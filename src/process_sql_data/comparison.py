"""Deterministic comparison of legacy and migrated output directory trees."""

from __future__ import annotations

import codecs
import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Callable


_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\d*\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?$"
)


@dataclass(frozen=True)
class Difference:
    """One human-readable incompatibility between two output trees."""

    path: str
    message: str


@dataclass(frozen=True)
class ComparisonResult:
    """Complete, bounded list of compatibility differences."""

    differences: tuple[Difference, ...]

    @property
    def matches(self) -> bool:
        return not self.differences

    def summary(self) -> str:
        if self.matches:
            return "Output trees are compatible."
        return "\n".join(
            f"{difference.path}: {difference.message}"
            for difference in self.differences
        )


def compare_output_trees(
    expected_root: Path | str,
    actual_root: Path | str,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
    max_differences: int = 200,
) -> ComparisonResult:
    """Compare directory structure, CSV contracts, and other file contents."""

    expected = Path(expected_root)
    actual = Path(actual_root)
    differences: list[Difference] = []

    def add(path: str, message: str) -> None:
        if len(differences) < max_differences:
            differences.append(Difference(path=path, message=message))

    if not expected.is_dir():
        add(".", f"expected root is not a directory: {expected}")
        return ComparisonResult(tuple(differences))
    if not actual.is_dir():
        add(".", f"actual root is not a directory: {actual}")
        return ComparisonResult(tuple(differences))

    expected_directories = _relative_paths(expected, Path.is_dir)
    actual_directories = _relative_paths(actual, Path.is_dir)
    _compare_path_sets(
        expected_directories,
        actual_directories,
        "directory",
        add,
    )

    expected_files = _relative_paths(expected, Path.is_file)
    actual_files = _relative_paths(actual, Path.is_file)
    _compare_path_sets(expected_files, actual_files, "file", add)

    for relative_path in sorted(expected_files & actual_files):
        expected_file = expected / relative_path
        actual_file = actual / relative_path
        if expected_file.suffix.lower() == ".csv":
            _compare_csv(
                expected_file,
                actual_file,
                relative_path,
                relative_tolerance,
                absolute_tolerance,
                add,
            )
        elif _sha256(expected_file) != _sha256(actual_file):
            add(relative_path, "file content differs")

    return ComparisonResult(tuple(differences))


def _relative_paths(
    root: Path,
    predicate: Callable[[Path], bool],
) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if predicate(path)
    }


def _compare_path_sets(
    expected: set[str],
    actual: set[str],
    kind: str,
    add: Callable[[str, str], None],
) -> None:
    for path in sorted(expected - actual):
        add(path, f"missing {kind}")
    for path in sorted(actual - expected):
        add(path, f"unexpected {kind}")


def _compare_csv(
    expected_path: Path,
    actual_path: Path,
    relative_path: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    add: Callable[[str, str], None],
) -> None:
    expected_encoding, expected_rows = _read_csv(expected_path)
    actual_encoding, actual_rows = _read_csv(actual_path)

    if expected_encoding != actual_encoding:
        add(
            relative_path,
            f"encoding differs: expected {expected_encoding}, "
            f"actual {actual_encoding}",
        )

    if not expected_rows or not actual_rows:
        if expected_rows != actual_rows:
            add(relative_path, "empty/non-empty CSV state differs")
        return

    if expected_rows[0] != actual_rows[0]:
        add(
            relative_path,
            f"header differs: expected {expected_rows[0]!r}, "
            f"actual {actual_rows[0]!r}",
        )

    expected_data = expected_rows[1:]
    actual_data = actual_rows[1:]
    if len(expected_data) != len(actual_data):
        add(
            relative_path,
            f"row count differs: expected {len(expected_data)}, "
            f"actual {len(actual_data)}",
        )

    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_data, actual_data),
        start=2,
    ):
        if len(expected_row) != len(actual_row):
            add(
                relative_path,
                f"column count differs at row {row_index}: "
                f"expected {len(expected_row)}, actual {len(actual_row)}",
            )
            continue

        for column_index, (expected_cell, actual_cell) in enumerate(
            zip(expected_row, actual_row),
            start=1,
        ):
            if _cells_match(
                expected_cell,
                actual_cell,
                relative_tolerance,
                absolute_tolerance,
            ):
                continue
            add(
                relative_path,
                f"value differs at row {row_index}, column {column_index}: "
                f"expected {expected_cell!r}, actual {actual_cell!r}",
            )


def _read_csv(path: Path) -> tuple[str, list[list[str]]]:
    raw = path.read_bytes()
    encoding, text = _decode_text(raw)
    rows = list(csv.reader(io.StringIO(text, newline="")))
    return encoding, rows


def _decode_text(raw: bytes) -> tuple[str, str]:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig", raw.decode("utf-8-sig")
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le", raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le")
    if raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be", raw[len(codecs.BOM_UTF16_BE) :].decode("utf-16-be")

    try:
        return "utf-8", raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return "gb18030", raw.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV is neither UTF-8 nor GB18030 text.") from exc


def _cells_match(
    expected: str,
    actual: str,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if expected == actual:
        return True

    if _INTEGER_PATTERN.fullmatch(expected) and _INTEGER_PATTERN.fullmatch(actual):
        return False
    if not _FLOAT_PATTERN.fullmatch(expected) or not _FLOAT_PATTERN.fullmatch(actual):
        return False

    return math.isclose(
        float(expected),
        float(actual),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
