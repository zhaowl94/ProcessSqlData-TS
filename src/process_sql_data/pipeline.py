"""File-level orchestration for the offline stages of the migration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import pandas

from .layout import LEGACY_OUTPUT_DIRECTORY_NAMES, ProjectLayout
from .run_state import RunMode
from .stages import (
    classify_series,
    group_by_device,
    interpolate_series,
    linear_trend_from_timestamps,
    merge_and_sort,
    parse_feature_rows,
    plot_histogram,
    plot_time_series,
    remove_outliers,
    select_arima_order,
    split_numeric_series,
    summary_statistics,
    timestamps_from_strings,
)


class PipelineError(RuntimeError):
    """Raised when an isolated file stage cannot complete safely."""


@dataclass(frozen=True)
class OfflinePipelineOptions:
    mode: RunMode = RunMode.LEGACY
    chunk_size: int = 10_000
    interpolation_points: int = 40_000
    arima_p_max: int = 10
    arima_q_max: int = 10
    arima_max_differences: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RunMode):
            raise ValueError("mode must be a RunMode value.")
        _positive_integer(self.chunk_size, "chunk_size")
        if self.interpolation_points < 2:
            raise ValueError("interpolation_points must be at least 2.")
        for value, name in (
            (self.arima_p_max, "arima_p_max"),
            (self.arima_q_max, "arima_q_max"),
            (self.arima_max_differences, "arima_max_differences"),
        ):
            _non_negative_integer(value, name)


@dataclass(frozen=True)
class OfflinePipelineSummary:
    run_id: str
    mode: str
    stage_file_counts: tuple[tuple[str, int], ...]

    def count_for(self, directory_name: str) -> int:
        for name, count in self.stage_file_counts:
            if name == directory_name:
                return count
        raise KeyError(f"Unknown output directory: {directory_name}")


def run_offline_stages(
    layout: ProjectLayout,
    run_id: str,
    *,
    options: OfflinePipelineOptions | None = None,
) -> OfflinePipelineSummary:
    """Run stages 2-14 entirely inside one prepared staging tree.

    ``DataRaw`` must already contain the exported pages. Every other staged
    output directory must be empty. This function never publishes or removes a
    live legacy output.
    """

    selected = options or OfflinePipelineOptions()
    directories = {
        name: layout.staged_output_path(run_id, name)
        for name in LEGACY_OUTPUT_DIRECTORY_NAMES
    }
    _validate_staging_tree(directories)

    actions: tuple[tuple[str, Callable[[], None]], ...] = (
        (
            "02_group_by_device",
            lambda: _stage_group_by_device(
                directories["DataRaw"],
                directories["DataProcess1"],
            ),
        ),
        (
            "03_parse_features",
            lambda: _stage_parse_features(
                directories["DataProcess1"],
                directories["DataProcess2"],
                mode=selected.mode,
                chunk_size=selected.chunk_size,
            ),
        ),
        (
            "04_merge_and_sort",
            lambda: _stage_merge_and_sort(
                directories["DataProcess2"],
                directories["DataProcess3"],
            ),
        ),
        (
            "05_split_time_series",
            lambda: _stage_split_time_series(
                directories["DataProcess3"],
                directories["DataProcess4"],
                mode=selected.mode,
            ),
        ),
        (
            "06_remove_outliers",
            lambda: _stage_remove_outliers(
                directories["DataProcess4"],
                directories["DataProcess5"],
            ),
        ),
        (
            "07_plot_time_series",
            lambda: _stage_plot_time_series(
                directories["DataProcess5"],
                directories["DataProcess6"],
            ),
        ),
        (
            "08_classify_series",
            lambda: _stage_classify_series(
                directories["DataProcess5"],
                directories["DataProcess7"],
            ),
        ),
        (
            "09_convert_timestamps",
            lambda: _stage_convert_timestamps(
                directories["DataProcess5"],
                directories["DataProcess8"],
            ),
        ),
        (
            "10_interpolate",
            lambda: _stage_interpolate(
                directories["DataProcess5"],
                directories["DataProcess9"],
                mode=selected.mode,
                points=selected.interpolation_points,
            ),
        ),
        (
            "11_fit_arima",
            lambda: _stage_fit_arima(
                directories["DataProcess9"],
                directories["DataProcess10"],
                p_max=selected.arima_p_max,
                q_max=selected.arima_q_max,
                max_differences=selected.arima_max_differences,
            ),
        ),
        (
            "12_plot_histograms",
            lambda: _stage_plot_histograms(
                directories["DataProcess9"],
                directories["DataProcess11"],
            ),
        ),
        (
            "13_summary_statistics",
            lambda: _stage_summary_statistics(
                directories["DataProcess9"],
                directories["DataProcess12"],
            ),
        ),
        (
            "14_linear_trend",
            lambda: _stage_linear_trend(
                directories["DataProcess9"],
                directories["DataProcess13"],
            ),
        ),
    )

    for stage_name, action in actions:
        try:
            action()
        except PipelineError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            raise PipelineError(
                f"Offline stage {stage_name} failed ({error_type})."
            ) from None

    return OfflinePipelineSummary(
        run_id=run_id,
        mode=selected.mode.value,
        stage_file_counts=tuple(
            (name, _count_files(path)) for name, path in directories.items()
        ),
    )


def _stage_group_by_device(source: Path, destination: Path) -> None:
    frames = [_read_csv(path) for path in _raw_page_files(source)]
    for device, frame in group_by_device(frames).items():
        filename = f"{_safe_component(device)}.csv"
        _write_csv(destination / filename, frame)


def _stage_parse_features(
    source: Path,
    destination: Path,
    *,
    mode: RunMode,
    chunk_size: int,
) -> None:
    for source_file in _csv_files(source):
        device_directory = destination / _safe_component(source_file.stem)
        device_directory.mkdir()
        frame = _strip_csv_index(_read_csv(source_file))
        for part_number, start in enumerate(range(0, len(frame), chunk_size)):
            chunk = frame.iloc[start : start + chunk_size]
            parsed = parse_feature_rows(chunk, mode=mode)
            _write_csv(device_directory / f"{part_number}.csv", parsed)


def _stage_merge_and_sort(source: Path, destination: Path) -> None:
    for device_directory in _subdirectories(source):
        chunks = [
            _strip_csv_index(_read_csv(path))
            for path in _numbered_csv_files(device_directory)
        ]
        if not chunks:
            continue
        merged = merge_and_sort(chunks)
        if not merged.empty:
            _write_csv(destination / f"{device_directory.name}.csv", merged)


def _stage_split_time_series(
    source: Path,
    destination: Path,
    *,
    mode: RunMode,
) -> None:
    for source_file in _csv_files(source):
        frame = _strip_csv_index(_read_csv(source_file))
        for feature, series in split_numeric_series(frame, mode=mode).items():
            safe_feature = _safe_component(feature)
            output = destination / f"{source_file.stem}_{safe_feature}.csv"
            _write_csv(output, series)


def _stage_remove_outliers(source: Path, destination: Path) -> None:
    for source_file in _csv_files(source):
        frame = _strip_csv_index(_read_csv(source_file))
        if frame.empty or len(frame.columns) != 2:
            continue
        filtered = remove_outliers(frame, value_column=str(frame.columns[1]))
        if not filtered.empty:
            _write_csv(destination / source_file.name, filtered)


def _stage_plot_time_series(source: Path, destination: Path) -> None:
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        if frame.empty:
            continue
        plot_time_series(
            frame,
            time_column=str(frame.columns[0]),
            value_column=str(frame.columns[1]),
            output_path=destination / f"{source_file.stem}.png",
        )


def _stage_classify_series(source: Path, destination: Path) -> None:
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        result = classify_series(frame.iloc[:, 1])
        virtual_input = f"../DataProcess5/{source_file.name}"
        (destination / f"{source_file.stem}.txt").write_text(
            virtual_input + result.legacy_message + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if result.should_continue:
            compatible = frame.copy()
            compatible.columns = ["time", "data"]
            _write_csv(destination / source_file.name, compatible)


def _stage_convert_timestamps(source: Path, destination: Path) -> None:
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        converted = timestamps_from_strings(
            frame,
            time_column=str(frame.columns[0]),
            value_column=str(frame.columns[1]),
        )
        _write_csv(destination / source_file.name, converted)


def _stage_interpolate(
    source: Path,
    destination: Path,
    *,
    mode: RunMode,
    points: int,
) -> None:
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        interpolated = interpolate_series(
            frame,
            time_column=str(frame.columns[0]),
            value_column=str(frame.columns[1]),
            mode=mode,
            points=points,
        )
        if not interpolated.empty:
            _write_csv(destination / source_file.name, interpolated)


def _stage_fit_arima(
    source: Path,
    destination: Path,
    *,
    p_max: int,
    q_max: int,
    max_differences: int,
) -> None:
    records: list[str] = []
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        result = select_arima_order(
            frame.iloc[:, 1],
            p_max=p_max,
            q_max=q_max,
            max_differences=max_differences,
        )
        virtual_input = f"../DataProcess9/{source_file.name}"
        records.append(f"{virtual_input}\n{result.legacy_message}\n\n")
    if records:
        (destination / "result1.txt").write_text(
            "".join(records),
            encoding="utf-8",
            newline="\n",
        )


def _stage_plot_histograms(source: Path, destination: Path) -> None:
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        plot_histogram(
            frame.iloc[:, 1],
            output_path=destination / f"{source_file.stem}.png",
        )


def _stage_summary_statistics(source: Path, destination: Path) -> None:
    named_series = {
        source_file.name: _two_column_frame(source_file).iloc[:, 1]
        for source_file in _csv_files(source)
    }
    result = summary_statistics(named_series)
    _write_csv(destination / "ResultAll.csv", result)


def _stage_linear_trend(source: Path, destination: Path) -> None:
    records: list[dict[str, object]] = []
    for source_file in _csv_files(source):
        frame = _two_column_frame(source_file)
        coefficient, intercept = linear_trend_from_timestamps(
            frame,
            time_column=str(frame.columns[0]),
            value_column=str(frame.columns[1]),
        )
        records.append(
            {
                "coef": coefficient,
                "intercept": intercept,
                "file": source_file.name,
            }
        )
    result = pandas.DataFrame.from_records(
        records,
        columns=["coef", "intercept", "file"],
    )
    _write_csv(destination / "ResultAll.csv", result)


def _validate_staging_tree(directories: dict[str, Path]) -> None:
    for name, path in directories.items():
        if not path.is_dir():
            raise PipelineError(f"Missing prepared staging directory: {name}")
        if name != "DataRaw" and next(path.iterdir(), None) is not None:
            raise PipelineError(
                f"Staged output directory must be empty before running: {name}"
            )
    _raw_page_files(directories["DataRaw"])


def _raw_page_files(directory: Path) -> tuple[Path, ...]:
    numbered: list[tuple[int, Path]] = []
    for entry in directory.iterdir():
        match = re.fullmatch(r"SqlData([1-9][0-9]*)\.csv", entry.name)
        if not entry.is_file() or match is None:
            raise PipelineError("DataRaw contains an unexpected entry.")
        numbered.append((int(match.group(1)), entry))
    if not numbered:
        raise PipelineError("DataRaw does not contain any exported CSV page.")
    numbered.sort(key=lambda item: item[0])
    expected = list(range(1, len(numbered) + 1))
    if [number for number, _ in numbered] != expected:
        raise PipelineError("DataRaw page numbering is not contiguous.")
    return tuple(path for _, path in numbered)


def _numbered_csv_files(directory: Path) -> tuple[Path, ...]:
    numbered: list[tuple[int, Path]] = []
    for entry in directory.iterdir():
        match = re.fullmatch(r"([0-9]+)\.csv", entry.name)
        if not entry.is_file() or match is None:
            raise PipelineError("Parsed feature directory has an unexpected entry.")
        numbered.append((int(match.group(1)), entry))
    numbered.sort(key=lambda item: item[0])
    return tuple(path for _, path in numbered)


def _csv_files(directory: Path) -> tuple[Path, ...]:
    files = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".csv"
    ]
    return tuple(sorted(files, key=lambda path: path.name.casefold()))


def _subdirectories(directory: Path) -> tuple[Path, ...]:
    directories = [entry for entry in directory.iterdir() if entry.is_dir()]
    return tuple(sorted(directories, key=lambda path: path.name.casefold()))


def _read_csv(path: Path) -> pandas.DataFrame:
    return pandas.read_csv(path, encoding="utf-8")


def _write_csv(path: Path, frame: pandas.DataFrame) -> None:
    if path.exists():
        raise PipelineError("A staged output file would be overwritten.")
    frame.to_csv(path, index=True, encoding="utf-8")


def _strip_csv_index(frame: pandas.DataFrame) -> pandas.DataFrame:
    result = frame
    while len(result.columns) and str(result.columns[0]).startswith("Unnamed:"):
        result = result.iloc[:, 1:]
    return result.copy()


def _two_column_frame(path: Path) -> pandas.DataFrame:
    frame = _strip_csv_index(_read_csv(path))
    if len(frame.columns) != 2:
        raise PipelineError("A time-series CSV must contain exactly two data columns.")
    return frame


def _safe_component(value: object) -> str:
    component = str(value)
    invalid = '<>:"/\\|?*'
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
    if (
        not component
        or component in {".", ".."}
        or len(component) > 128
        or component[-1] in {".", " "}
        or any(character in invalid or ord(character) < 32 for character in component)
        or component.casefold() in reserved
    ):
        raise PipelineError("Unsafe device or feature name.")
    return component


def _count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
