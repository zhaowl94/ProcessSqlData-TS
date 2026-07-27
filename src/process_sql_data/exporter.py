"""Read-only raw CSV export with explicit legacy/corrected pagination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas

from .config import DatabaseConfig
from .database import ConnectionProtocol, CursorProtocol
from .pagination import (
    CorrectedKeysetPaginator,
    LegacyOffsetPaginator,
    PaginationSafetyError,
    primary_key_discovery_plan,
)
from .run_state import RunMode


class ExportError(RuntimeError):
    """Raised when a raw export cannot preserve its safety contract."""


@dataclass(frozen=True)
class ExportSummary:
    mode: str
    page_count: int
    row_count: int
    files: tuple[Path, ...]
    key_columns: tuple[str, ...]


def export_raw_pages(
    connection: ConnectionProtocol,
    *,
    config: DatabaseConfig,
    mode: RunMode,
    output_directory: Path | str,
    batch_size: int = 10_000,
) -> ExportSummary:
    """Export raw pages into an empty staging directory.

    This function never removes or overwrites files. The caller must publish the
    completed staging tree through the runtime module.
    """

    output = Path(output_directory)
    _require_empty_output_directory(output)
    cursor = connection.cursor()
    try:
        if mode is RunMode.LEGACY:
            return _export_legacy(
                cursor,
                config=config,
                output=output,
                batch_size=batch_size,
            )
        return _export_corrected(
            cursor,
            config=config,
            output=output,
            batch_size=batch_size,
        )
    finally:
        cursor.close()


def _export_legacy(
    cursor: CursorProtocol,
    *,
    config: DatabaseConfig,
    output: Path,
    batch_size: int,
) -> ExportSummary:
    paginator = LegacyOffsetPaginator(
        schema=config.schema,
        table=config.table,
        batch_size=batch_size,
    )
    files: list[Path] = []
    row_count = 0
    page_number = 0

    while True:
        page_number += 1
        plan = paginator.next_page()
        cursor.execute(plan.query, plan.parameters)
        rows = cursor.fetchall()
        columns = _description_columns(cursor)
        file_path = output / f"SqlData{page_number}.csv"
        pandas.DataFrame.from_records(rows, columns=columns).to_csv(
            file_path,
            index=True,
            encoding="utf-8",
        )
        files.append(file_path)
        row_count += len(rows)
        if len(rows) < batch_size:
            break

    return ExportSummary(
        mode=RunMode.LEGACY.value,
        page_count=page_number,
        row_count=row_count,
        files=tuple(files),
        key_columns=(),
    )


def _export_corrected(
    cursor: CursorProtocol,
    *,
    config: DatabaseConfig,
    output: Path,
    batch_size: int,
) -> ExportSummary:
    discovery = primary_key_discovery_plan(
        schema=config.schema,
        table=config.table,
    )
    cursor.execute(discovery.query, discovery.parameters)
    key_columns = tuple(str(row[0]) for row in cursor.fetchall())
    try:
        paginator = CorrectedKeysetPaginator(
            schema=config.schema,
            table=config.table,
            key_columns=key_columns,
            batch_size=batch_size,
        )
    except PaginationSafetyError as exc:
        raise ExportError(str(exc)) from exc

    files: list[Path] = []
    row_count = 0
    page_number = 0
    plan = paginator.first_page()
    while True:
        page_number += 1
        cursor.execute(plan.query, plan.parameters)
        rows = cursor.fetchall()
        columns = _description_columns(cursor)
        file_path = output / f"SqlData{page_number}.csv"
        pandas.DataFrame.from_records(rows, columns=columns).to_csv(
            file_path,
            index=True,
            encoding="utf-8",
        )
        files.append(file_path)
        row_count += len(rows)
        if len(rows) < batch_size:
            break

        last_key = _extract_last_key(
            rows[-1],
            columns=columns,
            key_columns=key_columns,
        )
        plan = paginator.page_after(last_key)

    return ExportSummary(
        mode=RunMode.CORRECTED.value,
        page_count=page_number,
        row_count=row_count,
        files=tuple(files),
        key_columns=key_columns,
    )


def _description_columns(cursor: CursorProtocol) -> tuple[str, ...]:
    description = getattr(cursor, "description", None)
    if not description:
        raise ExportError("Database cursor did not provide column metadata.")

    columns: list[str] = []
    for item in description:
        name = getattr(item, "name", None)
        if name is None and isinstance(item, Sequence) and item:
            name = item[0]
        if not isinstance(name, str) or not name:
            raise ExportError("Database cursor returned an invalid column name.")
        columns.append(name)
    if len(columns) != len(set(columns)):
        raise ExportError("Database query returned duplicate column names.")
    return tuple(columns)


def _extract_last_key(
    row: Sequence[Any],
    *,
    columns: tuple[str, ...],
    key_columns: tuple[str, ...],
) -> tuple[Any, ...]:
    positions = {name: index for index, name in enumerate(columns)}
    missing = [name for name in key_columns if name not in positions]
    if missing:
        raise ExportError(
            f"Primary key column(s) missing from export: {', '.join(missing)}"
        )
    return tuple(row[positions[name]] for name in key_columns)


def _require_empty_output_directory(output: Path) -> None:
    if not output.is_dir():
        raise ExportError(f"Staging output directory does not exist: {output}")
    try:
        first_entry = next(output.iterdir())
    except StopIteration:
        return
    raise ExportError(
        f"Staging output directory must be empty; found: {first_entry.name}"
    )
