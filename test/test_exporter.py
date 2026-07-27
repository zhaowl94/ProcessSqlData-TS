from __future__ import annotations

from pathlib import Path
import unittest

import pandas

from process_sql_data.config import DatabaseConfig
from process_sql_data.exporter import ExportError, export_raw_pages
from process_sql_data.run_state import RunMode
from test import temporary_directory


class FakeCursor:
    def __init__(
        self,
        *,
        page_rows: list[list[tuple[object, ...]]],
        key_columns: tuple[str, ...] = ("id",),
    ) -> None:
        self.page_rows = list(page_rows)
        self.key_columns = key_columns
        self.description: tuple[tuple[str], ...] | None = None
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self._current_rows: list[tuple[object, ...]] = []
        self.closed = False

    def execute(
        self,
        query: str,
        parameters: tuple[object, ...] = (),
    ) -> None:
        self.executions.append((query, parameters))
        if "pg_index" in query:
            self._current_rows = [(column,) for column in self.key_columns]
            self.description = (("attname",),)
            return
        self._current_rows = self.page_rows.pop(0)
        self.description = (
            ("id",),
            ("sn",),
            ("date",),
            ("data",),
            ("slave_id",),
            ("type_id",),
        )

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._current_rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self._current_rows[0] if self._current_rows else None

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    def set_session(self, *, readonly: bool, autocommit: bool) -> None:
        raise AssertionError("Exporter must receive an already-safe connection.")

    def close(self) -> None:
        pass


def config() -> DatabaseConfig:
    return DatabaseConfig(
        host="db.example.invalid",
        port=5432,
        database="telemetry",
        user="readonly_user",
        password="test-password",
    )


def row(identifier: int) -> tuple[object, ...]:
    return (
        identifier,
        "device-a",
        f"2020-01-01 00:00:0{identifier}",
        '{"temperature":"12.5"}',
        1,
        2,
    )


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(temporary_directory()))
        self.output = self.root / "DataRaw"
        self.output.mkdir()

    def test_legacy_export_preserves_offset_and_file_names(self) -> None:
        cursor = FakeCursor(page_rows=[[row(3), row(4)], [row(5)]])

        summary = export_raw_pages(
            FakeConnection(cursor),
            config=config(),
            mode=RunMode.LEGACY,
            output_directory=self.output,
            batch_size=2,
        )

        self.assertEqual(summary.page_count, 2)
        self.assertEqual(summary.row_count, 3)
        self.assertEqual(
            [path.name for path in summary.files],
            ["SqlData1.csv", "SqlData2.csv"],
        )
        self.assertEqual(cursor.executions[0][1], (2, 2))
        self.assertEqual(cursor.executions[1][1], (2, 4))
        self.assertNotIn("ORDER BY", cursor.executions[0][0])
        self.assertTrue(cursor.closed)
        first_page = pandas.read_csv(self.output / "SqlData1.csv")
        self.assertEqual(first_page["id"].tolist(), [3, 4])

    def test_corrected_export_discovers_key_and_uses_last_row(self) -> None:
        cursor = FakeCursor(page_rows=[[row(1), row(2)], [row(3)]])

        summary = export_raw_pages(
            FakeConnection(cursor),
            config=config(),
            mode=RunMode.CORRECTED,
            output_directory=self.output,
            batch_size=2,
        )

        self.assertEqual(summary.key_columns, ("id",))
        self.assertEqual(summary.row_count, 3)
        self.assertIn("pg_index", cursor.executions[0][0])
        self.assertIn("ORDER BY", cursor.executions[1][0])
        self.assertEqual(cursor.executions[1][1], (2,))
        self.assertIn('WHERE ("id") > (%s)', cursor.executions[2][0])
        self.assertEqual(cursor.executions[2][1], (2, 2))

    def test_corrected_export_refuses_table_without_primary_key(self) -> None:
        cursor = FakeCursor(page_rows=[], key_columns=())

        with self.assertRaisesRegex(ExportError, "stable key"):
            export_raw_pages(
                FakeConnection(cursor),
                config=config(),
                mode=RunMode.CORRECTED,
                output_directory=self.output,
                batch_size=2,
            )

        self.assertTrue(cursor.closed)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_export_refuses_nonempty_staging_without_deleting_it(self) -> None:
        existing = self.output / "keep.txt"
        existing.write_text("keep", encoding="utf-8")
        cursor = FakeCursor(page_rows=[])

        with self.assertRaisesRegex(ExportError, "must be empty"):
            export_raw_pages(
                FakeConnection(cursor),
                config=config(),
                mode=RunMode.LEGACY,
                output_directory=self.output,
                batch_size=2,
            )

        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.assertFalse(cursor.closed)


if __name__ == "__main__":
    unittest.main()
