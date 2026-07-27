from __future__ import annotations

from pathlib import Path
import unittest

from process_sql_data.layout import ProjectLayout
from process_sql_data.runtime import (
    PublishError,
    prepare_staging,
    publish_staged_outputs,
    rollback_published_outputs,
)
from test import temporary_directory


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(self.enterContext(temporary_directory()))
        repository = root / "ProcessSqlData-TS"
        repository.mkdir()
        self.layout = ProjectLayout.from_entrypoint(
            repository / "ProcessSqlData Final.py"
        )
        self.names = ("DataRaw", "DataProcess1")

    def _write_file(self, directory: Path, value: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "value.txt").write_text(value, encoding="utf-8")

    def _read_file(self, directory: Path) -> str:
        return (directory / "value.txt").read_text(encoding="utf-8")

    def test_prepare_staging_creates_only_requested_directories(self) -> None:
        prepared = prepare_staging(
            self.layout,
            "run-prepare",
            directory_names=self.names,
        )
        self.assertEqual(
            prepared,
            tuple(
                self.layout.staged_output_path("run-prepare", name)
                for name in self.names
            ),
        )
        self.assertTrue(all(path.is_dir() for path in prepared))

    def test_publish_retains_backups_and_rollback_restores_them(self) -> None:
        run_id = "run-publish"
        prepare_staging(self.layout, run_id, directory_names=self.names)
        for name in self.names:
            self._write_file(self.layout.output_path(name), f"old-{name}")
            self._write_file(
                self.layout.staged_output_path(run_id, name),
                f"new-{name}",
            )

        outcome = publish_staged_outputs(
            self.layout,
            run_id,
            directory_names=self.names,
        )

        self.assertEqual(outcome.published, self.names)
        self.assertEqual(outcome.backed_up, self.names)
        for name in self.names:
            self.assertEqual(
                self._read_file(self.layout.output_path(name)),
                f"new-{name}",
            )
            self.assertEqual(
                self._read_file(self.layout.backup_output_path(run_id, name)),
                f"old-{name}",
            )

        rollback_published_outputs(
            self.layout,
            run_id,
            directory_names=self.names,
        )
        for name in self.names:
            self.assertEqual(
                self._read_file(self.layout.output_path(name)),
                f"old-{name}",
            )
            preserved = (
                self.layout.run_root(run_id) / "rolled-back" / name
            )
            self.assertEqual(self._read_file(preserved), f"new-{name}")

    def test_incomplete_staging_fails_before_existing_output_changes(self) -> None:
        run_id = "run-incomplete"
        first_stage = self.layout.staged_output_path(run_id, self.names[0])
        self._write_file(first_stage, "new")
        target = self.layout.output_path(self.names[0])
        self._write_file(target, "old")

        with self.assertRaisesRegex(PublishError, "Missing staged"):
            publish_staged_outputs(
                self.layout,
                run_id,
                directory_names=self.names,
            )

        self.assertEqual(self._read_file(target), "old")

    def test_mid_publish_failure_restores_all_existing_outputs(self) -> None:
        run_id = "run-failure"
        prepare_staging(self.layout, run_id, directory_names=self.names)
        for name in self.names:
            self._write_file(self.layout.output_path(name), f"old-{name}")
            self._write_file(
                self.layout.staged_output_path(run_id, name),
                f"new-{name}",
            )

        move_count = 0

        def fail_during_second_directory(source: Path, destination: Path) -> None:
            nonlocal move_count
            move_count += 1
            if move_count == 4:
                raise OSError("injected move failure")
            source.replace(destination)

        with self.assertRaisesRegex(PublishError, "rolled back"):
            publish_staged_outputs(
                self.layout,
                run_id,
                directory_names=self.names,
                _move=fail_during_second_directory,
            )

        for name in self.names:
            self.assertEqual(
                self._read_file(self.layout.output_path(name)),
                f"old-{name}",
            )

    def test_existing_backup_prevents_publish(self) -> None:
        run_id = "run-existing-backup"
        prepare_staging(self.layout, run_id, directory_names=self.names)
        self.layout.backup_output_path(run_id, self.names[0]).mkdir(
            parents=True
        )

        with self.assertRaisesRegex(PublishError, "Backup path already exists"):
            publish_staged_outputs(
                self.layout,
                run_id,
                directory_names=self.names,
            )


if __name__ == "__main__":
    unittest.main()
