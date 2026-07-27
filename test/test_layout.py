from __future__ import annotations

import json
from pathlib import Path
import unittest

from process_sql_data.layout import (
    LEGACY_OUTPUT_DIRECTORY_NAMES,
    PathSafetyError,
    ProjectLayout,
)
from test import REPOSITORY_ROOT, temporary_directory


class LayoutTests(unittest.TestCase):
    def test_entrypoint_anchors_outputs_to_repository_parent(self) -> None:
        with temporary_directory() as directory:
            workspace = Path(directory)
            repository = workspace / "ProcessSqlData-TS"
            repository.mkdir()
            entrypoint = repository / "ProcessSqlData Final.py"

            layout = ProjectLayout.from_entrypoint(entrypoint)

            self.assertEqual(layout.repository_root, repository.resolve())
            self.assertEqual(layout.data_root, workspace.resolve())
            self.assertEqual(
                layout.output_path("DataRaw"),
                workspace.resolve() / "DataRaw",
            )
            self.assertEqual(
                layout.output_path("DataProcess13"),
                workspace.resolve() / "DataProcess13",
            )

    def test_unknown_or_traversal_output_is_rejected(self) -> None:
        layout = ProjectLayout.from_entrypoint(
            REPOSITORY_ROOT / "ProcessSqlData Final.py"
        )
        for name in ("DataProcess14", "..", "../DataRaw", ""):
            with self.subTest(name=name):
                with self.assertRaises(PathSafetyError):
                    layout.output_path(name)

    def test_run_identifier_cannot_escape_runtime_root(self) -> None:
        layout = ProjectLayout.from_entrypoint(
            REPOSITORY_ROOT / "ProcessSqlData Final.py"
        )
        for run_id in ("../escape", "run/child", "", ".hidden"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(PathSafetyError):
                    layout.run_root(run_id)

        safe = layout.run_root("20260727T120000Z-ab12")
        self.assertEqual(
            safe,
            layout.runtime_root / "runs" / "20260727T120000Z-ab12",
        )

    def test_staging_and_backup_keep_legacy_directory_name(self) -> None:
        layout = ProjectLayout.from_entrypoint(
            REPOSITORY_ROOT / "ProcessSqlData Final.py"
        )
        self.assertEqual(
            layout.staged_output_path("run-1", "DataProcess4").name,
            "DataProcess4",
        )
        self.assertEqual(
            layout.backup_output_path("run-1", "DataProcess4").name,
            "DataProcess4",
        )

    def test_frozen_contract_lists_all_legacy_output_directories(self) -> None:
        contract_path = (
            REPOSITORY_ROOT / "test" / "fixtures" / "legacy_contract.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertEqual(
            contract["output_directories"],
            list(LEGACY_OUTPUT_DIRECTORY_NAMES),
        )
        self.assertEqual(len(LEGACY_OUTPUT_DIRECTORY_NAMES), 14)
        self.assertEqual(contract["legacy_batch_size"], 10_000)
        self.assertEqual(contract["legacy_first_offset"], 10_000)


if __name__ == "__main__":
    unittest.main()
