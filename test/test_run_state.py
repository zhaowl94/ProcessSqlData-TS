from __future__ import annotations

from pathlib import Path
import unittest

from process_sql_data.run_state import (
    LEGACY_STAGE_NAMES,
    InvalidStageTransition,
    RunManifest,
    RunMode,
    StageStatus,
)
from test import temporary_directory


CONFIG_FINGERPRINT = "config-0123456789abcdef"
INPUT_FINGERPRINT = "input-0123456789abcdef"
OUTPUT_FINGERPRINT = "output-0123456789abcdef"


class RunManifestTests(unittest.TestCase):
    def test_manifest_round_trip_contains_no_configuration_values(self) -> None:
        root = Path(self.enterContext(temporary_directory()))
        path = root / "manifest.json"
        manifest = RunManifest.create(
            run_id="run-1",
            mode=RunMode.LEGACY,
            config_fingerprint=CONFIG_FINGERPRINT,
            input_fingerprint=INPUT_FINGERPRINT,
        )
        manifest.save(path)
        loaded = RunManifest.load(path)
        raw = path.read_text(encoding="utf-8")

        self.assertEqual(loaded.run_id, "run-1")
        self.assertEqual(loaded.mode, RunMode.LEGACY.value)
        self.assertEqual(set(loaded.stages), set(LEGACY_STAGE_NAMES))
        self.assertNotIn("password", raw.lower())
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_completed_stage_can_only_be_reused_for_same_input(self) -> None:
        manifest = RunManifest.create(
            run_id="run-2",
            mode=RunMode.CORRECTED,
            config_fingerprint=CONFIG_FINGERPRINT,
            input_fingerprint=INPUT_FINGERPRINT,
        )
        stage = LEGACY_STAGE_NAMES[0]
        manifest.mark_started(stage, input_fingerprint=INPUT_FINGERPRINT)
        manifest.mark_completed(
            stage,
            output_fingerprint=OUTPUT_FINGERPRINT,
        )

        self.assertEqual(
            manifest.stages[stage].status,
            StageStatus.COMPLETED.value,
        )
        self.assertTrue(
            manifest.stage_can_be_reused(
                stage,
                input_fingerprint=INPUT_FINGERPRINT,
            )
        )
        self.assertFalse(
            manifest.stage_can_be_reused(
                stage,
                input_fingerprint="different-0123456789",
            )
        )

    def test_failed_stage_may_restart_but_completed_stage_may_not(self) -> None:
        manifest = RunManifest.create(
            run_id="run-3",
            mode=RunMode.LEGACY,
            config_fingerprint=CONFIG_FINGERPRINT,
            input_fingerprint=INPUT_FINGERPRINT,
        )
        stage = LEGACY_STAGE_NAMES[1]
        manifest.mark_started(stage, input_fingerprint=INPUT_FINGERPRINT)
        manifest.mark_failed(stage, error_type="ValueError: private row value")
        self.assertEqual(
            manifest.stages[stage].error_type,
            "ValueErrorprivaterowvalue",
        )

        manifest.mark_started(stage, input_fingerprint=INPUT_FINGERPRINT)
        manifest.mark_completed(
            stage,
            output_fingerprint=OUTPUT_FINGERPRINT,
        )
        with self.assertRaises(InvalidStageTransition):
            manifest.mark_started(stage, input_fingerprint=INPUT_FINGERPRINT)

    def test_resume_requires_same_mode_configuration_and_input(self) -> None:
        manifest = RunManifest.create(
            run_id="run-4",
            mode=RunMode.LEGACY,
            config_fingerprint=CONFIG_FINGERPRINT,
            input_fingerprint=INPUT_FINGERPRINT,
        )

        self.assertTrue(
            manifest.can_resume(
                mode=RunMode.LEGACY,
                config_fingerprint=CONFIG_FINGERPRINT,
                input_fingerprint=INPUT_FINGERPRINT,
            )
        )
        self.assertFalse(
            manifest.can_resume(
                mode=RunMode.CORRECTED,
                config_fingerprint=CONFIG_FINGERPRINT,
                input_fingerprint=INPUT_FINGERPRINT,
            )
        )
        self.assertFalse(
            manifest.can_resume(
                mode=RunMode.LEGACY,
                config_fingerprint="different-0123456789",
                input_fingerprint=INPUT_FINGERPRINT,
            )
        )


if __name__ == "__main__":
    unittest.main()
