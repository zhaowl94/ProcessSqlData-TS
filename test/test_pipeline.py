from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import unittest

import pandas

from process_sql_data.layout import (
    LEGACY_OUTPUT_DIRECTORY_NAMES,
    ProjectLayout,
)
from process_sql_data.pipeline import (
    OfflinePipelineOptions,
    PipelineError,
    run_offline_stages,
)
from process_sql_data.run_state import RunMode
from process_sql_data.runtime import prepare_staging
from test import temporary_directory


class OfflinePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(self.enterContext(temporary_directory()))
        repository = root / "ProcessSqlData-TS"
        repository.mkdir()
        self.layout = ProjectLayout.from_entrypoint(
            repository / "ProcessSqlData Final.py"
        )

    def _prepare_raw(
        self,
        run_id: str,
        *,
        device: str = "device-a",
        corrected_payload: bool = False,
    ) -> None:
        prepare_staging(self.layout, run_id)
        start = datetime(2020, 1, 1)
        rows = []
        for index in range(20):
            value = 1.25 + index * 0.5
            if corrected_payload:
                payload = (
                    f'{{"temperature": {value}, '
                    f'"label": "zone: east"}}'
                )
            else:
                payload = f'{{"temperature":"{value}"}}'
            rows.append(
                {
                    "sn": device,
                    "date": (start + timedelta(seconds=index)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "data": payload,
                    "slave_id": 1,
                    "type_id": 2,
                }
            )
        pandas.DataFrame(rows).to_csv(
            self.layout.staged_output_path(run_id, "DataRaw")
            / "SqlData1.csv",
            index=True,
            encoding="utf-8",
        )

    def test_all_offline_stages_run_only_inside_staging(self) -> None:
        run_id = "synthetic-e2e"
        self._prepare_raw(run_id)

        summary = run_offline_stages(
            self.layout,
            run_id,
            options=OfflinePipelineOptions(
                mode=RunMode.LEGACY,
                chunk_size=7,
                interpolation_points=32,
                arima_p_max=0,
                arima_q_max=0,
                arima_max_differences=1,
            ),
        )

        self.assertEqual(summary.mode, "legacy")
        self.assertEqual(summary.count_for("DataRaw"), 1)
        self.assertEqual(summary.count_for("DataProcess2"), 3)
        for name in LEGACY_OUTPUT_DIRECTORY_NAMES[1:]:
            self.assertGreater(summary.count_for(name), 0, name)
            self.assertFalse(self.layout.output_path(name).exists())

        series_name = "device-a_temperature"
        process6 = self.layout.staged_output_path(run_id, "DataProcess6")
        process7 = self.layout.staged_output_path(run_id, "DataProcess7")
        process8 = self.layout.staged_output_path(run_id, "DataProcess8")
        process10 = self.layout.staged_output_path(run_id, "DataProcess10")
        process12 = self.layout.staged_output_path(run_id, "DataProcess12")
        process13 = self.layout.staged_output_path(run_id, "DataProcess13")

        self.assertTrue(
            (process6 / f"{series_name}.png")
            .read_bytes()
            .startswith(b"\x89PNG\r\n\x1a\n")
        )
        classification = (
            process7 / f"{series_name}.txt"
        ).read_text(encoding="utf-8")
        self.assertTrue(classification.startswith("../DataProcess5/"))
        self.assertNotIn(".runtime", classification)

        converted = pandas.read_csv(
            process8 / f"{series_name}.csv",
            encoding="utf-8",
        )
        self.assertEqual(list(converted.columns[-2:]), ["time", "data"])

        arima_log = (process10 / "result1.txt").read_text(encoding="utf-8")
        self.assertIn("../DataProcess9/", arima_log)
        self.assertNotIn(".runtime", arima_log)

        summary_frame = pandas.read_csv(
            process12 / "ResultAll.csv",
            encoding="utf-8",
        )
        trend_frame = pandas.read_csv(
            process13 / "ResultAll.csv",
            encoding="utf-8",
        )
        self.assertEqual(list(summary_frame.columns[-3:]), ["var", "mean", "file"])
        self.assertEqual(
            list(trend_frame.columns[-3:]),
            ["coef", "intercept", "file"],
        )

    def test_rerun_refuses_nonempty_stage_instead_of_overwriting(self) -> None:
        run_id = "synthetic-rerun"
        self._prepare_raw(run_id)
        options = OfflinePipelineOptions(
            interpolation_points=16,
            arima_p_max=0,
            arima_q_max=0,
            arima_max_differences=1,
        )
        run_offline_stages(self.layout, run_id, options=options)

        with self.assertRaisesRegex(PipelineError, "must be empty"):
            run_offline_stages(self.layout, run_id, options=options)

    def test_corrected_mode_keeps_external_series_names(self) -> None:
        run_id = "synthetic-corrected"
        self._prepare_raw(run_id, corrected_payload=True)

        summary = run_offline_stages(
            self.layout,
            run_id,
            options=OfflinePipelineOptions(
                mode=RunMode.CORRECTED,
                interpolation_points=16,
                arima_p_max=0,
                arima_q_max=0,
                arima_max_differences=1,
            ),
        )

        expected_name = "device-a_temperature.csv"
        process4 = self.layout.staged_output_path(run_id, "DataProcess4")
        process5 = self.layout.staged_output_path(run_id, "DataProcess5")
        self.assertEqual(summary.mode, "corrected")
        self.assertTrue((process4 / expected_name).is_file())
        self.assertTrue((process5 / expected_name).is_file())
        self.assertFalse(self.layout.output_path("DataProcess5").exists())

    def test_device_name_cannot_escape_staging_tree(self) -> None:
        run_id = "synthetic-traversal"
        self._prepare_raw(run_id, device="../escape")

        with self.assertRaisesRegex(PipelineError, "Unsafe"):
            run_offline_stages(
                self.layout,
                run_id,
                options=OfflinePipelineOptions(
                    interpolation_points=8,
                    arima_p_max=0,
                    arima_q_max=0,
                ),
            )

        escaped = self.layout.staged_output_path(
            run_id,
            "DataProcess1",
        ).parent / "escape.csv"
        self.assertFalse(escaped.exists())

    def test_options_reject_unsafe_resource_bounds(self) -> None:
        with self.assertRaises(ValueError):
            OfflinePipelineOptions(chunk_size=0)
        with self.assertRaises(ValueError):
            OfflinePipelineOptions(interpolation_points=1)
        with self.assertRaises(ValueError):
            OfflinePipelineOptions(arima_p_max=-1)


if __name__ == "__main__":
    unittest.main()
