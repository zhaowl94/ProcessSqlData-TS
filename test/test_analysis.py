from __future__ import annotations

from pathlib import Path
import unittest

import numpy
import pandas

from process_sql_data.stages import (
    ClassificationKind,
    classify_series,
    plot_histogram,
    plot_time_series,
)
from test import temporary_directory


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(temporary_directory()))

    def test_empty_and_zero_variance_series_stop_analysis(self) -> None:
        empty = classify_series([])
        constant = classify_series([2.0, 2.0, 2.0])

        self.assertEqual(empty.kind, ClassificationKind.EMPTY)
        self.assertEqual(empty.legacy_message, "长度是0，不进一步分析")
        self.assertFalse(empty.should_continue)
        self.assertEqual(constant.kind, ClassificationKind.ZERO_VARIANCE)
        self.assertEqual(constant.legacy_message, "方差是0，不进一步分析")
        self.assertFalse(constant.should_continue)

    def test_autocorrelated_series_is_retained_as_signal(self) -> None:
        result = classify_series(numpy.arange(1.0, 101.0))

        self.assertEqual(result.kind, ClassificationKind.SIGNAL)
        self.assertIsNotNone(result.p_value)
        self.assertLess(result.p_value, 0.05)
        self.assertEqual(result.legacy_message, "非白噪声且方差非零")
        self.assertTrue(result.should_continue)

    def test_seeded_white_noise_is_classified_without_random_test_flakiness(self) -> None:
        # This fixed seed has a comfortably non-significant lag-1 statistic.
        # A seed is still used so the fixture remains reproducible.
        values = numpy.random.default_rng(0).normal(size=500)
        result = classify_series(values)

        self.assertEqual(result.kind, ClassificationKind.WHITE_NOISE)
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.legacy_message, "可能是白噪声，不进一步分析")

    def test_time_series_plot_writes_one_png_without_overwrite(self) -> None:
        frame = pandas.DataFrame(
            {
                "date": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:01",
                ],
                "value": [1.0, 2.0],
            }
        )
        output = self.root / "series.png"

        result = plot_time_series(
            frame,
            time_column="date",
            value_column="value",
            output_path=output,
        )

        self.assertEqual(result, output)
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        with self.assertRaisesRegex(ValueError, "already exists"):
            plot_time_series(
                frame,
                time_column="date",
                value_column="value",
                output_path=output,
            )

    def test_histogram_writes_png_and_validates_extension(self) -> None:
        output = self.root / "histogram.png"
        plot_histogram([1.0, 2.0, 3.0], output_path=output, bins=8)

        self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        with self.assertRaisesRegex(ValueError, ".png"):
            plot_histogram(
                [1.0],
                output_path=self.root / "histogram.jpg",
            )


if __name__ == "__main__":
    unittest.main()
