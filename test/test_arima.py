from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy

from process_sql_data.stages import (
    ArimaStatus,
    TransformError,
    select_arima_order,
)


class ArimaTests(unittest.TestCase):
    def test_zero_variance_short_circuits_without_fitting(self) -> None:
        with patch(
            "process_sql_data.stages.arima._candidate_bic"
        ) as candidate_bic:
            result = select_arima_order([3.0, 3.0, 3.0])

        candidate_bic.assert_not_called()
        self.assertEqual(result.status, ArimaStatus.ZERO_VARIANCE)
        self.assertIsNone(result.order)
        self.assertEqual(
            result.legacy_message,
            "方差为0，没有合适的ARIMA模型",
        )

    def test_grid_uses_first_lowest_bic_candidate(self) -> None:
        candidate_values = {
            (0, 1, 0): 20.0,
            (0, 1, 1): 10.0,
            (1, 1, 0): 10.0,
            (1, 1, 1): 15.0,
        }

        with (
            patch(
                "process_sql_data.stages.arima._difference_order",
                return_value=1,
            ),
            patch(
                "process_sql_data.stages.arima._candidate_bic",
                side_effect=lambda _values, order: candidate_values[order],
            ),
        ):
            result = select_arima_order(
                [1.0, 2.0, 4.0, 8.0],
                p_max=1,
                q_max=1,
            )

        self.assertEqual(result.status, ArimaStatus.SELECTED)
        self.assertEqual(result.order, (0, 1, 1))
        self.assertEqual(result.bic, 10.0)
        self.assertEqual(result.tested_candidates, 4)
        self.assertEqual(result.failed_candidates, 0)
        self.assertEqual(result.legacy_message, "最小的p值和q值为: 0、1")

    def test_failed_and_nonfinite_candidates_are_skipped(self) -> None:
        def candidate(_values, order):
            if order == (0, 0, 0):
                raise ValueError("unsupported")
            return numpy.nan

        with (
            patch(
                "process_sql_data.stages.arima._difference_order",
                return_value=0,
            ),
            patch(
                "process_sql_data.stages.arima._candidate_bic",
                side_effect=candidate,
            ),
        ):
            result = select_arima_order(
                [1.0, 2.0, 3.0],
                p_max=0,
                q_max=1,
            )

        self.assertEqual(result.status, ArimaStatus.NO_MODEL)
        self.assertEqual(result.tested_candidates, 2)
        self.assertEqual(result.failed_candidates, 2)
        self.assertEqual(result.legacy_message, "没有合适的ARIMA模型")

    def test_supported_statsmodels_api_is_exercised(self) -> None:
        values = numpy.random.default_rng(0).normal(size=80)

        result = select_arima_order(
            values,
            p_max=0,
            q_max=0,
            max_differences=1,
        )

        self.assertEqual(result.status, ArimaStatus.SELECTED)
        self.assertEqual(result.order, (0, 0, 0))
        self.assertTrue(numpy.isfinite(result.bic))

    def test_invalid_values_and_bounds_fail_closed(self) -> None:
        with self.assertRaises(TransformError):
            select_arima_order([1.0, numpy.inf])
        with self.assertRaises(TransformError):
            select_arima_order([1.0, "not-a-number"])
        with self.assertRaises(ValueError):
            select_arima_order([1.0, 2.0], p_max=-1)


if __name__ == "__main__":
    unittest.main()
