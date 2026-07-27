from __future__ import annotations

import math
import time
import unittest

import numpy
import pandas

from process_sql_data.run_state import RunMode
from process_sql_data.stages import (
    TransformError,
    group_by_device,
    interpolate_series,
    linear_trend,
    linear_trend_from_timestamps,
    merge_and_sort,
    parse_feature_rows,
    parse_payload,
    remove_outliers,
    split_numeric_series,
    summary_statistics,
    timestamps_from_strings,
)


class TransformTests(unittest.TestCase):
    def test_raw_pages_group_by_device_without_losing_page_order(self) -> None:
        first = pandas.DataFrame(
            [
                {
                    "sn": "device-b",
                    "date": "t1",
                    "data": '{"x":"1"}',
                    "slave_id": 1,
                    "type_id": 2,
                },
                {
                    "sn": "device-a",
                    "date": "t2",
                    "data": '{"x":"2"}',
                    "slave_id": 1,
                    "type_id": 2,
                },
            ]
        )
        second = pandas.DataFrame(
            [
                {
                    "sn": "device-a",
                    "date": "t3",
                    "data": '{"x":"3"}',
                    "slave_id": 1,
                    "type_id": 2,
                }
            ]
        )

        grouped = group_by_device([first, second])

        self.assertEqual(list(grouped), ["device-a", "device-b"])
        self.assertEqual(grouped["device-a"]["date"].tolist(), ["t2", "t3"])
        self.assertEqual(
            list(grouped["device-a"].columns),
            ["sn", "date", "data", "slave_id", "type_id"],
        )

    def test_payload_parser_preserves_both_legacy_encodings(self) -> None:
        self.assertEqual(
            parse_payload(
                "{u'temperature':u'12.5',u'pressure':u'8'}",
                mode=RunMode.LEGACY,
            ),
            {"temperature": "12.5", "pressure": "8"},
        )
        self.assertEqual(
            parse_payload(
                '{"temperature":"12.5","pressure":"8"}',
                mode=RunMode.LEGACY,
            ),
            {"temperature": "12.5", "pressure": "8"},
        )

    def test_corrected_payload_supports_spaces_and_colons_in_values(self) -> None:
        self.assertEqual(
            parse_payload(
                '{"label": "zone: east", "temperature": 12.5}',
                mode=RunMode.CORRECTED,
            ),
            {"label": "zone: east", "temperature": "12.5"},
        )
        with self.assertRaises(TransformError):
            parse_payload("not a mapping", mode=RunMode.CORRECTED)
        self.assertEqual(
            parse_payload("not a mapping", mode=RunMode.LEGACY),
            {},
        )

    def test_feature_rows_expand_dynamic_attributes_in_first_seen_order(self) -> None:
        frame = pandas.DataFrame(
            [
                {
                    "sn": "device-a",
                    "date": "2020-01-01 00:00:00",
                    "data": '{"temperature":"12.5"}',
                    "slave_id": 1,
                    "type_id": 2,
                },
                {
                    "sn": "device-a",
                    "date": "2020-01-01 00:00:01",
                    "data": '{"pressure":"8","temperature":"13"}',
                    "slave_id": 1,
                    "type_id": 2,
                },
            ]
        )

        result = parse_feature_rows(frame, mode=RunMode.LEGACY)

        self.assertEqual(
            list(result.columns),
            [
                "sn",
                "date",
                "temperature",
                "pressure",
                "slave_id",
                "type_id",
            ],
        )
        self.assertTrue(pandas.isna(result.loc[0, "pressure"]))
        self.assertEqual(result.loc[1, "temperature"], "13")

    def test_merge_and_sort_is_stable_for_equal_dates(self) -> None:
        first = pandas.DataFrame(
            {
                "date": ["2020-01-02 00:00:00", "2020-01-01 00:00:00"],
                "marker": ["late", "first"],
            }
        )
        second = pandas.DataFrame(
            {
                "date": ["2020-01-01 00:00:00"],
                "marker": ["second"],
            }
        )

        result = merge_and_sort([first, second])

        self.assertEqual(result["marker"].tolist(), ["first", "second", "late"])

    def test_split_series_distinguishes_legacy_and_corrected_numeric_values(self) -> None:
        frame = pandas.DataFrame(
            {
                "sn": ["a", "a", "a"],
                "date": ["t1", "t2", "t3"],
                "temperature": [1.5, "2.5", None],
                "slave_id": [1, 1, 1],
                "type_id": [2, 2, 2],
            }
        )

        legacy = split_numeric_series(frame, mode=RunMode.LEGACY)
        corrected = split_numeric_series(frame, mode=RunMode.CORRECTED)

        self.assertEqual(legacy["temperature"]["temperature"].tolist(), [1.5])
        self.assertEqual(
            corrected["temperature"]["temperature"].tolist(),
            [1.5, 2.5],
        )

    def test_outlier_filter_preserves_source_order(self) -> None:
        frame = pandas.DataFrame(
            {
                "date": [f"t{index}" for index in range(10)],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
            }
        )

        result = remove_outliers(frame, value_column="value")

        self.assertEqual(result["value"].tolist(), list(range(1, 10)))
        self.assertEqual(result["date"].tolist(), [f"t{i}" for i in range(9)])

    def test_timestamp_conversion_matches_legacy_local_time(self) -> None:
        frame = pandas.DataFrame(
            {
                "date": ["2020-01-01 00:00:00"],
                "value": [12.5],
            }
        )

        result = timestamps_from_strings(
            frame,
            time_column="date",
            value_column="value",
        )

        expected = time.mktime(
            time.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        )
        self.assertEqual(result.to_dict(orient="records"), [{"time": expected, "data": 12.5}])

    def test_interpolation_has_legacy_shape_and_linear_values(self) -> None:
        frame = pandas.DataFrame(
            {
                "date": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:10",
                ],
                "value": [0.0, 10.0],
            }
        )

        result = interpolate_series(
            frame,
            time_column="date",
            value_column="value",
            mode=RunMode.CORRECTED,
            points=3,
        )

        self.assertEqual(list(result.columns), [0, 1])
        numpy.testing.assert_allclose(result[1].to_numpy(), [0.0, 5.0, 10.0])

    def test_summary_statistics_use_sample_variance(self) -> None:
        result = summary_statistics(
            {
                "device-a.csv": pandas.Series([1.0, 2.0, 3.0]),
                "device-b.csv": pandas.Series([4.0, 4.0]),
            }
        )

        self.assertEqual(list(result.columns), ["var", "mean", "file"])
        self.assertTrue(math.isclose(result.loc[0, "var"], 1.0))
        self.assertTrue(math.isclose(result.loc[0, "mean"], 2.0))
        self.assertTrue(math.isclose(result.loc[1, "var"], 0.0))

    def test_linear_trend_matches_known_slope(self) -> None:
        frame = pandas.DataFrame(
            {
                "date": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:01",
                    "2020-01-01 00:00:02",
                ],
                "value": [3.0, 5.0, 7.0],
            }
        )

        coefficient, intercept = linear_trend(
            frame,
            time_column="date",
            value_column="value",
        )
        first_timestamp = time.mktime(
            time.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        )

        self.assertTrue(math.isclose(coefficient, 2.0, rel_tol=1e-9))
        self.assertTrue(
            math.isclose(
                coefficient * first_timestamp + intercept,
                3.0,
                abs_tol=1e-6,
            )
        )

    def test_linear_trend_accepts_interpolated_numeric_timestamps(self) -> None:
        frame = pandas.DataFrame(
            {
                "time": [100.0, 101.0, 102.0],
                "data": [3.0, 5.0, 7.0],
            }
        )

        coefficient, intercept = linear_trend_from_timestamps(
            frame,
            time_column="time",
            value_column="data",
        )

        self.assertTrue(math.isclose(coefficient, 2.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(intercept, -197.0, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
