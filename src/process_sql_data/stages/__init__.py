"""Pure, testable transformations corresponding to legacy pipeline stages."""

from .analysis import (
    ClassificationKind,
    ClassificationResult,
    classify_series,
    plot_histogram,
    plot_time_series,
)
from .arima import ArimaSelection, ArimaStatus, select_arima_order
from .transforms import (
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

__all__ = [
    "ArimaSelection",
    "ArimaStatus",
    "ClassificationKind",
    "ClassificationResult",
    "TransformError",
    "classify_series",
    "group_by_device",
    "interpolate_series",
    "linear_trend",
    "linear_trend_from_timestamps",
    "merge_and_sort",
    "parse_feature_rows",
    "parse_payload",
    "plot_histogram",
    "plot_time_series",
    "remove_outliers",
    "select_arima_order",
    "split_numeric_series",
    "summary_statistics",
    "timestamps_from_strings",
]
