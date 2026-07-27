"""DataFrame transformations extracted from the Python 2 monolith."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
import json
import math
import re
import time
from typing import Any

import numpy
import pandas
import scipy.interpolate

from ..run_state import RunMode


METADATA_COLUMNS = ("sn", "date", "slave_id", "type_id")
RAW_COLUMNS = ("sn", "date", "data", "slave_id", "type_id")
_PYTHON_UNICODE_PAIR = re.compile(r"u'[\w\W]*':u'[\w\W]*'")
_JSON_PAIR = re.compile(r'"[\w\W]*":"[\w\W]*"')


class TransformError(ValueError):
    """Raised when corrected mode encounters invalid input data."""


def group_by_device(
    frames: Iterable[pandas.DataFrame],
) -> dict[str, pandas.DataFrame]:
    """Group raw database pages into stable per-device tables."""

    materialized = [frame.loc[:, RAW_COLUMNS].copy() for frame in frames]
    if not materialized:
        return {}
    for frame in materialized:
        _require_columns(frame, RAW_COLUMNS)

    combined = pandas.concat(materialized, ignore_index=True, sort=False)
    if combined.empty:
        return {}

    device_values = sorted(
        combined["sn"].dropna().unique().tolist(),
        key=lambda value: str(value),
    )
    return {
        str(device): combined.loc[
            combined["sn"] == device,
            RAW_COLUMNS,
        ].reset_index(drop=True)
        for device in device_values
    }


def parse_payload(value: Any, *, mode: RunMode) -> dict[str, str]:
    """Parse the legacy text-encoded feature dictionary."""

    if not isinstance(value, str) or len(value) < 2:
        if mode is RunMode.LEGACY:
            return {}
        raise TransformError("Feature payload must be a non-empty string.")

    if mode is RunMode.CORRECTED:
        parsed = _parse_corrected_payload(value)
        if not isinstance(parsed, Mapping):
            raise TransformError("Feature payload must decode to an object.")
        return {str(key): str(item) for key, item in parsed.items()}

    text = value[1:-1].replace(" ", "")
    parts = text.split(",")
    if not parts:
        return {}

    result: dict[str, str] = {}
    if _PYTHON_UNICODE_PAIR.fullmatch(parts[0]):
        for part in parts:
            pieces = part.split(":")
            if len(pieces) != 2:
                return {}
            result[pieces[0][2:-1]] = pieces[1][2:-1]
    elif _JSON_PAIR.fullmatch(parts[0]):
        for part in parts:
            pieces = part.split(":")
            if len(pieces) != 2:
                return {}
            result[pieces[0][1:-1]] = pieces[1][1:-1]
    return result


def parse_feature_rows(
    frame: pandas.DataFrame,
    *,
    mode: RunMode,
) -> pandas.DataFrame:
    """Expand the `data` payload while preserving legacy metadata positions."""

    _require_columns(frame, METADATA_COLUMNS + ("data",))
    records: list[dict[str, Any]] = []
    attribute_order: list[str] = []

    for row in frame.to_dict(orient="records"):
        attributes = parse_payload(row["data"], mode=mode)
        if not attributes:
            continue
        for key in attributes:
            if key not in attribute_order:
                attribute_order.append(key)
        record: dict[str, Any] = {
            "sn": row["sn"],
            "date": row["date"],
            **attributes,
            "slave_id": row["slave_id"],
            "type_id": row["type_id"],
        }
        records.append(record)

    columns = ["sn", "date", *attribute_order, "slave_id", "type_id"]
    return pandas.DataFrame.from_records(records, columns=columns)


def merge_and_sort(frames: Iterable[pandas.DataFrame]) -> pandas.DataFrame:
    """Concatenate feature chunks and stably sort them by legacy date text."""

    materialized = [frame for frame in frames if not frame.empty]
    if not materialized:
        return pandas.DataFrame()
    merged = pandas.concat(materialized, ignore_index=True, sort=False)
    _require_columns(merged, ("date",))
    return merged.sort_values("date", kind="mergesort").reset_index(drop=True)


def split_numeric_series(
    frame: pandas.DataFrame,
    *,
    mode: RunMode,
) -> dict[str, pandas.DataFrame]:
    """Split one device table into separate non-null numeric time series."""

    _require_columns(frame, METADATA_COLUMNS)
    output: dict[str, pandas.DataFrame] = {}
    for column in frame.columns:
        if column in METADATA_COLUMNS:
            continue
        selected = frame.loc[:, ["date", column]].dropna()
        if mode is RunMode.LEGACY:
            mask = selected[column].map(lambda value: isinstance(value, float))
            numeric = selected.loc[mask].copy()
        else:
            converted = pandas.to_numeric(selected[column], errors="coerce")
            numeric = selected.loc[converted.notna()].copy()
            numeric[column] = converted.loc[converted.notna()].astype(float)
        output[str(column)] = numeric.reset_index(drop=True)
    return output


def remove_outliers(
    frame: pandas.DataFrame,
    *,
    value_column: str,
    hard_upper_limit: float = 62_100,
) -> pandas.DataFrame:
    """Apply the legacy quartile rule and hard upper measurement limit."""

    _require_columns(frame, (value_column,))
    if frame.empty:
        return frame.copy()
    values = pandas.to_numeric(frame[value_column], errors="coerce")
    valid_frame = frame.loc[values.notna()].copy()
    values = values.loc[values.notna()]
    if valid_frame.empty:
        return valid_frame

    sorted_values = numpy.sort(values.to_numpy(dtype=float))
    length = len(sorted_values)
    q25 = _legacy_quantile(sorted_values, 1, length)
    q75 = _legacy_quantile(sorted_values, 3, length)
    interquartile_range = q75 - q25
    lower = q25 - 1.5 * interquartile_range
    upper = q75 + 1.5 * interquartile_range
    keep = values.between(lower, upper, inclusive="both") & (
        values <= hard_upper_limit
    )
    return valid_frame.loc[keep].copy().reset_index(drop=True)


def timestamps_from_strings(
    frame: pandas.DataFrame,
    *,
    time_column: str,
    value_column: str,
) -> pandas.DataFrame:
    """Convert legacy local datetime strings to epoch seconds."""

    _require_columns(frame, (time_column, value_column))
    timestamps = [
        time.mktime(time.strptime(str(value), "%Y-%m-%d %H:%M:%S"))
        for value in frame[time_column]
    ]
    return pandas.DataFrame(
        {
            "time": timestamps,
            "data": frame[value_column].to_numpy(copy=True),
        }
    )


def interpolate_series(
    frame: pandas.DataFrame,
    *,
    time_column: str,
    value_column: str,
    mode: RunMode,
    points: int = 40_000,
) -> pandas.DataFrame:
    """Linearly interpolate one series using the historical fixed point count."""

    if points < 2:
        raise TransformError("Interpolation requires at least two output points.")
    _require_columns(frame, (time_column, value_column))
    if frame.empty:
        return pandas.DataFrame(columns=[0, 1])

    working = pandas.DataFrame(
        {
            "time": [
                time.mktime(time.strptime(str(value), "%Y-%m-%d %H:%M:%S"))
                for value in frame[time_column]
            ],
            "data": pandas.to_numeric(frame[value_column], errors="coerce"),
        }
    ).dropna()
    if working.empty:
        return pandas.DataFrame(columns=[0, 1])

    if mode is RunMode.CORRECTED:
        working = (
            working.sort_values("time", kind="mergesort")
            .drop_duplicates("time", keep="first")
            .reset_index(drop=True)
        )
        time_values = working["time"].to_numpy(dtype=float)
        data_values = working["data"].to_numpy(dtype=float)
    else:
        first_positions = numpy.flatnonzero(
            ~working["time"].duplicated(keep="first").to_numpy()
        )
        time_values = numpy.unique(working["time"].to_numpy(dtype=float))
        data_values = working["data"].to_numpy(dtype=float)[first_positions]

    if len(time_values) <= 1:
        return pandas.DataFrame(columns=[0, 1])
    if len(time_values) != len(data_values):
        raise TransformError(
            "Legacy timestamp ordering produced incompatible interpolation arrays."
        )

    new_time = numpy.linspace(
        float(time_values.min()),
        float(time_values.max()),
        points,
    )
    interpolator = scipy.interpolate.interp1d(
        time_values,
        data_values,
        kind="slinear",
    )
    new_data = interpolator(new_time)
    return pandas.DataFrame(numpy.column_stack([new_time, new_data]))


def summary_statistics(
    named_series: Mapping[str, pandas.Series],
) -> pandas.DataFrame:
    """Return the legacy sample variance, mean, and filename columns."""

    records = [
        {
            "var": series.var(),
            "mean": series.mean(),
            "file": filename,
        }
        for filename, series in named_series.items()
    ]
    return pandas.DataFrame.from_records(
        records,
        columns=["var", "mean", "file"],
    )


def linear_trend(
    frame: pandas.DataFrame,
    *,
    time_column: str,
    value_column: str,
) -> tuple[float, float]:
    """Fit the legacy one-variable linear regression without unused Lomb-Scargle."""

    _require_columns(frame, (time_column, value_column))
    if len(frame) < 2:
        raise TransformError("Linear trend requires at least two observations.")

    from sklearn.linear_model import LinearRegression

    timestamps = numpy.array(
        [
            time.mktime(time.strptime(str(value), "%Y-%m-%d %H:%M:%S"))
            for value in frame[time_column]
        ],
        dtype=float,
    ).reshape(-1, 1)
    values = pandas.to_numeric(frame[value_column], errors="raise").to_numpy(
        dtype=float
    )
    model = LinearRegression()
    model.fit(timestamps, values)
    return float(model.coef_[0]), float(model.intercept_)


def linear_trend_from_timestamps(
    frame: pandas.DataFrame,
    *,
    time_column: str,
    value_column: str,
) -> tuple[float, float]:
    """Fit the stage-14 trend after stage 10 has produced numeric timestamps."""

    _require_columns(frame, (time_column, value_column))
    if len(frame) < 2:
        raise TransformError("Linear trend requires at least two observations.")

    from sklearn.linear_model import LinearRegression

    timestamps = pandas.to_numeric(
        frame[time_column],
        errors="raise",
    ).to_numpy(dtype=float)
    values = pandas.to_numeric(
        frame[value_column],
        errors="raise",
    ).to_numpy(dtype=float)
    if not numpy.isfinite(timestamps).all() or not numpy.isfinite(values).all():
        raise TransformError("Linear trend requires finite values.")

    model = LinearRegression()
    model.fit(timestamps.reshape(-1, 1), values)
    return float(model.coef_[0]), float(model.intercept_)


def _parse_corrected_payload(value: str) -> Any:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    raise TransformError("Feature payload is neither JSON nor a Python literal.")


def _legacy_quantile(
    sorted_values: numpy.ndarray,
    numerator: int,
    length: int,
) -> float:
    position = numerator / 4.0 * length
    lower_index = max(0, int(math.floor(position)))
    upper_index = min(length - 1, int(math.ceil(position)))
    return float(
        (sorted_values[lower_index] + sorted_values[upper_index]) / 2.0
    )


def _require_columns(
    frame: pandas.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TransformError(f"Missing required column(s): {', '.join(missing)}")
