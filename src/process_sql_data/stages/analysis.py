"""Plotting and time-series classification extracted from legacy stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy
import pandas


class ClassificationKind(str, Enum):
    EMPTY = "empty"
    ZERO_VARIANCE = "zero_variance"
    WHITE_NOISE = "white_noise"
    SIGNAL = "signal"
    TEST_FAILED = "test_failed"


_LEGACY_MESSAGES = {
    ClassificationKind.EMPTY: "长度是0，不进一步分析",
    ClassificationKind.ZERO_VARIANCE: "方差是0，不进一步分析",
    ClassificationKind.WHITE_NOISE: "可能是白噪声，不进一步分析",
    ClassificationKind.SIGNAL: "非白噪声且方差非零",
    ClassificationKind.TEST_FAILED: "白噪声判定失败",
}


@dataclass(frozen=True)
class ClassificationResult:
    kind: ClassificationKind
    p_value: float | None

    @property
    def legacy_message(self) -> str:
        return _LEGACY_MESSAGES[self.kind]

    @property
    def should_continue(self) -> bool:
        return self.kind in {
            ClassificationKind.SIGNAL,
            ClassificationKind.TEST_FAILED,
        }


def classify_series(values: Iterable[float]) -> ClassificationResult:
    """Reproduce the variance/Ljung-Box decision with explicit outcomes."""

    series = pandas.Series(values, dtype="float64").dropna()
    if series.empty:
        return ClassificationResult(ClassificationKind.EMPTY, None)
    variance = series.var()
    if variance == 0:
        return ClassificationResult(ClassificationKind.ZERO_VARIANCE, None)

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(series, lags=[1], return_df=True)
        p_value = float(result["lb_pvalue"].iloc[0])
        if numpy.isnan(p_value):
            return ClassificationResult(ClassificationKind.TEST_FAILED, None)
    except Exception:
        return ClassificationResult(ClassificationKind.TEST_FAILED, None)

    kind = (
        ClassificationKind.WHITE_NOISE
        if p_value > 0.05
        else ClassificationKind.SIGNAL
    )
    return ClassificationResult(kind, p_value)


def plot_time_series(
    frame: pandas.DataFrame,
    *,
    time_column: str,
    value_column: str,
    output_path: Path | str,
) -> Path:
    """Write the legacy line chart through the non-interactive Agg backend."""

    _validate_plot_input(frame, time_column, value_column)
    matplotlib, pyplot = _plot_modules()
    times = pandas.to_datetime(frame[time_column], errors="raise")
    values = pandas.to_numeric(frame[value_column], errors="raise")
    output = _validate_plot_output(output_path)

    figure = pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    python_datetimes = times.array.to_pydatetime()
    axes.plot_date(matplotlib.dates.date2num(python_datetimes), values, "-")
    axes.set_xlabel("Time")
    axes.set_ylabel("Measurement")
    figure.savefig(output)
    pyplot.close(figure)
    return output


def plot_histogram(
    values: Iterable[float],
    *,
    output_path: Path | str,
    bins: int = 256,
) -> Path:
    """Write the historical 256-bin histogram without opening a GUI."""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("Histogram bins must be a positive integer.")
    _, pyplot = _plot_modules()
    numeric = pandas.to_numeric(pandas.Series(values), errors="raise")
    output = _validate_plot_output(output_path)

    figure = pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.hist(numeric, bins=bins)
    figure.savefig(output)
    pyplot.close(figure)
    return output


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["agg.path.chunksize"] = 10_000_000
    import matplotlib.dates
    import matplotlib.pyplot

    return matplotlib, matplotlib.pyplot


def _validate_plot_input(
    frame: pandas.DataFrame,
    time_column: str,
    value_column: str,
) -> None:
    missing = [
        name for name in (time_column, value_column) if name not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing plot column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Cannot plot an empty time series.")


def _validate_plot_output(output_path: Path | str) -> Path:
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Plot output must use the .png extension.")
    if not output.parent.is_dir():
        raise ValueError(f"Plot output directory does not exist: {output.parent}")
    if output.exists():
        raise ValueError(f"Plot output already exists: {output}")
    return output
