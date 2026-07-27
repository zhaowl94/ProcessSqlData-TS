"""Bounded ARIMA order selection using the supported statsmodels API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable
import warnings

import numpy
import pandas

from .transforms import TransformError


class ArimaStatus(str, Enum):
    ZERO_VARIANCE = "zero_variance"
    NO_MODEL = "no_model"
    SELECTED = "selected"


@dataclass(frozen=True)
class ArimaSelection:
    status: ArimaStatus
    order: tuple[int, int, int] | None
    bic: float | None
    tested_candidates: int
    failed_candidates: int

    @property
    def legacy_message(self) -> str:
        if self.status is ArimaStatus.ZERO_VARIANCE:
            return "方差为0，没有合适的ARIMA模型"
        if self.status is ArimaStatus.NO_MODEL:
            return "没有合适的ARIMA模型"
        assert self.order is not None
        p_value, _, q_value = self.order
        return f"最小的p值和q值为: {p_value}、{q_value}"


def select_arima_order(
    values: Iterable[float],
    *,
    p_max: int = 10,
    q_max: int = 10,
    max_differences: int = 3,
) -> ArimaSelection:
    """Select the lowest-BIC ARIMA order intended by legacy stage 11.

    The historical code used the ADF 5% critical value, then searched every
    ``(p, d, q)`` combination in row-major order. The old ARIMA implementation
    is no longer available; this function uses statsmodels' supported state
    space implementation and retains deterministic first-candidate tie
    breaking.
    """

    _validate_bound(p_max, "p_max")
    _validate_bound(q_max, "q_max")
    _validate_bound(max_differences, "max_differences")
    series = _finite_series(values)

    if series.empty:
        return _no_model()
    if series.var() == 0:
        return ArimaSelection(
            status=ArimaStatus.ZERO_VARIANCE,
            order=None,
            bic=None,
            tested_candidates=0,
            failed_candidates=0,
        )

    difference_order = _difference_order(
        series.to_numpy(dtype=float),
        max_differences=max_differences,
    )
    if difference_order is None:
        return _no_model()

    best_order: tuple[int, int, int] | None = None
    best_bic: float | None = None
    tested = 0
    failed = 0
    numeric = series.to_numpy(dtype=float)

    for p_value in range(p_max + 1):
        for q_value in range(q_max + 1):
            tested += 1
            order = (p_value, difference_order, q_value)
            try:
                candidate_bic = _candidate_bic(numeric, order)
            except Exception:
                # Individual grid points failed in the original implementation
                # too; other candidates must still be evaluated.
                failed += 1
                continue
            if not math.isfinite(candidate_bic):
                failed += 1
                continue
            if best_bic is None or candidate_bic < best_bic:
                best_bic = candidate_bic
                best_order = order

    if best_order is None:
        return ArimaSelection(
            status=ArimaStatus.NO_MODEL,
            order=None,
            bic=None,
            tested_candidates=tested,
            failed_candidates=failed,
        )
    return ArimaSelection(
        status=ArimaStatus.SELECTED,
        order=best_order,
        bic=best_bic,
        tested_candidates=tested,
        failed_candidates=failed,
    )


def _finite_series(values: Iterable[float]) -> pandas.Series:
    try:
        series = pandas.to_numeric(
            pandas.Series(list(values)),
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError) as exc:
        raise TransformError("ARIMA input must contain only numeric values.") from exc
    if not numpy.isfinite(series.to_numpy(dtype=float)).all():
        raise TransformError("ARIMA input must contain only finite values.")
    return series


def _difference_order(
    values: numpy.ndarray,
    *,
    max_differences: int,
) -> int | None:
    working = values
    for difference_order in range(max_differences + 1):
        try:
            if _is_stationary(working):
                return difference_order
        except (FloatingPointError, TypeError, ValueError):
            return None
        if difference_order == max_differences:
            break
        working = numpy.diff(working)
        if working.size == 0:
            return None
    return None


def _is_stationary(values: numpy.ndarray) -> bool:
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(values, autolag="AIC")
    return bool(result[0] < result[4]["5%"])


def _candidate_bic(
    values: numpy.ndarray,
    order: tuple[int, int, int],
) -> float:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = ARIMA(values, order=order).fit()
    return float(fitted.bic)


def _no_model() -> ArimaSelection:
    return ArimaSelection(
        status=ArimaStatus.NO_MODEL,
        order=None,
        bic=None,
        tested_candidates=0,
        failed_candidates=0,
    )


def _validate_bound(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
