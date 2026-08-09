"""Shared typed results and validation for Step 2 statistical research models."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ResidualModelResult:
    """Time-indexed research output; NaN marks dates without adequate prior data."""
    model_name: str
    target_ticker: str
    dates: pd.DatetimeIndex
    observed_series: pd.Series
    expected_series: pd.Series
    residual_series: pd.Series
    zscore_series: pd.Series
    estimation_window: int
    valid_observations: pd.Series
    parameters: pd.DataFrame = field(default_factory=pd.DataFrame)
    r_squared_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    state_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metadata: dict[str, Any] = field(default_factory=dict)

    def current_snapshot(self) -> "CurrentModelSnapshot":
        date = self.dates[-1]
        valid = bool(self.valid_observations.loc[date])
        return CurrentModelSnapshot(
            as_of_date=date, target_ticker=self.target_ticker, model_name=self.model_name,
            current_observed_value=_finite_or_none(self.observed_series.loc[date]),
            current_expected_value=_finite_or_none(self.expected_series.loc[date]),
            current_residual=_finite_or_none(self.residual_series.loc[date]),
            current_zscore=_finite_or_none(self.zscore_series.loc[date]), window=self.estimation_window,
            valid=valid, invalid_reason=None if valid else "Insufficient prior complete observations or zero residual variation.",
        )


@dataclass(frozen=True)
class CurrentModelSnapshot:
    """The latest model-time statistical result, not a recommendation."""
    as_of_date: pd.Timestamp
    target_ticker: str
    model_name: str
    current_observed_value: float | None
    current_expected_value: float | None
    current_residual: float | None
    current_zscore: float | None
    window: int
    valid: bool
    invalid_reason: str | None


def validate_price_series(prices: pd.Series, *, name: str = "prices") -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and sorted ascending")
    result = pd.to_numeric(prices.copy(deep=True), errors="raise").astype(float)
    if (result.dropna() <= 0).any():
        raise ValueError(f"{name} must contain strictly positive prices")
    return result


def prior_sample_zscore(values: pd.Series, window: int, *, epsilon: float = 1e-12) -> pd.Series:
    """Standardize each value using only preceding values; sample std uses ddof=1."""
    if window < 2:
        raise ValueError("window must be at least 2 for sample standard deviation")
    prior_mean = values.shift(1).rolling(window, min_periods=window).mean()
    prior_std = values.shift(1).rolling(window, min_periods=window).std(ddof=1)
    return ((values - prior_mean) / prior_std.where(prior_std.abs() > epsilon)).rename("zscore")


def _finite_or_none(value: object) -> float | None:
    return float(value) if value is not None and pd.notna(value) and np.isfinite(value) else None
