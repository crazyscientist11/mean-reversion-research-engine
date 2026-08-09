"""Simple prior-window rolling z-score benchmark on log prices."""
import numpy as np
import pandas as pd

from .common import ResidualModelResult, validate_price_series


def rolling_zscore_benchmark(prices: pd.Series, *, window: int = 60, epsilon: float = 1e-12) -> ResidualModelResult:
    """Evaluate each log price against mean/std estimated strictly through t-1.

    Standard deviation is the sample standard deviation (``ddof=1``).
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    checked = validate_price_series(prices)
    observed = np.log(checked).rename("log_price")
    baseline = observed.shift(1).rolling(window, min_periods=window).mean().rename("prior_log_price_mean")
    prior_std = observed.shift(1).rolling(window, min_periods=window).std(ddof=1).rename("prior_log_price_std")
    safe_std = prior_std.where(prior_std.abs() > epsilon)
    residual = (observed - baseline).rename("log_price_residual")
    zscore = (residual / safe_std).rename("zscore")
    valid = (baseline.notna() & safe_std.notna() & observed.notna() & zscore.notna()).rename("valid")
    return ResidualModelResult(
        model_name="Simple rolling z-score benchmark", target_ticker=str(prices.name or "target"), dates=checked.index,
        observed_series=observed, expected_series=baseline, residual_series=residual, zscore_series=zscore,
        estimation_window=window, valid_observations=valid,
        parameters=pd.DataFrame({"baseline_std": prior_std}, index=checked.index),
        metadata={"input": "log price", "standard_deviation": "sample (ddof=1)", "uses_prior_observations_only": True},
    )
