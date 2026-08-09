"""Rolling prior-window linear detrending of log prices."""
import numpy as np
import pandas as pd

from .common import ResidualModelResult, prior_sample_zscore, validate_price_series


def detrended_log_price_residual(prices: pd.Series, *, window: int = 126, zscore_window: int | None = None) -> ResidualModelResult:
    """Fit p_i = alpha + beta*time_i through t-1 and forecast log price at t."""
    if window < 2:
        raise ValueError("window must be at least 2")
    z_window = zscore_window or window
    checked = validate_price_series(prices)
    observed = np.log(checked).rename("log_price")
    expected = pd.Series(np.nan, index=checked.index, name="trend_implied_log_price")
    alpha = pd.Series(np.nan, index=checked.index, name="alpha")
    beta = pd.Series(np.nan, index=checked.index, name="beta")
    r_squared = pd.Series(np.nan, index=checked.index, name="r_squared")
    time = np.arange(window, dtype=float)
    design = np.column_stack((np.ones(window), time))
    for position in range(window, len(observed)):
        history = observed.iloc[position - window:position]
        if history.isna().any():
            continue
        coefficients, _, _, _ = np.linalg.lstsq(design, history.to_numpy(), rcond=None)
        fitted = design @ coefficients
        total = float(np.sum((history.to_numpy() - history.mean()) ** 2))
        unexplained = float(np.sum((history.to_numpy() - fitted) ** 2))
        date = observed.index[position]
        alpha.loc[date], beta.loc[date] = coefficients
        expected.loc[date] = coefficients[0] + coefficients[1] * window
        r_squared.loc[date] = 1.0 if np.isclose(total, 0.0) and np.isclose(unexplained, 0.0) else (1.0 - unexplained / total if total > 0 else np.nan)
    residual = (observed - expected).rename("trend_residual")
    zscore = prior_sample_zscore(residual, z_window).rename("residual_zscore")
    valid = (expected.notna() & zscore.notna()).rename("valid")
    return ResidualModelResult(
        model_name="Detrended log-price residual", target_ticker=str(prices.name or "target"), dates=checked.index,
        observed_series=observed, expected_series=expected, residual_series=residual, zscore_series=zscore,
        estimation_window=window, valid_observations=valid,
        parameters=pd.DataFrame({"alpha": alpha, "beta": beta}, index=checked.index), r_squared_series=r_squared,
        metadata={"expected_price": np.exp(expected), "zscore_window": z_window, "uses_prior_observations_only": True},
    )
