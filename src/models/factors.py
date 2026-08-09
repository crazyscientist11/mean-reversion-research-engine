"""Prior-window market and optional sector factor regression on log returns."""
import numpy as np
import pandas as pd

from .common import ResidualModelResult, prior_sample_zscore


def factor_residual_model(prices: pd.DataFrame, *, target_ticker: str, market_ticker: str, sector_ticker: str | None = None, window: int = 126, zscore_window: int | None = None, residual_state_window: int = 20) -> ResidualModelResult:
    """Fit returns through t-1, then evaluate t using contemporaneous factor returns."""
    if window < 3:
        raise ValueError("window must be at least 3")
    if residual_state_window < 1:
        raise ValueError("residual_state_window must be positive")
    tickers = [target_ticker, market_ticker] + ([sector_ticker] if sector_ticker is not None else [])
    if len(set(tickers)) != len(tickers):
        raise ValueError("target, market, and sector tickers must be distinct")
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"missing ticker columns: {', '.join(missing)}")
    selected = prices.loc[:, tickers].copy(deep=True)
    if not isinstance(selected.index, pd.DatetimeIndex) or not selected.index.is_unique or not selected.index.is_monotonic_increasing:
        raise ValueError("prices must use unique, ascending dates")
    selected = selected.apply(pd.to_numeric, errors="raise").astype(float)
    if (selected.dropna() <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    returns = np.log(selected / selected.shift(1))
    observed = returns[target_ticker].rename("actual_target_log_return")
    factor_columns = [market_ticker] + ([sector_ticker] if sector_ticker is not None else [])
    expected = pd.Series(np.nan, index=prices.index, name="predicted_target_log_return")
    alpha = pd.Series(np.nan, index=prices.index, name="alpha")
    beta_market = pd.Series(np.nan, index=prices.index, name="beta_market")
    beta_sector = pd.Series(np.nan, index=prices.index, name="beta_sector")
    r_squared = pd.Series(np.nan, index=prices.index, name="r_squared")
    for position in range(window + 1, len(returns)):
        history = returns.iloc[position - window:position].dropna(how="any")
        current = returns.iloc[position]
        if len(history) < window or current.isna().any():
            continue
        regressors = history[factor_columns].to_numpy()
        design = np.column_stack((np.ones(len(history)), regressors))
        response = history[target_ticker].to_numpy()
        coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        fitted = design @ coefficients
        total = float(np.sum((response - response.mean()) ** 2))
        unexplained = float(np.sum((response - fitted) ** 2))
        date = returns.index[position]
        alpha.loc[date] = coefficients[0]
        beta_market.loc[date] = coefficients[1]
        if sector_ticker is not None:
            beta_sector.loc[date] = coefficients[2]
        expected.loc[date] = coefficients[0] + float(np.dot(coefficients[1:], current[factor_columns].to_numpy()))
        r_squared.loc[date] = 1.0 if np.isclose(total, 0.0) and np.isclose(unexplained, 0.0) else (1.0 - unexplained / total if total > 0 else np.nan)
    residual = (observed - expected).rename("factor_residual_return")
    zscore = prior_sample_zscore(residual, zscore_window or window).rename("residual_zscore")
    state = residual.rolling(residual_state_window, min_periods=residual_state_window).sum().rename("accumulated_factor_residual")
    valid = (expected.notna() & zscore.notna()).rename("valid")
    return ResidualModelResult(
        model_name="Market/sector factor residual", target_ticker=target_ticker, dates=prices.index,
        observed_series=observed, expected_series=expected, residual_series=residual, zscore_series=zscore,
        estimation_window=window, valid_observations=valid,
        parameters=pd.DataFrame({"alpha": alpha, "beta_market": beta_market, "beta_sector": beta_sector}, index=prices.index),
        r_squared_series=r_squared, state_series=state,
        metadata={"market_ticker": market_ticker, "sector_ticker": sector_ticker, "residual_state_window": residual_state_window, "uses_prior_observations_only": True},
    )
