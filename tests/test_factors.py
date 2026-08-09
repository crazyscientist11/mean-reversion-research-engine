import numpy as np
import pandas as pd
import pytest

from src.models.factors import factor_residual_model


def synthetic_factor_prices(length: int = 80) -> pd.DataFrame:
    market = 0.005 * np.sin(np.arange(length) * 0.37)
    sector = 0.004 * np.cos(np.arange(length) * 0.23)
    target = 0.001 + 1.2 * market + 0.5 * sector
    returns = pd.DataFrame({"AAA": target, "SPY": market, "XLK": sector}, index=pd.date_range("2025-01-01", periods=length))
    return 100.0 * np.exp(returns.cumsum())


def test_factor_coefficients_and_residual_are_recovered() -> None:
    result = factor_residual_model(synthetic_factor_prices(), target_ticker="AAA", market_ticker="SPY", sector_ticker="XLK", window=20, zscore_window=5)
    date = result.dates[-1]
    assert result.parameters.loc[date, "alpha"] == pytest.approx(0.001, abs=1e-10)
    assert result.parameters.loc[date, "beta_market"] == pytest.approx(1.2, abs=1e-10)
    assert result.parameters.loc[date, "beta_sector"] == pytest.approx(0.5, abs=1e-10)
    assert result.residual_series.loc[date] == pytest.approx(0.0, abs=1e-10)


def test_factor_shock_is_a_positive_residual() -> None:
    prices = synthetic_factor_prices()
    prices.iloc[-1, prices.columns.get_loc("AAA")] *= np.exp(0.03)
    result = factor_residual_model(prices, target_ticker="AAA", market_ticker="SPY", sector_ticker="XLK", window=20, zscore_window=5)
    assert result.residual_series.iloc[-1] == pytest.approx(0.03, abs=1e-10)


def test_market_only_mode_leaves_sector_beta_empty() -> None:
    prices = synthetic_factor_prices().drop(columns="XLK")
    result = factor_residual_model(prices, target_ticker="AAA", market_ticker="SPY", window=20, zscore_window=5)
    assert result.parameters["beta_sector"].isna().all()
