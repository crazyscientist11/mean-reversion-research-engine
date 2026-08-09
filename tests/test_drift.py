import numpy as np
import pandas as pd
import pytest

from src.models.drift import detrended_log_price_residual


def test_linear_log_price_trend_is_recovered() -> None:
    log_price = 1.0 + 0.01 * np.arange(20)
    prices = pd.Series(np.exp(log_price), index=pd.date_range("2025-01-01", periods=20), name="AAA")
    result = detrended_log_price_residual(prices, window=8, zscore_window=3)
    date = prices.index[-1]
    assert result.expected_series.loc[date] == pytest.approx(log_price[-1])
    assert result.residual_series.loc[date] == pytest.approx(0.0, abs=1e-12)
    assert result.parameters.loc[date, "beta"] == pytest.approx(0.01)
    assert result.r_squared_series.loc[date] == pytest.approx(1.0)


def test_trend_shock_has_positive_residual() -> None:
    log_price = 1.0 + 0.01 * np.arange(16)
    log_price[-1] += 0.2
    prices = pd.Series(np.exp(log_price), index=pd.date_range("2025-01-01", periods=16), name="AAA")
    result = detrended_log_price_residual(prices, window=8, zscore_window=3)
    assert result.residual_series.iloc[-1] == pytest.approx(0.2)
