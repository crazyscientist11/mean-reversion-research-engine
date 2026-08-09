import numpy as np
import pandas as pd
import pytest

from src.models.benchmark import rolling_zscore_benchmark


def test_benchmark_uses_prior_log_price_window_only() -> None:
    prices = pd.Series(np.exp([0.0, 1.0, 2.0, 5.0]), index=pd.date_range("2025-01-01", periods=4), name="AAA")
    result = rolling_zscore_benchmark(prices, window=3)
    date = prices.index[-1]
    assert result.expected_series.loc[date] == pytest.approx(1.0)
    assert result.parameters.loc[date, "baseline_std"] == pytest.approx(1.0)
    assert result.zscore_series.loc[date] == pytest.approx(4.0)


def test_benchmark_zero_variation_is_invalid() -> None:
    prices = pd.Series(np.repeat(100.0, 5), index=pd.date_range("2025-01-01", periods=5))
    result = rolling_zscore_benchmark(prices, window=3)
    assert not result.valid_observations.iloc[-1]
    assert np.isnan(result.zscore_series.iloc[-1])
