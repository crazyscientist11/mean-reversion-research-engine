import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.models.benchmark import rolling_zscore_benchmark
from src.models.drift import detrended_log_price_residual
from src.models.factors import factor_residual_model


def prices(length: int = 50) -> pd.DataFrame:
    market = 0.004 * np.sin(np.arange(length) * 0.31)
    sector = 0.003 * np.cos(np.arange(length) * 0.17)
    target = 0.001 + 1.1 * market + 0.4 * sector + 0.0002 * np.sin(np.arange(length))
    return 100 * np.exp(pd.DataFrame({"AAA": target, "SPY": market, "XLK": sector}, index=pd.date_range("2025-01-01", periods=length)).cumsum())


def assert_unchanged_through(original, changed, cutoff) -> None:
    mask = original.dates <= cutoff
    assert_series_equal(original.expected_series.loc[mask], changed.expected_series.loc[mask])
    assert_series_equal(original.residual_series.loc[mask], changed.residual_series.loc[mask])
    assert_series_equal(original.zscore_series.loc[mask], changed.zscore_series.loc[mask])
    assert_frame_equal(original.parameters.loc[mask], changed.parameters.loc[mask])


def test_future_changes_do_not_affect_benchmark_or_trend() -> None:
    frame = prices()
    cutoff = frame.index[35]
    changed = frame.copy()
    changed.loc[changed.index > cutoff, "AAA"] *= 20
    benchmark = rolling_zscore_benchmark(frame["AAA"], window=10)
    altered_benchmark = rolling_zscore_benchmark(changed["AAA"], window=10)
    trend = detrended_log_price_residual(frame["AAA"], window=10, zscore_window=5)
    altered_trend = detrended_log_price_residual(changed["AAA"], window=10, zscore_window=5)
    assert_unchanged_through(benchmark, altered_benchmark, cutoff)
    assert_unchanged_through(trend, altered_trend, cutoff)


def test_future_changes_do_not_affect_factor_outputs_or_current_coefficients() -> None:
    frame = prices()
    cutoff = frame.index[35]
    changed = frame.copy()
    changed.loc[changed.index > cutoff, ["AAA", "SPY", "XLK"]] *= [3.0, 1.8, 2.2]
    original = factor_residual_model(frame, target_ticker="AAA", market_ticker="SPY", sector_ticker="XLK", window=10, zscore_window=5)
    altered = factor_residual_model(changed, target_ticker="AAA", market_ticker="SPY", sector_ticker="XLK", window=10, zscore_window=5)
    assert_unchanged_through(original, altered, cutoff)


def test_current_observation_is_excluded_from_each_estimation() -> None:
    frame = prices(35)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("AAA")] *= np.exp(0.5)
    benchmark = rolling_zscore_benchmark(frame["AAA"], window=10)
    altered_benchmark = rolling_zscore_benchmark(changed["AAA"], window=10)
    trend = detrended_log_price_residual(frame["AAA"], window=10, zscore_window=5)
    altered_trend = detrended_log_price_residual(changed["AAA"], window=10, zscore_window=5)
    factors = factor_residual_model(frame, target_ticker="AAA", market_ticker="SPY", sector_ticker="XLK", window=10, zscore_window=5)
    altered_factors = factor_residual_model(changed, target_ticker="AAA", market_ticker="SPY", sector_ticker="XLK", window=10, zscore_window=5)
    date = frame.index[-1]
    assert benchmark.expected_series.loc[date] == altered_benchmark.expected_series.loc[date]
    assert trend.expected_series.loc[date] == altered_trend.expected_series.loc[date]
    assert_frame_equal(factors.parameters.loc[[date]], altered_factors.parameters.loc[[date]])
