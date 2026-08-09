import numpy as np
from pandas.testing import assert_series_equal

from src.models.pairs import peer_comparison, rolling_pair_model

from .test_pairs import cointegrated_prices


def test_future_changes_do_not_affect_rolling_pair_outputs_through_cutoff() -> None:
    prices = cointegrated_prices()
    cutoff = prices.index[300]
    changed = prices.copy()
    changed.loc[changed.index > cutoff, "AAA"] *= 5.0
    original = rolling_pair_model(prices, target_ticker="AAA", peer_ticker="BBB", window=100)
    altered = rolling_pair_model(changed, target_ticker="AAA", peer_ticker="BBB", window=100)
    mask = original.dates <= cutoff
    for attribute in ("alpha_series", "beta_series", "spread_series", "zscore_series", "cointegration_statistic_series", "cointegration_pvalue_series"):
        assert_series_equal(getattr(original, attribute).loc[mask], getattr(altered, attribute).loc[mask])


def test_current_observation_is_excluded_from_pair_estimation_and_ranking() -> None:
    prices = cointegrated_prices(280)
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc("AAA")] *= np.exp(0.5)
    original = rolling_pair_model(prices, target_ticker="AAA", peer_ticker="BBB", window=100)
    altered = rolling_pair_model(changed, target_ticker="AAA", peer_ticker="BBB", window=100)
    date = prices.index[-1]
    assert original.alpha_series.loc[date] == altered.alpha_series.loc[date]
    assert original.beta_series.loc[date] == altered.beta_series.loc[date]
    assert original.cointegration_pvalue_series.loc[date] == altered.cointegration_pvalue_series.loc[date]
    prices["ALT"] = prices["BBB"].iloc[::-1].to_numpy()
    changed["ALT"] = changed["BBB"].iloc[::-1].to_numpy()
    original_table = peer_comparison(prices, target_ticker="AAA", candidate_peers=["BBB", "ALT"], window=100)
    altered_table = peer_comparison(changed, target_ticker="AAA", candidate_peers=["BBB", "ALT"], window=100)
    assert_series_equal(original_table["cointegration_pvalue"], altered_table["cointegration_pvalue"])
    assert_series_equal(original_table["research_rank"], altered_table["research_rank"])
