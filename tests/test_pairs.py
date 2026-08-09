import numpy as np
import pandas as pd
import pytest

from src.models.pairs import PairResearchState, normalized_pair_position, pair_research_state, rolling_pair_model


def cointegrated_prices(length: int = 360) -> pd.DataFrame:
    generator = np.random.default_rng(202503)
    log_peer = np.cumsum(generator.normal(0.0002, 0.01, size=length)) + 4.0
    noise = np.zeros(length)
    for index in range(1, length):
        noise[index] = 0.65 * noise[index - 1] + generator.normal(0, 0.008)
    log_target = 0.35 + 1.7 * log_peer + noise
    return pd.DataFrame({"AAA": np.exp(log_target), "BBB": np.exp(log_peer)}, index=pd.date_range("2020-01-01", periods=length))


def independent_random_walk_prices(length: int = 500) -> pd.DataFrame:
    generator = np.random.default_rng(711)
    return pd.DataFrame({
        "AAA": np.exp(4.0 + np.cumsum(generator.normal(0.001, 0.012, size=length))),
        "BBB": np.exp(4.5 + np.cumsum(generator.normal(-0.001, 0.011, size=length))),
    }, index=pd.date_range("2020-01-01", periods=length))


def test_cointegrated_pair_recovers_beta_and_cointegration() -> None:
    result = rolling_pair_model(cointegrated_prices(), target_ticker="AAA", peer_ticker="BBB", window=252)
    snapshot = result.current_snapshot()
    assert snapshot.valid
    assert snapshot.beta == pytest.approx(1.7, abs=0.12)
    assert snapshot.cointegration_pvalue is not None and snapshot.cointegration_pvalue < 0.05
    assert snapshot.spread_zscore is not None


def test_independent_random_walks_do_not_show_strong_cointegration_evidence() -> None:
    result = rolling_pair_model(independent_random_walk_prices(), target_ticker="AAA", peer_ticker="BBB", window=300)
    pvalue = result.current_snapshot().cointegration_pvalue
    assert pvalue is not None and pvalue > 0.10


def test_pair_position_and_relative_states_are_explicit() -> None:
    position = normalized_pair_position(1.5)
    assert position.target_units == 1.0
    assert position.peer_units == -1.5
    assert pair_research_state(2.1, threshold=2.0) is PairResearchState.TARGET_RELATIVELY_HIGH
    assert pair_research_state(-2.1, threshold=2.0) is PairResearchState.TARGET_RELATIVELY_LOW
    assert pair_research_state(0.2, threshold=2.0) is PairResearchState.NEUTRAL
