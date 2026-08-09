from src.models.pairs import peer_comparison

from .test_pairs import cointegrated_prices


def test_peer_comparison_is_a_transparent_research_table() -> None:
    prices = cointegrated_prices().rename(columns={"BBB": "GOOD"})
    prices["BAD"] = prices["GOOD"].iloc[::-1].to_numpy()
    table = peer_comparison(prices, target_ticker="AAA", candidate_peers=["GOOD", "BAD"], window=252)
    assert set(table["peer_ticker"]) == {"GOOD", "BAD"}
    assert {"cointegration_pvalue", "spread_volatility", "return_correlation", "research_rank"}.issubset(table.columns)
    assert table.iloc[0]["peer_ticker"] == "GOOD"
