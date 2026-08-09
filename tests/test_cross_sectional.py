from src.models.cross_sectional import CrossSectionalState, cross_sectional_residuals

from .test_pca_model import factor_prices
from src.models.pca_model import rolling_pca_residuals


def test_cross_sectional_ranking_finds_positive_idiosyncratic_extreme() -> None:
    prices = factor_prices()
    prices.iloc[-1, 0] *= 1.06
    pca = rolling_pca_residuals(prices, window=60, n_components=1)
    result = cross_sectional_residuals(pca.residual_returns, minimum_universe_size=4, zscore_threshold=1.5)
    table = result.current_table().set_index("ticker")
    assert table.loc["S0", "zscore"] > 1.5
    assert table.loc["S0", "relative_state"] == CrossSectionalState.RELATIVELY_HIGH.value
    assert table.loc["S0", "percentile"] == 1.0
