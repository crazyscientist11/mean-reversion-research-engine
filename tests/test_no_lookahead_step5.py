from pandas.testing import assert_frame_equal, assert_series_equal

from src.models.pca_model import rolling_pca_residuals

from .test_pca_model import factor_prices


def test_future_prices_do_not_change_pca_outputs_through_cutoff() -> None:
    prices = factor_prices()
    cutoff = prices.index[130]
    changed = prices.copy()
    changed.loc[changed.index > cutoff, "S0"] *= 4.0
    original = rolling_pca_residuals(prices, window=60, n_components=1)
    altered = rolling_pca_residuals(changed, window=60, n_components=1)
    mask = original.dates <= cutoff
    for attribute in ("reconstructed_returns", "residual_returns", "residual_zscores"):
        assert_frame_equal(getattr(original, attribute).loc[mask], getattr(altered, attribute).loc[mask])
    assert_series_equal(original.component_counts.loc[mask], altered.component_counts.loc[mask])
