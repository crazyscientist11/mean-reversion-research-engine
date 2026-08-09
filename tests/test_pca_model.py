import numpy as np
import pandas as pd

from src.models.pca_model import rolling_pca_residuals


def factor_prices(length: int = 180) -> pd.DataFrame:
    generator = np.random.default_rng(51)
    common = generator.normal(0, 0.012, size=length)
    loadings = np.array([0.8, 1.0, 1.2, 0.9, 1.1])
    noise = generator.normal(0, 0.001, size=(length, len(loadings)))
    returns = common[:, None] * loadings + noise
    return pd.DataFrame(np.exp(np.cumsum(returns, axis=0)) * 100.0, index=pd.date_range("2020-01-01", periods=length), columns=[f"S{i}" for i in range(len(loadings))])


def test_pca_identifies_dominant_common_factor_and_shock_sign() -> None:
    prices = factor_prices()
    prices.iloc[-1, 0] *= np.exp(0.05)
    result = rolling_pca_residuals(prices, window=60, n_components=1, residual_state_window=10)
    last_explained = result.explained_variance_by_component.dropna(how="all").iloc[-1, 0]
    assert last_explained > 0.8
    assert result.residual_returns.iloc[-1, 0] > 0.03
    assert abs(result.actual_returns.iloc[-1, 0] - result.reconstructed_returns.iloc[-1, 0]) > 0.03
    assert result.accumulated_residual_state.iloc[-1, 0] == result.accumulated_residual_state.iloc[-1, 0]
