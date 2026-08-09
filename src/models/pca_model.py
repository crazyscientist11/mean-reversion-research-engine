"""Prior-window PCA common-factor reconstruction on daily log returns."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAResidualResult:
    tickers: tuple[str, ...]
    dates: pd.DatetimeIndex
    actual_returns: pd.DataFrame
    reconstructed_returns: pd.DataFrame
    residual_returns: pd.DataFrame
    accumulated_residual_state: pd.DataFrame
    residual_zscores: pd.DataFrame
    valid_observations: pd.Series
    invalid_reasons: pd.Series
    estimation_window: int
    component_counts: pd.Series
    explained_variance_by_component: pd.DataFrame
    cumulative_explained_variance: pd.DataFrame
    latest_loadings: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def rolling_pca_residuals(prices: pd.DataFrame, *, window: int = 252, n_components: int | float = 1, residual_state_window: int = 20, epsilon: float = 1e-12) -> PCAResidualResult:
    """Fit scaler/PCA through t-1, then project and reconstruct the return at t."""
    if window < 3 or residual_state_window < 1:
        raise ValueError("window must be at least 3 and residual_state_window must be positive")
    _validate_prices(prices)
    if isinstance(n_components, int) and not 1 <= n_components <= len(prices.columns):
        raise ValueError("integer n_components must be between 1 and the universe size")
    if isinstance(n_components, float) and not 0.0 < n_components <= 1.0:
        raise ValueError("explained-variance n_components must be in (0, 1]")
    tickers = tuple(str(ticker) for ticker in prices.columns)
    returns = np.log(prices / prices.shift(1)).astype(float)
    frames = {name: pd.DataFrame(np.nan, index=prices.index, columns=tickers, dtype=float) for name in ("reconstructed", "residual", "zscore")}
    component_counts = pd.Series(np.nan, index=prices.index, name="component_count")
    valid = pd.Series(False, index=prices.index, name="valid")
    reasons = pd.Series("Insufficient prior complete returns.", index=prices.index, dtype=object, name="invalid_reason")
    latest_loadings = pd.DataFrame()
    latest_explained = np.array([])
    for position in range(window + 1, len(returns)):
        date = returns.index[position]
        training = returns.iloc[position - window:position]
        current = returns.iloc[position]
        if training.isna().any().any() or current.isna().any():
            reasons.loc[date] = "Missing return in training window or current observation."
            continue
        mean = training.mean(axis=0)
        scale = training.std(axis=0, ddof=0)
        if (scale.abs() <= epsilon).any() or not np.all(np.isfinite(scale)):
            reasons.loc[date] = "A training return standard deviation is effectively zero."
            continue
        standardized_training = ((training - mean) / scale).to_numpy()
        try:
            pca = PCA(n_components=n_components).fit(standardized_training)
        except ValueError as exc:
            reasons.loc[date] = f"PCA fitting failed: {exc}"
            continue
        standardized_current = ((current - mean) / scale).to_numpy()
        scores = pca.transform(standardized_current.reshape(1, -1))
        reconstructed_standardized = pca.inverse_transform(scores).reshape(-1)
        reconstructed = pd.Series(reconstructed_standardized * scale.to_numpy() + mean.to_numpy(), index=tickers)
        historical_reconstructed = pca.inverse_transform(pca.transform(standardized_training)) * scale.to_numpy() + mean.to_numpy()
        historical_residuals = training.to_numpy() - historical_reconstructed
        residual_mean = historical_residuals.mean(axis=0)
        residual_std = historical_residuals.std(axis=0, ddof=1)
        residual = current - reconstructed
        zscore = (residual.to_numpy() - residual_mean) / np.where(np.abs(residual_std) > epsilon, residual_std, np.nan)
        frames["reconstructed"].loc[date] = reconstructed
        frames["residual"].loc[date] = residual
        frames["zscore"].loc[date] = zscore
        component_counts.loc[date] = pca.n_components_
        valid.loc[date] = np.isfinite(zscore).all()
        reasons.loc[date] = "" if valid.loc[date] else "A PCA residual standard deviation is effectively zero."
        latest_loadings = pd.DataFrame(pca.components_.T, index=tickers, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
        latest_explained = pca.explained_variance_ratio_
    accumulated = frames["residual"].rolling(residual_state_window, min_periods=residual_state_window).sum()
    component_columns = [f"PC{i + 1}" for i in range(len(latest_explained))]
    explained = pd.DataFrame(np.nan, index=prices.index, columns=component_columns)
    cumulative = pd.DataFrame(np.nan, index=prices.index, columns=component_columns)
    if len(latest_explained):
        last_valid = valid[valid].index[-1]
        explained.loc[last_valid] = latest_explained
        cumulative.loc[last_valid] = np.cumsum(latest_explained)
    return PCAResidualResult(
        tickers=tickers, dates=prices.index, actual_returns=returns, reconstructed_returns=frames["reconstructed"], residual_returns=frames["residual"],
        accumulated_residual_state=accumulated, residual_zscores=frames["zscore"], valid_observations=valid, invalid_reasons=reasons,
        estimation_window=window, component_counts=component_counts, explained_variance_by_component=explained, cumulative_explained_variance=cumulative,
        latest_loadings=latest_loadings, metadata={"input": "daily log returns", "uses_prior_training_window_only": True, "residual_state_window": residual_state_window},
    )


def _validate_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must use a unique, ascending DatetimeIndex")
    if len(prices.columns) < 2:
        raise ValueError("PCA requires at least two securities")
    numeric = prices.apply(pd.to_numeric, errors="raise")
    if (numeric.dropna() <= 0).any().any():
        raise ValueError("prices must be strictly positive")
