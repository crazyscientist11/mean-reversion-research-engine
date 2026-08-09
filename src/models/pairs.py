"""Strictly prior-window static pair and cointegration research models."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


class PairResearchState(str, Enum):
    TARGET_RELATIVELY_HIGH = "TARGET_RELATIVELY_HIGH"
    TARGET_RELATIVELY_LOW = "TARGET_RELATIVELY_LOW"
    NEUTRAL = "NEUTRAL"
    INVALID = "INVALID"


@dataclass(frozen=True)
class PairPositionRepresentation:
    """Theoretical regression-exposure representation, not a dollar allocation."""
    target_units: float
    peer_units: float
    regression_beta: float
    normalization: str = "1 target log-price regression unit and -beta peer units"


@dataclass(frozen=True)
class CurrentPairSnapshot:
    as_of_date: pd.Timestamp
    target_ticker: str
    peer_ticker: str
    alpha: float | None
    beta: float | None
    spread: float | None
    spread_mean: float | None
    spread_standard_deviation: float | None
    spread_zscore: float | None
    r_squared: float | None
    cointegration_statistic: float | None
    cointegration_pvalue: float | None
    return_correlation: float | None
    relative_state: PairResearchState
    estimation_window: int
    valid: bool
    invalid_reason: str | None


@dataclass
class PairModelResult:
    model_name: str
    target_ticker: str
    peer_ticker: str
    dates: pd.DatetimeIndex
    alpha_series: pd.Series
    beta_series: pd.Series
    spread_series: pd.Series
    spread_mean_series: pd.Series
    spread_std_series: pd.Series
    zscore_series: pd.Series
    r_squared_series: pd.Series
    cointegration_statistic_series: pd.Series
    cointegration_pvalue_series: pd.Series
    return_correlation_series: pd.Series
    valid_observations: pd.Series
    invalid_reasons: pd.Series
    estimation_window: int
    zscore_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def current_snapshot(self) -> CurrentPairSnapshot:
        date = self.dates[-1]
        valid = bool(self.valid_observations.loc[date])
        zscore = _finite_or_none(self.zscore_series.loc[date])
        return CurrentPairSnapshot(
            as_of_date=date, target_ticker=self.target_ticker, peer_ticker=self.peer_ticker,
            alpha=_finite_or_none(self.alpha_series.loc[date]), beta=_finite_or_none(self.beta_series.loc[date]),
            spread=_finite_or_none(self.spread_series.loc[date]), spread_mean=_finite_or_none(self.spread_mean_series.loc[date]),
            spread_standard_deviation=_finite_or_none(self.spread_std_series.loc[date]), spread_zscore=zscore,
            r_squared=_finite_or_none(self.r_squared_series.loc[date]),
            cointegration_statistic=_finite_or_none(self.cointegration_statistic_series.loc[date]),
            cointegration_pvalue=_finite_or_none(self.cointegration_pvalue_series.loc[date]),
            return_correlation=_finite_or_none(self.return_correlation_series.loc[date]),
            relative_state=pair_research_state(zscore, self.zscore_threshold) if valid else PairResearchState.INVALID,
            estimation_window=self.estimation_window, valid=valid,
            invalid_reason=None if valid else str(self.invalid_reasons.loc[date]),
        )


def normalized_pair_position(beta: float) -> PairPositionRepresentation:
    """Represent the log-price regression as +1 target and -beta peer units.

    This is a theoretical regression-exposure normalization. It does not convert
    units into shares, dollars, beta-neutral exposure, or market neutrality.
    """
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    return PairPositionRepresentation(target_units=1.0, peer_units=-float(beta), regression_beta=float(beta))


def pair_research_state(zscore: float | None, threshold: float = 2.0) -> PairResearchState:
    if threshold <= 0:
        raise ValueError("z-score threshold must be positive")
    if zscore is None or not np.isfinite(zscore):
        return PairResearchState.INVALID
    if zscore >= threshold:
        return PairResearchState.TARGET_RELATIVELY_HIGH
    if zscore <= -threshold:
        return PairResearchState.TARGET_RELATIVELY_LOW
    return PairResearchState.NEUTRAL


def rolling_pair_model(prices: pd.DataFrame, *, target_ticker: str, peer_ticker: str, window: int = 252, zscore_threshold: float = 2.0, epsilon: float = 1e-12) -> PairModelResult:
    """Fit each pair relationship through t-1, then evaluate the observed t price.

    For each date the prior-window regression supplies alpha, beta, historical
    residual mean/std (sample ``ddof=1``), Engle–Granger statistic/p-value, and
    return correlation. The current observation is excluded from every estimate.
    """
    if window < 20:
        raise ValueError("window must be at least 20 for rolling cointegration diagnostics")
    if zscore_threshold <= 0:
        raise ValueError("z-score threshold must be positive")
    if target_ticker == peer_ticker:
        raise ValueError("target and peer tickers must differ")
    for ticker in (target_ticker, peer_ticker):
        if ticker not in prices.columns:
            raise ValueError(f"missing ticker column: {ticker}")
    selected = prices.loc[:, [target_ticker, peer_ticker]].copy(deep=True)
    _validate_pair_prices(selected)
    logs = np.log(selected)
    index = selected.index
    series = {name: pd.Series(np.nan, index=index, dtype=float, name=name) for name in (
        "alpha", "beta", "spread", "spread_mean", "spread_std", "zscore", "r_squared", "cointegration_statistic", "cointegration_pvalue", "return_correlation"
    )}
    valid = pd.Series(False, index=index, name="valid")
    reasons = pd.Series("Insufficient prior complete observations.", index=index, dtype=object, name="invalid_reason")
    for position in range(window, len(index)):
        date = index[position]
        history = logs.iloc[position - window:position]
        current = logs.iloc[position]
        if history.isna().any().any() or current.isna().any():
            reasons.loc[date] = "Missing target or peer price in the prior window or current observation."
            continue
        response = history[target_ticker].to_numpy()
        peer = history[peer_ticker].to_numpy()
        design = np.column_stack((np.ones(window), peer))
        coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        fitted = design @ coefficients
        historical_spread = response - fitted
        spread_mean = float(np.mean(historical_spread))
        spread_std = float(np.std(historical_spread, ddof=1))
        total = float(np.sum((response - response.mean()) ** 2))
        unexplained = float(np.sum((response - fitted) ** 2))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cointegration_statistic, cointegration_pvalue, _ = coint(response, peer, trend="c", autolag="aic")
        except (ValueError, np.linalg.LinAlgError):
            reasons.loc[date] = "Cointegration test could not be estimated for the prior window."
            continue
        historical_returns = history.diff().dropna(how="any")
        correlation = historical_returns[target_ticker].corr(historical_returns[peer_ticker])
        spread = float(current[target_ticker] - coefficients[0] - coefficients[1] * current[peer_ticker])
        series["alpha"].loc[date], series["beta"].loc[date] = coefficients
        series["spread"].loc[date] = spread
        series["spread_mean"].loc[date], series["spread_std"].loc[date] = spread_mean, spread_std
        series["r_squared"].loc[date] = 1.0 if np.isclose(total, 0.0) and np.isclose(unexplained, 0.0) else (1.0 - unexplained / total if total > 0 else np.nan)
        series["cointegration_statistic"].loc[date] = cointegration_statistic
        series["cointegration_pvalue"].loc[date] = cointegration_pvalue
        series["return_correlation"].loc[date] = correlation
        if not np.isfinite(spread_std) or abs(spread_std) <= epsilon:
            reasons.loc[date] = "Prior-window spread standard deviation is effectively zero."
            continue
        series["zscore"].loc[date] = (spread - spread_mean) / spread_std
        if not np.isfinite(cointegration_statistic) or not np.isfinite(cointegration_pvalue):
            reasons.loc[date] = "Cointegration test returned a non-finite result."
            continue
        valid.loc[date] = True
        reasons.loc[date] = ""
    return PairModelResult(
        model_name="Rolling static pair spread", target_ticker=target_ticker, peer_ticker=peer_ticker, dates=index,
        alpha_series=series["alpha"], beta_series=series["beta"], spread_series=series["spread"],
        spread_mean_series=series["spread_mean"], spread_std_series=series["spread_std"], zscore_series=series["zscore"],
        r_squared_series=series["r_squared"], cointegration_statistic_series=series["cointegration_statistic"],
        cointegration_pvalue_series=series["cointegration_pvalue"], return_correlation_series=series["return_correlation"],
        valid_observations=valid, invalid_reasons=reasons, estimation_window=window, zscore_threshold=zscore_threshold,
        metadata={"uses_prior_observations_only": True, "regression": "log(target) = alpha + beta * log(peer)", "spread_std": "sample (ddof=1)"},
    )


def peer_comparison(prices: pd.DataFrame, *, target_ticker: str, candidate_peers: Iterable[str], as_of_date: pd.Timestamp | None = None, window: int = 252, zscore_threshold: float = 2.0) -> pd.DataFrame:
    """Return a transparent prior-window research table; it does not choose a pair."""
    cutoff_prices = prices.loc[:as_of_date].copy(deep=True) if as_of_date is not None else prices.copy(deep=True)
    records: list[dict[str, Any]] = []
    for peer in dict.fromkeys(candidate_peers):
        if peer == target_ticker:
            continue
        try:
            result = rolling_pair_model(cutoff_prices, target_ticker=target_ticker, peer_ticker=peer, window=window, zscore_threshold=zscore_threshold)
            snapshot = result.current_snapshot()
            records.append({
                "peer_ticker": peer, "valid": snapshot.valid, "invalid_reason": snapshot.invalid_reason,
                "cointegration_pvalue": snapshot.cointegration_pvalue, "cointegration_statistic": snapshot.cointegration_statistic,
                "spread_volatility": snapshot.spread_standard_deviation, "spread_zscore": snapshot.spread_zscore,
                "r_squared": snapshot.r_squared, "return_correlation": snapshot.return_correlation,
                "valid_observation_count": int(result.valid_observations.sum()), "relative_state": snapshot.relative_state.value,
            })
        except ValueError as exc:
            records.append({"peer_ticker": peer, "valid": False, "invalid_reason": str(exc), "cointegration_pvalue": np.nan, "cointegration_statistic": np.nan, "spread_volatility": np.nan, "spread_zscore": np.nan, "r_squared": np.nan, "return_correlation": np.nan, "valid_observation_count": 0, "relative_state": PairResearchState.INVALID.value})
    table = pd.DataFrame(records)
    if table.empty:
        return pd.DataFrame(columns=["peer_ticker", "valid", "invalid_reason", "cointegration_pvalue", "cointegration_statistic", "spread_volatility", "spread_zscore", "r_squared", "return_correlation", "valid_observation_count", "relative_state", "research_rank"])
    table = table.sort_values(["valid", "cointegration_pvalue", "peer_ticker"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    table["research_rank"] = np.where(table["valid"], np.arange(1, len(table) + 1), np.nan)
    return table


def _validate_pair_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must use a unique, ascending DatetimeIndex")
    numeric = prices.apply(pd.to_numeric, errors="raise")
    if (numeric.dropna() <= 0).any().any():
        raise ValueError("prices must be strictly positive")


def _finite_or_none(value: object) -> float | None:
    return float(value) if value is not None and pd.notna(value) and np.isfinite(value) else None
