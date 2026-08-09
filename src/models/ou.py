"""Reusable trading-day Ornstein–Uhlenbeck dynamics estimation for state series."""
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OU_METHOD = Literal["AR1", "MLE"]


@dataclass(frozen=True)
class OUFitResult:
    method: OU_METHOD
    theta: float | None
    kappa: float | None
    sigma: float | None
    half_life: float | None
    reversion_75_time: float | None
    reversion_90_time: float | None
    log_likelihood: float | None
    aic: float | None
    bic: float | None
    n_observations: int
    dt: float
    time_unit: str
    valid: bool
    invalid_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OUComparisonResult:
    ar1: OUFitResult
    mle: OUFitResult

    def as_table(self) -> pd.DataFrame:
        rows = []
        for result in (self.ar1, self.mle):
            rows.append({
                "method": result.method, "valid": result.valid, "theta": result.theta,
                "kappa_per_trading_day": result.kappa, "sigma_per_sqrt_trading_day": result.sigma,
                "half_life_trading_days": result.half_life, "reversion_75_trading_days": result.reversion_75_time,
                "reversion_90_trading_days": result.reversion_90_time, "log_likelihood": result.log_likelihood,
                "aic": result.aic, "bic": result.bic, "invalid_reason": result.invalid_reason,
            })
        table = pd.DataFrame(rows)
        if self.ar1.valid and self.mle.valid:
            table["theta_difference_ar1_minus_mle"] = self.ar1.theta - self.mle.theta  # type: ignore[operator]
            table["kappa_difference_ar1_minus_mle"] = self.ar1.kappa - self.mle.kappa  # type: ignore[operator]
            table["sigma_difference_ar1_minus_mle"] = self.ar1.sigma - self.mle.sigma  # type: ignore[operator]
            table["half_life_difference_ar1_minus_mle"] = self.ar1.half_life - self.mle.half_life  # type: ignore[operator]
        return table


def fit_ou_ar1(values: pd.Series | Sequence[float] | np.ndarray, *, dt: float = 1.0, time_unit: str = "trading days") -> OUFitResult:
    """Fit X[t+1] = a + b X[t] and map a valid 0<b<1 fit to OU parameters."""
    data, failure = _validate_values(values, dt)
    if failure:
        return _invalid("AR1", len(data), dt, time_unit, failure)
    previous, following = data[:-1], data[1:]
    design = np.column_stack((np.ones(len(previous)), previous))
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, following, rcond=None)
    except np.linalg.LinAlgError:
        return _invalid("AR1", len(data), dt, time_unit, "AR(1) regression failed.")
    intercept, autoregressive_coefficient = map(float, coefficients)
    if not (_EPSILON < autoregressive_coefficient < 1.0 - _EPSILON):
        return _invalid("AR1", len(data), dt, time_unit, "AR(1) coefficient is not strictly inside the mean-reverting interval (0, 1).", {"intercept": intercept, "ar1_coefficient": autoregressive_coefficient})
    innovations = following - design @ coefficients
    innovation_variance = float(np.mean(innovations**2))
    if not np.isfinite(innovation_variance) or innovation_variance <= _EPSILON:
        return _invalid("AR1", len(data), dt, time_unit, "AR(1) innovation variance is effectively zero.", {"intercept": intercept, "ar1_coefficient": autoregressive_coefficient})
    kappa = -np.log(autoregressive_coefficient) / dt
    theta = intercept / (1.0 - autoregressive_coefficient)
    sigma = np.sqrt(innovation_variance * 2.0 * kappa / (1.0 - autoregressive_coefficient**2))
    if not all(np.isfinite(value) and value > 0 for value in (kappa, sigma)) or not np.isfinite(theta):
        return _invalid("AR1", len(data), dt, time_unit, "AR(1)-derived OU parameters are non-finite.")
    log_likelihood = _conditional_log_likelihood(data, theta, kappa, sigma, dt)
    return _valid_result("AR1", theta, kappa, sigma, log_likelihood, len(data), dt, time_unit, {"intercept": intercept, "ar1_coefficient": autoregressive_coefficient, "innovation_variance": innovation_variance})


def fit_ou_mle(values: pd.Series | Sequence[float] | np.ndarray, *, dt: float = 1.0, time_unit: str = "trading days") -> OUFitResult:
    """Estimate theta, kappa, and sigma by exact conditional transition likelihood."""
    data, failure = _validate_values(values, dt)
    if failure:
        return _invalid("MLE", len(data), dt, time_unit, failure)
    ar1_start = fit_ou_ar1(data, dt=dt, time_unit=time_unit)
    sample_std = float(np.std(data, ddof=1))
    start_theta = ar1_start.theta if ar1_start.valid else float(np.mean(data))
    start_kappa = ar1_start.kappa if ar1_start.valid else 0.1 / dt
    start_sigma = ar1_start.sigma if ar1_start.valid else max(sample_std * np.sqrt(2.0 * start_kappa), 1e-4)
    initial = np.array([start_theta, np.log(start_kappa), np.log(start_sigma)], dtype=float)
    bounds = [(None, None), (np.log(1e-8), np.log(100.0 / dt)), (np.log(1e-10), np.log(max(sample_std * 1000.0, 1.0)))]
    try:
        optimization = minimize(_negative_log_likelihood_transformed, initial, args=(data, dt), method="L-BFGS-B", bounds=bounds)
    except Exception as exc:  # scipy may raise on numerical optimizer failures
        return _invalid("MLE", len(data), dt, time_unit, f"OU optimizer raised {type(exc).__name__}.")
    if not optimization.success or not np.isfinite(optimization.fun) or not np.all(np.isfinite(optimization.x)):
        return _invalid("MLE", len(data), dt, time_unit, f"OU optimizer failed: {optimization.message}")
    theta, log_kappa, log_sigma = optimization.x
    kappa, sigma = float(np.exp(log_kappa)), float(np.exp(log_sigma))
    if not np.isfinite(theta) or not np.isfinite(kappa) or not np.isfinite(sigma) or kappa <= 0 or sigma <= 0:
        return _invalid("MLE", len(data), dt, time_unit, "OU optimizer returned invalid parameters.")
    return _valid_result("MLE", float(theta), kappa, sigma, -float(optimization.fun), len(data), dt, time_unit, {"optimizer_success": True, "optimizer_message": str(optimization.message), "optimizer_iterations": int(optimization.nit)})


def fit_ou_both(values: pd.Series | Sequence[float] | np.ndarray, *, dt: float = 1.0, time_unit: str = "trading days") -> OUComparisonResult:
    """Fit independent AR(1) and exact-MLE estimators to the same generic state."""
    return OUComparisonResult(ar1=fit_ou_ar1(values, dt=dt, time_unit=time_unit), mle=fit_ou_mle(values, dt=dt, time_unit=time_unit))


def simulate_ou(*, theta: float, kappa: float, sigma: float, n_observations: int, dt: float = 1.0, x0: float | None = None, seed: int = 0) -> np.ndarray:
    """Generate deterministic testing data from exact OU transitions; never market data."""
    if n_observations < 2 or kappa <= 0 or sigma <= 0 or dt <= 0:
        raise ValueError("n_observations >= 2, kappa > 0, sigma > 0, and dt > 0 are required")
    decay = np.exp(-kappa * dt)
    transition_std = np.sqrt(_transition_variance(kappa, sigma, dt))
    generator = np.random.default_rng(seed)
    result = np.empty(n_observations, dtype=float)
    result[0] = theta if x0 is None else x0
    for index in range(1, n_observations):
        result[index] = theta + (result[index - 1] - theta) * decay + generator.normal(0.0, transition_std)
    return result


def conditional_expectation(fit: OUFitResult, current_value: float, horizon: float) -> float:
    """OU conditional expectation at a horizon measured in `fit.time_unit`."""
    _require_valid_fit(fit, horizon)
    return float(fit.theta + (current_value - fit.theta) * np.exp(-fit.kappa * horizon))  # type: ignore[operator]


def conditional_variance(fit: OUFitResult, horizon: float) -> float:
    """OU model-implied conditional variance, not a guaranteed outcome band."""
    _require_valid_fit(fit, horizon)
    return float(_transition_variance(fit.kappa, fit.sigma, horizon))  # type: ignore[arg-type]


def conditional_standard_deviation(fit: OUFitResult, horizon: float) -> float:
    return float(np.sqrt(conditional_variance(fit, horizon)))


def conditional_band(fit: OUFitResult, current_value: float, horizon: float, *, standard_deviations: float = 1.96) -> tuple[float, float, float]:
    """Return mean/lower/upper OU model-implied distribution band."""
    if standard_deviations < 0:
        raise ValueError("standard_deviations must be non-negative")
    mean = conditional_expectation(fit, current_value, horizon)
    deviation = standard_deviations * conditional_standard_deviation(fit, horizon)
    return mean, mean - deviation, mean + deviation


def rolling_ou_fits(values: pd.Series, *, method: OU_METHOD = "AR1", window: int = 252, dt: float = 1.0, time_unit: str = "trading days") -> pd.DataFrame:
    """Fit each date using only values available through that date, never future values."""
    if not isinstance(values, pd.Series) or not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("rolling OU input must be a Series with a DatetimeIndex")
    if not values.index.is_unique or not values.index.is_monotonic_increasing or window < 3:
        raise ValueError("input dates must be unique/ascending and window must be at least 3")
    fitter = fit_ou_ar1 if method == "AR1" else fit_ou_mle if method == "MLE" else None
    if fitter is None:
        raise ValueError("method must be 'AR1' or 'MLE'")
    rows: list[dict[str, Any]] = []
    for position in range(window - 1, len(values)):
        fit = fitter(values.iloc[position - window + 1:position + 1], dt=dt, time_unit=time_unit)
        rows.append({"date": values.index[position], "theta": fit.theta, "kappa": fit.kappa, "sigma": fit.sigma, "half_life": fit.half_life, "valid": fit.valid, "invalid_reason": fit.invalid_reason})
    return pd.DataFrame(rows).set_index("date")


_EPSILON = 1e-12


def _validate_values(values: pd.Series | Sequence[float] | np.ndarray, dt: float) -> tuple[np.ndarray, str | None]:
    data = np.asarray(values, dtype=float)
    if dt <= 0 or not np.isfinite(dt):
        return data, "dt must be positive and finite."
    if data.ndim != 1 or len(data) < 20:
        return data, "At least 20 finite observations are required."
    if not np.all(np.isfinite(data)):
        return data, "Input contains NaN or infinite observations."
    if float(np.var(data)) <= _EPSILON:
        return data, "Input variance is effectively zero."
    return data, None


def _negative_log_likelihood_transformed(parameters: np.ndarray, data: np.ndarray, dt: float) -> float:
    theta, log_kappa, log_sigma = parameters
    kappa, sigma = np.exp(log_kappa), np.exp(log_sigma)
    if not np.isfinite(theta) or not np.isfinite(kappa) or not np.isfinite(sigma):
        return np.inf
    log_likelihood = _conditional_log_likelihood(data, theta, kappa, sigma, dt)
    return -log_likelihood if np.isfinite(log_likelihood) else np.inf


def _conditional_log_likelihood(data: np.ndarray, theta: float, kappa: float, sigma: float, dt: float) -> float:
    variance = _transition_variance(kappa, sigma, dt)
    if variance <= 0 or not np.isfinite(variance):
        return -np.inf
    mean = theta + (data[:-1] - theta) * np.exp(-kappa * dt)
    errors = data[1:] - mean
    return float(-0.5 * (len(errors) * np.log(2.0 * np.pi * variance) + np.sum(errors**2) / variance))


def _transition_variance(kappa: float, sigma: float, horizon: float) -> float:
    return float(sigma**2 * (-np.expm1(-2.0 * kappa * horizon)) / (2.0 * kappa))


def _valid_result(method: OU_METHOD, theta: float, kappa: float, sigma: float, log_likelihood: float, n_observations: int, dt: float, time_unit: str, metadata: dict[str, Any]) -> OUFitResult:
    n_transitions = n_observations - 1
    return OUFitResult(
        method=method, theta=theta, kappa=kappa, sigma=sigma,
        half_life=float(np.log(2.0) / kappa), reversion_75_time=float(-np.log(0.25) / kappa), reversion_90_time=float(-np.log(0.10) / kappa),
        log_likelihood=log_likelihood, aic=float(2 * 3 - 2 * log_likelihood), bic=float(np.log(n_transitions) * 3 - 2 * log_likelihood),
        n_observations=n_observations, dt=dt, time_unit=time_unit, valid=True, invalid_reason=None, metadata=metadata,
    )


def _invalid(method: OU_METHOD, n_observations: int, dt: float, time_unit: str, reason: str, metadata: dict[str, Any] | None = None) -> OUFitResult:
    return OUFitResult(method, None, None, None, None, None, None, None, None, None, n_observations, dt, time_unit, False, reason, metadata or {})


def _require_valid_fit(fit: OUFitResult, horizon: float) -> None:
    if not fit.valid or fit.theta is None or fit.kappa is None or fit.sigma is None:
        raise ValueError(f"A valid OU fit is required: {fit.invalid_reason or 'invalid fit'}")
    if horizon < 0 or not np.isfinite(horizon):
        raise ValueError("horizon must be finite and non-negative")
