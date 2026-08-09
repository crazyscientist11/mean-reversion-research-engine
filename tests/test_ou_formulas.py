import numpy as np
import pytest

from src.models.ou import OUFitResult, conditional_band, conditional_expectation, conditional_standard_deviation, conditional_variance, fit_ou_both


def known_fit() -> OUFitResult:
    kappa = np.log(2.0) / 5.0
    return OUFitResult("AR1", 1.0, kappa, 0.2, 5.0, -np.log(0.25) / kappa, -np.log(0.10) / kappa, None, None, None, 100, 1.0, "trading days", True, None)


def test_reversion_times_and_conditional_moments() -> None:
    fit = known_fit()
    assert fit.half_life == pytest.approx(5.0)
    assert fit.reversion_75_time == pytest.approx(10.0)
    assert fit.reversion_90_time == pytest.approx(-np.log(0.10) / fit.kappa)
    assert conditional_expectation(fit, 3.0, 5.0) == pytest.approx(2.0)
    expected_variance = fit.sigma**2 / (2 * fit.kappa) * (1 - np.exp(-2 * fit.kappa * 5.0))
    assert conditional_variance(fit, 5.0) == pytest.approx(expected_variance)
    assert conditional_standard_deviation(fit, 5.0) == pytest.approx(np.sqrt(expected_variance))
    mean, lower, upper = conditional_band(fit, 3.0, 5.0)
    assert mean == pytest.approx(2.0)
    assert lower < mean < upper


def test_comparison_exposes_both_methods(synthetic_ou) -> None:
    comparison = fit_ou_both(synthetic_ou)
    assert set(comparison.as_table()["method"]) == {"AR1", "MLE"}


@pytest.fixture
def synthetic_ou() -> np.ndarray:
    from src.models.ou import simulate_ou
    return simulate_ou(theta=0.5, kappa=0.08, sigma=0.12, n_observations=500, seed=9)
