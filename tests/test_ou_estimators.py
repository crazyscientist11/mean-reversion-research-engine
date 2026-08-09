import numpy as np
import pytest

from src.models.ou import fit_ou_ar1, fit_ou_mle, simulate_ou


@pytest.fixture
def synthetic_ou() -> np.ndarray:
    return simulate_ou(theta=0.5, kappa=0.08, sigma=0.12, n_observations=3000, dt=1.0, x0=-0.2, seed=42)


def test_ar1_recovers_seeded_ou_parameters(synthetic_ou) -> None:
    result = fit_ou_ar1(synthetic_ou)
    assert result.valid
    assert result.theta == pytest.approx(0.5, abs=0.08)
    assert result.kappa == pytest.approx(0.08, rel=0.18)
    assert result.sigma == pytest.approx(0.12, rel=0.10)


def test_exact_mle_recovers_seeded_ou_parameters(synthetic_ou) -> None:
    result = fit_ou_mle(synthetic_ou)
    assert result.valid
    assert result.theta == pytest.approx(0.5, abs=0.08)
    assert result.kappa == pytest.approx(0.08, rel=0.18)
    assert result.sigma == pytest.approx(0.12, rel=0.10)
    assert result.log_likelihood is not None
    assert result.aic is not None and result.bic is not None
