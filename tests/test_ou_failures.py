from types import SimpleNamespace

import numpy as np
import pytest

from src.models import ou


@pytest.mark.parametrize("values", [np.ones(30), np.arange(10, dtype=float), np.array([1.0] * 29 + [np.nan]), np.array([1.0] * 29 + [np.inf])])
def test_invalid_input_is_never_presented_as_valid(values) -> None:
    assert not ou.fit_ou_ar1(values).valid
    assert not ou.fit_ou_mle(values).valid


def test_random_walk_like_ar_coefficient_is_invalid() -> None:
    values = np.arange(100, dtype=float)
    result = ou.fit_ou_ar1(values)
    assert not result.valid
    assert "coefficient" in result.invalid_reason.lower()


def test_optimizer_failure_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(ou, "minimize", lambda *args, **kwargs: SimpleNamespace(success=False, fun=1.0, x=np.zeros(3), message="forced failure"))
    result = ou.fit_ou_mle(np.linspace(-1, 1, 50) + 0.01 * np.sin(np.arange(50)))
    assert not result.valid
    assert "optimizer failed" in result.invalid_reason.lower()
