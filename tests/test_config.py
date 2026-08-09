import pytest

from src.config import DataParameters, Frequency, PredictionStorageParameters, ResearchWindowParameters


def test_valid_defaults() -> None:
    assert DataParameters().frequency is Frequency.DAILY
    assert PredictionStorageParameters().allow_overwrite is False


def test_invalid_frequency_rejected() -> None:
    with pytest.raises(ValueError):
        DataParameters(frequency="daily")  # type: ignore[arg-type]


def test_invalid_windows_rejected() -> None:
    with pytest.raises(ValueError):
        ResearchWindowParameters(short_window=126, medium_window=60)

