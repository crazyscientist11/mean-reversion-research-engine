import pandas as pd
import pytest

from src.data.validation import validate_price_frame


def test_validation_does_not_mutate_input(prices) -> None:
    original = prices.copy(deep=True)
    result = validate_price_frame(prices)
    assert result.equals(original)
    assert result is not prices
    assert prices.equals(original)


def test_unsorted_dates_rejected(prices) -> None:
    with pytest.raises(ValueError, match="sorted"):
        validate_price_frame(prices.iloc[::-1])

