import numpy as np
import pytest

from src.data.transforms import common_complete_rows, log_prices, log_returns, simple_returns


def test_return_and_log_transforms(prices) -> None:
    assert simple_returns(prices).iloc[0, 0] == pytest.approx(0.1)
    assert log_returns(prices).iloc[0, 0] == pytest.approx(np.log(1.1))
    assert log_prices(prices).iloc[0, 0] == pytest.approx(np.log(100))


def test_common_complete_rows_returns_copy(prices) -> None:
    prices.iloc[1, 1] = np.nan
    complete = common_complete_rows(prices)
    assert len(complete) == 2
    complete.iloc[0, 0] = 999
    assert prices.iloc[0, 0] == 100
