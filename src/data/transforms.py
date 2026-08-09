"""Basic non-mutating price transforms. No model fitting occurs here."""
import numpy as np
import pandas as pd

from .validation import validate_price_frame


def _positive(frame: pd.DataFrame) -> pd.DataFrame:
    return validate_price_frame(frame, require_positive_prices=True)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return _positive(prices).pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    checked = _positive(prices)
    return np.log(checked / checked.shift(1)).iloc[1:]


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(_positive(prices))


def common_complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing only rows complete across all columns."""
    return frame.dropna(how="any").copy(deep=True)

