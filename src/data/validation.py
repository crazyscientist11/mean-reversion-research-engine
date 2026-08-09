"""Validation and summary helpers for aligned price frames."""
from typing import Any

import pandas as pd


def validate_price_frame(frame: pd.DataFrame, *, require_positive_prices: bool = True) -> pd.DataFrame:
    """Return a validated copy; never mutate the caller's frame."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("price data must use a DatetimeIndex")
    if frame.empty or not len(frame.columns):
        raise ValueError("price data must contain dates and at least one ticker")
    if not frame.index.is_unique:
        raise ValueError("dates must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending")
    if any(not isinstance(name, str) or not name.strip() for name in frame.columns):
        raise ValueError("ticker column names must be non-empty strings")
    checked = frame.copy(deep=True)
    for ticker in checked.columns:
        try:
            checked[ticker] = pd.to_numeric(checked[ticker], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ticker {ticker!r} contains nonnumeric prices") from exc
    if require_positive_prices and (checked.dropna() <= 0).any().any():
        raise ValueError("ordinary equity prices must be strictly positive")
    return checked


def summarize_prices(frame: pd.DataFrame, *, frequency: str, source: str) -> dict[str, Any]:
    """Create the dashboard-ready data quality summary."""
    return {
        "start_date": frame.index.min(), "end_date": frame.index.max(),
        "observation_count": len(frame), "ticker_count": len(frame.columns),
        "tickers": tuple(str(column) for column in frame.columns),
        "missing_values_by_ticker": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "complete_row_count": int(frame.notna().all(axis=1).sum()),
        "frequency": frequency, "source": source,
    }

