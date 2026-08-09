"""Internal data schemas shared by providers and future research modules."""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import Frequency


@dataclass(frozen=True)
class MarketDataBundle:
    """Aligned adjusted-price data, with explicit provenance and data quality metadata."""
    prices: pd.DataFrame
    frequency: Frequency
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    tickers: tuple[str, ...]
    missing_values_by_ticker: dict[str, int]
    number_of_observations: int
    number_of_complete_observations: int
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "observation_count": self.number_of_observations,
            "ticker_count": len(self.tickers),
            "tickers": self.tickers,
            "missing_values_by_ticker": self.missing_values_by_ticker.copy(),
            "complete_row_count": self.number_of_complete_observations,
            "frequency": self.frequency.value,
            "source": self.source_name,
        }
