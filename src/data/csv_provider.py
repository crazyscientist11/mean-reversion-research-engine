"""CSV-only Step 1 market-data provider."""
from pathlib import Path

import pandas as pd

from src.config import DataParameters
from src.schemas import MarketDataBundle
from .validation import summarize_prices, validate_price_frame


class CSVDataProvider:
    source_name = "CSV"

    def __init__(self, parameters: DataParameters | None = None) -> None:
        self.parameters = parameters or DataParameters()

    def load(self, path: str | Path) -> MarketDataBundle:
        raw = pd.read_csv(path)
        date_column = self.parameters.date_column
        if date_column not in raw.columns:
            raise ValueError(f"CSV must include date column {date_column!r}")
        if len(raw.columns) < 2:
            raise ValueError("CSV must include at least one ticker column")
        dates = pd.to_datetime(raw[date_column], errors="raise")
        if dates.isna().any():
            raise ValueError("dates cannot be missing")
        if dates.duplicated().any():
            raise ValueError("dates must be unique")
        prices = raw.drop(columns=[date_column]).copy(deep=True)
        prices.index = pd.DatetimeIndex(dates, name=date_column)
        prices = prices.sort_index()
        prices = validate_price_frame(prices, require_positive_prices=self.parameters.require_positive_prices)
        if self.parameters.drop_incomplete_rows:
            prices = prices.dropna(how="any")
        if len(prices) < self.parameters.minimum_observations:
            raise ValueError("insufficient observations after validation")
        summary = summarize_prices(prices, frequency=self.parameters.frequency.value, source=self.source_name)
        return MarketDataBundle(
            prices=prices, frequency=self.parameters.frequency,
            start_date=summary["start_date"], end_date=summary["end_date"],
            tickers=summary["tickers"], missing_values_by_ticker=summary["missing_values_by_ticker"],
            number_of_observations=summary["observation_count"],
            number_of_complete_observations=summary["complete_row_count"],
            source_name=self.source_name, metadata={"input_path": str(path), "raw_row_count": len(raw)},
        )

