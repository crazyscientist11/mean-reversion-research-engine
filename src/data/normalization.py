"""Explicit CSV normalization; never fills or fabricates observations."""
from dataclasses import dataclass
from typing import Iterable
import pandas as pd

@dataclass(frozen=True)
class SeriesMapping:
    ticker: str
    date_column: str
    price_column: str

def normalize_repeated_pairs(raw: pd.DataFrame, mappings: Iterable[SeriesMapping]) -> pd.DataFrame:
    """Outer-merge mapped series on exact observed dates, with no forward/back fill."""
    pieces=[]
    seen=set()
    for mapping in mappings:
        if not mapping.ticker.strip() or mapping.ticker in seen: raise ValueError("ticker mappings must be non-empty and unique")
        if mapping.date_column not in raw or mapping.price_column not in raw: raise ValueError("mapped columns are missing")
        dates=pd.to_datetime(raw[mapping.date_column],errors="coerce")
        values=pd.to_numeric(raw[mapping.price_column],errors="coerce")
        piece=pd.DataFrame({"Date":dates,mapping.ticker:values}).dropna(subset=["Date"])
        if piece.Date.duplicated().any(): raise ValueError(f"duplicate observed dates for {mapping.ticker}")
        pieces.append(piece.set_index("Date")); seen.add(mapping.ticker)
    if not pieces: raise ValueError("at least one explicit series mapping is required")
    return pd.concat(pieces,axis=1,join="outer",sort=False).sort_index()
