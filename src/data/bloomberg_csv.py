"""CSV detection and loading for common Bloomberg Excel exports, without Bloomberg APIs."""
from pathlib import Path
import pandas as pd
from .normalization import SeriesMapping, normalize_repeated_pairs

def detect_csv_format(raw: pd.DataFrame) -> str:
    names=[str(column).strip().lower() for column in raw.columns]
    if "date" in names and len(raw.columns)>=2: return "wide"
    return "repeated_pairs_requires_mapping"

def load_bloomberg_csv(path: str|Path, *, mappings: list[SeriesMapping]|None=None) -> pd.DataFrame:
    raw=pd.read_csv(path)
    return normalize_bloomberg_frame(raw,mappings=mappings)

def normalize_bloomberg_frame(raw: pd.DataFrame, *, mappings: list[SeriesMapping]|None=None) -> pd.DataFrame:
    detected=detect_csv_format(raw)
    if detected=="wide":
        date_column=next(column for column in raw.columns if str(column).strip().lower()=="date")
        result=raw.drop(columns=[date_column]).apply(pd.to_numeric,errors="raise")
        result.index=pd.DatetimeIndex(pd.to_datetime(raw[date_column],errors="raise"),name="Date")
        return result.sort_index()
    if not mappings: raise ValueError("Repeated date/price exports require explicit ticker, date-column, and price-column mappings.")
    return normalize_repeated_pairs(raw,mappings)
