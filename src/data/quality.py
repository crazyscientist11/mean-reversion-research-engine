"""Descriptive quality and model-readiness reports; flags do not silently alter data."""
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DataQualityReport:
    start_date: pd.Timestamp; end_date: pd.Timestamp; observation_count: int; ticker_count: int
    missing_values_by_ticker: dict[str,int]; duplicate_dates: int; nonpositive_prices: dict[str,int]
    unsorted_dates: bool; complete_row_count: int; approximate_trading_years: float
    coverage: dict[str,dict[str,str|None]]; large_one_day_jumps: dict[str,int]
    def to_dict(self)->dict[str,Any]: return asdict(self)

@dataclass(frozen=True)
class ModelReadinessReport:
    checks: dict[str,dict[str,str]]
    def to_frame(self)->pd.DataFrame: return pd.DataFrame([{"model":name,**value} for name,value in self.checks.items()])

def data_quality_report(frame: pd.DataFrame, *, jump_threshold: float=.25) -> DataQualityReport:
    if not isinstance(frame.index,pd.DatetimeIndex): raise ValueError("prices need a DatetimeIndex")
    numeric=frame.apply(pd.to_numeric,errors="coerce")
    coverage={}
    for ticker in numeric:
        values=numeric[ticker].dropna(); coverage[str(ticker)]={"start":None if values.empty else str(values.index.min().date()),"end":None if values.empty else str(values.index.max().date())}
    jumps=(numeric.pct_change().abs()>jump_threshold).sum()
    return DataQualityReport(frame.index.min(),frame.index.max(),len(frame),len(frame.columns),{str(k):int(v) for k,v in numeric.isna().sum().items()},int(frame.index.duplicated().sum()),{str(k):int(v) for k,v in (numeric<=0).sum().items()},not frame.index.is_monotonic_increasing,int(numeric.notna().all(axis=1).sum()),len(frame)/252,coverage,{str(k):int(v) for k,v in jumps.items()})

def model_readiness_report(frame: pd.DataFrame, *, benchmark_window:int=60, regression_window:int=126, pair_window:int=252, minimum_pca_tickers:int=3) -> ModelReadinessReport:
    n=len(frame); tickers=len(frame.columns); complete=int(frame.notna().all(axis=1).sum())
    def ready(condition:bool,reason:str)->dict[str,str]: return {"status":"sufficient" if condition else "insufficient","reason":reason}
    return ModelReadinessReport({"Basic benchmark":ready(n>=benchmark_window+1,f"requires {benchmark_window+1} observations"),"Trend model":ready(n>=regression_window+1,f"requires {regression_window+1} observations"),"Factor model":ready(tickers>=2 and complete>=regression_window+1,"requires target/factor columns and complete regression history"),"Pair 252-day model":ready(tickers>=2 and complete>=pair_window+1,"requires two series and 253 common complete rows"),"PCA":ready(tickers>=minimum_pca_tickers and complete>=regression_window+1,"requires selected universe and complete training history"),"OU":ready(n>=20,"requires at least 20 residual observations after residual construction"),"Walk-forward":ready(n>=max(pair_window+21,benchmark_window+21),"requires training history plus forward outcomes")})
