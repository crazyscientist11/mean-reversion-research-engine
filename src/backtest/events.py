"""Forward event outcomes kept separate from historical signal construction."""
from dataclasses import asdict, dataclass
from typing import Iterable
import pandas as pd

@dataclass(frozen=True)
class EventStudyRecord:
    signal_date: pd.Timestamp; execution_date: pd.Timestamp | None; horizon_days: int
    raw_target_return: float | None; direction_adjusted_return: float | None
    residual_change: float | None; distance_toward_equilibrium: float | None
    fraction_reverted: float | None; exit_reached: bool | None; stop_reached: bool | None
    mfe: float | None; mae: float | None

def build_event_study(prices: pd.Series, residuals: pd.Series, candidates: pd.Series, *, exit_boundary: float = 0.0, horizons: Iterable[int] = (1,3,5,10,20), stop_boundary: float | None = None) -> pd.DataFrame:
    """Measure outcomes after t+1 execution; this function never creates signals."""
    rows=[]; prices=prices.astype(float)
    for i, date in enumerate(prices.index):
        if not bool(candidates.loc[date]) or i+1 >= len(prices): continue
        entry_price=float(prices.iloc[i+1]); initial=float(residuals.loc[date]); direction=1 if initial < exit_boundary else -1
        for horizon in horizons:
            end=min(i+1+horizon, len(prices)-1); future=prices.iloc[i+1:end+1]; end_residual=float(residuals.iloc[end])
            raw=float(prices.iloc[end]/entry_price-1); fraction=None if abs(initial-exit_boundary)<1e-12 else (initial-end_residual)/(initial-exit_boundary)
            path=residuals.iloc[i+1:end+1]; exit_hit=bool((path>=exit_boundary).any()) if direction==1 else bool((path<=exit_boundary).any())
            stop_hit=None if stop_boundary is None else (bool((path<=stop_boundary).any()) if direction==1 else bool((path>=stop_boundary).any()))
            returns=future/entry_price-1
            rows.append(asdict(EventStudyRecord(date, prices.index[i+1], horizon, raw, direction*raw, end_residual-initial, abs(initial-exit_boundary)-abs(end_residual-exit_boundary), fraction, exit_hit, stop_hit, float(direction*returns.max() if direction==1 else -returns.min()), float(direction*returns.min() if direction==1 else -returns.max()))))
    return pd.DataFrame(rows)
