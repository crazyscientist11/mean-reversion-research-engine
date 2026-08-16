"""Transparent fixed-z benchmark settings; these are not optimal thresholds."""
from dataclasses import dataclass
import pandas as pd
from src.decision import PositionState, benchmark_action
from src.models.common import prior_sample_zscore

@dataclass(frozen=True)
class BenchmarkConfig:
    window: int = 60
    entry_z: float = 2.0
    exit_z: float = .5
    stop_z: float = 4.0

def benchmark_signals(prices: pd.Series, config: BenchmarkConfig = BenchmarkConfig()) -> pd.DataFrame:
    """Signals at t use only prior log prices; execution belongs to t+1."""
    logs = prices.astype(float).map(__import__('math').log)
    zscores = prior_sample_zscore(logs, config.window)
    position = PositionState.FLAT
    rows=[]
    for date, zscore in zscores.items():
        if pd.isna(zscore): action="INSUFFICIENT_DATA"
        else:
            action = benchmark_action(float(zscore), position, config.entry_z, config.exit_z, config.stop_z)
            if action.value == "ENTER_LONG": position=PositionState.LONG_RESIDUAL
            elif action.value == "ENTER_SHORT": position=PositionState.SHORT_RESIDUAL
            elif action.value in {"EXIT_LONG","STOP_LONG","EXIT_SHORT","STOP_SHORT"}: position=PositionState.FLAT
            action=action.value
        rows.append({"date":date,"zscore":zscore,"action":action})
    return pd.DataFrame(rows).set_index("date")
