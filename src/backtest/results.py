"""Descriptive results, calibration, and bucket summaries; no performance selection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .accounting import TradeRecord

@dataclass(frozen=True)
class BacktestResults:
    metrics: dict[str,float]; equity: pd.Series; drawdown: pd.Series; calibration: pd.DataFrame; buckets: pd.DataFrame

def summarize_results(trades: list[TradeRecord], events: pd.DataFrame, predicted_probabilities: pd.Series | None=None) -> BacktestResults:
    net=pd.Series([trade.net_pnl for trade in trades], dtype=float); gross=pd.Series([trade.gross_pnl for trade in trades], dtype=float); costs=pd.Series([trade.costs for trade in trades], dtype=float)
    equity=net.cumsum(); peak=equity.cummax(); drawdown=equity-peak
    wins=net[net>0]; losses=net[net<0]
    metrics={"candidate_count":float(len(events)),"trade_count":float(len(trades)),"exit_count":float(events.get("exit_reached",pd.Series(dtype=bool)).sum()),"stop_count":float(events.get("stop_reached",pd.Series(dtype=bool)).sum()),"average_fraction_reverted":float(events["fraction_reverted"].mean()) if "fraction_reverted" in events else np.nan,"median_fraction_reverted":float(events["fraction_reverted"].median()) if "fraction_reverted" in events else np.nan,"gross_pnl":float(gross.sum()),"costs":float(costs.sum()),"net_pnl":float(net.sum()),"win_rate":float((net>0).mean()) if len(net) else np.nan,"average_win":float(wins.mean()) if len(wins) else np.nan,"average_loss":float(losses.mean()) if len(losses) else np.nan,"maximum_drawdown":float(drawdown.min()) if len(drawdown) else 0.,"turnover":float(sum(t.gross_notional for t in trades)),"average_holding_period":float(np.mean([t.holding_days for t in trades])) if trades else np.nan,"median_holding_period":float(np.median([t.holding_days for t in trades])) if trades else np.nan}
    calibration=pd.DataFrame()
    if predicted_probabilities is not None and len(events):
        table=pd.DataFrame({"predicted":predicted_probabilities.reindex(events.index),"realized":events.get("exit_reached")}).dropna()
        if not table.empty:
            table["bucket"]=pd.cut(table.predicted,[.5,.6,.7,.8,.9,1.000001],right=False)
            calibration=table.groupby("bucket",observed=False).agg(observations=("realized","size"),average_predicted_probability=("predicted","mean"),actual_exit_frequency=("realized","mean")).reset_index()
    buckets=pd.DataFrame() if events.empty else events.groupby("horizon_days").agg(observations=("horizon_days","size"),average_fraction_reverted=("fraction_reverted","mean"),realized_exit_frequency=("exit_reached","mean")).reset_index()
    return BacktestResults(metrics,equity,drawdown,calibration,buckets)
