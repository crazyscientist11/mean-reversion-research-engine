"""Strict daily close walk-forward runner. Signals at t execute at close t+1."""
from dataclasses import dataclass
from typing import Callable
import pandas as pd
from src.costs import CostModel
from src.decision import PositionState, ResearchAction
from .accounting import TradeRecord
from .baseline import BenchmarkConfig, benchmark_signals
from .events import build_event_study
from .results import BacktestResults, summarize_results

SignalFunction = Callable[[pd.Series, PositionState], ResearchAction]

@dataclass(frozen=True)
class BacktestConfig:
    benchmark: BenchmarkConfig = BenchmarkConfig()
    costs: CostModel = CostModel()
    notional: float = 1.0
    execution_lag: int = 1

def _trade_records(prices: pd.Series, signals: pd.DataFrame, config: BacktestConfig) -> list[TradeRecord]:
    """Single-stock accounting. It never calls a residual strategy market neutral."""
    open_trade=None; records=[]
    for i, (date,row) in enumerate(signals.iterrows()):
        execution=i+config.execution_lag
        if execution>=len(prices): break
        action=row.action; price=float(prices.iloc[execution])
        if open_trade is None and action in {"ENTER_LONG","ENTER_SHORT"}:
            open_trade=(date,execution,price,action)
        elif open_trade is not None and action in {"EXIT_LONG","STOP_LONG","EXIT_SHORT","STOP_SHORT"}:
            entry_date,entry_index,entry_price,entry_action=open_trade; sign=1 if entry_action=="ENTER_LONG" else -1
            gross=config.notional*sign*(price/entry_price-1); costs=config.costs.trade_cost(config.notional,True)+config.costs.trade_cost(config.notional,False)
            records.append(TradeRecord(entry_date,date,entry_action,config.notional,gross,costs,gross-costs,execution-entry_index)); open_trade=None
    return records

def run_backtest(prices: pd.Series, *, config: BacktestConfig = BacktestConfig(), full_signal: SignalFunction | None = None) -> dict[str, BacktestResults | pd.DataFrame]:
    """Run benchmark and an optional full-model signal hook chronologically.

    The hook receives a copy of prices through t and the current position only.
    It must return a frozen-at-t action; no future data is supplied.
    """
    if not isinstance(prices.index,pd.DatetimeIndex) or not prices.index.is_monotonic_increasing or not prices.index.is_unique: raise ValueError("prices need a unique ascending DatetimeIndex")
    benchmark=benchmark_signals(prices,config.benchmark)
    residual=benchmark.zscore.astype(float)
    candidates=benchmark.action.isin(["ENTER_LONG","ENTER_SHORT"])
    events=build_event_study(prices,residual,candidates,exit_boundary=0.,stop_boundary=config.benchmark.stop_z)
    baseline_results=summarize_results(_trade_records(prices,benchmark,config),events)
    if full_signal is None: return {"benchmark":baseline_results,"events":events,"benchmark_signals":benchmark}
    position=PositionState.FLAT; actions=[]
    for i,date in enumerate(prices.index):
        action=full_signal(prices.iloc[:i+1].copy(),position)
        if not isinstance(action,ResearchAction): raise TypeError("full_signal must return ResearchAction")
        actions.append(action.value)
        if action is ResearchAction.ENTER_LONG: position=PositionState.LONG_RESIDUAL
        elif action is ResearchAction.ENTER_SHORT: position=PositionState.SHORT_RESIDUAL
        elif action in {ResearchAction.EXIT_LONG,ResearchAction.STOP_LONG,ResearchAction.EXIT_SHORT,ResearchAction.STOP_SHORT}: position=PositionState.FLAT
    full=pd.DataFrame({"zscore":residual,"action":actions},index=prices.index)
    full_events=build_event_study(prices,residual,full.action.isin(["ENTER_LONG","ENTER_SHORT"]),exit_boundary=0.,stop_boundary=config.benchmark.stop_z)
    return {"benchmark":baseline_results,"full_model":summarize_results(_trade_records(prices,full,config),full_events),"events":events,"full_events":full_events,"benchmark_signals":benchmark,"full_signals":full}
