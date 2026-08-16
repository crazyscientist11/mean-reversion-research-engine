import numpy as np
import pandas as pd
from src.backtest import BenchmarkConfig, chronological_split, run_backtest
from src.decision import ResearchAction

def prices(n=100): return pd.Series(np.exp(np.linspace(4,4.3,n)+.04*np.sin(np.arange(n))), index=pd.date_range("2020-01-01",periods=n))

def test_chronological_split_has_no_shuffle():
    split=chronological_split(prices().index)
    assert split.train.max() < split.validation.min() < split.test.min()

def test_signals_through_cutoff_are_unchanged_by_radical_future_data():
    original=prices(120); altered=original.copy(); cutoff=original.index[80]; altered.iloc[81:]=altered.iloc[81:]*1000
    one=run_backtest(original,config=__import__('src.backtest.engine',fromlist=['BacktestConfig']).BacktestConfig(BenchmarkConfig(window=20)))
    two=run_backtest(altered,config=__import__('src.backtest.engine',fromlist=['BacktestConfig']).BacktestConfig(BenchmarkConfig(window=20)))
    pd.testing.assert_frame_equal(one["benchmark_signals"].loc[:cutoff],two["benchmark_signals"].loc[:cutoff])

def test_full_hook_receives_only_history_through_current_date():
    seen=[]
    def signal(history, position): seen.append(history.index.max()); return ResearchAction.WAIT
    data=prices(80); run_backtest(data,config=__import__('src.backtest.engine',fromlist=['BacktestConfig']).BacktestConfig(BenchmarkConfig(window=20)),full_signal=signal)
    assert seen == list(data.index)

def test_event_study_and_result_metrics_are_reported():
    output=run_backtest(prices(),config=__import__('src.backtest.engine',fromlist=['BacktestConfig']).BacktestConfig(BenchmarkConfig(window=20,entry_z=.1)))
    assert {"candidate_count","net_pnl","maximum_drawdown"} <= output["benchmark"].metrics.keys()
    assert set(output["events"].horizon_days.unique()) <= {1,3,5,10,20}
