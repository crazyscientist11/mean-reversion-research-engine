"""Strict chronological backtest research utilities."""
from .baseline import BenchmarkConfig, benchmark_signals
from .engine import BacktestConfig, run_backtest
from .events import EventStudyRecord, build_event_study
from .results import BacktestResults, summarize_results
from .walk_forward import ChronologicalSplit, chronological_split
