"""Backtesting and performance evaluation modules."""

from portfolio_ml.backtesting.engine import BacktestEngine
from portfolio_ml.backtesting.benchmark_runner import BenchmarkRunner
from portfolio_ml.backtesting.slice_eval import SliceEvaluator, compare_strategies_by_regime

__all__ = [
    "BacktestEngine",
    "BenchmarkRunner",
    "SliceEvaluator",
    "compare_strategies_by_regime",
]
