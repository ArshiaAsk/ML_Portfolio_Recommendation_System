"""Run all baseline strategies and compare their performance.

This module orchestrates walk-forward backtests for the baseline strategies
(Equal Weight, Minimum Variance, Risk Parity) and produces a summary DataFrame
for easy comparison.

Design
------
- Each strategy is rebalanced at the specified frequency using only data
  available up to the rebalance date (no lookahead).
- Covariance matrices are computed over a rolling lookback window from the
  training split determined by the walk-forward splitter.
- Transaction costs are applied uniformly across all strategies.

Example
-------
>>> import pandas as pd
>>> import numpy as np
>>> from portfolio_ml.backtesting.benchmark_runner import BenchmarkRunner
>>> from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
>>> dates = pd.date_range("2018-01-01", periods=1000, freq="B")
>>> prices = pd.DataFrame(
...     np.random.randn(1000, 5).cumsum(axis=0) + 100,
...     index=dates,
...     columns=[f"ETF{i}" for i in range(5)],
... )
>>> splitter = WalkForwardSplitter(lookback_window=252, rebalance_freq=21)
>>> runner = BenchmarkRunner(
...     price_data=prices,
...     splitter=splitter,
...     transaction_cost=0.0005,
... )
>>> summary = runner.run_all()
>>> print(summary)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from portfolio_ml.backtesting.engine import BacktestEngine
from portfolio_ml.modeling.baselines import (
    EqualWeightStrategy,
    MinVarianceStrategy,
    RiskParityStrategy,
)
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """Run and compare baseline portfolio strategies.

    Args:
        price_data: DataFrame with datetime index and one column per asset.
        splitter: Walk-forward splitter defining train/test periods.
        transaction_cost: Proportional transaction cost (e.g. 0.0005).
        lookback_days: Number of days to use for covariance estimation at
            each rebalance. Defaults to 252 (1 year).
        initial_capital: Starting portfolio value.

    Attributes:
        results_: Dict mapping strategy name to backtest result dict (set
            after :meth:`run_all`).
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        splitter: WalkForwardSplitter,
        transaction_cost: float = 0.0005,
        lookback_days: int = 252,
        initial_capital: float = 1_000_000.0,
    ):
        self.price_data = price_data
        self.splitter = splitter
        self.transaction_cost = transaction_cost
        self.lookback_days = lookback_days
        self.initial_capital = initial_capital
        self.results_: Dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> pd.DataFrame:
        """Execute backtests for all baseline strategies.

        Returns:
            Summary DataFrame with one row per strategy and columns for each
            metric (Return, Volatility, Sharpe, MaxDrawdown, Turnover).
        """
        strategies = {
            "EqualWeight": EqualWeightStrategy(),
            "MinVariance": MinVarianceStrategy(),
            "RiskParity": RiskParityStrategy(),
        }

        results = {}
        for name, strategy in strategies.items():
            logger.info("BenchmarkRunner: running strategy '%s'...", name)
            weights_schedule = self._compute_weights_schedule(strategy)
            engine = BacktestEngine(
                price_data=self.price_data,
                weights_schedule=weights_schedule,
                transaction_cost=self.transaction_cost,
                initial_capital=self.initial_capital,
            )
            result = engine.run()
            results[name] = result
            logger.info(
                "  %s: Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
                name,
                result["metrics"]["sharpe_ratio"],
                result["metrics"]["annualised_return"] * 100,
                result["metrics"]["max_drawdown"] * 100,
            )

        self.results_ = results
        return self._build_summary_df()

    def get_strategy_results(self, strategy_name: str) -> dict | None:
        """Retrieve detailed results for a specific strategy.

        Args:
            strategy_name: One of ``"EqualWeight"``, ``"MinVariance"``,
                ``"RiskParity"``.

        Returns:
            Dict with keys ``"metrics"``, ``"daily_returns"``,
            ``"portfolio_value"``, ``"turnover"``, or None if not run yet.
        """
        if self.results_ is None:
            logger.warning("get_strategy_results: run_all() has not been called yet.")
            return None
        return self.results_.get(strategy_name)

    # ------------------------------------------------------------------
    # Internal: weights schedule computation
    # ------------------------------------------------------------------

    def _compute_weights_schedule(self, strategy) -> Dict[pd.Timestamp, np.ndarray]:
        """Compute portfolio weights at each rebalance date using only past data.

        Args:
            strategy: An instance of PortfolioStrategy (EqualWeight, MinVariance, etc.).

        Returns:
            Dict mapping rebalance dates to weight arrays.
        """
        dates = self.price_data.index
        returns = self.price_data.pct_change().fillna(0.0)
        n_assets = self.price_data.shape[1]

        splits = self.splitter.split(dates)
        weights_schedule = {}

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # Rebalance happens at the first day of the test window
            rebal_date = dates[test_idx[0]]

            # Use only training data up to (but not including) the test window
            train_dates = dates[train_idx]
            train_returns = returns.iloc[train_idx]

            # Restrict to the most recent `lookback_days` within the training window
            if len(train_returns) > self.lookback_days:
                train_returns = train_returns.iloc[-self.lookback_days :]

            # Compute covariance from training data
            cov_matrix = train_returns.cov().values
            # Handle edge case: not enough data or all-zero returns
            if cov_matrix.shape != (n_assets, n_assets) or np.all(cov_matrix == 0):
                logger.warning(
                    "Fold %d: insufficient or zero-variance data. Using equal weights.",
                    fold_idx,
                )
                cov_matrix = np.eye(n_assets) * 1e-4

            # Allocate weights
            if isinstance(strategy, EqualWeightStrategy):
                weights = strategy.allocate(n_assets=n_assets)
            else:
                # MinVariance and RiskParity need covariance
                weights = strategy.allocate(cov_matrix=cov_matrix)

            weights_schedule[rebal_date] = weights

        logger.debug(
            "_compute_weights_schedule: generated %d rebalance dates.",
            len(weights_schedule),
        )
        return weights_schedule

    # ------------------------------------------------------------------
    # Internal: summary table
    # ------------------------------------------------------------------

    def _build_summary_df(self) -> pd.DataFrame:
        """Build a comparison DataFrame from stored results."""
        if self.results_ is None:
            return pd.DataFrame()

        rows = []
        for strategy_name, result in self.results_.items():
            metrics = result["metrics"]
            rows.append(
                {
                    "Strategy": strategy_name,
                    "Annualised Return": metrics["annualised_return"],
                    "Annualised Volatility": metrics["annualised_volatility"],
                    "Sharpe Ratio": metrics["sharpe_ratio"],
                    "Max Drawdown": metrics["max_drawdown"],
                    "Avg Turnover": metrics["avg_turnover"],
                    "Total Return": metrics["total_return"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
        return df
