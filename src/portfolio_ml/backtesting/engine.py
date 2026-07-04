"""Backtesting engine for portfolio strategies.

Given a sequence of portfolio weights over time and corresponding asset prices,
this module computes realistic portfolio returns accounting for transaction
costs, rebalancing, and standard performance metrics.

Key metrics
-----------
- **Cumulative return**: Total portfolio return over the backtest period.
- **Annualised return**: Geometric mean return scaled to 252 trading days.
- **Annualised volatility**: Standard deviation of daily returns × √252.
- **Sharpe ratio**: Annualised return / annualised volatility (no risk-free rate).
- **Maximum drawdown**: Largest peak-to-trough decline in portfolio value.
- **Turnover**: Average absolute change in portfolio weights at rebalancing.

Example
-------
>>> import pandas as pd
>>> import numpy as np
>>> from portfolio_ml.backtesting.engine import BacktestEngine
>>> dates = pd.date_range("2020-01-01", periods=252, freq="B")
>>> prices = pd.DataFrame(
...     np.random.randn(252, 3).cumsum(axis=0) + 100,
...     index=dates,
...     columns=["SPY", "TLT", "GLD"],
... )
>>> # Assume equal-weight rebalanced monthly (21 days)
>>> weights_schedule = {dates[i]: np.array([1/3, 1/3, 1/3]) for i in range(0, 252, 21)}
>>> engine = BacktestEngine(
...     price_data=prices,
...     weights_schedule=weights_schedule,
...     transaction_cost=0.0005,
... )
>>> results = engine.run()
>>> results["metrics"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestEngine:
    """Execute a backtest given price data and a rebalancing schedule.

    Args:
        price_data: DataFrame with datetime index and one column per asset.
            Prices must be adjusted for splits/dividends.
        weights_schedule: Dict mapping rebalancing dates (pd.Timestamp) to
            1-D numpy arrays of portfolio weights.  Weights must be
            non-negative and sum to ~1.0 for each date.
        transaction_cost: Proportional cost applied to turnover at each
            rebalance (e.g. 0.0005 = 5 basis points per dollar traded).
        initial_capital: Starting portfolio value in dollars.

    Attributes:
        daily_returns_: Series of portfolio daily returns (set after :meth:`run`).
        portfolio_value_: Series of portfolio value over time (set after :meth:`run`).
        turnover_: Series of turnover at each rebalance date (set after :meth:`run`).
    """

    price_data: pd.DataFrame
    weights_schedule: Dict[pd.Timestamp, np.ndarray]
    transaction_cost: float = 0.0005
    initial_capital: float = 1_000_000.0

    # Computed after run()
    daily_returns_: pd.Series | None = None
    portfolio_value_: pd.Series | None = None
    turnover_: pd.Series | None = None

    def __post_init__(self) -> None:
        if self.price_data.empty:
            raise ValueError("price_data cannot be empty.")
        if not self.weights_schedule:
            raise ValueError("weights_schedule cannot be empty.")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be >= 0.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the backtest and return performance metrics.

        Returns:
            Dictionary with keys:
                - ``"metrics"``: dict of scalar performance metrics
                - ``"daily_returns"``: pd.Series of daily portfolio returns
                - ``"portfolio_value"``: pd.Series of portfolio value over time
                - ``"turnover"``: pd.Series of turnover at rebalance dates
        """
        dates = self.price_data.index
        n_assets = self.price_data.shape[1]
        n_days = len(dates)

        # Convert weights_schedule to sorted list of (date, weights)
        rebalance_dates = sorted(self.weights_schedule.keys())
        if rebalance_dates[0] > dates[0]:
            logger.warning(
                "First rebalance date %s is after first price date %s. "
                "Portfolio will have zero allocation until first rebalance.",
                rebalance_dates[0],
                dates[0],
            )

        # Initialise portfolio state
        portfolio_value = np.zeros(n_days)
        portfolio_value[0] = self.initial_capital
        current_weights = np.zeros(n_assets)
        turnover_list = []

        # Build a mapping: date -> weights for fast lookup
        weights_map = {d: self.weights_schedule[d] for d in rebalance_dates}

        for i in range(n_days):
            date = dates[i]
            prices_today = self.price_data.iloc[i].values

            # Check if today is a rebalancing date
            if date in weights_map:
                target_weights = weights_map[date]
                target_weights = self._normalise_weights(target_weights)

                # Compute turnover (sum of absolute weight changes)
                turnover = np.sum(np.abs(target_weights - current_weights))
                turnover_list.append((date, turnover))

                # Apply transaction cost: reduce portfolio value by (turnover * cost)
                cost = turnover * self.transaction_cost * portfolio_value[i]
                portfolio_value[i] -= cost

                # Update current weights
                current_weights = target_weights.copy()

            # Compute today's return if not the first day
            if i > 0:
                prices_yesterday = self.price_data.iloc[i - 1].values
                # Avoid division by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    asset_returns = prices_today / prices_yesterday - 1
                    asset_returns = np.nan_to_num(asset_returns, nan=0.0, posinf=0.0, neginf=0.0)

                # Portfolio return is weighted sum of asset returns
                portfolio_return = np.dot(current_weights, asset_returns)
                portfolio_value[i] = portfolio_value[i - 1] * (1 + portfolio_return)

        # Store results
        self.portfolio_value_ = pd.Series(portfolio_value, index=dates, name="portfolio_value")
        self.daily_returns_ = self.portfolio_value_.pct_change().fillna(0.0)
        self.turnover_ = pd.Series(dict(turnover_list), name="turnover")

        # Compute summary metrics
        metrics = self._compute_metrics()

        logger.info(
            "BacktestEngine: completed backtest over %d days with %d rebalances.",
            n_days,
            len(rebalance_dates),
        )

        return {
            "metrics": metrics,
            "daily_returns": self.daily_returns_,
            "portfolio_value": self.portfolio_value_,
            "turnover": self.turnover_,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_weights(self, w: np.ndarray) -> np.ndarray:
        """Ensure weights are non-negative and sum to 1."""
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total < 1e-12:
            # Fallback to equal weight if all zeros
            return np.ones_like(w) / len(w)
        return w / total

    def _compute_metrics(self) -> dict:
        """Calculate standard performance metrics."""
        if self.daily_returns_ is None or self.portfolio_value_ is None:
            raise RuntimeError("run() must be called before computing metrics.")

        returns = self.daily_returns_.values
        pv = self.portfolio_value_.values

        # Total return
        total_return = (pv[-1] / pv[0]) - 1

        # Annualised return (geometric)
        n_days = len(returns)
        n_years = n_days / 252.0
        annualised_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Annualised volatility
        vol_daily = returns.std(ddof=1)
        annualised_volatility = vol_daily * np.sqrt(252)

        # Sharpe ratio (no risk-free rate)
        sharpe = annualised_return / annualised_volatility if annualised_volatility > 1e-9 else 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()

        # Average turnover
        if self.turnover_ is not None and len(self.turnover_) > 0:
            avg_turnover = self.turnover_.mean()
        else:
            avg_turnover = 0.0

        return {
            "total_return": total_return,
            "annualised_return": annualised_return,
            "annualised_volatility": annualised_volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_turnover": avg_turnover,
            "n_days": n_days,
        }
