"""Slice-based performance evaluation for regime analysis.

Evaluating portfolio strategies over the entire backtest period can mask
important differences in performance across market regimes (bull/bear,
high/low volatility, etc.).  This module computes performance metrics on
user-defined slices of the backtest timeline.

Use cases
---------
- **Regime robustness**: Does the strategy perform well in both bull and bear markets?
- **Volatility sensitivity**: How does performance degrade during high-volatility periods?
- **Crisis testing**: What happens during known market crises (2008, 2020, etc.)?

Example
-------
>>> import pandas as pd
>>> from portfolio_ml.backtesting.slice_eval import SliceEvaluator
>>> from portfolio_ml.backtesting.engine import BacktestEngine
>>> # Assume engine has been run
>>> daily_returns = engine.daily_returns_
>>> slices = {
...     "bull_2019": ("2019-01-01", "2019-12-31"),
...     "covid_crash": ("2020-02-01", "2020-04-30"),
... }
>>> evaluator = SliceEvaluator(daily_returns)
>>> slice_metrics = evaluator.evaluate_slices(slices)
>>> print(slice_metrics)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


class SliceEvaluator:
    """Compute performance metrics over specified time slices.

    Args:
        daily_returns: Series of portfolio daily returns with a datetime index.

    Example:
        >>> evaluator = SliceEvaluator(daily_returns)
        >>> slices = {"2020_Q1": ("2020-01-01", "2020-03-31")}
        >>> metrics = evaluator.evaluate_slices(slices)
    """

    def __init__(self, daily_returns: pd.Series):
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            raise ValueError("daily_returns must have a DatetimeIndex.")
        self.daily_returns = daily_returns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_slices(
        self, slices: Dict[str, Tuple[str, str]]
    ) -> pd.DataFrame:
        """Compute metrics for each named time slice.

        Args:
            slices: Dict mapping slice name to (start_date, end_date) tuples.
                Dates can be strings (e.g. "2020-01-01") or pd.Timestamp.

        Returns:
            DataFrame with one row per slice and columns:
                - Slice
                - Start
                - End
                - N_Days
                - Total_Return
                - Annualised_Return
                - Annualised_Volatility
                - Sharpe_Ratio
                - Max_Drawdown
        """
        results = []
        for slice_name, (start, end) in slices.items():
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            slice_returns = self._extract_slice(start_ts, end_ts)

            if slice_returns.empty:
                logger.warning(
                    "Slice '%s' (%s to %s) has no data.",
                    slice_name,
                    start,
                    end,
                )
                continue

            metrics = self._compute_slice_metrics(slice_returns)
            metrics["Slice"] = slice_name
            metrics["Start"] = start_ts
            metrics["End"] = end_ts
            results.append(metrics)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df[
            [
                "Slice",
                "Start",
                "End",
                "N_Days",
                "Total_Return",
                "Annualised_Return",
                "Annualised_Volatility",
                "Sharpe_Ratio",
                "Max_Drawdown",
            ]
        ]
        logger.info("SliceEvaluator: evaluated %d slices.", len(df))
        return df

    def detect_regimes(
        self,
        volatility_threshold: float = 0.02,
        return_threshold: float = 0.0,
    ) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
        """Automatically detect bull/bear and high/low volatility regimes.

        Uses a simple rule-based approach:
        - **Bull**: periods where rolling 63-day return > return_threshold
        - **Bear**: periods where rolling 63-day return < -return_threshold
        - **High Vol**: periods where rolling 21-day volatility > volatility_threshold
        - **Low Vol**: periods where rolling 21-day volatility < volatility_threshold

        Args:
            volatility_threshold: Daily volatility threshold for high/low vol.
            return_threshold: Rolling 63-day return threshold for bull/bear.

        Returns:
            Dict mapping regime labels to (start, end) tuples.

        Notes:
            This is a simplified heuristic.  For production use consider more
            sophisticated regime detection (Hidden Markov Models, change-point
            detection, etc.).
        """
        returns = self.daily_returns
        rolling_63d = returns.rolling(window=63, min_periods=1).sum()
        rolling_21d_vol = returns.rolling(window=21, min_periods=1).std()

        regimes = {}

        # Bull markets
        bull_mask = rolling_63d > return_threshold
        if bull_mask.any():
            bull_start = returns.index[bull_mask][0]
            bull_end = returns.index[bull_mask][-1]
            regimes["bull"] = (bull_start, bull_end)

        # Bear markets
        bear_mask = rolling_63d < -return_threshold
        if bear_mask.any():
            bear_start = returns.index[bear_mask][0]
            bear_end = returns.index[bear_mask][-1]
            regimes["bear"] = (bear_start, bear_end)

        # High volatility
        high_vol_mask = rolling_21d_vol > volatility_threshold
        if high_vol_mask.any():
            hv_start = returns.index[high_vol_mask][0]
            hv_end = returns.index[high_vol_mask][-1]
            regimes["high_vol"] = (hv_start, hv_end)

        # Low volatility
        low_vol_mask = rolling_21d_vol < volatility_threshold
        if low_vol_mask.any():
            lv_start = returns.index[low_vol_mask][0]
            lv_end = returns.index[low_vol_mask][-1]
            regimes["low_vol"] = (lv_start, lv_end)

        logger.info("detect_regimes: identified %d regimes.", len(regimes))
        return regimes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_slice(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Return daily_returns filtered to [start, end]."""
        mask = (self.daily_returns.index >= start) & (self.daily_returns.index <= end)
        return self.daily_returns[mask]

    def _compute_slice_metrics(self, slice_returns: pd.Series) -> dict:
        """Compute standard metrics for a return slice."""
        returns = slice_returns.values
        n_days = len(returns)

        # Total return
        cumulative = (1 + returns).prod() - 1

        # Annualised return
        n_years = n_days / 252.0
        annualised_return = (
            (1 + cumulative) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        )

        # Annualised volatility
        vol_daily = returns.std(ddof=1) if n_days > 1 else 0.0
        annualised_volatility = vol_daily * np.sqrt(252)

        # Sharpe ratio
        sharpe = (
            annualised_return / annualised_volatility
            if annualised_volatility > 1e-9
            else 0.0
        )

        # Maximum drawdown
        cumulative_arr = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_arr)
        drawdown = (cumulative_arr / running_max) - 1
        max_drawdown = drawdown.min()

        return {
            "N_Days": n_days,
            "Total_Return": cumulative,
            "Annualised_Return": annualised_return,
            "Annualised_Volatility": annualised_volatility,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": max_drawdown,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def compare_strategies_by_regime(
    strategy_returns: Dict[str, pd.Series],
    slices: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Compare multiple strategies across the same set of time slices.

    Args:
        strategy_returns: Dict mapping strategy name to its daily returns Series.
        slices: Dict mapping slice name to (start, end) tuples.

    Returns:
        Multi-index DataFrame with (Slice, Strategy) as index and metrics as columns.

    Example:
        >>> returns = {
        ...     "EqualWeight": ew_returns,
        ...     "MinVariance": mv_returns,
        ... }
        >>> slices = {"2020_Q1": ("2020-01-01", "2020-03-31")}
        >>> df = compare_strategies_by_regime(returns, slices)
    """
    all_results = []
    for strategy_name, returns in strategy_returns.items():
        evaluator = SliceEvaluator(returns)
        slice_df = evaluator.evaluate_slices(slices)
        slice_df["Strategy"] = strategy_name
        all_results.append(slice_df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.set_index(["Slice", "Strategy"])
    logger.info(
        "compare_strategies_by_regime: compared %d strategies over %d slices.",
        len(strategy_returns),
        len(slices),
    )
    return combined
