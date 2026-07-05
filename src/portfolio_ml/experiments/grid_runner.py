"""Systematic experiment runner for baseline portfolio strategies.

This module turns the Phase 3.1 Sprint 1 requirements into a reusable
config-driven workflow:

1. Load a parameter grid from configs/experiments.yaml.
2. Generate walk-forward splits for each (lookback, rebalance) pair.
3. Run EqualWeight / RiskParity / MinVariance backtests.
4. Rank runs by a simple composite score based on Sharpe, MDD, and turnover.
5. Persist a consolidated summary and log to MLflow when available.
"""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_ml.backtesting.benchmark_runner import BenchmarkRunner
from portfolio_ml.config.loader import ExperimentConfig, load_experiment_config
from portfolio_ml.experiments.tracking import log_artifact, log_metrics, log_params, start_experiment
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentGridRunner:
    """Run a systematic grid of portfolio experiments."""

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or load_experiment_config()
        self.project_root = Path(__file__).resolve().parents[3]
        self.output_dir = self.project_root / self.config.paths.output_dir

    def run(self) -> pd.DataFrame:
        """Execute the configured grid of experiments and return a ranked summary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.mlflow.tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow.tracking_uri
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        prices = self._load_price_data()
        runs: list[dict[str, Any]] = []

        for lookback, rebalance, strategy_name in product(
            self.config.lookback_windows,
            self.config.rebalance_frequencies,
            self.config.strategies,
        ):
            splitter = WalkForwardSplitter(
                lookback_window=lookback,
                rebalance_freq=rebalance,
                window_type=self.config.backtest.window_type,
            )
            runner = BenchmarkRunner(
                price_data=prices,
                splitter=splitter,
                transaction_cost=self.config.transaction_cost,
                lookback_days=self.config.backtest.lookback_days,
                initial_capital=self.config.initial_capital,
            )
            summary = runner.run_all()
            row = summary.loc[summary["Strategy"] == strategy_name].iloc[0].to_dict()

            score = self._compute_ranking_score(row)
            run_record = {
                "lookback_window": lookback,
                "rebalance_freq": rebalance,
                "strategy": strategy_name,
                "score": score,
                **row,
            }
            runs.append(run_record)

            self._log_run(run_record, splitter, prices)

        results = pd.DataFrame(runs)
        results = results.sort_values(["score", "Sharpe Ratio"], ascending=[False, False]).reset_index(drop=True)
        results.to_csv(self.output_dir / "experiment_summary.csv", index=False)
        logger.info("Saved %d experiment runs to %s", len(results), self.output_dir / "experiment_summary.csv")
        return results

    def _load_price_data(self) -> pd.DataFrame:
        price_dir = self.project_root / self.config.paths.price_data_dir
        price_files = sorted(price_dir.glob("**/daily_prices.parquet"))
        if not price_files:
            raise FileNotFoundError(
                f"No price data found in {price_dir}. Run the preprocessing pipeline first."
            )

        frames = [pd.read_parquet(path) for path in price_files]
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        prices = combined.pivot(index="date", columns="symbol", values="adj_close").sort_index()
        return prices.dropna(axis=1, how="all")

    def _compute_ranking_score(self, row: dict[str, Any]) -> float:
        sharpe = float(row.get("Sharpe Ratio", 0.0))
        mdd = abs(float(row.get("Max Drawdown", 0.0)))
        turnover = float(row.get("Avg Turnover", 0.0))
        return (
            self.config.ranking.sharpe_weight * sharpe
            - self.config.ranking.mdd_weight * mdd
            - self.config.ranking.turnover_weight * turnover
        )

    def _log_run(self, row: dict[str, Any], splitter: WalkForwardSplitter, prices: pd.DataFrame) -> None:
        with start_experiment(
            self.config.mlflow.experiment_name,
            run_name=f"{row['strategy']}_{row['lookback_window']}_{row['rebalance_freq']}",
        ):
            log_params(
                {
                    "strategy": row["strategy"],
                    "lookback_window": row["lookback_window"],
                    "rebalance_freq": row["rebalance_freq"],
                    "window_type": self.config.backtest.window_type,
                    "transaction_cost": self.config.transaction_cost,
                }
            )
            log_metrics(
                {
                    "sharpe_ratio": float(row.get("Sharpe Ratio", 0.0)),
                    "max_drawdown": float(row.get("Max Drawdown", 0.0)),
                    "avg_turnover": float(row.get("Avg Turnover", 0.0)),
                    "score": float(row.get("score", 0.0)),
                }
            )
            summary_path = self.output_dir / "experiment_summary.csv"
            if summary_path.exists():
                log_artifact(summary_path)


if __name__ == "__main__":
    runner = ExperimentGridRunner()
    summary = runner.run()
    print(summary.to_string(index=False))
