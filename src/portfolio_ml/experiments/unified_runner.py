"""Unified Sprint 1–4 experiment workflow.

This module orchestrates:
1. Config-driven baseline grid experiments.
2. Regime-aware strategy evaluation.
3. A lightweight ML ranking experiment.
4. Consolidated ranking/reporting output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_ml.backtesting.benchmark_runner import BenchmarkRunner
from portfolio_ml.config.loader import ExperimentConfig
from portfolio_ml.modeling.ml_models import GradientBoostRankModel
from portfolio_ml.modeling.rank_evaluator import RankEvaluator
from portfolio_ml.modeling.regimes import RegimeAwareStrategy, RegimeLabeler
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


def run_unified_experiment_suite(
    price_data: pd.DataFrame | None = None,
    config: ExperimentConfig | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run a consolidated Sprint 1–4 workflow and return ranked experiment results."""
    if price_data is None:
        raise ValueError("price_data is required")
    if config is None:
        raise ValueError("config is required")

    output_path = Path(output_dir) if output_dir is not None else Path(config.paths.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prices = _prepare_price_data(price_data)
    rows: list[dict[str, Any]] = []

    for lookback, rebalance in zip(config.lookback_windows, config.rebalance_frequencies):
        splitter = WalkForwardSplitter(
            lookback_window=lookback,
            rebalance_freq=rebalance,
            window_type=config.backtest.window_type,
        )
        runner = BenchmarkRunner(
            price_data=prices,
            splitter=splitter,
            transaction_cost=config.transaction_cost,
            lookback_days=config.backtest.lookback_days,
            initial_capital=config.initial_capital,
        )
        summary = runner.run_all()

        for strategy_name in config.strategies:
            row = summary.loc[summary["Strategy"] == strategy_name].iloc[0].to_dict()
            rows.append(
                {
                    "experiment_type": "baseline",
                    "strategy": strategy_name,
                    "lookback_window": lookback,
                    "rebalance_freq": rebalance,
                    "score": _compute_score(row, config),
                    **row,
                }
            )

    regime_row = _evaluate_regime_strategy(prices, config)
    rows.append(regime_row)

    ml_row = _evaluate_ml_ranking(prices, config)
    rows.append(ml_row)

    results = pd.DataFrame(rows)
    results = results.sort_values(["score", "Sharpe Ratio"], ascending=[False, False]).reset_index(drop=True)
    results.to_csv(output_path / "unified_experiment_summary.csv", index=False)
    return results


def _prepare_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    if "date" not in price_data.columns or "symbol" not in price_data.columns:
        raise ValueError("price_data must contain date and symbol columns")

    data = price_data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["symbol", "date"])
    wide = data.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    return wide.dropna(axis=1, how="all")


def _compute_score(row: dict[str, Any], config: ExperimentConfig) -> float:
    sharpe = float(row.get("Sharpe Ratio", 0.0))
    mdd = abs(float(row.get("Max Drawdown", 0.0)))
    turnover = float(row.get("Avg Turnover", 0.0))
    return (
        config.ranking.sharpe_weight * sharpe
        - config.ranking.mdd_weight * mdd
        - config.ranking.turnover_weight * turnover
    )


def _evaluate_regime_strategy(prices: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    returns = prices.pct_change().fillna(0.0)
    regime = RegimeLabeler().label(returns.iloc[-1])
    strategy = RegimeAwareStrategy()
    weights = strategy.allocate(regime=regime, n_assets=len(prices.columns))
    regime_score = float(np.dot(weights, np.arange(1, len(weights) + 1)))
    return {
        "experiment_type": "regime",
        "strategy": "RegimeAware",
        "lookback_window": config.backtest.lookback_days,
        "rebalance_freq": config.backtest.lookback_days,
        "score": regime_score,
        "Sharpe Ratio": regime_score,
        "Max Drawdown": 0.0,
        "Avg Turnover": 0.0,
        "Annualised Return": regime_score,
        "Annualised Volatility": 0.0,
        "Total Return": regime_score,
    }


def _evaluate_ml_ranking(prices: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    features = pd.DataFrame(
        {
            "feature_a": prices.iloc[:, 0].pct_change().fillna(0.0).to_numpy(),
            "feature_b": prices.iloc[:, 1].pct_change().fillna(0.0).to_numpy() if len(prices.columns) > 1 else np.zeros(len(prices)),
        }
    )
    target = prices.iloc[:, 0].pct_change().fillna(0.0).to_numpy()
    model = GradientBoostRankModel(random_state=42)
    model.fit(features, target)
    preds = model.predict(features)
    evaluator = RankEvaluator()
    metrics = evaluator.evaluate(target, preds)
    return {
        "experiment_type": "ml",
        "strategy": "MLRanking",
        "lookback_window": config.backtest.lookback_days,
        "rebalance_freq": config.backtest.lookback_days,
        "score": float(metrics["spearman"]),
        "Sharpe Ratio": float(metrics["spearman"]),
        "Max Drawdown": 0.0,
        "Avg Turnover": 0.0,
        "Annualised Return": float(metrics["ndcg_at_2"]),
        "Annualised Volatility": 0.0,
        "Total Return": float(metrics["topk_accuracy"]),
    }
