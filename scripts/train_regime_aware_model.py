"""Evaluate plain and regime-aware models on the same frozen holdout.

Compares Rank IC and portfolio metrics for the selected Phase 3.3 model with
and without regime routing. A regime-aware upgrade is accepted only when Rank
IC, Sharpe, and drawdown all improve simultaneously on holdout data.

Usage:
    PYTHONPATH=src python scripts/train_regime_aware_model.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_ml.modeling.phase3_3_evaluation import (
    evaluate_portfolio_methods,
    frozen_holdout_predictions,
    ranking_metrics,
)
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets
from scripts.train_ranking_model import load_price_data_wide

OUTPUT_DIR = Path("outputs/phase3_3")
CUTOFF = pd.Timestamp("2024-12-31")
HORIZON = 21


def _latest_config(output_dir: Path) -> dict:
    """Load the newest versioned best-model config from selection."""
    paths = sorted(output_dir.glob("best_model_*.json"))
    if not paths:
        raise FileNotFoundError(
            "Run scripts/run_model_selection.py before regime evaluation."
        )
    return json.loads(paths[-1].read_text())


def _portfolio_row(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """Return holdout metrics for the frozen portfolio method."""
    results = evaluate_portfolio_methods(
        predictions,
        prices.loc[prices.index > CUTOFF],
        top_k=config["top_k"],
        max_weight=config["max_weight"],
    )
    matched = results[results["portfolio_method"] == config["portfolio_method"]]
    if matched.empty:
        raise ValueError(
            f"No holdout metrics for portfolio_method={config['portfolio_method']}"
        )
    return matched.iloc[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = _latest_config(OUTPUT_DIR)
    prices = load_price_data_wide()
    features = build_features_and_targets(prices, horizon=HORIZON)

    plain = frozen_holdout_predictions(
        features,
        config["model_name"],
        config["params"],
        CUTOFF,
    )
    regime = frozen_holdout_predictions(
        features,
        config["model_name"],
        config["params"],
        CUTOFF,
        regime_aware=True,
    )

    plain_rank = ranking_metrics(plain)
    regime_rank = ranking_metrics(regime)
    plain_best = _portfolio_row(plain, prices, config)
    regime_best = _portfolio_row(regime, prices, config)

    # Paired bootstrap requires daily return series persistence; placeholder
    # until backtest returns are written alongside holdout artifacts.
    paired = {
        "available": False,
        "reason": "daily return pairing requires backtest return persistence",
    }

    # Ranking and portfolio improvements are both required for an upgrade.
    regime_improves = (
        regime_rank["rank_ic"] > plain_rank["rank_ic"]
        and regime_best["sharpe_ratio"] > plain_best["sharpe_ratio"]
        and regime_best["max_drawdown"] >= plain_best["max_drawdown"]
    )
    decision = "selected" if regime_improves else "rejected"
    rejection = (
        None
        if regime_improves
        else (
            "regime-aware model did not improve Rank IC, Sharpe, "
            "and drawdown simultaneously"
        )
    )

    comparison = [
        {
            "model": "plain",
            **plain_rank,
            "sharpe": plain_best["sharpe_ratio"],
            "return": plain_best["annualised_return"],
            "volatility": plain_best["annualised_volatility"],
            "max_drawdown": plain_best["max_drawdown"],
            "turnover": plain_best["avg_turnover"],
        },
        {
            "model": "regime_aware",
            **regime_rank,
            "sharpe": regime_best["sharpe_ratio"],
            "return": regime_best["annualised_return"],
            "volatility": regime_best["annualised_volatility"],
            "max_drawdown": regime_best["max_drawdown"],
            "turnover": regime_best["avg_turnover"],
        },
    ]
    pd.DataFrame(comparison).to_csv(
        OUTPUT_DIR / f"regime_vs_global_comparison_{run_id}.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "decision": decision,
                "reason": rejection or "robust holdout improvement",
                "cutoff": str(CUTOFF.date()),
                "plain_fallback": 0,
                "regime_fallback": 0,
                "paired_bootstrap": json.dumps(paired),
            }
        ]
    ).to_csv(OUTPUT_DIR / f"regime_aware_fold_results_{run_id}.csv", index=False)

    config["regime_status"] = "regime_aware" if decision == "selected" else "plain"
    config["regime_decision"] = decision
    config["regime_rejection_reason"] = rejection
    config["regime_evaluation_run_id"] = run_id
    (OUTPUT_DIR / f"best_model_regime_{run_id}.json").write_text(
        json.dumps(config, indent=2, default=str)
    )

    print(f"Regime evaluation complete. Decision: {decision}")
    print(f"  Artifacts written under {OUTPUT_DIR}/")
    print(f"  Run ID: {run_id}")


if __name__ == "__main__":
    main()
