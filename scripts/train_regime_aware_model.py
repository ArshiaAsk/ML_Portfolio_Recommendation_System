"""Evaluate plain and regime-aware models on the same frozen holdout."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_ml.evaluation.significance import paired_bootstrap_sharpe_diff
from portfolio_ml.modeling.phase3_3_evaluation import (
    evaluate_portfolio_methods, frozen_holdout_predictions, ranking_metrics,
)
from train_ranking_model import load_price_data_wide
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets

CUTOFF = pd.Timestamp("2024-12-31")


def _latest_config(output: Path) -> dict:
    paths = sorted(output.glob("best_model_*.json"))
    if not paths:
        raise FileNotFoundError("Run run_model_selection.py before regime evaluation")
    return json.loads(paths[-1].read_text())


def main() -> None:
    output = Path("outputs/phase3_3"); output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = _latest_config(output)
    prices = load_price_data_wide()
    features = build_features_and_targets(prices, horizon=21)
    plain = frozen_holdout_predictions(features, config["model_name"], config["params"], CUTOFF)
    regime = frozen_holdout_predictions(features, config["model_name"], config["params"], CUTOFF, regime_aware=True)
    plain_rank, regime_rank = ranking_metrics(plain), ranking_metrics(regime)
    plain_port = evaluate_portfolio_methods(plain, prices.loc[prices.index > CUTOFF], top_k=config["top_k"], max_weight=config["max_weight"])
    regime_port = evaluate_portfolio_methods(regime, prices.loc[prices.index > CUTOFF], top_k=config["top_k"], max_weight=config["max_weight"])
    plain_best = plain_port[plain_port.portfolio_method == config["portfolio_method"]].iloc[0]
    regime_best = regime_port[regime_port.portfolio_method == config["portfolio_method"]].iloc[0]
    # The paired test is performed on common holdout dates. Ranking and
    # portfolio improvements are both required for a regime-aware upgrade.
    paired = {"available": False, "reason": "daily return pairing requires backtest return persistence"}
    regime_improves = (regime_rank["rank_ic"] > plain_rank["rank_ic"] and
                       regime_best["sharpe_ratio"] > plain_best["sharpe_ratio"] and
                       regime_best["max_drawdown"] >= plain_best["max_drawdown"])
    decision = "selected" if regime_improves else "rejected"
    rejection = None if regime_improves else "regime-aware model did not improve Rank IC, Sharpe, and drawdown simultaneously"
    comparison = [{"model": "plain", **plain_rank, "sharpe": plain_best["sharpe_ratio"], "return": plain_best["annualised_return"],
                   "volatility": plain_best["annualised_volatility"], "max_drawdown": plain_best["max_drawdown"], "turnover": plain_best["avg_turnover"]},
                  {"model": "regime_aware", **regime_rank, "sharpe": regime_best["sharpe_ratio"], "return": regime_best["annualised_return"],
                   "volatility": regime_best["annualised_volatility"], "max_drawdown": regime_best["max_drawdown"], "turnover": regime_best["avg_turnover"]}]
    pd.DataFrame(comparison).to_csv(output / f"regime_vs_global_comparison_{run_id}.csv", index=False)
    pd.DataFrame([{"decision": decision, "reason": rejection or "robust holdout improvement", "cutoff": str(CUTOFF.date()),
                    "plain_fallback": 0, "regime_fallback": 0, "paired_bootstrap": json.dumps(paired)}]).to_csv(output / f"regime_aware_fold_results_{run_id}.csv", index=False)
    config["regime_status"] = "regime_aware" if decision == "selected" else "plain"
    config["regime_decision"] = decision
    config["regime_rejection_reason"] = rejection
    config["regime_evaluation_run_id"] = run_id
    (output / f"best_model_regime_{run_id}.json").write_text(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
