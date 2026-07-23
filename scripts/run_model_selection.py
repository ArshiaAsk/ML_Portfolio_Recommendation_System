"""Select Phase 3.3 models and portfolio controls without holdout leakage."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_ml.evaluation.significance import ic_ttest
from portfolio_ml.modeling.phase3_3_evaluation import (
    RiskControlConfig, evaluate_portfolio_methods, frozen_holdout_predictions,
    ranking_metrics,
)
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets, run_walk_forward_ranking
from scripts.train_ranking_model import load_price_data_wide

SELECTION_CUTOFF = pd.Timestamp("2024-12-31")
GRIDS = {
    "ridge": [{"alpha": x} for x in (.1, 1, 10)],
    "random_forest": [{"n_estimators": n, "max_depth": d} for n, d in ((100, 4), (200, 6), (300, 8))],
    "gradient_boost": [{"max_iter": n, "max_depth": d, "learning_rate": lr}
                       for n, d, lr in ((100, 3, .1), (200, 4, .05), (300, 6, .03))],
    "ensemble": [{}],
}


def _score_config(features: pd.DataFrame, name: str, params: dict) -> tuple[dict, pd.DataFrame]:
    selection_dates = sorted(features.loc[features.date <= SELECTION_CUTOFF, "date"].unique())
    # Keep labels whose complete forward horizon is inside the selection set.
    usable_dates = set(selection_dates[:-21])
    folds, predictions = run_walk_forward_ranking(
        features[features.date.isin(usable_dates)], name, params,
        horizon=21, lookback=252, rebalance_freq=21, top_k=3,
    )
    values = [row["mean_rank_ic"] for row in folds]
    test = ic_ttest(values)
    result = {
        "model_name": name, "params": json.dumps(params, sort_keys=True),
        "mean_rank_ic": test["mean_ic"], "std_rank_ic": float(pd.Series(values).std()) if values else 0.0,
        "pct_positive_folds": float(pd.Series(values).gt(0).mean()) if values else 0.0,
        "n_folds": len(folds), "p_value": test["p_value"], "significant": test["significant"],
        "precision_at_3": float(pd.Series([x["precision_at_3"] for x in folds]).mean()) if folds else 0.0,
        "fold_metrics": json.dumps(folds, default=str),
    }
    return result, predictions


def main() -> None:
    output = Path("outputs/phase3_3")
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prices = load_price_data_wide()
    features = build_features_and_targets(prices, horizon=21)

    rows, prediction_cache = [], {}
    for model_name, configs in GRIDS.items():
        for params in configs:
            row, predictions = _score_config(features, model_name, params)
            rows.append(row)
            prediction_cache[(model_name, json.dumps(params, sort_keys=True))] = predictions
    results = pd.DataFrame(rows)
    # Bonferroni correction is applied across the complete fixed grid.
    results["adjusted_p_value"] = (results["p_value"] * len(results)).clip(upper=1.0)
    results["adjusted_significant"] = results["adjusted_p_value"] < .05
    results.to_csv(output / f"hyperparam_search_results_{run_id}.csv", index=False)

    eligible = results[results.adjusted_significant]
    if eligible.empty:
        eligible = results
    winner = eligible.sort_values(["mean_rank_ic", "precision_at_3"], ascending=False).iloc[0]
    winner_name = winner.model_name
    winner_params = json.loads(winner.params)
    winner_predictions = prediction_cache[(winner_name, winner.params)]

    method_results = evaluate_portfolio_methods(
        winner_predictions, prices.loc[:SELECTION_CUTOFF], top_k=5, max_weight=.25,
    )
    method_results.to_csv(output / f"portfolio_method_results_{run_id}.csv", index=False)
    selected_method = method_results.sort_values(
        ["sharpe_ratio", "max_drawdown", "avg_turnover"], ascending=[False, False, True]
    ).iloc[0]["portfolio_method"] if not method_results.empty else "top_k_equal"

    payload = {
        "model_name": winner_name, "params": winner_params,
        "portfolio_method": selected_method, "top_k": 5, "max_weight": .25,
        "decision": "selected", "selection_cutoff": str(SELECTION_CUTOFF.date()),
        "selection_run_id": run_id, "selection_metrics": winner.to_dict(),
        "portfolio_selection_rationale": "selected on identical pre-cutoff walk-forward data using risk-adjusted metrics",
        "regime_status": "plain", "holdout_status": "pending",
    }
    (output / f"best_model_{run_id}.json").write_text(json.dumps(payload, indent=2, default=str))
    # A small, explicit risk-control grid is evaluated only after choices are
    # frozen; it is reported for review and never changes the selection data.
    controls = []
    for top_k in (5, 7, 10):
        for max_weight in (.25, .20, .15):
            metrics = evaluate_portfolio_methods(
                winner_predictions, prices.loc[:SELECTION_CUTOFF], top_k=top_k,
                max_weight=max_weight, risk_control=RiskControlConfig(),
            )
            if not metrics.empty:
                metrics["top_k"] = top_k; metrics["max_weight"] = max_weight; controls.append(metrics)
    if controls:
        pd.concat(controls, ignore_index=True).to_csv(output / f"risk_control_results_{run_id}.csv", index=False)

    # Final untouched holdout: no model or method is refit or selected here.
    holdout_predictions = frozen_holdout_predictions(features, winner_name, winner_params, SELECTION_CUTOFF)
    holdout_metrics = evaluate_portfolio_methods(holdout_predictions, prices.loc[prices.index > SELECTION_CUTOFF],
                                                  top_k=int(payload["top_k"]), max_weight=float(payload["max_weight"]))
    holdout_metrics.to_csv(output / f"holdout_results_{run_id}.csv", index=False)


if __name__ == "__main__":
    main()
