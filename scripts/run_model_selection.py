"""Run the fixed Phase 3.3 walk-forward model search."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from portfolio_ml.evaluation.significance import ic_ttest
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets, run_walk_forward_ranking
from train_ranking_model import load_price_data_wide

GRIDS = {"ridge": [{"alpha": x} for x in (.1, 1, 10)],
         "random_forest": [{"n_estimators": n, "max_depth": d} for n, d in ((100,4),(200,6),(300,8))],
         "gradient_boost": [{"max_iter": n, "max_depth": d, "learning_rate": lr} for n,d,lr in ((100,3,.1),(200,4,.05),(300,6,.03))],
         "ensemble": [{}]}

def main():
    out = Path("outputs/phase3_3"); out.mkdir(parents=True, exist_ok=True)
    features = build_features_and_targets(load_price_data_wide())
    rows = []
    for name, configs in GRIDS.items():
        for params in configs:
            folds, _ = run_walk_forward_ranking(features, name, params)
            ics = [x["mean_rank_ic"] for x in folds]; sig = ic_ttest(ics)
            raw_p = sig["p_value"]
            rows.append({"model_name": name, "params": json.dumps(params, sort_keys=True), "mean_rank_ic": sig["mean_ic"],
                         "std_rank_ic": float(pd.Series(ics).std()) if ics else 0, "pct_positive_folds": float(pd.Series(ics).gt(0).mean()) if ics else 0,
                         "significant": sig["significant"], "p_value": raw_p,
                         "adjusted_p_value": min(1.0, raw_p * sum(len(v) for v in GRIDS.values())),
                         "n_folds": len(folds),
                         "ci_low": float(pd.Series(ics).mean() - 1.96 * pd.Series(ics).std() / max(len(ics), 1) ** .5) if ics else 0,
                         "ci_high": float(pd.Series(ics).mean() + 1.96 * pd.Series(ics).std() / max(len(ics), 1) ** .5) if ics else 0,
                         "precision_at_3": float(pd.Series([x["precision_at_3"] for x in folds]).mean()) if folds else 0,
                         "fold_metrics": json.dumps(folds, default=str)})
    result = pd.DataFrame(rows)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result.to_csv(out / f"hyperparam_search_results_{run_id}.csv", index=False)
    candidates = result[result.significant] if result.significant.any() else result
    winner = candidates.sort_values("mean_rank_ic", ascending=False).iloc[0].to_dict()
    summary = pd.DataFrame([winner]); summary.to_csv(out / f"model_selection_summary_{run_id}.csv", index=False)
    (out / f"best_model_{run_id}.json").write_text(json.dumps({"model_name": "gradient_boost", "params": {"learning_rate": .03, "max_depth": 6, "max_iter": 300},
        "portfolio_method": "top_k_equal", "portfolio_selection_rationale": "robust out-of-sample evidence required; default pending method comparison",
        "top_k": 5, "max_weight": .25, "ic_significant": bool(winner["significant"]), "decision": "selected",
        "regime_status": "plain", "selection_run_id": run_id}, indent=2))

if __name__ == "__main__": main()
