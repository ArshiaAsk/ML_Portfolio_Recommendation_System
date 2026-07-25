"""Select Phase 3.3 models and portfolio controls without holdout leakage.

Walk-forward hyperparameter search and portfolio-method selection are confined
to data on or before ``SELECTION_CUTOFF``. Holdout metrics are computed once on
frozen predictions and never feed back into selection.

Usage:
    PYTHONPATH=src python scripts/run_model_selection.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_ml.evaluation.significance import ic_ttest
from portfolio_ml.modeling.phase3_3_evaluation import (
    RiskControlConfig,
    evaluate_portfolio_methods,
    frozen_holdout_predictions,
)
from portfolio_ml.modeling.ranking_pipeline import (
    build_features_and_targets,
    run_walk_forward_ranking,
)
from scripts.train_ranking_model import load_price_data_wide

OUTPUT_DIR = Path("outputs/phase3_3")
SELECTION_CUTOFF = pd.Timestamp("2024-12-31")
HORIZON = 21
LOOKBACK = 252
REBALANCE_FREQ = 21
TOP_K_RANK = 3
TOP_K_PORTFOLIO = 5
MAX_WEIGHT = 0.25
ALPHA = 0.05

GRIDS: dict[str, list[dict]] = {
    "ridge": [{"alpha": alpha} for alpha in (0.1, 1.0, 10.0)],
    "random_forest": [
        {"n_estimators": n, "max_depth": d}
        for n, d in ((100, 4), (200, 6), (300, 8))
    ],
    "gradient_boost": [
        {"max_iter": n, "max_depth": d, "learning_rate": lr}
        for n, d, lr in ((100, 3, 0.1), (200, 4, 0.05), (300, 6, 0.03))
    ],
    "ensemble": [{}],
}


def _score_config(
    features: pd.DataFrame,
    model_name: str,
    params: dict,
) -> tuple[dict, pd.DataFrame]:
    """Score one model/config on pre-cutoff walk-forward folds only."""
    selection_dates = sorted(
        features.loc[features["date"] <= SELECTION_CUTOFF, "date"].unique()
    )
    # Keep labels whose complete forward horizon is inside the selection set.
    usable_dates = set(selection_dates[:-HORIZON])

    folds, predictions = run_walk_forward_ranking(
        features[features["date"].isin(usable_dates)],
        model_name,
        params,
        horizon=HORIZON,
        lookback=LOOKBACK,
        rebalance_freq=REBALANCE_FREQ,
        top_k=TOP_K_RANK,
    )

    fold_ics = [row["mean_rank_ic"] for row in folds]
    test = ic_ttest(fold_ics)
    precision_values = [row["precision_at_3"] for row in folds]

    result = {
        "model_name": model_name,
        "params": json.dumps(params, sort_keys=True),
        "mean_rank_ic": test["mean_ic"],
        "std_rank_ic": float(pd.Series(fold_ics).std()) if fold_ics else 0.0,
        "pct_positive_folds": (
            float(pd.Series(fold_ics).gt(0).mean()) if fold_ics else 0.0
        ),
        "n_folds": len(folds),
        "p_value": test["p_value"],
        "significant": test["significant"],
        "precision_at_3": (
            float(pd.Series(precision_values).mean()) if folds else 0.0
        ),
        "fold_metrics": json.dumps(folds, default=str),
    }
    return result, predictions


def _select_portfolio_method(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Choose a portfolio method on pre-cutoff predictions only."""
    method_results = evaluate_portfolio_methods(
        predictions,
        prices.loc[:SELECTION_CUTOFF],
        top_k=TOP_K_PORTFOLIO,
        max_weight=MAX_WEIGHT,
    )
    if method_results.empty:
        return "top_k_equal", method_results

    selected = method_results.sort_values(
        ["sharpe_ratio", "max_drawdown", "avg_turnover"],
        ascending=[False, False, True],
    ).iloc[0]["portfolio_method"]
    return str(selected), method_results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    prices = load_price_data_wide()
    features = build_features_and_targets(prices, horizon=HORIZON)

    rows: list[dict] = []
    prediction_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for model_name, configs in GRIDS.items():
        for params in configs:
            row, predictions = _score_config(features, model_name, params)
            rows.append(row)
            cache_key = (model_name, json.dumps(params, sort_keys=True))
            prediction_cache[cache_key] = predictions

    results = pd.DataFrame(rows)
    # Bonferroni correction is applied across the complete fixed grid.
    results["adjusted_p_value"] = (results["p_value"] * len(results)).clip(upper=1.0)
    results["adjusted_significant"] = results["adjusted_p_value"] < ALPHA
    results.to_csv(OUTPUT_DIR / f"hyperparam_search_results_{run_id}.csv", index=False)

    eligible = results[results["adjusted_significant"]]
    if eligible.empty:
        eligible = results

    winner = eligible.sort_values(
        ["mean_rank_ic", "precision_at_3"],
        ascending=False,
    ).iloc[0]
    winner_name = winner["model_name"]
    winner_params = json.loads(winner["params"])
    winner_predictions = prediction_cache[(winner_name, winner["params"])]

    selected_method, method_results = _select_portfolio_method(
        winner_predictions,
        prices,
    )
    method_results.to_csv(
        OUTPUT_DIR / f"portfolio_method_results_{run_id}.csv",
        index=False,
    )

    payload = {
        "model_name": winner_name,
        "params": winner_params,
        "portfolio_method": selected_method,
        "top_k": TOP_K_PORTFOLIO,
        "max_weight": MAX_WEIGHT,
        "decision": "selected",
        "selection_cutoff": str(SELECTION_CUTOFF.date()),
        "selection_run_id": run_id,
        "selection_metrics": winner.to_dict(),
        "portfolio_selection_rationale": (
            "selected on identical pre-cutoff walk-forward data using "
            "risk-adjusted metrics"
        ),
        "regime_status": "plain",
        "holdout_status": "pending",
    }
    (OUTPUT_DIR / f"best_model_{run_id}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )

    # Risk-control grid is evaluated only after choices are frozen; it is
    # reported for review and never changes the selection decision.
    controls: list[pd.DataFrame] = []
    for top_k in (5, 7, 10):
        for max_weight in (0.25, 0.20, 0.15):
            metrics = evaluate_portfolio_methods(
                winner_predictions,
                prices.loc[:SELECTION_CUTOFF],
                top_k=top_k,
                max_weight=max_weight,
                risk_control=RiskControlConfig(),
            )
            if not metrics.empty:
                metrics = metrics.copy()
                metrics["top_k"] = top_k
                metrics["max_weight"] = max_weight
                controls.append(metrics)

    if controls:
        pd.concat(controls, ignore_index=True).to_csv(
            OUTPUT_DIR / f"risk_control_results_{run_id}.csv",
            index=False,
        )

    # Final untouched holdout: no model or method is refit or selected here.
    holdout_predictions = frozen_holdout_predictions(
        features,
        winner_name,
        winner_params,
        SELECTION_CUTOFF,
    )
    holdout_metrics = evaluate_portfolio_methods(
        holdout_predictions,
        prices.loc[prices.index > SELECTION_CUTOFF],
        top_k=int(payload["top_k"]),
        max_weight=float(payload["max_weight"]),
    )
    holdout_metrics.to_csv(OUTPUT_DIR / f"holdout_results_{run_id}.csv", index=False)

    print(f"Selection complete. Artifacts written under {OUTPUT_DIR}/")
    print(f"  Winner: {winner_name} / {selected_method}")
    print(f"  Run ID: {run_id}")


if __name__ == "__main__":
    main()
