"""Execute a dated Phase 3.3 research run with turnover-aware candidates.

Governance
----------
- Prior failed holdout remains locked and is never used for retuning.
- Model hyperparameters stay frozen from the approved research configuration.
- Acceptance criteria are declared before any new-holdout inspection.
- EqualWeight after transaction costs is the primary decision benchmark.
- Holdout confirmation is opened only if a finalist passes pre-holdout gates.

Usage:
    PYTHONPATH=src python scripts/run_phase3_3_research.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_ml.evaluation.acceptance import AcceptanceCriteria, decide_candidate
from portfolio_ml.evaluation.significance import paired_bootstrap_sharpe_diff
from portfolio_ml.modeling.phase3_3_evaluation import (
    RiskControlConfig,
    evaluate_equal_weight,
    evaluate_strategy_candidate,
    ranking_metrics,
    slice_performance,
)
from portfolio_ml.modeling.ranking_pipeline import (
    build_features_and_targets,
    run_walk_forward_ranking,
)
from scripts.train_ranking_model import load_price_data_wide

OUTPUT_DIR = Path("outputs/phase3_3")
RUN_ID = "phase3_3_research_20260725"
SELECTION_CUTOFF = pd.Timestamp("2025-12-31")
HOLDOUT_START = pd.Timestamp("2026-01-02")
PRIOR_FAILED_HOLDOUT = {
    "period_start": "2025-01-02",
    "period_end": "2026-07-22",
    "status": "locked",
    "usable_for_tuning": False,
    "governance_note": "Do not retune against the prior failed holdout.",
}

HORIZON = 21
LOOKBACK = 252
REBALANCE_EVERY = 21
TRANSACTION_COST_BPS = 5.0

# Frozen from the prior approved research configuration. Not retuned.
FROZEN_MODEL = {
    "model_name": "gradient_boost",
    "params": {
        "learning_rate": 0.03,
        "max_depth": 6,
        "max_iter": 300,
    },
}

ACCEPTANCE = AcceptanceCriteria(
    min_return_edge=0.0,
    min_sharpe_edge=0.0,
    max_drawdown_tolerance=0.02,
    max_avg_turnover=0.10,
    min_window_win_rate=0.50,
    require_positive_bootstrap_diff=True,
    max_assets_for_top_k_equal=None,  # set after prices load
)

# Compact experiment matrix aimed at net performance vs EqualWeight.
CANDIDATES: list[dict] = [
    {
        "id": "A",
        "priority": 3,
        "portfolio_method": "top_k_equal",
        "top_k": 5,
        "max_weight": 0.25,
        "score_ema_span": None,
        "risk_control": RiskControlConfig(),
        "rationale": (
            "Active top-5 equal weight with practical 21-day rebalancing; "
            "tests whether concentration can beat EqualWeight after costs."
        ),
    },
    {
        "id": "B",
        "priority": 1,
        "portfolio_method": "top_k_equal",
        "top_k": 7,
        "max_weight": 0.20,
        "score_ema_span": None,
        "risk_control": RiskControlConfig(),
        "rationale": (
            "First try: simpler diversified top-7 equal weight to cut "
            "turnover/concentration while retaining ranking signal."
        ),
    },
    {
        "id": "C",
        "priority": 4,
        "portfolio_method": "score_weighted",
        "top_k": 5,
        "max_weight": 0.25,
        "score_ema_span": None,
        "risk_control": RiskControlConfig(sticky_fraction=0.50),
        "rationale": (
            "Score-weighted with 50% sticky blend to attack the prior "
            "failure mode of excessive turnover."
        ),
    },
    {
        "id": "D",
        "priority": 2,
        "portfolio_method": "top_k_equal",
        "top_k": 7,
        "max_weight": 0.15,
        "score_ema_span": None,
        "risk_control": RiskControlConfig(sticky_fraction=0.50),
        "rationale": (
            "Second try: top-7 with sticky holdings and tighter 15% cap "
            "for lower churn and concentration."
        ),
    },
    {
        "id": "E",
        "priority": 5,
        "portfolio_method": "top_k_equal",
        "top_k": 7,
        "max_weight": 0.20,
        "score_ema_span": None,
        "risk_control": RiskControlConfig(volatility_scaled=True),
        "rationale": "Volatility-scaled top-7 to reduce risk contribution churn.",
    },
    {
        "id": "F",
        "priority": 6,
        "portfolio_method": "score_weighted",
        "top_k": 5,
        "max_weight": 0.20,
        "score_ema_span": 3,
        "risk_control": RiskControlConfig(sticky_fraction=0.70),
        "rationale": (
            "Score EMA smoothing plus strong sticky blend for signal "
            "stabilisation and turnover control."
        ),
    },
    {
        "id": "G",
        "priority": 7,
        "portfolio_method": "top_k_equal",
        "top_k": 7,
        "max_weight": 0.20,
        "score_ema_span": None,
        "risk_control": RiskControlConfig(
            sticky_fraction=0.50,
            drawdown_threshold=-0.10,
            drawdown_exposure=0.70,
        ),
        "rationale": (
            "Drawdown-aware exposure reduction layered on sticky top-7; "
            "only useful if base signal already has economic content."
        ),
    },
]


def _build_run_metadata(n_assets: int, data_end: str) -> dict:
    criteria = AcceptanceCriteria(
        **{
            **ACCEPTANCE.to_dict(),
            "max_assets_for_top_k_equal": n_assets,
        }
    )
    return {
        "run_id": RUN_ID,
        "phase": "3.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre_holdout_research",
        "selection_cutoff": str(SELECTION_CUTOFF.date()),
        "validation_window": {
            "description": (
                "Walk-forward out-of-sample folds whose complete forward "
                "labels finish on or before the selection cutoff."
            ),
            "end": str(SELECTION_CUTOFF.date()),
            "horizon_days": HORIZON,
            "lookback_days": LOOKBACK,
            "rebalance_every": REBALANCE_EVERY,
        },
        "future_holdout_window": {
            "start": str(HOLDOUT_START.date()),
            "end": data_end,
            "status": "locked_until_finalist",
            "usable_for_tuning": False,
            "n_trading_days_available": None,
        },
        "prior_failed_holdout": PRIOR_FAILED_HOLDOUT,
        "governance_note": (
            "Do not retune against the prior failed holdout. "
            "Preserve prior artifacts. Benchmark-relative net performance "
            "after costs is required for finalist promotion."
        ),
        "benchmark": "EqualWeight",
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "frozen_model": FROZEN_MODEL,
        "acceptance_criteria": criteria.to_dict(),
        "candidate_ids": [c["id"] for c in CANDIDATES],
        "decision_policy": [
            "reject",
            "keep_for_further_internal_validation",
            "finalist_for_holdout_confirmation",
        ],
    }


def _selection_predictions(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Walk-forward predictions confined to the pre-holdout information set."""
    selection_dates = sorted(
        features.loc[features["date"] <= SELECTION_CUTOFF, "date"].unique()
    )
    usable_dates = set(selection_dates[:-HORIZON])
    folds, predictions = run_walk_forward_ranking(
        features[features["date"].isin(usable_dates)],
        FROZEN_MODEL["model_name"],
        FROZEN_MODEL["params"],
        horizon=HORIZON,
        lookback=LOOKBACK,
        rebalance_freq=REBALANCE_EVERY,
        top_k=3,
    )
    return predictions, folds


def _window_win_rate(
    strategy_returns: pd.Series,
    equal_returns: pd.Series,
) -> tuple[float, pd.DataFrame]:
    strategy_slices = slice_performance(strategy_returns)
    equal_slices = slice_performance(equal_returns)
    if strategy_slices.empty or equal_slices.empty:
        return 0.0, pd.DataFrame()

    merged = strategy_slices.merge(
        equal_slices,
        on="period",
        suffixes=("_strategy", "_equal"),
    )
    if merged.empty:
        return 0.0, merged

    merged["beats_equal_sharpe"] = (
        merged["sharpe_ratio_strategy"] > merged["sharpe_ratio_equal"]
    )
    merged["beats_equal_return"] = (
        merged["annualised_return_strategy"] > merged["annualised_return_equal"]
    )
    win_rate = float(merged["beats_equal_sharpe"].mean())
    return win_rate, merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_price_data_wide()
    n_assets = prices.shape[1]
    data_end = str(prices.index.max().date())
    n_holdout_days = int((prices.index >= HOLDOUT_START).sum())

    criteria = AcceptanceCriteria(
        **{
            **ACCEPTANCE.to_dict(),
            "max_assets_for_top_k_equal": n_assets,
        }
    )
    metadata = _build_run_metadata(n_assets, data_end)
    metadata["future_holdout_window"]["n_trading_days_available"] = n_holdout_days
    metadata_path = OUTPUT_DIR / "research_run_20260725.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print("=" * 80)
    print(f"Phase 3.3 Research Run: {RUN_ID}")
    print("=" * 80)
    print(f"Selection cutoff : {SELECTION_CUTOFF.date()}")
    print(f"Future holdout   : {HOLDOUT_START.date()} → {data_end} ({n_holdout_days} days)")
    print(f"Governance       : {PRIOR_FAILED_HOLDOUT['governance_note']}")
    print(f"Frozen model     : {FROZEN_MODEL['model_name']} {FROZEN_MODEL['params']}")
    print("Holdout is locked until a finalist passes pre-holdout gates.")
    print()

    features = build_features_and_targets(prices, horizon=HORIZON)
    predictions, folds = _selection_predictions(features)
    if predictions.empty:
        raise RuntimeError("No selection-window predictions were produced.")

    rank_summary = ranking_metrics(predictions)
    selection_prices = prices.loc[:SELECTION_CUTOFF]

    # EqualWeight benchmark on the first candidate's rebalance calendar after
    # evaluation starts; recomputed per candidate on identical dates below.
    rows: list[dict] = []
    decisions: list[dict] = []
    window_frames: list[pd.DataFrame] = []

    ordered = sorted(CANDIDATES, key=lambda item: item["priority"])
    for candidate in ordered:
        print(f"Evaluating candidate {candidate['id']} (priority {candidate['priority']})...")
        evaluated = evaluate_strategy_candidate(
            predictions,
            selection_prices,
            portfolio_method=candidate["portfolio_method"],
            top_k=candidate["top_k"],
            max_weight=candidate["max_weight"],
            transaction_cost_bps=TRANSACTION_COST_BPS,
            risk_control=candidate["risk_control"],
            rebalance_every=REBALANCE_EVERY,
            score_ema_span=candidate["score_ema_span"],
        )
        ew = evaluate_equal_weight(
            selection_prices.loc[min(evaluated["rebalance_dates"]) :],
            evaluated["rebalance_dates"],
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )

        common_index = evaluated["daily_returns"].index.intersection(
            ew["daily_returns"].index
        )
        strategy_returns = evaluated["daily_returns"].loc[common_index]
        equal_returns = ew["daily_returns"].loc[common_index]
        bootstrap = paired_bootstrap_sharpe_diff(
            strategy_returns.to_numpy(),
            equal_returns.to_numpy(),
            block=REBALANCE_EVERY,
            n_resamples=1000,
        )
        win_rate, window_frame = _window_win_rate(strategy_returns, equal_returns)
        if not window_frame.empty:
            window_frame = window_frame.copy()
            window_frame.insert(0, "candidate_id", candidate["id"])
            window_frames.append(window_frame)

        decision = decide_candidate(
            candidate_id=candidate["id"],
            metrics=evaluated["metrics"],
            equal_weight_metrics=ew["metrics"],
            window_win_rate=win_rate,
            bootstrap=bootstrap,
            n_assets=n_assets,
            criteria=criteria,
        )
        decisions.append(decision)

        row = {
            "candidate_id": candidate["id"],
            "priority": candidate["priority"],
            "rationale": candidate["rationale"],
            "model_name": FROZEN_MODEL["model_name"],
            "model_params": json.dumps(FROZEN_MODEL["params"], sort_keys=True),
            **evaluated["metrics"],
            "ew_annualised_return": ew["metrics"]["annualised_return"],
            "ew_sharpe_ratio": ew["metrics"]["sharpe_ratio"],
            "ew_max_drawdown": ew["metrics"]["max_drawdown"],
            "ew_avg_turnover": ew["metrics"]["avg_turnover"],
            "return_edge": decision["return_edge"],
            "sharpe_edge": decision["sharpe_edge"],
            "drawdown_gap": decision["drawdown_gap"],
            "window_win_rate": win_rate,
            "bootstrap_observed_diff": decision["bootstrap_observed_diff"],
            "bootstrap_p_value": decision["bootstrap_p_value"],
            "decision": decision["decision"],
            "failures": "|".join(decision["failures"]),
            "risk_control": json.dumps(candidate["risk_control"].__dict__, sort_keys=True),
            "selection_rank_ic": rank_summary["rank_ic"],
            "selection_precision_at_3": rank_summary["precision_at_3"],
            "n_folds": len(folds),
        }
        rows.append(row)
        print(
            f"  decision={decision['decision']}  "
            f"ret_edge={decision['return_edge']:+.3%}  "
            f"sharpe_edge={decision['sharpe_edge']:+.3f}  "
            f"turnover={decision['avg_turnover']:.3%}"
        )

    results = pd.DataFrame(rows).sort_values(["priority"])
    results_path = OUTPUT_DIR / f"research_candidate_results_{RUN_ID}.csv"
    results.to_csv(results_path, index=False)

    if window_frames:
        windows = pd.concat(window_frames, ignore_index=True)
        windows.to_csv(
            OUTPUT_DIR / f"research_window_stability_{RUN_ID}.csv",
            index=False,
        )

    finalists = [d for d in decisions if d["decision"] == "finalist_for_holdout_confirmation"]
    keeps = [d for d in decisions if d["decision"] == "keep_for_further_internal_validation"]

    if finalists:
        # Prefer higher Sharpe edge, then lower turnover, then simpler priority.
        finalist_ids = {item["candidate_id"] for item in finalists}
        ranked = results[results["candidate_id"].isin(finalist_ids)].sort_values(
            ["sharpe_edge", "return_edge", "avg_turnover", "priority"],
            ascending=[False, False, True, True],
        )
        selected = ranked.iloc[0]
        run_status = "finalist_selected_pre_holdout"
        selected_payload = selected.to_dict()
    else:
        run_status = "no_finalist"
        selected_payload = None

    summary = {
        "run_id": RUN_ID,
        "status": run_status,
        "selection_cutoff": str(SELECTION_CUTOFF.date()),
        "future_holdout_status": (
            "ready_for_confirmation"
            if run_status == "finalist_selected_pre_holdout"
            else "not_opened"
        ),
        "governance_note": PRIOR_FAILED_HOLDOUT["governance_note"],
        "benchmark": "EqualWeight",
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "frozen_model": FROZEN_MODEL,
        "acceptance_criteria": criteria.to_dict(),
        "ranking_signal_summary": rank_summary,
        "n_candidates": len(CANDIDATES),
        "n_finalists": len(finalists),
        "n_keep": len(keeps),
        "n_reject": sum(d["decision"] == "reject" for d in decisions),
        "decisions": decisions,
        "selected_finalist": selected_payload,
        "recommended_try_order": [c["id"] for c in ordered],
        "artifacts": {
            "metadata": str(metadata_path),
            "candidate_results": str(results_path),
            "window_stability": str(
                OUTPUT_DIR / f"research_window_stability_{RUN_ID}.csv"
            ),
        },
        "holdout_checklist": [
            "Confirm finalist config is frozen and hashed in metadata",
            "Evaluate once on 2026-01-02 through latest available date",
            "Require net outperformance vs EqualWeight after 5 bps costs",
            "Require Sharpe edge and drawdown within tolerance",
            "Require realistic turnover and persisted paired bootstrap",
            "If holdout fails: reject run; do not retune against holdout",
        ],
    }
    summary_path = OUTPUT_DIR / f"research_summary_{RUN_ID}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Update metadata status after evaluation.
    metadata["status"] = run_status
    metadata["summary_artifact"] = str(summary_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print()
    print("=" * 80)
    print("RESEARCH SUMMARY")
    print("=" * 80)
    print(f"Status          : {run_status}")
    print(f"Finalists       : {len(finalists)}")
    print(f"Keep for review : {len(keeps)}")
    print(f"Rejected        : {summary['n_reject']}")
    if selected_payload is not None:
        print(
            f"Selected        : {selected_payload['candidate_id']} "
            f"({selected_payload['portfolio_method']}, k={selected_payload['top_k']})"
        )
        print("Next action     : open locked holdout confirmation once only")
    else:
        print("Selected        : none")
        print("Next action     : do not open holdout; revise hypotheses or reject run")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
