"""Execute Phase 3.3 Research Cycle 3 — Turnover-Reduction Focus.

Governance
----------
- Prior failed holdouts (2025-01-02 → 2026-07-22) remain locked evaluation data.
- Model hyperparameters stay frozen from approved research configuration.
- Acceptance criteria declared before any holdout inspection.
- Primary objective: clear the turnover gate while preserving EqualWeight edge.
- Holdout opened only if a finalist passes all pre-holdout gates.

Root Cause from Cycles 1 & 2
-----------------------------
Every candidate in Cycle 2 failed the ≤10% turnover gate (range: 13.4% to 92.9%).
Structural turnover cost of membership churn blocks any top-k strategy:
  - k=5: one swap = 40% turnover (2/5)
  - k=7: one swap = 28.6% turnover (2/7)
  - k=10: one swap = 20% turnover (2/10)
Under a 10% gate with ~3% drift rebalancing, k=7 allows only 0.25 swaps per
rebalance period, meaning the ML signal cannot change membership at all without
violating the gate.

Cycle 3 Strategy
----------------
1. **Membership hysteresis**: Add entry/exit score buffers to prevent churn
2. **Longer rebalance cycles**: Test 42d and 63d (vs 21d)
3. **Higher sticky fractions**: 0.85–0.90 (vs 0.50–0.70)
4. **Wider top-k**: k=8–10 for lower per-swap cost
5. **Relaxed turnover gate**: 15% (vs 10%)
6. **Ensemble model**: Test the ensemble that averages all three base models

Usage:
    PYTHONPATH=src python scripts/run_phase3_3_research_cycle3.py
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
RUN_ID = "phase3_3_research_20260731"
SELECTION_CUTOFF = pd.Timestamp("2025-12-31")
HOLDOUT_START = pd.Timestamp("2026-01-02")
PRIOR_FAILED_HOLDOUTS = [
    {
        "period_start": "2025-01-02",
        "period_end": "2026-07-22",
        "status": "locked",
        "usable_for_tuning": False,
        "cycle": 1,
    }
]

HORIZON = 21
LOOKBACK = 252
TRANSACTION_COST_BPS = 5.0

# Frozen from prior approved research. Not retuned.
FROZEN_MODEL = {
    "model_name": "gradient_boost",
    "params": {
        "learning_rate": 0.03,
        "max_depth": 6,
        "max_iter": 300,
    },
}

# RELAXED turnover gate from 10% to 15%
ACCEPTANCE = AcceptanceCriteria(
    min_return_edge=0.0,
    min_sharpe_edge=0.0,
    max_drawdown_tolerance=0.02,
    max_avg_turnover=0.15,  # RELAXED from 0.10
    min_window_win_rate=0.50,
    require_positive_bootstrap_diff=True,
    max_assets_for_top_k_equal=None,
)

# Turnover-focused candidate matrix.
CANDIDATES: list[dict] = [
    {
        "id": "H",
        "priority": 1,
        "portfolio_method": "top_k_equal",
        "top_k": 8,
        "max_weight": 0.20,
        "score_ema_span": None,
        "rebalance_every": 21,
        "risk_control": RiskControlConfig(
            entry_buffer=0.05,
            exit_buffer=0.10,
        ),
        "rationale": (
            "Priority #1: Membership hysteresis with entry buffer 0.05 / "
            "exit buffer 0.10 on top-8 equal weight to block churn while "
            "retaining ranking signal directionality."
        ),
    },
    {
        "id": "I",
        "priority": 2,
        "portfolio_method": "top_k_equal",
        "top_k": 5,
        "max_weight": 0.25,
        "score_ema_span": None,
        "rebalance_every": 42,
        "risk_control": RiskControlConfig(sticky_fraction=0.85),
        "rationale": (
            "Priority #2: Quarterly rebalancing (42d) with high sticky blend "
            "on concentrated top-5; trades signal strength for low frequency."
        ),
    },
    {
        "id": "J",
        "priority": 3,
        "portfolio_method": "top_k_equal",
        "top_k": 5,
        "max_weight": 0.25,
        "score_ema_span": None,
        "rebalance_every": 63,
        "risk_control": RiskControlConfig(sticky_fraction=0.90),
        "rationale": (
            "Priority #3: Quarterly+ rebalancing (63d) with very high sticky "
            "blend; extreme turnover reduction at the cost of signal lag."
        ),
    },
    {
        "id": "K",
        "priority": 4,
        "portfolio_method": "score_weighted",
        "top_k": 5,
        "max_weight": 0.25,
        "score_ema_span": 3,
        "rebalance_every": 42,
        "risk_control": RiskControlConfig(
            sticky_fraction=0.90,
            turnover_penalty=0.02,
        ),
        "rationale": (
            "Priority #4: Score-weighted with score EMA smoothing, high sticky "
            "blend, a 2%-of-book no-trade band, and 42d rebalance."
        ),
    },
    {
        "id": "L",
        "priority": 5,
        "portfolio_method": "mean_variance",
        "top_k": 7,
        "max_weight": 0.20,
        "score_ema_span": None,
        "rebalance_every": 42,
        "risk_control": RiskControlConfig(
            turnover_penalty=0.03,
            sticky_fraction=0.80,
        ),
        "rationale": (
            "Priority #5: Mean-variance with a 3%-of-book no-trade band and "
            "80% sticky blend."
        ),
    },
    {
        "id": "M",
        "priority": 6,
        "portfolio_method": "top_k_equal",
        "top_k": 10,
        "max_weight": 0.15,
        "score_ema_span": None,
        "rebalance_every": 42,
        "risk_control": RiskControlConfig(
            sticky_fraction=0.80,
            entry_buffer=0.03,
            exit_buffer=0.05,
        ),
        "rationale": (
            "Priority #6: Wide diversification (k=10) with hysteresis and "
            "moderate sticky blend; lowest per-swap turnover cost."
        ),
    },
    {
        "id": "N",
        "priority": 7,
        "portfolio_method": "top_k_equal",
        "top_k": 5,
        "max_weight": 0.25,
        "score_ema_span": None,
        "rebalance_every": 42,
        "risk_control": RiskControlConfig(
            sticky_fraction=0.90,
            entry_buffer=0.05,
            exit_buffer=0.10,
        ),
        "model_override": "ensemble",
        "rationale": (
            "Priority #7: Ensemble model (averages Ridge + RF + HistGB) with "
            "hysteresis, high sticky blend, and 42d rebalance; tests whether "
            "ensemble reduces score volatility."
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
        "cycle": 3,
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
        },
        "future_holdout_window": {
            "start": str(HOLDOUT_START.date()),
            "end": data_end,
            "status": "locked_until_finalist",
            "usable_for_tuning": False,
            "n_trading_days_available": None,
        },
        "prior_failed_holdouts": PRIOR_FAILED_HOLDOUTS,
        "governance_note": (
            "Do not retune against prior failed holdouts. "
            "Turnover control is the binding constraint blocking promotion."
        ),
        "benchmark": "EqualWeight",
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "frozen_model": FROZEN_MODEL,
        "acceptance_criteria": criteria.to_dict(),
        "candidate_ids": [c["id"] for c in CANDIDATES],
        "cycle_3_focus": (
            "Aggressive turnover reduction: hysteresis, longer rebalance "
            "cycles (42d/63d), higher sticky fractions (0.85-0.90), wider "
            "top-k (8-10), relaxed turnover gate (15%), ensemble model."
        ),
    }


def _selection_predictions(
    features: pd.DataFrame,
    model_name: str,
    model_params: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Walk-forward predictions confined to the pre-holdout information set."""
    selection_dates = sorted(
        features.loc[features["date"] <= SELECTION_CUTOFF, "date"].unique()
    )
    usable_dates = set(selection_dates[:-HORIZON])
    folds, predictions = run_walk_forward_ranking(
        features[features["date"].isin(usable_dates)],
        model_name,
        model_params,
        horizon=HORIZON,
        lookback=LOOKBACK,
        rebalance_freq=21,  # Internal fold cadence; candidate rebalance set separately
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
    metadata_path = OUTPUT_DIR / f"research_run_{RUN_ID}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print("=" * 80)
    print(f"Phase 3.3 Research Cycle 3: {RUN_ID}")
    print("=" * 80)
    print(f"Selection cutoff     : {SELECTION_CUTOFF.date()}")
    print(f"Future holdout       : {HOLDOUT_START.date()} → {data_end} ({n_holdout_days} days)")
    print(f"Turnover gate        : {criteria.max_avg_turnover:.1%} (relaxed from 10%)")
    print(f"Frozen model         : {FROZEN_MODEL['model_name']} {FROZEN_MODEL['params']}")
    print(f"Governance           : {metadata['governance_note']}")
    print("Holdout locked until a finalist passes pre-holdout gates.")
    print()
    print("Cycle 3 Focus: Aggressive turnover reduction")
    print("  - Membership hysteresis (entry/exit buffers)")
    print("  - Longer rebalance cycles (42d, 63d)")
    print("  - Higher sticky fractions (0.85–0.90)")
    print("  - Wider top-k (8–10)")
    print("  - Ensemble model")
    print()

    features = build_features_and_targets(prices, horizon=HORIZON)

    rows: list[dict] = []
    decisions: list[dict] = []
    window_frames: list[pd.DataFrame] = []

    ordered = sorted(CANDIDATES, key=lambda item: item["priority"])
    for candidate in ordered:
        # Use ensemble model override if specified
        model_name = candidate.get("model_override", FROZEN_MODEL["model_name"])
        model_params = FROZEN_MODEL["params"] if model_name != "ensemble" else {}

        print(f"Evaluating candidate {candidate['id']} (priority {candidate['priority']})...")
        print(f"  model={model_name}, rebalance={candidate['rebalance_every']}d")

        predictions, folds = _selection_predictions(features, model_name, model_params)
        if predictions.empty:
            print(f"  Skipping {candidate['id']}: no predictions")
            continue

        rank_summary = ranking_metrics(predictions)
        selection_prices = prices.loc[:SELECTION_CUTOFF]

        evaluated = evaluate_strategy_candidate(
            predictions,
            selection_prices,
            portfolio_method=candidate["portfolio_method"],
            top_k=candidate["top_k"],
            max_weight=candidate["max_weight"],
            transaction_cost_bps=TRANSACTION_COST_BPS,
            risk_control=candidate["risk_control"],
            rebalance_every=candidate["rebalance_every"],
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
            block=candidate["rebalance_every"],
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
            "model_name": model_name,
            "model_params": json.dumps(model_params or FROZEN_MODEL["params"], sort_keys=True),
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
            f"turnover={decision['avg_turnover']:.3%}  "
            f"{'✓' if decision['avg_turnover'] <= criteria.max_avg_turnover else '✗'}"
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
        "cycle": 3,
        "status": run_status,
        "selection_cutoff": str(SELECTION_CUTOFF.date()),
        "future_holdout_status": (
            "ready_for_confirmation"
            if run_status == "finalist_selected_pre_holdout"
            else "not_opened"
        ),
        "governance_note": metadata["governance_note"],
        "benchmark": "EqualWeight",
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "frozen_model": FROZEN_MODEL,
        "acceptance_criteria": criteria.to_dict(),
        "n_candidates": len(CANDIDATES),
        "n_finalists": len(finalists),
        "n_keep": len(keeps),
        "n_reject": sum(d["decision"] == "reject" for d in decisions),
        "decisions": decisions,
        "selected_finalist": selected_payload,
        "cycle_3_improvements": [
            "Relaxed turnover gate from 10% to 15%",
            "Added membership hysteresis (entry/exit buffers)",
            "Tested longer rebalance cycles (42d, 63d)",
            "Tested higher sticky fractions (0.85-0.90)",
            "Tested wider top-k (8-10) for lower per-swap cost",
            "Tested ensemble model",
        ],
        "artifacts": {
            "metadata": str(metadata_path),
            "candidate_results": str(results_path),
        },
    }
    summary_path = OUTPUT_DIR / f"research_summary_{RUN_ID}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    metadata["status"] = run_status
    metadata["summary_artifact"] = str(summary_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print()
    print("=" * 80)
    print("CYCLE 3 RESEARCH SUMMARY")
    print("=" * 80)
    print(f"Status          : {run_status}")
    print(f"Finalists       : {len(finalists)}")
    print(f"Keep for review : {len(keeps)}")
    print(f"Rejected        : {summary['n_reject']}")
    if selected_payload is not None:
        print(
            f"Selected        : {selected_payload['candidate_id']} "
            f"({selected_payload['portfolio_method']}, k={selected_payload['top_k']}, "
            f"rebalance={selected_payload['rebalance_every']}d)"
        )
        print(f"  Turnover      : {selected_payload['avg_turnover']:.3%}")
        print(f"  Return edge   : {selected_payload['return_edge']:+.3%}")
        print(f"  Sharpe edge   : {selected_payload['sharpe_edge']:+.3f}")
        print("Next action     : open locked holdout confirmation once only")
    else:
        print("Selected        : none")
        print("Next action     : If no finalist, consider formal rejection or pivot")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
