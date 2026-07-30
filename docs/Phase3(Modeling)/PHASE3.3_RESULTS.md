# Phase 3.3 Results — Model Robustness, Regime Awareness & Recommendations

**Completion Date:** 2026-07-25 (superseded prior run: 2026-07-23)
**Objective:** Extend the validated Phase 3.2 ranking pipeline with systematic model comparison, statistical significance checks, regime-aware routing, stronger output validation, and a final inference script — while holding out a locked evaluation window to confirm whether any selected strategy actually beats EqualWeight after costs.

---

## Executive Summary

Phase 3.3 built the full research and inference infrastructure required to move beyond Phase 3.2's ranking prototype, but **the resulting strategies did not clear their own pre-declared acceptance bars**. Two research cycles were run:

1. **Initial cycle (2026-07-23):** Selected a score-weighted, regime-aware HistGradientBoosting model pre-holdout, then **failed the locked 2025-01-02 → 2026-07-22 holdout** on return, Sharpe, drawdown, and turnover versus EqualWeight.
2. **Follow-on research run (2026-07-25):** Reset governance with a new selection cutoff (`2025-12-31`) and a compact, turnover-aware experiment matrix. Every candidate beat EqualWeight gross of the turnover gate, but **all seven candidates failed the pre-declared turnover acceptance criterion**, so **no finalist was promoted** and the new future holdout (`2026-01-02` → `2026-07-22`) was **never opened**.

✅ **Shared modeling pipelines** — `ranking_pipeline.py` and `two_stage_pipeline.py` reuse the leak-free Phase 3.2 walk-forward design
✅ **Model factory + ensemble** — Ridge, Random Forest, HistGradientBoosting (`gradient_boost`), and an averaged ensemble
✅ **Statistical evaluation** — one-sided IC t-tests, Bonferroni-adjusted p-values, paired block-bootstrap Sharpe differences
✅ **Regime-aware routing** — four-regime (`Bull/Bear` × `HighVol/LowVol`) model router with a global fallback
✅ **Output validation** — recommendation schema, weight, and freshness checks that reject stale or malformed outputs
❌ **Holdout confirmation** — the selected strategy from the first cycle failed the locked holdout
❌ **Turnover-aware research run** — no candidate cleared the ≤10% average turnover gate in the second cycle

**Verdict:** Phase 3.3 is **not completed** and **not production ready**. The infrastructure (model search, regime routing, significance testing, output validation, recommendation generation) is in place and tested, but no strategy has yet demonstrated robust, cost-adjusted outperformance versus EqualWeight on unseen data. Turnover control is the binding constraint blocking promotion.

---

## 1. What Was Built

### 1.1 Shared Modeling Pipelines

- `src/portfolio_ml/modeling/ranking_pipeline.py`
  - `build_features_and_targets(...)` — feature set v2 + forward ranking targets
  - `run_walk_forward_ranking(...)` — train-only scaling, walk-forward folds
- `src/portfolio_ml/modeling/two_stage_pipeline.py`
  - `run_two_stage_walk_forward(...)` — connects model scores to the Phase 3.2 portfolio construction methods

Training never uses future target values; the final horizon of each asset is excluded automatically because its forward return is unknown.

### 1.2 Model Factory and Ensemble

`src/portfolio_ml/modeling/ml_models/factory.py` provides `build_model(...)` for `ridge`, `random_forest`, `gradient_boost` (`HistGradientBoostingRegressor`), and `ensemble` (averages cross-sectional rank-normalized predictions from the three base models). No new project dependencies were introduced; `gradient_boost` retains its Phase 3.2 name for compatibility while now using sklearn's `HistGradientBoostingRegressor`.

### 1.3 Statistical Evaluation

`src/portfolio_ml/evaluation/significance.py` provides:

- `ic_ttest(...)` — one-sided one-sample test of whether mean Rank IC is above zero
- `paired_bootstrap_sharpe_diff(...)` — block bootstrap for paired strategy vs. baseline returns

Model-search output includes confidence intervals and Bonferroni-adjusted p-values across the fixed hyperparameter grid.

### 1.4 Regime-Aware Model

`RegimeAwareRankModel` routes predictions using existing feature-set-v2 signals:

- `bm_ret_21d` sign → `Bull` or `Bear`
- `vol_regime_flag` → `HighVol` or `LowVol`

This yields four regimes (`Bull_LowVol`, `Bull_HighVol`, `Bear_LowVol`, `Bear_HighVol`). A regime-specific model trains only when it has at least 200 rows; sparse or unseen regimes fall back to the global model.

### 1.5 Recommendation Output Validation

`src/portfolio_ml/evaluation/output_validation.py` validates:

- Required model metadata and stable decisions (`candidate`, `selected`, `rejected`)
- Unique symbols and non-empty recommendations
- No NaN scores or weights; non-negative weights; weights sum to one; configured max weight respected
- Exact disclaimer: `Not investment advice.`
- Feature freshness — data older than three calendar days is rejected rather than silently published (this caught a real case: generation date `2026-07-18` used feature date `2026-07-02` and was correctly rejected)

---

## 2. Research Cycle 1 — Model Selection and Holdout (2026-07-23)

### 2.1 Model Selection

`scripts/run_model_selection.py` searched horizon `21`, lookback `252`, rebalance `21` over a fixed grid:

- Ridge alpha: `0.1`, `1`, `10`
- Random Forest `(n_estimators, max_depth)`: `(100,4)`, `(200,6)`, `(300,8)`
- HistGradientBoosting `(max_iter, max_depth, learning_rate)`: `(100,3,0.1)`, `(200,4,0.05)`, `(300,6,0.03)`
- Ensemble: one default configuration

Winner (highest mean Rank IC among statistically significant configurations):

```json
{
  "model_name": "gradient_boost",
  "params": {"learning_rate": 0.03, "max_depth": 6, "max_iter": 300}
}
```

Regime-aware training (`scripts/train_regime_aware_model.py`) was directionally better than the plain model on observed holdout metrics but lacked paired bootstrap confirmation, so it was not accepted as a replacement on its own.

### 2.2 Holdout Confirmation — Failed

The locked holdout covered `2025-01-02` through `2026-07-22` (388 trading days). After 5 bps transaction costs:

| Metric | Selected (score-weighted) | EqualWeight |
|---|---:|---:|
| Annualized return | 8.36% | 22.36% |
| Sharpe | 0.60 | 1.70 |
| Max drawdown | −15.96% | −12.83% |
| Avg turnover | 19.51% | 0.26% |

The selected strategy failed on return, Sharpe, drawdown, and turnover.

| Status field | Value |
|---|---|
| `selected_pre_holdout` | `true` |
| `holdout_status` | `failed` |
| `production_ready` | `false` |
| `final_decision` | `rejected_after_holdout` |

Artifacts: `outputs/phase3_3/holdout_confirmation_20260723.json`, `outputs/phase3_3/phase3_3_status_20260723.json`. **This holdout is locked evaluation data and must not be retuned against.**

---

## 3. Research Cycle 2 — Turnover-Aware Reset (2026-07-25)

### 3.1 Root-Cause Diagnosis

Using the failed holdout only as structural (not tunable) evidence, five likely causes were identified:

1. Excessive turnover — score-weighted membership/weights churned far faster than EqualWeight, and costs converted a fragile gross signal into weak net performance.
2. Weak return persistence — mean Rank IC was positive pre-holdout but negative on the holdout for the plain model.
3. Poor net alpha after costs — the strategy lost against a cheap diversified benchmark under identical cost assumptions.
4. Regime-aware improvements lacked paired bootstrap confirmation.
5. Wrong objective — prior selection optimized Rank IC/Sharpe among active methods rather than explicit net outperformance vs. EqualWeight after costs.

### 3.2 New Governance

| Field | Value |
|---|---|
| Run ID | `phase3_3_research_20260725` |
| Selection cutoff | `2025-12-31` |
| Future holdout (locked) | `2026-01-02` → `2026-07-22` (138 trading days) |
| Prior failed holdout | `2025-01-02` → `2026-07-22` — locked, not used for retuning |
| Frozen model | HistGB / `gradient_boost` (`max_iter=300`, `max_depth=6`, `lr=0.03`) |

Supporting code: `scripts/run_phase3_3_research.py`, `src/portfolio_ml/evaluation/acceptance.py`, `src/portfolio_ml/modeling/phase3_3_evaluation.py`.

### 3.3 Compact, Turnover-Aware Candidate Matrix

Model hyperparameters stayed frozen; only portfolio/turnover controls varied across candidates A–G (top-k equal weight, score-weighted with sticky blends, volatility scaling, drawdown-aware exposure reduction, and regime-aware routing gated on regime consistency).

### 3.4 Acceptance Criteria (declared before opening the holdout)

A finalist requires **all** of:

1. Net annualized return exceeds EqualWeight after costs
2. Net Sharpe exceeds EqualWeight
3. Max drawdown no worse than EqualWeight by more than 2 percentage points
4. Average turnover ≤ 10%
5. Beats EqualWeight Sharpe in ≥ 50% of calendar-year validation windows
6. Paired bootstrap observed Sharpe difference > 0
7. Not a trivial full-universe equal-weight clone

### 3.5 Pre-Holdout Results — No Finalist

EqualWeight benchmark on the research calendar (after 5 bps): ann. return ≈ 12.4%, Sharpe ≈ 0.84, max DD ≈ −26.8%, avg turnover ≈ 1.2%.

| Candidate | Ann. return | Sharpe | Max DD | Turnover | vs EW return | vs EW Sharpe | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 33.8% | 2.26 | −21.0% | 92.9% | +21.4 pp | +1.42 | reject (turnover) |
| B | 25.8% | 1.70 | −23.9% | 54.5% | +13.4 pp | +0.86 | reject (turnover) |
| C | 19.2% | 1.29 | −25.3% | 23.5% | +6.8 pp | +0.44 | reject (turnover) |
| D | 19.4% | 1.28 | −25.8% | 27.6% | +7.0 pp | +0.43 | reject (turnover) |
| E | 25.4% | 1.78 | −22.9% | 59.9% | +13.0 pp | +0.93 | reject (turnover) |
| F | 15.0% | 0.94 | −28.9% | 13.4% | +2.6 pp | +0.10 | reject (turnover, drawdown) |
| G | 19.2% | 1.27 | −25.8% | 27.3% | +6.8 pp | +0.43 | reject (turnover) |

Every candidate beat EqualWeight on gross return and Sharpe, but **all seven failed the ≤10% average-turnover gate** (closest: candidate F at 13.4% turnover, which also failed the drawdown-tolerance gate). Bootstrap Sharpe-difference p-values clustered near 0.5, so pre-holdout edges were not statistically confirmed even before the turnover failure. **No finalist was promoted; the 2026 future holdout was never opened.**

### 3.6 Recommended Next Steps (not yet executed)

If a follow-on dated research run is approved, try (in order, with a new run ID and no retuning against locked holdouts):

1. Membership hysteresis / trade buffer on top-k equal weight
2. Sticky fraction 0.8–0.9 plus 42-day rebalance
3. Explicit turnover penalty in candidate ranking with EqualWeight net Sharpe as the primary score

If those still cannot clear the turnover gate while preserving a meaningful EqualWeight edge, Phase 3.3 should be formally rejected rather than forced to completion.

---

## 4. Workflow

Run in this order from the repository root. No new dependencies are required.

```bash
pip install -e ".[dev]"
```

Ensure processed price files exist under `data/processed/daily_prices/year=*/daily_prices.parquet` with at least `date, symbol, adj_close` columns.

### 4.1 Model selection

```bash
PYTHONPATH=src python3 scripts/run_model_selection.py
```

Loads processed prices, builds feature set v2, excludes labels crossing the selection cutoff, evaluates every model configuration on identical walk-forward folds, selects the best model, compares the three portfolio methods, evaluates risk controls, fits through the cutoff, and evaluates the untouched holdout.

### 4.2 Regime-aware comparison

```bash
PYTHONPATH=src python3 scripts/train_regime_aware_model.py
```

Run after model selection; compares plain vs. regime-aware predictions on the same holdout dates using the latest versioned model configuration.

### 4.3 Turnover-aware research run

```bash
PYTHONPATH=src:. python3 scripts/run_phase3_3_research.py
```

Executes the governed, turnover-aware candidate matrix against a declared acceptance policy without opening the future holdout unless a finalist is promoted.

### 4.4 Generate a recommendation

```bash
PYTHONPATH=src python3 scripts/generate_recommendation.py
```

Loads price history, builds feature set v2, trains the selected model on rows with valid forward targets, scores the latest feature date, applies the selected two-stage portfolio method, validates freshness/scores/metadata/weights, and writes timestamped CSV and JSON files. Generation fails with a clear error if the latest market data is stale (>3 calendar days old).

**Do not run `generate_recommendation.py` as a production recommendation until a finalist passes a future holdout.**

### 4.5 Run tests

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src pytest -q tests/test_phase33_output_validation.py
```

---

## 5. Output Artifacts

All Phase 3.3 artifacts are written below `outputs/phase3_3/`. Versioned names prevent new runs from overwriting prior experiment artifacts.

| Artifact | Description |
|---|---|
| `hyperparam_search_results_<run-id>.csv` | One row per model configuration: metrics, folds, confidence intervals, raw/adjusted p-values, serialized per-fold results |
| `model_selection_summary_<run-id>.csv` | Summary of the selected research configuration |
| `best_model_<run-id>.json` | Versioned model, parameters, portfolio settings, decision, and selection rationale |
| `portfolio_method_results_<run-id>.csv` | Comparison of `top_k_equal`, `score_weighted`, `mean_variance` |
| `risk_control_results_<run-id>.csv` | Top-K and concentration-control results |
| `holdout_results_<run-id>.csv` / `holdout_confirmation_20260723.json` | Final untouched holdout metrics and confirmation decision |
| `regime_aware_fold_results_<run-id>.csv` | Regime decision and evaluation metadata |
| `regime_vs_global_comparison_<run-id>.csv` | Plain versus regime-aware comparison record |
| `phase3_3_status_<date>.json` | Governance status snapshot (`selected_pre_holdout`, `holdout_status`, `production_ready`, `final_decision`) |
| `research_run_20260725.json`, `research_summary_phase3_3_research_20260725.json` | Turnover-aware research run metadata and summary |
| `research_candidate_results_phase3_3_research_20260725.csv` | Per-candidate pre-holdout metrics and decisions |
| `research_window_stability_phase3_3_research_20260725.csv` | Calendar-year window stability for each candidate |
| `recommendation_<YYYYMMDD>_<timestamp>.csv` / `.json` | Per-asset score, final portfolio weight, and full metadata including the disclaimer `Not investment advice.` |

---

## 6. Selection and Governance Rules

- The model winner is the highest mean Rank IC among statistically significant configurations; if none is significant, the highest mean Rank IC is retained and marked non-significant. Multiple comparisons are reported via adjusted p-values.
- Portfolio methods (`top_k_equal`, `score_weighted`, `mean_variance`) must be compared on identical out-of-sample periods using annualized return/volatility, Sharpe, max drawdown, turnover, transaction-cost-adjusted returns, and bootstrap confidence intervals. **Sharpe alone is not sufficient evidence** for replacing the selected method.
- Regime-aware selection must use paired identical folds and report Rank IC, Precision@3, portfolio statistics, fallback usage, and per-regime sample counts. If improvement is not robust and economically meaningful, the plain winner remains final and the rejection reason is recorded in `best_model.json`.
- Locked holdouts (both the initial `2025-01-02`–`2026-07-22` window and the follow-on `2026-01-02`–`2026-07-22` window) are evaluation data only. **Do not retune against a locked holdout** — start a new dated research run with a new cutoff instead.

---

## 7. Known Limitations

- A full workflow run requires local processed price data and the project's existing Python dependencies.
- A stale-data failure in recommendation generation is intentional; refresh the Phase 1/2 market-data pipeline before generating a recommendation.
- Research outputs are not investment advice and do not account for taxes, liquidity, execution slippage beyond configured transaction costs, or user risk preferences.
- Any generated recommendation is an inference artifact, not evidence of future performance.
- Turnover control remains the binding constraint across both research cycles; no candidate has yet combined acceptable turnover with a statistically confirmed edge over EqualWeight.

---

## 8. Conclusion

**Can ML add value over EqualWeight after realistic costs?** Not yet demonstrated. Phase 3.3 delivered the full research infrastructure — model factory, significance testing, regime-aware routing, output validation, and a governed research-run process — and used it honestly to reject two candidate strategies rather than force a weak result into production:

- Cycle 1's score-weighted regime-aware HistGB strategy **failed its locked holdout** on every headline metric versus EqualWeight.
- Cycle 2's turnover-aware matrix showed every candidate beating EqualWeight on gross return/Sharpe, but **none cleared the pre-declared turnover gate**, so no finalist was promoted and the new holdout stayed locked.

**Phase 3.3 is not completed and not production ready.** The next dated research run should target turnover reduction specifically (membership hysteresis, higher sticky fractions, longer rebalance cycles, or an explicit turnover penalty in candidate ranking) before any strategy is considered for holdout confirmation again.

---

**End of Phase 3.3 Results Report**
++++++
