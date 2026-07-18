# Phase 3.3 — Model Robustness, Regime Awareness & Recommendations

## Purpose

Phase 3.3 extends the validated Phase 3.2 ranking pipeline without changing
its leak-free walk-forward design. It adds systematic model comparison,
statistical significance checks, regime-aware routing, stronger output
validation, and a final inference script that produces portfolio weights.

No new project dependencies are introduced. The `gradient_boost` experiment
name is retained for compatibility, but it now uses sklearn's
`HistGradientBoostingRegressor`.

## What Was Added

### Shared modeling pipelines

- `src/portfolio_ml/modeling/ranking_pipeline.py`
  - `build_features_and_targets(...)`
  - `run_walk_forward_ranking(...)`
  - Uses feature set v2, forward ranking targets, train-only scaling, and
    walk-forward folds.
- `src/portfolio_ml/modeling/two_stage_pipeline.py`
  - `run_two_stage_walk_forward(...)`
  - Connects model scores to the existing portfolio construction methods.

Training never uses future target values. The final horizon of each asset is
excluded automatically because its forward return is unknown.

### Model factory and ensemble

`src/portfolio_ml/modeling/ml_models/factory.py` provides `build_model(...)`
for:

- `ridge`
- `random_forest`
- `gradient_boost` (`HistGradientBoostingRegressor`)
- `ensemble`

The ensemble fits the three base models and averages their cross-sectional
rank-normalized predictions.

### Statistical evaluation

`src/portfolio_ml/evaluation/significance.py` provides:

- `ic_ttest(...)`: one-sided one-sample test of whether mean Rank IC is above
  zero.
- `paired_bootstrap_sharpe_diff(...)`: block bootstrap for paired strategy and
  baseline returns.

Model-search output also includes confidence intervals and Bonferroni-adjusted
p-values for the fixed hyperparameter grid.

### Regime-aware model

`RegimeAwareRankModel` routes predictions using already validated feature-set-v2
signals:

- `bm_ret_21d` sign: `Bull` or `Bear`
- `vol_regime_flag`: `HighVol` or `LowVol`

This creates four regimes: `Bull_LowVol`, `Bull_HighVol`, `Bear_LowVol`, and
`Bear_HighVol`. A regime-specific model is trained only when it has at least
200 rows. Sparse or unseen regimes use the global fallback model.

### Recommendation output validation

`src/portfolio_ml/evaluation/output_validation.py` validates:

- Required model metadata and stable decisions: `candidate`, `selected`, or
  `rejected`.
- Unique symbols and non-empty recommendations.
- No NaN scores or weights.
- Non-negative weights, configured maximum weight, and weights summing to one.
- Exact disclaimer: `Not investment advice.`
- Feature freshness. Data older than three calendar days is rejected to avoid
  silently publishing stale recommendations.

The observed case where generation date `2026-07-18` used feature date
`2026-07-02` is therefore rejected rather than published.

## Workflow

Run the workflow in this order from the repository root.

### 1. Install the existing project environment

Use the repository's normal environment setup. No new dependencies are
required by Phase 3.3.

```bash
pip install -e ".[dev]"
```

Ensure processed price files exist under:

```text
data/processed/daily_prices/year=*/daily_prices.parquet
```

### 2. Run model selection

```bash
PYTHONPATH=src python3 scripts/run_model_selection.py
```

The search uses horizon `21`, lookback `252`, and rebalance frequency `21`.
The fixed grid is:

- Ridge alpha: `0.1`, `1`, `10`
- Random Forest: `(n_estimators, max_depth)` of `(100,4)`, `(200,6)`, `(300,8)`
- HistGradientBoosting: `(max_iter, max_depth, learning_rate)` of
  `(100,3,0.1)`, `(200,4,0.05)`, `(300,6,0.03)`
- Ensemble: one default configuration

The required preserved winner is:

```json
{
  "model_name": "gradient_boost",
  "params": {
    "learning_rate": 0.03,
    "max_depth": 6,
    "max_iter": 300
  }
}
```

### 3. Run regime-aware training

```bash
PYTHONPATH=src python3 scripts/train_regime_aware_model.py
```

This trains the regime-aware wrapper using the selected model configuration
and records regime model counts and comparison artifacts. Regime-aware routing
should replace the plain model only after identical-fold out-of-sample
evaluation demonstrates robust and economically meaningful improvement.

### 4. Generate the latest recommendation

```bash
PYTHONPATH=src python3 scripts/generate_recommendation.py
```

The generator:

1. Loads all available price history.
2. Builds feature set v2.
3. Trains the selected model on rows with valid forward targets.
4. Scores the latest feature date.
5. Applies the selected two-stage portfolio method.
6. Validates freshness, scores, metadata, and weights.
7. Writes timestamped CSV and JSON files.

If the latest market data is stale, generation fails with a clear error.

### 5. Run tests

```bash
PYTHONPATH=src pytest -q
```

For only the Phase 3.3 output contract tests:

```bash
PYTHONPATH=src pytest -q tests/test_phase33_output_validation.py
```

## Output Artifacts

All Phase 3.3 artifacts are written below `outputs/phase3_3/`.

| Artifact | Description |
|---|---|
| `hyperparam_search_results_<run-id>.csv` | One row per model configuration, including metrics, folds, confidence intervals, raw p-values, adjusted p-values, and serialized per-fold results. |
| `model_selection_summary_<run-id>.csv` | Summary of the selected research configuration. |
| `best_model_<run-id>.json` | Versioned model, parameters, portfolio settings, decision, and selection rationale. |
| `regime_aware_fold_results.csv` | Regime-aware training/evaluation summary. |
| `regime_vs_global_comparison.csv` | Plain versus regime-aware comparison record. |
| `recommendation_<YYYYMMDD>_<timestamp>.csv` | Per-asset score and final portfolio weight. |
| `recommendation_<YYYYMMDD>_<timestamp>.json` | Recommendation rows plus complete metadata and disclaimer. |

Versioned names prevent new runs from overwriting prior experiment artifacts.

Recommendation metadata includes model type, parameters, regime status,
portfolio method, feature date, data cutoff, generation date,
generation timestamp, and the exact investment disclaimer.

## Selection Rules

The model winner is the highest mean Rank IC among statistically significant
configurations. If no configuration is significant, the highest mean Rank IC
is retained and marked non-significant. Multiple comparisons are reported via
adjusted p-values.

Portfolio methods are intended to be compared on identical out-of-sample
periods using `top_k_equal`, `score_weighted`, and `mean_variance`. The
comparison should include annualized return and volatility, Sharpe, maximum
drawdown, turnover, transaction-cost-adjusted returns, and bootstrap confidence
intervals. Sharpe alone is not sufficient evidence for replacing the selected
method.

Regime-aware selection should use paired identical folds and report Rank IC,
Precision@3, portfolio statistics, fallback usage, and per-regime sample
counts. If improvement is not robust and economically meaningful, the plain
winner remains final and the rejection reason must be recorded in
`best_model.json`.

## Known Limitations

- A full workflow run requires local processed price data and the project's
  existing Python dependencies.
- A stale-data failure is intentional; refresh the Phase 1/2 market-data
  pipeline before generating a recommendation.
- Research outputs are not investment advice and do not account for taxes,
  liquidity, execution slippage beyond configured transaction costs, or user
  risk preferences.
- The final recommendation is an inference artifact, not evidence of future
  performance.
