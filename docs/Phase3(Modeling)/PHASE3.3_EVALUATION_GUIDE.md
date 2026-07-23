# Phase 3.3 Evaluation Guide

This document explains the corrected Phase 3.3 evaluation workflow and how to
run it safely.

## Objective

Phase 3.3 evaluates ranking models, portfolio construction methods, regime-aware
routing, and risk controls without using the final holdout period for selection.
The workflow preserves the validated Phase 3.2 feature and walk-forward design.

## Evaluation Timeline

The workflow uses three information periods:

| Period | Purpose |
|---|---|
| Data before `2024-12-31` | Model and portfolio selection |
| `2025-01-01` onward | Untouched final holdout |
| Final latest data | Recommendation generation after choices are frozen |

Because a 21-day target uses future prices, the final 21 pre-cutoff trading
dates are excluded from model-selection training. This prevents labels from
crossing the `2024-12-31` boundary.

## Components Implemented

### `phase33_evaluation.py`

Located at:

```text
src/portfolio_ml/modeling/phase33_evaluation.py
```

Important functions:

- `frozen_holdout_predictions(...)`
  - Fits a model once using data available before the cutoff.
  - Scores only dates after the cutoff.
  - Supports both plain and regime-aware models.
- `ranking_metrics(...)`
  - Computes Rank IC, Rank IC dispersion, and Precision@3.
- `evaluate_portfolio_methods(...)`
  - Evaluates `top_k_equal`, `score_weighted`, and `mean_variance` on the same dates.
- `RiskControlConfig`
  - Supports volatility scaling, sticky holdings, turnover smoothing, and
    drawdown-aware exposure reduction.

### Model selection

`scripts/run_model_selection.py` evaluates:

- Ridge with alpha values `0.1`, `1`, and `10`
- Random Forest configurations
- HistGradientBoosting configurations under the public name `gradient_boost`
- The ensemble model

Each candidate uses identical walk-forward settings:

```text
horizon       = 21 trading days
lookback      = 252 trading days
rebalance     = 21 trading days
selection end = 2024-12-31
```

The winner is selected from computed results using Rank IC and adjusted
significance results. No model is selected from the holdout period.

### Regime-aware evaluation

`scripts/train_regime_aware_model.py` compares the plain and regime-aware
models on the same frozen holdout dates.

Regimes are derived from feature-set-v2 columns:

- `bm_ret_21d >= 0`: Bull, otherwise Bear
- `vol_regime_flag > 0`: HighVol, otherwise LowVol

The four possible regimes are:

```text
Bull_LowVol
Bull_HighVol
Bear_LowVol
Bear_HighVol
```

Regime-aware routing is accepted only when it improves the required ranking and
portfolio criteria. Otherwise, the plain model remains selected and the reason
is persisted in the versioned model JSON.

## Installation and Prerequisites

Use the existing project environment. Phase 3.3 does not add dependencies.

```bash
pip install -e ".[dev]"
```

Processed price data must exist at:

```text
data/processed/daily_prices/year=*/daily_prices.parquet
```

The files must contain at least:

```text
date, symbol, adj_close
```

## Execution Workflow

Run all commands from the repository root.

### 1. Run model and portfolio selection

```bash
PYTHONPATH=src python3 scripts/run_model_selection.py
```

This command:

1. Loads all processed prices.
2. Builds feature set v2 and ranking targets.
3. Excludes labels that cross the selection cutoff.
4. Runs every model configuration on identical walk-forward folds.
5. Selects the best computed model.
6. Compares all three portfolio methods.
7. Evaluates top-K and maximum-weight risk controls.
8. Fits the selected model through `2024-12-31`.
9. Evaluates the untouched `2025+` holdout.

### 2. Run regime-aware comparison

```bash
PYTHONPATH=src python3 scripts/train_regime_aware_model.py
```

Run this after model selection. It loads the latest versioned model
configuration and compares plain versus regime-aware predictions using the
same holdout data.

### 3. Generate a recommendation

```bash
PYTHONPATH=src python3 scripts/generate_recommendation.py
```

The recommendation generator uses the latest versioned model configuration and
performs final inference only. It does not run research selection again.

The generator validates:

- No missing scores or weights
- Unique symbols
- Non-negative weights
- Maximum individual weight
- Weights sum to one
- Valid model and portfolio metadata
- Fresh feature data
- Exact disclaimer: `Not investment advice.`

If the latest feature date is more than three calendar days old, generation
fails rather than producing a stale recommendation.

## Output Files

Outputs are written to:

```text
outputs/phase3_3/
```

The workflow creates timestamped artifacts:

| File | Contents |
|---|---|
| `hyperparam_search_results_<run-id>.csv` | Candidate model metrics and per-fold results |
| `portfolio_method_results_<run-id>.csv` | Three portfolio-method comparison |
| `risk_control_results_<run-id>.csv` | Top-K and concentration-control results |
| `holdout_results_<run-id>.csv` | Final untouched holdout metrics |
| `best_model_<run-id>.json` | Frozen model and portfolio choices |
| `regime_vs_global_comparison_<run-id>.csv` | Plain versus regime-aware comparison |
| `regime_aware_fold_results_<run-id>.csv` | Regime decision and evaluation metadata |
| `recommendation_<date>_<timestamp>.csv` | Asset scores and final weights |
| `recommendation_<date>_<timestamp>.json` | Recommendation data and metadata |

Existing artifacts are not overwritten.

## How to Interpret Results

### Model results

Review:

- `mean_rank_ic`: average cross-sectional ranking quality
- `std_rank_ic`: stability across folds
- `precision_at_3`: top-three selection quality
- `n_folds`: number of completed walk-forward folds
- `p_value`: raw one-sided IC test p-value
- `adjusted_p_value`: multiple-comparison-adjusted p-value
- `fold_metrics`: date ranges, sample counts, asset counts, and missingness

### Portfolio results

Review:

- `annualised_return`
- `annualised_volatility`
- `sharpe_ratio`
- `max_drawdown`
- `avg_turnover`
- Transaction-cost-adjusted performance

Do not select a method based on Sharpe alone. Prefer a simpler method when
performance is comparable but it has lower turnover, lower concentration, and
more acceptable drawdown.

### Holdout decision

The holdout is for confirmation, not tuning. It should answer:

- Does the selected strategy outperform EqualWeight after costs?
- Is the Sharpe improvement meaningful?
- Is maximum drawdown acceptable?
- Is turnover economically realistic?
- Does regime-aware routing improve consistently?

If the holdout is weak, do not retune against it. Start a new dated research
run with a new cutoff and document the change.

## Tests

Run the complete test suite:

```bash
PYTHONPATH=src pytest -q
```

Run focused Phase 3.3 validation tests:

```bash
PYTHONPATH=src pytest -q tests/test_phase33_output_validation.py
```

Run data-contract tests:

```bash
PYTHONPATH=src pytest -q tests/test_data_validation.py
```

## Limitations

- Results depend on the quality and freshness of processed market data.
- The final holdout cannot prove future performance.
- Transaction costs are modeled proportionally and may not capture all execution
  effects.
- The strategy does not incorporate investor-specific objectives, taxes,
  liquidity constraints, or personalized risk tolerance.
- All generated recommendations are explicitly not investment advice.
