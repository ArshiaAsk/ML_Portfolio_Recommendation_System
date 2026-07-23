# Phase 3.3 Research Run — 2025-12-31 Selection Cutoff

## Governance

Run ID: `phase3_3_research_20260723`

Selection cutoff: `2025-12-31`

The previous holdout (`2025-01-02`–`2026-07-22`) is locked evaluation data. It
must not be used to choose models, parameters, portfolio rules, risk controls,
or regime logic in this run. This run starts a new research cycle and does not
overwrite prior artifacts.

## Prior Failure: Structural Lessons

The previous score-weighted strategy failed after costs:

- It returned `8.36%` annualized versus `22.36%` for EqualWeight.
- Its Sharpe was `0.60` versus `1.70` for EqualWeight.
- Its drawdown was `-15.96%` versus `-12.83%`.
- Its average turnover was `19.51%` versus `0.26%`.
- Regime-aware routing improved the plain model directionally, but paired
  bootstrap evidence was unavailable.

These are structural lessons, not parameter targets. They motivate lower
turnover, more diversified exposure, and explicit net-performance acceptance
criteria without tuning against the locked holdout.

## Ranked Hypotheses

| Rank | Hypothesis | Expected impact | Simplicity |
|---:|---|---|---|
| 1 | Rebalance only every 21 trading days and retain existing holdings between rebalances | High turnover reduction | High |
| 2 | Prefer top-k equal weighting over score-weighting when scores are noisy | High concentration/turnover reduction | High |
| 3 | Add a turnover penalty or sticky-holdings blend to new target weights | Medium-high net performance improvement | Medium |
| 4 | Evaluate `top_k` values 7 and 10 to reduce concentration | Medium drawdown reduction | High |
| 5 | Apply volatility-scaled positions with the same weight cap | Medium risk reduction | Medium |
| 6 | Use drawdown-aware exposure reduction rather than changing model parameters | Medium tail-risk reduction | Medium |
| 7 | Require regime consistency before enabling regime-aware routing | Medium robustness improvement | Medium |

## Narrow Experiment Matrix

The matrix is intentionally compact:

| Candidate | Model | Portfolio rule | Controls |
|---|---|---|---|
| A | Frozen selected HistGB config | top-k equal, k=7 | 21-day rebalance |
| B | Frozen selected HistGB config | top-k equal, k=10 | 21-day rebalance |
| C | Frozen selected HistGB config | score-weighted | sticky holdings, 50% blend |
| D | Frozen selected HistGB config | top-k equal, k=7 | volatility scaling |
| E | Frozen selected HistGB config | top-k equal, k=7 | sticky holdings + 15% cap |
| F | Frozen selected HistGB config | top-k equal, k=10 | drawdown exposure reduction |
| G | Frozen selected HistGB config | regime-aware routing | only if regime consistency passes |

No broad hyperparameter sweep is permitted. Model parameters remain frozen from
the approved research configuration unless a separate research decision is
recorded before evaluation.

## Pre-Holdout Evaluation Policy

All candidates use identical expanding walk-forward validation dates ending at
`2025-12-31`. For every candidate, record:

- Net annualized return after 5 bps costs
- Net Sharpe
- Maximum drawdown
- Average and annualized turnover
- Return and Sharpe advantage versus EqualWeight
- Results for each validation window
- Bull/Bear and HighVol/LowVol consistency
- Paired block-bootstrap confidence intervals

Candidates are rejected if they improve statistical metrics but lose money
versus EqualWeight after costs, have materially worse drawdown, or require
unrealistic turnover.

## Finalist Selection Rules

A candidate becomes a finalist only if it satisfies all of the following on
pre-holdout validation:

1. Net annualized return is above EqualWeight.
2. Net Sharpe is above EqualWeight and the lower paired confidence bound is
   not materially negative.
3. Maximum drawdown is no worse than EqualWeight within a documented tolerance.
4. Turnover is economically plausible and materially below the prior strategy.
5. Improvement is present in a majority of validation windows.
6. No single regime accounts for all of the improvement.
7. Paired bootstrap results are persisted.

If no candidate passes, the run ends as `no_finalist`; no holdout is opened.

## Future Holdout Package

For each finalist, freeze:

- Model name and parameters
- Portfolio method and all controls
- Feature version
- Cutoff date
- Benchmark and transaction costs
- Expected artifact hashes/configuration
- Acceptance rubric

The future holdout starts after the new selection cutoff and is evaluated once.
It must answer whether the finalist beats EqualWeight after costs, improves
Sharpe meaningfully, controls drawdown and turnover, and remains consistent
across regimes. The holdout cannot be used for tuning.

## Execution

Create the governed run metadata first, then execute the narrow pre-holdout
selection workflow:

```bash
PYTHONPATH=src python3 scripts/run_model_selection.py \
  --selection-cutoff 2025-12-31 \
  --run-id phase3_3_research_20260723
```

Then inspect the versioned artifacts in `outputs/phase3_3/`. Only finalists
that satisfy the rubric may receive a holdout package. Do not run
`generate_recommendation.py` as a production recommendation until a future
holdout passes.

## Metadata Draft

The machine-readable metadata is stored at:

```text
outputs/phase3_3/research_run_20260723.json
```
