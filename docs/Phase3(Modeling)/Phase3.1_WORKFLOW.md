# Phase 3.1 Workflow Execution

## 1. Architectural decisions for Sprint 1

### Config-driven experiment contract
The first sprint now uses a dedicated experiment config file at configs/experiments.yaml with a structured schema for:

- experiment metadata
- search space over lookback windows and rebalance frequencies
- strategy selection
- backtest defaults and ranking weights
- MLflow logging output

The loader in src/portfolio_ml/config/loader.py now resolves paths relative to the project root and exposes a typed ExperimentConfig object for runtime use.

### Why this structure
- Keeps path, model, and parameter choices out of Python logic.
- Makes the experiment grid reproducible and easy to extend.
- Aligns with the existing package layout under src/portfolio_ml.
- Allows the same runner to be reused for later sprints without changing the core orchestration.

### Proposed schema
```yaml
experiment:
  experiment_name: "baseline_grid"
  lookback_windows: [126, 252, 504]
  rebalance_frequencies: [5, 21, 63]
  strategies: ["EqualWeight", "RiskParity", "MinVariance"]
  transaction_cost: 0.0005
  initial_capital: 1000000.0
  paths:
    price_data_dir: "data/processed/daily_prices"
    output_dir: "outputs/phase3_experiment_grid"
  backtest:
    transaction_cost: 0.0005
    initial_capital: 1000000.0
    lookback_days: 252
    window_type: "expanding"
  ranking:
    sharpe_weight: 0.5
    mdd_weight: 0.3
    turnover_weight: 0.2
  mlflow:
    experiment_name: "portfolio_baseline_grid"
    tracking_uri: null
    log_artifacts: true
```

## 2. Implemented code structure

### New/updated modules
- src/portfolio_ml/config/loader.py
  - Added ExperimentConfig, ExperimentPathsConfig, ExperimentBacktestConfig, ExperimentRankingConfig, and ExperimentMLflowConfig.
  - Added load_experiment_config().

- src/portfolio_ml/experiments/grid_runner.py
  - Added a config-driven experiment grid runner.
  - Iterates over the Cartesian product of lookback window, rebalance frequency, and strategy.
  - Uses the existing WalkForwardSplitter and BenchmarkRunner.
  - Produces a consolidated summary CSV and logs each run to MLflow when available.

- configs/experiments.yaml
  - Added the first sprint config file.

- tests/test_experiment_config.py
  - Added regression coverage for the new experiment config schema.

## 3. Execution workflow

### How to run
```bash
PYTHONPATH=src python -m portfolio_ml.experiments.grid_runner
```

### Output artifacts
- outputs/phase3_experiment_grid/experiment_summary.csv
- MLflow runs under the local mlruns store when MLflow is available.

## 4. Verification evidence

### Configuration tests
Verified with:
```bash
pytest -q tests/test_experiment_config.py tests/test_config_loader.py
```
Result: 6 passed in 0.03s.

### End-to-end runner execution
Verified with:
```bash
PYTHONPATH=src python -m portfolio_ml.experiments.grid_runner
```
Result: the runner completed successfully and wrote the consolidated summary file to outputs/phase3_experiment_grid/experiment_summary.csv.

## 5. Observed experiment results

The grid runner produced 27 combinations across:
- lookback windows: 126, 252, 504
- rebalance frequencies: 5, 21, 63
- strategies: EqualWeight, RiskParity, MinVariance

### Top-ranked configurations by composite score
1. EqualWeight, lookback 252, rebalance 5
2. EqualWeight, lookback 252, rebalance 21
3. EqualWeight, lookback 252, rebalance 63
4. RiskParity, lookback 252, rebalance 5
5. MinVariance, lookback 252, rebalance 5

### Notes from the current run
- The highest-scoring settings in this run were all based on a 252-day lookback window.
- EqualWeight consistently ranked highly under the current scoring function.
- The current implementation uses a simple composite score based on Sharpe, drawdown, and turnover and can be extended in later sprints as more features and regime-aware logic are introduced.

## 6. Sprint 2 update

### Feature expansion implemented
Sprint 2 now adds a feature-store-friendly expansion to the asset-level feature builder in src/portfolio_ml/transformations/build_features.py:

- Macro features:
  - macro_vix
  - macro_ten_year_yield
  - macro_dxy
- Cross-asset / relational features:
  - rolling_corr_spy_21d
  - spy_relative_momentum_21d

These features are computed using only historical data and are designed to remain compatible with later modeling steps.

### Leakage validation
A lightweight leakage validator is now available from src/portfolio_ml/features/leakage.py and exposed via src/portfolio_ml/features/__init__.py. It provides a structured report that confirms the expanded feature set does not introduce future-data leakage in the current Sprint 2 implementation.

### Verification evidence
Verified with:
```bash
pytest -q tests/test_feature_generation.py
```
Result: 10 passed in 0.42s.

### Sprint 3 update

Sprint 3 now adds a lightweight ML ranking layer built on a gradient-boosting regressor and a rank-based evaluation utility:

- GradientBoostRankModel in src/portfolio_ml/modeling/ml_models/ranking.py
- RankEvaluator in src/portfolio_ml/modeling/rank_evaluator.py
- Exposed through src/portfolio_ml/modeling/__init__.py and src/portfolio_ml/modeling/ml_models/__init__.py
- Regression tests added in tests/test_ml_ranking.py

These components provide a simple benchmark for comparing ML-based ranking predictions against heuristic baselines in a reproducible way.

### Verification evidence
Verified with:
```bash
pytest -q tests/test_ml_ranking.py
```
Result: 2 passed in 1.15s.

### Sprint 4 update

Sprint 4 now adds a lightweight regime layer:

- RegimeLabeler in src/portfolio_ml/modeling/regimes.py
- RegimeAwareStrategy in src/portfolio_ml/modeling/regimes.py
- Exposed through src/portfolio_ml/modeling/__init__.py
- Regression tests added in tests/test_regime_strategy.py

The regime wrapper uses heuristics to select among the existing baseline strategies, providing a simple starting point for regime-aware portfolio allocation.

### Verification evidence
Verified with:
```bash
pytest -q tests/test_regime_strategy.py
```
Result: 2 passed in 1.09s.

## Sprint 1–4 unified workflow

The project now has a single entry point for the full Phase 3.1 workflow.

### What is included
- systematic baseline grid experiments
- regime-aware strategy evaluation
- a lightweight ML ranking experiment
- consolidated ranking output in CSV form

### Entry points
- Python module: src/portfolio_ml/experiments/unified_runner.py
- CLI script: scripts/run_unified_experiments.py

### How to run
```bash
PYTHONPATH=src python scripts/run_unified_experiments.py
```

### Verification evidence
Verified with:
```bash
pytest -q tests/test_unified_runner.py
```
Result: 1 passed in 2.10s.
