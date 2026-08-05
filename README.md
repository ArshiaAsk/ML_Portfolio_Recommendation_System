# ML Portfolio Recommendation System

## Version 1.0 — Research Complete

[![Project Status: Complete](https://img.shields.io/badge/status-v1.0%20complete-success)](https://github.com/yourusername/ML_Portfolio_Recommendation_System)
[![Engineering: Production-Grade](https://img.shields.io/badge/engineering-production--grade-blue)](https://github.com/yourusername/ML_Portfolio_Recommendation_System)
[![Research: Rigorous](https://img.shields.io/badge/research-rigorous-informational)](https://github.com/yourusername/ML_Portfolio_Recommendation_System)

**A production-grade ML research system designed to reject weak strategies before deployment.**

---

## Project Status

| Component | Status | Outcome |
|-----------|--------|---------|
| **Engineering** | ✅ **Complete** | Production-quality data pipelines, validation, testing, and backtesting infrastructure |
| **Research** | ✅ **Complete** | Hypothesis tested through walk-forward validation, locked holdout evaluation, and failure attribution analysis |
| **ML Strategy** | ❌ **Rejected** | Did not outperform EqualWeight baseline after realistic transaction costs |
| **Production Baseline** | ✅ **Selected** | **EqualWeight** — 22.36% annualized return (1.70 Sharpe) on 2025–2026 holdout |
| **Repository** | 📦 **Stable** | Version 1.0 complete. Future research (if any) will be in new version branches |

---

## What This Project Is

This is an **end-to-end ML research system** for ETF portfolio optimization—not just a model, but a complete production-oriented platform that:

- Ingests and validates market data with strict quality controls
- Engineers 25 leakage-free features through rigorous temporal constraints
- Trains ranking models using 70+ walk-forward folds over 8 years
- Evaluates strategies on a **locked holdout period** never used for tuning
- Applies realistic transaction costs (5 basis points)
- Compares ML strategies against baseline alternatives (EqualWeight, MinVariance, RiskParity)
- Performs statistical significance testing (IC t-tests, paired bootstrap)
- Documents failure modes through forensic attribution analysis
- **Makes production decisions based on evidence, not backtest optimism**

The system was designed from the ground up to answer one question honestly:

> **"Can machine learning consistently beat a simple EqualWeight ETF portfolio after accounting for real-world trading costs?"**

**Answer: No, it cannot—at least not in the tested 10-asset universe with the current feature set.**

---

## Why This Repository Matters

Most portfolio optimization projects end after a promising backtest.

This project continues **through to a deployment decision**.

### What Makes This Different

**We built a research system, not just a model.**

The architecture includes:

- **Data Engineering:** ETL/ELT pipelines with Pandera schema validation and Parquet-based data lake
- **Leakage Prevention:** 12 automated tests verify no forward-looking information enters features
- **Time-Series Validation:** Walk-forward splitter with expanding windows—train/test never overlap
- **Honest Benchmarking:** EqualWeight, MinVariance, RiskParity baselines evaluated under identical conditions
- **Production Standards:** Transaction costs, turnover limits, drawdown constraints, and staleness checks
- **Experiment Governance:** Locked holdout periods, pre-declared acceptance criteria, Bonferroni-adjusted p-values
- **Research Integrity:** Negative results documented openly; weak strategies rejected before deployment

**The system did exactly what it was designed to do: identify that the ML strategy was not ready for production.**

---

## Final Research Outcome

### The Hypothesis

> "Machine learning can extract predictive signals from cross-sectional and time-series features to outperform an EqualWeight ETF portfolio after realistic transaction costs."

### The Research Journey

**Phase 1 (Data Engineering):**
- Built a validated data pipeline for 10 ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, DBC, DIA)
- 2,131 trading days per symbol (2018–2026), zero data quality issues
- Parquet-based data lake with DuckDB query layer

**Phase 2 (Exploratory Analysis):**
- Confirmed dataset integrity: balanced panel, no duplicates, expected correlations
- Validated financial coherence: equity cluster (SPY–QQQ ≈ 0.94), TLT hedge (≈ -0.14), GLD diversifier (≈ 0.12)

**Phase 3.1–3.2 (Model Development):**
- Engineered 25 leakage-free features (cross-sectional ranks, momentum, volatility, drawdown, regime flags)
- Trained Ridge, RandomForest, and GradientBoosting ranking models
- Best model (GradientBoosting): Mean Rank IC = **0.131** (71.9% positive folds, p < 0.001)
- Pre-holdout performance: 15.15% annualized return vs. 11.49% EqualWeight (+3.66 pp edge)

**Phase 3.3 (Locked Holdout Evaluation):**
- Selected model on pre-holdout data (cutoff: 2024-12-31)
- Evaluated on locked holdout (2025-01-02 to 2026-07-22, 388 trading days)
- **Holdout result:**
  - **ML Strategy:** 8.36% annualized, 0.60 Sharpe, -15.96% max drawdown
  - **EqualWeight Baseline:** 22.36% annualized, 1.70 Sharpe, -12.83% max drawdown
  - **Deficit:** ML underperformed by **14.00 percentage points**

**Failure Attribution Analysis:**
- Rank IC collapsed from +0.1244 (pre-holdout) to **-0.0408** (holdout)
- Signal inverted: only 41% of holdout folds had positive Rank IC (vs. 69% pre-holdout)
- Transaction costs contributed only **0.12 pp** to the 14.00 pp deficit
- **Root cause:** Poor generalization (99.14% of failure), not transaction costs (0.86%)

**Additional Research Cycles:**
- Tested 7 turnover-aware variants with hysteresis, longer rebalance periods, and sticky allocations
- Best candidate (H): 17.83% return, but 17.38% turnover (violated 15% acceptance gate)
- Low-turnover candidates: Either lost Sharpe vs. EqualWeight or became trivial clones

### The Evidence

| Metric | Pre-Holdout (In-Sample) | Holdout (Out-of-Sample) | Verdict |
|--------|-------------------------|-------------------------|---------|
| Mean Rank IC | +0.1244 | **-0.0408** | ❌ Signal inverted |
| Positive Folds | 69.0% | 41.2% | ❌ Majority negative |
| Annualized Return | 15.15% | **8.36%** | ❌ Lost 6.79 pp |
| Sharpe Ratio | 0.860 | **0.600** | ❌ Below EqualWeight (1.70) |
| Return vs. EqualWeight | +3.66 pp | **-14.00 pp** | ❌ Catastrophic underperformance |

**Counterfactual Analysis:**

- **With zero transaction costs:** ML would still lose by 13.88 pp (99.14% of deficit remains)
- **With zero turnover:** ML would still lose by 13.99 pp
- **Conclusion:** The ML strategy failed because it could not generalize, not because of friction costs

### The Production Decision

**Status:** **ML strategy rejected. EqualWeight selected as production baseline.**

**Rationale:**

1. The ML strategy underperformed EqualWeight on every metric (return, Sharpe, max drawdown)
2. The predictive signal was negative on holdout data (Rank IC = -0.041)
3. Even with zero transaction costs, the strategy would lose by 13.88 pp annually
4. EqualWeight achieved 22.36% return with 0.26% turnover—a strong, low-cost benchmark

**Acceptance Criteria (all failed):**

- ❌ Net return must exceed EqualWeight (failed by 14.00 pp)
- ❌ Net Sharpe must exceed EqualWeight (0.60 vs. 1.70)
- ❌ Max drawdown no worse than EqualWeight + 2pp tolerance (-15.96% vs. -12.83%)
- ❌ Turnover ≤ 15% (actual: 19.51%)

**The system correctly rejected the ML strategy.**

---

## Why Version 1.0 Is a Success

This repository demonstrates that **production ML requires more than a good backtest.**

### What Was Achieved

✅ **Engineering Excellence:**
- Production-quality data pipelines with validation and versioning
- Leakage-free feature engineering (12 automated tests)
- Walk-forward validation with 70+ folds
- Realistic transaction cost modeling
- Comprehensive test suite (pytest coverage)

✅ **Research Integrity:**
- Hypothesis clearly stated upfront
- Locked holdout never used for tuning
- Pre-declared acceptance criteria
- Statistical significance testing (t-tests, bootstrap)
- Failure attribution analysis

✅ **Honest Model Selection:**
- Compared ML against baselines under identical conditions
- Rejected weak strategies before deployment
- Documented negative results openly
- Selected EqualWeight as the production-ready alternative

✅ **Production Standards:**
- Transaction cost accounting
- Turnover constraints
- Staleness checks
- Output validation
- Experiment governance

### What Was Learned

**Good backtests are not enough.** In-sample performance (15.15% return, Rank IC = 0.124) did not survive the transition to out-of-sample holdout data.

**Locked holdouts matter.** Without a locked holdout, the project would have deployed a strategy that lost 14 pp annually.

**Simplicity often wins.** EqualWeight (10-asset basket, rebalanced annually) beat complex ML with 70+ features and monthly rebalancing.

**Weak signals disappear under friction.** A modest Rank IC (0.124) is insufficient when turnover costs 1–2 pp annually.

**Research should be designed to reject weak ideas.** The system rejected the ML strategy—that's a feature, not a bug.

**Negative results are valuable when obtained rigorously.** This repository proves the methodology was sound, even though the hypothesis was not supported.

---

---

## Architecture Overview (Version 1.0)

The system follows a modern data lake / MLOps architecture with distinct layers, from raw ingestion through to a validated portfolio recommendation:

```
Data Sources (Yahoo Finance)
    ↓
Raw Data Ingestion
    ↓
Raw Storage / Data Lake (Parquet)                          — Phase 1
    ↓
Data Validation (Pandera schemas)
    ↓
Cleaning & Standardization
    ↓
Processed Analytical Tables (Parquet)
    ↓
Feature Generation (technical, macro, cross-asset)
    ↓
Feature Tables (Parquet)
    ↓
DuckDB Query Layer
    ↓
Exploratory Data Analysis (notebook + EDA report)           — Phase 2
  • integrity checks • missingness audit
  • return distribution • correlation structure
    ↓
Leakage-Safe Feature Set v2 (cross-sectional, TS, regime)   — Phase 3.1 / 3.2
    ↓
Forward-Looking Targets (1d / 5d / 21d returns, rank, class)
    ↓
Walk-Forward Splitter (expanding / rolling, no lookahead)
    ↓
    ├── Baseline Strategies: EqualWeight / MinVariance / RiskParity
    └── ML Ranking Models: Ridge / RandomForest / GradientBoost / Ensemble
    ↓
    ├── Backtesting Engine (Sharpe, MaxDD, Turnover)
    └── Rank IC / Precision@K + Significance Testing (t-test, bootstrap Sharpe diff)
    ↓
Model Selection (best mean Rank IC, statistically significant,
Bonferroni-adjusted)                                        — Phase 3.3
    ↓
Regime-Aware Routing (Bull/Bear × High/Low Vol)
    ↓
Two-Stage Portfolio Construction
(top-K equal / score-weighted / mean-variance)
    ↓
Output Validation (weights, freshness, disclaimer)
    ↓
Recommendation Artifact (CSV + JSON)
    ↓
MLflow Experiment Tracking (throughout)
    ↓
Future: Serving, Monitoring, Deployment
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Dependency Management | pyproject.toml |
| Data Source | yfinance |
| Data Transformation | pandas |
| Data Validation | pandera |
| Storage Format | Parquet (pyarrow) |
| Query Engine | DuckDB |
| Pipeline Orchestration | Prefect |
| EDA & Visualization | Jupyter, matplotlib, seaborn |
| ML Models | scikit-learn (Ridge, RandomForest, HistGradientBoosting) |
| Statistics | scipy (t-tests, bootstrap) |
| Experiment Tracking | MLflow |
| Configuration | YAML |
| Testing | pytest |
| Logging | Python logging |

---

## Repository Structure

```
portfolio-ml-system/
  README.md                       # This file
  pyproject.toml                  # Python project configuration
  .gitignore                      # Git ignore rules
  .env.example                    # Environment variables template

  configs/
    assets.yaml                   # Asset universe configuration (10 ETFs)
    pipeline.yaml                 # Pipeline configuration (dates, paths)
    experiments.yaml               # Phase 3 experiment grid / backtest / MLflow config

  data/
    raw/                          # Raw ingested data (partitioned Parquet)
    processed/                    # Clean daily prices (partitioned Parquet)
    marts/                        # Feature tables (partitioned Parquet)
    validation_reports/           # Validation reports (JSON)
    portfolio_ml.duckdb           # DuckDB database file

  src/
    portfolio_ml/
      config/
        loader.py                 # Configuration loader (incl. experiment config)
      data_sources/
        base.py                   # Abstract data source interface
        yahoo.py                  # Yahoo Finance implementation
      ingestion/
        ingest_prices.py          # Raw price ingestion orchestration
      validation/
        schemas.py                # Pandera validation schemas
        checks.py                 # Validation check functions
      transformations/
        clean_prices.py           # Clean raw prices
        build_features.py         # Generate feature table (+ macro/cross-asset)
      features/
        targets.py                 # Forward-looking return/volatility targets
        feature_set_v2.py          # Cross-sectional, TS momentum, regime features
        ranking_targets.py         # Rank/class targets + rank-aware metrics
        leakage.py                 # Leakage validation report
      modeling/
        walk_forward.py            # Time-series CV splitter (expanding/rolling)
        scalers.py                 # Train-only scaler wrapper
        baselines.py               # EqualWeight / MinVariance / RiskParity
        rank_evaluator.py          # Rank IC / Precision@K evaluation
        regimes.py                 # Regime labeling + regime-aware strategy
        two_stage_portfolio.py     # Signal → weight construction (3 methods)
        ranking_pipeline.py        # Shared walk-forward ranking pipeline
        two_stage_pipeline.py      # Shared walk-forward two-stage pipeline
        ml_models/
          ranking.py                # Gradient-boost ranking model
          factory.py                # Model factory (ridge/RF/GBM/ensemble)
      backtesting/
        engine.py                  # Backtest engine (Sharpe, MaxDD, turnover)
        benchmark_runner.py        # Baseline comparison runner
        slice_eval.py              # Regime-based / slice evaluation
      evaluation/
        significance.py            # IC t-test, paired bootstrap Sharpe diff
        output_validation.py       # Recommendation output contract checks
      experiments/
        grid_runner.py             # Config-driven experiment grid runner
        unified_runner.py          # Unified Sprint 1-4 experiment runner
        schema.py                  # Stable experiment output schema (v2)
        tracking.py                # MLflow experiment tracking helpers
      storage/
        paths.py                  # Path helpers for data lake
        parquet.py                # Parquet read/write utilities
        duckdb_client.py          # DuckDB client wrapper
      pipelines/
        daily_market_data_pipeline.py  # Main Prefect pipeline
      utils/
        logging.py                # Centralized logging
        dates.py                  # Date utilities

  notebooks/
    01_data_exploration.ipynb     # Phase 2 EDA notebook (integrity, distributions, correlations)

  scripts/
    demo_quickstart.py             # 5-minute baseline demo
    run_baseline.py                 # Full baseline strategy run
    run_unified_experiments.py      # Sprint 1-4 unified experiment entry point
    train_ranking_model.py          # Walk-forward ML ranking model training
    run_two_stage_portfolio.py      # Two-stage portfolio construction + backtest
    run_model_selection.py          # Phase 3.3 hyperparameter search + model selection
    train_regime_aware_model.py     # Regime-aware model training/comparison
    generate_recommendation.py      # Final validated recommendation generator
    evaluate_by_regime.py           # Strategy performance by market regime
    verify_phase_3_2.py             # Phase 3.2 validation checks

  outputs/
    phase3_baseline/                # Baseline strategy returns & summary
    phase3_experiment_grid/         # Sprint 1-4 experiment grid results
    phase3_ml/                      # Raw-return regression model results
    phase3_regime/                  # Strategy performance by regime
    phase3_3/                       # Model search, recommendations, regime comparison

  docs/
    Phase1(Data_Engineering)/       # Phase 1 quickstart, summary, verification
    Phase2(EDA)/                    # EDA_report.md — full Phase 2 write-up
    Phase3(Modeling)/                # Phase 3 workflow, quickstart, and results docs

  tests/
    test_config_loader.py         # Config loader tests
    test_yahoo_source.py          # Yahoo Finance source tests
    test_data_validation.py       # Validation tests
    test_clean_prices.py          # Clean prices transformation tests
    test_feature_generation.py    # Feature generation tests
    test_leakage.py                # Leakage prevention test suite (12 tests)
    test_experiment_config.py      # Experiment config schema tests
    test_ml_ranking.py             # ML ranking model tests
    test_regime_strategy.py        # Regime-aware strategy tests
    test_unified_runner.py         # Unified experiment runner tests
    test_walk_forward.py           # Walk-forward splitter tests
    test_phase_3_2.py              # Phase 3.2 schema/feature/target/weight tests (29)
    test_phase33_output_validation.py  # Phase 3.3 recommendation output contract tests
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or poetry

### Setup

```bash
# Clone the repository
cd /path/to/ML_Portfolio_Recommendation_System

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e ".[dev]"
```

The `[dev]` extra includes testing and linting tools.

---

## How to Run the Daily Pipeline

The main pipeline fetches, validates, cleans, and generates features for all configured assets.

### Run the full pipeline

```bash
python -m portfolio_ml.pipelines.daily_market_data_pipeline
```

### What happens:

1. **Load configs** from `configs/assets.yaml` and `configs/pipeline.yaml`
2. **Extract prices** from Yahoo Finance for 10 symbols from 2018-01-01 to today
3. **Validate raw prices** using Pandera schemas
4. **Clean prices**: normalize, compute returns, remove duplicates
5. **Validate clean prices**
6. **Save clean prices** to `data/processed/daily_prices/`
7. **Build features**: rolling returns, volatility, momentum, drawdown, targets
8. **Validate features**
9. **Save features** to `data/marts/features/`
10. **Create DuckDB views** over all Parquet datasets

### Expected output

After a successful run, you'll have:

```
data/
  raw/source=yfinance/interval=1d/symbol=SPY/year=2018/prices.parquet
  raw/source=yfinance/interval=1d/symbol=SPY/year=2019/prices.parquet
  ...
  processed/daily_prices/year=2018/daily_prices.parquet
  processed/daily_prices/year=2019/daily_prices.parquet
  ...
  marts/features/year=2018/asset_daily_features.parquet
  marts/features/year=2019/asset_daily_features.parquet
  ...
  portfolio_ml.duckdb
  validation_reports/raw_daily_prices_20260623T120000.json
  validation_reports/clean_daily_prices_20260623T120000.json
  validation_reports/asset_daily_features_20260623T120000.json
```

---

## Data Storage Design

### Raw Layer

**Path pattern:**

```
data/raw/source={source}/interval={interval}/symbol={symbol}/year={year}/prices.parquet
```

**Example:**

```
data/raw/source=yfinance/interval=1d/symbol=SPY/year=2024/prices.parquet
```

**Partitioning:** source, interval, symbol, year

**Grain:** One row per symbol per trading date

**Primary key:** symbol + date + source

### Processed Layer

**Path pattern:**

```
data/processed/daily_prices/year={year}/daily_prices.parquet
```

**Partitioning:** year

**Grain:** One row per symbol per trading date

**Primary key:** symbol + date

### Feature Layer

**Path pattern:**

```
data/marts/features/year={year}/asset_daily_features.parquet
```

**Partitioning:** year

**Grain:** One row per symbol per trading date

**Primary key:** symbol + date

---

## Data Contracts

### Raw Daily Prices

**Required columns:**

- `date` (date): Trading date
- `symbol` (string): Ticker symbol
- `open`, `high`, `low`, `close` (float): OHLC prices
- `adj_close` (float): Adjusted closing price
- `volume` (float): Trading volume
- `dividends` (float): Dividend per share
- `stock_splits` (float): Stock split ratio
- `source` (string): Data source identifier
- `ingested_at` (timestamp): UTC ingestion timestamp

**Quality rules:**

- `adj_close` must be positive
- `volume` must be >= 0
- `high >= low`, `high >= open`, `high >= close`, `low <= open`, `low <= close`
- No duplicate rows for `symbol + date + source`

### Clean Daily Prices

**Required columns:**

- `date`, `symbol`, `open`, `high`, `low`, `close`, `adj_close`, `volume`
- `return_1d` (float): Daily simple return (null for first row per symbol)
- `log_return_1d` (float): Daily log return (null for first row per symbol)
- `is_trading_day` (bool): Always True
- `data_quality_flag` (string): "ok" or issue description
- `created_at` (timestamp): UTC record creation timestamp

**Quality rules:**

- `adj_close` must be positive
- `volume` must be >= 0
- No duplicate rows for `symbol + date`

### Feature Table

**Required columns:**

- `date`, `symbol`
- `return_1d`, `log_return_1d`, `return_5d`, `return_21d`, `return_63d`
- `volatility_21d`, `volatility_63d`
- `momentum_21d`, `momentum_63d`
- `drawdown`
- `rolling_volume_21d`
- `price_to_ma_21`, `price_to_ma_63`
- `target_return_1d`, `target_return_5d` (labels — use future data)
- `created_at`

**Quality rules:**

- No duplicate rows for `symbol + date`
- Features must not use future data (except targets)
- Targets are intentionally forward-looking labels

---

## How to Query Data with DuckDB

### Command-line interface

```bash
duckdb data/portfolio_ml.duckdb
```

### Example queries

```sql
-- View summary stats per symbol
SELECT
    symbol,
    COUNT(*) AS rows,
    MIN(date) AS first_date,
    MAX(date) AS last_date
FROM clean_daily_prices
GROUP BY symbol
ORDER BY symbol;

-- Compute average daily return and volatility
SELECT
    symbol,
    AVG(return_1d) AS avg_daily_return,
    STDDEV(return_1d) AS daily_volatility,
    COUNT(*) AS observations
FROM clean_daily_prices
GROUP BY symbol
ORDER BY symbol;

-- Inspect recent feature values
SELECT *
FROM asset_daily_features
WHERE date >= '2024-01-01'
ORDER BY symbol, date
LIMIT 20;
```

### Python API

```python
from portfolio_ml.storage.duckdb_client import DuckDBClient

client = DuckDBClient(db_path="data/portfolio_ml.duckdb")
with client:
    client.create_views()
    
    # Run a query
    df = client.query("SELECT * FROM clean_daily_prices LIMIT 10")
    print(df)
    
    # Use helper methods
    summary = client.summary_stats()
    print(summary)
```

---

## How to Run Tests

```bash
pytest
```

### Run specific test files

```bash
pytest tests/test_config_loader.py
pytest tests/test_yahoo_source.py
pytest tests/test_data_validation.py
pytest tests/test_clean_prices.py
pytest tests/test_feature_generation.py
pytest tests/test_leakage.py
pytest tests/test_walk_forward.py
pytest tests/test_experiment_config.py
pytest tests/test_ml_ranking.py
pytest tests/test_regime_strategy.py
pytest tests/test_unified_runner.py
pytest tests/test_phase_3_2.py
pytest tests/test_phase33_output_validation.py
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage

```bash
pytest --cov=portfolio_ml
```

---

## Asset Universe

The system currently tracks 10 ETFs across multiple asset classes:

| Symbol | Name | Asset Class |
|--------|------|-------------|
| SPY | S&P 500 ETF | Equity |
| QQQ | Nasdaq 100 ETF | Equity |
| IWM | Russell 2000 ETF | Equity |
| EFA | Developed Markets ex-US ETF | Equity |
| EEM | Emerging Markets ETF | Equity |
| TLT | Long-Term Treasury Bond ETF | Bond |
| GLD | Gold ETF | Commodity |
| VNQ | Real Estate ETF | Real Estate |
| DBC | Commodities ETF | Commodity |
| DIA | Dow Jones Industrial Average ETF | Equity |

To modify the asset list, edit `configs/assets.yaml`.

---

## Phase 2 — Exploratory Data Analysis

Before any modeling, the engineered Phase 1 dataset was audited end-to-end in `notebooks/01_data_exploration.ipynb`, with findings written up in [`docs/Phase2(EDA)/EDA_report.md`](<docs/Phase2(EDA)/EDA_report.md>).

### What was checked

- **Structural integrity** — 21,310 rows × 19 columns, a perfectly balanced panel of 2,131 rows per symbol across all 10 ETFs, and **0 duplicate** `symbol + date` records.
- **Missing-value audit** — missingness appears only in rolling-window features (e.g. 210 NaNs in `return_21d`/`momentum_21d`, 630 in `return_63d`/`momentum_63d`), exactly matching `window_size × 10 assets`. This confirms missingness is a product of correct rolling-window construction, not data loss.
- **Return distribution sanity check** — daily returns (`return_1d`) have mean ≈ 0.00044, std ≈ 0.0126, min ≈ -17.7%, max ≈ +12.0%: consistent with fat-tailed, real-world ETF behavior.
- **Correlation structure vs. financial theory** — the correlation matrix confirms expected relationships:
  - a strong equity cluster (SPY–QQQ ≈ 0.94, SPY–DIA ≈ 0.94, SPY–IWM ≈ 0.87)
  - `TLT` as a mild equity hedge (TLT–SPY ≈ -0.14)
  - `GLD` as a low-correlation diversifier (GLD–SPY ≈ 0.12)
  - `DBC` with moderate, but distinct, equity linkage (≈ 0.25–0.32)

### Outcome

The EDA confirmed the dataset is both **technically clean** and **financially coherent**, and cleared the project to move from data validation into **Phase 3: modeling and backtesting**. Two follow-up risk notes were carried forward into Phase 3: (1) manually spot-check extreme single-day moves, and (2) run a dedicated leakage audit before training any model — both of which are addressed by `tests/test_leakage.py` and the feature-set leakage validator in Phase 3.

---

## Phase 3 — Modeling, Backtesting & Recommendation

Phase 3 builds a leakage-free, walk-forward modeling and portfolio-construction pipeline on top of the validated Phase 1/2 dataset, delivered across three sub-phases (3.1 → 3.2 → 3.3). Full write-ups live under [`docs/Phase3(Modeling)/`](<docs/Phase3(Modeling)/>).

### Core Building Blocks

| Module | Purpose |
|---|---|
| `features/targets.py` | Forward-looking targets (`future_1d/5d/21d_return`, `forward_volatility`) — explicitly shifted to prevent backward leakage |
| `modeling/walk_forward.py` | `WalkForwardSplitter` — expanding or rolling time-series CV; train/test never overlap, train always strictly precedes test |
| `modeling/scalers.py` | `TrainOnlyScaler` — raises if `transform()` is called before `fit()` on training data only |
| `modeling/baselines.py` | `EqualWeightStrategy`, `MinVarianceStrategy`, `RiskParityStrategy` |
| `backtesting/engine.py` + `benchmark_runner.py` | Portfolio backtest with transaction costs, Sharpe, Max Drawdown, turnover |
| `backtesting/slice_eval.py` | Regime detection (bull/bear/high-vol/low-vol) and per-regime strategy comparison |
| `experiments/tracking.py` | MLflow experiment tracking (params, metrics, artifacts) with graceful fallback if MLflow is unavailable |

All of the above are covered by `tests/test_leakage.py` (12 tests) verifying targets are forward-only, splits never overlap, scalers never see test data, and rolling/cross-sectional features never peek forward.

### 3.1 — Config-Driven Experiment Grid & Sprints 1-4

Full workflow: [`docs/Phase3(Modeling)/Phase3.1_WORKFLOW.md`](<docs/Phase3(Modeling)/Phase3.1_WORKFLOW.md>)

- **Sprint 1 — Experiment grid runner:** `configs/experiments.yaml` drives a Cartesian-product grid over lookback windows (`126/252/504`), rebalance frequencies (`5/21/63`), and strategies (EqualWeight/RiskParity/MinVariance), executed by `experiments/grid_runner.py` and ranked by a composite Sharpe/MaxDD/turnover score.
- **Sprint 2 — Feature expansion:** added macro placeholders (`macro_vix`, `macro_ten_year_yield`, `macro_dxy`) and cross-asset features (`rolling_corr_spy_21d`, `spy_relative_momentum_21d`) to `transformations/build_features.py`, plus a leakage validator in `features/leakage.py`.
- **Sprint 3 — ML ranking layer:** `GradientBoostRankModel` (`modeling/ml_models/ranking.py`) and `RankEvaluator` (`modeling/rank_evaluator.py`) — a first benchmark comparing ML-ranked predictions against heuristic baselines.
- **Sprint 4 — Regime layer:** `RegimeLabeler` and `RegimeAwareStrategy` (`modeling/regimes.py`) route allocation across simple market-regime heuristics.
- **Unified entry point:** `scripts/run_unified_experiments.py` runs the full baseline grid, ML ranking experiment, and regime-aware evaluation in one pass, writing a consolidated ranking CSV.

```bash
PYTHONPATH=src python scripts/run_unified_experiments.py
```

### 3.2 — Feature Set v2, Ranking Targets & Two-Stage Portfolios

Full results: [`docs/Phase3(Modeling)/PHASE3.2_RESULTS.md`](<docs/Phase3(Modeling)/PHASE3.2_RESULTS.md>)

Phase 3.2 replaced noisy raw-return regression (R² ≈ -0.17) with a **ranking-based** approach and connected ML signals to portfolio construction correctly:

- **Stable experiment schema** — `experiments/schema.py` defines 31 canonical output columns with validation (catches missing identifiers, impossible Sharpe/MDD values, out-of-range Rank IC, etc.).
- **Feature set v2** (`features/feature_set_v2.py`, 25 features, all leakage-safe): cross-sectional ranks/z-scores/relative momentum, trailing return & volatility, drawdown-from-high, benchmark correlation/beta, and volatility/drawdown regime flags.
- **Ranking targets & metrics** (`features/ranking_targets.py`): cross-sectional percentile rank, tertile/binary class targets, Rank IC (Spearman) and Precision@K.
- **Two-stage portfolio construction** (`modeling/two_stage_portfolio.py`): ML score → portfolio weights via `top_k_equal`, `score_weighted`, or `mean_variance`.

**Headline result (Ridge + Top-K-Equal, vs. EqualWeight baseline):**

| Metric | Ridge Ranking + Top-K-Equal | EqualWeight Baseline |
|---|---:|---:|
| Annualised Return | 15.15% | 11.49% |
| Sharpe Ratio | 0.860 | 0.837 |
| Max Drawdown | -33.03% | -26.76% |
| Mean Rank IC | 0.036 (62.1% positive folds) | — |

```bash
PYTHONPATH=src python scripts/train_ranking_model.py --model ridge --rebalance-freq 21 --lookback 252
PYTHONPATH=src python scripts/run_two_stage_portfolio.py --model ridge --portfolio-method top_k_equal --top-k 5 --max-weight 0.25
PYTHONPATH=src python scripts/verify_phase_3_2.py   # 7 automated schema/leakage/weight checks
```

**Verdict:** ranking adds modest Sharpe improvement over baselines but with higher drawdown — a real but weak signal that motivated Phase 3.3's model search and regime awareness.

### 3.3 — Model Selection, Regime Awareness & Recommendation Generation

Full results: [`docs/Phase3(Modeling)/PHASE3.3_RESULTS.md`](<docs/Phase3(Modeling)/PHASE3.3_RESULTS.md>)

Phase 3.3 adds systematic model comparison, statistical rigor, regime-aware routing, and a production-style recommendation generator on top of the validated Phase 3.2 pipeline:

- **Model factory & ensemble** (`modeling/ml_models/factory.py`): `ridge`, `random_forest`, `gradient_boost` (`HistGradientBoostingRegressor`), and a rank-averaged `ensemble` of all three.
- **Statistical significance** (`evaluation/significance.py`): one-sided `ic_ttest` on mean Rank IC, and `paired_bootstrap_sharpe_diff` for strategy-vs-baseline Sharpe comparisons, with Bonferroni-adjusted p-values across the hyperparameter grid.
- **Regime-aware model** (`RegimeAwareRankModel`): routes to a per-regime model (`Bull/Bear_LowVol/HighVol`, ≥ 200 rows required) or falls back to the global model.
- **Output validation** (`evaluation/output_validation.py`): enforces unique symbols, non-NaN scores/weights, weights ≥ 0 summing to 1 within `max_weight`, the exact disclaimer `"Not investment advice."`, and rejects recommendations built on data older than 3 calendar days.

**Selected model (hyperparameter search winner):**

| model_name | params | mean Rank IC | significant | p-value |
|---|---|---:|---:|---:|
| `gradient_boost` | `learning_rate=0.03, max_depth=6, max_iter=300` | 0.131 (71.9% positive folds) | ✅ | 1.04e-06 |

```bash
PYTHONPATH=src python3 scripts/run_model_selection.py        # hyperparameter search + model selection
PYTHONPATH=src python3 scripts/train_regime_aware_model.py   # regime-aware training/comparison
PYTHONPATH=src python3 scripts/generate_recommendation.py    # final validated recommendation (CSV + JSON)
PYTHONPATH=src pytest -q tests/test_phase33_output_validation.py
```

Outputs land in `outputs/phase3_3/`: `hyperparam_search_results*.csv`, `model_selection_summary*.csv`, `best_model.json`, `regime_aware_fold_results.csv`, `regime_vs_global_comparison.csv`, and versioned `recommendation_<date>_<timestamp>.{csv,json}` files. Recommendation JSON always includes model metadata, regime status, portfolio method, feature/data-cutoff dates, and the investment disclaimer.

**Known limitations:** research outputs are not investment advice; they exclude taxes, liquidity, and slippage beyond configured transaction costs; and stale-data rejection is intentional — refresh the Phase 1 pipeline before generating a new recommendation.

---

---

## Lessons Learned

### 1. Good Backtests Are Not Enough

In-sample performance metrics can be misleading. The ML strategy showed promising results during walk-forward training (15.15% return, 0.860 Sharpe, Mean Rank IC = 0.124), but these metrics collapsed on the locked holdout period. Without a true out-of-sample test on data never seen during development, this project would have deployed a failing strategy.

**Takeaway:** Always reserve a locked holdout period for final validation. Never tune against your holdout data.

### 2. Locked Holdouts Matter

The locked holdout period (2025-01-02 to 2026-07-22) was declared before model selection and never used for hyperparameter tuning, feature engineering, or acceptance criteria adjustments. This discipline revealed the truth: the ML strategy's edge was not robust.

**Takeaway:** Treat the holdout as a production environment. If you wouldn't deploy without testing in production, you shouldn't deploy without testing on a locked holdout.

### 3. Simplicity Often Wins

EqualWeight—a simple 10-asset basket rebalanced annually—achieved 22.36% annualized return with 1.70 Sharpe and minimal turnover (0.26%). The ML strategy, despite 25 features and sophisticated ranking models, could not beat this baseline after accounting for realistic trading costs.

**Takeaway:** Complex models must earn their complexity. If a simple baseline is strong, the ML system must demonstrably outperform it—not just in backtest, but in production-like conditions.

### 4. Weak Signals Disappear Under Friction

A Rank IC of 0.124 (pre-holdout) seems promising, but it's insufficient when:
- The universe has only 10 assets (limited opportunity)
- Assets are highly correlated (SPY–QQQ ≈ 0.94)
- Monthly rebalancing generates 19.51% turnover
- Transaction costs are realistic (5 basis points)

**Takeaway:** Signal strength must be calibrated against implementation costs. For this universe and rebalancing frequency, a Rank IC ≥ 0.15–0.20 would be needed to generate sustainable alpha after costs.

### 5. Research Should Be Designed to Reject Weak Ideas

The acceptance criteria (return > EqualWeight, Sharpe > EqualWeight, turnover ≤ 15%, statistical significance) were deliberately conservative. Seven turnover-aware candidates were tested; all failed at least one criterion. The system did not relax criteria to force a weak strategy into production.

**Takeaway:** Set high standards and enforce them. A research system that never rejects anything is not doing its job.

### 6. Negative Results Are Valuable When Obtained Rigorously

This project proves:
- The data pipeline is sound (12 leakage tests pass)
- The features are correctly constructed (walk-forward validation)
- The evaluation is honest (locked holdout, transaction costs, statistical tests)
- The methodology is rigorous (failure attribution analysis)

The negative result is not a failure—it's a success of the research process.

**Takeaway:** Document negative results openly. They contribute to the body of knowledge and demonstrate research integrity.

### 7. Transaction Costs Are Often a Scapegoat

The failure attribution analysis revealed that transaction costs contributed only **0.12 pp** (0.86%) to the 14.00 pp deficit. The primary cause was **signal inversion** (99.14% contribution)—the model's predictions became negatively correlated with returns on the holdout period.

**Takeaway:** Don't blame transaction costs for a fundamentally weak signal. If your strategy would still fail with zero costs, the problem is the signal, not the friction.

### 8. Turnover Is a Feature, Not a Bug

High turnover (19.51% for the ML strategy vs. 0.26% for EqualWeight) is not inherently bad—it's a signal that the strategy is actively trading. But it must be justified by alpha. If turnover does not generate sufficient edge to cover its costs, it's wasteful.

**Takeaway:** Turnover penalties should be part of the objective function during model training, not an afterthought during evaluation.

### 9. Strong Baselines Set the Bar High

EqualWeight performed exceptionally well on the holdout period (22.36% return, 1.70 Sharpe). This is a strong benchmark. Any active strategy must justify its existence by beating this baseline convincingly—not by a few basis points, but by enough to cover management effort, monitoring costs, and implementation risk.

**Takeaway:** Choose your baseline carefully. If it's weak, you'll promote mediocre strategies. If it's strong, you'll only promote truly exceptional ones.

### 10. Walk-Forward Validation Is Necessary But Not Sufficient

Walk-forward validation (70+ folds over 8 years) correctly identified that the GradientBoosting model had the best in-sample Rank IC (0.131). But it could not predict that this signal would invert on the holdout period. Walk-forward validation prevents overfitting within the training period, but it cannot guarantee out-of-sample robustness.

**Takeaway:** Use walk-forward validation for model selection, but always confirm on a locked holdout before deployment.

---

## Repository Lifecycle

### Version 1.0: Complete

Version 1.0 of this project is **feature-complete and stable**.

- ✅ Data engineering pipelines (ingestion, validation, cleaning, feature generation)
- ✅ Exploratory data analysis and reporting
- ✅ Leakage-free feature engineering (25 features, 12 automated tests)
- ✅ Walk-forward model training and validation (70+ folds)
- ✅ Baseline strategies (EqualWeight, MinVariance, RiskParity)
- ✅ ML ranking models (Ridge, RandomForest, GradientBoosting, Ensemble)
- ✅ Two-stage portfolio construction (top-k, score-weighted, mean-variance)
- ✅ Backtesting engine with transaction costs
- ✅ Statistical significance testing (t-tests, bootstrap)
- ✅ Locked holdout evaluation
- ✅ Failure attribution analysis
- ✅ Production decision (EqualWeight selected)

**No further development is planned for Version 1.0.** The repository will remain stable and maintained for reference.

### Future Research (Version 2.0+)

If new research directions are pursued, they will be developed in separate branches or a new repository. Potential directions include:

**Alternative Universes:**
- Larger stock panels (50+ assets for higher cross-sectional dispersion)
- Sector rotation strategies (SPDR sector ETFs)
- Factor-based portfolios (size, value, momentum, quality)
- International equity markets
- Fixed income instruments

**Alternative Signals:**
- Fundamental data (earnings, book value, cash flow)
- Alternative data (sentiment, web traffic, satellite imagery)
- Cross-asset relative value
- Macro regime indicators (yield curve, credit spreads, VIX)

**Alternative Approaches:**
- Longer prediction horizons (quarterly, semi-annual)
- Lower rebalancing frequency (reduce turnover costs)
- Turnover-penalized training objectives
- Deep learning models (LSTM, Transformer)
- Reinforcement learning for portfolio construction

**Deployment Research:**
- Live trading simulation with paper money
- Model monitoring and drift detection
- Adaptive retraining schedules
- Risk overlay systems (drawdown control, volatility targeting)

**Important Note:** These are **NEW research questions**, not unfinished work. Version 1.0 answered the question it set out to answer. Future versions would ask different questions in different contexts.

---

## For Recruiters and Portfolio Reviewers

This repository is designed to demonstrate:

✅ **Production-Grade Engineering:**  
Data pipelines, validation, testing, logging, configuration management, and modular architecture that would work in a real production environment.

✅ **Research Integrity:**  
Honest evaluation, locked holdouts, pre-declared acceptance criteria, statistical rigor, and transparent documentation of negative results.

✅ **Domain Expertise:**  
Understanding of portfolio theory, risk metrics, transaction costs, time-series validation, and financial markets.

✅ **End-to-End Ownership:**  
From data ingestion through model training, backtesting, evaluation, and final production decision—not just model training in isolation.

✅ **Mature Judgment:**  
The ability to reject a strategy that "worked in backtest" because the evidence does not support deployment.

**What this project is NOT:**
- ❌ A failed project (the research succeeded; the hypothesis was rejected)
- ❌ Unfinished work (Version 1.0 is complete)
- ❌ A "moonshot" with no production standards
- ❌ An over-optimized backtest with no holdout validation

**What makes this different from typical portfolio repos:**
- Most stop after a good backtest. This one continues to a deployment decision.
- Most hide negative results. This one documents them openly.
- Most lack production infrastructure. This one has data lakes, validation, testing, and experiment tracking.
- Most lack statistical rigor. This one has t-tests, bootstrap, locked holdouts, and failure attribution analysis.

If you're evaluating this repository for a role in quantitative research, machine learning engineering, or data science, the key questions to ask are:

1. **Can they build production systems?** (Yes—see architecture, testing, validation)
2. **Can they conduct rigorous research?** (Yes—see walk-forward validation, locked holdouts, statistical tests)
3. **Can they make honest decisions?** (Yes—see rejection of ML strategy despite "good backtest")
4. **Can they document and communicate?** (Yes—see this README, EDA report, Phase 3 results, failure attribution analysis)

---

---

## Engineering Principles

This project follows production-grade engineering practices:

✅ **Config-driven, not hardcoded**  
✅ **Idempotent pipeline runs** (safe to run multiple times)  
✅ **Clear separation** between raw, processed, and feature layers  
✅ **No lookahead bias** in features (targets are intentionally forward-looking)  
✅ **Validation before downstream processing**  
✅ **Small reusable modules**  
✅ **Testable components**  
✅ **Readable logging**  
✅ **Reproducible data transformations**  
✅ **Production-style project structure**

---

## Troubleshooting

### Pipeline fails with network errors

Check your internet connection. Yahoo Finance requires outbound HTTP access.

### DuckDB views are empty

Run the pipeline first to generate Parquet files. DuckDB views read from these files.

### Validation fails

Check the validation reports in `data/validation_reports/` for detailed error messages.

### Tests fail

Ensure you've installed dev dependencies with `pip install -e ".[dev]"`.

---

## Contributing

This is a personal portfolio project. Contributions are welcome via pull requests.

### Development workflow

1. Create a feature branch
2. Make changes
3. Run tests: `pytest`
4. Run linter: `ruff check .` (optional)
5. Commit and push
6. Open a pull request

---

## License

This project is for educational and portfolio purposes. No license specified.

---

## Contact

For questions or feedback, please open an issue on the repository.

---

## Acknowledgments

- Data source: [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://github.com/ranaroussi/yfinance)
- Validation: [Pandera](https://pandera.readthedocs.io/)
- Query engine: [DuckDB](https://duckdb.org/)
- Orchestration: [Prefect](https://www.prefect.io/)
- Modeling: [scikit-learn](https://scikit-learn.org/) and [scipy](https://scipy.org/)
- Experiment tracking: [MLflow](https://mlflow.org/)
