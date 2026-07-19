# ML Portfolio Recommendation System

A production-style, end-to-end ML system for daily ETF portfolio recommendations — from raw market data ingestion through exploratory analysis, walk-forward ML modeling, regime-aware routing, and a final, validated portfolio recommendation artifact.

## Project Purpose

This system ingests daily market data for 10 ETFs, validates data quality, transforms raw data into clean analytical tables, generates leakage-safe feature tables, explores and audits the data, and trains/evaluates walk-forward ML ranking models that feed a two-stage portfolio construction and recommendation pipeline.

**Completed phases:**

- **Phase 1 — Data Engineering:** ETL/ELT pipelines, data validation, reproducible feature generation.
- **Phase 2 — Exploratory Data Analysis:** Data quality audit, financial sanity checks, correlation structure review.
- **Phase 3 — Modeling & Recommendation:** Leakage-free walk-forward evaluation, baseline & ML ranking models, regime-aware routing, statistical significance testing, and a validated recommendation generator.

**Future work:** Model ensembling/tuning, richer risk management, serving/productionization (see [Future Roadmap](#future-roadmap)).

---

## Architecture Overview

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

## Future Roadmap

Phases 1–3 above cover data engineering, EDA, baseline/ML modeling, backtesting, two-stage portfolio construction, and recommendation generation. Remaining future work:

### Model Improvement (Phase 4)

- Hyperparameter tuning beyond the fixed grid and richer ensembling
- Additional model families (XGBoost, LightGBM, LSTM/Transformer)
- Turnover-penalized and drawdown-aware objective functions
- Robust, paired out-of-sample comparison of portfolio methods (`top_k_equal` vs `score_weighted` vs `mean_variance`)

### Serving & Productionization (Phase 5)

- FastAPI endpoint for latest recommendations
- Scheduled daily reports
- Dashboard (Streamlit or Dash)
- Data drift monitoring
- Model performance monitoring
- Alerting
- Cloud object storage (S3)
- CI/CD pipeline

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
