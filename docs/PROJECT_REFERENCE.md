# Project Reference

Supplementary reference material for the [ML Portfolio Recommendation System](../README.md): full repository layout, data storage/contract specs, DuckDB usage, and troubleshooting. The README covers the big picture — this doc covers the details you need once you're actually working in the codebase.

---

## Repository Structure

```
portfolio-ml-system/
  README.md                       # Project overview
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
    generate_architecture_gif.py    # Renders docs/assets/architecture.gif

  outputs/
    phase3_baseline/                # Baseline strategy returns & summary
    phase3_experiment_grid/         # Sprint 1-4 experiment grid results
    phase3_ml/                      # Raw-return regression model results
    phase3_regime/                  # Strategy performance by regime
    phase3_3/                       # Model search, recommendations, regime comparison

  docs/
    assets/                         # README media (architecture GIF, etc.)
    Phase1(Data_Engineering)/       # Phase 1 quickstart, summary, verification
    Phase2(EDA)/                    # EDA_report.md — full Phase 2 write-up
    Phase3(Modeling)/                # Phase 3 workflow, quickstart, and results docs
    PROJECT_REFERENCE.md            # This file

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

## Querying Data with DuckDB

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

## Running the Full Test Suite

```bash
pytest                       # all tests
pytest -v                    # verbose
pytest --cov=portfolio_ml    # with coverage
```

Individual suites:

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

## Regenerating the Architecture GIF

The animated diagram at the top of the README (`docs/assets/architecture.gif`) is generated with `matplotlib`:

```bash
python3 scripts/generate_architecture_gif.py
```

Edit the `STAGES` list in that script to update stage labels or phase tags; re-run to regenerate the GIF.
