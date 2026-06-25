# ML Portfolio Recommendation System — Data Engineering Foundation

A production-style data engineering foundation for an ML-driven portfolio recommendation system. This project focuses on building robust data pipelines, validation, and feature engineering infrastructure before model development.

## Project Purpose

This system ingests daily market data for 10 ETFs, validates data quality, transforms raw data into clean analytical tables, and generates feature tables ready for future ML modeling and portfolio optimization.

**Current focus:** Data engineering, ETL/ELT pipelines, data validation, and reproducible feature generation.

**Future work:** Predictive modeling, backtesting, portfolio optimization, and production deployment.

---

## Architecture Overview

The system follows a modern data lake architecture with distinct layers:

```
Data Sources (Yahoo Finance)
    ↓
Raw Data Ingestion
    ↓
Raw Storage / Data Lake (Parquet)
    ↓
Data Validation (Pandera schemas)
    ↓
Cleaning & Standardization
    ↓
Processed Analytical Tables (Parquet)
    ↓
Feature Generation
    ↓
Feature Tables (Parquet)
    ↓
DuckDB Query Layer
    ↓
Future: Model Training & Portfolio Optimization
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

  data/
    raw/                          # Raw ingested data (partitioned Parquet)
    processed/                    # Clean daily prices (partitioned Parquet)
    marts/                        # Feature tables (partitioned Parquet)
    validation_reports/           # Validation reports (JSON)
    portfolio_ml.duckdb           # DuckDB database file

  src/
    portfolio_ml/
      config/
        loader.py                 # Configuration loader
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
        build_features.py         # Generate feature table
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
    01_data_exploration.ipynb     # Placeholder for exploratory analysis

  tests/
    test_config_loader.py         # Config loader tests
    test_yahoo_source.py          # Yahoo Finance source tests
    test_data_validation.py       # Validation tests
    test_clean_prices.py          # Clean prices transformation tests
    test_feature_generation.py    # Feature generation tests
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

## Future Roadmap

### Modeling (Phase 2)

- Baseline historical mean model
- Rolling mean return model
- Linear regression
- Ridge/Lasso regularization
- Tree-based models (Random Forest)
- Gradient boosting (XGBoost, LightGBM)

### Backtesting (Phase 3)

- Walk-forward validation
- Transaction cost modeling
- Rebalancing frequency analysis
- Portfolio turnover metrics
- Benchmark comparison (SPY)

### Portfolio Optimization (Phase 4)

- Minimum variance portfolio
- Mean-variance optimization
- Maximum Sharpe ratio
- Risk parity
- Weight constraints (e.g., 0 ≤ w_i ≤ 0.3)

Example objective:

```
max_w  (w^T μ - r_f) / sqrt(w^T Σ w)

subject to:
  sum(w_i) = 1
  0 ≤ w_i ≤ 0.3
```

### Serving & Productionization (Phase 5)

- FastAPI endpoint for latest recommendations
- Scheduled daily reports
- Dashboard (Streamlit or Dash)
- MLflow tracking
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
