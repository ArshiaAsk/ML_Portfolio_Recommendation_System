# ML Portfolio Recommendation System — Quickstart Guide

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
cd /home/arshiaask/projects/ML_Portfolio_Recommendation_System

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

### Step 2: Run the Pipeline

```bash
python -m portfolio_ml.pipelines.daily_market_data_pipeline
```

This will:
- Fetch daily prices for 10 ETFs from 2018-01-01 to today
- Validate all data quality rules
- Generate clean analytical tables
- Build 17 features per asset
- Create DuckDB views for SQL queries

**Runtime:** ~2-3 minutes for first run (depends on network speed)

### Step 3: Run Tests

```bash
pytest
```

### Step 4: Query Your Data

```bash
duckdb data/portfolio_ml.duckdb
```

```sql
-- View data summary
SELECT symbol, COUNT(*) AS rows, MIN(date) AS start, MAX(date) AS end
FROM clean_daily_prices
GROUP BY symbol
ORDER BY symbol;

-- Calculate returns and volatility
SELECT 
    symbol,
    AVG(return_1d) * 252 AS annualized_return,
    STDDEV(return_1d) * SQRT(252) AS annualized_volatility
FROM clean_daily_prices
GROUP BY symbol
ORDER BY annualized_return DESC;

-- View recent features
SELECT * FROM asset_daily_features 
WHERE date >= '2024-06-01' 
ORDER BY symbol, date
LIMIT 20;
```

---

## 📁 What Gets Created

After the first successful run:

```
data/
├── raw/                         # ~16,000 raw price records
├── processed/                   # ~16,000 clean price records  
├── marts/                       # ~16,000 feature records
├── validation_reports/          # 3 JSON validation reports
└── portfolio_ml.duckdb          # Query-ready database
```

---

## 🔄 Re-running the Pipeline

The pipeline is **idempotent** — you can run it multiple times safely:

```bash
# Fetch latest data (updates existing + adds new dates)
python -m portfolio_ml.pipelines.daily_market_data_pipeline
```

Existing data is overwritten in place. No duplicates are created.

---

## 🐛 Troubleshooting

### Network errors during data fetch
- Check internet connection
- Yahoo Finance may be temporarily unavailable
- Pipeline will log warnings for failed symbols but continue

### Validation failures
- Check `data/validation_reports/` for detailed error messages
- Common causes: missing data, price anomalies, schema changes

### Import errors
- Ensure you activated the virtual environment: `source .venv/bin/activate`
- Verify installation: `pip list | grep portfolio-ml`

---

## 📊 Next Steps

Once your data pipeline is running:

1. **Explore in Jupyter:**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Build ML models:**
   - Baseline mean reversion
   - Linear regression
   - XGBoost/LightGBM

3. **Implement backtesting:**
   - Walk-forward validation
   - Transaction costs
   - Sharpe ratio tracking

4. **Portfolio optimization:**
   - Mean-variance
   - Risk parity
   - Maximum Sharpe

---

## 📚 Documentation

- `README.md` — Full project documentation
- `VERIFICATION.md` — Implementation status report
- `configs/assets.yaml` — Asset universe
- `configs/pipeline.yaml` — Pipeline parameters

---

## 💡 Pro Tips

**Daily updates:**
```bash
# Add to cron for daily 6 PM updates
0 18 * * 1-5 cd /path/to/project && .venv/bin/python -m portfolio_ml.pipelines.daily_market_data_pipeline
```

**Custom date range:**
Edit `configs/pipeline.yaml`:
```yaml
pipeline:
  start_date: "2020-01-01"  # Change this
  end_date: null            # null = today
```

**Add more assets:**
Edit `configs/assets.yaml` and add new symbols following the existing format.

---

Enjoy building your portfolio recommendation system! 🎯
