# Portfolio Recommendation System — Exploratory Data Analysis Report

**Project:** ML-Based Portfolio Recommendation System  
**Phase:** Data Exploration & Validation  
**Status:** Completed and Verified  
**Date:** 2026-06-30

---

## 1. Overview

This report summarizes the exploratory data analysis (EDA) and data quality audit performed on the engineered dataset used for the portfolio recommendation system.

The objective of this EDA was to verify that:

- the dataset is structurally sound,
- feature engineering outputs are consistent with financial expectations,
- missing values are explainable and non-problematic,
- asset relationships are economically meaningful,
- the data pipeline is ready for downstream modeling and backtesting.

The dataset was produced through a layered ELT-style pipeline:

1. **Raw ingestion**
2. **Validation / Bronze**
3. **Cleaning & normalization / Silver**
4. **Feature engineering / Gold**
5. **Analytical inspection via DuckDB and notebook-based EDA**

---

## 2. Dataset Summary

The final engineered dataset contains daily observations for 10 diversified ETFs spanning equities, international markets, fixed income, commodities, gold, and real estate.

### Asset Universe

- SPY
- QQQ
- IWM
- EFA
- EEM
- TLT
- GLD
- VNQ
- DBC
- DIA

### Dataset Shape

- **Rows:** 21,310
- **Columns:** 19

### Per-Asset Row Count

Each symbol contains exactly **2,131 rows**, indicating a balanced panel dataset.

This is an important quality signal because it confirms:

- consistent historical coverage across all assets,
- no silent truncation for individual symbols,
- simpler downstream modeling and rolling-window analysis.

---

## 3. Data Integrity Checks

### 3.1 Duplicate Records

- **Duplicate rows:** 0

This confirms that the pipeline does not generate duplicate symbol-date records and preserves the one-row-per-symbol-per-date contract.

### 3.2 Structural Consistency

The dataset appears to satisfy the expected financial time-series structure:

- one observation per `symbol` and `date`,
- consistent asset coverage,
- clean alignment between raw prices and engineered features.

No structural anomalies were observed in the validation summary shared from the notebook.

---

## 4. Missing Value Analysis

Missing values were detected only in rolling-window-derived features, which is expected and correct.

### Observed Pattern

Examples from the notebook audit:

- **21-day features** such as `return_21d` and `momentum_21d` contain **210 missing values**
- **63-day features** such as `return_63d` and `momentum_63d` contain **630 missing values**

### Interpretation

These missing values arise naturally because rolling features require sufficient lookback history before they can be computed.

Given 10 assets:

- `21 × 10 = 210`
- `63 × 10 = 630`

This confirms that the missingness pattern is not due to data loss or corruption, but rather due to proper feature construction logic.

### Conclusion

The missing values are:

- **expected**
- **mathematically consistent**
- **non-random**
- **not a data quality issue**

---

## 5. Descriptive Statistics of Daily Returns

The distribution of `return_1d` was inspected as part of the financial sanity check.

### Summary Statistics

| Metric | Value |
|---|---:|
| Mean | 0.000438 |
| Standard Deviation | 0.012623 |
| Minimum | -0.177277 |
| Maximum | 0.120031 |

### Interpretation

These values are broadly consistent with real-world financial return behavior:

- The **mean** is close to zero, which is typical for daily returns.
- The **standard deviation** is in a plausible range for a multi-asset ETF universe.
- The distribution exhibits **fat tails**, which is expected in financial markets.
- The presence of large negative and positive moves suggests that market stress periods are preserved in the data.

### Outliers

The minimum and maximum daily returns indicate the presence of extreme market moves:

- **Min:** -17.73%
- **Max:** +12.00%

These are large but still plausible values in historical ETF data, particularly during crisis periods, macro shocks, or periods of elevated volatility.

At this stage, these extreme values should be treated as **valid candidates for market events**, not automatically as errors.

---

## 6. Correlation Structure and Economic Sanity Check

A correlation matrix of asset returns was reviewed to verify whether inter-asset relationships are economically sensible.

### Key Findings

#### 6.1 Strong Equity Cluster

Broad equity ETFs show high positive correlation with one another, including relationships such as:

- **SPY – QQQ:** approximately 0.94
- **SPY – DIA:** approximately 0.94
- **SPY – IWM:** approximately 0.87
- **SPY – EFA:** approximately 0.85

This is expected and confirms that the equity portion of the universe behaves like a coherent risk cluster.

#### 6.2 Bonds as Defensive Exposure

`TLT` shows mildly negative correlation with major equity ETFs, for example:

- **TLT – SPY:** approximately -0.14
- **TLT – DIA:** approximately -0.17
- **TLT – QQQ:** approximately -0.09

This is a strong positive sign for portfolio construction, since it indicates that long-duration Treasuries can provide diversification benefits during equity stress.

#### 6.3 Gold as a Diversifier

`GLD` exhibits low correlation with equities, such as:

- **GLD – SPY:** approximately 0.12
- **GLD – QQQ:** approximately 0.13
- **GLD – DIA:** approximately 0.09

This supports the expectation that gold behaves as a partial hedge or low-correlation diversifier.

#### 6.4 Commodities with Moderate Equity Linkage

`DBC` shows moderate correlation with equities, roughly in the range of:

- **0.25 to 0.32**

This is also financially reasonable, as commodities often share some macro sensitivity with risk assets while remaining distinct from pure equity beta.

---

## 7. Feature Engineering Sanity Check

The EDA results strongly suggest that feature engineering was implemented correctly.

### Supporting Evidence

- Rolling-window features produce exactly the expected number of initial missing values.
- Asset-level row counts are balanced.
- Return distributions are financially plausible.
- Inter-asset correlations match known market structure.
- No duplicate rows were detected.

### Interpretation

This indicates that:

- rolling calculations were applied correctly,
- symbol-level grouping was likely handled properly,
- there is no immediate evidence of cross-asset contamination,
- the transformed dataset is suitable for downstream modeling.

---

## 8. Risk Notes and Follow-Up Checks

Although the dataset is in strong condition, a few follow-up checks remain good practice before production-grade modeling.

### 8.1 Extreme Return Validation

Very large one-day moves should be manually inspected at least once to confirm that they reflect genuine market behavior rather than:

- split adjustment errors,
- dividend adjustment issues,
- bad ticks,
- upstream source anomalies.

This is especially relevant for observations above approximately 10% in absolute daily return.

### 8.2 Leakage Audit

Before training any predictive or allocation model, it remains essential to confirm that:

- all features are computed using only information available at time \( t \),
- all targets are shifted properly into the future,
- no lookahead bias was introduced during transformation or joining.

### 8.3 Rolling Backtest Readiness

The dataset appears suitable for walk-forward evaluation, but the backtesting layer should still enforce:

- time-aware train/test separation,
- rebalancing logic based only on past information,
- realistic weight generation and portfolio accounting.

---

## 9. Final Assessment

The dataset passes the EDA and quality audit with strong results.

### Summary of Findings

- **21,310 rows** across **10 assets**
- **19 columns** in the engineered dataset
- **Balanced coverage** across all symbols
- **0 duplicate rows**
- **Missing values fully explained** by rolling-window mechanics
- **Return distribution financially plausible**
- **Correlation matrix economically consistent**
- **No obvious structural issues detected**

### Conclusion

The project has successfully completed the **data engineering foundation phase** and the dataset is now **validated and ready for Phase 6: Modeling and Backtesting**.

This provides a reliable basis for implementing baseline portfolio models such as:

- Mean-Variance Optimization
- Risk Parity
- Minimum Variance
- Heuristic allocation baselines
- Walk-forward backtesting workflows

---

## 10. Recommended Next Step

The recommended next step is to build a **first baseline portfolio construction model** and evaluate it in a strictly walk-forward framework.

A sensible progression would be:

1. build a simple benchmark allocator,
2. implement Mean-Variance Optimization,
3. compare against equal-weight and risk-parity baselines,
4. evaluate Sharpe ratio, volatility, drawdown, and turnover,
5. ensure the full process remains free of lookahead bias.

---

## 11. Closing Note

The EDA confirms that the current system is not only technically clean, but also financially coherent.  
The observed relationships across equities, bonds, gold, and commodities provide a strong foundation for realistic portfolio optimization experiments.

The project is ready to transition from **data validation** to **model development and backtesting**.
