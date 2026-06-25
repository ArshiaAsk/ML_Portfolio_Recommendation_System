"""Tests for clean_prices transformation."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest
import numpy as np

from portfolio_ml.transformations.clean_prices import clean_daily_prices


def test_clean_prices_basic():
    """Test that clean_daily_prices produces expected output columns."""
    raw = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "symbol": ["SPY", "SPY"],
        "open": [450.0, 451.0],
        "high": [452.0, 453.0],
        "low": [449.0, 450.0],
        "close": [451.0, 452.0],
        "adj_close": [451.0, 452.0],
        "volume": [1000000, 1100000],
        "source": ["yfinance", "yfinance"],
        "ingested_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
    })
    
    clean = clean_daily_prices(raw)
    
    assert "return_1d" in clean.columns
    assert "log_return_1d" in clean.columns
    assert "is_trading_day" in clean.columns
    assert "data_quality_flag" in clean.columns
    assert "created_at" in clean.columns
    assert len(clean) == 2


def test_clean_prices_return_computation():
    """Test that return_1d and log_return_1d are computed correctly."""
    raw = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "symbol": ["SPY", "SPY", "SPY"],
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.0, 101.0, 102.0],
        "adj_close": [100.0, 101.0, 102.0],
        "volume": [1000000, 1100000, 1200000],
        "source": ["yfinance", "yfinance", "yfinance"],
        "ingested_at": [datetime.now(timezone.utc)] * 3,
    })
    
    clean = clean_daily_prices(raw)
    
    # First row should have null return
    assert pd.isna(clean.loc[0, "return_1d"])
    assert pd.isna(clean.loc[0, "log_return_1d"])
    
    # Second row: (101 / 100) - 1 = 0.01
    assert abs(clean.loc[1, "return_1d"] - 0.01) < 1e-6
    # log(101 / 100) ≈ 0.00995
    assert abs(clean.loc[1, "log_return_1d"] - np.log(101 / 100)) < 1e-6
    
    # Third row: (102 / 101) - 1 ≈ 0.0099
    expected_ret = (102 / 101) - 1
    assert abs(clean.loc[2, "return_1d"] - expected_ret) < 1e-6


def test_clean_prices_sorted_by_symbol_date():
    """Test that output is sorted by symbol, then date."""
    raw = pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)],
        "symbol": ["QQQ", "QQQ", "SPY", "SPY"],
        "open": [300.0, 299.0, 450.0, 449.0],
        "high": [301.0, 300.0, 451.0, 450.0],
        "low": [299.0, 298.0, 449.0, 448.0],
        "close": [300.0, 299.0, 450.0, 449.0],
        "adj_close": [300.0, 299.0, 450.0, 449.0],
        "volume": [1000000] * 4,
        "source": ["yfinance"] * 4,
        "ingested_at": [datetime.now(timezone.utc)] * 4,
    })
    
    clean = clean_daily_prices(raw)
    
    # Should be sorted: QQQ 1/1, QQQ 1/2, SPY 1/1, SPY 1/2
    assert list(clean["symbol"]) == ["QQQ", "QQQ", "SPY", "SPY"]
    assert list(clean["date"]) == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 1),
        date(2024, 1, 2),
    ]


def test_clean_prices_removes_duplicates():
    """Test that exact duplicate rows are removed."""
    raw = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 1)],  # Duplicate
        "symbol": ["SPY", "SPY"],
        "open": [450.0, 450.0],
        "high": [452.0, 452.0],
        "low": [449.0, 449.0],
        "close": [451.0, 451.0],
        "adj_close": [451.0, 451.0],
        "volume": [1000000, 1000000],
        "source": ["yfinance", "yfinance"],
        "ingested_at": [datetime.now(timezone.utc)] * 2,
    })
    
    clean = clean_daily_prices(raw)
    
    # Should only have 1 row after deduplication
    assert len(clean) == 1


def test_clean_prices_handles_multiple_symbols():
    """Test that returns are computed per symbol independently."""
    raw = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
        "symbol": ["SPY", "SPY", "QQQ", "QQQ"],
        "open": [100.0, 101.0, 200.0, 202.0],
        "high": [101.0, 102.0, 201.0, 203.0],
        "low": [99.0, 100.0, 199.0, 201.0],
        "close": [100.0, 101.0, 200.0, 202.0],
        "adj_close": [100.0, 101.0, 200.0, 202.0],
        "volume": [1000000] * 4,
        "source": ["yfinance"] * 4,
        "ingested_at": [datetime.now(timezone.utc)] * 4,
    })
    
    clean = clean_daily_prices(raw)
    
    # First row of each symbol should have null return
    spy_rows = clean[clean["symbol"] == "SPY"]
    qqq_rows = clean[clean["symbol"] == "QQQ"]
    
    assert pd.isna(spy_rows.iloc[0]["return_1d"])
    assert pd.isna(qqq_rows.iloc[0]["return_1d"])
    
    # Second row of each symbol should have computed return
    assert abs(spy_rows.iloc[1]["return_1d"] - 0.01) < 1e-6
    assert abs(qqq_rows.iloc[1]["return_1d"] - 0.01) < 1e-6


def test_clean_prices_missing_columns():
    """Test that missing required columns raises ValueError."""
    raw = pd.DataFrame({
        "date": [date(2024, 1, 1)],
        "symbol": ["SPY"],
        # Missing adj_close, volume, etc.
    })
    
    with pytest.raises(ValueError, match="missing required input columns"):
        clean_daily_prices(raw)
