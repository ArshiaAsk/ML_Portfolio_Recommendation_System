"""Tests for Yahoo Finance data source."""

from datetime import date

import pandas as pd
import pytest

from portfolio_ml.data_sources.yahoo import YahooFinanceSource


def test_yahoo_source_fetch_daily_prices():
    """Test that YahooFinanceSource returns data with required columns."""
    source = YahooFinanceSource()
    symbols = ["SPY"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)
    
    df = source.fetch_daily_prices(symbols, start, end)
    
    assert not df.empty
    assert "date" in df.columns
    assert "symbol" in df.columns
    assert "adj_close" in df.columns
    assert "volume" in df.columns
    assert "source" in df.columns
    assert "ingested_at" in df.columns
    assert all(df["symbol"] == "SPY")
    assert all(df["source"] == "yfinance")


def test_yahoo_source_multiple_symbols():
    """Test fetching multiple symbols."""
    source = YahooFinanceSource()
    symbols = ["SPY", "QQQ"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)
    
    df = source.fetch_daily_prices(symbols, start, end)
    
    assert not df.empty
    assert set(df["symbol"].unique()) == {"SPY", "QQQ"}


def test_yahoo_source_invalid_symbol_graceful():
    """Test that invalid symbols log warnings but don't crash."""
    source = YahooFinanceSource()
    # Mix valid and invalid
    symbols = ["SPY", "INVALID_TICKER_XYZ123"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)
    
    # Should return data for SPY but warn about invalid ticker
    df = source.fetch_daily_prices(symbols, start, end)
    assert not df.empty
    assert "SPY" in df["symbol"].unique()


def test_yahoo_source_all_invalid_raises():
    """Test that all invalid symbols raises RuntimeError."""
    source = YahooFinanceSource()
    symbols = ["INVALID_TICKER_XYZ123", "ANOTHER_INVALID_ABC999"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 10)
    
    with pytest.raises(RuntimeError, match="Failed to fetch data for all requested symbols"):
        source.fetch_daily_prices(symbols, start, end)


def test_yahoo_source_date_column_is_date():
    """Test that the date column contains date objects, not timestamps."""
    source = YahooFinanceSource()
    symbols = ["SPY"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 5)
    
    df = source.fetch_daily_prices(symbols, start, end)
    
    assert not df.empty
    # Check that all values in date column are date objects
    sample_date = df["date"].iloc[0]
    assert isinstance(sample_date, date)
