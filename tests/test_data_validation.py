"""Tests for Pandera validation schemas and checks."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest
import pandera as pa

from portfolio_ml.validation.schemas import (
    RawDailyPricesSchema,
    CleanDailyPricesSchema,
    AssetDailyFeaturesSchema,
)
from portfolio_ml.validation.checks import (
    validate_raw_prices,
    validate_clean_prices,
    validate_asset_features,
)


def test_raw_daily_prices_schema_valid():
    """Test that valid raw data passes schema validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "symbol": ["SPY", "SPY"],
        "open": [450.0, 451.0],
        "high": [452.0, 453.0],
        "low": [449.0, 450.0],
        "close": [451.0, 452.0],
        "adj_close": [451.0, 452.0],
        "volume": [1000000.0, 1100000.0],
        "dividends": [0.0, 0.0],
        "stock_splits": [0.0, 0.0],
        "source": ["yfinance", "yfinance"],
        "ingested_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
    })
    validated = RawDailyPricesSchema.validate(df)
    assert len(validated) == 2


def test_raw_daily_prices_schema_negative_adj_close():
    """Test that negative adj_close fails validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1)],
        "symbol": ["SPY"],
        "open": [450.0],
        "high": [452.0],
        "low": [449.0],
        "close": [451.0],
        "adj_close": [-1.0],  # Invalid
        "volume": [1000000.0],
        "dividends": [0.0],
        "stock_splits": [0.0],
        "source": ["yfinance"],
        "ingested_at": [datetime.now(timezone.utc)],
    })
    with pytest.raises(pa.errors.SchemaError):
        RawDailyPricesSchema.validate(df)


def test_raw_daily_prices_schema_negative_volume():
    """Test that negative volume fails validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1)],
        "symbol": ["SPY"],
        "open": [450.0],
        "high": [452.0],
        "low": [449.0],
        "close": [451.0],
        "adj_close": [451.0],
        "volume": [-1000.0],  # Invalid
        "dividends": [0.0],
        "stock_splits": [0.0],
        "source": ["yfinance"],
        "ingested_at": [datetime.now(timezone.utc)],
    })
    with pytest.raises(pa.errors.SchemaError):
        RawDailyPricesSchema.validate(df)


def test_raw_daily_prices_schema_ohlc_inconsistency():
    """Test that high < low fails validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1)],
        "symbol": ["SPY"],
        "open": [450.0],
        "high": [448.0],  # Invalid: high < low
        "low": [449.0],
        "close": [451.0],
        "adj_close": [451.0],
        "volume": [1000000.0],
        "dividends": [0.0],
        "stock_splits": [0.0],
        "source": ["yfinance"],
        "ingested_at": [datetime.now(timezone.utc)],
    })
    with pytest.raises(pa.errors.SchemaError):
        RawDailyPricesSchema.validate(df)


def test_validate_raw_prices_duplicate_keys(tmp_path):
    """Test that duplicate symbol/date/source fails validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 1)],  # Duplicate
        "symbol": ["SPY", "SPY"],
        "open": [450.0, 450.0],
        "high": [452.0, 452.0],
        "low": [449.0, 449.0],
        "close": [451.0, 451.0],
        "adj_close": [451.0, 451.0],
        "volume": [1000000.0, 1000000.0],
        "dividends": [0.0, 0.0],
        "stock_splits": [0.0, 0.0],
        "source": ["yfinance", "yfinance"],
        "ingested_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_prices(df, reports_dir=str(tmp_path))


def test_clean_daily_prices_schema_valid():
    """Test that valid clean data passes schema validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "symbol": ["SPY", "SPY"],
        "open": [450.0, 451.0],
        "high": [452.0, 453.0],
        "low": [449.0, 450.0],
        "close": [451.0, 452.0],
        "adj_close": [451.0, 452.0],
        "volume": [1000000.0, 1100000.0],
        "return_1d": [None, 0.002217],
        "log_return_1d": [None, 0.002215],
        "is_trading_day": [True, True],
        "data_quality_flag": ["ok", "ok"],
        "created_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
    })
    validated = CleanDailyPricesSchema.validate(df)
    assert len(validated) == 2


def test_asset_daily_features_schema_valid():
    """Test that valid feature data passes schema validation."""
    df = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "symbol": ["SPY", "SPY"],
        "return_1d": [None, 0.002217],
        "log_return_1d": [None, 0.002215],
        "return_5d": [None, None],
        "return_21d": [None, None],
        "return_63d": [None, None],
        "volatility_21d": [None, None],
        "volatility_63d": [None, None],
        "momentum_21d": [None, None],
        "momentum_63d": [None, None],
        "drawdown": [0.0, 0.0],
        "rolling_volume_21d": [1000000.0, 1050000.0],
        "price_to_ma_21": [0.0, 0.001],
        "price_to_ma_63": [0.0, 0.001],
        "target_return_1d": [0.002217, None],
        "target_return_5d": [None, None],
        "created_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
    })
    validated = AssetDailyFeaturesSchema.validate(df)
    assert len(validated) == 2
