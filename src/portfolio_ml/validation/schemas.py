"""Pandera data contracts for raw, clean, and feature tables."""

from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# ---------------------------------------------------------------------------
# Raw Daily Prices Schema
# ---------------------------------------------------------------------------

RawDailyPricesSchema = DataFrameSchema(
    columns={
        "date": Column(
            pa.Object,
            nullable=False,
            description="Trading date (Python date object)",
        ),
        "symbol": Column(
            pa.String,
            nullable=False,
            description="Ticker symbol",
        ),
        "open": Column(
            pa.Float,
            checks=Check.greater_than(0),
            nullable=True,
            description="Opening price",
        ),
        "high": Column(
            pa.Float,
            checks=Check.greater_than(0),
            nullable=True,
            description="Intraday high price",
        ),
        "low": Column(
            pa.Float,
            checks=Check.greater_than(0),
            nullable=True,
            description="Intraday low price",
        ),
        "close": Column(
            pa.Float,
            checks=Check.greater_than(0),
            nullable=True,
            description="Closing price",
        ),
        "adj_close": Column(
            pa.Float,
            checks=Check.greater_than(0),
            nullable=False,
            description="Adjusted closing price",
        ),
        "volume": Column(
            pa.Float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Trading volume",
        ),
        "dividends": Column(
            pa.Float,
            nullable=True,
            description="Dividend per share",
        ),
        "stock_splits": Column(
            pa.Float,
            nullable=True,
            description="Stock split ratio",
        ),
        "source": Column(
            pa.String,
            nullable=False,
            description="Data source identifier",
        ),
        "ingested_at": Column(
            nullable=False,
            description="UTC ingestion timestamp",
        ),
    },
    checks=[
        # OHLC consistency: high >= low
        Check(
            lambda df: (df["high"].isna() | df["low"].isna() | (df["high"] >= df["low"])),
            element_wise=False,
            error="high must be >= low",
        ),
        # OHLC consistency: high >= open
        Check(
            lambda df: (df["high"].isna() | df["open"].isna() | (df["high"] >= df["open"])),
            element_wise=False,
            error="high must be >= open",
        ),
        # OHLC consistency: high >= close
        Check(
            lambda df: (df["high"].isna() | df["close"].isna() | (df["high"] >= df["close"])),
            element_wise=False,
            error="high must be >= close",
        ),
        # OHLC consistency: low <= open
        Check(
            lambda df: (df["low"].isna() | df["open"].isna() | (df["low"] <= df["open"])),
            element_wise=False,
            error="low must be <= open",
        ),
        # OHLC consistency: low <= close
        Check(
            lambda df: (df["low"].isna() | df["close"].isna() | (df["low"] <= df["close"])),
            element_wise=False,
            error="low must be <= close",
        ),
    ],
    name="RawDailyPrices",
    coerce=False,
)


# ---------------------------------------------------------------------------
# Clean Daily Prices Schema
# ---------------------------------------------------------------------------

CleanDailyPricesSchema = DataFrameSchema(
    columns={
        "date": Column(
            pa.Object,
            nullable=False,
            description="Trading date",
        ),
        "symbol": Column(
            pa.String,
            nullable=False,
            description="Ticker symbol",
        ),
        "open": Column(pa.Float, nullable=True),
        "high": Column(pa.Float, nullable=True),
        "low": Column(pa.Float, nullable=True),
        "close": Column(pa.Float, nullable=True),
        "adj_close": Column(
            pa.Float,
            checks=Check.greater_than(0),
            nullable=False,
        ),
        "volume": Column(
            pa.Float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "return_1d": Column(
            pa.Float,
            nullable=True,
            description="Daily simple return (null for first row per symbol)",
        ),
        "log_return_1d": Column(
            pa.Float,
            nullable=True,
            description="Daily log return (null for first row per symbol)",
        ),
        "is_trading_day": Column(
            pa.Bool,
            nullable=False,
        ),
        "data_quality_flag": Column(
            pa.String,
            nullable=False,
        ),
        "created_at": Column(
            nullable=False,
            description="UTC timestamp of record creation",
        ),
    },
    name="CleanDailyPrices",
    coerce=False,
)


# ---------------------------------------------------------------------------
# Asset Daily Features Schema
# ---------------------------------------------------------------------------

AssetDailyFeaturesSchema = DataFrameSchema(
    columns={
        "date": Column(pa.Object, nullable=False),
        "symbol": Column(pa.String, nullable=False),
        "return_1d": Column(pa.Float, nullable=True),
        "log_return_1d": Column(pa.Float, nullable=True),
        "return_5d": Column(pa.Float, nullable=True),
        "return_21d": Column(pa.Float, nullable=True),
        "return_63d": Column(pa.Float, nullable=True),
        "volatility_21d": Column(pa.Float, nullable=True),
        "volatility_63d": Column(pa.Float, nullable=True),
        "momentum_21d": Column(pa.Float, nullable=True),
        "momentum_63d": Column(pa.Float, nullable=True),
        "drawdown": Column(pa.Float, nullable=True),
        "rolling_volume_21d": Column(pa.Float, nullable=True),
        "price_to_ma_21": Column(pa.Float, nullable=True),
        "price_to_ma_63": Column(pa.Float, nullable=True),
        "target_return_1d": Column(pa.Float, nullable=True),
        "target_return_5d": Column(pa.Float, nullable=True),
        "created_at": Column(nullable=False),
    },
    name="AssetDailyFeatures",
    coerce=False,
)
