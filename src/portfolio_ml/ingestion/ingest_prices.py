"""Orchestrates raw price ingestion: fetch + partition-write raw Parquet."""

from __future__ import annotations

from datetime import date

import pandas as pd

from portfolio_ml.data_sources.base import MarketDataSource
from portfolio_ml.storage.parquet import write_partitioned_parquet
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_raw_prices(
    source: MarketDataSource,
    symbols: list[str],
    start_date: date,
    end_date: date,
    raw_data_dir: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch and persist raw daily prices to the raw data lake layer.

    Prices are partitioned by ``source / interval / symbol / year``.
    Running this function multiple times for the same date range is
    idempotent — existing Parquet files are overwritten in place.

    Args:
        source: A :class:`MarketDataSource` implementation.
        symbols: List of ticker symbols.
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        raw_data_dir: Base raw data directory path.
        interval: Data interval string (default ``"1d"``).

    Returns:
        Combined raw prices DataFrame.
    """
    df = source.fetch_daily_prices(symbols, start_date, end_date)
    logger.info("Saving %d raw price rows to '%s'", len(df), raw_data_dir)

    write_partitioned_parquet(
        df=df,
        base_path=raw_data_dir,
        partition_cols=["source", "interval", "symbol", "year"],
        filename="prices.parquet",
        extra_partition_fn=_add_interval_and_year,
    )
    logger.info("Raw prices saved to '%s'", raw_data_dir)
    return df


def _add_interval_and_year(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Add ``interval`` and ``year`` partition columns to the DataFrame."""
    df = df.copy()
    df["interval"] = interval
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df
