"""Path helpers for the data lake directory layout."""

from __future__ import annotations

from pathlib import Path


def raw_parquet_path(
    base_dir: str,
    source: str,
    interval: str,
    symbol: str,
    year: int,
) -> Path:
    """Return the full path for a raw Parquet file.

    Layout::

        {base_dir}/source={source}/interval={interval}/symbol={symbol}/year={year}/prices.parquet

    Args:
        base_dir: Root raw data directory (e.g. ``"data/raw"``).
        source: Data source name (e.g. ``"yfinance"``).
        interval: Data interval (e.g. ``"1d"``).
        symbol: Ticker symbol (e.g. ``"SPY"``).
        year: Calendar year (e.g. ``2024``).

    Returns:
        :class:`pathlib.Path` to the Parquet file.
    """
    return (
        Path(base_dir)
        / f"source={source}"
        / f"interval={interval}"
        / f"symbol={symbol}"
        / f"year={year}"
        / "prices.parquet"
    )


def processed_parquet_path(base_dir: str, year: int) -> Path:
    """Return the full path for a processed daily prices Parquet file.

    Layout::

        {base_dir}/daily_prices/year={year}/daily_prices.parquet

    Args:
        base_dir: Root processed data directory.
        year: Calendar year.

    Returns:
        :class:`pathlib.Path` to the Parquet file.
    """
    return Path(base_dir) / "daily_prices" / f"year={year}" / "daily_prices.parquet"


def features_parquet_path(base_dir: str, year: int) -> Path:
    """Return the full path for a feature table Parquet file.

    Layout::

        {base_dir}/features/year={year}/asset_daily_features.parquet

    Args:
        base_dir: Root marts data directory.
        year: Calendar year.

    Returns:
        :class:`pathlib.Path` to the Parquet file.
    """
    return Path(base_dir) / "features" / f"year={year}" / "asset_daily_features.parquet"


def validation_report_path(base_dir: str, table_name: str, timestamp: str) -> Path:
    """Return path for a validation report JSON file.

    Args:
        base_dir: Root validation reports directory.
        table_name: Name of the table being validated.
        timestamp: ISO timestamp string used as filename suffix.

    Returns:
        :class:`pathlib.Path` to the JSON report file.
    """
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    return Path(base_dir) / f"{table_name}_{safe_ts}.json"
