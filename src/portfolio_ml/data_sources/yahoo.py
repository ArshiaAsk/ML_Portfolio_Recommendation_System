"""Yahoo Finance implementation of MarketDataSource using yfinance."""

from __future__ import annotations

from datetime import date, timezone
from typing import Sequence

import pandas as pd
import yfinance as yf

from portfolio_ml.data_sources.base import MarketDataSource
from portfolio_ml.utils.dates import now_utc
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "source",
    "ingested_at",
]

_SOURCE_NAME = "yfinance"


class YahooFinanceSource(MarketDataSource):
    """Fetches daily OHLCV data from Yahoo Finance via yfinance.

    Data is fetched for all symbols in a single batch call, then
    reshaped into a long-format normalized DataFrame.
    """

    def fetch_daily_prices(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV prices from Yahoo Finance.

        Args:
            symbols: Ticker symbols (e.g. ``["SPY", "QQQ"]``).
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            Normalized long-format DataFrame.

        Raises:
            RuntimeError: If no data is returned for any symbol.
        """
        symbols_list = list(symbols)
        logger.info(
            "Fetching daily prices for %d symbols from %s to %s via yfinance",
            len(symbols_list),
            start_date,
            end_date,
        )

        ingested_at = now_utc()

        # yfinance end date is exclusive — add one day
        end_date_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        frames: list[pd.DataFrame] = []
        failed_symbols: list[str] = []

        for symbol in symbols_list:
            try:
                ticker = yf.Ticker(symbol)
                raw = ticker.history(
                    start=str(start_date),
                    end=str(end_date_exclusive.date()),
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                )

                if raw.empty:
                    logger.warning("No data returned for symbol '%s'", symbol)
                    failed_symbols.append(symbol)
                    continue

                df = self._normalize_ticker_df(raw, symbol, ingested_at)
                frames.append(df)
                logger.debug(
                    "Fetched %d rows for symbol '%s'", len(df), symbol
                )

            except Exception as exc:
                logger.warning("Failed to fetch data for symbol '%s': %s", symbol, exc)
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(
                "No data obtained for %d symbol(s): %s",
                len(failed_symbols),
                failed_symbols,
            )

        if not frames:
            raise RuntimeError(
                f"Failed to fetch data for all requested symbols: {symbols_list}. "
                "Check network connectivity and ticker validity."
            )

        result = pd.concat(frames, ignore_index=True)
        logger.info(
            "Fetched %d total rows for %d symbol(s) (skipped %d)",
            len(result),
            len(frames),
            len(failed_symbols),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_ticker_df(
        self,
        raw: pd.DataFrame,
        symbol: str,
        ingested_at: pd.Timestamp,
    ) -> pd.DataFrame:
        """Reshape a single-ticker yfinance DataFrame to normalized form.

        Args:
            raw: Raw DataFrame from ``ticker.history()``.
            symbol: Ticker symbol string.
            ingested_at: UTC timestamp of ingestion.

        Returns:
            Normalized DataFrame with required output columns.
        """
        df = raw.copy()

        # Handle MultiIndex columns (can appear when fetching multiple tickers
        # via yf.download; less common for single-ticker but handled defensively)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(part) for part in col if part).lower()
                for col in df.columns
            ]
        else:
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # Reset index so Date becomes a column
        df = df.reset_index()

        # Normalize the date index column name
        date_col = next(
            (c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")),
            None,
        )
        if date_col is None:
            raise ValueError(
                f"Cannot find date column in yfinance output for {symbol}. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.rename(columns={date_col: "date"})

        # Convert timezone-aware timestamps → plain date
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_localize(None).dt.normalize().dt.date
        else:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # Map yfinance column names → standardized names
        column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj close": "adj_close",
            "adj_close": "adj_close",
            "volume": "volume",
            "dividends": "dividends",
            "stock splits": "stock_splits",
            "stock_splits": "stock_splits",
            "capital gains": "capital_gains",
        }
        df = df.rename(columns={c: column_map[c] for c in df.columns if c in column_map})

        # Ensure required price columns exist
        for col in ("adj_close", "dividends", "stock_splits"):
            if col not in df.columns:
                df[col] = 0.0

        # Add metadata columns
        df["symbol"] = symbol
        df["source"] = _SOURCE_NAME
        df["ingested_at"] = ingested_at

        # Cast numeric columns
        for col in ("open", "high", "low", "close", "adj_close", "dividends", "stock_splits"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)

        # Select only required output columns (drop extras like capital_gains)
        missing_cols = [c for c in _REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Normalized DataFrame for '{symbol}' is missing columns: {missing_cols}"
            )

        return df[_REQUIRED_OUTPUT_COLUMNS]
