"""Abstract base class for market data sources."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Sequence

import pandas as pd


class MarketDataSource(ABC):
    """Abstract interface for market data providers.

    All concrete implementations must return a normalized DataFrame
    with one row per symbol per trading date.
    """

    @abstractmethod
    def fetch_daily_prices(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV market data for the given symbols.

        Args:
            symbols: Sequence of ticker symbols to fetch.
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame with columns:
                date, symbol, open, high, low, close, adj_close,
                volume, dividends, stock_splits, source, ingested_at

        Raises:
            RuntimeError: If fetching fails for all requested symbols.
        """
        raise NotImplementedError
