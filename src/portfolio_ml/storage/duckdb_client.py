"""DuckDB client for querying Parquet data lake as SQL views."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

_VIEW_DEFINITIONS: dict[str, str] = {
    "raw_daily_prices": "SELECT * FROM read_parquet('{raw_glob}', hive_partitioning=true)",
    "clean_daily_prices": "SELECT * FROM read_parquet('{processed_glob}')",
    "asset_daily_features": "SELECT * FROM read_parquet('{features_glob}')",
}


class DuckDBClient:
    """Thin wrapper around a DuckDB connection for the portfolio data lake.

    Args:
        db_path: Path to the DuckDB database file. Defaults to in-memory.
        raw_data_dir: Path to the raw data lake root.
        processed_data_dir: Path to the processed data root.
        marts_data_dir: Path to the feature marts root.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        marts_data_dir: str = "data/marts",
    ) -> None:
        self.db_path = db_path
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.marts_data_dir = Path(marts_data_dir)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> "DuckDBClient":
        """Open the DuckDB connection.

        Returns:
            Self, for chaining.
        """
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        logger.info("Connected to DuckDB at '%s'", self.db_path)
        return self

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckDBClient":
        return self.connect()

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBClient is not connected. Call connect() first.")
        return self._conn

    def create_views(self) -> None:
        """Create or replace SQL views over the Parquet data lake.

        Views are created only for directories that actually contain
        Parquet files; otherwise a warning is logged and the view is
        skipped to avoid DuckDB errors on empty globs.
        """
        raw_glob = str(self.raw_data_dir / "**" / "*.parquet")
        processed_glob = str(self.processed_data_dir / "**" / "*.parquet")
        features_glob = str(self.marts_data_dir / "**" / "*.parquet")

        view_configs = {
            "raw_daily_prices": (self.raw_data_dir, raw_glob, True),
            "clean_daily_prices": (self.processed_data_dir, processed_glob, False),
            "asset_daily_features": (self.marts_data_dir, features_glob, False),
        }

        for view_name, (data_dir, glob_path, hive) in view_configs.items():
            if not data_dir.exists() or not list(data_dir.rglob("*.parquet")):
                logger.warning(
                    "Skipping view '%s': no Parquet files found under '%s'",
                    view_name,
                    data_dir,
                )
                continue

            if hive:
                sql = (
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true)"
                )
            else:
                sql = (
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{glob_path}')"
                )

            self.conn.execute(sql)
            logger.info("Created DuckDB view '%s'", view_name)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return the result as a DataFrame.

        Args:
            sql: SQL query string.

        Returns:
            Query result as a :class:`pandas.DataFrame`.
        """
        return self.conn.execute(sql).df()

    def summary_stats(self) -> pd.DataFrame:
        """Return per-symbol row counts and date ranges from clean_daily_prices.

        Returns:
            Summary DataFrame or empty DataFrame if view not available.
        """
        try:
            return self.query(
                """
                SELECT
                    symbol,
                    COUNT(*) AS rows,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date
                FROM clean_daily_prices
                GROUP BY symbol
                ORDER BY symbol
                """
            )
        except Exception as exc:
            logger.warning("Could not compute summary stats: %s", exc)
            return pd.DataFrame()

    def avg_returns(self) -> pd.DataFrame:
        """Return per-symbol average return and volatility from clean_daily_prices.

        Returns:
            Analytics DataFrame or empty DataFrame if view not available.
        """
        try:
            return self.query(
                """
                SELECT
                    symbol,
                    AVG(return_1d)    AS avg_daily_return,
                    STDDEV(return_1d) AS daily_volatility,
                    COUNT(*)          AS observations
                FROM clean_daily_prices
                GROUP BY symbol
                ORDER BY symbol
                """
            )
        except Exception as exc:
            logger.warning("Could not compute avg returns: %s", exc)
            return pd.DataFrame()
