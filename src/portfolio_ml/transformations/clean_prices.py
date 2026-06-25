"""Clean and standardize raw daily prices into the processed layer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from portfolio_ml.utils.dates import now_utc
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_INPUT_COLS = [
    "date", "symbol", "open", "high", "low", "close", "adj_close", "volume",
]

_OUTPUT_COLS = [
    "date", "symbol", "open", "high", "low", "close", "adj_close", "volume",
    "return_1d", "log_return_1d", "is_trading_day", "data_quality_flag", "created_at",
]


def clean_daily_prices(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Transform raw OHLCV data into a clean analytical table.

    Steps performed:
    1. Normalize column names to snake_case.
    2. Convert ``date`` to :class:`datetime.date`.
    3. Cast numeric columns.
    4. Sort by ``symbol``, then ``date``.
    5. Remove exact duplicate rows.
    6. Enforce one row per ``symbol/date`` (keep last on conflict).
    7. Compute ``return_1d`` and ``log_return_1d`` per symbol without
       lookahead bias (values are null for the first row per symbol).
    8. Flag rows with missing ``adj_close`` values.
    9. Add ``is_trading_day = True`` and ``created_at`` metadata.

    Args:
        raw_prices: Raw DataFrame from the ingestion layer.

    Returns:
        Clean daily prices DataFrame.

    Raises:
        ValueError: If required input columns are missing.
    """
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in raw_prices.columns]
    if missing:
        raise ValueError(
            f"clean_daily_prices: missing required input columns: {missing}"
        )

    df = raw_prices.copy()

    # --- Normalize column names -------------------------------------------
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # --- Normalize date column -------------------------------------------
    if not pd.api.types.is_object_dtype(df["date"]) or not isinstance(
        df["date"].dropna().iloc[0] if len(df) > 0 else None,
        __import__("datetime").date,
    ):
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # --- Cast numeric columns --------------------------------------------
    for col in ("open", "high", "low", "close", "adj_close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # --- Sort -------------------------------------------------------------
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # --- Remove exact duplicates -----------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d exact duplicate rows.", dropped)

    # --- Enforce one row per symbol/date (keep last on conflict) ----------
    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped %d conflicting duplicate rows (kept last) for symbol/date.", dropped
        )

    # --- Compute daily returns per symbol --------------------------------
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df["return_1d"] = df.groupby("symbol")["adj_close"].transform(
        lambda s: s / s.shift(1) - 1
    )
    df["log_return_1d"] = df.groupby("symbol")["adj_close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    # --- Data quality flag -----------------------------------------------
    df["data_quality_flag"] = "ok"
    missing_adj_close = df["adj_close"].isna()
    if missing_adj_close.any():
        df.loc[missing_adj_close, "data_quality_flag"] = "missing_adj_close"
        logger.warning(
            "%d rows flagged as 'missing_adj_close'.", missing_adj_close.sum()
        )

    # --- Metadata --------------------------------------------------------
    df["is_trading_day"] = True
    df["created_at"] = now_utc()

    # --- Select output columns -------------------------------------------
    available = [c for c in _OUTPUT_COLS if c in df.columns]
    df = df[available].reset_index(drop=True)

    logger.info(
        "clean_daily_prices: produced %d rows for %d symbol(s).",
        len(df),
        df["symbol"].nunique(),
    )
    return df
