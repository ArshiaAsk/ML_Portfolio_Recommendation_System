"""Generate the asset daily feature table from clean daily prices.

All features use only current and past data (no lookahead bias).
Target variables (``target_return_1d``, ``target_return_5d``) use future
data intentionally as supervised learning labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_ml.utils.dates import now_utc
from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_INPUT_COLS = ["date", "symbol", "adj_close", "volume", "return_1d", "log_return_1d"]

_OUTPUT_COLS = [
    "date",
    "symbol",
    "return_1d",
    "log_return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    "volatility_21d",
    "volatility_63d",
    "momentum_21d",
    "momentum_63d",
    "drawdown",
    "rolling_volume_21d",
    "price_to_ma_21",
    "price_to_ma_63",
    "target_return_1d",
    "target_return_5d",
    "created_at",
]


def build_asset_daily_features(clean_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling and target features for each asset.

    Features are computed per symbol independently to avoid cross-symbol
    contamination. All rolling lookback features use ``min_periods=1``
    so that rows with insufficient history retain partial values rather
    than being dropped; however, lookback features that genuinely need
    N periods (e.g. return_5d) will remain null until enough history
    is available.

    Args:
        clean_prices: Output of :func:`clean_daily_prices`.

    Returns:
        Feature DataFrame with one row per symbol per date.

    Raises:
        ValueError: If required input columns are missing.
    """
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in clean_prices.columns]
    if missing:
        raise ValueError(
            f"build_asset_daily_features: missing required input columns: {missing}"
        )

    df = clean_prices.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    results: list[pd.DataFrame] = []
    for symbol, grp in df.groupby("symbol", sort=True):
        grp = grp.copy().reset_index(drop=True)
        feat = _compute_symbol_features(grp)
        results.append(feat)

    if not results:
        logger.warning("build_asset_daily_features: no results produced.")
        return pd.DataFrame(columns=_OUTPUT_COLS)

    out = pd.concat(results, ignore_index=True)
    out["created_at"] = now_utc()

    # Reorder output columns
    available = [c for c in _OUTPUT_COLS if c in out.columns]
    out = out[available].reset_index(drop=True)

    logger.info(
        "build_asset_daily_features: produced %d rows for %d symbol(s).",
        len(out),
        out["symbol"].nunique(),
    )
    return out


def _compute_symbol_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a single symbol's time series.

    Args:
        grp: Clean prices DataFrame for one symbol, sorted by date.

    Returns:
        Feature DataFrame for the symbol.
    """
    adj = grp["adj_close"]
    vol = grp["volume"]
    ret = grp["return_1d"]

    # --- Multi-period returns (past only) ---------------------------------
    grp["return_5d"] = adj / adj.shift(5) - 1
    grp["return_21d"] = adj / adj.shift(21) - 1
    grp["return_63d"] = adj / adj.shift(63) - 1

    # --- Volatility (rolling std of daily returns, past only) -------------
    grp["volatility_21d"] = ret.rolling(window=21, min_periods=5).std()
    grp["volatility_63d"] = ret.rolling(window=63, min_periods=10).std()

    # --- Momentum (same as multi-period returns) --------------------------
    grp["momentum_21d"] = grp["return_21d"]
    grp["momentum_63d"] = grp["return_63d"]

    # --- Rolling volume ---------------------------------------------------
    grp["rolling_volume_21d"] = vol.rolling(window=21, min_periods=1).mean()

    # --- Price vs moving average -----------------------------------------
    ma_21 = adj.rolling(window=21, min_periods=1).mean()
    ma_63 = adj.rolling(window=63, min_periods=1).mean()
    grp["price_to_ma_21"] = adj / ma_21 - 1
    grp["price_to_ma_63"] = adj / ma_63 - 1

    # --- Drawdown (relative to rolling historical max) --------------------
    rolling_max = adj.cummax()
    grp["drawdown"] = adj / rolling_max - 1

    # --- Target variables (use future data — labels only) -----------------
    grp["target_return_1d"] = adj.shift(-1) / adj - 1
    grp["target_return_5d"] = adj.shift(-5) / adj - 1

    return grp
