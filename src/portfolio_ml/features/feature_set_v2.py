"""Feature Set v2 — leakage-safe, richer feature engineering for ranking models.

Feature groups
--------------
A. Cross-sectional features
   - z-score of returns vs universe, relative momentum, cross-sectional ranks

B. Time-series momentum / risk features
   - multi-window trailing returns, rolling volatility, drawdown, MA ratios

C. Regime / risk-state features
   - market trailing return/vol, volatility regime flag, drawdown regime flag

D. Macro placeholder
   - VIX proxy from cross-asset vol (real macro can be added via ``inject_macro``).

Leakage rules enforced
-----------------------
- All rolling features use only data at or before the current row.
- Cross-sectional aggregates are computed over all assets available *at the
  same date*, never mixing future dates.
- Feature matrix is sorted by (symbol, date) before returning.
- Forward labels are NOT included here; use ``features.targets`` for those.

Usage
-----
>>> from portfolio_ml.features.feature_set_v2 import build_feature_set_v2
>>> prices_wide = ...  # pd.DataFrame, DatetimeIndex, columns = asset tickers
>>> features_long = build_feature_set_v2(prices_wide)
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_WINDOWS = {
    "return_windows": [5, 21, 63, 126],
    "vol_windows": [21, 63],
    "corr_window": 63,
    "beta_window": 63,
    "ma_window": 63,
    "cs_rank_windows": [21, 63],
}


def build_feature_set_v2(
    prices_wide: pd.DataFrame,
    benchmark_col: str | None = None,
    feature_windows: dict | None = None,
) -> pd.DataFrame:
    """Build feature set v2 from a wide-format price matrix.

    Args:
        prices_wide: DataFrame with DatetimeIndex and one column per asset
            (adjusted-close prices).  Must be sorted by date.
        benchmark_col: Column name to use as market benchmark.  If None and
            ``SPY`` is present it will be used automatically; otherwise the
            equal-weighted average is used.
        feature_windows: Override default window configuration. Partial
            override is supported.

    Returns:
        Long-format DataFrame with columns:
            date, symbol, <feature columns>

        Sorted by (date, symbol).  No target columns are included.

    Notes:
        - Minimum history required before meaningful features exist: 126 rows.
        - Rows with insufficient history will have NaN for rolling features.
        - NaN filling is intentionally NOT done here; callers can impute.
    """
    if prices_wide.empty:
        raise ValueError("build_feature_set_v2: prices_wide is empty.")
    if not isinstance(prices_wide.index, pd.DatetimeIndex):
        raise ValueError("build_feature_set_v2: prices_wide must have a DatetimeIndex.")

    windows = {**FEATURE_WINDOWS, **(feature_windows or {})}

    prices_wide = prices_wide.sort_index()

    # Pick benchmark
    benchmark = _pick_benchmark(prices_wide, benchmark_col)

    logger.info(
        "build_feature_set_v2: %d assets, %d dates, benchmark=%s",
        prices_wide.shape[1],
        len(prices_wide),
        benchmark.name if hasattr(benchmark, "name") else "equal-weight",
    )

    # Compute per-asset features (long format)
    per_asset_frames = []
    for symbol in prices_wide.columns:
        sym_prices = prices_wide[symbol].dropna()
        if len(sym_prices) < 5:
            logger.warning("build_feature_set_v2: %s has fewer than 5 non-NaN prices, skipping.", symbol)
            continue
        sym_df = _compute_asset_features(symbol, sym_prices, benchmark, windows)
        per_asset_frames.append(sym_df)

    if not per_asset_frames:
        raise ValueError("build_feature_set_v2: no assets had sufficient data.")

    long_df = pd.concat(per_asset_frames, ignore_index=True)

    # Compute cross-sectional features
    long_df = _add_cross_sectional_features(long_df, windows)

    long_df = long_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    logger.info(
        "build_feature_set_v2: produced %d rows, %d features for %d assets.",
        len(long_df),
        len([c for c in long_df.columns if c not in ("date", "symbol")]),
        long_df["symbol"].nunique(),
    )
    return long_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excludes date and symbol)."""
    return [c for c in df.columns if c not in ("date", "symbol")]


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print a summary of the feature matrix.

    Args:
        df: Feature DataFrame (long format).

    Returns:
        Summary DataFrame with feature name, dtype, non-null count, null pct,
        mean, std.
    """
    feat_cols = get_feature_columns(df)
    summary_rows = []
    for col in feat_cols:
        series = df[col]
        non_null = series.notna().sum()
        summary_rows.append({
            "feature": col,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "null_pct": round(100.0 * (1 - non_null / len(series)), 2),
            "mean": round(series.mean(), 6) if pd.api.types.is_numeric_dtype(series) else None,
            "std": round(series.std(), 6) if pd.api.types.is_numeric_dtype(series) else None,
        })
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Per-asset feature computation (time-series)
# ---------------------------------------------------------------------------


def _compute_asset_features(
    symbol: str,
    prices: pd.Series,
    benchmark: pd.Series,
    windows: dict,
) -> pd.DataFrame:
    """Compute all time-series features for a single asset.

    Args:
        symbol: Ticker symbol.
        prices: Adj-close prices indexed by date (no NaN).
        benchmark: Benchmark price series (may be shorter/longer than prices).
        windows: Window configuration dict.

    Returns:
        DataFrame with columns: date, symbol, <features>
    """
    df = pd.DataFrame({"adj_close": prices})
    df.index.name = "date"
    df = df.reset_index()
    df["symbol"] = symbol

    adj = df["adj_close"]

    # --- B. Time-series momentum / risk features ---

    for w in windows["return_windows"]:
        col = f"ret_{w}d"
        df[col] = adj / adj.shift(w) - 1

    for w in windows["vol_windows"]:
        log_ret = np.log(adj / adj.shift(1))
        df[f"vol_{w}d"] = log_ret.rolling(window=w, min_periods=max(5, w // 4)).std() * np.sqrt(252)

    # Rolling drawdown (from 252-day rolling high)
    rolling_max = adj.rolling(window=252, min_periods=1).max()
    df["rolling_drawdown"] = adj / rolling_max - 1

    # Distance from rolling 252-day high
    df["dist_from_high_252d"] = df["rolling_drawdown"]

    # MA ratio (price / SMA_63 - 1)
    sma_63 = adj.rolling(window=windows["ma_window"], min_periods=max(5, windows["ma_window"] // 4)).mean()
    df["price_to_ma_63"] = adj / sma_63 - 1

    # Volatility change (vol_21 / vol_63 - 1)
    if "vol_21d" in df.columns and "vol_63d" in df.columns:
        df["vol_change"] = (
            df["vol_21d"] / df["vol_63d"].replace(0, np.nan) - 1
        )

    # --- C. Benchmark-relative and regime features ---

    # Align benchmark to asset dates (benchmark has DatetimeIndex, df has date column)
    bm_aligned = benchmark.reindex(df["date"].values).ffill()
    bm_aligned.index = df.index  # Reset to integer index to match df

    bm_ret = bm_aligned / bm_aligned.shift(1) - 1
    df["bm_ret_21d"] = bm_aligned / bm_aligned.shift(21) - 1
    df["bm_ret_63d"] = bm_aligned / bm_aligned.shift(63) - 1

    bm_log_ret = np.log(bm_aligned / bm_aligned.shift(1))
    df["bm_vol_21d"] = bm_log_ret.rolling(window=21, min_periods=5).std() * np.sqrt(252)

    # Volatility regime flag (asset vol_21d vs trailing 63-day quantiles)
    if "vol_21d" in df.columns:
        df["vol_regime_flag"] = (
            df["vol_21d"] > df["vol_21d"].rolling(window=63, min_periods=21).quantile(0.75)
        ).astype(float)

    # Drawdown regime flag (drawdown < -10%)
    df["drawdown_regime_flag"] = (df["rolling_drawdown"] < -0.10).astype(float)

    # --- Rolling beta and correlation to benchmark ---
    asset_log_ret = np.log(adj / adj.shift(1))
    bm_log_ret_series = bm_log_ret.values if hasattr(bm_log_ret, "values") else bm_log_ret

    bm_series_aligned = pd.Series(bm_log_ret_series, index=df.index)
    corr_window = windows["corr_window"]

    df["rolling_corr_bm"] = asset_log_ret.rolling(window=corr_window, min_periods=corr_window // 4).corr(bm_series_aligned)

    # Rolling beta = cov(asset, bm) / var(bm)
    joint_cov = asset_log_ret.rolling(window=corr_window, min_periods=corr_window // 4).cov(bm_series_aligned)
    bm_var = bm_series_aligned.rolling(window=corr_window, min_periods=corr_window // 4).var()
    df["rolling_beta_bm"] = joint_cov / bm_var.replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Cross-sectional features
# ---------------------------------------------------------------------------


def _add_cross_sectional_features(df: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Add cross-sectional (per-date-across-assets) features.

    These are computed on the full long DataFrame grouped by date, so they
    capture how each asset ranks within its contemporaneous peer group.

    Args:
        df: Long-format feature DataFrame with date and symbol columns.
        windows: Window configuration.

    Returns:
        df with additional cross-sectional columns.
    """
    cs_rank_windows = windows["cs_rank_windows"]  # e.g. [21, 63]

    # Rank columns to create cross-sectionally
    for w in cs_rank_windows:
        ret_col = f"ret_{w}d"
        if ret_col not in df.columns:
            continue

        rank_col = f"cs_rank_ret_{w}d"
        zscore_col = f"cs_zscore_ret_{w}d"
        rel_mom_col = f"cs_rel_mom_{w}d"

        grp = df.groupby("date")[ret_col]

        # Cross-sectional percentile rank [0, 1]
        df[rank_col] = grp.rank(pct=True, na_option="keep")

        # Cross-sectional z-score
        cs_mean = grp.transform("mean")
        cs_std = grp.transform("std")
        df[zscore_col] = (df[ret_col] - cs_mean) / cs_std.replace(0, np.nan)

        # Relative momentum (asset return - universe median)
        cs_median = grp.transform("median")
        df[rel_mom_col] = df[ret_col] - cs_median

    # Cross-sectional relative volatility (vol_21d vs universe)
    vol_col = "vol_21d"
    if vol_col in df.columns:
        grp_vol = df.groupby("date")[vol_col]
        cs_mean_vol = grp_vol.transform("mean")
        cs_std_vol = grp_vol.transform("std")
        df["cs_rel_vol_21d"] = (df[vol_col] - cs_mean_vol) / cs_std_vol.replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Benchmark selection helper
# ---------------------------------------------------------------------------


def _pick_benchmark(prices_wide: pd.DataFrame, benchmark_col: str | None) -> pd.Series:
    """Select or construct a benchmark series.

    Priority:
    1. Explicitly requested column (if in prices_wide).
    2. ``SPY`` column (if present).
    3. Equal-weighted average of all assets.

    Returns:
        pd.Series with DatetimeIndex named "benchmark".
    """
    if benchmark_col and benchmark_col in prices_wide.columns:
        bm = prices_wide[benchmark_col].copy()
        bm.name = benchmark_col
        return bm

    if "SPY" in prices_wide.columns:
        bm = prices_wide["SPY"].copy()
        bm.name = "SPY"
        logger.info("_pick_benchmark: using SPY as benchmark.")
        return bm

    bm = prices_wide.mean(axis=1)
    bm.name = "equal_weight_bm"
    logger.info("_pick_benchmark: SPY not found; using equal-weight benchmark.")
    return bm
