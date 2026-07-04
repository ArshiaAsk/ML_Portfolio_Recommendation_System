"""Explicit target variable definitions for portfolio return prediction.

All targets are computed as *forward* returns, meaning they look into the
future relative to the feature date.  By design this introduces lookahead if
features and targets are combined before the train/test split.  The correct
usage pattern is:

    1.  Compute features on the training window only (no future data present).
    2.  Attach targets to the **training** rows by calling these functions on
        the *full* labelled dataset and then restricting to the train split.
    3.  Never expose target values for test-period dates when fitting any
        scaler or model.

This module does NOT perform the split itself – that is the responsibility of
``WalkForwardSplitter`` in :mod:`portfolio_ml.modeling.walk_forward`.

Example
-------
>>> import pandas as pd
>>> from portfolio_ml.features.targets import compute_all_targets
>>> prices = pd.DataFrame({
...     "date": pd.date_range("2021-01-01", periods=30),
...     "symbol": ["SPY"] * 30,
...     "adj_close": range(100, 130),
... })
>>> df = compute_all_targets(prices)
>>> df[["date", "future_1d_return", "future_5d_return"]].head()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_COLS = {"date", "symbol", "adj_close"}


def _validate(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"targets: missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Individual target functions
# ---------------------------------------------------------------------------


def future_1d_return(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    col_name: str = "future_1d_return",
) -> pd.Series:
    """Forward 1-day simple return.

    For a row at date *t* the value is ``P_{t+1} / P_t - 1``.
    The last row per symbol will be NaN (no future price available).

    Args:
        df: DataFrame sorted by (symbol, date) with a price column.
        price_col: Name of the adjusted close price column.
        col_name: Output series name.

    Returns:
        pd.Series of forward 1-day returns aligned to ``df.index``.
    """
    _validate(df)
    result = pd.Series(np.nan, index=df.index, name=col_name)
    for _, grp in df.groupby("symbol", sort=False):
        idx = grp.index
        price = grp[price_col].values
        # shift(-1): next day's price
        fwd = np.empty(len(price))
        fwd[:-1] = price[1:] / price[:-1] - 1
        fwd[-1] = np.nan
        result.loc[idx] = fwd
    return result


def future_5d_return(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    col_name: str = "future_5d_return",
) -> pd.Series:
    """Forward 5-day (weekly) simple return.

    ``P_{t+5} / P_t - 1`` for each row.  Last 5 rows per symbol are NaN.

    Args:
        df: DataFrame sorted by (symbol, date).
        price_col: Adjusted close price column.
        col_name: Output series name.

    Returns:
        pd.Series of forward 5-day returns.
    """
    _validate(df)
    result = pd.Series(np.nan, index=df.index, name=col_name)
    for _, grp in df.groupby("symbol", sort=False):
        idx = grp.index
        price = grp[price_col].values
        horizon = 5
        fwd = np.empty(len(price))
        fwd[:-horizon] = price[horizon:] / price[:-horizon] - 1
        fwd[-horizon:] = np.nan
        result.loc[idx] = fwd
    return result


def future_21d_return(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    col_name: str = "future_21d_return",
) -> pd.Series:
    """Forward 21-day (monthly) simple return.

    ``P_{t+21} / P_t - 1`` for each row.  Last 21 rows per symbol are NaN.

    Args:
        df: DataFrame sorted by (symbol, date).
        price_col: Adjusted close price column.
        col_name: Output series name.

    Returns:
        pd.Series of forward 21-day returns.
    """
    _validate(df)
    result = pd.Series(np.nan, index=df.index, name=col_name)
    for _, grp in df.groupby("symbol", sort=False):
        idx = grp.index
        price = grp[price_col].values
        horizon = 21
        fwd = np.empty(len(price))
        fwd[:-horizon] = price[horizon:] / price[:-horizon] - 1
        fwd[-horizon:] = np.nan
        result.loc[idx] = fwd
    return result


def forward_volatility(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    horizon: int = 21,
    col_name: str = "forward_volatility",
) -> pd.Series:
    """Realised volatility over the next *horizon* trading days.

    Computed as the annualised standard deviation of forward daily log-returns.
    Last ``horizon`` rows per symbol will be NaN.

    Args:
        df: DataFrame sorted by (symbol, date).
        price_col: Adjusted close price column.
        horizon: Number of future trading days over which to measure volatility.
        col_name: Output series name.

    Returns:
        pd.Series of forward realised volatility (annualised).
    """
    _validate(df)
    result = pd.Series(np.nan, index=df.index, name=col_name)
    ann_factor = np.sqrt(252)
    for _, grp in df.groupby("symbol", sort=False):
        idx = grp.index
        price = grp[price_col].values
        log_ret = np.log(price[1:] / price[:-1])  # length n-1
        n = len(price)
        fvol = np.full(n, np.nan)
        for i in range(n - horizon):
            # log returns for days i+1 .. i+horizon (the horizon days *after* day i)
            segment = log_ret[i : i + horizon]
            fvol[i] = segment.std(ddof=1) * ann_factor
        result.loc[idx] = fvol
    return result


# ---------------------------------------------------------------------------
# Convenience: compute all targets at once
# ---------------------------------------------------------------------------


def compute_all_targets(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    include_forward_vol: bool = True,
    vol_horizon: int = 21,
) -> pd.DataFrame:
    """Attach all forward-looking target columns to *df*.

    The returned DataFrame is a copy of *df* with additional columns:
    - ``future_1d_return``
    - ``future_5d_return``
    - ``future_21d_return``
    - ``forward_volatility`` (optional, annualised)

    Args:
        df: Input DataFrame with columns ``date``, ``symbol``, ``adj_close``.
            Must be sorted by (symbol, date) or will be sorted internally.
        price_col: Name of the price column to use.
        include_forward_vol: Whether to include the forward volatility target.
        vol_horizon: Horizon in trading days for the volatility target.

    Returns:
        Copy of *df* with target columns appended.

    Notes:
        **Leakage warning**: The returned DataFrame contains rows for all
        dates including the test period.  You MUST restrict to the training
        window before fitting any model or scaler.  Use
        :class:`portfolio_ml.modeling.walk_forward.WalkForwardSplitter`
        to obtain correct train/test index sets.
    """
    _validate(df)
    out = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    out["future_1d_return"] = future_1d_return(out, price_col)
    out["future_5d_return"] = future_5d_return(out, price_col)
    out["future_21d_return"] = future_21d_return(out, price_col)
    if include_forward_vol:
        out["forward_volatility"] = forward_volatility(out, price_col, horizon=vol_horizon)

    logger.info(
        "compute_all_targets: added targets for %d rows / %d symbols.",
        len(out),
        out["symbol"].nunique(),
    )
    return out
