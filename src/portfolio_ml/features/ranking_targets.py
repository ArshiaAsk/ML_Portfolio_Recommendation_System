"""Cross-sectional ranking and classification targets for ML models.

All targets are FORWARD-LOOKING and must NOT be included in features.
They are provided here for use when constructing training labels after the
train/test split.

Workflow
--------
1. Build feature matrix (feature_set_v2.build_feature_set_v2).
2. Call this module to attach forward labels to the FULL feature frame.
3. In each walk-forward fold, train ONLY on the training split.
4. Evaluate on the held-out test split.

Target types
------------
- ``future_return_21d``: raw forward 21-day return (regression baseline)
- ``cs_rank_21d``: percentile rank of forward return within each date [0, 1]
- ``cs_class_3``: 3-class label (0=bottom-third, 1=middle-third, 2=top-third)
- ``cs_class_binary``: binary label (1 = top-50%, 0 = bottom-50%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def add_ranking_targets(
    df: pd.DataFrame,
    horizon: int = 21,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Attach forward-return ranking targets to a long-format feature DataFrame.

    Args:
        df: Long-format DataFrame with columns ``date``, ``symbol``,
            and a price column (used to compute forward returns).
        horizon: Forward-return horizon in trading days.
        price_col: Price column name.

    Returns:
        Copy of ``df`` with additional target columns:
            - ``future_return_{horizon}d``    raw forward return
            - ``cs_rank_{horizon}d``          cross-sectional percentile rank [0, 1]
            - ``cs_class_3_{horizon}d``       tertile class: 0 / 1 / 2
            - ``cs_class_binary_{horizon}d``  binary: 1 = top 50%

    Notes:
        The last ``horizon`` rows per symbol will have NaN targets because
        there is no future price available.  Always drop NaN before fitting.
    """
    required = {"date", "symbol", price_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_ranking_targets: missing columns {missing}")

    out = df.sort_values(["symbol", "date"]).copy()

    # --- 1. Forward raw return (per symbol) ---
    fwd_ret_col = f"future_return_{horizon}d"

    def _fwd_return(grp: pd.DataFrame | pd.Series) -> pd.Series:
        price = (grp[price_col] if isinstance(grp, pd.DataFrame) else grp).values
        fwd = np.full(len(price), np.nan)
        fwd[:-horizon] = price[horizon:] / price[:-horizon] - 1
        return pd.Series(fwd, index=grp.index, name=fwd_ret_col)

    out[fwd_ret_col] = out.groupby("symbol", group_keys=False)[[price_col]].apply(_fwd_return)

    # --- 2. Cross-sectional rank within each date ---
    cs_rank_col = f"cs_rank_{horizon}d"
    out[cs_rank_col] = out.groupby("date")[fwd_ret_col].rank(pct=True, na_option="keep")

    # --- 3. Tertile class (0=bottom, 1=mid, 2=top) ---
    cs_class3_col = f"cs_class_3_{horizon}d"
    def _tertile(series: pd.Series) -> pd.Series:
        valid = series.notna()
        result = pd.Series(np.nan, index=series.index)
        if valid.sum() < 3:
            return result
        result.loc[valid] = pd.qcut(
            series.loc[valid], q=3, labels=[0, 1, 2], duplicates="drop"
        ).astype(float)
        return result

    out[cs_class3_col] = out.groupby("date")[fwd_ret_col].transform(_tertile)

    # --- 4. Binary label (top 50%) ---
    cs_bin_col = f"cs_class_binary_{horizon}d"
    def _binary(series: pd.Series) -> pd.Series:
        valid = series.notna()
        result = pd.Series(np.nan, index=series.index)
        if valid.sum() < 2:
            return result
        median = series.loc[valid].median()
        result.loc[valid] = (series.loc[valid] >= median).astype(float)
        return result

    out[cs_bin_col] = out.groupby("date")[fwd_ret_col].transform(_binary)

    logger.info(
        "add_ranking_targets: horizon=%d, non-NaN forward returns=%d/%d",
        horizon,
        out[fwd_ret_col].notna().sum(),
        len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Rank-aware metrics
# ---------------------------------------------------------------------------


def rank_ic(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Rank Information Coefficient (Spearman correlation).

    Args:
        actual: True forward returns or ranks.
        predicted: Model predicted scores / probabilities.

    Returns:
        Spearman rank correlation in [-1, 1].  Returns 0 if insufficient data.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    if mask.sum() < 2:
        return 0.0
    corr, _ = stats.spearmanr(a[mask], p[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def spearman_ic(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Alias for rank_ic (Spearman IC is the same metric)."""
    return rank_ic(actual, predicted)


def mean_rank_ic(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    date_col: str = "date",
) -> dict[str, float]:
    """Compute mean and std of rank IC across multiple dates.

    Args:
        df: Long-format DataFrame with actual, predicted, and date columns.
        actual_col: Column with true forward returns.
        predicted_col: Column with model scores.
        date_col: Date column name.

    Returns:
        Dict with keys: ``mean_rank_ic``, ``std_rank_ic``,
        ``n_dates``, ``pct_positive_ic``.
    """
    ics = []
    for date_val, grp in df.groupby(date_col):
        valid = grp[[actual_col, predicted_col]].dropna()
        if len(valid) < 2:
            continue
        ic = rank_ic(valid[actual_col].values, valid[predicted_col].values)
        ics.append(ic)

    if not ics:
        return {
            "mean_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "n_dates": 0,
            "pct_positive_ic": 0.0,
        }

    ics_arr = np.array(ics)
    return {
        "mean_rank_ic": float(np.mean(ics_arr)),
        "std_rank_ic": float(np.std(ics_arr)),
        "n_dates": len(ics_arr),
        "pct_positive_ic": float(np.mean(ics_arr > 0.0)),
    }


def precision_at_k(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    k: int = 3,
) -> float:
    """Proportion of top-K predicted assets that are actually in the top-K.

    Args:
        actual: True forward returns (higher = better).
        predicted: Model scores (higher = predicted better).
        k: Number of top assets to evaluate.

    Returns:
        Precision@K in [0, 1].
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]

    k = min(k, len(a))
    if k < 1:
        return 0.0

    true_top_k = set(np.argsort(a)[-k:])
    pred_top_k = set(np.argsort(p)[-k:])
    return len(true_top_k & pred_top_k) / k


def top_k_hit_rate(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    date_col: str = "date",
    k: int = 3,
) -> dict[str, float]:
    """Average Precision@K across all dates.

    Args:
        df: Long-format DataFrame.
        actual_col: True forward return column.
        predicted_col: Predicted score column.
        date_col: Date column.
        k: Number of top assets.

    Returns:
        Dict with ``mean_precision_at_k``, ``n_dates``.
    """
    scores = []
    for _, grp in df.groupby(date_col):
        valid = grp[[actual_col, predicted_col]].dropna()
        if len(valid) < 2:
            continue
        p_at_k = precision_at_k(valid[actual_col].values, valid[predicted_col].values, k=k)
        scores.append(p_at_k)

    if not scores:
        return {"mean_precision_at_k": 0.0, "n_dates": 0}

    return {
        "mean_precision_at_k": float(np.mean(scores)),
        "n_dates": len(scores),
    }


def compute_all_rank_metrics(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    date_col: str = "date",
    k: int = 3,
) -> dict[str, float]:
    """Convenience wrapper: compute all rank-aware metrics at once.

    Returns:
        Dict with all metrics from ``mean_rank_ic`` and ``top_k_hit_rate``.
    """
    ic_metrics = mean_rank_ic(df, actual_col, predicted_col, date_col)
    pak_metrics = top_k_hit_rate(df, actual_col, predicted_col, date_col, k=k)
    return {**ic_metrics, **pak_metrics}
