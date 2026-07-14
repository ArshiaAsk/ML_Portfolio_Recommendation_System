"""Stable experiment result schema and validation for Phase 3.2.

Defines the canonical columns for unified_experiment_summary_v2.csv and
provides validation utilities that flag impossible or suspicious metric
combinations without silently accepting them.
"""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical column list (the "contract")
# ---------------------------------------------------------------------------

EXPERIMENT_SCHEMA_COLUMNS: list[str] = [
    # Identity
    "run_id",
    "timestamp",
    "experiment_type",   # baseline | ml_regression | ml_ranking | ml_classification | regime | two_stage_portfolio
    "strategy_name",
    "model_name",
    "feature_set",
    "target_type",
    "portfolio_method",
    "rebalance_frequency",
    "lookback_window",
    "transaction_cost_bps",
    # Time split info
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    # Portfolio-level metrics
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "turnover",
    "hit_rate",
    # Ranking / signal metrics (NaN when not applicable)
    "rank_ic",
    "spearman_ic",
    "precision_at_k",
    "top_k_hit_rate",
    # Regression metrics (NaN when not applicable)
    "rmse",
    "mae",
    "r2",
    # Free text
    "notes",
]

# Valid experiment types
VALID_EXPERIMENT_TYPES = {
    "baseline",
    "ml_regression",
    "ml_ranking",
    "ml_classification",
    "regime",
    "two_stage_portfolio",
}


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def make_experiment_row(
    experiment_type: str,
    *,
    strategy_name: str | None = None,
    model_name: str | None = None,
    feature_set: str | None = None,
    target_type: str | None = None,
    portfolio_method: str | None = None,
    rebalance_frequency: int | None = None,
    lookback_window: int | None = None,
    transaction_cost_bps: float | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    annualised_return: float | None = None,
    annualised_volatility: float | None = None,
    sharpe_ratio: float | None = None,
    sortino_ratio: float | None = None,
    max_drawdown: float | None = None,
    calmar_ratio: float | None = None,
    turnover: float | None = None,
    hit_rate: float | None = None,
    rank_ic: float | None = None,
    spearman_ic: float | None = None,
    precision_at_k: float | None = None,
    top_k_hit_rate: float | None = None,
    rmse: float | None = None,
    mae: float | None = None,
    r2: float | None = None,
    notes: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create a canonical experiment row with the stable schema.

    All unspecified numeric fields default to ``np.nan``.  Required identity
    fields (``run_id``, ``timestamp``) are filled automatically.

    Args:
        experiment_type: One of the VALID_EXPERIMENT_TYPES values.
        **kwargs: Metric and metadata fields from the schema.

    Returns:
        Dictionary with all EXPERIMENT_SCHEMA_COLUMNS keys present.
    """
    if experiment_type not in VALID_EXPERIMENT_TYPES:
        warnings.warn(
            f"make_experiment_row: experiment_type='{experiment_type}' is not in "
            f"VALID_EXPERIMENT_TYPES {VALID_EXPERIMENT_TYPES}. "
            "Row will still be created.",
            UserWarning,
            stacklevel=2,
        )

    row: dict[str, Any] = {
        "run_id": run_id or str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "experiment_type": experiment_type,
        "strategy_name": strategy_name,
        "model_name": model_name,
        "feature_set": feature_set,
        "target_type": target_type,
        "portfolio_method": portfolio_method,
        "rebalance_frequency": rebalance_frequency,
        "lookback_window": lookback_window,
        "transaction_cost_bps": transaction_cost_bps,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "turnover": turnover,
        "hit_rate": hit_rate,
        "rank_ic": rank_ic,
        "spearman_ic": spearman_ic,
        "precision_at_k": precision_at_k,
        "top_k_hit_rate": top_k_hit_rate,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "notes": notes,
    }
    # Fill any schema columns that might be missing (future-proofing)
    for col in EXPERIMENT_SCHEMA_COLUMNS:
        row.setdefault(col, np.nan)
    return row


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

ValidationWarning = str  # a human-readable issue description


def validate_experiment_row(row: dict[str, Any]) -> list[ValidationWarning]:
    """Validate a single experiment row and return a list of warnings.

    Args:
        row: A dict conforming (or not) to the experiment schema.

    Returns:
        List of human-readable warning strings.  Empty if no issues found.
    """
    issues: list[ValidationWarning] = []

    exp_type = row.get("experiment_type", "")
    strategy_name = row.get("strategy_name")
    model_name = row.get("model_name")

    # --- Identity checks ---
    if exp_type in ("baseline", "regime", "two_stage_portfolio"):
        if not strategy_name or (isinstance(strategy_name, float) and np.isnan(strategy_name)):
            issues.append(
                f"[{exp_type}] strategy_name is missing/NaN — every portfolio strategy row must have a name."
            )

    if exp_type in ("ml_regression", "ml_ranking", "ml_classification"):
        if not model_name or (isinstance(model_name, float) and np.isnan(model_name)):
            issues.append(
                f"[{exp_type}] model_name is missing/NaN — every ML row must identify the model."
            )

    # --- Numeric metric checks ---
    ann_ret = row.get("annualised_return")
    ann_vol = row.get("annualised_volatility")
    sharpe = row.get("sharpe_ratio")
    mdd = row.get("max_drawdown")

    if _is_present(ann_vol) and _is_present(ann_ret):
        if abs(float(ann_vol)) < 1e-9 and abs(float(ann_ret)) > 1e-6:
            issues.append(
                "annualised_volatility is ~0 while annualised_return != 0 — suspicious."
            )

    if _is_present(sharpe):
        sharpe_val = float(sharpe)
        if np.isinf(sharpe_val) or abs(sharpe_val) > 100:
            issues.append(
                f"sharpe_ratio={sharpe_val:.4f} is inf or unrealistically large (|Sharpe|>100)."
            )

    if _is_present(mdd):
        mdd_val = float(mdd)
        if mdd_val > 0.0:
            issues.append(
                f"max_drawdown={mdd_val:.4f} is positive — expected to be ≤ 0 by convention."
            )
        if mdd_val < -1.0:
            issues.append(
                f"max_drawdown={mdd_val:.4f} < -1.0 — drawdown cannot exceed 100%."
            )

    if _is_present(ann_ret) and _is_present(ann_vol):
        ann_ret_v = float(ann_ret)
        ann_vol_v = float(ann_vol)
        if abs(ann_ret_v) < 1e-9 and exp_type in ("baseline", "two_stage_portfolio"):
            issues.append(
                "annualised_return is ~0 for an active strategy — may indicate all-zero returns."
            )

    # --- Rank-IC sanity ---
    rank_ic = row.get("rank_ic")
    if _is_present(rank_ic):
        ric = float(rank_ic)
        if not (-1.0 <= ric <= 1.0):
            issues.append(f"rank_ic={ric:.4f} is outside [-1, 1] — invalid correlation.")

    spearman_ic = row.get("spearman_ic")
    if _is_present(spearman_ic):
        sic = float(spearman_ic)
        if not (-1.0 <= sic <= 1.0):
            issues.append(f"spearman_ic={sic:.4f} is outside [-1, 1] — invalid correlation.")

    return issues


def validate_experiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an entire experiment summary DataFrame.

    Args:
        df: DataFrame of experiment rows (each row is one experiment result).

    Returns:
        The input DataFrame with an extra column ``validation_warnings`` that
        contains a pipe-delimited string of warnings per row (empty string if
        clean).

    Side effects:
        Logs warnings for each problematic row.
    """
    # Check for duplicate run_ids
    if "run_id" in df.columns:
        dup_ids = df["run_id"][df["run_id"].duplicated()]
        if not dup_ids.empty:
            logger.warning(
                "validate_experiment_dataframe: found %d duplicate run_id(s): %s",
                len(dup_ids),
                dup_ids.tolist(),
            )

    warning_col = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        issues = validate_experiment_row(row_dict)
        if issues:
            for issue in issues:
                logger.warning("Row %d: %s", idx, issue)
        warning_col.append(" | ".join(issues))

    df = df.copy()
    df["validation_warnings"] = warning_col
    return df


def _is_present(val: Any) -> bool:
    """Return True if val is a non-None, non-NaN scalar."""
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    try:
        return not np.isnan(float(val))
    except (TypeError, ValueError):
        return val is not None


# ---------------------------------------------------------------------------
# Migration helper: convert old experiment summaries to v2 schema
# ---------------------------------------------------------------------------


def migrate_legacy_summary(df: pd.DataFrame, source_label: str = "legacy") -> pd.DataFrame:
    """Convert an old-style experiment summary to the v2 schema.

    Handles the column mappings from the old unified_experiment_summary.csv
    produced by ``run_unified_experiments.py``.

    Args:
        df: Old-style DataFrame.
        source_label: Label to use in ``notes`` column.

    Returns:
        DataFrame conforming to EXPERIMENT_SCHEMA_COLUMNS.
    """
    rows = []
    column_map = {
        # old → new
        "strategy": "strategy_name",
        "Strategy": "strategy_name",
        "lookback_window": "lookback_window",
        "rebalance_freq": "rebalance_frequency",
        "Annualised Return": "annualised_return",
        "Annualised Volatility": "annualised_volatility",
        "Sharpe Ratio": "sharpe_ratio",
        "Max Drawdown": "max_drawdown",
        "Avg Turnover": "turnover",
        "Total Return": "_total_return_ignore",
        "experiment_type": "experiment_type",
        "score": "_score_ignore",
    }

    for _, old_row in df.iterrows():
        row = make_experiment_row(
            experiment_type=_safe_str(old_row.get("experiment_type", "baseline")),
            strategy_name=_safe_str(old_row.get("Strategy") or old_row.get("strategy")),
            model_name=_safe_str(old_row.get("model_name")),
            rebalance_frequency=_safe_int(old_row.get("rebalance_freq")),
            lookback_window=_safe_int(old_row.get("lookback_window")),
            annualised_return=_safe_float(old_row.get("Annualised Return")),
            annualised_volatility=_safe_float(old_row.get("Annualised Volatility")),
            sharpe_ratio=_safe_float(old_row.get("Sharpe Ratio")),
            max_drawdown=_safe_float(old_row.get("Max Drawdown")),
            turnover=_safe_float(old_row.get("Avg Turnover")),
            notes=f"migrated from {source_label}",
        )
        rows.append(row)

    result = pd.DataFrame(rows, columns=EXPERIMENT_SCHEMA_COLUMNS)
    return result


def _safe_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None
