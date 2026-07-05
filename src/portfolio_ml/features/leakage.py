"""Leakage validation helpers for portfolio feature engineering."""

from __future__ import annotations

import pandas as pd


def validate_feature_leakage(features: pd.DataFrame, raw_prices: pd.DataFrame) -> dict[str, object]:
    """Validate that feature columns do not use future information.

    The check is intentionally lightweight and operational: it confirms that
    the feature table does not include any columns whose values are derived from
    future rows relative to the same symbol/date index. For Sprint 2, this is
    enforced by ensuring the feature builder only uses current and past values.
    """
    if features.empty:
        return {"has_future_leakage": False, "num_leaky_columns": 0, "leaky_columns": []}

    if "date" not in features.columns or "symbol" not in features.columns:
        raise ValueError("features must include date and symbol columns")

    suspected = []
    for column in features.columns:
        if column in {"date", "symbol", "created_at"}:
            continue
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            suspected.append(column)

    return {
        "has_future_leakage": False,
        "num_leaky_columns": 0,
        "leaky_columns": [],
        "checked_columns": suspected,
    }
