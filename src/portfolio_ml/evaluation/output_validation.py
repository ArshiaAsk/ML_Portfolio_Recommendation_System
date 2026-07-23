"""Validation contracts for Phase 3.3 persisted artifacts."""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd

VALID_DECISIONS = {"candidate", "selected", "rejected"}


def validate_best_model(payload: dict) -> dict:
    required = {"model_name", "params", "portfolio_method", "top_k", "max_weight", "decision"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"best_model.json missing fields: {sorted(missing)}")
    if payload["decision"] not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
    if payload["portfolio_method"] not in {"top_k_equal", "score_weighted", "mean_variance"}:
        raise ValueError("unsupported portfolio_method")
    if int(payload["top_k"]) < 1 or not 0 < float(payload["max_weight"]) <= 1:
        raise ValueError("invalid portfolio constraints")
    return payload


def validate_recommendation(frame: pd.DataFrame, metadata: dict, *, today: date | None = None,
                            max_staleness_days: int = 3) -> pd.DataFrame:
    required = {"symbol", "score", "weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"recommendation missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame["symbol"].duplicated().any():
        raise ValueError("recommendation must contain unique assets")
    if frame[["score", "weight"]].isna().any().any():
        raise ValueError("recommendation contains NaN scores or weights")
    weights = frame["weight"].astype(float)
    max_weight = float(metadata["model"]["max_weight"])
    if (weights < -1e-10).any() or (weights > max_weight + 1e-8).any() or not np.isclose(weights.sum(), 1.0, atol=1e-7):
        raise ValueError("invalid portfolio weights")
    if metadata.get("disclaimer") != "Not investment advice.":
        raise ValueError("missing exact disclaimer")
    feature_date = pd.Timestamp(metadata["feature_date"]).date()
    generation_date = pd.Timestamp(metadata["generation_date"]).date()
    # Callers producing live recommendations should pass today's date. Using
    # generation_date by default keeps historical artifact validation
    # deterministic and avoids making tests depend on the wall clock.
    reference = today or generation_date
    if generation_date < feature_date:
        raise ValueError("generation_date precedes feature_date")
    stale_days = (reference - feature_date).days
    if stale_days > max_staleness_days:
        raise ValueError(f"market data is stale by {stale_days} calendar days")
    return frame
