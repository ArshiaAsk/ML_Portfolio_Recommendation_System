"""Contract tests for Phase 3.3 best-model and recommendation validation."""

import pandas as pd
import pytest

from portfolio_ml.evaluation.output_validation import (
    validate_best_model,
    validate_recommendation,
)


def _model_payload() -> dict:
    return {
        "model_name": "gradient_boost",
        "params": {},
        "portfolio_method": "top_k_equal",
        "top_k": 2,
        "max_weight": 0.75,
        "decision": "selected",
    }


def _metadata(feature_date: str = "2026-07-18") -> dict:
    return {
        "generation_date": "2026-07-18",
        "feature_date": feature_date,
        "model": _model_payload(),
        "disclaimer": "Not investment advice.",
    }


def test_best_model_rejects_typo_decision() -> None:
    payload = _model_payload()
    payload["decision"] = "condidate"
    with pytest.raises(ValueError):
        validate_best_model(payload)


def test_recommendation_rejects_stale_data() -> None:
    frame = pd.DataFrame(
        {"symbol": ["A", "B"], "score": [0.2, 0.1], "weight": [0.5, 0.5]}
    )
    with pytest.raises(ValueError, match="stale"):
        validate_recommendation(
            frame,
            _metadata("2026-07-02"),
            today=pd.Timestamp("2026-07-18").date(),
        )


def test_recommendation_validates_nan() -> None:
    frame = pd.DataFrame(
        {"symbol": ["A", "B"], "score": [0.2, None], "weight": [0.5, 0.5]}
    )
    with pytest.raises(ValueError, match="NaN"):
        validate_recommendation(frame, _metadata())


def test_recommendation_accepts_valid_contract() -> None:
    frame = pd.DataFrame(
        {"symbol": ["A", "B"], "score": [0.2, 0.1], "weight": [0.5, 0.5]}
    )
    validate_recommendation(frame, _metadata())
