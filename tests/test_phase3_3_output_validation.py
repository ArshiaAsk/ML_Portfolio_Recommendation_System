import pandas as pd
import pytest

from portfolio_ml.evaluation.output_validation import validate_best_model, validate_recommendation

def model():
    return {"model_name": "gradient_boost", "params": {}, "portfolio_method": "top_k_equal", "top_k": 2, "max_weight": .75, "decision": "selected"}

def metadata(feature_date="2026-07-18"):
    return {"generation_date": "2026-07-18", "feature_date": feature_date, "model": model(), "disclaimer": "Not investment advice."}

def test_best_model_rejects_typo_decision():
    payload = model(); payload["decision"] = "condidate"
    with pytest.raises(ValueError): validate_best_model(payload)

def test_recommendation_rejects_stale_data():
    frame = pd.DataFrame({"symbol": ["A", "B"], "score": [.2, .1], "weight": [.5, .5]})
    with pytest.raises(ValueError, match="stale"):
        validate_recommendation(frame, metadata("2026-07-02"), today=pd.Timestamp("2026-07-18").date())

def test_recommendation_validates_nan():
    frame = pd.DataFrame({"symbol": ["A", "B"], "score": [.2, None], "weight": [.5, .5]})
    with pytest.raises(ValueError, match="NaN"):
        validate_recommendation(frame, metadata())

def test_recommendation_accepts_valid_contract():
    frame = pd.DataFrame({"symbol": ["A", "B"], "score": [.2, .1], "weight": [.5, .5]})
    validate_recommendation(frame, metadata())
