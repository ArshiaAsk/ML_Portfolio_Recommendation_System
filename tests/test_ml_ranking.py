"""Tests for Sprint 3 ranking-model experiments."""

import numpy as np
import pandas as pd

from portfolio_ml.modeling.ml_models import GradientBoostRankModel
from portfolio_ml.modeling.rank_evaluator import RankEvaluator


def test_gradient_boost_rank_model_training_and_prediction():
    """The ranking model should train on tabular features and produce ranked scores."""
    X = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_b": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    y = pd.Series([0.2, 0.1, 0.4, 0.3, 0.6, 0.5])

    model = GradientBoostRankModel(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.ptp(preds) > 0.0
    assert np.corrcoef(preds, y)[0, 1] > 0.0


def test_rank_evaluator_computes_expected_metrics():
    """The evaluator should return rank-based metrics from actual and predicted scores."""
    actual = pd.Series([0.4, 0.2, 0.1, 0.3])
    predicted = pd.Series([0.3, 0.1, 0.2, 0.4])

    evaluator = RankEvaluator()
    metrics = evaluator.evaluate(actual, predicted)

    assert metrics["spearman"] > 0.0
    assert metrics["ndcg_at_2"] >= 0.0
    assert metrics["topk_accuracy"] >= 0.0
