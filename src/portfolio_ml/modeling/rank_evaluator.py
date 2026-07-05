"""Rank-based evaluation utilities for portfolio ranking models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class RankEvaluator:
    """Compute rank-oriented metrics for predicted vs actual scores."""

    def evaluate(self, actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
        actual_arr = np.asarray(actual, dtype=float)
        predicted_arr = np.asarray(predicted, dtype=float)

        if len(actual_arr) != len(predicted_arr):
            raise ValueError("actual and predicted must have the same length")

        corr = spearmanr(actual_arr, predicted_arr).correlation
        if np.isnan(corr):
            corr = 0.0

        actual_ranked = np.argsort(-actual_arr)
        predicted_ranked = np.argsort(-predicted_arr)
        topk = np.mean(actual_ranked[:2] == predicted_ranked[:2])

        dcg = self._dcg(actual_arr, predicted_arr, k=2)
        ideal_dcg = self._dcg(actual_arr, actual_arr, k=2)
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        return {
            "spearman": float(corr),
            "ndcg_at_2": float(ndcg),
            "topk_accuracy": float(topk),
        }

    def _dcg(self, actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
        order = np.argsort(-predicted)
        gains = actual[order][:k]
        discounts = np.log2(np.arange(2, len(gains) + 2))
        return float(np.sum((2**gains - 1) / discounts))
