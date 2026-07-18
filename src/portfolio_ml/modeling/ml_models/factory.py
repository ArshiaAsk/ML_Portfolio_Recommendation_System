"""Sklearn-only model factory used by Phase 3.3."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


class EnsembleRankModel:
    """Average rank-normalised predictions from three fixed base models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "ridge": build_model("ridge", random_state=random_state),
            "random_forest": build_model("random_forest", random_state=random_state),
            "gradient_boost": build_model("gradient_boost", random_state=random_state),
        }

    @staticmethod
    def _rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
        return (order + 1.0) / max(len(values), 1)

    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.vstack([self._rank(np.asarray(m.predict(X))) for m in self.models.values()])
        return predictions.mean(axis=0)


def build_model(name: str, **params):
    """Build a supported ranking estimator.

    ``gradient_boost`` intentionally maps to HistGradientBoostingRegressor;
    the public name remains stable for experiment schemas and CLI users.
    """
    name = name.lower()
    random_state = params.pop("random_state", 42)
    if name == "ridge":
        return Ridge(alpha=params.pop("alpha", 1.0), **params)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.pop("n_estimators", 100),
            max_depth=params.pop("max_depth", 6),
            random_state=random_state, n_jobs=params.pop("n_jobs", -1), **params
        )
    if name == "gradient_boost":
        return HistGradientBoostingRegressor(
            max_iter=params.pop("max_iter", 100),
            max_depth=params.pop("max_depth", 4),
            learning_rate=params.pop("learning_rate", 0.05),
            random_state=random_state, **params
        )
    if name == "ensemble":
        if params:
            raise ValueError(f"Unsupported ensemble parameters: {sorted(params)}")
        return EnsembleRankModel(random_state=random_state)
    raise ValueError(f"Unknown model: {name}")
