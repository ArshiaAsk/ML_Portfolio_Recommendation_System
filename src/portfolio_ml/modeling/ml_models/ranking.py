"""Simple ranking-model implementations for portfolio experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostRankModel:
    """A lightweight gradient boosting regressor for ranking-style targets."""

    def __init__(self, random_state: int = 42, n_estimators: int = 100):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_ = GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "GradientBoostRankModel":
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
