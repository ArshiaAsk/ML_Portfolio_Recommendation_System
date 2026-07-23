"""Regime-routed ranking model using validated v2 regime features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .factory import build_model

class RegimeAwareRankModel:
    def __init__(self, model_name="ridge", model_params=None, min_training_rows=200):
        self.model_name, self.model_params = model_name, model_params or {}
        self.min_training_rows = min_training_rows
        self.global_model = None
        self.regime_models = {}

    @staticmethod
    def regime_labels(frame):
        trend = np.where(frame.get("bm_ret_21d", 0).fillna(0).to_numpy() >= 0, "Bull", "Bear")
        vol = np.where(frame.get("vol_regime_flag", 0).fillna(0).to_numpy() > 0, "HighVol", "LowVol")
        return pd.Series([f"{a}_{b}" for a, b in zip(trend, vol)], index=frame.index)

    def fit(self, X, y):
        labels = self.regime_labels(X)
        target = pd.Series(np.asarray(y), index=X.index)
        self.global_model = build_model(self.model_name, **self.model_params).fit(X, target.to_numpy())
        for regime, indexes in labels.groupby(labels).groups.items():
            if len(indexes) >= self.min_training_rows:
                self.regime_models[regime] = build_model(
                    self.model_name, **self.model_params
                ).fit(X.loc[indexes], target.loc[indexes].to_numpy())
        return self

    def predict(self, X):
        if self.global_model is None: raise RuntimeError("RegimeAwareRankModel must be fitted before predict")
        labels = self.regime_labels(X)
        output = pd.Series(self.global_model.predict(X), index=X.index, dtype=float)
        for regime, indexes in labels.groupby(labels).groups.items():
            if regime in self.regime_models:
                output.loc[indexes] = self.regime_models[regime].predict(X.loc[indexes])
        return output.to_numpy()
