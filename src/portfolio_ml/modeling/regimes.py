"""Heuristic regime labels and a regime-aware strategy wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_ml.modeling.baselines import EqualWeightStrategy, MinVarianceStrategy, RiskParityStrategy


class RegimeLabeler:
    """Heuristically classify a return series into trend/volatility regimes."""

    def label(self, returns: pd.Series | np.ndarray) -> dict[str, str]:
        values = np.asarray(returns, dtype=float)
        if values.size == 0:
            return {"trend": "Neutral", "volatility": "LowVol"}

        recent_return = float(values[-1]) if values.size else 0.0
        rolling_mean = float(np.nanmean(values[-5:])) if values.size >= 5 else float(np.nanmean(values))
        volatility = float(np.nanstd(values[-5:])) if values.size >= 5 else float(np.nanstd(values))

        if recent_return > 0.0 or rolling_mean > 0.0:
            trend = "Bull"
        elif recent_return < 0.0 or rolling_mean < 0.0:
            trend = "Bear"
        else:
            trend = "Neutral"

        volatility_label = "HighVol" if volatility > 0.02 else "LowVol"
        return {"trend": trend, "volatility": volatility_label}


class RegimeAwareStrategy:
    """Select a baseline strategy according to a simple regime heuristic."""

    def __init__(self):
        self.equal_weight = EqualWeightStrategy()
        self.min_variance = MinVarianceStrategy()
        self.risk_parity = RiskParityStrategy()

    def allocate(self, cov_matrix: np.ndarray | None = None, regime: dict[str, str] | None = None, n_assets: int | None = None) -> np.ndarray:
        regime = regime or {"trend": "Neutral", "volatility": "LowVol"}
        if n_assets is None or n_assets < 1:
            raise ValueError("n_assets must be provided and >= 1")

        trend = regime.get("trend", "Neutral")
        volatility = regime.get("volatility", "LowVol")

        if trend == "Bear" or volatility == "HighVol":
            strategy = self.risk_parity if cov_matrix is not None else self.equal_weight
        else:
            strategy = self.min_variance if cov_matrix is not None else self.equal_weight

        if cov_matrix is not None and strategy is self.min_variance:
            return strategy.allocate(cov_matrix=cov_matrix)
        if cov_matrix is not None and strategy is self.risk_parity:
            return strategy.allocate(cov_matrix=cov_matrix)
        return self.equal_weight.allocate(n_assets=n_assets)
