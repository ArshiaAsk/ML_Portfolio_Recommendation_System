"""Two-stage ML portfolio construction.

Stage 1 — Signal generation
    - Model predicts ranking score (or class probability) per asset.
    - Signals are normalized cross-sectionally.

Stage 2 — Portfolio construction
    Three allocation methods:
    1. ``top_k_equal``      : select top-K by score, equal-weight.
    2. ``score_weighted``   : weight proportional to positive scores, capped.
    3. ``mean_variance``    : use scores as expected-return proxy, optimise
                              via mean-variance with long-only + max-weight
                              constraints.

All methods:
    - Long-only by default.
    - No leverage (sum(weights) = 1).
    - Max weight cap configurable (default 0.25).
    - Handle missing predictions safely (zero or equal fallback).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TwoStageConfig:
    """Configuration for two-stage portfolio construction.

    Args:
        portfolio_method: ``"top_k_equal"`` | ``"score_weighted"`` | ``"mean_variance"``.
        top_k: Number of assets to select (for ``top_k_equal``).
        max_weight: Maximum weight for any single asset.
        min_weight: Minimum weight (0 for long-only).
        cov_window: Trailing days for covariance estimation.
        cov_shrinkage: Ledoit-Wolf-style shrinkage towards identity (0=none, 1=full).
        score_floor: Minimum normalised score required to receive positive weight
                     in ``score_weighted`` mode.  Assets below this are excluded.
        rebalance_frequency: How often (in trading days) to rebalance.
        transaction_cost_bps: Transaction cost in basis points.
    """
    portfolio_method: str = "top_k_equal"
    top_k: int = 5
    max_weight: float = 0.25
    min_weight: float = 0.0
    cov_window: int = 63
    cov_shrinkage: float = 0.1
    score_floor: float = 0.0
    rebalance_frequency: int = 21
    transaction_cost_bps: float = 5.0

    def __post_init__(self) -> None:
        valid_methods = {"top_k_equal", "score_weighted", "mean_variance"}
        if self.portfolio_method not in valid_methods:
            raise ValueError(
                f"portfolio_method must be one of {valid_methods}, got '{self.portfolio_method}'"
            )
        if not (0 < self.max_weight <= 1.0):
            raise ValueError(f"max_weight must be in (0, 1], got {self.max_weight}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


# ---------------------------------------------------------------------------
# Signal normalisation
# ---------------------------------------------------------------------------


def normalise_scores_cross_sectionally(scores: pd.Series) -> pd.Series:
    """Normalise scores to [0, 1] cross-sectionally using percentile rank.

    Args:
        scores: Series indexed by asset symbols.

    Returns:
        Normalised scores in [0, 1].  NaN inputs remain NaN.
    """
    valid = scores.dropna()
    if valid.empty:
        return scores * np.nan
    normalised = valid.rank(pct=True)
    return normalised.reindex(scores.index)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def compute_weights_from_scores(
    scores: pd.Series,
    asset_names: List[str],
    config: TwoStageConfig,
    returns_history: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Compute portfolio weights from ML signal scores.

    Args:
        scores: Signal scores indexed by asset name (higher = better).
                NaN means "no signal for this asset".
        asset_names: Ordered list of asset names (defines weight vector order).
        config: Two-stage portfolio configuration.
        returns_history: Historical returns DataFrame (DatetimeIndex, columns=assets).
                         Required for ``mean_variance`` method.

    Returns:
        1-D weight array aligned to ``asset_names``.
    """
    scores = scores.reindex(asset_names).fillna(0.0)
    n = len(asset_names)

    if config.portfolio_method == "top_k_equal":
        weights = _top_k_equal_weights(scores, config.top_k, n)

    elif config.portfolio_method == "score_weighted":
        weights = _score_weighted(scores, config.max_weight, config.score_floor)

    elif config.portfolio_method == "mean_variance":
        weights = _mean_variance_weights(
            scores,
            asset_names,
            returns_history,
            config,
        )
    else:
        raise ValueError(f"Unknown portfolio_method: {config.portfolio_method}")

    # Apply constraints and normalise
    weights = np.clip(weights, config.min_weight, config.max_weight)
    total = weights.sum()
    if total < 1e-9:
        # Fallback to equal weight
        logger.warning("compute_weights_from_scores: all weights zero, falling back to equal weight.")
        weights = np.ones(n) / n
    else:
        weights = weights / total

    return weights


def _top_k_equal_weights(scores: pd.Series, top_k: int, n: int) -> np.ndarray:
    """Select top-K assets by score and equal-weight them."""
    weights = np.zeros(n)
    k = min(top_k, n)
    top_indices = scores.values.argsort()[-k:]
    weights[top_indices] = 1.0 / k
    return weights


def _score_weighted(scores: pd.Series, max_weight: float, score_floor: float) -> np.ndarray:
    """Weight assets proportional to their positive scores."""
    vals = scores.values.copy()
    vals[vals < score_floor] = 0.0
    vals = np.clip(vals, 0.0, None)
    total = vals.sum()
    if total < 1e-9:
        return vals
    # Initial proportional weights
    weights = vals / total
    # Apply cap and redistribute surplus (simple iterative capping)
    for _ in range(20):
        over_cap = weights > max_weight
        if not over_cap.any():
            break
        surplus = (weights[over_cap] - max_weight).sum()
        weights[over_cap] = max_weight
        uncapped = ~over_cap
        if uncapped.sum() > 0 and weights[uncapped].sum() > 1e-9:
            weights[uncapped] += surplus * (weights[uncapped] / weights[uncapped].sum())
    return weights


def _mean_variance_weights(
    scores: pd.Series,
    asset_names: List[str],
    returns_history: Optional[pd.DataFrame],
    config: TwoStageConfig,
) -> np.ndarray:
    """Mean-variance optimisation using scores as expected-return proxy."""
    n = len(asset_names)

    # Build covariance matrix
    if returns_history is not None and len(returns_history) >= config.cov_window // 4:
        hist = returns_history[asset_names].iloc[-config.cov_window:]
        cov = hist.cov().fillna(0.0).values
    else:
        # Fallback: identity (proportional to scores only)
        cov = np.eye(n) * 0.0001
        logger.warning("_mean_variance_weights: insufficient history; using diagonal covariance.")

    # Apply shrinkage towards scaled identity
    if config.cov_shrinkage > 0:
        target = np.eye(n) * np.diag(cov).mean()
        cov = (1 - config.cov_shrinkage) * cov + config.cov_shrinkage * target

    cov += np.eye(n) * 1e-8  # numerical stability

    mu = scores.values.astype(float)
    # Scale mu to avoid numerical issues
    mu_range = np.ptp(mu)
    if mu_range > 1e-9:
        mu = (mu - mu.min()) / mu_range  # normalise to [0, 1]
    else:
        return np.ones(n) / n  # flat scores → equal weight

    # Optimise (maximise Sharpe-like objective)
    def neg_score_over_vol(w: np.ndarray) -> float:
        port_ret = float(w @ mu)
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        return -port_ret / port_vol

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(config.min_weight, config.max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_score_over_vol,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if not result.success:
        logger.warning("_mean_variance_weights: optimisation did not converge: %s", result.message)

    weights = result.x
    weights = np.clip(weights, 0.0, config.max_weight)
    return weights


# ---------------------------------------------------------------------------
# Turnover computation
# ---------------------------------------------------------------------------


def compute_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """Compute one-way portfolio turnover.

    Args:
        old_weights: Previous rebalancing weights.
        new_weights: New weights after signal update.

    Returns:
        Sum of absolute weight changes (one-way).
    """
    return float(np.sum(np.abs(new_weights - old_weights)))
