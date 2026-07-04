"""Baseline portfolio allocation strategies for benchmarking ML models.

These simple strategies serve as performance baselines for more sophisticated
machine-learning–based allocators.  All strategies produce portfolio weights
that sum to 1.0 and lie in [0, 1] (long-only, fully invested).

Strategies
----------
1. **Equal Weight**: ``1/N`` allocation to each asset.  No input required.
2. **Minimum Variance**: Solve the quadratic program that minimises portfolio
   variance subject to weights summing to 1 and being non-negative.
3. **Risk Parity**: Allocate so that each asset contributes equally to the
   total portfolio volatility.  Approximated via iterative optimisation.

Example
-------
>>> import numpy as np
>>> from portfolio_ml.modeling.baselines import (
...     EqualWeightStrategy,
...     MinVarianceStrategy,
...     RiskParityStrategy,
... )
>>> returns = np.random.randn(252, 5) * 0.01  # 5 assets, 252 days
>>> cov = np.cov(returns.T)
>>> ew = EqualWeightStrategy()
>>> w_ew = ew.allocate(n_assets=5)
>>> mv = MinVarianceStrategy()
>>> w_mv = mv.allocate(cov_matrix=cov)
>>> rp = RiskParityStrategy()
>>> w_rp = rp.allocate(cov_matrix=cov)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class PortfolioStrategy(ABC):
    """Abstract base for portfolio allocation strategies."""

    @abstractmethod
    def allocate(self, **kwargs) -> np.ndarray:
        """Return a 1-D array of portfolio weights summing to 1.0."""
        pass


# ---------------------------------------------------------------------------
# 1. Equal Weight
# ---------------------------------------------------------------------------


class EqualWeightStrategy(PortfolioStrategy):
    """Naïve 1/N equal-weight allocation.

    Each asset receives the same weight regardless of historical returns,
    volatility, or correlations.  This is a surprisingly robust baseline.

    Example:
        >>> ew = EqualWeightStrategy()
        >>> weights = ew.allocate(n_assets=10)
        >>> weights.sum()
        1.0
    """

    def allocate(self, n_assets: int, **kwargs) -> np.ndarray:
        """Return equal weights for *n_assets*.

        Args:
            n_assets: Number of assets in the universe.

        Returns:
            1-D array of length *n_assets* with each element = 1 / n_assets.
        """
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1.")
        w = np.ones(n_assets) / n_assets
        logger.debug("EqualWeightStrategy: allocated 1/%d to each asset.", n_assets)
        return w


# ---------------------------------------------------------------------------
# 2. Minimum Variance
# ---------------------------------------------------------------------------


class MinVarianceStrategy(PortfolioStrategy):
    """Global minimum-variance portfolio (long-only).

    Solves:
        min  w^T Σ w
        s.t. w^T 1 = 1,  w >= 0

    where Σ is the covariance matrix.  No expected return estimates are used.

    Args:
        allow_short: If ``True``, remove the non-negativity constraint.  By
            default only long positions are allowed.

    Example:
        >>> mv = MinVarianceStrategy()
        >>> cov = np.eye(5) * 0.01  # diagonal covariance
        >>> weights = mv.allocate(cov_matrix=cov)
    """

    def __init__(self, allow_short: bool = False):
        self.allow_short = allow_short

    def allocate(self, cov_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """Compute minimum-variance weights.

        Args:
            cov_matrix: Covariance matrix of shape ``(n, n)``.

        Returns:
            1-D array of portfolio weights.

        Raises:
            ValueError: If covariance matrix is not square or positive definite.
        """
        cov = np.asarray(cov_matrix)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov_matrix must be a square 2-D array.")

        n = cov.shape[0]
        if n < 1:
            raise ValueError("cov_matrix dimension must be >= 1.")

        # Add small ridge for numerical stability
        cov_reg = cov + np.eye(n) * 1e-8

        def objective(w):
            return 0.5 * w @ cov_reg @ w

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n if not self.allow_short else [(None, None)] * n
        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            logger.warning(
                "MinVarianceStrategy: optimisation did not converge: %s", result.message
            )

        weights = result.x
        # Enforce sum-to-one due to numerical drift
        weights = np.clip(weights, 0.0, 1.0) if not self.allow_short else weights
        weights /= weights.sum()

        logger.debug(
            "MinVarianceStrategy: allocated with variance=%.6f", objective(weights)
        )
        return weights


# ---------------------------------------------------------------------------
# 3. Risk Parity
# ---------------------------------------------------------------------------


class RiskParityStrategy(PortfolioStrategy):
    """Risk-parity allocation (equal risk contribution).

    Each asset contributes the same amount to overall portfolio volatility.
    The optimisation target is:

        min  Σ_i (RC_i - target)^2

    where ``RC_i = w_i * (Σ w)_i`` is the risk contribution of asset *i*,
    and ``target = portfolio_vol / n``.

    Args:
        max_iter: Maximum number of optimisation iterations.

    Example:
        >>> rp = RiskParityStrategy()
        >>> cov = np.eye(3) * 0.04
        >>> weights = rp.allocate(cov_matrix=cov)
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter

    def allocate(self, cov_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """Compute risk-parity weights.

        Args:
            cov_matrix: Covariance matrix of shape ``(n, n)``.

        Returns:
            1-D array of portfolio weights.
        """
        cov = np.asarray(cov_matrix)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov_matrix must be square.")

        n = cov.shape[0]
        cov_reg = cov + np.eye(n) * 1e-8

        def portfolio_variance(w):
            return w @ cov_reg @ w

        def risk_contributions(w):
            """Marginal contribution to risk (MCR) times weight."""
            pvar = portfolio_variance(w)
            if pvar < 1e-12:
                return np.zeros(n)
            pvol = np.sqrt(pvar)
            # Marginal contribution to variance
            mcv = cov_reg @ w
            # Risk contribution: w_i * (dVar/dw_i) / (2 * portfolio_vol)
            rc = w * mcv / pvol
            return rc

        def objective(w):
            """Sum of squared deviations from equal risk contribution."""
            rc = risk_contributions(w)
            target = rc.sum() / n  # equal share of total risk
            return np.sum((rc - target) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": self.max_iter},
        )

        if not result.success:
            logger.warning(
                "RiskParityStrategy: optimisation did not converge: %s", result.message
            )

        weights = result.x
        weights = np.clip(weights, 0.0, 1.0)
        weights /= weights.sum()

        logger.debug(
            "RiskParityStrategy: allocated with objective=%.6f", objective(weights)
        )
        return weights
