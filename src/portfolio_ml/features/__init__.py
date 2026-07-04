"""Feature engineering and target definition modules."""

from portfolio_ml.features.targets import (
    compute_all_targets,
    forward_volatility,
    future_1d_return,
    future_21d_return,
    future_5d_return,
)

__all__ = [
    "compute_all_targets",
    "future_1d_return",
    "future_5d_return",
    "future_21d_return",
    "forward_volatility",
]
