"""Feature engineering and target definition modules."""

from portfolio_ml.features.leakage import validate_feature_leakage
from portfolio_ml.features.targets import (
    compute_all_targets,
    forward_volatility,
    future_1d_return,
    future_21d_return,
    future_5d_return,
)
from portfolio_ml.features.feature_set_v2 import (
    build_feature_set_v2,
    feature_summary,
    get_feature_columns,
)
from portfolio_ml.features.ranking_targets import (
    add_ranking_targets,
    compute_all_rank_metrics,
    mean_rank_ic,
    precision_at_k,
    rank_ic,
    spearman_ic,
    top_k_hit_rate,
)

__all__ = [
    # v1 targets
    "compute_all_targets",
    "future_1d_return",
    "future_5d_return",
    "future_21d_return",
    "forward_volatility",
    "validate_feature_leakage",
    # v2 features
    "build_feature_set_v2",
    "feature_summary",
    "get_feature_columns",
    # ranking targets & metrics
    "add_ranking_targets",
    "rank_ic",
    "spearman_ic",
    "mean_rank_ic",
    "precision_at_k",
    "top_k_hit_rate",
    "compute_all_rank_metrics",
]
