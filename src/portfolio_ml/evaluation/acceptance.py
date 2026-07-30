"""Acceptance policy for Phase 3.3 research finalists.

Criteria are declared before holdout inspection. Beating the plain model is
never sufficient; EqualWeight after costs is the primary benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Pre-declared gates for promoting a candidate to holdout finalist."""

    min_return_edge: float = 0.0
    min_sharpe_edge: float = 0.0
    max_drawdown_tolerance: float = 0.02
    max_avg_turnover: float = 0.10
    min_window_win_rate: float = 0.50
    require_positive_bootstrap_diff: bool = True
    max_assets_for_top_k_equal: int | None = None
    governance_note: str = (
        "Do not retune against the prior failed holdout. "
        "EqualWeight after costs is the decision benchmark."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def decide_candidate(
    *,
    candidate_id: str,
    metrics: dict,
    equal_weight_metrics: dict,
    window_win_rate: float,
    bootstrap: dict,
    n_assets: int,
    criteria: AcceptanceCriteria,
) -> dict:
    """Assign reject / keep / finalist using pre-declared acceptance criteria."""
    return_edge = float(metrics["annualised_return"]) - float(
        equal_weight_metrics["annualised_return"]
    )
    sharpe_edge = float(metrics["sharpe_ratio"]) - float(
        equal_weight_metrics["sharpe_ratio"]
    )
    drawdown_gap = float(metrics["max_drawdown"]) - float(
        equal_weight_metrics["max_drawdown"]
    )
    turnover = float(metrics["avg_turnover"])
    top_k = int(metrics.get("top_k", 0) or 0)
    method = str(metrics.get("portfolio_method", ""))

    checks = {
        "return_exceeds_equal_weight": return_edge > criteria.min_return_edge,
        "sharpe_exceeds_equal_weight": sharpe_edge > criteria.min_sharpe_edge,
        "drawdown_within_tolerance": drawdown_gap >= -criteria.max_drawdown_tolerance,
        "turnover_acceptable": turnover <= criteria.max_avg_turnover,
        "window_stability": window_win_rate >= criteria.min_window_win_rate,
        "bootstrap_non_negative": (
            float(bootstrap.get("observed_diff", 0.0)) > 0.0
            if criteria.require_positive_bootstrap_diff
            else True
        ),
        "not_trivial_equal_weight": not (
            method == "top_k_equal"
            and criteria.max_assets_for_top_k_equal is not None
            and top_k >= criteria.max_assets_for_top_k_equal
        ),
    }

    failures = [name for name, passed in checks.items() if not passed]

    if not failures:
        decision = "finalist_for_holdout_confirmation"
    elif checks["return_exceeds_equal_weight"] or checks["sharpe_exceeds_equal_weight"]:
        decision = "keep_for_further_internal_validation"
    else:
        decision = "reject"

    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "return_edge": return_edge,
        "sharpe_edge": sharpe_edge,
        "drawdown_gap": drawdown_gap,
        "avg_turnover": turnover,
        "window_win_rate": window_win_rate,
        "bootstrap_observed_diff": float(bootstrap.get("observed_diff", 0.0)),
        "bootstrap_p_value": float(bootstrap.get("p_value", 1.0)),
        "checks": checks,
        "failures": failures,
    }
