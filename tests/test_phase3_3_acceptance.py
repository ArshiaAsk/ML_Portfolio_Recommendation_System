"""Tests for Phase 3.3 finalist acceptance policy."""

from portfolio_ml.evaluation.acceptance import AcceptanceCriteria, decide_candidate


def _metrics(**overrides) -> dict:
    base = {
        "annualised_return": 0.12,
        "sharpe_ratio": 0.90,
        "max_drawdown": -0.15,
        "avg_turnover": 0.05,
        "portfolio_method": "top_k_equal",
        "top_k": 7,
    }
    base.update(overrides)
    return base


def _equal(**overrides) -> dict:
    base = {
        "annualised_return": 0.10,
        "sharpe_ratio": 0.80,
        "max_drawdown": -0.14,
        "avg_turnover": 0.01,
    }
    base.update(overrides)
    return base


def test_finalist_when_all_gates_pass() -> None:
    decision = decide_candidate(
        candidate_id="B",
        metrics=_metrics(),
        equal_weight_metrics=_equal(),
        window_win_rate=0.75,
        bootstrap={"observed_diff": 0.2, "p_value": 0.04},
        n_assets=10,
        criteria=AcceptanceCriteria(max_assets_for_top_k_equal=10),
    )
    assert decision["decision"] == "finalist_for_holdout_confirmation"
    assert decision["failures"] == []


def test_reject_when_worse_than_equal_weight() -> None:
    decision = decide_candidate(
        candidate_id="A",
        metrics=_metrics(annualised_return=0.05, sharpe_ratio=0.40),
        equal_weight_metrics=_equal(),
        window_win_rate=0.25,
        bootstrap={"observed_diff": -0.3, "p_value": 0.80},
        n_assets=10,
        criteria=AcceptanceCriteria(max_assets_for_top_k_equal=10),
    )
    assert decision["decision"] == "reject"


def test_rejects_trivial_full_universe_top_k() -> None:
    decision = decide_candidate(
        candidate_id="X",
        metrics=_metrics(top_k=10),
        equal_weight_metrics=_equal(),
        window_win_rate=1.0,
        bootstrap={"observed_diff": 0.1, "p_value": 0.2},
        n_assets=10,
        criteria=AcceptanceCriteria(max_assets_for_top_k_equal=10),
    )
    assert decision["decision"] != "finalist_for_holdout_confirmation"
    assert "not_trivial_equal_weight" in decision["failures"]
