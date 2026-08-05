"""Tests for Phase 3.3 Cycle 3 turnover-reduction mechanisms.

Covers membership hysteresis, the no-trade band, config validation, and the
relaxed acceptance gate. These verify the controls do what they claim before
any conclusion is drawn from aggregate backtest metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_ml.evaluation.acceptance import AcceptanceCriteria, decide_candidate
from portfolio_ml.modeling.phase3_3_evaluation import (
    RiskControlConfig,
    _apply_turnover_penalty,
    apply_membership_hysteresis,
)
from portfolio_ml.modeling.two_stage_portfolio import compute_turnover

ASSETS = ["A", "B", "C", "D", "E", "F"]


def _weights(held: dict[str, float]) -> np.ndarray:
    return np.array([held.get(a, 0.0) for a in ASSETS], dtype=float)


# ---------------------------------------------------------------------------
# Membership hysteresis
# ---------------------------------------------------------------------------


class TestMembershipHysteresis:
    def test_first_rebalance_is_plain_top_k(self):
        """With no incumbents, hysteresis must reduce to plain top-k."""
        scores = pd.Series([0.9, 0.8, 0.7, 0.4, 0.3, 0.1], index=ASSETS)
        out = apply_membership_hysteresis(
            scores, ASSETS, None, top_k=3, entry_buffer=0.05, exit_buffer=0.10
        )
        selected = [a for a in ASSETS if out[a] > -1.0]
        assert selected == ["A", "B", "C"]

    def test_selects_exactly_top_k(self):
        scores = pd.Series([0.9, 0.8, 0.7, 0.4, 0.3, 0.1], index=ASSETS)
        prev = _weights({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        out = apply_membership_hysteresis(
            scores, ASSETS, prev, top_k=3, entry_buffer=0.05, exit_buffer=0.10
        )
        assert sum(out[a] > -1.0 for a in ASSETS) == 3

    def test_incumbent_retained_within_exit_buffer(self):
        """A holding that slips just below the cutoff is not sold."""
        # Cutoff (3rd best) = 0.70. Incumbent C at 0.65 is within exit_buffer.
        scores = pd.Series(
            {"A": 0.95, "B": 0.90, "C": 0.65, "D": 0.70, "E": 0.30, "F": 0.10}
        )
        prev = _weights({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        out = apply_membership_hysteresis(
            scores, ASSETS, prev, top_k=3, entry_buffer=0.05, exit_buffer=0.10
        )
        assert out["C"] > -1.0, "incumbent within exit buffer should be retained"
        assert out["D"] == -1.0, "challenger not clearing entry buffer should stay out"

    def test_incumbent_dropped_beyond_exit_buffer(self):
        """A holding that decays well past the buffer is sold."""
        scores = pd.Series(
            {"A": 0.95, "B": 0.90, "C": 0.20, "D": 0.80, "E": 0.30, "F": 0.10}
        )
        prev = _weights({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        out = apply_membership_hysteresis(
            scores, ASSETS, prev, top_k=3, entry_buffer=0.05, exit_buffer=0.10
        )
        assert out["C"] == -1.0, "decayed incumbent should be dropped"
        assert out["D"] > -1.0, "clear challenger should enter"

    def test_hysteresis_reduces_turnover_vs_plain_top_k(self):
        """The core claim: hysteresis cuts turnover on a churning signal."""
        rng = np.random.default_rng(0)
        k = 3
        plain_prev = None
        hyst_prev = None
        plain_turnover = 0.0
        hyst_turnover = 0.0

        for _ in range(40):
            # Persistent ranking plus noise: marginal names jitter near cutoff.
            base = np.array([0.9, 0.75, 0.60, 0.55, 0.40, 0.2])
            scores = pd.Series(
                np.clip(base + rng.normal(0, 0.08, len(ASSETS)), 0, 1), index=ASSETS
            )

            plain = apply_membership_hysteresis(
                scores, ASSETS, None, top_k=k, entry_buffer=0.0, exit_buffer=0.0
            )
            plain_w = np.array(
                [1.0 / k if plain[a] > -1.0 else 0.0 for a in ASSETS]
            )

            hyst = apply_membership_hysteresis(
                scores, ASSETS, hyst_prev, top_k=k, entry_buffer=0.05, exit_buffer=0.10
            )
            hyst_w = np.array([1.0 / k if hyst[a] > -1.0 else 0.0 for a in ASSETS])

            if plain_prev is not None:
                plain_turnover += compute_turnover(plain_prev, plain_w)
                hyst_turnover += compute_turnover(hyst_prev, hyst_w)

            plain_prev, hyst_prev = plain_w, hyst_w

        assert hyst_turnover < plain_turnover, (
            f"hysteresis turnover {hyst_turnover:.3f} should be below "
            f"plain top-k {plain_turnover:.3f}"
        )

    def test_book_never_under_invested(self):
        """Even when few names qualify, exactly k slots are filled."""
        # All challengers far below cutoff; incumbents all decayed.
        scores = pd.Series(
            {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05, "F": 0.06}
        )
        prev = _weights({"A": 0.5, "B": 0.5})
        out = apply_membership_hysteresis(
            scores, ASSETS, prev, top_k=4, entry_buffer=0.20, exit_buffer=0.01
        )
        assert sum(out[a] > -1.0 for a in ASSETS) == 4

    def test_excluded_sentinel_is_finite(self):
        """Sentinel must stay finite so mean-variance normalisation is safe."""
        scores = pd.Series([0.9, 0.8, 0.7, 0.4, 0.3, 0.1], index=ASSETS)
        out = apply_membership_hysteresis(
            scores, ASSETS, None, top_k=2, entry_buffer=0.05, exit_buffer=0.05
        )
        assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------------------
# No-trade band
# ---------------------------------------------------------------------------


class TestTurnoverPenalty:
    def test_disabled_penalty_is_identity(self):
        prev = _weights({"A": 0.5, "B": 0.5})
        target = _weights({"A": 0.6, "B": 0.4})
        out = _apply_turnover_penalty(target, prev, 0.0)
        np.testing.assert_allclose(out, target)

    def test_no_previous_book_is_identity(self):
        target = _weights({"A": 0.6, "B": 0.4})
        out = _apply_turnover_penalty(target, None, 0.05)
        np.testing.assert_allclose(out, target)

    def test_small_trades_suppressed(self):
        """Trades under the band are dropped, so the book barely moves."""
        prev = _weights({"A": 0.5, "B": 0.5})
        target = _weights({"A": 0.51, "B": 0.49})  # 1% trade
        out = _apply_turnover_penalty(target, prev, 0.05)
        np.testing.assert_allclose(out, prev, atol=1e-9)

    def test_large_trades_execute_net_of_band(self):
        prev = _weights({"A": 0.5, "B": 0.5})
        target = _weights({"A": 0.9, "B": 0.1})
        out = _apply_turnover_penalty(target, prev, 0.05)
        assert out[0] > prev[0], "large trade should still move the book"
        assert out[0] < target[0], "and should execute net of the band"

    def test_not_equivalent_to_proportional_shrinkage(self):
        """Guards against collapsing back into sticky_fraction behaviour."""
        prev = _weights({"A": 0.5, "B": 0.3, "C": 0.2})
        # One large trade, one tiny trade.
        target = _weights({"A": 0.8, "B": 0.19, "C": 0.01})
        band = _apply_turnover_penalty(target, prev, 0.05)

        proportional = prev + (target - prev) * (1 - 0.05)
        proportional = proportional / proportional.sum()

        assert not np.allclose(band, proportional, atol=1e-6), (
            "no-trade band must differ from proportional shrinkage"
        )

    def test_output_is_normalised_and_long_only(self):
        prev = _weights({"A": 0.5, "B": 0.5})
        target = _weights({"A": 0.95, "B": 0.05})
        out = _apply_turnover_penalty(target, prev, 0.03)
        assert out.min() >= 0.0
        assert out.sum() == pytest.approx(1.0)

    def test_reduces_turnover_on_noisy_targets(self):
        rng = np.random.default_rng(7)
        prev_a = prev_b = np.ones(len(ASSETS)) / len(ASSETS)
        raw_turnover = banded_turnover = 0.0

        for _ in range(50):
            target = np.clip(
                np.ones(len(ASSETS)) / len(ASSETS) + rng.normal(0, 0.01, len(ASSETS)),
                0,
                None,
            )
            target = target / target.sum()
            banded = _apply_turnover_penalty(target, prev_b, 0.02)
            raw_turnover += compute_turnover(prev_a, target)
            banded_turnover += compute_turnover(prev_b, banded)
            prev_a, prev_b = target, banded

        assert banded_turnover < raw_turnover


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestRiskControlValidation:
    def test_defaults_disable_new_controls(self):
        cfg = RiskControlConfig()
        assert cfg.entry_buffer == 0.0
        assert cfg.exit_buffer == 0.0
        assert cfg.turnover_penalty == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sticky_fraction": 1.5},
            {"sticky_fraction": -0.1},
            {"entry_buffer": -0.01},
            {"exit_buffer": -0.01},
            {"turnover_penalty": -0.01},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            RiskControlConfig(**kwargs)


# ---------------------------------------------------------------------------
# Relaxed acceptance gate
# ---------------------------------------------------------------------------


class TestCycle3AcceptanceGate:
    @staticmethod
    def _metrics(turnover: float) -> dict:
        return {
            "annualised_return": 0.18,
            "sharpe_ratio": 1.20,
            "max_drawdown": -0.26,
            "avg_turnover": turnover,
            "portfolio_method": "top_k_equal",
            "top_k": 8,
        }

    @staticmethod
    def _ew() -> dict:
        return {
            "annualised_return": 0.124,
            "sharpe_ratio": 0.845,
            "max_drawdown": -0.268,
            "avg_turnover": 0.012,
        }

    def _decide(self, turnover: float, max_turnover: float) -> dict:
        return decide_candidate(
            candidate_id="T",
            metrics=self._metrics(turnover),
            equal_weight_metrics=self._ew(),
            window_win_rate=1.0,
            bootstrap={"observed_diff": 0.35, "p_value": 0.2},
            n_assets=len(ASSETS),
            criteria=AcceptanceCriteria(max_avg_turnover=max_turnover),
        )

    def test_cycle2_gate_rejects_13_percent(self):
        """Reproduces the Cycle 2 failure mode at the old 10% gate."""
        d = self._decide(0.13, 0.10)
        assert "turnover_acceptable" in d["failures"]

    def test_cycle3_gate_admits_13_percent(self):
        """The relaxed 15% gate lets an otherwise-passing candidate through."""
        d = self._decide(0.13, 0.15)
        assert d["failures"] == []
        assert d["decision"] == "finalist_for_holdout_confirmation"

    def test_relaxed_gate_still_rejects_high_turnover(self):
        d = self._decide(0.28, 0.15)
        assert "turnover_acceptable" in d["failures"]
        assert d["decision"] != "finalist_for_holdout_confirmation"
