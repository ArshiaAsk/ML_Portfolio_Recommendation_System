"""Tests for Sprint 4 regime labeling and regime-aware strategies."""

import pandas as pd

from portfolio_ml.modeling.regimes import RegimeLabeler, RegimeAwareStrategy


def test_regime_labeler_classifies_simple_series():
    """The labeler should produce Bull/Bear and HighVol/LowVol labels."""
    prices = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0], index=pd.date_range("2024-01-01", periods=6, freq="D"))
    returns = prices.pct_change().dropna()

    labeler = RegimeLabeler()
    regime = labeler.label(returns)

    assert regime["trend"] == "Bull"
    assert regime["volatility"] == "LowVol"


def test_regime_aware_strategy_selects_baseline():
    """The regime wrapper should select a baseline strategy based on regime labels."""
    strategy = RegimeAwareStrategy()
    weights = strategy.allocate(
        cov_matrix=None,
        regime={"trend": "Bear", "volatility": "HighVol"},
        n_assets=3,
    )

    assert len(weights) == 3
    assert sum(weights) == 1.0
