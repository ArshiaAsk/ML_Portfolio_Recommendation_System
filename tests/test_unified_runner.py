"""Tests for the unified Sprint 1–4 experiment workflow."""

import pandas as pd

from portfolio_ml.config.loader import load_experiment_config
from portfolio_ml.experiments.unified_runner import run_unified_experiment_suite


def test_run_unified_experiment_suite_returns_summary(tmp_path):
    """The unified runner should return a ranked summary for a small synthetic dataset."""
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    assets = ["SPY", "QQQ"]
    rows = []
    for i, date in enumerate(dates):
        for asset in assets:
            base = 100.0 if asset == "SPY" else 90.0
            price = base * (1 + 0.001 * i + (0.002 if asset == "QQQ" else 0.0))
            rows.append(
                {
                    "date": date,
                    "symbol": asset,
                    "adj_close": price,
                    "volume": 1_000_000,
                    "return_1d": None if i == 0 else (price / (base * (1 + 0.001 * (i - 1) + (0.002 if asset == "QQQ" else 0.0))) - 1),
                    "log_return_1d": None if i == 0 else 0.0,
                }
            )

    long_prices = pd.DataFrame(rows)
    config = load_experiment_config()
    config.lookback_windows = [20]
    config.rebalance_frequencies = [5]
    config.strategies = ["EqualWeight", "RiskParity", "MinVariance"]

    summary = run_unified_experiment_suite(
        price_data=long_prices,
        config=config,
        output_dir=tmp_path,
    )

    assert not summary.empty
    assert {"strategy", "score"}.issubset(summary.columns)
    assert summary["strategy"].isin(["EqualWeight", "RiskParity", "MinVariance", "RegimeAware", "MLRanking"]).all()
