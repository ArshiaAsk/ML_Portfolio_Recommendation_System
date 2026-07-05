"""Tests for experiment configuration loading."""

from portfolio_ml.config.loader import load_experiment_config


def test_load_experiment_config():
    """Experiment config should expose the expected grid and output settings."""
    cfg = load_experiment_config()

    assert cfg.experiment_name == "baseline_grid"
    assert cfg.lookback_windows == [126, 252, 504]
    assert cfg.rebalance_frequencies == [5, 21, 63]
    assert cfg.strategies == ["EqualWeight", "RiskParity", "MinVariance"]
    assert cfg.transaction_cost == 0.0005
    assert cfg.initial_capital == 1_000_000.0
    assert cfg.paths.output_dir == "outputs/phase3_experiment_grid"
    assert cfg.mlflow.experiment_name == "portfolio_baseline_grid"
