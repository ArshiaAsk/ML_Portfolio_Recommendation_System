"""Significance tests for walk-forward model selection."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ic_ttest(ics, alpha: float = 0.05) -> dict:
    """One-sided t-test that mean Rank IC is greater than zero."""
    values = np.asarray(list(ics), dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return {
            "mean_ic": float(np.mean(values)) if len(values) else 0.0,
            "t_stat": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "n": int(len(values)),
            "alpha": alpha,
        }

    result = stats.ttest_1samp(values, 0.0, alternative="greater")
    return {
        "mean_ic": float(values.mean()),
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "n": int(len(values)),
        "alpha": alpha,
    }


def paired_bootstrap_sharpe_diff(
    strategy_returns,
    baseline_returns,
    block: int = 21,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Block-bootstrap the annualised Sharpe difference of paired returns."""
    strategy = np.asarray(strategy_returns, dtype=float)
    baseline = np.asarray(baseline_returns, dtype=float)
    n = min(len(strategy), len(baseline))
    strategy, baseline = strategy[:n], baseline[:n]

    valid = np.isfinite(strategy) & np.isfinite(baseline)
    strategy, baseline = strategy[valid], baseline[valid]

    if len(strategy) < 2:
        return {"observed_diff": 0.0, "p_value": 1.0, "significant": False}

    def sharpe(returns: np.ndarray) -> float:
        std = np.std(returns, ddof=1)
        if std <= 0:
            return 0.0
        return float(np.sqrt(252) * np.mean(returns) / std)

    observed = sharpe(strategy) - sharpe(baseline)
    blocks = [
        np.arange(start, min(start + block, len(strategy)))
        for start in range(0, len(strategy), block)
    ]
    rng = np.random.default_rng(random_state)

    diffs = []
    for _ in range(n_resamples):
        sample_idx = np.concatenate(
            [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        )[: len(strategy)]
        diffs.append(sharpe(strategy[sample_idx]) - sharpe(baseline[sample_idx]))

    diffs = np.asarray(diffs)
    p_value = float(np.mean(np.abs(diffs) >= abs(observed)))
    return {
        "observed_diff": float(observed),
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "n": int(len(strategy)),
        "block": block,
        "n_resamples": n_resamples,
        "alpha": alpha,
    }
