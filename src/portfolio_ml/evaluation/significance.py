"""Significance tests for walk-forward model selection."""
from __future__ import annotations

import numpy as np
from scipy import stats


def ic_ttest(ics, alpha: float = 0.05) -> dict:
    values = np.asarray(list(ics), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"mean_ic": float(np.mean(values)) if len(values) else 0.0,
                "t_stat": float("nan"), "p_value": 1.0, "significant": False,
                "n": int(len(values)), "alpha": alpha}
    result = stats.ttest_1samp(values, 0.0, alternative="greater")
    return {"mean_ic": float(values.mean()), "t_stat": float(result.statistic),
            "p_value": float(result.pvalue), "significant": bool(result.pvalue < alpha),
            "n": int(len(values)), "alpha": alpha}


def paired_bootstrap_sharpe_diff(strategy_returns, baseline_returns, block: int = 21,
                                 n_resamples: int = 1000, alpha: float = 0.05,
                                 random_state: int = 42) -> dict:
    """Block-bootstrap the annualised Sharpe difference of paired returns."""
    a = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(baseline_returns, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2:
        return {"observed_diff": 0.0, "p_value": 1.0, "significant": False}
    def sharpe(x):
        return float(np.sqrt(252) * np.mean(x) / np.std(x, ddof=1)) if np.std(x, ddof=1) > 0 else 0.0
    observed = sharpe(a) - sharpe(b)
    blocks = [np.arange(i, min(i + block, len(a))) for i in range(0, len(a), block)]
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_resamples):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])[:len(a)]
        diffs.append(sharpe(a[sample]) - sharpe(b[sample]))
    diffs = np.asarray(diffs)
    p_value = float(np.mean(np.abs(diffs) >= abs(observed)))
    return {"observed_diff": float(observed), "p_value": p_value,
            "significant": bool(p_value < alpha), "n": int(len(a)), "block": block,
            "n_resamples": n_resamples, "alpha": alpha}
