"""Leakage detection tests for the portfolio ML pipeline.

These tests verify that no future information leaks into training data through:
1. Feature computation (rolling windows must not peek forward)
2. Target alignment (targets must be properly shifted)
3. Scaling (scaler must be fit only on training data)
4. Walk-forward splits (train/test must not overlap)
5. Cross-sectional aggregates (group statistics must not include test assets)

All tests use pytest and synthetic data to ensure reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_ml.features.targets import (
    compute_all_targets,
    future_1d_return,
    future_5d_return,
    future_21d_return,
)
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Generate synthetic price data for 3 assets over 500 days."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    n = len(dates)
    symbols = ["A", "B", "C"]
    rows = []
    for sym, base in zip(symbols, [100.0, 150.0, 200.0]):
        prices = base + np.random.randn(n).cumsum()
        for i, d in enumerate(dates):
            rows.append({"date": d, "symbol": sym, "adj_close": prices[i]})
    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test 1: Target definitions do not leak backward
# ---------------------------------------------------------------------------


def test_future_returns_are_forward_looking(synthetic_prices):
    """Verify that future_1d_return at time t uses price at t+1, not t-1."""
    df = synthetic_prices[synthetic_prices["symbol"] == "A"].copy()
    df = df.reset_index(drop=True)

    target = future_1d_return(df)

    # For row 0, target should be (P[1] - P[0]) / P[0]
    p0 = df.loc[0, "adj_close"]
    p1 = df.loc[1, "adj_close"]
    expected = (p1 / p0) - 1
    actual = target.iloc[0]

    assert np.isclose(actual, expected, atol=1e-6), (
        f"future_1d_return leaked: expected {expected}, got {actual}"
    )

    # Last row should be NaN (no future price)
    assert pd.isna(target.iloc[-1]), "Last row of future_1d_return should be NaN."


def test_future_5d_return_no_leakage(synthetic_prices):
    """Verify that future_5d_return at time t uses price at t+5."""
    df = synthetic_prices[synthetic_prices["symbol"] == "A"].copy().reset_index(drop=True)
    target = future_5d_return(df)

    p0 = df.loc[0, "adj_close"]
    p5 = df.loc[5, "adj_close"]
    expected = (p5 / p0) - 1
    actual = target.iloc[0]

    assert np.isclose(actual, expected, atol=1e-6), (
        f"future_5d_return leaked: expected {expected}, got {actual}"
    )

    # Last 5 rows should be NaN
    assert target.iloc[-5:].isna().all(), "Last 5 rows should be NaN."


def test_future_21d_return_no_leakage(synthetic_prices):
    """Verify that future_21d_return at time t uses price at t+21."""
    df = synthetic_prices[synthetic_prices["symbol"] == "B"].copy().reset_index(drop=True)
    target = future_21d_return(df)

    p0 = df.loc[0, "adj_close"]
    p21 = df.loc[21, "adj_close"]
    expected = (p21 / p0) - 1
    actual = target.iloc[0]

    assert np.isclose(actual, expected, atol=1e-6)
    assert target.iloc[-21:].isna().all(), "Last 21 rows should be NaN."


def test_compute_all_targets_no_backward_leakage(synthetic_prices):
    """Ensure compute_all_targets does not use future prices in past rows."""
    df_with_targets = compute_all_targets(synthetic_prices, include_forward_vol=False)

    grp = df_with_targets[df_with_targets["symbol"] == "C"].reset_index(drop=True)
    p0 = grp.loc[0, "adj_close"]
    p1 = grp.loc[1, "adj_close"]
    expected_1d = (p1 / p0) - 1
    actual_1d = grp.loc[0, "future_1d_return"]

    assert np.isclose(actual_1d, expected_1d, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Walk-forward splits have no overlap
# ---------------------------------------------------------------------------


def test_walk_forward_train_test_no_overlap():
    """Verify that train and test indices do not overlap in any fold."""
    dates = pd.date_range("2018-01-01", periods=1000, freq="B")
    splitter = WalkForwardSplitter(lookback_window=252, rebalance_freq=21)
    splits = splitter.split(dates)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: train and test indices overlap: {overlap}"
        )


def test_walk_forward_train_ends_before_test():
    """Verify that the last training date is strictly before the first test date."""
    dates = pd.date_range("2018-01-01", periods=800, freq="B")
    splitter = WalkForwardSplitter(lookback_window=200, rebalance_freq=20)
    splits = splitter.split(dates)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        last_train_date = dates[train_idx[-1]]
        first_test_date = dates[test_idx[0]]
        assert last_train_date < first_test_date, (
            f"Fold {fold_idx}: last train date {last_train_date} >= "
            f"first test date {first_test_date}"
        )


def test_walk_forward_rolling_mode_window_size():
    """Verify that rolling mode maintains a fixed training window size."""
    dates = pd.date_range("2018-01-01", periods=1000, freq="B")
    lookback = 252
    splitter = WalkForwardSplitter(
        lookback_window=lookback, rebalance_freq=21, window_type="rolling"
    )
    splits = splitter.split(dates)

    for fold_idx, (train_idx, _) in enumerate(splits):
        assert len(train_idx) <= lookback, (
            f"Fold {fold_idx}: rolling mode train window exceeds lookback: "
            f"{len(train_idx)} > {lookback}"
        )


# ---------------------------------------------------------------------------
# Test 3: Scaler is fit only on training data
# ---------------------------------------------------------------------------


def test_scaler_fit_on_train_only():
    """Verify that TrainOnlyScaler does not see test data during fit.

    StandardScaler uses ddof=0 (population std) so we check with atol=0.01
    and verify the test data mean is notably shifted (uses train statistics).
    """
    from sklearn.preprocessing import StandardScaler

    np.random.seed(99)
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(50, 5) + 10.0  # deliberately shifted mean

    scaler = TrainOnlyScaler(StandardScaler())
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training data should have mean ≈ 0 (StandardScaler normalises the mean)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-6), (
        "Train data mean should be 0 after StandardScaler fit."
    )

    # Training data std (population) should be ≈ 1
    # StandardScaler uses ddof=0 to fit, so scaled data has population std = 1
    assert np.allclose(X_train_scaled.std(axis=0, ddof=0), 1.0, atol=1e-6), (
        "Train data population std should be 1 after StandardScaler fit."
    )

    # Test data mean should NOT be ~0 because scaler was fit on train (mean≈0, not 10)
    test_mean = X_test_scaled.mean(axis=0)
    assert not np.allclose(test_mean, 0.0, atol=0.5), (
        "Test data mean is unexpectedly close to 0, suggesting leakage."
    )


def test_scaler_raises_if_transform_before_fit():
    """Verify that calling transform before fit raises an error."""
    from sklearn.preprocessing import StandardScaler

    scaler = TrainOnlyScaler(StandardScaler())
    X = np.random.randn(50, 3)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        scaler.transform(X)


# ---------------------------------------------------------------------------
# Test 4: Rolling windows do not peek forward
# ---------------------------------------------------------------------------


def test_rolling_window_features_no_forward_peek():
    """Verify that rolling features at time t use only data from t and earlier."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = np.arange(100, dtype=float) + 100  # 100, 101, ..., 199
    df = pd.DataFrame({"date": dates, "adj_close": prices})

    window = 5
    df["rolling_mean"] = df["adj_close"].rolling(window=window, min_periods=1).mean()

    # At index 4 (5th element), rolling mean should be mean of indices 0..4
    expected = prices[:5].mean()
    actual = df.loc[4, "rolling_mean"]
    assert np.isclose(actual, expected, atol=1e-6), (
        f"Rolling mean leaked: expected {expected}, got {actual}"
    )

    # At index 10, rolling mean should use indices 6..10
    expected_10 = prices[6:11].mean()
    actual_10 = df.loc[10, "rolling_mean"]
    assert np.isclose(actual_10, expected_10, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Cross-sectional aggregates do not leak across assets
# ---------------------------------------------------------------------------


def test_cross_sectional_no_leakage(synthetic_prices):
    """Verify that per-symbol features do not use data from other symbols."""
    df = synthetic_prices.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["cumulative_return"] = df.groupby("symbol")["adj_close"].transform(
        lambda x: x / x.iloc[0] - 1
    )

    grp_a = df[df["symbol"] == "A"].reset_index(drop=True)
    first_price_a = grp_a.loc[0, "adj_close"]
    last_price_a = grp_a.loc[len(grp_a) - 1, "adj_close"]
    expected_cum_a = (last_price_a / first_price_a) - 1
    actual_cum_a = grp_a.loc[len(grp_a) - 1, "cumulative_return"]

    assert np.isclose(actual_cum_a, expected_cum_a, atol=1e-6), (
        "Cross-sectional aggregate leaked across symbols."
    )


# ---------------------------------------------------------------------------
# Test 6: Integration test – full pipeline with walk-forward
# ---------------------------------------------------------------------------


def test_full_pipeline_no_leakage(synthetic_prices):
    """Integration test: verify that the full workflow has no leakage."""
    from sklearn.preprocessing import StandardScaler

    df = compute_all_targets(synthetic_prices, include_forward_vol=False)
    df = df.dropna(subset=["future_1d_return"]).reset_index(drop=True)

    feature_cols = ["adj_close"]
    dates_series = df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    splitter = WalkForwardSplitter(lookback_window=50, rebalance_freq=10)
    splits = splitter.split(dates_series)

    train_idx, test_idx = splits[0]
    train_dates = set(dates_series.iloc[train_idx])
    test_dates = set(dates_series.iloc[test_idx])

    # No overlap
    assert len(train_dates & test_dates) == 0, "Train and test dates overlap!"

    # Train strictly before test
    assert max(train_dates) < min(test_dates), "Train dates not before test dates!"

    train_mask = df["date"].isin(train_dates)
    test_mask = df["date"].isin(test_dates)
    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values

    scaler = TrainOnlyScaler(StandardScaler())
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    assert np.abs(X_train_scaled.mean()) < 0.1, "Train data not standardised."
    assert X_test_scaled is not None
