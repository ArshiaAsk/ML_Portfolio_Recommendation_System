"""Tests for feature generation transformation."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest
import numpy as np

from portfolio_ml.transformations.build_features import build_asset_daily_features


def test_build_features_basic():
    """Test that build_asset_daily_features produces expected output columns."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 11)],
        "symbol": ["SPY"] * 10,
        "adj_close": [100.0 + i for i in range(10)],
        "volume": [1000000] * 10,
        "return_1d": [None] + [0.01] * 9,
        "log_return_1d": [None] + [np.log(1.01)] * 9,
    })
    
    features = build_asset_daily_features(clean)
    
    assert "return_5d" in features.columns
    assert "return_21d" in features.columns
    assert "return_63d" in features.columns
    assert "volatility_21d" in features.columns
    assert "volatility_63d" in features.columns
    assert "momentum_21d" in features.columns
    assert "momentum_63d" in features.columns
    assert "drawdown" in features.columns
    assert "rolling_volume_21d" in features.columns
    assert "price_to_ma_21" in features.columns
    assert "price_to_ma_63" in features.columns
    assert "target_return_1d" in features.columns
    assert "target_return_5d" in features.columns
    assert "created_at" in features.columns
    assert len(features) == 10


def test_build_features_return_5d():
    """Test that return_5d is computed correctly."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 11)],
        "symbol": ["SPY"] * 10,
        "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "volume": [1000000] * 10,
        "return_1d": [None] + [0.01] * 9,
        "log_return_1d": [None] + [0.01] * 9,
    })
    
    features = build_asset_daily_features(clean)
    
    # First 5 rows should have null return_5d
    assert all(pd.isna(features["return_5d"].iloc[:5]))
    
    # 6th row (index 5): (105 / 100) - 1 = 0.05
    assert abs(features.loc[5, "return_5d"] - 0.05) < 1e-6


def test_build_features_volatility():
    """Test that volatility is computed from rolling window of returns."""
    # Create data with known volatility
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 26)],
        "symbol": ["SPY"] * 25,
        "adj_close": [100.0 + i * 0.5 for i in range(25)],
        "volume": [1000000] * 25,
        "return_1d": [None] + [0.005] * 24,  # Constant return
        "log_return_1d": [None] + [0.005] * 24,
    })
    
    features = build_asset_daily_features(clean)
    
    # Volatility should be computed from row 21 onward (21-day window)
    # With constant returns, volatility should be 0
    assert features["volatility_21d"].iloc[-1] == 0.0


def test_build_features_targets_use_future():
    """Test that target variables use future data correctly."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 11)],
        "symbol": ["SPY"] * 10,
        "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "volume": [1000000] * 10,
        "return_1d": [None] + [0.01] * 9,
        "log_return_1d": [None] + [0.01] * 9,
    })
    
    features = build_asset_daily_features(clean)
    
    # target_return_1d for first row should be (101 / 100) - 1 = 0.01
    assert abs(features.loc[0, "target_return_1d"] - 0.01) < 1e-6
    
    # Last row should have null target_return_1d (no future data)
    assert pd.isna(features.loc[9, "target_return_1d"])


def test_build_features_drawdown():
    """Test that drawdown is computed correctly."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 6)],
        "symbol": ["SPY"] * 5,
        "adj_close": [100.0, 110.0, 105.0, 108.0, 95.0],  # Peak at 110, then drop
        "volume": [1000000] * 5,
        "return_1d": [None, 0.1, -0.045, 0.029, -0.12],
        "log_return_1d": [None, 0.095, -0.046, 0.028, -0.128],
    })
    
    features = build_asset_daily_features(clean)
    
    # At index 2 (price=105), rolling max = 110, drawdown = (105/110) - 1 = -0.045
    expected_dd = (105.0 / 110.0) - 1
    assert abs(features.loc[2, "drawdown"] - expected_dd) < 1e-6
    
    # At index 4 (price=95), rolling max = 110, drawdown = (95/110) - 1 ≈ -0.136
    expected_dd = (95.0 / 110.0) - 1
    assert abs(features.loc[4, "drawdown"] - expected_dd) < 1e-6


def test_build_features_multiple_symbols():
    """Test that features are computed per symbol independently."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 6)] * 2,
        "symbol": ["SPY"] * 5 + ["QQQ"] * 5,
        "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 200.0, 202.0, 204.0, 206.0, 208.0],
        "volume": [1000000] * 10,
        "return_1d": [None, 0.01, 0.01, 0.01, 0.01, None, 0.01, 0.01, 0.01, 0.01],
        "log_return_1d": [None, 0.01, 0.01, 0.01, 0.01, None, 0.01, 0.01, 0.01, 0.01],
    })
    
    features = build_asset_daily_features(clean)
    
    # Should have 10 rows (5 per symbol)
    assert len(features) == 10
    assert features["symbol"].nunique() == 2


def test_build_features_no_lookahead_in_features():
    """Test that features do not use future data."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, i) for i in range(1, 6)],
        "symbol": ["SPY"] * 5,
        "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0],
        "volume": [1000000] * 5,
        "return_1d": [None, 0.01, 0.01, 0.01, 0.01],
        "log_return_1d": [None, 0.01, 0.01, 0.01, 0.01],
    })
    
    features = build_asset_daily_features(clean)
    
    # At index 2 (date 2024-01-03):
    # - return_1d should use data from index 1 and 2 only
    # - return_5d should be null (needs 5 days history)
    # - volatility_21d should be null (insufficient history)
    
    assert not pd.isna(features.loc[2, "return_1d"])
    assert pd.isna(features.loc[2, "return_5d"])  # Not enough history


def test_build_features_missing_columns():
    """Test that missing required columns raises ValueError."""
    clean = pd.DataFrame({
        "date": [date(2024, 1, 1)],
        "symbol": ["SPY"],
        # Missing adj_close, volume, return_1d, log_return_1d
    })
    
    with pytest.raises(ValueError, match="missing required input columns"):
        build_asset_daily_features(clean)
