import pandas as pd

from portfolio_ml.modeling.walk_forward import WalkForwardSplitter


def test_split_auto_reduces_lookback_to_available_history():
    dates = pd.date_range("2018-01-02", periods=251, freq="B")
    splitter = WalkForwardSplitter(lookback_window=252, rebalance_freq=21)

    splits = splitter.split(dates)

    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 230
    assert len(test_idx) == 21
