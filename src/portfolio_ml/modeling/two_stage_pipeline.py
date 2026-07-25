"""Reusable walk-forward two-stage portfolio loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from portfolio_ml.modeling.ml_models.factory import build_model
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.two_stage_portfolio import (
    TwoStageConfig,
    compute_weights_from_scores,
    normalise_scores_cross_sectionally,
)
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter


def run_two_stage_walk_forward(
    prices_wide: pd.DataFrame,
    model_name: str = "gradient_boost",
    model_params: dict | None = None,
    ts_config: TwoStageConfig | None = None,
    horizon: int = 21,
    lookback: int = 252,
    rebalance_freq: int = 21,
    model_factory: Callable[..., Any] = build_model,
) -> dict[pd.Timestamp, np.ndarray]:
    """Fit ranking models walk-forward and map scores to portfolio weights.

    Returns a rebalance schedule mapping each test date to an asset-weight
    vector aligned with ``prices_wide.columns``.
    """
    config = ts_config or TwoStageConfig(rebalance_frequency=rebalance_freq)
    features = build_features_and_targets(prices_wide, horizon=horizon)
    target = f"cs_rank_{horizon}d"
    excluded = {
        "date",
        "symbol",
        f"future_return_{horizon}d",
        target,
        f"cs_class_3_{horizon}d",
        f"cs_class_binary_{horizon}d",
    }
    columns = [
        col
        for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    ]
    dates = features["date"].drop_duplicates().sort_values().reset_index(drop=True)
    assets = list(prices_wide.columns)
    equal_weight = np.ones(len(assets)) / len(assets)
    schedule: dict[pd.Timestamp, np.ndarray] = {}
    params = model_params or {}

    for train_idx, test_idx in WalkForwardSplitter(lookback, rebalance_freq).split(dates):
        train = features[
            features["date"].isin(set(dates.iloc[train_idx]))
        ].dropna(subset=[target])
        test_date = dates.iloc[test_idx[0]]
        test = features[features["date"] == test_date]

        if len(train) < 20 or test.empty:
            schedule[test_date] = equal_weight.copy()
            continue

        X_train = train[columns]
        valid = X_train.notna().all(axis=1)
        X_train = X_train.loc[valid]
        y_train = train.loc[valid, target]

        scaler = TrainOnlyScaler(StandardScaler())
        model = model_factory(model_name, **params)
        model.fit(
            np.nan_to_num(scaler.fit_transform(X_train), nan=0.0),
            y_train.to_numpy(),
        )

        X_test = test[columns].fillna(X_train.mean())
        scores = pd.Series(
            model.predict(np.nan_to_num(scaler.transform(X_test), nan=0.0)),
            index=test["symbol"],
        )

        history = None
        if config.portfolio_method == "mean_variance":
            history = prices_wide.pct_change().iloc[-config.cov_window :]

        schedule[test_date] = compute_weights_from_scores(
            normalise_scores_cross_sectionally(scores),
            assets,
            config,
            history,
        )

    return schedule
