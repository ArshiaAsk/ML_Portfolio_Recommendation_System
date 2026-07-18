"""Reusable walk-forward two-stage portfolio loop."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets
from portfolio_ml.modeling.ml_models.factory import build_model
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
from portfolio_ml.modeling.two_stage_portfolio import compute_weights_from_scores, normalise_scores_cross_sectionally, TwoStageConfig

def run_two_stage_walk_forward(prices_wide, model_name="gradient_boost", model_params=None,
                               ts_config=None, horizon=21, lookback=252, rebalance_freq=21,
                               model_factory=build_model):
    config = ts_config or TwoStageConfig(rebalance_frequency=rebalance_freq)
    features = build_features_and_targets(prices_wide, horizon=horizon)
    target = f"cs_rank_{horizon}d"
    excluded = {"date", "symbol", f"future_return_{horizon}d", target, f"cs_class_3_{horizon}d", f"cs_class_binary_{horizon}d"}
    cols = [c for c in features if c not in excluded and pd.api.types.is_numeric_dtype(features[c])]
    dates = features.date.drop_duplicates().sort_values().reset_index(drop=True)
    assets = list(prices_wide.columns); schedule = {}
    for train_idx, test_idx in WalkForwardSplitter(lookback, rebalance_freq).split(dates):
        train = features[features.date.isin(set(dates.iloc[train_idx]))].dropna(subset=[target])
        test_date = dates.iloc[test_idx[0]]; test = features[features.date == test_date]
        if len(train) < 20 or test.empty: schedule[test_date] = np.ones(len(assets))/len(assets); continue
        X = train[cols]; mask = X.notna().all(axis=1); X = X[mask]
        scaler = TrainOnlyScaler(StandardScaler()); model = model_factory(model_name, **(model_params or {}))
        model.fit(np.nan_to_num(scaler.fit_transform(X), nan=0), train.loc[mask, target].to_numpy())
        scores = pd.Series(model.predict(np.nan_to_num(scaler.transform(test[cols].fillna(X.mean())), nan=0)), index=test.symbol)
        history = prices_wide.pct_change().iloc[-config.cov_window:] if config.portfolio_method == "mean_variance" else None
        schedule[test_date] = compute_weights_from_scores(normalise_scores_cross_sectionally(scores), assets, config, history)
    return schedule
