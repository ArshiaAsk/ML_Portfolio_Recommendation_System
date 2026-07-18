"""Reusable feature construction and walk-forward ranking evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from portfolio_ml.features.feature_set_v2 import build_feature_set_v2, get_feature_columns
from portfolio_ml.features.ranking_targets import add_ranking_targets, rank_ic, precision_at_k
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter
from portfolio_ml.modeling.ml_models.factory import build_model

def build_features_and_targets(prices_wide, horizon=21, feature_set="v2"):
    if feature_set != "v2":
        raise ValueError("Only feature set v2 is supported by the Phase 3.3 pipeline")
    out = add_ranking_targets(build_feature_set_v2(prices_wide), horizon=horizon, price_col="adj_close")
    return out.drop(columns=["adj_close"], errors="ignore")

def run_walk_forward_ranking(features_df, model_name="gradient_boost", model_params=None,
                             horizon=21, lookback=252, rebalance_freq=21, top_k=3):
    target, rank_target = f"future_return_{horizon}d", f"cs_rank_{horizon}d"
    excluded = {"date", "symbol", target, rank_target, f"cs_class_3_{horizon}d", f"cs_class_binary_{horizon}d"}
    cols = [c for c in features_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(features_df[c])]
    dates = features_df.date.drop_duplicates().sort_values().reset_index(drop=True)
    results, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(WalkForwardSplitter(lookback, rebalance_freq).split(dates)):
        train = features_df[features_df.date.isin(set(dates.iloc[train_idx]))].dropna(subset=[rank_target])
        test = features_df[features_df.date.isin(set(dates.iloc[test_idx]))]
        if len(train) < 20 or test.empty: continue
        X = train[cols]; mask = X.notna().all(axis=1)
        X = X[mask]; y = train.loc[mask, rank_target]
        scaler = TrainOnlyScaler(StandardScaler()); Xs = np.nan_to_num(scaler.fit_transform(X), nan=0.0)
        model = build_model(model_name, **(model_params or {})).fit(Xs, y.to_numpy())
        means = X.mean(); Xt = test[cols].fillna(means)
        pred = model.predict(np.nan_to_num(scaler.transform(Xt), nan=0.0))
        frame = test[["date", "symbol", target]].copy(); frame["predicted_score"] = pred; frame["fold"] = fold
        predictions.append(frame)
        ics, paks = [], []
        for _, group in frame.dropna(subset=[target]).groupby("date"):
            if len(group) >= 2:
                ics.append(rank_ic(group[target].to_numpy(), group.predicted_score.to_numpy()))
                paks.append(precision_at_k(group[target].to_numpy(), group.predicted_score.to_numpy(), top_k))
        results.append({"fold": fold, "train_start": str(min(train.date).date()), "train_end": str(max(train.date).date()),
                        "test_start": str(min(test.date).date()), "test_end": str(max(test.date).date()),
                        "mean_rank_ic": float(np.mean(ics)) if ics else 0.0,
                        f"precision_at_{top_k}": float(np.mean(paks)) if paks else 0.0,
                        "n_train_rows": len(X), "n_test_rows": len(test),
                        "n_train_assets": int(train.symbol.nunique()), "n_test_assets": int(test.symbol.nunique()),
                        "feature_missingness": float(train[cols].isna().mean().mean())})
    return results, pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
