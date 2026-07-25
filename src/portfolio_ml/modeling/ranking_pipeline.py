"""Reusable feature construction and walk-forward ranking evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from portfolio_ml.features.feature_set_v2 import build_feature_set_v2
from portfolio_ml.features.ranking_targets import (
    add_ranking_targets,
    precision_at_k,
    rank_ic,
)
from portfolio_ml.modeling.ml_models.factory import build_model
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter


def build_features_and_targets(
    prices_wide: pd.DataFrame,
    horizon: int = 21,
    feature_set: str = "v2",
) -> pd.DataFrame:
    """Build feature set v2 and attach cross-sectional ranking targets."""
    if feature_set != "v2":
        raise ValueError("Only feature set v2 is supported by the Phase 3.3 pipeline")

    features = build_feature_set_v2(prices_wide)
    labeled = add_ranking_targets(features, horizon=horizon, price_col="adj_close")
    return labeled.drop(columns=["adj_close"], errors="ignore")


def _feature_columns(features_df: pd.DataFrame, horizon: int) -> list[str]:
    """Numeric columns excluding identifiers and label targets."""
    excluded = {
        "date",
        "symbol",
        f"future_return_{horizon}d",
        f"cs_rank_{horizon}d",
        f"cs_class_3_{horizon}d",
        f"cs_class_binary_{horizon}d",
    }
    return [
        col
        for col in features_df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features_df[col])
    ]


def run_walk_forward_ranking(
    features_df: pd.DataFrame,
    model_name: str = "gradient_boost",
    model_params: dict | None = None,
    horizon: int = 21,
    lookback: int = 252,
    rebalance_freq: int = 21,
    top_k: int = 3,
) -> tuple[list[dict], pd.DataFrame]:
    """Run walk-forward ranking folds with train-only scaling.

    Returns:
        fold_results: per-fold Rank IC / Precision@K summary rows
        predictions: concatenated out-of-sample prediction frame
    """
    target = f"future_return_{horizon}d"
    rank_target = f"cs_rank_{horizon}d"
    columns = _feature_columns(features_df, horizon)
    dates = features_df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    fold_results: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    params = model_params or {}
    splitter = WalkForwardSplitter(lookback, rebalance_freq)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dates)):
        train = features_df[
            features_df["date"].isin(set(dates.iloc[train_idx]))
        ].dropna(subset=[rank_target])
        test = features_df[features_df["date"].isin(set(dates.iloc[test_idx]))]

        if len(train) < 20 or test.empty:
            continue

        X_train = train[columns]
        valid = X_train.notna().all(axis=1)
        X_train = X_train.loc[valid]
        y_train = train.loc[valid, rank_target]

        scaler = TrainOnlyScaler(StandardScaler())
        X_scaled = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
        model = build_model(model_name, **params).fit(X_scaled, y_train.to_numpy())

        train_means = X_train.mean()
        X_test = test[columns].fillna(train_means)
        predicted = model.predict(np.nan_to_num(scaler.transform(X_test), nan=0.0))

        pred_frame = test[["date", "symbol", target]].copy()
        pred_frame["predicted_score"] = predicted
        pred_frame["fold"] = fold
        prediction_frames.append(pred_frame)

        ics: list[float] = []
        paks: list[float] = []
        for _, group in pred_frame.dropna(subset=[target]).groupby("date"):
            if len(group) < 2:
                continue
            actual = group[target].to_numpy()
            scores = group["predicted_score"].to_numpy()
            ics.append(rank_ic(actual, scores))
            paks.append(precision_at_k(actual, scores, top_k))

        fold_results.append(
            {
                "fold": fold,
                "train_start": str(min(train["date"]).date()),
                "train_end": str(max(train["date"]).date()),
                "test_start": str(min(test["date"]).date()),
                "test_end": str(max(test["date"]).date()),
                "mean_rank_ic": float(np.mean(ics)) if ics else 0.0,
                f"precision_at_{top_k}": float(np.mean(paks)) if paks else 0.0,
                "n_train_rows": len(X_train),
                "n_test_rows": len(test),
                "n_train_assets": int(train["symbol"].nunique()),
                "n_test_assets": int(test["symbol"].nunique()),
                "feature_missingness": float(train[columns].isna().mean().mean()),
            }
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return fold_results, predictions
