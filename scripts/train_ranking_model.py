"""Train a ranking ML model with walk-forward validation and rank-aware metrics.

Phase 3.2 — Ranking Model Training

This script implements the ranking-based ML approach:
1. Load price data from processed parquet files.
2. Build feature set v2 (cross-sectional + TS momentum + regime features).
3. Attach cross-sectional ranking targets (forward 21-day returns → ranks).
4. Run walk-forward training with proper train-only scaling and fitting.
5. Evaluate with rank-aware metrics (Rank IC, Precision@K).
6. Save results to the v2 unified experiment schema.

Usage:
    python scripts/train_ranking_model.py [options]

Options:
    --feature-set      v1 | v2 (default: v2)
    --target-horizon   forward return horizon in days (default: 21)
    --model            ridge | gradient_boost | random_forest (default: gradient_boost)
    --output-dir       output directory (default: outputs/phase3_2)
    --lookback         walk-forward training lookback in days (default: 252)
    --rebalance-freq   rebalance frequency in days (default: 21)
    --top-k            K for Precision@K metric (default: 3)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from portfolio_ml.experiments.schema import (
    EXPERIMENT_SCHEMA_COLUMNS,
    make_experiment_row,
    validate_experiment_dataframe,
)
from portfolio_ml.features.feature_set_v2 import build_feature_set_v2, feature_summary, get_feature_columns
from portfolio_ml.features.ranking_targets import (
    add_ranking_targets,
    compute_all_rank_metrics,
    rank_ic,
)
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ranking ML model with walk-forward validation (Phase 3.2)"
    )
    parser.add_argument("--feature-set", choices=["v1", "v2"], default="v2")
    parser.add_argument("--target-horizon", type=int, default=21)
    parser.add_argument(
        "--model",
        choices=["ridge", "gradient_boost", "random_forest"],
        default="gradient_boost",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/phase3_2")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance-freq", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_price_data_wide() -> pd.DataFrame:
    """Load all processed daily prices and pivot to wide format."""
    print("Loading processed price data...")
    price_files = sorted(glob.glob("data/processed/daily_prices/year=*/daily_prices.parquet"))
    if not price_files:
        raise FileNotFoundError(
            "No price data found in data/processed/daily_prices/. "
            "Run Phase 2 pipeline first."
        )
    dfs = [pd.read_parquet(f) for f in price_files]
    all_data = pd.concat(dfs, ignore_index=True)
    all_data["date"] = pd.to_datetime(all_data["date"])
    prices_wide = all_data.pivot(index="date", columns="symbol", values="adj_close")
    prices_wide = prices_wide.sort_index().dropna(axis=1, how="all")
    print(f"  Price matrix: {prices_wide.shape[0]} days × {prices_wide.shape[1]} assets")
    print(f"  Date range: {prices_wide.index.min().date()} – {prices_wide.index.max().date()}")
    print(f"  Assets: {', '.join(prices_wide.columns.tolist())}")
    return prices_wide


def build_features_and_targets(
    prices_wide: pd.DataFrame,
    horizon: int = 21,
    feature_set: str = "v2",
) -> pd.DataFrame:
    """Build the feature/target DataFrame for model training."""
    print(f"\nBuilding feature set {feature_set} with horizon={horizon}d...")

    # Build v2 features
    features_df = build_feature_set_v2(prices_wide)

    # Note: adj_close is already in features_df from build_feature_set_v2
    # Attach forward ranking targets
    features_df = add_ranking_targets(features_df, horizon=horizon, price_col="adj_close")

    # Drop the adj_close column (not a feature — just used for targets)
    features_df = features_df.drop(columns=["adj_close"], errors="ignore")

    # Summary
    feat_cols = get_feature_columns(features_df)
    target_col = f"future_return_{horizon}d"
    valid_rows = features_df.dropna(subset=[target_col])
    print(f"  Total rows: {len(features_df)}")
    print(f"  Rows with valid targets: {len(valid_rows)}")
    print(f"  Feature columns: {len(feat_cols)}")

    summary = feature_summary(features_df)
    print("\nFeature summary (top 10 by null%):")
    print(summary.sort_values("null_pct", ascending=False).head(10).to_string(index=False))

    return features_df


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _make_model(model_name: str):
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    elif model_name == "gradient_boost":
        return GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    elif model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------


def run_walk_forward_ranking(
    features_df: pd.DataFrame,
    model_name: str = "gradient_boost",
    horizon: int = 21,
    lookback: int = 252,
    rebalance_freq: int = 21,
    top_k: int = 3,
) -> tuple[list[dict], pd.DataFrame]:
    """Run walk-forward ranking model training and evaluation.

    Returns:
        Tuple of (fold_results list, combined predictions DataFrame).
    """
    target_col = f"future_return_{horizon}d"
    cs_rank_col = f"cs_rank_{horizon}d"

    # Identify feature columns (exclude date, symbol, targets)
    non_feature_cols = {
        "date", "symbol",
        f"future_return_{horizon}d",
        f"cs_rank_{horizon}d",
        f"cs_class_3_{horizon}d",
        f"cs_class_binary_{horizon}d",
    }
    all_cols = set(features_df.columns)
    feat_cols = sorted(all_cols - non_feature_cols)
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(features_df[c])]

    print(f"\nUsing {len(feat_cols)} feature columns for {model_name}")

    # Get unique sorted dates
    dates_series = features_df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    splitter = WalkForwardSplitter(
        lookback_window=lookback,
        rebalance_freq=rebalance_freq,
    )
    splits = list(splitter.split(dates_series))
    print(f"Walk-forward folds: {len(splits)}")

    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_dates = set(dates_series.iloc[train_idx])
        test_dates = set(dates_series.iloc[test_idx])

        train_df = features_df[features_df["date"].isin(train_dates)].copy()
        test_df = features_df[features_df["date"].isin(test_dates)].copy()

        # Drop rows with NaN targets in training set
        train_df = train_df.dropna(subset=[target_col])

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        # Prepare X, y for training (rank target is better for ranking models)
        X_train = train_df[feat_cols].copy()
        # Use cross-sectional rank as target (cleaner than raw return)
        y_train = train_df[cs_rank_col].copy()
        
        # Use raw return as secondary target for evaluation
        y_train_ret = train_df[target_col].copy()

        # Drop NaN in training features
        mask_train = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]

        if len(X_train) < 20:
            continue

        # Scale features (train-only scaler)
        scaler = TrainOnlyScaler(StandardScaler())
        X_train_scaled = scaler.fit_transform(X_train)

        # Fill any remaining NaN with column mean (impute on train stats)
        X_train_arr = np.nan_to_num(X_train_scaled if isinstance(X_train_scaled, np.ndarray) else X_train_scaled.values, nan=0.0)

        # Fit model
        model = _make_model(model_name)
        model.fit(X_train_arr, y_train.values)

        # Predict on test set
        X_test = test_df[feat_cols].copy()
        # Use train column means for imputation of test NaN (no leakage)
        train_col_means = {c: float(X_train[c].mean()) for c in feat_cols}

        for col in feat_cols:
            X_test[col] = X_test[col].fillna(train_col_means.get(col, 0.0))

        X_test_scaled = scaler.transform(X_test)
        X_test_arr = np.nan_to_num(X_test_scaled if isinstance(X_test_scaled, np.ndarray) else X_test_scaled.values, nan=0.0)

        preds = model.predict(X_test_arr)

        # Store predictions with metadata
        pred_df = test_df[["date", "symbol", target_col]].copy()
        pred_df["predicted_score"] = preds
        pred_df["fold"] = fold_idx
        all_predictions.append(pred_df)

        # Per-fold rank metrics
        valid = pred_df.dropna(subset=[target_col])
        fold_rics = []
        for date_val, grp in valid.groupby("date"):
            if len(grp) >= 2:
                ic = rank_ic(grp[target_col].values, grp["predicted_score"].values)
                fold_rics.append(ic)

        mean_ic = float(np.mean(fold_rics)) if fold_rics else 0.0
        p_at_k = 0.0
        if valid.groupby("date").ngroups >= 1:
            from portfolio_ml.features.ranking_targets import precision_at_k as pak
            p_at_k_scores = []
            for _, grp in valid.groupby("date"):
                if len(grp) >= 2:
                    p_at_k_scores.append(pak(grp[target_col].values, grp["predicted_score"].values, k=top_k))
            p_at_k = float(np.mean(p_at_k_scores)) if p_at_k_scores else 0.0

        fold_result = {
            "fold": fold_idx,
            "train_start": min(train_dates),
            "train_end": max(train_dates),
            "test_start": min(test_dates),
            "test_end": max(test_dates),
            "n_train_rows": len(X_train),
            "n_test_rows": len(test_df),
            "n_dates_test": len(test_dates),
            "mean_rank_ic": mean_ic,
            f"precision_at_{top_k}": p_at_k,
        }
        fold_results.append(fold_result)

        if fold_idx < 3 or fold_idx == len(splits) - 1:
            print(
                f"  Fold {fold_idx}: "
                f"train {min(train_dates).date()}–{max(train_dates).date()} ({len(X_train)}), "
                f"test {min(test_dates).date()}–{max(test_dates).date()} | "
                f"Rank IC={mean_ic:.4f}, P@{top_k}={p_at_k:.4f}"
            )

    combined_preds = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return fold_results, combined_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Phase 3.2 — Ranking Model Training")
    print(f"  Feature set  : {args.feature_set}")
    print(f"  Model        : {args.model}")
    print(f"  Horizon      : {args.target_horizon} days")
    print(f"  Lookback     : {args.lookback} days")
    print(f"  Rebalance    : {args.rebalance_freq} days")
    print(f"  Top-K        : {args.top_k}")
    print("=" * 80)

    # 1. Load data
    prices_wide = load_price_data_wide()

    # 2. Build features + targets
    features_df = build_features_and_targets(
        prices_wide,
        horizon=args.target_horizon,
        feature_set=args.feature_set,
    )

    # 3. Walk-forward training
    print("\n" + "=" * 80)
    print("Walk-forward ranking model training")
    print("=" * 80)

    fold_results, combined_preds = run_walk_forward_ranking(
        features_df,
        model_name=args.model,
        horizon=args.target_horizon,
        lookback=args.lookback,
        rebalance_freq=args.rebalance_freq,
        top_k=args.top_k,
    )

    if not fold_results:
        print("\n⚠️  No valid folds produced. Check data availability.")
        return

    fold_df = pd.DataFrame(fold_results)

    # 4. Aggregate metrics
    mean_ic = fold_df["mean_rank_ic"].mean()
    std_ic = fold_df["mean_rank_ic"].std()
    pct_positive_ic = (fold_df["mean_rank_ic"] > 0).mean()
    pak_col = f"precision_at_{args.top_k}"
    mean_pak = fold_df[pak_col].mean()

    print("\n" + "=" * 80)
    print(f"AGGREGATE RESULTS — {args.model.upper()}")
    print("=" * 80)
    print(f"  Mean Rank IC        : {mean_ic:.4f} ± {std_ic:.4f}")
    print(f"  % Positive IC folds : {pct_positive_ic:.1%}")
    print(f"  Mean Precision@{args.top_k}  : {mean_pak:.4f}")
    print(f"  Folds completed     : {len(fold_df)}")

    # 5. Save fold results
    fold_file = output_dir / f"ranking_fold_results_{args.model}_{args.feature_set}.csv"
    fold_df.to_csv(fold_file, index=False)
    print(f"\n✅ Fold results saved to {fold_file}")

    if not combined_preds.empty:
        preds_file = output_dir / f"ranking_predictions_{args.model}_{args.feature_set}.csv"
        combined_preds.to_csv(preds_file, index=False)
        print(f"✅ Predictions saved to {preds_file}")

    # 6. Build v2 experiment row and append to unified summary
    experiment_row = make_experiment_row(
        experiment_type="ml_ranking",
        model_name=args.model,
        feature_set=args.feature_set,
        target_type=f"cs_rank_{args.target_horizon}d",
        rebalance_frequency=args.rebalance_freq,
        lookback_window=args.lookback,
        rank_ic=mean_ic,
        spearman_ic=mean_ic,
        precision_at_k=mean_pak,
        notes=f"walk-forward {len(fold_df)} folds, P@{args.top_k}={mean_pak:.4f}",
    )

    # Load or create unified v2 summary
    summary_file = output_dir / "unified_experiment_summary_v2.csv"
    if summary_file.exists():
        existing = pd.read_csv(summary_file)
        # Ensure all schema columns exist
        for col in EXPERIMENT_SCHEMA_COLUMNS:
            if col not in existing.columns:
                existing[col] = np.nan
        updated = pd.concat(
            [existing, pd.DataFrame([experiment_row])],
            ignore_index=True,
        )
    else:
        updated = pd.DataFrame([experiment_row])

    updated = validate_experiment_dataframe(updated)
    updated.to_csv(summary_file, index=False)
    print(f"✅ Updated unified summary: {summary_file}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  python scripts/run_two_stage_portfolio.py --model gradient_boost")
    print("  python scripts/verify_phase_3_2.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
