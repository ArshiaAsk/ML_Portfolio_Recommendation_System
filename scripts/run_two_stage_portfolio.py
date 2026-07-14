"""Two-stage ML portfolio: signal generation → portfolio construction.

Phase 3.2 — Two-Stage Portfolio Construction

Workflow:
1. Load price data.
2. Build feature set v2.
3. Walk-forward: fit ranking model → generate scores per date → construct portfolio.
4. Backtest the resulting portfolio.
5. Compare vs EqualWeight / RiskParity / MinVariance.
6. Save results to v2 unified experiment summary.

Usage:
    python scripts/run_two_stage_portfolio.py [options]

Options:
    --model              ridge | gradient_boost | random_forest (default: gradient_boost)
    --portfolio-method   top_k_equal | score_weighted | mean_variance (default: top_k_equal)
    --top-k              K for top-K selection (default: 5)
    --max-weight         max weight per asset (default: 0.25)
    --target-horizon     forward return horizon in days (default: 21)
    --lookback           walk-forward lookback in days (default: 252)
    --rebalance-freq     rebalance frequency in days (default: 21)
    --transaction-cost   cost in bps (default: 5)
    --output-dir         output directory (default: outputs/phase3_2)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from portfolio_ml.backtesting.engine import BacktestEngine
from portfolio_ml.experiments.schema import (
    EXPERIMENT_SCHEMA_COLUMNS,
    make_experiment_row,
    validate_experiment_dataframe,
)
from portfolio_ml.features.feature_set_v2 import build_feature_set_v2, get_feature_columns
from portfolio_ml.features.ranking_targets import add_ranking_targets
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores, normalise_scores_cross_sectionally
from portfolio_ml.modeling.walk_forward import WalkForwardSplitter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage ML portfolio (Phase 3.2)")
    parser.add_argument("--model", choices=["ridge", "gradient_boost", "random_forest"], default="gradient_boost")
    parser.add_argument("--portfolio-method", choices=["top_k_equal", "score_weighted", "mean_variance"], default="top_k_equal")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--target-horizon", type=int, default=21)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance-freq", type=int, default=21)
    parser.add_argument("--transaction-cost", type=float, default=5.0, help="cost in bps")
    parser.add_argument("--output-dir", type=str, default="outputs/phase3_2")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prices_wide() -> pd.DataFrame:
    price_files = sorted(glob.glob("data/processed/daily_prices/year=*/daily_prices.parquet"))
    if not price_files:
        raise FileNotFoundError("No price data found. Run Phase 2 pipeline first.")
    dfs = [pd.read_parquet(f) for f in price_files]
    all_data = pd.concat(dfs, ignore_index=True)
    all_data["date"] = pd.to_datetime(all_data["date"])
    prices_wide = all_data.pivot(index="date", columns="symbol", values="adj_close")
    return prices_wide.sort_index().dropna(axis=1, how="all")


def _make_model(model_name: str):
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    elif model_name == "gradient_boost":
        return GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    elif model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main walk-forward two-stage pipeline
# ---------------------------------------------------------------------------


def run_two_stage(
    prices_wide: pd.DataFrame,
    model_name: str,
    ts_config: TwoStageConfig,
    horizon: int,
    lookback: int,
    rebalance_freq: int,
) -> dict[pd.Timestamp, np.ndarray]:
    """Run walk-forward two-stage pipeline; return weights schedule."""
    print("\nBuilding feature set v2...")
    features_df = build_feature_set_v2(prices_wide)

    # Note: adj_close is already in features_df from build_feature_set_v2
    features_df = add_ranking_targets(features_df, horizon=horizon, price_col="adj_close")
    features_df = features_df.drop(columns=["adj_close"], errors="ignore")

    target_col = f"future_return_{horizon}d"
    cs_rank_col = f"cs_rank_{horizon}d"

    non_feature_cols = {
        "date", "symbol",
        f"future_return_{horizon}d",
        f"cs_rank_{horizon}d",
        f"cs_class_3_{horizon}d",
        f"cs_class_binary_{horizon}d",
    }
    feat_cols = sorted(set(features_df.columns) - non_feature_cols)
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(features_df[c])]

    dates_series = features_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    splitter = WalkForwardSplitter(lookback_window=lookback, rebalance_freq=rebalance_freq)
    splits = list(splitter.split(dates_series))

    asset_names = prices_wide.columns.tolist()
    weights_schedule: dict[pd.Timestamp, np.ndarray] = {}

    print(f"Walk-forward folds: {len(splits)}")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_dates = set(dates_series.iloc[train_idx])
        test_dates = set(dates_series.iloc[test_idx])

        train_df = features_df[features_df["date"].isin(train_dates)].copy()
        test_df = features_df[features_df["date"].isin(test_dates)].copy()

        train_df = train_df.dropna(subset=[cs_rank_col])
        if len(train_df) < 50:
            # Fallback: equal weight
            rebal_date = prices_wide.index[prices_wide.index >= min(test_dates)][0]
            n = len(asset_names)
            weights_schedule[rebal_date] = np.ones(n) / n
            continue

        X_train = train_df[feat_cols].copy()
        y_train = train_df[cs_rank_col].copy()
        mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 20:
            rebal_date = prices_wide.index[prices_wide.index >= min(test_dates)][0]
            n = len(asset_names)
            weights_schedule[rebal_date] = np.ones(n) / n
            continue

        scaler = TrainOnlyScaler(StandardScaler())
        X_train_arr = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)

        model = _make_model(model_name)
        model.fit(X_train_arr, y_train.values)

        # Get training-set column means for test imputation
        train_col_means = {c: float(X_train[c].mean()) for c in feat_cols}

        # Compute weights at each rebalance date in the test window
        # Use first date in test window as rebalance date
        rebal_date = prices_wide.index[prices_wide.index >= min(test_dates)][0]

        # Get test assets at the rebalance date
        test_at_rebal = test_df[test_df["date"] == test_df["date"].min()].copy()
        if test_at_rebal.empty:
            n = len(asset_names)
            weights_schedule[rebal_date] = np.ones(n) / n
            continue

        X_rebal = test_at_rebal[feat_cols].copy()
        for col in feat_cols:
            X_rebal[col] = X_rebal[col].fillna(train_col_means.get(col, 0.0))

        X_rebal_arr = np.nan_to_num(scaler.transform(X_rebal), nan=0.0)
        scores_arr = model.predict(X_rebal_arr)

        scores = pd.Series(scores_arr, index=test_at_rebal["symbol"].values)
        scores = normalise_scores_cross_sectionally(scores)

        # Historical returns for mean-variance
        returns_hist = None
        if ts_config.portfolio_method == "mean_variance":
            returns_hist = prices_wide.pct_change().iloc[-ts_config.cov_window:]

        weights = compute_weights_from_scores(
            scores,
            asset_names=asset_names,
            config=ts_config,
            returns_history=returns_hist,
        )
        weights_schedule[rebal_date] = weights

        if fold_idx < 2 or fold_idx == len(splits) - 1:
            top_assets = [asset_names[i] for i in np.argsort(weights)[-3:][::-1]]
            print(f"  Fold {fold_idx}: rebal_date={rebal_date.date()}, top assets={top_assets}")

    return weights_schedule


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 3.2 — Two-Stage Portfolio Construction")
    print(f"  Model            : {args.model}")
    print(f"  Portfolio method : {args.portfolio_method}")
    print(f"  Top-K            : {args.top_k}")
    print(f"  Max weight       : {args.max_weight}")
    print(f"  Horizon          : {args.target_horizon}d")
    print(f"  TX cost          : {args.transaction_cost} bps")
    print("=" * 80)

    # Load data
    print("\nLoading price data...")
    prices_wide = _load_prices_wide()
    print(f"  {prices_wide.shape[0]} days × {prices_wide.shape[1]} assets")

    ts_config = TwoStageConfig(
        portfolio_method=args.portfolio_method,
        top_k=args.top_k,
        max_weight=args.max_weight,
        cov_window=args.lookback,
        rebalance_frequency=args.rebalance_freq,
        transaction_cost_bps=args.transaction_cost,
    )

    # Run two-stage
    weights_schedule = run_two_stage(
        prices_wide=prices_wide,
        model_name=args.model,
        ts_config=ts_config,
        horizon=args.target_horizon,
        lookback=args.lookback,
        rebalance_freq=args.rebalance_freq,
    )

    if not weights_schedule:
        print("\n⚠️  No weight schedule produced. Insufficient data.")
        return

    print(f"\nProduced {len(weights_schedule)} rebalance dates.")

    # Backtest
    tx_cost_pct = args.transaction_cost / 10_000.0
    engine = BacktestEngine(
        price_data=prices_wide,
        weights_schedule=weights_schedule,
        transaction_cost=tx_cost_pct,
        initial_capital=1_000_000.0,
    )
    result = engine.run()
    metrics = result["metrics"]

    print("\n" + "=" * 80)
    print(f"ML TWO-STAGE PORTFOLIO PERFORMANCE ({args.model.upper()} + {args.portfolio_method.upper()})")
    print("=" * 80)
    print(f"  Annualised Return  : {metrics['annualised_return']:.2%}")
    print(f"  Annualised Vol     : {metrics['annualised_volatility']:.2%}")
    print(f"  Sharpe Ratio       : {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown']:.2%}")
    print(f"  Avg Turnover       : {metrics['avg_turnover']:.4f}")
    print(f"  Total Return       : {metrics['total_return']:.2%}")

    # Comparison vs baselines (from existing baseline summary)
    baseline_file = Path("outputs/phase3_baseline/baseline_summary.csv")
    if baseline_file.exists():
        baselines = pd.read_csv(baseline_file)
        print("\n" + "=" * 80)
        print("COMPARISON vs BASELINES")
        print("=" * 80)
        print(f"  {'Strategy':<30} {'Ann.Ret':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10}")
        print("-" * 70)
        for _, row in baselines.iterrows():
            print(
                f"  {row['Strategy']:<30} "
                f"{row['Annualised Return']:>10.2%} "
                f"{row['Annualised Volatility']:>10.2%} "
                f"{row['Sharpe Ratio']:>10.3f} "
                f"{row['Max Drawdown']:>10.2%}"
            )
        print(
            f"  {'ML-' + args.model + ' (' + args.portfolio_method + ')':<30} "
            f"{metrics['annualised_return']:>10.2%} "
            f"{metrics['annualised_volatility']:>10.2%} "
            f"{metrics['sharpe_ratio']:>10.3f} "
            f"{metrics['max_drawdown']:>10.2%}"
        )

    # Save daily returns
    returns_file = output_dir / f"ml_{args.model}_{args.portfolio_method}_returns.csv"
    result["daily_returns"].to_csv(returns_file)

    # Build v2 experiment row
    strategy_name = f"ML_{args.model}_{args.portfolio_method}"
    experiment_row = make_experiment_row(
        experiment_type="two_stage_portfolio",
        strategy_name=strategy_name,
        model_name=args.model,
        feature_set="v2",
        target_type=f"cs_rank_{args.target_horizon}d",
        portfolio_method=args.portfolio_method,
        rebalance_frequency=args.rebalance_freq,
        lookback_window=args.lookback,
        transaction_cost_bps=args.transaction_cost,
        annualised_return=metrics["annualised_return"],
        annualised_volatility=metrics["annualised_volatility"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        turnover=metrics["avg_turnover"],
        notes=f"top_k={args.top_k}, max_weight={args.max_weight}",
    )

    # Append to v2 unified summary
    summary_file = output_dir / "unified_experiment_summary_v2.csv"
    if summary_file.exists():
        existing = pd.read_csv(summary_file)
        for col in EXPERIMENT_SCHEMA_COLUMNS:
            if col not in existing.columns:
                existing[col] = np.nan
        updated = pd.concat([existing, pd.DataFrame([experiment_row])], ignore_index=True)
    else:
        # Also include baseline rows if available
        rows = [experiment_row]
        if baseline_file.exists():
            baselines = pd.read_csv(baseline_file)
            for _, brow in baselines.iterrows():
                baseline_exp_row = make_experiment_row(
                    experiment_type="baseline",
                    strategy_name=brow["Strategy"],
                    rebalance_frequency=21,
                    lookback_window=252,
                    transaction_cost_bps=5.0,
                    annualised_return=brow["Annualised Return"],
                    annualised_volatility=brow["Annualised Volatility"],
                    sharpe_ratio=brow["Sharpe Ratio"],
                    max_drawdown=brow["Max Drawdown"],
                    turnover=brow["Avg Turnover"],
                    notes="from Phase 3 baseline run",
                )
                rows.insert(0, baseline_exp_row)
        updated = pd.DataFrame(rows)

    updated = validate_experiment_dataframe(updated)
    updated.to_csv(summary_file, index=False)
    print(f"\n✅ Updated unified summary: {summary_file}")
    print(f"✅ Daily returns saved: {returns_file}")

    print("\nNext steps:")
    print("  python scripts/verify_phase_3_2.py")
    print("  Review outputs/phase3_2/unified_experiment_summary_v2.csv")


if __name__ == "__main__":
    main()
