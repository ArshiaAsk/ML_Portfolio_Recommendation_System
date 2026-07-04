"""Train ML models for return prediction with proper walk-forward validation.

This script demonstrates supervised learning for portfolio optimization:
1. Load feature data with target variables
2. Setup walk-forward cross-validation (prevent look-ahead bias)
3. Train ML models to predict future returns
4. Evaluate prediction accuracy across folds
5. Log experiments with MLflow

Usage:
    python scripts/train_ml_model.py

Requirements:
    - Feature data in data/marts/features/
    - pip install scikit-learn mlflow
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from portfolio_ml.modeling import WalkForwardSplitter, standard_scaler

# Enable MLflow file store
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

from portfolio_ml.experiments import log_metrics, log_params, start_experiment


def load_feature_data() -> pd.DataFrame:
    """Load all feature data with targets.
    
    Returns:
        DataFrame with features and target_return_* columns
    """
    print("Loading feature data...")
    
    feature_files = sorted(glob.glob('data/marts/features/year=*/asset_daily_features.parquet'))
    
    if not feature_files:
        raise FileNotFoundError(
            "No feature data found in data/marts/features/. "
            "Run Phase 2 pipeline first."
        )
    
    print(f"Found {len(feature_files)} feature files")
    
    dfs = []
    for file in feature_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    all_features = pd.concat(dfs, ignore_index=True)
    
    # Convert date to datetime
    all_features['date'] = pd.to_datetime(all_features['date'])
    
    # Sort by date and symbol
    all_features = all_features.sort_values(['date', 'symbol'])
    
    print(f"Loaded {len(all_features)} rows")
    print(f"Date range: {all_features['date'].min()} to {all_features['date'].max()}")
    print(f"Symbols: {all_features['symbol'].nunique()}")
    print(f"Columns: {all_features.columns.tolist()}")
    
    return all_features


def prepare_ml_data(df: pd.DataFrame, target_col: str = 'target_return_5d'):
    """Prepare features and target for ML.
    
    Args:
        df: Feature dataframe
        target_col: Target column name
    
    Returns:
        Tuple of (features_df, feature_columns)
    """
    # Define feature columns (exclude date, symbol, targets, metadata)
    feature_cols = [
        'return_1d', 'log_return_1d', 'return_5d', 'return_21d', 'return_63d',
        'volatility_21d', 'volatility_63d', 'momentum_21d', 'momentum_63d',
        'drawdown', 'rolling_volume_21d', 'price_to_ma_21', 'price_to_ma_63'
    ]
    
    # Keep only rows where target is not NaN
    df_clean = df.dropna(subset=[target_col]).copy()
    
    print(f"\nPreparing ML data:")
    print(f"  Target: {target_col}")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Rows after dropping NaN targets: {len(df_clean)}")
    
    # Check for NaN in features
    nan_counts = df_clean[feature_cols].isna().sum()
    if nan_counts.any():
        print(f"\n⚠️  Warning: NaN values in features:")
        print(nan_counts[nan_counts > 0])
        print("  Filling NaN with 0 (consider better imputation for production)")
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    
    return df_clean, feature_cols


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'target_return_5d',
    model_name: str = 'RandomForest'
):
    """Train ML model with walk-forward validation.
    
    Args:
        df: Feature dataframe (must include 'date' column)
        feature_cols: List of feature column names
        target_col: Target column name
        model_name: 'RandomForest' or 'Ridge'
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name} model")
    print(f"{'='*80}")
    
    # Setup walk-forward splitter
    dates = df['date'].unique()
    splitter = WalkForwardSplitter(
        lookback_window=252,  # 1 year training
        rebalance_freq=21,    # Monthly evaluation
    )
    
    splits = list(splitter.split(dates))
    print(f"Walk-forward folds: {len(splits)}")
    
    # Store results for each fold
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        
        # Split data
        train_df = df[df['date'].isin(train_dates)]
        test_df = df[df['date'].isin(test_dates)]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Scale features (fit on train only!)
        scaler = standard_scaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_name == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'Ridge':
            model = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        fold_results.append({
            'fold': fold_idx,
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max(),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
        })
        
        if fold_idx < 3 or fold_idx == len(splits) - 1:  # Print first 3 and last
            print(f"\nFold {fold_idx}:")
            print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()} ({len(train_df)} samples)")
            print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()} ({len(test_df)} samples)")
            print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
    
    results_df = pd.DataFrame(fold_results)
    
    # Aggregate metrics
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS ({model_name})")
    print(f"{'='*80}")
    print(f"Mean RMSE: {results_df['rmse'].mean():.6f} ± {results_df['rmse'].std():.6f}")
    print(f"Mean MAE:  {results_df['mae'].mean():.6f} ± {results_df['mae'].std():.6f}")
    print(f"Mean R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
    
    return results_df


def main():
    """Main training pipeline."""
    
    # 1. Load data
    df = load_feature_data()
    df_clean, feature_cols = prepare_ml_data(df, target_col='target_return_5d')
    
    # Create output directory
    output_dir = Path('outputs/phase3_ml')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Train RandomForest
    print("\n" + "="*80)
    print("EXPERIMENT 1: Random Forest Regressor")
    print("="*80)
    
    with start_experiment("portfolio_ml_phase3", run_name="random_forest_5d"):
        log_params({
            "model": "RandomForest",
            "target": "target_return_5d",
            "n_estimators": 100,
            "max_depth": 10,
            "lookback_window": 252,
            "rebalance_freq": 21,
        })
        
        rf_results = train_and_evaluate(
            df_clean, feature_cols, 
            target_col='target_return_5d',
            model_name='RandomForest'
        )
        
        # Log aggregate metrics
        log_metrics({
            "mean_rmse": rf_results['rmse'].mean(),
            "mean_mae": rf_results['mae'].mean(),
            "mean_r2": rf_results['r2'].mean(),
            "std_rmse": rf_results['rmse'].std(),
        })
        
        # Save results
        rf_file = output_dir / 'random_forest_results.csv'
        rf_results.to_csv(rf_file, index=False)
        print(f"\n✅ Random Forest results saved to {rf_file}")
    
    # 3. Train Ridge Regression
    print("\n" + "="*80)
    print("EXPERIMENT 2: Ridge Regression")
    print("="*80)
    
    with start_experiment("portfolio_ml_phase3", run_name="ridge_5d"):
        log_params({
            "model": "Ridge",
            "target": "target_return_5d",
            "alpha": 1.0,
            "lookback_window": 252,
            "rebalance_freq": 21,
        })
        
        ridge_results = train_and_evaluate(
            df_clean, feature_cols,
            target_col='target_return_5d',
            model_name='Ridge'
        )
        
        log_metrics({
            "mean_rmse": ridge_results['rmse'].mean(),
            "mean_mae": ridge_results['mae'].mean(),
            "mean_r2": ridge_results['r2'].mean(),
            "std_rmse": ridge_results['rmse'].std(),
        })
        
        ridge_file = output_dir / 'ridge_results.csv'
        ridge_results.to_csv(ridge_file, index=False)
        print(f"\n✅ Ridge results saved to {ridge_file}")
    
    # 4. Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': ['RandomForest', 'Ridge'],
        'Mean RMSE': [rf_results['rmse'].mean(), ridge_results['rmse'].mean()],
        'Mean MAE': [rf_results['mae'].mean(), ridge_results['mae'].mean()],
        'Mean R²': [rf_results['r2'].mean(), ridge_results['r2'].mean()],
    })
    
    print(comparison.to_string(index=False))
    
    comparison_file = output_dir / 'model_comparison.csv'
    comparison.to_csv(comparison_file, index=False)
    
    print(f"\n✅ All results saved to {output_dir}/")
    print("\nNext steps:")
    print("  1. Review MLflow UI: mlflow ui --backend-store-uri ./mlruns")
    print("  2. Run: python scripts/evaluate_by_regime.py")
    print("  3. Experiment with feature engineering")


if __name__ == "__main__":
    main()
