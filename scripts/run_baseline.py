"""Run baseline portfolio strategies and generate performance reports.

This script demonstrates the complete workflow for Phase 3:
1. Load processed price data
2. Pivot to wide format (one column per asset)
3. Setup walk-forward cross-validation
4. Run baseline strategies (Equal Weight, Min Variance, Risk Parity)
5. Generate performance metrics and comparison reports

Usage:
    python scripts/run_baseline.py

Requirements:
    - Processed daily prices in data/processed/daily_prices/
    - pip install scipy scikit-learn mlflow
"""

import glob
from pathlib import Path

import pandas as pd

from portfolio_ml.backtesting import BenchmarkRunner
from portfolio_ml.modeling import WalkForwardSplitter


def load_price_data() -> pd.DataFrame:
    """Load all processed daily prices and pivot to wide format.
    
    Returns:
        DataFrame with DatetimeIndex and one column per asset (adj_close prices)
    """
    print("Loading processed price data...")
    
    # Load all partitioned parquet files
    price_files = sorted(glob.glob('data/processed/daily_prices/year=*/daily_prices.parquet'))
    
    if not price_files:
        raise FileNotFoundError(
            "No price data found in data/processed/daily_prices/. "
            "Run Phase 2 pipeline first."
        )
    
    print(f"Found {len(price_files)} parquet files")
    
    # Read all files
    dfs = []
    for file in price_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all data
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_data)} rows, {all_data['symbol'].nunique()} unique symbols")
    
    # Convert date to datetime
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    # Pivot to wide format: rows = dates, columns = symbols
    prices_wide = all_data.pivot(index='date', columns='symbol', values='adj_close')
    
    # Sort by date
    prices_wide = prices_wide.sort_index()
    
    # Drop any columns with all NaN
    prices_wide = prices_wide.dropna(axis=1, how='all')
    
    print(f"\nPivoted to wide format:")
    print(f"  Date range: {prices_wide.index.min()} to {prices_wide.index.max()}")
    print(f"  Number of assets: {len(prices_wide.columns)}")
    print(f"  Shape: {prices_wide.shape}")
    print(f"\nAssets: {', '.join(prices_wide.columns.tolist())}")
    
    return prices_wide


def main():
    """Run baseline strategy backtests."""
    
    # 1. Load data
    prices = load_price_data()
    
    # Check for sufficient data
    if len(prices) < 300:
        raise ValueError(f"Insufficient data: only {len(prices)} days. Need at least 300.")
    
    # 2. Setup walk-forward cross-validation
    print("\n" + "="*80)
    print("Setting up walk-forward cross-validation...")
    print("="*80)
    
    splitter = WalkForwardSplitter(
        lookback_window=252,   # 1 year training window
        rebalance_freq=21,     # Monthly rebalancing (approx 21 trading days)
    )
    
    # Preview splits
    dates = prices.index.to_numpy()
    splits = list(splitter.split(dates))
    print(f"Generated {len(splits)} walk-forward folds")
    print(f"First fold train dates: {dates[splits[0][0]].min()} to {dates[splits[0][0]].max()}")
    print(f"First fold test dates: {dates[splits[0][1]].min()} to {dates[splits[0][1]].max()}")
    
    # 3. Run benchmark strategies
    print("\n" + "="*80)
    print("Running baseline strategies...")
    print("="*80)
    
    runner = BenchmarkRunner(
        price_data=prices,
        splitter=splitter,
        transaction_cost=0.0005,  # 5 basis points
    )
    
    # Run all baseline strategies
    print("\nRunning benchmarks (this may take a few minutes)...")
    summary = runner.run_all()
    
    # 4. Display results
    print("\n" + "="*80)
    print("BASELINE STRATEGY PERFORMANCE")
    print("="*80)
    print(summary.to_string(index=False))
    
    # 5. Save results
    output_dir = Path('outputs/phase3_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / 'baseline_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\n✅ Results saved to {summary_file}")
    
    # Save detailed results for each strategy
    for strategy_name in summary['Strategy'].values:
        results = runner.get_strategy_results(strategy_name)
        
        # Save daily returns
        returns_file = output_dir / f'{strategy_name.lower()}_returns.csv'
        results['daily_returns'].to_csv(returns_file)
        
        # Save weights (if available)
        if 'weights' in results and results['weights'] is not None:
            weights_file = output_dir / f'{strategy_name.lower()}_weights.csv'
            results['weights'].to_csv(weights_file)
        
        print(f"   Saved {strategy_name} details")
    
    print(f"\n✅ All results saved to {output_dir}/")
    print("\nNext steps:")
    print("  1. Review performance metrics above")
    print("  2. Run: python scripts/train_ml_model.py")
    print("  3. Run: python scripts/evaluate_by_regime.py")


if __name__ == "__main__":
    main()
