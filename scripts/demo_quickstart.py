"""Quick demo of Phase 3 modeling workflow.

This is a minimal example showing the complete Phase 3 pipeline.

Usage:
    python scripts/demo_quickstart.py
"""

import glob

import pandas as pd

from portfolio_ml.backtesting import BenchmarkRunner
from portfolio_ml.modeling import WalkForwardSplitter


def main():
    """Run minimal Phase 3 demo."""
    
    print("="*80)
    print("PHASE 3 MODELING - QUICK START DEMO")
    print("="*80)
    
    # 1. Load price data
    print("\n1. Loading price data...")
    price_files = sorted(glob.glob('data/processed/daily_prices/year=*/daily_prices.parquet'))
    
    dfs = [pd.read_parquet(f) for f in price_files]
    all_data = pd.concat(dfs, ignore_index=True)
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    # Pivot to wide format
    prices = all_data.pivot(index='date', columns='symbol', values='adj_close')
    prices = prices.sort_index()
    
    print(f"   Loaded {prices.shape[0]} days, {prices.shape[1]} assets")
    print(f"   Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    
    # 2. Setup walk-forward validation
    print("\n2. Setting up walk-forward cross-validation...")
    splitter = WalkForwardSplitter(
        lookback_window=252,   # 1 year training
        rebalance_freq=21,     # Monthly rebalancing
    )
    print("   ✓ Configured: 252-day lookback, 21-day rebalancing")
    
    # 3. Run baseline benchmarks
    print("\n3. Running baseline strategies...")
    runner = BenchmarkRunner(
        price_data=prices,
        splitter=splitter,
        transaction_cost=0.0005,  # 5 bps
    )
    
    summary = runner.run_all()
    
    # 4. Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(summary.to_string(index=False))
    
    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("  • Run full baseline: python scripts/run_baseline.py")
    print("  • Train ML models: python scripts/train_ml_model.py")
    print("  • Evaluate by regime: python scripts/evaluate_by_regime.py")


if __name__ == "__main__":
    main()
