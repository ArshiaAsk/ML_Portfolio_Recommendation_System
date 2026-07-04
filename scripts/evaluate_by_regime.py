"""Evaluate portfolio strategies by market regime.

This script demonstrates regime-based performance analysis:
1. Load baseline strategy returns
2. Define market regimes (bull, bear, high volatility, etc.)
3. Compute metrics for each regime
4. Compare strategies across regimes

Usage:
    python scripts/evaluate_by_regime.py

Requirements:
    - Baseline results from run_baseline.py
"""

from pathlib import Path

import pandas as pd

from portfolio_ml.backtesting import SliceEvaluator, compare_strategies_by_regime


def load_strategy_returns(output_dir: Path) -> dict:
    """Load daily returns for all strategies.
    
    Args:
        output_dir: Directory containing baseline results
    
    Returns:
        Dict mapping strategy names to return Series
    """
    print("Loading strategy returns...")
    
    strategies = {}
    
    for returns_file in output_dir.glob('*_returns.csv'):
        strategy_name = returns_file.stem.replace('_returns', '').title().replace('_', '')
        
        returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        returns.index.name = 'date'
        
        # Assume single column with returns
        if len(returns.columns) == 1:
            strategies[strategy_name] = returns.iloc[:, 0]
        else:
            strategies[strategy_name] = returns['return']  # or first column
        
        print(f"  Loaded {strategy_name}: {len(strategies[strategy_name])} days")
    
    return strategies


def define_regimes(returns: pd.Series) -> dict:
    """Define market regime time slices.
    
    Args:
        returns: Daily returns series to determine date range
    
    Returns:
        Dict mapping regime names to (start_date, end_date) tuples
    """
    # Get date range
    start = returns.index.min()
    end = returns.index.max()
    
    print(f"\nDate range: {start.date()} to {end.date()}")
    
    # Define regimes based on known market events
    # Adjust these based on your data's date range
    regimes = {}
    
    # Year-based slices
    for year in range(start.year, end.year + 1):
        regimes[f'Year_{year}'] = (f'{year}-01-01', f'{year}-12-31')
    
    # COVID crash (if in range)
    if start <= pd.Timestamp('2020-02-01') <= end:
        regimes['COVID_Crash'] = ('2020-02-01', '2020-04-30')
        regimes['COVID_Recovery'] = ('2020-05-01', '2020-12-31')
    
    # Recent bear market (if in range)
    if start <= pd.Timestamp('2022-01-01') <= end:
        regimes['Bear_2022'] = ('2022-01-01', '2022-10-31')
    
    # Bull runs
    if start <= pd.Timestamp('2019-01-01') <= end:
        regimes['Bull_2019'] = ('2019-01-01', '2019-12-31')
    
    if start <= pd.Timestamp('2023-11-01') <= end:
        regimes['Bull_Late_2023'] = ('2023-11-01', '2023-12-31')
    
    # Filter regimes to only those within data range
    valid_regimes = {}
    for name, (s, e) in regimes.items():
        s_ts = pd.Timestamp(s)
        e_ts = pd.Timestamp(e)
        if s_ts <= end and e_ts >= start:
            valid_regimes[name] = (s, e)
    
    print(f"\nDefined {len(valid_regimes)} market regimes:")
    for name, (s, e) in valid_regimes.items():
        print(f"  {name}: {s} to {e}")
    
    return valid_regimes


def main():
    """Main evaluation pipeline."""
    
    # 1. Load baseline results
    baseline_dir = Path('outputs/phase3_baseline')
    
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"Baseline results not found at {baseline_dir}. "
            "Run scripts/run_baseline.py first."
        )
    
    strategy_returns = load_strategy_returns(baseline_dir)
    
    if not strategy_returns:
        raise ValueError("No strategy returns found. Check baseline output format.")
    
    # 2. Define market regimes
    # Use the first strategy's returns to determine date range
    sample_returns = list(strategy_returns.values())[0]
    regimes = define_regimes(sample_returns)
    
    # 3. Evaluate each strategy by regime
    print("\n" + "="*80)
    print("EVALUATING STRATEGIES BY REGIME")
    print("="*80)
    
    output_dir = Path('outputs/phase3_regime')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for strategy_name, returns in strategy_returns.items():
        print(f"\n{strategy_name}:")
        print("-" * 60)
        
        evaluator = SliceEvaluator(returns)
        regime_metrics = evaluator.evaluate_slices(regimes)
        
        print(regime_metrics.to_string(index=False))
        
        # Save results
        output_file = output_dir / f'{strategy_name.lower()}_by_regime.csv'
        regime_metrics.to_csv(output_file, index=False)
    
    # 4. Compare strategies across regimes
    print("\n" + "="*80)
    print("STRATEGY COMPARISON BY REGIME")
    print("="*80)
    
    comparison = compare_strategies_by_regime(strategy_returns, regimes)
    
    # Reset index to make Slice and Strategy regular columns
    comparison_flat = comparison.reset_index()
    
    # Display comparison for each regime
    for regime_name in regimes.keys():
        regime_data = comparison_flat[comparison_flat['Slice'] == regime_name]
        
        if not regime_data.empty:
            print(f"\n{regime_name}:")
            # Display key metrics
            display_cols = ['Strategy', 'Total_Return', 'Sharpe_Ratio', 'Max_Drawdown']
            print(regime_data[display_cols].to_string(index=False))
    
    # Save full comparison
    comparison_file = output_dir / 'strategy_comparison_by_regime.csv'
    comparison_flat.to_csv(comparison_file, index=False)
    
    print(f"\n✅ All results saved to {output_dir}/")
    
    # 5. Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Best strategy per regime (by Sharpe ratio)
    for regime_name in regimes.keys():
        regime_data = comparison_flat[comparison_flat['Slice'] == regime_name]
        
        if not regime_data.empty and not regime_data['Sharpe_Ratio'].isna().all():
            best_idx = regime_data['Sharpe_Ratio'].idxmax()
            best = regime_data.loc[best_idx]
            print(f"{regime_name}: Best strategy = {best['Strategy']} (Sharpe: {best['Sharpe_Ratio']:.2f})")
    
    print("\nNext steps:")
    print("  1. Analyze which strategies perform best in different regimes")
    print("  2. Consider regime-switching portfolio strategies")
    print("  3. Develop ML models to predict regime transitions")


if __name__ == "__main__":
    main()
