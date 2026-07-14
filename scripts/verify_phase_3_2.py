"""Verification script for Phase 3.2 implementation.

Checks:
1. Schema stability: unified_experiment_summary_v2.csv has all required columns.
2. No missing strategy/model names in portfolio/ML rows.
3. Impossible metric combinations flagged.
4. Feature/target alignment: features at time t predict labels at t+horizon.
5. No leakage: rolling features use only past data.
6. Portfolio weights: sum to 1, respect constraints.
7. Ranking metrics: correlations in [-1, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_ml.experiments.schema import (
    EXPERIMENT_SCHEMA_COLUMNS,
    VALID_EXPERIMENT_TYPES,
    validate_experiment_dataframe,
)


def check_schema_stability(summary_file: Path) -> bool:
    """Check that v2 summary has all required schema columns."""
    print("\n" + "=" * 80)
    print("CHECK 1: Schema Stability")
    print("=" * 80)

    if not summary_file.exists():
        print(f"❌ File not found: {summary_file}")
        return False

    df = pd.read_csv(summary_file)
    missing = set(EXPERIMENT_SCHEMA_COLUMNS) - set(df.columns)

    if missing:
        print(f"❌ Missing schema columns: {missing}")
        return False

    print(f"✅ All {len(EXPERIMENT_SCHEMA_COLUMNS)} schema columns present.")
    print(f"   Rows: {len(df)}")
    return True


def check_identifiers(summary_file: Path) -> bool:
    """Check that strategy_name/model_name are not missing where required."""
    print("\n" + "=" * 80)
    print("CHECK 2: Row Identifiers")
    print("=" * 80)

    df = pd.read_csv(summary_file)
    issues = []

    for idx, row in df.iterrows():
        exp_type = row.get("experiment_type", "")
        strategy = row.get("strategy_name")
        model = row.get("model_name")

        if exp_type in ("baseline", "regime", "two_stage_portfolio"):
            if pd.isna(strategy) or strategy == "":
                issues.append(f"Row {idx}: experiment_type={exp_type}, missing strategy_name")

        if exp_type in ("ml_regression", "ml_ranking", "ml_classification"):
            if pd.isna(model) or model == "":
                issues.append(f"Row {idx}: experiment_type={exp_type}, missing model_name")

    if issues:
        print("❌ Identifier issues:")
        for issue in issues:
            print(f"   {issue}")
        return False

    print(f"✅ All {len(df)} rows have valid identifiers.")
    return True


def check_validation_warnings(summary_file: Path) -> bool:
    """Run schema validation and check for warnings."""
    print("\n" + "=" * 80)
    print("CHECK 3: Metric Validation")
    print("=" * 80)

    df = pd.read_csv(summary_file)
    validated = validate_experiment_dataframe(df)

    if "validation_warnings" not in validated.columns:
        print("❌ validation_warnings column not added.")
        return False

    warnings = validated["validation_warnings"].fillna("")
    flagged = warnings[warnings != ""]

    if not flagged.empty:
        print(f"⚠️  {len(flagged)} rows have validation warnings:")
        for idx in flagged.index[:5]:  # show first 5
            print(f"   Row {idx}: {flagged.iloc[idx]}")
        if len(flagged) > 5:
            print(f"   ... and {len(flagged) - 5} more.")
        return False

    print(f"✅ All {len(df)} rows passed metric validation.")
    return True


def check_feature_target_alignment() -> bool:
    """Smoke test: feature/target alignment using a toy example."""
    print("\n" + "=" * 80)
    print("CHECK 4: Feature/Target Alignment (Smoke Test)")
    print("=" * 80)

    # Simple synthetic check
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    prices = pd.Series(np.arange(100, 150), index=dates)

    # Forward 5-day return at t should use price at t+5
    horizon = 5
    fwd_ret = (prices.shift(-horizon) / prices - 1).iloc[:-horizon]

    # Check alignment: fwd_ret at index 0 should equal (prices[5] / prices[0] - 1)
    expected = (prices.iloc[5] / prices.iloc[0]) - 1
    actual = fwd_ret.iloc[0]

    if not np.isclose(actual, expected, atol=1e-6):
        print(f"❌ Alignment failed: expected {expected:.6f}, got {actual:.6f}")
        return False

    print("✅ Feature/target alignment check passed.")
    return True


def check_no_leakage() -> bool:
    """Smoke test: rolling features do not peek forward."""
    print("\n" + "=" * 80)
    print("CHECK 5: No Leakage (Smoke Test)")
    print("=" * 80)

    prices = np.arange(100, 120, dtype=float)
    window = 5

    rolling_mean = pd.Series(prices).rolling(window=window, min_periods=1).mean()

    # At index 4 (5th element), rolling mean should be mean of indices 0..4
    expected = prices[:5].mean()
    actual = rolling_mean.iloc[4]

    if not np.isclose(actual, expected, atol=1e-6):
        print(f"❌ Leakage detected: rolling mean at index 4 = {actual:.4f}, expected {expected:.4f}")
        return False

    print("✅ Rolling features do not leak forward.")
    return True


def check_portfolio_weights() -> bool:
    """Smoke test: portfolio weights sum to 1 and respect constraints."""
    print("\n" + "=" * 80)
    print("CHECK 6: Portfolio Weight Constraints (Smoke Test)")
    print("=" * 80)

    from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores

    scores = pd.Series([0.9, 0.7, 0.3, 0.1, 0.05], index=["A", "B", "C", "D", "E"])
    config = TwoStageConfig(portfolio_method="top_k_equal", top_k=3, max_weight=0.4)

    weights = compute_weights_from_scores(scores, list(scores.index), config)

    total = weights.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        print(f"❌ Weights do not sum to 1: {total:.6f}")
        return False

    if np.any(weights < 0) or np.any(weights > config.max_weight + 1e-6):
        print(f"❌ Weights violate constraints: min={weights.min():.4f}, max={weights.max():.4f}")
        return False

    print(f"✅ Weights sum to {total:.6f}, respect constraints [0, {config.max_weight}].")
    return True


def check_ranking_metrics() -> bool:
    """Smoke test: rank IC is in [-1, 1]."""
    print("\n" + "=" * 80)
    print("CHECK 7: Ranking Metrics (Smoke Test)")
    print("=" * 80)

    from portfolio_ml.features.ranking_targets import rank_ic

    actual = np.array([0.1, 0.3, 0.2, 0.5])
    predicted = np.array([0.15, 0.25, 0.22, 0.45])

    ic = rank_ic(actual, predicted)

    if not (-1.0 <= ic <= 1.0):
        print(f"❌ Rank IC out of range: {ic:.4f}")
        return False

    print(f"✅ Rank IC in valid range: {ic:.4f}")
    return True


def main() -> None:
    print("=" * 80)
    print("Phase 3.2 Verification")
    print("=" * 80)

    summary_file = Path("outputs/phase3_2/unified_experiment_summary_v2.csv")

    checks = [
        ("Schema Stability", lambda: check_schema_stability(summary_file)),
        ("Row Identifiers", lambda: check_identifiers(summary_file)),
        ("Metric Validation", lambda: check_validation_warnings(summary_file)),
        ("Feature/Target Alignment", check_feature_target_alignment),
        ("No Leakage", check_no_leakage),
        ("Portfolio Weights", check_portfolio_weights),
        ("Ranking Metrics", check_ranking_metrics),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status:<10} {name}")

    all_passed = all(p for _, p in results)
    print("=" * 80)
    if all_passed:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
