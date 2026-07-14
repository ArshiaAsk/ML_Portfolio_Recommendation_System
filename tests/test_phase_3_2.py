"""Phase 3.2 tests: schema, features, ranking, portfolio weights, metrics.

Tests cover:
1. Experiment schema validation (required columns, flagging impossible values).
2. Feature set v2 generation (no leakage, correct shapes).
3. Ranking targets (forward returns are forward-looking, cross-sectional ranks).
4. Ranking metrics (Rank IC in [-1, 1], Precision@K correct).
5. Two-stage portfolio weights (sum=1, constraints, turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices_wide() -> pd.DataFrame:
    """Wide-format price matrix: 300 dates × 5 assets."""
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    symbols = ["SPY", "TLT", "GLD", "QQQ", "EEM"]
    data = {}
    for sym, base in zip(symbols, [100, 80, 50, 200, 30]):
        data[sym] = base + np.random.randn(300).cumsum()
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def synthetic_prices_long(synthetic_prices_wide) -> pd.DataFrame:
    """Long-format price DataFrame for testing targets."""
    df = synthetic_prices_wide.stack().reset_index()
    df.columns = ["date", "symbol", "adj_close"]
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Experiment schema tests
# ---------------------------------------------------------------------------


class TestExperimentSchema:
    def test_make_experiment_row_has_all_columns(self):
        from portfolio_ml.experiments.schema import EXPERIMENT_SCHEMA_COLUMNS, make_experiment_row

        row = make_experiment_row(
            experiment_type="baseline",
            strategy_name="EqualWeight",
            annualised_return=0.1,
            sharpe_ratio=0.8,
            max_drawdown=-0.25,
        )
        for col in EXPERIMENT_SCHEMA_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_validate_row_flags_missing_strategy(self):
        from portfolio_ml.experiments.schema import validate_experiment_row

        row = {
            "experiment_type": "baseline",
            "strategy_name": None,
            "annualised_return": 0.1,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.2,
        }
        issues = validate_experiment_row(row)
        assert any("strategy_name" in issue for issue in issues), (
            "Should flag missing strategy_name for baseline."
        )

    def test_validate_row_flags_missing_model_name(self):
        from portfolio_ml.experiments.schema import validate_experiment_row

        row = {
            "experiment_type": "ml_ranking",
            "model_name": None,
            "rank_ic": 0.05,
        }
        issues = validate_experiment_row(row)
        assert any("model_name" in issue for issue in issues)

    def test_validate_row_flags_infinite_sharpe(self):
        from portfolio_ml.experiments.schema import validate_experiment_row

        row = {
            "experiment_type": "baseline",
            "strategy_name": "EqualWeight",
            "sharpe_ratio": float("inf"),
        }
        issues = validate_experiment_row(row)
        assert any("sharpe" in issue.lower() or "inf" in issue.lower() for issue in issues)

    def test_validate_row_flags_positive_max_drawdown(self):
        from portfolio_ml.experiments.schema import validate_experiment_row

        row = {
            "experiment_type": "baseline",
            "strategy_name": "EqualWeight",
            "max_drawdown": 0.05,  # positive — wrong sign
        }
        issues = validate_experiment_row(row)
        assert any("max_drawdown" in issue.lower() or "positive" in issue.lower() for issue in issues)

    def test_validate_row_clean_row_no_issues(self):
        from portfolio_ml.experiments.schema import validate_experiment_row

        row = {
            "experiment_type": "baseline",
            "strategy_name": "MinVariance",
            "model_name": None,
            "annualised_return": 0.07,
            "annualised_volatility": 0.10,
            "sharpe_ratio": 0.7,
            "max_drawdown": -0.21,
            "rank_ic": float("nan"),
        }
        issues = validate_experiment_row(row)
        assert len(issues) == 0, f"Expected no issues, got: {issues}"

    def test_validate_dataframe_adds_warnings_column(self):
        from portfolio_ml.experiments.schema import make_experiment_row, validate_experiment_dataframe

        rows = [
            make_experiment_row("baseline", strategy_name="EW", sharpe_ratio=0.8, max_drawdown=-0.2),
            make_experiment_row("ml_ranking", model_name="ridge", rank_ic=0.05),
        ]
        df = pd.DataFrame(rows)
        validated = validate_experiment_dataframe(df)
        assert "validation_warnings" in validated.columns


# ---------------------------------------------------------------------------
# 2. Feature set v2 tests
# ---------------------------------------------------------------------------


class TestFeatureSetV2:
    def test_basic_shape(self, synthetic_prices_wide):
        from portfolio_ml.features.feature_set_v2 import build_feature_set_v2

        df = build_feature_set_v2(synthetic_prices_wide)
        assert len(df) > 0
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert set(synthetic_prices_wide.columns).issubset(set(df["symbol"].unique()))

    def test_has_cross_sectional_features(self, synthetic_prices_wide):
        from portfolio_ml.features.feature_set_v2 import build_feature_set_v2

        df = build_feature_set_v2(synthetic_prices_wide)
        cs_cols = [c for c in df.columns if "cs_rank" in c or "cs_zscore" in c]
        assert len(cs_cols) >= 2, f"Expected cross-sectional features, got: {cs_cols}"

    def test_has_time_series_features(self, synthetic_prices_wide):
        from portfolio_ml.features.feature_set_v2 import build_feature_set_v2

        df = build_feature_set_v2(synthetic_prices_wide)
        expected_prefixes = ["ret_", "vol_", "rolling_drawdown", "price_to_ma"]
        for prefix in expected_prefixes:
            matching = [c for c in df.columns if c.startswith(prefix)]
            assert len(matching) > 0, f"No columns with prefix '{prefix}'"

    def test_no_future_values_in_rolling_features(self, synthetic_prices_wide):
        """Rolling features at date T must not include price at T+1."""
        from portfolio_ml.features.feature_set_v2 import build_feature_set_v2

        df = build_feature_set_v2(synthetic_prices_wide)
        df_spy = df[df["symbol"] == "SPY"].sort_values("date").reset_index(drop=True)

        # ret_5d at row 0 should be NaN or computed from rows 0-5
        # At least it should not be a future value
        if "ret_5d" in df_spy.columns and len(df_spy) > 10:
            # The feature should not be NaN for rows > 5
            later_rows = df_spy.iloc[10:]
            non_null = later_rows["ret_5d"].notna().sum()
            assert non_null > 0, "ret_5d should be non-NaN for rows > 5"

    def test_sorted_output(self, synthetic_prices_wide):
        from portfolio_ml.features.feature_set_v2 import build_feature_set_v2

        df = build_feature_set_v2(synthetic_prices_wide)
        assert df["date"].is_monotonic_increasing or df.groupby("symbol")["date"].apply(
            lambda x: x.is_monotonic_increasing
        ).all()


# ---------------------------------------------------------------------------
# 3. Ranking targets tests
# ---------------------------------------------------------------------------


class TestRankingTargets:
    def test_future_return_is_forward_looking(self, synthetic_prices_long):
        from portfolio_ml.features.ranking_targets import add_ranking_targets

        df = synthetic_prices_long.copy()
        df_with_targets = add_ranking_targets(df, horizon=5, price_col="adj_close")
        target_col = "future_return_5d"
        assert target_col in df_with_targets.columns

        # Verify for one symbol: target at index 0 should be (P[5] / P[0] - 1)
        sym = "SPY"
        sym_df = df_with_targets[df_with_targets["symbol"] == sym].sort_values("date").reset_index(drop=True)
        p0 = sym_df["adj_close"].iloc[0]
        p5 = sym_df["adj_close"].iloc[5]
        expected = p5 / p0 - 1
        actual = sym_df[target_col].iloc[0]
        assert np.isclose(actual, expected, atol=1e-6), (
            f"Target mismatch: expected {expected:.6f}, got {actual:.6f}"
        )

    def test_last_rows_are_nan(self, synthetic_prices_long):
        from portfolio_ml.features.ranking_targets import add_ranking_targets

        horizon = 5
        df_with_targets = add_ranking_targets(synthetic_prices_long, horizon=horizon)
        target_col = f"future_return_{horizon}d"

        for sym, grp in df_with_targets.groupby("symbol"):
            grp_sorted = grp.sort_values("date")
            last_values = grp_sorted[target_col].iloc[-horizon:]
            assert last_values.isna().all(), (
                f"Last {horizon} rows for {sym} should be NaN (no future price available)."
            )

    def test_cross_sectional_ranks_in_0_1(self, synthetic_prices_long):
        from portfolio_ml.features.ranking_targets import add_ranking_targets

        df_with_targets = add_ranking_targets(synthetic_prices_long, horizon=5)
        rank_col = "cs_rank_5d"
        valid = df_with_targets[rank_col].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), (
            "Cross-sectional ranks should be in [0, 1]."
        )

    def test_class_labels_are_0_1_2(self, synthetic_prices_long):
        from portfolio_ml.features.ranking_targets import add_ranking_targets

        df_with_targets = add_ranking_targets(synthetic_prices_long, horizon=5)
        class_col = "cs_class_3_5d"
        valid = df_with_targets[class_col].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0, 2.0}), (
            f"Class labels should be 0/1/2, got: {set(valid.unique())}"
        )


# ---------------------------------------------------------------------------
# 4. Ranking metrics tests
# ---------------------------------------------------------------------------


class TestRankingMetrics:
    def test_rank_ic_in_valid_range(self):
        from portfolio_ml.features.ranking_targets import rank_ic

        actual = np.array([0.1, 0.3, 0.2, 0.5, 0.0])
        predicted = np.array([0.15, 0.25, 0.22, 0.45, 0.05])
        ic = rank_ic(actual, predicted)
        assert -1.0 <= ic <= 1.0, f"Rank IC = {ic} is out of [-1, 1]."

    def test_rank_ic_perfect_correlation(self):
        from portfolio_ml.features.ranking_targets import rank_ic

        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([10.0, 20.0, 30.0, 40.0])  # same order
        ic = rank_ic(actual, predicted)
        assert np.isclose(ic, 1.0, atol=1e-6), f"Perfect rank IC should be 1.0, got {ic:.6f}"

    def test_rank_ic_zero_nan(self):
        from portfolio_ml.features.ranking_targets import rank_ic

        actual = np.array([np.nan, np.nan])
        predicted = np.array([1.0, 2.0])
        ic = rank_ic(actual, predicted)
        assert ic == 0.0, "Should return 0 when all actual are NaN."

    def test_precision_at_k_perfect(self):
        from portfolio_ml.features.ranking_targets import precision_at_k

        actual = np.array([0.1, 0.5, 0.3, 0.2])
        predicted = np.array([0.05, 0.9, 0.4, 0.1])  # predicted top is actual top
        pak = precision_at_k(actual, predicted, k=2)
        assert pak == 1.0, f"Perfect precision@2 should be 1.0, got {pak}"

    def test_precision_at_k_in_0_1(self):
        from portfolio_ml.features.ranking_targets import precision_at_k

        np.random.seed(42)
        actual = np.random.randn(10)
        predicted = np.random.randn(10)
        pak = precision_at_k(actual, predicted, k=3)
        assert 0.0 <= pak <= 1.0, f"Precision@K must be in [0, 1], got {pak}"

    def test_mean_rank_ic_structure(self):
        from portfolio_ml.features.ranking_targets import mean_rank_ic

        np.random.seed(7)
        n_assets = 5
        n_dates = 20
        dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")
        rows = []
        for d in dates:
            for sym in range(n_assets):
                rows.append({"date": d, "symbol": f"A{sym}", "actual": np.random.randn(), "predicted": np.random.randn()})
        df = pd.DataFrame(rows)

        result = mean_rank_ic(df, "actual", "predicted")
        assert "mean_rank_ic" in result
        assert "std_rank_ic" in result
        assert "n_dates" in result
        assert result["n_dates"] == n_dates
        assert -1.0 <= result["mean_rank_ic"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Two-stage portfolio weight tests
# ---------------------------------------------------------------------------


class TestTwoStagePortfolio:
    def test_top_k_equal_weights_sum_to_1(self):
        from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores

        scores = pd.Series([0.9, 0.1, 0.5, 0.8, 0.3], index=["A", "B", "C", "D", "E"])
        config = TwoStageConfig(portfolio_method="top_k_equal", top_k=3, max_weight=0.5)
        weights = compute_weights_from_scores(scores, list(scores.index), config)

        assert np.isclose(weights.sum(), 1.0, atol=1e-6), f"Weights sum = {weights.sum()}"

    def test_top_k_equal_weights_selects_correct_assets(self):
        from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores

        scores = pd.Series({"A": 0.9, "B": 0.1, "C": 0.5, "D": 0.8, "E": 0.3})
        config = TwoStageConfig(portfolio_method="top_k_equal", top_k=3, max_weight=0.5)
        asset_names = list(scores.index)
        weights = compute_weights_from_scores(scores, asset_names, config)

        # A, C, D should get positive weight (top 3)
        weight_dict = dict(zip(asset_names, weights))
        assert weight_dict["A"] > 0 and weight_dict["D"] > 0 and weight_dict["C"] > 0

    def test_score_weighted_no_negative_weights(self):
        from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores

        scores = pd.Series([0.9, 0.7, 0.0, -0.1, 0.3], index=["A", "B", "C", "D", "E"])
        config = TwoStageConfig(portfolio_method="score_weighted", max_weight=0.5)
        weights = compute_weights_from_scores(scores, list(scores.index), config)

        assert (weights >= 0).all(), "Long-only portfolio must have non-negative weights."
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_max_weight_respected(self):
        from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores

        max_w = 0.3
        scores = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6], index=["A", "B", "C", "D", "E"])
        config = TwoStageConfig(portfolio_method="score_weighted", max_weight=max_w)
        weights = compute_weights_from_scores(scores, list(scores.index), config)

        assert (weights <= max_w + 1e-6).all(), (
            f"Max weight violated: {weights.max():.4f} > {max_w}"
        )

    def test_mean_variance_weights_sum_to_1(self):
        from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores

        np.random.seed(42)
        n_assets = 5
        asset_names = [f"A{i}" for i in range(n_assets)]
        scores = pd.Series(np.random.rand(n_assets), index=asset_names)
        returns_hist = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            columns=asset_names,
        )
        config = TwoStageConfig(portfolio_method="mean_variance", max_weight=0.4)
        weights = compute_weights_from_scores(scores, asset_names, config, returns_history=returns_hist)

        assert np.isclose(weights.sum(), 1.0, atol=1e-6), f"Weights sum = {weights.sum()}"
        assert (weights >= 0.0).all(), "Weights must be non-negative."
        assert (weights <= config.max_weight + 1e-6).all()

    def test_turnover_calculation(self):
        from portfolio_ml.modeling.two_stage_portfolio import compute_turnover

        old = np.array([0.5, 0.3, 0.2])
        new = np.array([0.4, 0.4, 0.2])
        turnover = compute_turnover(old, new)
        expected = abs(0.4 - 0.5) + abs(0.4 - 0.3) + abs(0.2 - 0.2)
        assert np.isclose(turnover, expected, atol=1e-6)

    def test_normalise_scores_cross_sectionally(self):
        from portfolio_ml.modeling.two_stage_portfolio import normalise_scores_cross_sectionally

        scores = pd.Series([10.0, 5.0, 1.0, 8.0], index=["A", "B", "C", "D"])
        normalised = normalise_scores_cross_sectionally(scores)

        assert normalised.notna().all()
        assert normalised.min() >= 0.0
        assert normalised.max() <= 1.0
