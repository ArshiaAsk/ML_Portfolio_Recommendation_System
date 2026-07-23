"""Leakage-safe evaluation helpers for Phase 3.3 selection and holdout work."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from portfolio_ml.backtesting.engine import BacktestEngine
from portfolio_ml.evaluation.significance import paired_bootstrap_sharpe_diff
from portfolio_ml.features.ranking_targets import precision_at_k, rank_ic
from portfolio_ml.modeling.ml_models.factory import build_model
from portfolio_ml.modeling.ml_models.regime_aware import RegimeAwareRankModel
from portfolio_ml.modeling.scalers import TrainOnlyScaler
from portfolio_ml.modeling.two_stage_portfolio import (
    TwoStageConfig, compute_turnover, compute_weights_from_scores,
    normalise_scores_cross_sectionally,
)


@dataclass(frozen=True)
class RiskControlConfig:
    """Optional controls evaluated without changing the base pipeline."""
    volatility_scaled: bool = False
    sticky_fraction: float = 0.0
    drawdown_threshold: float | None = None
    drawdown_exposure: float = 1.0
    turnover_penalty: float = 0.0


def _feature_columns(frame: pd.DataFrame, horizon: int) -> list[str]:
    excluded = {"date", "symbol", f"future_return_{horizon}d", f"cs_rank_{horizon}d",
                f"cs_class_3_{horizon}d", f"cs_class_binary_{horizon}d"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def frozen_holdout_predictions(features: pd.DataFrame, model_name: str, params: dict,
                               cutoff: str | pd.Timestamp, horizon: int = 21,
                               regime_aware: bool = False) -> pd.DataFrame:
    """Fit once through ``cutoff`` and score each date after the cutoff."""
    cutoff = pd.Timestamp(cutoff)
    target = f"cs_rank_{horizon}d"
    # A target at date t uses prices t+horizon. Remove the final horizon
    # pre-cutoff dates so no label can read beyond the frozen information set.
    pre_cutoff_dates = sorted(features.loc[features["date"] <= cutoff, "date"].unique())
    usable_dates = set(pre_cutoff_dates[:-horizon]) if len(pre_cutoff_dates) > horizon else set()
    train = features[features["date"].isin(usable_dates) & features[target].notna()].copy()
    test = features[features["date"] > cutoff].copy()
    columns = _feature_columns(features, horizon)
    if train.empty or test.empty:
        raise ValueError("holdout requires non-empty pre-cutoff training and post-cutoff test data")
    X_train = train[columns]
    valid = X_train.notna().all(axis=1)
    X_train = X_train.loc[valid]
    y_train = train.loc[valid, target]
    scaler = TrainOnlyScaler(StandardScaler())
    train_scaled = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    train_means = X_train.mean()
    model = (RegimeAwareRankModel(model_name, params) if regime_aware else build_model(model_name, **params))
    # Regime labels must remain available to the wrapper, while base models use
    # the same scaled feature matrix and train-only imputation.
    if regime_aware:
        # Preserve the raw regime-signal semantics (sign and flag) for the
        # router; scaling those columns would change the regime definition.
        model.fit(X_train.fillna(train_means), y_train.to_numpy())
    else:
        model.fit(train_scaled, y_train.to_numpy())
    rows = []
    for date, group in test.groupby("date", sort=True):
        X_test = group[columns].fillna(train_means)
        scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0)
        scores = model.predict(X_test if regime_aware else scaled)
        result = group[["date", "symbol", f"future_return_{horizon}d"]].copy()
        result["predicted_score"] = scores
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def ranking_metrics(predictions: pd.DataFrame, horizon: int = 21, top_k: int = 3) -> dict:
    target = f"future_return_{horizon}d"
    ics, paks = [], []
    for _, group in predictions.dropna(subset=[target]).groupby("date"):
        if len(group) >= 2:
            ics.append(rank_ic(group[target].to_numpy(), group.predicted_score.to_numpy()))
            paks.append(precision_at_k(group[target].to_numpy(), group.predicted_score.to_numpy(), top_k))
    return {"rank_ic": float(np.mean(ics)) if ics else 0.0,
            "rank_ic_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
            "precision_at_3": float(np.mean(paks)) if paks else 0.0,
            "n_folds_or_dates": len(ics)}


def _risk_adjust_weights(weights: np.ndarray, returns_history: pd.DataFrame | None,
                         previous: np.ndarray | None, control: RiskControlConfig,
                         drawdown: float = 0.0) -> np.ndarray:
    adjusted = weights.copy()
    if control.volatility_scaled and returns_history is not None:
        vols = returns_history.std().replace(0, np.nan).fillna(returns_history.std().mean()).to_numpy()
        adjusted = adjusted / np.maximum(vols, 1e-8)
    if previous is not None and control.sticky_fraction > 0:
        adjusted = control.sticky_fraction * previous + (1 - control.sticky_fraction) * adjusted
    if control.drawdown_threshold is not None and drawdown <= control.drawdown_threshold:
        adjusted *= control.drawdown_exposure
        adjusted = adjusted + (1 - adjusted.sum()) / len(adjusted)
    return np.clip(adjusted, 0, None) / max(adjusted.sum(), 1e-12)


def evaluate_portfolio_methods(predictions: pd.DataFrame, prices: pd.DataFrame,
                               top_k: int = 5, max_weight: float = .25,
                               transaction_cost_bps: float = 5.0,
                               risk_control: RiskControlConfig | None = None) -> pd.DataFrame:
    """Evaluate all allocation methods on exactly the same prediction dates."""
    control = risk_control or RiskControlConfig()
    assets = list(prices.columns)
    output = []
    for method in ("top_k_equal", "score_weighted", "mean_variance"):
        config = TwoStageConfig(portfolio_method=method, top_k=top_k, max_weight=max_weight,
                                transaction_cost_bps=transaction_cost_bps,
                                rebalance_frequency=21)
        schedule, previous = {}, None
        for date, group in predictions.groupby("date", sort=True):
            scores = normalise_scores_cross_sectionally(pd.Series(group.predicted_score.to_numpy(), index=group.symbol))
            history = prices.loc[:date].pct_change().iloc[-config.cov_window:] if (method == "mean_variance" or control.volatility_scaled) else None
            weights = compute_weights_from_scores(scores, assets, config, history)
            weights = _risk_adjust_weights(weights, history, previous, control)
            schedule[pd.Timestamp(date)] = weights
            previous = weights
        if not schedule:
            continue
        aligned_prices = prices.loc[min(schedule):]
        result = BacktestEngine(aligned_prices, schedule, transaction_cost=transaction_cost_bps / 10000).run()
        metrics = result["metrics"]
        row = {"portfolio_method": method, **metrics,
               "transaction_cost_bps": transaction_cost_bps,
               "risk_control": control.__dict__}
        output.append(row)
    return pd.DataFrame(output)


def compare_to_equal_weight(strategy_returns: pd.Series, prices: pd.DataFrame,
                             block: int = 21, n_resamples: int = 1000) -> dict:
    equal = prices.loc[strategy_returns.index].pct_change().mean(axis=1).fillna(0.0)
    return paired_bootstrap_sharpe_diff(strategy_returns.to_numpy(), equal.to_numpy(), block, n_resamples)
