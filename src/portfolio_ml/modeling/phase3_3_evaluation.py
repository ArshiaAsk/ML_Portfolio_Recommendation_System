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
    TwoStageConfig,
    compute_weights_from_scores,
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
    """Numeric feature columns excluding identifiers and label targets."""
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
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def frozen_holdout_predictions(
    features: pd.DataFrame,
    model_name: str,
    params: dict,
    cutoff: str | pd.Timestamp,
    horizon: int = 21,
    regime_aware: bool = False,
) -> pd.DataFrame:
    """Fit once through ``cutoff`` and score each date after the cutoff."""
    cutoff = pd.Timestamp(cutoff)
    target = f"cs_rank_{horizon}d"

    # A target at date t uses prices through t+horizon. Drop the final horizon
    # pre-cutoff dates so no label can read beyond the frozen information set.
    pre_cutoff_dates = sorted(
        features.loc[features["date"] <= cutoff, "date"].unique()
    )
    usable_dates = (
        set(pre_cutoff_dates[:-horizon])
        if len(pre_cutoff_dates) > horizon
        else set()
    )

    train = features[
        features["date"].isin(usable_dates) & features[target].notna()
    ].copy()
    test = features[features["date"] > cutoff].copy()
    columns = _feature_columns(features, horizon)

    if train.empty or test.empty:
        raise ValueError(
            "holdout requires non-empty pre-cutoff training and post-cutoff test data"
        )

    X_train = train[columns]
    valid = X_train.notna().all(axis=1)
    X_train = X_train.loc[valid]
    y_train = train.loc[valid, target]

    scaler = TrainOnlyScaler(StandardScaler())
    train_scaled = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    train_means = X_train.mean()

    if regime_aware:
        # Preserve raw regime-signal semantics (sign and flag) for the router;
        # scaling those columns would change the regime definition.
        model = RegimeAwareRankModel(model_name, params)
        model.fit(X_train.fillna(train_means), y_train.to_numpy())
    else:
        model = build_model(model_name, **params)
        model.fit(train_scaled, y_train.to_numpy())

    rows: list[pd.DataFrame] = []
    for _, group in test.groupby("date", sort=True):
        X_test = group[columns].fillna(train_means)
        if regime_aware:
            scores = model.predict(X_test)
        else:
            scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0)
            scores = model.predict(scaled)

        result = group[["date", "symbol", f"future_return_{horizon}d"]].copy()
        result["predicted_score"] = scores
        rows.append(result)

    return pd.concat(rows, ignore_index=True)


def ranking_metrics(
    predictions: pd.DataFrame,
    horizon: int = 21,
    top_k: int = 3,
) -> dict:
    """Aggregate Rank IC and Precision@K across prediction dates."""
    target = f"future_return_{horizon}d"
    ics: list[float] = []
    paks: list[float] = []

    for _, group in predictions.dropna(subset=[target]).groupby("date"):
        if len(group) < 2:
            continue
        actual = group[target].to_numpy()
        scores = group["predicted_score"].to_numpy()
        ics.append(rank_ic(actual, scores))
        paks.append(precision_at_k(actual, scores, top_k))

    return {
        "rank_ic": float(np.mean(ics)) if ics else 0.0,
        "rank_ic_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
        "precision_at_3": float(np.mean(paks)) if paks else 0.0,
        "n_folds_or_dates": len(ics),
    }


def _risk_adjust_weights(
    weights: np.ndarray,
    returns_history: pd.DataFrame | None,
    previous: np.ndarray | None,
    control: RiskControlConfig,
    drawdown: float = 0.0,
) -> np.ndarray:
    """Apply optional post-hoc risk overlays to a weight vector."""
    adjusted = weights.copy()

    if control.volatility_scaled and returns_history is not None:
        vols = (
            returns_history.std()
            .replace(0, np.nan)
            .fillna(returns_history.std().mean())
            .to_numpy()
        )
        adjusted = adjusted / np.maximum(vols, 1e-8)

    if previous is not None and control.sticky_fraction > 0:
        adjusted = (
            control.sticky_fraction * previous
            + (1.0 - control.sticky_fraction) * adjusted
        )

    if (
        control.drawdown_threshold is not None
        and drawdown <= control.drawdown_threshold
    ):
        adjusted *= control.drawdown_exposure
        adjusted = adjusted + (1.0 - adjusted.sum()) / len(adjusted)

    total = max(adjusted.sum(), 1e-12)
    return np.clip(adjusted, 0.0, None) / total


def evaluate_portfolio_methods(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    top_k: int = 5,
    max_weight: float = 0.25,
    transaction_cost_bps: float = 5.0,
    risk_control: RiskControlConfig | None = None,
) -> pd.DataFrame:
    """Evaluate all allocation methods on exactly the same prediction dates."""
    control = risk_control or RiskControlConfig()
    assets = list(prices.columns)
    output: list[dict] = []

    for method in ("top_k_equal", "score_weighted", "mean_variance"):
        config = TwoStageConfig(
            portfolio_method=method,
            top_k=top_k,
            max_weight=max_weight,
            transaction_cost_bps=transaction_cost_bps,
            rebalance_frequency=21,
        )
        schedule: dict[pd.Timestamp, np.ndarray] = {}
        previous: np.ndarray | None = None

        for date, group in predictions.groupby("date", sort=True):
            scores = normalise_scores_cross_sectionally(
                pd.Series(group["predicted_score"].to_numpy(), index=group["symbol"])
            )
            needs_history = method == "mean_variance" or control.volatility_scaled
            history = (
                prices.loc[:date].pct_change().iloc[-config.cov_window :]
                if needs_history
                else None
            )
            weights = compute_weights_from_scores(scores, assets, config, history)
            weights = _risk_adjust_weights(weights, history, previous, control)
            schedule[pd.Timestamp(date)] = weights
            previous = weights

        if not schedule:
            continue

        aligned_prices = prices.loc[min(schedule) :]
        result = BacktestEngine(
            aligned_prices,
            schedule,
            transaction_cost=transaction_cost_bps / 10000,
        ).run()
        metrics = result["metrics"]
        output.append(
            {
                "portfolio_method": method,
                **metrics,
                "transaction_cost_bps": transaction_cost_bps,
                "risk_control": control.__dict__,
            }
        )

    return pd.DataFrame(output)


def compare_to_equal_weight(
    strategy_returns: pd.Series,
    prices: pd.DataFrame,
    block: int = 21,
    n_resamples: int = 1000,
) -> dict:
    """Paired block-bootstrap Sharpe difference vs equal-weight benchmark."""
    equal = (
        prices.loc[strategy_returns.index]
        .pct_change()
        .mean(axis=1)
        .fillna(0.0)
    )
    return paired_bootstrap_sharpe_diff(
        strategy_returns.to_numpy(),
        equal.to_numpy(),
        block,
        n_resamples,
    )


def thin_prediction_dates(
    predictions: pd.DataFrame,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Keep every N-th unique prediction date to enforce practical rebalancing."""
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be >= 1")
    dates = sorted(predictions["date"].unique())
    keep = set(dates[::rebalance_every])
    return predictions[predictions["date"].isin(keep)].copy()


def smooth_prediction_scores(
    predictions: pd.DataFrame,
    span: int | None,
) -> pd.DataFrame:
    """Optionally EMA-smooth scores by symbol to reduce day-to-day churn."""
    if span is None or span <= 1:
        return predictions.copy()

    out = predictions.sort_values(["symbol", "date"]).copy()
    out["predicted_score"] = out.groupby("symbol")["predicted_score"].transform(
        lambda series: series.ewm(span=span, adjust=False).mean()
    )
    return out


def _build_weight_schedule(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    portfolio_method: str,
    top_k: int,
    max_weight: float,
    transaction_cost_bps: float,
    risk_control: RiskControlConfig,
) -> dict[pd.Timestamp, np.ndarray]:
    """Map prediction dates to portfolio weights with optional risk overlays."""
    assets = list(prices.columns)
    config = TwoStageConfig(
        portfolio_method=portfolio_method,
        top_k=top_k,
        max_weight=max_weight,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_frequency=21,
    )
    schedule: dict[pd.Timestamp, np.ndarray] = {}
    previous: np.ndarray | None = None
    previous_date: pd.Timestamp | None = None
    peak_value = 1.0
    portfolio_value = 1.0

    for date, group in predictions.groupby("date", sort=True):
        date_ts = pd.Timestamp(date)
        drawdown = 0.0
        if previous is not None and previous_date is not None:
            # Approximate inter-rebalance return for drawdown overlays.
            start_px = prices.loc[previous_date, assets].to_numpy(dtype=float)
            end_px = prices.loc[date_ts, assets].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                asset_ret = end_px / start_px - 1.0
                asset_ret = np.nan_to_num(asset_ret, nan=0.0, posinf=0.0, neginf=0.0)
            portfolio_value *= 1.0 + float(np.dot(previous, asset_ret))
            peak_value = max(peak_value, portfolio_value)
            drawdown = portfolio_value / peak_value - 1.0

        scores = normalise_scores_cross_sectionally(
            pd.Series(group["predicted_score"].to_numpy(), index=group["symbol"])
        )
        needs_history = (
            portfolio_method == "mean_variance" or risk_control.volatility_scaled
        )
        history = (
            prices.loc[:date_ts].pct_change().iloc[-config.cov_window :]
            if needs_history
            else None
        )
        weights = compute_weights_from_scores(scores, assets, config, history)
        weights = _risk_adjust_weights(
            weights,
            history,
            previous,
            risk_control,
            drawdown=drawdown,
        )
        schedule[date_ts] = weights
        previous = weights
        previous_date = date_ts

    return schedule


def evaluate_equal_weight(
    prices: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Backtest an equal-weight schedule on the same rebalance calendar."""
    if not rebalance_dates:
        raise ValueError("rebalance_dates cannot be empty")

    n_assets = prices.shape[1]
    equal = np.ones(n_assets) / n_assets
    schedule = {pd.Timestamp(date): equal.copy() for date in rebalance_dates}
    aligned = prices.loc[min(schedule) :]
    result = BacktestEngine(
        aligned,
        schedule,
        transaction_cost=transaction_cost_bps / 10000,
    ).run()
    return {
        "metrics": {
            "portfolio_method": "equal_weight",
            **result["metrics"],
            "transaction_cost_bps": transaction_cost_bps,
        },
        "daily_returns": result["daily_returns"],
        "portfolio_value": result["portfolio_value"],
        "turnover": result["turnover"],
    }


def evaluate_strategy_candidate(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    portfolio_method: str,
    top_k: int = 5,
    max_weight: float = 0.25,
    transaction_cost_bps: float = 5.0,
    risk_control: RiskControlConfig | None = None,
    rebalance_every: int = 21,
    score_ema_span: int | None = None,
) -> dict:
    """Evaluate one portfolio candidate and return metrics plus daily returns."""
    control = risk_control or RiskControlConfig()
    scored = smooth_prediction_scores(predictions, score_ema_span)
    thinned = thin_prediction_dates(scored, rebalance_every=rebalance_every)
    schedule = _build_weight_schedule(
        thinned,
        prices,
        portfolio_method=portfolio_method,
        top_k=top_k,
        max_weight=max_weight,
        transaction_cost_bps=transaction_cost_bps,
        risk_control=control,
    )
    if not schedule:
        raise ValueError("candidate produced an empty rebalance schedule")

    aligned = prices.loc[min(schedule) :]
    result = BacktestEngine(
        aligned,
        schedule,
        transaction_cost=transaction_cost_bps / 10000,
    ).run()

    return {
        "metrics": {
            "portfolio_method": portfolio_method,
            **result["metrics"],
            "transaction_cost_bps": transaction_cost_bps,
            "top_k": top_k,
            "max_weight": max_weight,
            "rebalance_every": rebalance_every,
            "score_ema_span": score_ema_span,
            "risk_control": control.__dict__,
            "n_rebalances": len(schedule),
        },
        "daily_returns": result["daily_returns"],
        "portfolio_value": result["portfolio_value"],
        "turnover": result["turnover"],
        "rebalance_dates": sorted(schedule.keys()),
    }


def slice_performance(
    daily_returns: pd.Series,
    *,
    freq: str = "YE",
) -> pd.DataFrame:
    """Compute annualised return/Sharpe by calendar slice for stability checks."""
    rows: list[dict] = []
    grouped = daily_returns.groupby(pd.Grouper(freq=freq))
    for period, returns in grouped:
        if returns.empty or returns.notna().sum() < 5:
            continue
        total = float((1.0 + returns).prod() - 1.0)
        n_years = len(returns) / 252.0
        ann_ret = (1.0 + total) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
        vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
        sharpe = ann_ret / vol if vol > 1e-9 else 0.0
        rows.append(
            {
                "period": str(pd.Timestamp(period).date()),
                "n_days": int(len(returns)),
                "annualised_return": float(ann_ret),
                "sharpe_ratio": float(sharpe),
            }
        )
    return pd.DataFrame(rows)
