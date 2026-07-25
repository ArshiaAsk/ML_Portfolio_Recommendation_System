"""Generate the latest Phase 3.3 portfolio recommendation.

Loads the most recent ``best_model_*.json`` artifact, fits the selected ranking
model on all available labeled history, scores the latest cross-section, and
writes a validated, versioned recommendation CSV/JSON pair.

Usage:
    PYTHONPATH=src python scripts/generate_recommendation.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_ml.evaluation.output_validation import (
    DISCLAIMER,
    validate_best_model,
    validate_recommendation,
)
from portfolio_ml.modeling.ml_models.factory import build_model
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets
from portfolio_ml.modeling.two_stage_portfolio import (
    TwoStageConfig,
    compute_weights_from_scores,
    normalise_scores_cross_sectionally,
)
from scripts.train_ranking_model import load_price_data_wide

OUTPUT_DIR = Path("outputs/phase3_3")
DEFAULT_HORIZON = 21


def _latest_best_model_path(output_dir: Path) -> Path:
    """Return the newest versioned best-model config, or the unversioned fallback."""
    candidates = sorted(output_dir.glob("best_model_*.json"))
    if candidates:
        return candidates[-1]
    fallback = output_dir / "best_model.json"
    if not fallback.exists():
        raise FileNotFoundError(
            f"No best_model_*.json found in {output_dir}. "
            "Run scripts/run_model_selection.py first."
        )
    return fallback


def _feature_columns(frame: pd.DataFrame, horizon: int) -> list[str]:
    """Numeric feature columns excluding date/symbol/label targets."""
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


def _build_recommendation(
    prices: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Fit on labeled history and produce weights for the latest feature date."""
    horizon = int(config.get("horizon", DEFAULT_HORIZON))
    frame = build_features_and_targets(prices, horizon=horizon)
    target = f"cs_rank_{horizon}d"
    columns = _feature_columns(frame, horizon)

    train = frame.dropna(subset=[target])
    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date]

    train_means = train[columns].mean()
    model = build_model(config["model_name"], **config.get("params", {}))
    model.fit(train[columns].fillna(train_means), train[target])

    scores = pd.Series(
        model.predict(latest[columns].fillna(train_means)),
        index=latest["symbol"],
    )
    ts_config = TwoStageConfig(
        portfolio_method=config.get("portfolio_method", "top_k_equal"),
        top_k=int(config.get("top_k", 5)),
        max_weight=float(config.get("max_weight", 0.25)),
    )
    assets = list(prices.columns)
    weights = compute_weights_from_scores(
        normalise_scores_cross_sectionally(scores),
        assets,
        ts_config,
    )

    recommendation = pd.DataFrame(
        {
            "symbol": assets,
            "score": scores.reindex(assets),
            "weight": weights,
        }
    )

    stamp = date.today().strftime("%Y%m%d")
    metadata = {
        "generation_date": stamp,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_date": str(latest_date.date()),
        "data_cutoff": str(prices.index.max().date()),
        "model_type": config["model_name"],
        "model": config,
        "regime_status": config.get("regime_status", "plain"),
        "portfolio_method": config["portfolio_method"],
        "disclaimer": DISCLAIMER,
    }
    return recommendation, metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = _latest_best_model_path(OUTPUT_DIR)
    config = validate_best_model(json.loads(config_path.read_text()))

    prices = load_price_data_wide()
    recommendation, metadata = _build_recommendation(prices, config)
    validate_recommendation(recommendation, metadata, today=date.today())

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamp = metadata["generation_date"]
    csv_path = OUTPUT_DIR / f"recommendation_{stamp}_{version}.csv"
    json_path = OUTPUT_DIR / f"recommendation_{stamp}_{version}.json"

    recommendation.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "recommendation": recommendation.to_dict("records"),
            },
            indent=2,
        )
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
