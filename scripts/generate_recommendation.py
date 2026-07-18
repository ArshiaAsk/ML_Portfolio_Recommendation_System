"""Generate the latest Phase 3.3 portfolio recommendation."""
from __future__ import annotations
import json
from datetime import date, datetime, timezone
from pathlib import Path
import pandas as pd
from portfolio_ml.evaluation.output_validation import validate_best_model, validate_recommendation
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets
from portfolio_ml.modeling.ml_models.factory import build_model
from portfolio_ml.modeling.two_stage_portfolio import TwoStageConfig, compute_weights_from_scores, normalise_scores_cross_sectionally
from scripts.train_ranking_model import load_price_data_wide

def main():
    out = Path("outputs/phase3_3")
    candidates = sorted(out.glob("best_model_*.json"))
    config_path = candidates[-1] if candidates else out / "best_model.json"
    config = validate_best_model(json.loads(config_path.read_text()))
    prices = load_price_data_wide(); horizon = int(config.get("horizon", 21))
    frame = build_features_and_targets(prices, horizon=horizon)
    target = f"cs_rank_{horizon}d"; excluded = {"date", "symbol", f"future_return_{horizon}d", target, f"cs_class_3_{horizon}d", f"cs_class_binary_{horizon}d"}
    columns = [c for c in frame if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    train = frame.dropna(subset=[target]); latest_date = frame.date.max(); latest = frame[frame.date == latest_date]
    means = train[columns].mean(); model = build_model(config["model_name"], **config.get("params", {})); model.fit(train[columns].fillna(means), train[target])
    scores = pd.Series(model.predict(latest[columns].fillna(means)), index=latest.symbol)
    ts = TwoStageConfig(portfolio_method=config.get("portfolio_method", "top_k_equal"), top_k=int(config.get("top_k", 5)), max_weight=float(config.get("max_weight", .25)))
    assets = list(prices.columns); weights = compute_weights_from_scores(normalise_scores_cross_sectionally(scores), assets, ts)
    recommendation = pd.DataFrame({"symbol": assets, "score": scores.reindex(assets), "weight": weights})
    stamp = date.today().strftime("%Y%m%d")
    metadata = {"generation_date": stamp, "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "feature_date": str(latest_date.date()), "data_cutoff": str(prices.index.max().date()),
                "model_type": config["model_name"], "model": config, "regime_status": config.get("regime_status", "plain"),
                "portfolio_method": config["portfolio_method"], "disclaimer": "Not investment advice."}
    validate_recommendation(recommendation, metadata)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recommendation.to_csv(out / f"recommendation_{stamp}_{version}.csv", index=False)
    (out / f"recommendation_{stamp}_{version}.json").write_text(json.dumps({"metadata": metadata, "recommendation": recommendation.to_dict("records")}, indent=2))

if __name__ == "__main__": main()
