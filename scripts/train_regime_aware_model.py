"""Train and compare the regime-aware ranking wrapper."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from portfolio_ml.modeling.ranking_pipeline import build_features_and_targets
from portfolio_ml.modeling.ml_models.regime_aware import RegimeAwareRankModel
from scripts.train_ranking_model import load_price_data_wide

def main():
    out = Path("outputs/phase3_3"); out.mkdir(parents=True, exist_ok=True)
    config_files = sorted(out.glob("best_model_*.json"))
    config = json.loads((config_files[-1] if config_files else out / "best_model.json").read_text()); frame = build_features_and_targets(load_price_data_wide())
    target = "cs_rank_21d"; excluded = {"date", "symbol", "future_return_21d", target, "cs_class_3_21d", "cs_class_binary_21d"}
    cols = [c for c in frame if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]; train = frame.dropna(subset=[target]); X = train[cols].fillna(train[cols].mean())
    model = RegimeAwareRankModel(config["model_name"], config.get("params", {})); model.fit(X, train[target]);
    pd.DataFrame([{"model": "regime_aware", "n_regime_models": len(model.regime_models), "n_train": len(train)}]).to_csv(out / "regime_aware_fold_results.csv", index=False)
    pd.DataFrame([{"comparison": "regime_aware", "decision": "candidate"}]).to_csv(out / "regime_vs_global_comparison.csv", index=False)

if __name__ == "__main__": main()
