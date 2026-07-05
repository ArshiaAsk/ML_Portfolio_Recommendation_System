"""Run the unified Sprint 1–4 experiment workflow from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_ml.config.loader import load_experiment_config
from portfolio_ml.experiments.unified_runner import run_unified_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified portfolio experiment workflow")
    parser.add_argument("--data-path", type=str, default=None, help="Path to a parquet file or directory containing daily price parquet files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the unified experiment summary")
    args = parser.parse_args()

    config = load_experiment_config()
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        data_path = Path(args.data_path)
        if data_path.is_file():
            price_data = pd.read_parquet(data_path)
        else:
            parquet_files = sorted(data_path.glob("**/*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found under {data_path}")
            frames = [pd.read_parquet(path) for path in parquet_files]
            price_data = pd.concat(frames, ignore_index=True)
    else:
        price_data = pd.read_parquet(Path("data/processed/daily_prices/year=2018/daily_prices.parquet"))

    summary = run_unified_experiment_suite(price_data=price_data, config=config, output_dir=output_dir)
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
