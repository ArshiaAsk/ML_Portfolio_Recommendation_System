"""Daily market data pipeline: fetch → validate → clean → features → DuckDB."""

from __future__ import annotations

from datetime import date

from prefect import flow, task

from portfolio_ml.config.loader import load_assets_config, load_pipeline_config
from portfolio_ml.data_sources.yahoo import YahooFinanceSource
from portfolio_ml.ingestion.ingest_prices import ingest_raw_prices
from portfolio_ml.storage.duckdb_client import DuckDBClient
from portfolio_ml.storage.parquet import read_parquet_dataset, write_partitioned_parquet
from portfolio_ml.transformations.build_features import build_asset_daily_features
from portfolio_ml.transformations.clean_prices import clean_daily_prices
from portfolio_ml.utils.dates import parse_date, resolve_end_date
from portfolio_ml.utils.logging import get_logger
from portfolio_ml.validation.checks import (
    validate_asset_features,
    validate_clean_prices,
    validate_raw_prices,
)

logger = get_logger(__name__)


@task(name="load_configs")
def load_configs() -> tuple:
    """Load assets and pipeline configuration from YAML files."""
    assets_cfg = load_assets_config()
    pipeline_cfg = load_pipeline_config()
    return assets_cfg, pipeline_cfg


@task(name="extract_prices")
def extract_prices(
    symbols: list[str],
    start_date: date,
    end_date: date,
    raw_data_dir: str,
    interval: str,
):
    """Fetch and persist raw prices from Yahoo Finance."""
    source = YahooFinanceSource()
    raw_df = ingest_raw_prices(
        source=source,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        raw_data_dir=raw_data_dir,
        interval=interval,
    )
    logger.info("extract_prices: %d rows fetched and saved.", len(raw_df))
    return raw_df


@task(name="validate_raw_prices")
def validate_raw_prices_task(raw_df, reports_dir: str):
    """Validate raw prices against schema."""
    validated = validate_raw_prices(raw_df, reports_dir=reports_dir)
    logger.info("validate_raw_prices: validation passed.")
    return validated


@task(name="clean_prices")
def clean_prices_task(raw_df):
    """Clean and standardize raw prices."""
    clean_df = clean_daily_prices(raw_df)
    logger.info("clean_prices: produced %d clean rows.", len(clean_df))
    return clean_df


@task(name="validate_clean_prices")
def validate_clean_prices_task(clean_df, reports_dir: str):
    """Validate clean prices against schema."""
    validated = validate_clean_prices(clean_df, reports_dir=reports_dir)
    logger.info("validate_clean_prices: validation passed.")
    return validated


@task(name="save_clean_prices")
def save_clean_prices_task(clean_df, processed_data_dir: str):
    """Persist clean prices to processed layer partitioned by year."""
    clean_df_with_year = clean_df.copy()
    clean_df_with_year["year"] = clean_df_with_year["date"].apply(
        lambda d: d.year if hasattr(d, "year") else d
    )

    write_partitioned_parquet(
        df=clean_df_with_year,
        base_path=f"{processed_data_dir}/daily_prices",
        partition_cols=["year"],
        filename="daily_prices.parquet",
    )
    logger.info("save_clean_prices: saved to '%s/daily_prices'", processed_data_dir)


@task(name="build_features")
def build_features_task(clean_df):
    """Generate asset daily features."""
    features_df = build_asset_daily_features(clean_df)
    logger.info("build_features: produced %d feature rows.", len(features_df))
    return features_df


@task(name="validate_features")
def validate_features_task(features_df, reports_dir: str):
    """Validate features against schema."""
    validated = validate_asset_features(features_df, reports_dir=reports_dir)
    logger.info("validate_features: validation passed.")
    return validated


@task(name="save_features")
def save_features_task(features_df, marts_data_dir: str):
    """Persist features to marts layer partitioned by year."""
    features_df_with_year = features_df.copy()
    features_df_with_year["year"] = features_df_with_year["date"].apply(
        lambda d: d.year if hasattr(d, "year") else d
    )

    write_partitioned_parquet(
        df=features_df_with_year,
        base_path=f"{marts_data_dir}/features",
        partition_cols=["year"],
        filename="asset_daily_features.parquet",
    )
    logger.info("save_features: saved to '%s/features'", marts_data_dir)


@task(name="create_duckdb_views")
def create_duckdb_views_task(
    duckdb_path: str,
    raw_data_dir: str,
    processed_data_dir: str,
    marts_data_dir: str,
):
    """Create or update DuckDB views over the Parquet data lake."""
    client = DuckDBClient(
        db_path=duckdb_path,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        marts_data_dir=marts_data_dir,
    )
    with client:
        client.create_views()
        logger.info("create_duckdb_views: views created in '%s'", duckdb_path)


@flow(name="daily_market_data_pipeline")
def daily_market_data_pipeline():
    """Orchestrate the full daily market data ETL pipeline.

    Steps:
        1. Load configs
        2. Extract raw prices
        3. Validate raw prices
        4. Clean prices
        5. Validate clean prices
        6. Save clean prices
        7. Build features
        8. Validate features
        9. Save features
        10. Create DuckDB views
    """
    logger.info("=" * 80)
    logger.info("Starting daily_market_data_pipeline")
    logger.info("=" * 80)

    # Load configs
    assets_cfg, pipeline_cfg = load_configs()
    symbols = assets_cfg.symbols
    start_date = parse_date(pipeline_cfg.start_date)
    end_date = resolve_end_date(pipeline_cfg.end_date)
    logger.info("Pipeline config: start_date=%s, end_date=%s", start_date, end_date)

    # Extract raw prices
    raw_df = extract_prices(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        raw_data_dir=pipeline_cfg.raw_data_dir,
        interval=pipeline_cfg.interval,
    )

    # Validate raw prices
    validated_raw = validate_raw_prices_task(
        raw_df, reports_dir=pipeline_cfg.validation_reports_dir
    )

    # Clean prices
    clean_df = clean_prices_task(validated_raw)

    # Validate clean prices
    validated_clean = validate_clean_prices_task(
        clean_df, reports_dir=pipeline_cfg.validation_reports_dir
    )

    # Save clean prices
    save_clean_prices_task(validated_clean, pipeline_cfg.processed_data_dir)

    # Build features
    features_df = build_features_task(validated_clean)

    # Validate features
    validated_features = validate_features_task(
        features_df, reports_dir=pipeline_cfg.validation_reports_dir
    )

    # Save features
    save_features_task(validated_features, pipeline_cfg.marts_data_dir)

    # Create DuckDB views
    create_duckdb_views_task(
        duckdb_path=pipeline_cfg.duckdb_path,
        raw_data_dir=pipeline_cfg.raw_data_dir,
        processed_data_dir=pipeline_cfg.processed_data_dir,
        marts_data_dir=pipeline_cfg.marts_data_dir,
    )

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    daily_market_data_pipeline()
