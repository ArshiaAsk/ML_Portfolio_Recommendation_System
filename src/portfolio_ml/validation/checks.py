"""Validation orchestration: run Pandera schemas and persist reports."""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from portfolio_ml.utils.dates import now_utc
from portfolio_ml.utils.logging import get_logger
from portfolio_ml.validation.schemas import (
    AssetDailyFeaturesSchema,
    CleanDailyPricesSchema,
    RawDailyPricesSchema,
)

logger = get_logger(__name__)


def _check_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    table_name: str,
) -> None:
    """Raise ValueError if duplicate primary keys are found.

    Args:
        df: DataFrame to check.
        key_cols: Columns forming the composite primary key.
        table_name: Name used in the error message.

    Raises:
        ValueError: If any duplicate key combination is found.
    """
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        n = dupes.sum()
        examples = df[dupes][key_cols].head(5).to_dict(orient="records")
        raise ValueError(
            f"[{table_name}] Found {n} duplicate rows on key {key_cols}. "
            f"Example duplicates: {examples}"
        )


def _save_report(
    table_name: str,
    status: str,
    details: dict,
    reports_dir: str,
) -> None:
    """Persist a validation report as JSON.

    Args:
        table_name: Table being validated.
        status: ``"passed"`` or ``"failed"``.
        details: Dict with additional info (error messages, row counts, etc.).
        reports_dir: Directory where reports are written.
    """
    ts = now_utc().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{table_name}_{ts}.json"

    report = {
        "table": table_name,
        "status": status,
        "timestamp": now_utc().isoformat(),
        **details,
    }
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Validation report saved to '%s'", out_path)


def validate_raw_prices(
    df: pd.DataFrame,
    reports_dir: str = "data/validation_reports",
) -> pd.DataFrame:
    """Validate raw daily prices against the RawDailyPricesSchema.

    Args:
        df: Raw prices DataFrame to validate.
        reports_dir: Directory where validation reports are written.

    Returns:
        The validated DataFrame (unchanged if validation passes).

    Raises:
        pa.errors.SchemaError: If schema validation fails.
        ValueError: If duplicate primary keys are found.
    """
    table_name = "raw_daily_prices"
    logger.info("Validating raw daily prices (%d rows)...", len(df))

    try:
        _check_duplicates(df, ["symbol", "date", "source"], table_name)
        validated = RawDailyPricesSchema.validate(df, lazy=True)
        _save_report(
            table_name, "passed", {"row_count": len(df)}, reports_dir
        )
        logger.info("Raw daily prices validation passed (%d rows).", len(df))
        return validated
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        _save_report(
            table_name,
            "failed",
            {"row_count": len(df), "error": str(exc)},
            reports_dir,
        )
        raise
    except ValueError as exc:
        _save_report(
            table_name,
            "failed",
            {"row_count": len(df), "error": str(exc)},
            reports_dir,
        )
        raise


def validate_clean_prices(
    df: pd.DataFrame,
    reports_dir: str = "data/validation_reports",
) -> pd.DataFrame:
    """Validate clean daily prices against the CleanDailyPricesSchema.

    Args:
        df: Clean prices DataFrame to validate.
        reports_dir: Directory where validation reports are written.

    Returns:
        The validated DataFrame.

    Raises:
        pa.errors.SchemaError: If schema validation fails.
        ValueError: If duplicate primary keys are found.
    """
    table_name = "clean_daily_prices"
    logger.info("Validating clean daily prices (%d rows)...", len(df))

    try:
        _check_duplicates(df, ["symbol", "date"], table_name)
        validated = CleanDailyPricesSchema.validate(df, lazy=True)
        _save_report(
            table_name, "passed", {"row_count": len(df)}, reports_dir
        )
        logger.info("Clean daily prices validation passed (%d rows).", len(df))
        return validated
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        _save_report(
            table_name,
            "failed",
            {"row_count": len(df), "error": str(exc)},
            reports_dir,
        )
        raise
    except ValueError as exc:
        _save_report(
            table_name,
            "failed",
            {"row_count": len(df), "error": str(exc)},
            reports_dir,
        )
        raise


def validate_asset_features(
    df: pd.DataFrame,
    reports_dir: str = "data/validation_reports",
) -> pd.DataFrame:
    """Validate asset daily features against the AssetDailyFeaturesSchema.

    Args:
        df: Features DataFrame to validate.
        reports_dir: Directory where validation reports are written.

    Returns:
        The validated DataFrame.

    Raises:
        pa.errors.SchemaError: If schema validation fails.
        ValueError: If duplicate primary keys are found.
    """
    table_name = "asset_daily_features"
    logger.info("Validating asset daily features (%d rows)...", len(df))

    try:
        _check_duplicates(df, ["symbol", "date"], table_name)
        validated = AssetDailyFeaturesSchema.validate(df, lazy=True)
        _save_report(
            table_name, "passed", {"row_count": len(df)}, reports_dir
        )
        logger.info("Asset daily features validation passed (%d rows).", len(df))
        return validated
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        _save_report(
            table_name,
            "failed",
            {"row_count": len(df), "error": str(exc)},
            reports_dir,
        )
        raise
    except ValueError as exc:
        _save_report(
            table_name,
            "failed",
            {"row_count": len(df), "error": str(exc)},
            reports_dir,
        )
        raise
