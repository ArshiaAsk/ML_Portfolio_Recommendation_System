"""Configuration loader for portfolio_ml.

Reads YAML config files and returns typed dataclasses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AssetConfig:
    symbol: str
    name: str
    asset_class: str
    currency: str
    source: str


@dataclass
class AssetsConfig:
    assets: list[AssetConfig]

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.assets]


@dataclass
class PipelineConfig:
    start_date: str
    end_date: Optional[str]
    interval: str
    raw_data_dir: str
    processed_data_dir: str
    marts_data_dir: str
    validation_reports_dir: str
    duckdb_path: str


def _resolve_path(config_path: str) -> Path:
    """Resolve a config path relative to the project root.

    Priority:
    1. Absolute path as-is.
    2. Relative to the current working directory.
    3. Relative to the project root (two levels up from this file).
    """
    p = Path(config_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    # Fall back to project root resolution
    project_root = Path(__file__).parent.parent.parent.parent
    candidate = project_root / config_path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Config file not found: '{config_path}'. "
        f"Tried absolute, CWD-relative, and project-root-relative paths."
    )


def load_assets_config(path: str = "configs/assets.yaml") -> AssetsConfig:
    """Load and parse the assets configuration YAML.

    Args:
        path: Path to assets.yaml (relative or absolute).

    Returns:
        :class:`AssetsConfig` instance.

    Raises:
        FileNotFoundError: If the config file cannot be located.
        ValueError: If the YAML is missing required keys.
    """
    resolved = _resolve_path(path)
    with resolved.open() as fh:
        raw = yaml.safe_load(fh)

    if "assets" not in raw:
        raise ValueError(f"assets.yaml is missing top-level 'assets' key: {resolved}")

    assets = [AssetConfig(**item) for item in raw["assets"]]
    config = AssetsConfig(assets=assets)
    logger.info("Loaded %d assets from %s", len(assets), resolved)
    return config


def load_pipeline_config(path: str = "configs/pipeline.yaml") -> PipelineConfig:
    """Load and parse the pipeline configuration YAML.

    Args:
        path: Path to pipeline.yaml (relative or absolute).

    Returns:
        :class:`PipelineConfig` instance.

    Raises:
        FileNotFoundError: If the config file cannot be located.
        ValueError: If required keys are missing.
    """
    resolved = _resolve_path(path)
    with resolved.open() as fh:
        raw = yaml.safe_load(fh)

    if "pipeline" not in raw:
        raise ValueError(f"pipeline.yaml is missing top-level 'pipeline' key: {resolved}")

    cfg = raw["pipeline"]
    required_keys = [
        "start_date",
        "interval",
        "raw_data_dir",
        "processed_data_dir",
        "marts_data_dir",
        "validation_reports_dir",
        "duckdb_path",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"pipeline.yaml is missing keys: {missing}")

    config = PipelineConfig(
        start_date=cfg["start_date"],
        end_date=cfg.get("end_date"),
        interval=cfg["interval"],
        raw_data_dir=cfg["raw_data_dir"],
        processed_data_dir=cfg["processed_data_dir"],
        marts_data_dir=cfg["marts_data_dir"],
        validation_reports_dir=cfg["validation_reports_dir"],
        duckdb_path=cfg["duckdb_path"],
    )
    logger.info("Loaded pipeline config from %s", resolved)
    return config
