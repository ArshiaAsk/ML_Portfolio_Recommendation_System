"""Configuration loader for portfolio_ml.

Reads YAML config files and returns typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


@dataclass
class ExperimentPathsConfig:
    price_data_dir: str
    output_dir: str


@dataclass
class ExperimentBacktestConfig:
    transaction_cost: float
    initial_capital: float
    lookback_days: int
    window_type: str


@dataclass
class ExperimentRankingConfig:
    sharpe_weight: float
    mdd_weight: float
    turnover_weight: float


@dataclass
class ExperimentMLflowConfig:
    experiment_name: str
    tracking_uri: Optional[str] = None
    log_artifacts: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str
    lookback_windows: list[int]
    rebalance_frequencies: list[int]
    strategies: list[str]
    transaction_cost: float
    initial_capital: float
    paths: ExperimentPathsConfig
    backtest: ExperimentBacktestConfig
    ranking: ExperimentRankingConfig
    mlflow: ExperimentMLflowConfig


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

    project_root = Path(__file__).parent.parent.parent.parent
    candidate = project_root / config_path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Config file not found: '{config_path}'. "
        f"Tried absolute, CWD-relative, and project-root-relative paths."
    )


def _load_yaml(path: str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    with resolved.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping at the top level of {resolved}")
    return raw


def load_assets_config(path: str = "configs/assets.yaml") -> AssetsConfig:
    """Load and parse the assets configuration YAML."""
    raw = _load_yaml(path)
    if "assets" not in raw:
        raise ValueError(f"assets.yaml is missing top-level 'assets' key: {path}")

    resolved = _resolve_path(path)
    assets = [AssetConfig(**item) for item in raw["assets"]]
    config = AssetsConfig(assets=assets)
    logger.info("Loaded %d assets from %s", len(assets), resolved)
    return config


def load_pipeline_config(path: str = "configs/pipeline.yaml") -> PipelineConfig:
    """Load and parse the pipeline configuration YAML."""
    raw = _load_yaml(path)
    if "pipeline" not in raw:
        raise ValueError(f"pipeline.yaml is missing top-level 'pipeline' key: {path}")

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
    logger.info("Loaded pipeline config from %s", _resolve_path(path))
    return config


def load_experiment_config(path: str = "configs/experiments.yaml") -> ExperimentConfig:
    """Load and parse the systematic experiment configuration YAML."""
    raw = _load_yaml(path)
    if "experiment" not in raw:
        raise ValueError(f"experiments.yaml is missing top-level 'experiment' key: {path}")

    cfg = raw["experiment"]
    required_keys = [
        "experiment_name",
        "lookback_windows",
        "rebalance_frequencies",
        "strategies",
        "transaction_cost",
        "initial_capital",
        "paths",
        "backtest",
        "ranking",
        "mlflow",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"experiments.yaml is missing keys: {missing}")

    paths_cfg = cfg["paths"]
    backtest_cfg = cfg["backtest"]
    ranking_cfg = cfg["ranking"]
    mlflow_cfg = cfg["mlflow"]

    config = ExperimentConfig(
        experiment_name=cfg["experiment_name"],
        lookback_windows=list(cfg["lookback_windows"]),
        rebalance_frequencies=list(cfg["rebalance_frequencies"]),
        strategies=list(cfg["strategies"]),
        transaction_cost=float(cfg["transaction_cost"]),
        initial_capital=float(cfg["initial_capital"]),
        paths=ExperimentPathsConfig(
            price_data_dir=paths_cfg["price_data_dir"],
            output_dir=paths_cfg["output_dir"],
        ),
        backtest=ExperimentBacktestConfig(
            transaction_cost=float(backtest_cfg["transaction_cost"]),
            initial_capital=float(backtest_cfg["initial_capital"]),
            lookback_days=int(backtest_cfg["lookback_days"]),
            window_type=str(backtest_cfg["window_type"]),
        ),
        ranking=ExperimentRankingConfig(
            sharpe_weight=float(ranking_cfg["sharpe_weight"]),
            mdd_weight=float(ranking_cfg["mdd_weight"]),
            turnover_weight=float(ranking_cfg["turnover_weight"]),
        ),
        mlflow=ExperimentMLflowConfig(
            experiment_name=str(mlflow_cfg["experiment_name"]),
            tracking_uri=mlflow_cfg.get("tracking_uri"),
            log_artifacts=bool(mlflow_cfg.get("log_artifacts", True)),
        ),
    )
    logger.info("Loaded experiment config from %s", _resolve_path(path))
    return config
