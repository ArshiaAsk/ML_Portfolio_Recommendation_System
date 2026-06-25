"""Tests for configuration loader."""

import pytest
from pathlib import Path

from portfolio_ml.config.loader import (
    load_assets_config,
    load_pipeline_config,
    _resolve_path,
)


def test_load_assets_config():
    """Test that assets.yaml loads correctly."""
    cfg = load_assets_config()
    assert len(cfg.assets) == 10
    assert cfg.symbols == ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC", "DIA"]
    assert cfg.assets[0].symbol == "SPY"
    assert cfg.assets[0].name == "S&P 500 ETF"
    assert cfg.assets[0].asset_class == "equity"
    assert cfg.assets[0].currency == "USD"
    assert cfg.assets[0].source == "yfinance"


def test_load_pipeline_config():
    """Test that pipeline.yaml loads correctly."""
    cfg = load_pipeline_config()
    assert cfg.start_date == "2018-01-01"
    assert cfg.end_date is None
    assert cfg.interval == "1d"
    assert cfg.raw_data_dir == "data/raw"
    assert cfg.processed_data_dir == "data/processed"
    assert cfg.marts_data_dir == "data/marts"
    assert cfg.validation_reports_dir == "data/validation_reports"
    assert cfg.duckdb_path == "data/portfolio_ml.duckdb"


def test_load_assets_config_missing_file():
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_assets_config("nonexistent/path.yaml")


def test_resolve_path_absolute():
    """Test that _resolve_path handles absolute paths."""
    # Create a temp file
    tmp = Path("/tmp/test_config.yaml")
    tmp.write_text("test: value\n")
    resolved = _resolve_path(str(tmp))
    assert resolved.exists()
    assert resolved.is_absolute()
    tmp.unlink()


def test_resolve_path_nonexistent():
    """Test that _resolve_path raises FileNotFoundError for nonexistent paths."""
    with pytest.raises(FileNotFoundError):
        _resolve_path("this/does/not/exist.yaml")
