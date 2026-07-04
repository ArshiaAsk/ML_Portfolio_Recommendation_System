"""MLflow experiment tracking utilities for portfolio ML pipelines.

This module provides a thin wrapper around MLflow for logging parameters,
metrics, and artifacts during model development and backtesting.  All
experiments are logged to a local MLflow tracking server or to a configured
remote URI.

Setup
-----
Before using these utilities, ensure MLflow is installed:

    pip install mlflow

By default, MLflow logs to a local ``./mlruns`` directory.  To use a remote
tracking server, set the environment variable:

    export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000

Example
-------
>>> from portfolio_ml.experiments.tracking import (
...     start_experiment,
...     log_params,
...     log_metrics,
...     log_artifact,
...     end_run,
... )
>>> with start_experiment("baseline_comparison"):
...     log_params({"strategy": "MinVariance", "lookback": 252})
...     log_metrics({"sharpe_ratio": 1.23, "max_drawdown": -0.15})
...     log_artifact("results.csv")
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import MLflow to avoid hard dependency
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning(
        "MLflow not installed. Experiment tracking will be disabled. "
        "Install with: pip install mlflow"
    )


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


@contextmanager
def start_experiment(
    experiment_name: str,
    run_name: str | None = None,
    nested: bool = False,
) -> Generator[None, None, None]:
    """Start an MLflow experiment run as a context manager.

    Args:
        experiment_name: Name of the experiment (created if it doesn't exist).
        run_name: Optional human-readable name for this run.
        nested: If ``True``, this run is nested inside an active parent run.

    Yields:
        None. Use :func:`log_params`, :func:`log_metrics`, :func:`log_artifact`
        within the context to log data.

    Example:
        >>> with start_experiment("portfolio_baselines", run_name="equal_weight"):
        ...     log_params({"strategy": "EqualWeight"})
        ...     log_metrics({"sharpe": 1.2})
    """
    if not _MLFLOW_AVAILABLE:
        logger.warning("start_experiment: MLflow not available, skipping.")
        yield
        return

    try:
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info("Created new MLflow experiment: %s (ID=%s)", experiment_name, experiment_id)
        else:
            experiment_id = experiment.experiment_id

        mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested)
        logger.info("Started MLflow run: experiment=%s, run_name=%s", experiment_name, run_name)
        yield
    finally:
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended MLflow run.")


def log_params(params: Dict[str, Any]) -> None:
    """Log a dictionary of parameters to the active MLflow run.

    Args:
        params: Dict of parameter names and values.  Values are converted to
            strings internally by MLflow.

    Example:
        >>> log_params({"lookback_window": 252, "rebalance_freq": 21})
    """
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    try:
        mlflow.log_params(params)
        logger.debug("Logged params: %s", params)
    except Exception as e:
        logger.error("Failed to log params: %s", e)


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics to the active MLflow run.

    Args:
        metrics: Dict of metric names and float values.
        step: Optional step number (useful for logging training curves).

    Example:
        >>> log_metrics({"sharpe_ratio": 1.45, "max_drawdown": -0.12})
    """
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    try:
        mlflow.log_metrics(metrics, step=step)
        logger.debug("Logged metrics: %s", metrics)
    except Exception as e:
        logger.error("Failed to log metrics: %s", e)


def log_artifact(local_path: str | Path) -> None:
    """Log a file or directory as an artifact in the active MLflow run.

    Args:
        local_path: Path to a file or directory to upload.

    Example:
        >>> log_artifact("outputs/backtest_results.csv")
    """
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    try:
        path_obj = Path(local_path)
        if not path_obj.exists():
            logger.warning("log_artifact: path does not exist: %s", local_path)
            return
        mlflow.log_artifact(str(path_obj))
        logger.debug("Logged artifact: %s", local_path)
    except Exception as e:
        logger.error("Failed to log artifact: %s", e)


def log_figure(fig, artifact_file: str) -> None:
    """Log a matplotlib figure as an artifact.

    Args:
        fig: A matplotlib Figure object.
        artifact_file: Filename to save the figure as (e.g. "plot.png").

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> log_figure(fig, "line_plot.png")
    """
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    try:
        mlflow.log_figure(fig, artifact_file)
        logger.debug("Logged figure: %s", artifact_file)
    except Exception as e:
        logger.error("Failed to log figure: %s", e)


def end_run() -> None:
    """Explicitly end the active MLflow run.

    This is optional when using :func:`start_experiment` as a context manager,
    but can be called manually if needed.
    """
    if not _MLFLOW_AVAILABLE:
        return
    if mlflow.active_run():
        mlflow.end_run()
        logger.info("Ended MLflow run.")


# ---------------------------------------------------------------------------
# Query utilities
# ---------------------------------------------------------------------------


def list_experiments() -> list[dict]:
    """List all MLflow experiments in the tracking server.

    Returns:
        List of dicts with keys ``experiment_id``, ``name``, ``lifecycle_stage``.
    """
    if not _MLFLOW_AVAILABLE:
        return []
    client = MlflowClient()
    experiments = client.search_experiments()
    return [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "lifecycle_stage": exp.lifecycle_stage,
        }
        for exp in experiments
    ]


def get_run_metrics(run_id: str) -> dict:
    """Retrieve all metrics for a specific run.

    Args:
        run_id: The MLflow run ID.

    Returns:
        Dict mapping metric names to their latest values.
    """
    if not _MLFLOW_AVAILABLE:
        return {}
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics


def get_run_params(run_id: str) -> dict:
    """Retrieve all parameters for a specific run.

    Args:
        run_id: The MLflow run ID.

    Returns:
        Dict mapping parameter names to their values (strings).
    """
    if not _MLFLOW_AVAILABLE:
        return {}
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.params
