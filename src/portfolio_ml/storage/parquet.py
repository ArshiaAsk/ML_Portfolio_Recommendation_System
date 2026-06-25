"""Parquet read/write utilities for the data lake."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_ml.utils.logging import get_logger

logger = get_logger(__name__)


def write_partitioned_parquet(
    df: pd.DataFrame,
    base_path: str,
    partition_cols: list[str],
    filename: str,
    extra_partition_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> None:
    """Write a DataFrame to partitioned Parquet files under *base_path*.

    Each combination of partition column values produces one file::

        {base_path}/col1={val1}/col2={val2}/.../filename

    Existing files are overwritten (idempotent). Partition columns are
    **not** stored inside the Parquet file to avoid column duplication.

    Args:
        df: DataFrame to persist.
        base_path: Root output directory.
        partition_cols: Columns to partition by (in order).
        filename: File name for each partition file.
        extra_partition_fn: Optional callable that adds extra partition
            columns to the DataFrame before partitioning. It receives
            the full DataFrame and returns a modified copy.
    """
    if df.empty:
        logger.warning("write_partitioned_parquet called with empty DataFrame; skipping.")
        return

    working_df = df.copy()
    if extra_partition_fn is not None:
        working_df = extra_partition_fn(working_df)

    missing = [c for c in partition_cols if c not in working_df.columns]
    if missing:
        raise ValueError(
            f"Partition columns missing from DataFrame: {missing}. "
            f"Available columns: {list(working_df.columns)}"
        )

    groups = working_df.groupby(partition_cols, sort=False)
    files_written = 0

    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        # Build directory path from partition column values
        parts = "/".join(
            f"{col}={val}" for col, val in zip(partition_cols, key_tuple)
        )
        out_dir = Path(base_path) / parts
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        # Drop partition columns from the stored file
        store_df = group.drop(columns=partition_cols, errors="ignore").reset_index(drop=True)

        table = pa.Table.from_pandas(store_df, preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy")
        files_written += 1

    logger.info(
        "Wrote %d partition file(s) to '%s'", files_written, base_path
    )


def read_parquet_dataset(path: str) -> pd.DataFrame:
    """Read all Parquet files under *path* recursively into one DataFrame.

    Args:
        path: Root directory (or file) to read from.

    Returns:
        Combined DataFrame. Empty DataFrame if no files found.
    """
    root = Path(path)
    if not root.exists():
        logger.warning("read_parquet_dataset: path does not exist: '%s'", path)
        return pd.DataFrame()

    if root.is_file():
        return pd.read_parquet(str(root))

    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("read_parquet_dataset: no Parquet files found under '%s'", path)
        return pd.DataFrame()

    frames = [pd.read_parquet(str(f)) for f in parquet_files]
    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Read %d rows from %d Parquet file(s) under '%s'",
        len(result),
        len(parquet_files),
        path,
    )
    return result


def write_single_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a single Parquet file.

    Creates parent directories as needed.

    Args:
        df: DataFrame to persist.
        path: Full file path (including filename).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(out_path), compression="snappy")
    logger.debug("Wrote %d rows to '%s'", len(df), out_path)
