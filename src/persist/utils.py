import gzip
import hashlib
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .schema import PARTITION_COLUMNS


def compress_string(text: str) -> str:
    """
    Compress a string using gzip.

    Args:
        text: The string to compress

    Returns:
        A compressed string
    """
    return gzip.compress(text.encode("utf-8"))


def get_string_hash(text: str) -> str:
    """
    Generate a SHA-256 hash of the input string.

    Args:
        text: The string to hash

    Returns:
        A string containing the hexadecimal representation of the hash
    """
    hash_obj = hashlib.sha256(text.encode("utf-8"))
    return hash_obj.hexdigest()


def create_directory(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def get_use_case_for_metric(
    metric_name: str, version: str, metric_to_use_case_map: Dict[str, str]
) -> str:
    """
    Determine the use case for a given metric name based on the provided mapping.

    Args:
        metric_name: The name of the metric
        version: The version of the metric
        metric_to_use_case_map: Mapping of metric names to use cases

    Returns:
        The use case string

    Raises:
        ValueError: If the metric name is not found in the mapping
    """
    if metric_name in metric_to_use_case_map:
        return metric_to_use_case_map[metric_name].format(version=version)
    else:
        raise ValueError(f"No use case found for metric: {metric_name}")


def create_partition_path(partition_values: Dict[str, Any]) -> str:
    """
    Create a partition path string from partition values according to the schema order.

    Args:
        partition_values: Dictionary of partition column names to values

    Returns:
        String representing the partition path

    Raises:
        ValueError: If a required partition column is missing
    """
    # Build the path using the order defined in PARTITION_COLUMNS

    partition_path = ""
    for col in PARTITION_COLUMNS:
        if col in partition_values:
            value = partition_values[col].strip() or "default"
            partition_path += f"{col}={value}/"
        else:
            raise ValueError(f"Partition column '{col}' not found in partition values")

    return partition_path


def safe_to_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df: The pandas DataFrame to save
        filepath: The path to the CSV file
        kwargs: Additional arguments to pass to the pandas to_csv method
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    
    
def safe_to_parquet(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save a pandas DataFrame to a Parquet file.

    Args:
        df: The pandas DataFrame to save
        filepath: The path to the Parquet file
        kwargs: Additional arguments to pass to the pandas to_parquet method
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)