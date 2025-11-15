import datetime
import hashlib
import json
import math
import os
import uuid
from logging import getLogger
from typing import Any, List

import pandas as pd
from pydantic import BaseModel

from common.metric_config import METRIC_TO_USE_CASE_MAP
from evaluation.config_model import FileNamingConfig, ModelMetadata
from persist.schema import MetricDataRecord, MetricScore, PartitionInfo
from persist.utils import (
    compress_string,
    create_directory,
    create_partition_path,
    get_string_hash,
    get_use_case_for_metric,
    safe_to_csv,
    safe_to_parquet,
)
from scoring.aggregate_scoring import DatapointScore, MetricType

logger = getLogger(__name__)


def generate_filename(config: FileNamingConfig, provider: str, model: str) -> str:
    """
    Generate a hash-based filename for parquet files.

    Args:
        config: FileNamingConfig object with file naming settings
        provider: Provider/organization name
        model: Model name or identifier

    Returns:
        str: Generated filename in format "{type_of_run}-{hash}.parquet"
    """
    # Create hash from all components
    hash_input = (
        f"{config.run_identifier}"
        f"-{provider}-{model}"
        f"-{config.year}-{config.month}-{config.day}"
    )
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()

    return f"{config.type_of_run}-{hash_value}.parquet"


def create_metric_record(
    datapoint: DatapointScore,
    base_record: MetricDataRecord,
    partition_info: PartitionInfo,
    metric_type: MetricType,
    score: Any,
) -> MetricScore:
    """
    Create a record for a single metric score with partition information.
    """

    def all_attributes_are_nan(object: BaseModel) -> bool:
        return all(
            isinstance(v, (int, float)) and math.isnan(v)
            for v in object.model_dump().values()
            if isinstance(v, (int, float))
        )

    def inspect_datapoint_name(datapoint_name: str) -> str:
        """Inspect the datapoint name and return a string."""
        if datapoint_name[0] is not None and datapoint_name[1] is None:
            return "pred_exists_but_gt_missing"
        elif datapoint_name[0] is None and datapoint_name[1] is not None:
            return "gt_exists_but_pred_missing"
        else:
            return ""

    # Clone the base record to avoid modifying the original
    record = base_record.model_copy()

    # Set metric name
    metric_name = metric_type.value
    record.metric_name = metric_name

    # Generate a deterministic ID by hashing relevant fields
    id_components = f"{datapoint.name}_{datapoint.dataset}_{metric_name}"
    record.uuid = f"{uuid.uuid4().hex[:8]}_{get_string_hash(id_components)[:8]}"

    # Handle different score types
    if isinstance(score, (float, int)):
        # Store float directly as metric_value
        record.metric_value = score
        record.metric_json = None
        record.metric_nan = None
    else:
        if all_attributes_are_nan(score.scores):
            # For complex objects, store null in metric_value and JSON in metric_json
            record.metric_value = None
            record.metric_json = None
            record.metric_nan = inspect_datapoint_name(datapoint.name)
        else:
            record.metric_value = None
            record.metric_json = json.dumps(score.scores.model_dump())
            record.metric_nan = None

    # Create partition information using the custom use case
    use_case = get_use_case_for_metric(metric_name, datapoint.version, METRIC_TO_USE_CASE_MAP)
    partition_info.use_case = use_case

    metric_score = MetricScore(data=record, partition=partition_info)
    # Return both the data record and partition info
    return metric_score


def datapoint_to_records(datapoint: DatapointScore, config: ModelMetadata) -> List[MetricScore]:
    """
    Convert a DatapointScore object to a list of records with partition information.

    Args:
        datapoint: The DatapointScore object containing metric scores
        config: The ModelMetadata object containing the model configuration

    Returns:
        List of dictionaries representing records with partition information
    """
    # Get current timestamp
    now = datetime.datetime.now()

    # Create base record with data columns

    source_filename = datapoint.name[0] if datapoint.name[0] else datapoint.name[1]
    oi = config.output_identifier
    output_hash = get_string_hash(oi) if oi else None
    pred_filepath = datapoint.sample_file.name if datapoint.sample_file else None
    compressed_pred = compress_string(pred_filepath) if pred_filepath else None
    gt_filepath = datapoint.ground_truth_file.name if datapoint.ground_truth_file else None
    compressed_gt = compress_string(gt_filepath) if gt_filepath else None

    # Create base record as Pydantic model
    base_record = MetricDataRecord(
        uuid=str(uuid.uuid4()),
        dataset=datapoint.dataset,
        metric_name="",  # Will be set in create_metric_record
        context_version=config.context_version,
        context_sub_sub_label=config.context_sub_sub_label,
        output_identifier=output_hash,
        source_filename=source_filename,
        prediction_filename=compressed_pred,
        ground_truth_filename=compressed_gt,
        created_at=now,
        hour=now.strftime("%H"),
    )

    # Create partition info as Pydantic model
    partition_info = PartitionInfo(
        context_type=config.context_type,
        use_case="",  # Will be set in create_metric_record
        context_label=config.context_label,
        context_sub_label=config.context_sub_label,
        year=now.strftime("%Y"),
        month=now.strftime("%m"),
        day=now.strftime("%d"),
    )

    # Generate records for each metric
    return [
        create_metric_record(
            datapoint, base_record.model_copy(), partition_info.model_copy(), metric_type, score
        )
        for metric_type, score in datapoint.scores.items()
    ]


# Initialize S3 client once (lazily)
_s3_client = None


def get_s3_client():
    """Get or initialize the S3 client."""
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3")
    return _s3_client


def write_metric_to_s3(
    df: pd.DataFrame,
    partition_path: str,
    base_path: str,
    naming_config: FileNamingConfig,
    provider: str,
    model: str,
) -> int:
    """
    Write records to S3.
    """
    # S3 storage
    import io

    s3_client = get_s3_client()

    # Generate file name and path
    file_name = generate_filename(naming_config, provider, model)
    # Hard coded, this path is not likely to change, but the bucket name can change
    s3_key = f"metadata/metric_score/{partition_path}{file_name}"

    # Write DataFrame to a buffer
    buffer = io.BytesIO()
    df.to_parquet(buffer, compression="snappy", index=False)
    buffer.seek(0)

    # Upload to S3
    bucket = base_path.replace("s3://", "")
    s3_client.upload_fileobj(buffer, bucket, s3_key)
    logger.info(f"Wrote {len(df)} records to s3://{bucket}/{s3_key}")
    return len(df)


def write_metric_to_local(
    df: pd.DataFrame,
    partition_path: str,
    base_path: str,
    naming_config: FileNamingConfig,
    provider: str,
    model: str,
) -> int:
    """
    Write records to local storage.
    """
    # Local storage
    output_file = os.path.join(
        base_path, partition_path, generate_filename(naming_config, provider, model)
    )

    output_dir = os.path.dirname(output_file)
    create_directory(output_dir)

    # Write files
    # Write to csv for debugging
    safe_to_csv(df, output_file.replace(".parquet", ".csv"), index=False)
    safe_to_parquet(df, output_file, compression="snappy", index=False)
    logger.info(f"Wrote {len(df)} records to {output_file}")

    return len(df)


def write_records_to_storage(
    records: List[MetricScore],
    base_path: str,
    naming_config: FileNamingConfig,
    provider: str,
    model: str,
) -> int:
    """
    Write records to storage (local filesystem or S3).

    Args:
        records: List of dictionaries with 'data' and 'partition' keys
        base_path: Base path where files will be written
        naming_config: The FileNamingConfig object containing the file naming configuration
        provider: The provider name
        model: The model name
    Returns:
        Number of records written
    """
    record_count = 0

    # Group by all partition values
    partition_groups = {}
    for record in records:
        data = record.data.model_dump()
        partition = record.partition.model_dump()

        # Create partition path
        partition_path = create_partition_path(partition)

        # Initialize group if needed
        if partition_path not in partition_groups:
            partition_groups[partition_path] = []

        # Add data record to group
        partition_groups[partition_path].append(data)

    # Write each partition group
    for partition_path, data_records in partition_groups.items():
        # Create DataFrame for this partition
        group_df = pd.DataFrame(data_records)
        write_to_s3 = base_path.startswith("s3://")
        storage_writer = write_metric_to_s3 if write_to_s3 else write_metric_to_local
        # Write to storage
        record_count += storage_writer(
            group_df, partition_path, base_path, naming_config, provider, model
        )

    return record_count


def write_metrics_to_parquet(
    datapoints: List[DatapointScore],
    config: ModelMetadata,
    base_path: str,
    naming_config: FileNamingConfig,
    provider: str,
    model: str,
) -> int:
    """
    Writes metrics to local parquet files or s3 bucket.

    Args:
        datapoints: List of DatapointScore objects
        config: The ModelMetadata object containing the model configuration
        base_path: The base path or s3 bucket where the parquet files will be written
        naming_config: The FileNamingConfig object containing the file naming configuration
        provider: The provider name
        model: The model name

    Returns:
        Number of records written
    """
    all_records = []
    for datapoint in datapoints:
        records = datapoint_to_records(datapoint=datapoint, config=config)
        all_records.extend(records)

    return write_records_to_storage(all_records, base_path, naming_config, provider, model)
