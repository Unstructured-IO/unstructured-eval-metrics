import re
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from common.metric_config import (
    CONFIG_GROUND_TRUTH_FORMATS,
    CONFIG_SAMPLE_FORMATS,
)
from common.metric_types import MetricType


class DatasetConfig(BaseSettings):
    name: str  # dataset in the Glue table
    metrics: List[MetricType]
    to_evaluate_dir: Path
    ground_truth_dir: Path
    # These can override the global formats
    sample_formats: Optional[Dict[str, str]] = None
    ground_truth_formats: Optional[Dict[str, str]] = None
    exclude_elements: Optional[List[str]] = None

    @field_validator("to_evaluate_dir", "ground_truth_dir")
    def validate_directories(cls, v):
        """Validate that directories exist."""
        if not v.exists():
            raise ValueError(f"Directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return v


class ModelMetadata(BaseSettings):
    """
    Configuration for the document parsing system or model
    used to generate the prediction results.
    """

    context_type: str
    context_label: str = Field(
        default="default", description="Label for the context, e.g., model name"
    )
    context_sub_label: str = Field(
        default="default", description="Sub label for the context, e.g., environment"
    )
    context_version: Optional[str] = Field(
        default="default",
        description="Version identifier for the evaluated system or model if applicable",
    )
    context_sub_sub_label: Optional[str] = Field(
        default="default",
        description="Tertiary label for additional categorization, e.g., provider name",
    )
    output_identifier: Optional[str] = Field(
        default="default", description="Identifier linking to the raw output data path"
    )

    @model_validator(mode="after")
    def update_empty_string(self) -> "ModelMetadata":
        # the following attributes can not be empty string or it will break the database crawler
        self.context_type = self.context_type.strip() or "default"
        self.context_label = self.context_label.strip() or "default"
        self.context_sub_label = self.context_sub_label.strip() or "default"
        return self


class StorageConfig(BaseSettings):
    """Configuration for metric storage."""

    store_metrics: bool = Field(
        default=True, description="Whether to store metrics in local directory or s3"
    )
    base_path: str = Field(
        default="s3://utic-prod-dataset-registry",
        description="Base path for storing metrics. Path should be s3 bucket or local directory",
    )
    output_csv: Path = Field(
        default_factory=lambda: Path("report/evaluation_report_scores.csv"),
        description="Output CSV path",
    )

    @field_validator("base_path")
    def validate_base_path(cls, v):
        if isinstance(v, str):
            # S3 path validation
            if v.startswith("s3://"):
                s3_pattern = r"^s3://([^/]+)(/.*)?$"
                if not re.match(s3_pattern, v):
                    raise ValueError(f"Invalid S3 path format: {v}")
                return v
            # Local path validation
            else:
                # Convert to Path to check if it's a valid path
                path = Path(v)
                if not path.exists():
                    raise ValueError(f"Directory does not exist: {v}")
                return v
        raise ValueError(f"Expected string, got {type(v)}")


class FileNamingConfig(BaseSettings):
    """Configuration for parquet file naming."""

    type_of_run: str = Field(
        default="eval", description="Type of evaluation run (e.g., 'eval', 'test', 'benchmark')"
    )
    run_identifier: str = Field(
        default="",
        description="Run identifier for the evaluation run, e.g., data registry data tag",
    )
    year: Optional[str] = Field(
        default=None, description="Year for file naming (defaults to current year if not provided)"
    )
    month: Optional[str] = Field(
        default=None,
        description="Month for file naming (defaults to current month if not provided)",
    )
    day: Optional[str] = Field(
        default=None, description="Day for file naming (defaults to current day if not provided)"
    )


class FormatConfig(BaseSettings):
    """Configuration for data formats."""

    sample_formats: Dict[str, str] = Field(default_factory=CONFIG_SAMPLE_FORMATS)
    ground_truth_formats: Dict[str, str] = Field(default_factory=CONFIG_GROUND_TRUTH_FORMATS)


class BaseEvaluationConfig(BaseSettings):
    """Base configuration shared by all evaluation types."""

    model_metadata: Optional[ModelMetadata] = None
    storage: StorageConfig = Field(default_factory=StorageConfig)
    file_naming: Optional[FileNamingConfig] = None
    formats: FormatConfig = Field(default_factory=FormatConfig)
    datasets: List[DatasetConfig] = Field(default_factory=list)

    @field_validator("datasets")
    def validate_datasets(cls, v):
        """Ensure there is at least one dataset configured."""
        if not v:
            raise ValueError("At least one dataset must be specified")
        return v

    @property
    def sample_formats(self) -> Dict[str, str]:
        return self.formats.sample_formats

    @property
    def ground_truth_formats(self) -> Dict[str, str]:
        return self.formats.ground_truth_formats

    def get_merged_formats_for_dataset(
        self, dataset_name: str
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Get merged sample and ground truth formats for a specific dataset.

        This method copies the global formats and overrides any keys defined
        in the dataset-specific formats.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Tuple containing:
            - Merged sample formats dictionary
            - Merged ground truth formats dictionary
        """
        # Start with copies of the global formats
        merged_sample_formats = self.sample_formats.copy()
        merged_ground_truth_formats = self.ground_truth_formats.copy()

        # Find the specified dataset
        for dataset in self.datasets:
            if dataset.name == dataset_name:
                # Override with dataset-specific formats if provided
                if hasattr(dataset, "sample_formats") and dataset.sample_formats:
                    merged_sample_formats.update(dataset.sample_formats)

                if hasattr(dataset, "ground_truth_formats") and dataset.ground_truth_formats:
                    merged_ground_truth_formats.update(dataset.ground_truth_formats)

                break
        return merged_sample_formats, merged_ground_truth_formats


class AggregateEvaluationConfig(BaseEvaluationConfig):
    """Configuration for the aggregate evaluation report."""

    class ConfigDict:
        """Pydantic configuration."""

        extra = "forbid"  # Prevent extra fields


class DocumentEvaluationConfig(BaseEvaluationConfig):
    """Configuration for the document level evaluation report."""

    merge_table_counts: bool = Field(
        True, description="Whether to merge table counts into the main DataFrame"
    )
