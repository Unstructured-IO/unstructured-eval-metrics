import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from common.metric_types import MetricType
from scoring.element_consistency_scoring import (
    ElementConsistencyScore,
    calculate_f1_from_confusion_matrix,
)
from scoring.table_scoring import TablesScore

logging.basicConfig(level=logging.INFO)


class ScoreContext(BaseModel):
    """
    Attributes:
        stats (Dict[str, Dict[str, List[float]]]):
            A nested dictionary tracking raw score values in a list of DatapointScore objects.
            - First level key: Metric type
            - Second level key: Metric name
            - Value: List of float scores collected for that metric

        counts (Dict[str, Dict[str, int]]):
            A nested dictionary tracking count statistics in a list of DatapointScore objects.
            - First level key: Metric type
            - Second level key: Count type ('total', 'valid', 'zeros', 'skipped')
            - Value: Integer count for the specified statistic
    """

    stats: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    counts: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: defaultdict(
            lambda: {"total": 0, "valid": 0, "zeros": 0, "skipped": 0}
        )
    )
    matrix: Dict[str, Dict[Tuple[str, str], int]] = Field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )


@dataclass
class ScoreFilterConfig:
    """
    Configuration for filtering scores in aggregate statistics calculations.
    """

    ignore_NA: bool = True
    ignore_zeros: bool = False
    treat_low_value_as_NA: bool = False
    treat_NA_as_zero: bool = False
    treat_zero_as_NA: bool = False
    low_value_as_NA_threshold: float = 0.00001


class ScoreValidator:
    def __init__(self, config: ScoreFilterConfig):
        self.config = config

    def update_score_based_on_config(self, score) -> float:
        """Update a score based on the filter configuration."""
        if not isinstance(score, (int, float)):
            return score
        if math.isnan(score) and self.config.treat_NA_as_zero:
            return 0.0
        if score == 0.0 and self.config.treat_zero_as_NA:
            return float("nan")
        elif score <= self.config.low_value_as_NA_threshold and self.config.treat_low_value_as_NA:
            return float("nan")

        return score

    def is_valid_value(self, score) -> bool:
        """Check if a value is valid based on the filter configuration."""
        if score is None or not isinstance(score, (int, float)):
            return False
        if math.isnan(score):
            return self.config.treat_NA_as_zero and not self.config.ignore_NA
        if score == 0.0:
            return not (self.config.treat_zero_as_NA or self.config.ignore_zeros)
        elif score <= self.config.low_value_as_NA_threshold:
            return not self.config.treat_low_value_as_NA
        return True


@dataclass
class DatapointScore:
    """
    Scores for a single datapoint.

    Args:
        name (str): Name of the datapoint file or a description of the datapoint.
        sample_formats (Dict[MetricType, str]): Format for sample file input for a specific metric.
        ground_truth_formats (Dict[MetricType, str]): Format for ground truth file input
            for a specific metric. Same as sample_format.
        scores (Dict[MetricType, Union[float, TablesScore, ElementConsistencyScore]]):
            Scores for the datapoint.
        version (str): Version of the score to use.
        dataset (str): Name of the dataset the datapoint belongs to.
        sample_file (Path): Optional path to the sample file for the datapoint.
        ground_truth_file (Path): Optional path to the ground truth file for the datapoint.
    Note:
        - For CCT, the format is "v1" or "plain_text_v1" or "unstructured_json_v1".
        - For percent_tokens_found, the format is "v1" or "plain_text_v1" or "unstructured_json_v1".
        - For tables, the format is "html" or "cells" or "text"
    """

    name: str
    sample_formats: Dict[MetricType, str]
    ground_truth_formats: Dict[MetricType, str]
    scores: Dict[MetricType, Union[float, TablesScore, ElementConsistencyScore]] = field(
        default_factory=dict
    )
    version: str = "v1"
    dataset: str = "industrial-eval-dataset"
    sample_file: Optional[Path] = None
    ground_truth_file: Optional[Path] = None

    def __getitem__(self, metric_type: MetricType):
        """Get a score for a specific metric."""
        return self.scores[metric_type]

    def __setitem__(
        self,
        metric_type: MetricType,
        score: Union[float, TablesScore, ElementConsistencyScore],
    ):
        """Add a score for a specific metric."""
        self.scores[metric_type] = score

    def get_scores(
        self,
    ) -> Dict[MetricType, Union[float, TablesScore, ElementConsistencyScore]]:
        """Get all scores for the datapoint."""
        return self.scores


def _process_numeric_metric(
    metric_type: MetricType,
    score: float,
    dp_score: DatapointScore,
    score_context: ScoreContext,
    score_validator: ScoreValidator,
):
    """
    Process numeric metrics like CCT, PERCENT_TOKENS_FOUND (ADDED), ELEMENT_ALIGNMENT
    """
    score_context.counts[metric_type]["total"] += 1
    score = score_validator.update_score_based_on_config(score)
    valid = score_validator.is_valid_value(score)

    if valid:
        score_context.counts[metric_type]["valid"] += 1
        score_context.stats[metric_type][metric_type.value].append(score)
        if score == 0.0:
            score_context.counts[metric_type]["zeros"] += 1
    elif score == 0:
        score_context.counts[metric_type]["zeros"] += 1
    else:
        score_context.counts[metric_type]["skipped"] += 1


def _process_basemodel_metric(
    metric_type: MetricType,
    score: BaseModel,
    dp_score: DatapointScore,
    score_context: ScoreContext,
    score_validator: ScoreValidator,
    scores_to_ignore: Optional[List[str]] = ["detection_f_beta"],
):
    """
    Process BaseModel metrics (TablesScore, ElementConsistencyScore, etc.)
    """
    score_context.counts[metric_type]["total"] += 1

    if hasattr(score, "get_scores_version"):
        version_score = score.get_scores_version(dp_score.version)

        has_valid_metrics = False
        has_zero_metrics = True

        for key, value in version_score.model_dump().items():
            if key in scores_to_ignore:
                continue  # Skip aggregation for this parameter
            value = score_validator.update_score_based_on_config(value)
            valid = score_validator.is_valid_value(value)

            if valid:
                has_valid_metrics = True
                score_context.stats[metric_type][key].append(value)
            elif value == 0:
                has_zero_metrics = True

        if has_valid_metrics:
            score_context.counts[metric_type]["valid"] += 1
            if has_zero_metrics:
                score_context.counts[metric_type]["zeros"] += 1
        elif has_zero_metrics:
            score_context.counts[metric_type]["zeros"] += 1
        else:
            score_context.counts[metric_type]["skipped"] += 1
    else:
        logging.warning(f"No scores found for {dp_score.name} {metric_type.name}")


def is_float_nan(x):
    return isinstance(x, float) and math.isnan(x)


def _process_matrix_metric(
    metric_type: MetricType,
    score: BaseModel,
    dp_score: DatapointScore,
    score_context: ScoreContext,
    score_validator: ScoreValidator,
    scores_to_ignore: Optional[List[str]] = ["detection_f_beta"],
):
    if is_float_nan(score):
        return

    for key, count in score.matrix.items():
        score_context.matrix[metric_type][key] = (
            score_context.matrix[metric_type].get(key, 0) + count
        )


def _calculate_aggregate_statistics(
    score_context: ScoreContext,
) -> Dict[MetricType, Dict[str, float]]:
    """
    Calculate final statistics from collected values.

    Args:
        stats: Dictionary with collected valid score values
        counts: Dictionary with counts of different value types

    Returns:
        Dict[MetricType, Dict[str, float]]: Aggregate statistics
    """
    from statistics import mean, stdev

    aggregate_stats = {}

    for metric_type, matrix in score_context.matrix.items():
        aggregate_stats[metric_type] = {}
        aggregate_stats[metric_type]["matrix"] = matrix
        aggregate_stats[metric_type]["detection_f1"] = calculate_f1_from_confusion_matrix(
            aggregate_stats[metric_type]["matrix"]
        )

    for metric_type, metrics in score_context.stats.items():
        aggregate_stats[metric_type] = {}
        count_info = score_context.counts[metric_type]

        # Add count information
        aggregate_stats[metric_type]["valid"] = count_info["valid"]
        aggregate_stats[metric_type]["total"] = count_info["total"]
        aggregate_stats[metric_type]["zeros"] = count_info["zeros"]
        aggregate_stats[metric_type]["skipped"] = count_info["skipped"]

        # Calculate statistics for each metric
        for metric_name, values in metrics.items():
            if not values:
                continue

            # Calculate basic statistics
            aggregate_stats[metric_type][f"{metric_name}_mean"] = round(mean(values), 3)
            aggregate_stats[metric_type][f"{metric_name}_min"] = round(min(values), 3)
            aggregate_stats[metric_type][f"{metric_name}_max"] = round(max(values), 3)

            # Add standard deviation for multiple values
            if len(values) > 1:
                try:
                    aggregate_stats[metric_type][f"{metric_name}_std"] = round(stdev(values), 3)
                except Exception as e:
                    logging.warning(f"Error calculating std for {metric_name}: {e}")
                    aggregate_stats[metric_type][f"{metric_name}_std"] = 0.0

    return aggregate_stats
