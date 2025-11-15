import concurrent.futures
import inspect
import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from common.metric_config import SCORE_FUNCS, STATS_PROCESSORS, VALID_FORMATS
from common.metric_types import MetricType
from processing.aggregate import (
    DatapointScore,
    ScoreContext,
    ScoreFilterConfig,
    ScoreValidator,
    _calculate_aggregate_statistics,
    _process_basemodel_metric,
)

logging.basicConfig(level=logging.INFO)


@dataclass
class AggregateScore:
    """
    Aggregate scores across multiple datapoints.

    Args:
        datapoint_scores (List[DatapointScore]): List of datapoint scores.
        aggregate_stats (Dict[MetricType, Dict[str, float]]): Aggregate statistics.
    Note:
        The key for the value dict in aggregate_stats can be:
        - "cct": cct score
        - "percent_tokens_found": percent tokens found score
        - "percent_tokens_added": percent tokens added score
        - "element_alignment": element alignment score
        - "overall": overall score
        - "detection_f1": detection f1 score
        - "cell_level_content_acc": cell level content accuracy
        - "cell_level_index_acc": cell level index accuracy
    """

    datapoint_scores: List[DatapointScore] = field(default_factory=list)
    aggregate_stats: Dict[MetricType, Dict[str, float]] = field(default_factory=dict)

    def add_datapoint_score(self, datapoint_score: DatapointScore):
        """Add a datapoint score to the collection and track format counts."""
        self.datapoint_scores.append(datapoint_score)

    def calculate_aggregate_stats(
        self, score_validator: ScoreValidator
    ) -> Dict[MetricType, Dict[str, float]]:
        """
        Calculate aggregate statistics across all datapoints.

        This method calculates various statistics for each metric type across all datapoints:
        - For CCT, PERCENT_TOKENS_FOUND, PERCENT_TOKENS_ADDED, ELEMENT_ALIGNMENT:
            mean, min, max, and std of the scores
        - For TABLES: mean, min, max, and std for each table metric (overall, detection_f1, etc.)

        Special handling:
        - N/A values:
            N/A values can be no ground truth while there is a prediction.
            N/A values are not included in statistics by default.
        - Zero values:
            Zero values can also be false prediction when there is no tables in ground truth.
            Zero values are included in statistics by default.
        - Counts of zeros, valid, skipped, and total values are tracked

        Returns:
            Dict[MetricType, Dict[str, float]]: Aggregate statistics
        """

        score_context = ScoreContext()

        logging.info(f"Calculating aggregate stats for {len(self.datapoint_scores)} datapoints")

        # Process all datapoints and collect scores
        for dp_score in self.datapoint_scores:
            for metric_type, score in dp_score.scores.items():
                if metric_type in STATS_PROCESSORS:
                    STATS_PROCESSORS[metric_type](
                        metric_type, score, dp_score, score_context, score_validator
                    )

        # Calculate final statistics from collected values
        self.aggregate_stats = _calculate_aggregate_statistics(score_context)

        return self.aggregate_stats

    def __getitem__(self, metric_type: MetricType) -> Dict[str, float]:
        """Get aggregate statistics for a specific metric type."""
        return self.aggregate_stats[metric_type]

    def get_datapoint_scores(self) -> List[DatapointScore]:
        """Get all datapoint scores."""
        return self.datapoint_scores

    def get_aggregate_stats(self) -> Dict[MetricType, Dict[str, float]]:
        """Get aggregate statistics."""
        return self.aggregate_stats


def _validate_input_format(metric_type: MetricType, input_format: str, version: str) -> str:
    """
    Check if the format of the score is valid based on score type and version.

    Args:
        metric_type: The type of score being calculated (CCT, PERCENT_TOKENS_FOUND, TABLES)
        input_format: The format string to validate
        version: The version string that determines valid formats

    Returns:
        str: The validated input format

    Raises:
        ValueError: If the input format is not valid for the given score type and version
    """

    valid_formats = VALID_FORMATS

    if version not in valid_formats:
        raise ValueError(
            f"Unsupported version: {version}. \
            Supported versions are: {list(valid_formats.keys())}"
        )

    if metric_type not in valid_formats[version]:
        raise ValueError(f"Score type {metric_type} not supported in version {version}")

    if input_format not in valid_formats[version][metric_type]:
        valid_formats_str = ", ".join(valid_formats[version][metric_type])
        raise ValueError(
            f"Invalid format '{input_format}' for {metric_type.name} in version {version}. "
            f"Valid formats are: {valid_formats_str}"
        )
    return input_format


def _process_datapoint(
    args, metric_types_enums, sample_formats, ground_truth_formats, exclude_elements, version
):
    """
    Process a single datapoint and return a DatapointScore object.

    Args:
        args: A tuple containing the sample file, ground truth file, and name of the datapoint
        metric_types_enums: A list of MetricType objects
        sample_formats: A dictionary of sample formats
        ground_truth_formats: A dictionary of ground truth formats
        exclude_elements: The elements to exclude from the evaluation
        version: The version of the score to use

    Returns:
        A DatapointScore object with scores for each metric type
    """

    def _get_metric_score(
        name,
        metric_type,
        sample_file,
        ground_truth_file,
        sample_format,
        ground_truth_format,
        exclude_elements,
        version,
    ):
        """
        Get the score for a specific metric type.
        """
        if None in name:
            # Check if this metric type uses BaseModel processing by looking at stats processor
            if STATS_PROCESSORS.get(metric_type) == _process_basemodel_metric:
                # Get a sample score to check if it has get_scores_with_values method
                score_func = SCORE_FUNCS[metric_type]
                sig = inspect.signature(score_func)
                return_annotation = sig.return_annotation
                if hasattr(return_annotation, "get_scores_with_values"):
                    return return_annotation(version=version).get_scores_with_values(float("nan"))
            return float("nan")

        # Calculate actual score
        _validate_input_format(metric_type, sample_format, version)
        _validate_input_format(metric_type, ground_truth_format, version)

        score_func = SCORE_FUNCS[metric_type]
        sig = inspect.signature(score_func)
        args = {
            "sample_filename": sample_file,
            "ground_truth_filename": ground_truth_file,
            "sample_format": sample_format,
            "ground_truth_format": ground_truth_format,
        }
        if "exclude_elements" in sig.parameters:
            args["exclude_elements"] = exclude_elements

        return score_func(**args)

    sample_file, ground_truth_file, name = args

    # Create datapoint score with format information
    datapoint_score = DatapointScore(
        name=name,
        sample_formats=sample_formats,
        ground_truth_formats=ground_truth_formats,
        version=version,
        sample_file=sample_file,
        ground_truth_file=ground_truth_file,
    )

    # Calculate scores for each metric type
    for metric_type in metric_types_enums:
        datapoint_score[metric_type] = _get_metric_score(
            name,
            metric_type,
            sample_file,
            ground_truth_file,
            sample_formats[metric_type],
            ground_truth_formats[metric_type],
            exclude_elements,
            version,
        )

    return datapoint_score


def aggregate_scores(
    metric_types: List[str],
    to_evaluate: List[Path],
    ground_truth: List[Path],
    sample_formats: Dict[str, str],
    ground_truth_formats: Dict[str, str],
    datapoint_names: List[str],
    exclude_elements: Optional[List[str]] = None,
    version: Optional[str] = "v1",
    score_validator: Optional[ScoreValidator] = None,
    max_workers: Optional[int] = None,
) -> AggregateScore:
    """
    Calculate scores for multiple evaluation datapoints with mixed formats.

    Args:
        metric_types (List[str]): Types of scores to calculate
            (each element is a score type,
            e.g. ["cct", "tables", "percent_tokens_found"])
        to_evaluate (List[Path]):
            List of paths to formatted datapoints to evaluate (each element is str or json)
        ground_truth (List[Path]):
            List of paths to formatted ground truth datapoints (each element is str or json)
        sample_formats (Dict[str, str]): A dictionary to specify sample formats
            for each score type e.g. {"cct": "v1", "tables": "html", "percent_tokens_found": "v1"}
        ground_truth_formats (Dict[str, str]): A dictionary to specify
            ground truth formats for each score type
        datapoint_names (List[str]): List of display names for each datapoint
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
        version (Optional[str]): Version of the score to use
        score_validator (Optional[ScoreValidator]): Configuration for filtering scores
        max_workers (Optional[int]): Maximum number of worker processes to use
    Returns:
        An AggregateScore object containing individual datapoint scores and aggregate statistics

    Note:
        - to_evaluate and ground_truth supports str (plain text) or list of dicts
            (Unstructured Elements JSON)
    """
    # Initialize the aggregate score object
    aggregate_score = AggregateScore()

    # Convert string score types to enum
    metric_types_enums = [MetricType(s) for s in metric_types]

    # Create a list of inputs for each worker
    datapoint_inputs = list(zip(to_evaluate, ground_truth, datapoint_names))

    # Process datapoints in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with the fixed arguments
        worker_func = partial(
            _process_datapoint,
            metric_types_enums=metric_types_enums,
            sample_formats=sample_formats,
            ground_truth_formats=ground_truth_formats,
            exclude_elements=exclude_elements,
            version=version,
        )

        # Submit all tasks and collect results as they complete
        datapoint_scores = list(executor.map(worker_func, datapoint_inputs))

    for datapoint_score in datapoint_scores:
        aggregate_score.add_datapoint_score(datapoint_score)

    # Calculate aggregate statistics
    aggregate_score.calculate_aggregate_stats(score_validator)

    return aggregate_score


def get_aligned_paths(to_evaluate_dir, ground_truth_dir):
    to_evaluate_dir = Path(to_evaluate_dir)
    ground_truth_dir = Path(ground_truth_dir)

    # Populate list of datapoint paths
    to_evaluate_paths = [
        path.resolve()
        for path in to_evaluate_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    ]

    ground_truth_paths = [
        path.resolve()
        for path in ground_truth_dir.glob("*")
        if path.is_file() and not path.name.startswith(".")
    ]

    to_evaluate_dict = {path.stem: path for path in to_evaluate_paths}
    ground_truth_dict = {
        path.stem.split("__")[0] if "__" in path.stem else path.stem: path
        for path in ground_truth_paths
    }

    # Get all unique filenames from both directories
    all_names = set(to_evaluate_dict.keys()) | set(ground_truth_dict.keys())

    # Create aligned lists, adding None when a file doesn't exist in one of the directories
    aligned_to_evaluate = [to_evaluate_dict.get(name, None) for name in all_names]
    aligned_ground_truth = [ground_truth_dict.get(name, None) for name in all_names]

    return aligned_to_evaluate, aligned_ground_truth


def load_files_and_aggregate_scores(
    metric_types: List[str],
    to_evaluate_dir: str,
    ground_truth_dir: str,
    sample_formats: Dict[str, str],
    ground_truth_formats: Dict[str, str],
    exclude_elements: Optional[List[str]] = None,
    version: Optional[str] = "v1",
    score_validator: Optional[ScoreValidator] = ScoreValidator(ScoreFilterConfig()),
) -> AggregateScore:
    """
    Load files and aggregate scores from a directory of files.

    Args:
        metric_types (List[str]): Types of scores to calculate for all datapoints
            (each element is a score type, e.g. ["cct", "tables", "percent_tokens_found"])
        to_evaluate_dir (str): Directory containing the files to evaluate
        ground_truth_dir (str): Directory containing the ground truth files
        sample_format (Dict[str, str]): A dictionary to specify sample formats
            for each score type e.g. {"cct": "v1", "tables": "html", "percent_tokens_found": "v1"}
        ground_truth_format (Dict[str, str]): A dictionary to specify
            ground truth formats for each score type. Same as sample_format.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
        version (Optional[str]): Version of the score to use
        score_validator (Optional[ScoreValidator]): Configuration for filtering scores
    Returns:
        An AggregateScore object containing individual datapoint scores and aggregate statistics

    Note:
        A set of documents should have same sample format and ground truth format and version.
        TODO: The version may be different for each score type.
    """

    to_evaluate_paths, ground_truth_paths = get_aligned_paths(to_evaluate_dir, ground_truth_dir)

    datapoint_names = []

    # Populate list of datapoint names
    datapoint_names = [
        (
            path.stem if path is not None else None,
            ground_truth_path.stem if ground_truth_path is not None else None,
        )
        for path, ground_truth_path in zip(to_evaluate_paths, ground_truth_paths)
    ]

    return aggregate_scores(
        metric_types=metric_types,
        to_evaluate=to_evaluate_paths,
        ground_truth=ground_truth_paths,
        sample_formats=sample_formats,
        ground_truth_formats=ground_truth_formats,
        datapoint_names=datapoint_names,
        exclude_elements=exclude_elements,
        version=version,
        score_validator=score_validator,
    )
