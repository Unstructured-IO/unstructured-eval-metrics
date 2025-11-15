import argparse
import json
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel

from evaluation.config_model import DocumentEvaluationConfig
from persist.parquet import write_metrics_to_parquet
from persist.utils import safe_to_csv
from scoring.aggregate_scoring import (
    DatapointScore,
    MetricType,
    get_aligned_paths,
    load_files_and_aggregate_scores,
)


def datapoint_scores_to_scores_df(datapoints: List[DatapointScore]) -> pd.DataFrame:
    """
    Convert datapoint scores to a pandas DataFrame with dynamic column generation.

    This function dynamically creates columns based on the available metric types
    and their corresponding score attributes in the datapoints, expanding
    metric types with sub-metrics like TablesScore into separate columns for each attribute.

    Args:
        datapoints: List of datapoint objects containing scores

    Returns:
        pandas.DataFrame: DataFrame with one row per unique datapoint name and
                         columns for each metric and sub-metric
    """
    rows = {}

    # Identify all possible metrics and their attributes
    all_metrics = set()
    all_sub_metrics = set()

    for dp in datapoints:
        for metric_type, score in dp.scores.items():
            metric_name = metric_type.value.lower()
            # If this is a metric object with attributes (like TablesScore)
            # add all the sub-metrics to all_sub_metrics to be used as columns
            if hasattr(score, "scores") and hasattr(score.scores, "__dict__"):
                for attr_name in vars(score.scores).keys():
                    all_sub_metrics.add(attr_name)
            else:
                # add the metric name to all_metrics to be used as a column
                all_metrics.add(metric_name)

    # Create rows with all possible columns initialized to None
    for dp in datapoints:
        # skip datapoints without prediction or ground truth
        if not dp.name[0] or not dp.name[1]:
            continue
        name = dp.name[0]

        if "__" in name:
            # this is added to support the new data registry gt file naming format,
            # where the name is a combination of the filename, format and use case
            name = name.split("__")[0]

        # Create new row if name doesn't exist yet
        if name not in rows:
            row_data = {
                "name": name,
                **dict.fromkeys(all_metrics),
                **dict.fromkeys(all_sub_metrics),
            }
            rows[name] = row_data

        # Update the row with actual values from this datapoint
        for metric_type, score in dp.scores.items():
            metric_name = metric_type.value.lower()
            if isinstance(score, BaseModel) and hasattr(score, "scores"):
                table_scores = score.scores
                for sub_metric_name, sub_metric_value in vars(table_scores).items():
                    rows[name][sub_metric_name] = sub_metric_value
            else:
                # For simple metrics (just store the value)
                rows[name][metric_name] = score

    # Convert dictionary of rows to a list for DataFrame creation
    return pd.DataFrame(list(rows.values()))


def count_tables_for_dataset(
    to_evaluate_dir: str, ground_truth_dir: str
) -> Dict[str, Dict[str, float]]:
    """
    Count tables in prediction files and ground truth files for each document.

    This function compares the number of tables found in prediction files against
    ground truth files by processing JSON data in both directories. It aligns files
    from both directories and counts elements with "table" type in each file.

    Args:
        to_evaluate_dir: str
            Directory containing prediction JSON files to evaluate
        ground_truth_dir: str
            Directory containing ground truth JSON files for comparison

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping filenames to dictionaries
            containing:
            - 'predicted_table_count': Number of tables in prediction file
            - 'ground_truth_table_count': Number of tables in ground truth file

    """

    table_counts = defaultdict(
        lambda: {"predicted_table_count": float("nan"), "ground_truth_table_count": float("nan")}
    )

    aligned_to_evaluate_files, aligned_ground_truth_files = get_aligned_paths(
        to_evaluate_dir, ground_truth_dir
    )

    for aligned_to_evaluate, aligned_ground_truth in zip(
        aligned_to_evaluate_files, aligned_ground_truth_files
    ):

        # Use filename as key
        key = aligned_to_evaluate.stem if aligned_to_evaluate else aligned_ground_truth.stem

        # Count tables in prediction file if available
        if aligned_to_evaluate is not None:
            with open(aligned_to_evaluate) as f:
                data = json.load(f)
                predicted_table_count = sum(1 for d in data if d.get("type", "").lower() == "table")
            table_counts[key]["predicted_table_count"] = predicted_table_count

        # Count tables in ground truth file if available
        if aligned_ground_truth is not None:
            with open(aligned_ground_truth) as f:
                data = json.load(f)
                ground_truth_table_count = sum(
                    1 for d in data if d.get("type", "").lower() == "table"
                )
            table_counts[key]["ground_truth_table_count"] = ground_truth_table_count

    return dict(table_counts)


def process_all_datasets(
    config: DocumentEvaluationConfig,
) -> Tuple[List[DatapointScore], Dict[str, Dict[str, float]]]:
    """
    Process all datasets and return datapoint scores and table counts.
    """
    all_datapoints = []
    all_table_counts = {}
    # Process each dataset
    for dataset in config.datasets:
        sample_formats, ground_truth_formats = config.get_merged_formats_for_dataset(dataset.name)
        exclude_elements = dataset.exclude_elements

        logging.info(f"Processing dataset: {dataset.name}")
        datapoints = process_dataset(
            metric_types=dataset.metrics,
            to_evaluate_dir=str(dataset.to_evaluate_dir),
            ground_truth_dir=str(dataset.ground_truth_dir),
            exclude_elements=exclude_elements,
            sample_formats=sample_formats,
            ground_truth_formats=ground_truth_formats,
        )
        # update dataset metadata for each datapoint
        for datapoint in datapoints:
            datapoint.dataset = dataset.name

        # Only collect table counts if explicitly configured
        if "tables" in dataset.metrics and getattr(config, "merge_table_counts", False):
            table_counts = count_tables_for_dataset(
                str(dataset.to_evaluate_dir), str(dataset.ground_truth_dir)
            )
            all_table_counts.update(table_counts)

        all_datapoints.extend(datapoints)
    return all_datapoints, all_table_counts


def process_dataset(
    metric_types: List[MetricType],
    to_evaluate_dir: str,
    ground_truth_dir: str,
    exclude_elements: List[str],
    sample_formats: Dict[str, str],
    ground_truth_formats: Dict[str, str],
    version: str = "v1",
) -> List[DatapointScore]:
    """
    Process a single dataset and return datapoint scores and table counts.

    Args:
        metric_types: List[MetricType]
                The metrics to evaluate
        to_evaluate_dir: str
            The directory containing the files to evaluate
        ground_truth_dir: str
            The directory containing the ground truth files
        exclude_elements: List[str]
            The elements to exclude from the evaluation
        sample_formats: Dict[str, str]
            The sample formats for each metric
        ground_truth_formats: Dict[str, str]
            The ground truth formats for each metric
        version: str
            The version of the metrics

    Returns:
        List[DatapointScore]: A list of datapoint scores
    """
    aggregate_score = load_files_and_aggregate_scores(
        metric_types=metric_types,
        to_evaluate_dir=to_evaluate_dir,
        ground_truth_dir=ground_truth_dir,
        sample_formats=sample_formats,
        ground_truth_formats=ground_truth_formats,
        exclude_elements=exclude_elements,
        version=version,
    )
    return aggregate_score.get_datapoint_scores()


def get_config_from_args() -> DocumentEvaluationConfig:
    """
    Parse command line arguments and create a configuration.

    Returns:
        EvaluationConfig: The configuration
    """
    parser = argparse.ArgumentParser(
        description="Evaluate document processing results against ground truth."
    )
    parser.add_argument("--config", type=str, help="Path to a JSON config file")
    parser.add_argument("--output_csv", type=str, help="Path to the output CSV file")

    args = parser.parse_args()

    # Load configuration from file
    with open(args.config) as f:
        config_data = json.load(f)
    config_data["storage"]["output_csv"] = args.output_csv if args.output_csv else ""
    config = DocumentEvaluationConfig(**config_data)

    return config


def generate_doc_eval_metrics(config_file_path: str, output_csv_path: str = None) -> pd.DataFrame:
    """
    Generate document evaluation metrics from a config file.

    Args:
        config_file_path: Path to the JSON config file
        output_csv_path: Optional path to save CSV output

    Returns:
        pandas.DataFrame: DataFrame with evaluation results
    """
    # Load configuration from file
    with open(config_file_path) as f:
        config_data = json.load(f)
    config_data["storage"]["output_csv"] = output_csv_path or ""
    config = DocumentEvaluationConfig(**config_data)

    return generate_doc_eval_metrics_from_config(config)


def generate_doc_eval_metrics_from_config(config: DocumentEvaluationConfig) -> pd.DataFrame:
    """
    Generate document evaluation metrics from a config object.

    Args:
        config: DocumentEvaluationConfig object

    Returns:
        pandas.DataFrame: DataFrame with evaluation results
    """
    start_time = time.time()

    all_datapoints, all_table_counts = process_all_datasets(config)

    if config.storage.store_metrics and config.storage.base_path:

        # Extract dataset name from the first datapoint (all should have the same dataset)
        provider = config.model_metadata.context_sub_sub_label or "unknown"
        model = config.model_metadata.context_label or "unknown"

        write_metrics_to_parquet(
            all_datapoints,
            config=config.model_metadata,
            base_path=config.storage.base_path,
            naming_config=config.file_naming,
            provider=provider,
            model=model,
        )

    # always save locally if output path is given
    if config.storage.output_csv:
        logging.info("storing results locally")

        # Create scores DataFrame
        scores_df = datapoint_scores_to_scores_df(all_datapoints)

        # Start with scores DataFrame as the base result
        result = scores_df

        # Process table counts if available and configured
        table_counts_df = None
        if all_table_counts and getattr(config, "merge_table_counts", False):
            table_counts_rows = []
            for key, value in all_table_counts.items():
                row = {"filename": key}
                row.update(value)
                table_counts_rows.append(row)

            if table_counts_rows:
                table_counts_df = pd.DataFrame(table_counts_rows)

                result = pd.merge(
                    left=result,
                    right=table_counts_df,
                    left_on="name",
                    right_on="filename",
                    how="left",
                )
                if "filename" in result.columns:
                    result = result.drop("filename", axis=1)

        # Save the main results DataFrame
        safe_to_csv(result, config.storage.output_csv, index=False)

    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")

    return result


def main():
    """
    Main function to run the evaluation.

    This function orchestrates the overall evaluation process:
    1. Loads configuration from a JSON file
    2. Processes each dataset to calculate metrics
    3. Generates DataFrames for scores and optionally for table counts and metadata
    4. Saves the results to CSV file(s)

    Table counts and metadata are only processed when explicitly configured.
    """
    config = get_config_from_args()
    generate_doc_eval_metrics_from_config(config)


if __name__ == "__main__":
    main()
