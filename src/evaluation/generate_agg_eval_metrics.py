import argparse
import json
import logging
import time
from collections import defaultdict
from typing import Dict

import pandas as pd

from common.metric_types import MetricType
from evaluation.config_model import AggregateEvaluationConfig
from evaluation.generate_doc_eval_metrics import process_all_datasets
from persist.utils import safe_to_csv
from processing.aggregate import ScoreFilterConfig, ScoreValidator
from scoring.aggregate_scoring import AggregateScore


def aggregate_stats_to_dataframe(
    aggregate_stats: Dict[MetricType, Dict[str, float]]
) -> pd.DataFrame:
    """
    Convert aggregate statistics dictionary to a pandas DataFrame.

    Args:
        aggregate_stats: Dictionary of metrics with their statistics
                         {metric_type: {stat_name: value, ...}, ...}

    Returns:
        pandas DataFrame with metrics as rows and statistics as columns
    """
    # Initialize lists to store data
    metrics = []

    # Process all metrics and gather all possible statistic types
    all_stat_types = set()
    for metric_type, stats in aggregate_stats.items():
        metrics.append(metric_type)
        for stat_name in stats.keys():
            all_stat_types.add(stat_name)

    # Initialize dictionary for DataFrame
    df_data = {"metric": metrics}

    # Fill in all statistic values
    for stat_name in all_stat_types:
        df_data[stat_name] = [
            (
                aggregate_stats[metric][stat_name]
                if type(aggregate_stats[metric].get(stat_name, float("nan"))) is defaultdict
                else aggregate_stats[metric].get(stat_name, float("nan"))
            )
            for metric in metrics
        ]

    # Create DataFrame
    return pd.DataFrame(df_data)


def get_config_from_args() -> AggregateEvaluationConfig:
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
    config_data["storage"]["output_csv"] = args.output_csv
    config = AggregateEvaluationConfig(**config_data)

    return config


def generate_agg_eval_metrics(config_file_path: str, output_csv_path: str = None) -> pd.DataFrame:
    """
    Generate aggregate evaluation metrics from a config file.

    Args:
        config_file_path: Path to the JSON config file
        output_csv_path: Optional path to save CSV output

    Returns:
        pandas.DataFrame: DataFrame with aggregate evaluation results
    """
    # Load configuration from file
    with open(config_file_path) as f:
        config_data = json.load(f)
    config_data["storage"]["output_csv"] = output_csv_path or ""
    config = AggregateEvaluationConfig(**config_data)

    return generate_agg_eval_metrics_from_config(config)


def generate_agg_eval_metrics_from_config(config: AggregateEvaluationConfig) -> pd.DataFrame:
    """
    Generate aggregate evaluation metrics from a config object.

    Args:
        config: AggregateEvaluationConfig object

    Returns:
        pandas.DataFrame: DataFrame with aggregate evaluation results
    """
    start_time = time.time()

    all_datapoints, _ = process_all_datasets(config)
    # Calculate aggregate stats
    aggregate_score = AggregateScore(datapoint_scores=all_datapoints)
    score_validator = ScoreValidator(ScoreFilterConfig())
    aggregate_score.calculate_aggregate_stats(score_validator)

    # Create aggregate stats DataFrame
    aggregate_stats_df = aggregate_stats_to_dataframe(aggregate_score.get_aggregate_stats())

    safe_to_csv(aggregate_stats_df, config.storage.output_csv, index=False)
    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")

    return aggregate_stats_df


def main():
    """
    Main function to run the evaluation.

    This function orchestrates the aggregated evaluation process:
    1. Loads aggregate evaluation configuration from a JSON file
    2. Processes each dataset to calculate the aggregate stats
    3. Generates DataFrames for aggregate stats
    4. Saves the results to CSV files according to configuration
    """
    config = get_config_from_args()
    generate_agg_eval_metrics_from_config(config)


if __name__ == "__main__":
    main()
