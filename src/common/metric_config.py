"""
Centralized metric configuration using MetricRegistry.

This module provides the single source of truth for all metric configurations
and ensures alignment across STATS_PROCESSORS, SCORE_FUNCS, and METRIC_TO_USE_CASE_MAP
and VALID_FORMATS.
"""

from processing.aggregate import (
    _process_basemodel_metric,
    _process_matrix_metric,
    _process_numeric_metric,
)
from scoring.cct_adjustment import score_cct_adjustment_by_filename
from scoring.content_scoring import (
    score_cct_by_filename,
    score_tokens_added_by_filename,
    score_tokens_found_by_filename,
)
from scoring.element_consistency_scoring import score_element_consistency_by_filename
from scoring.structure_scoring import score_element_alignment_by_filename
from scoring.table_scoring import score_tables_by_filename

from .metric_registry import MetricRegistry
from .metric_types import MetricType

# Create the central metric registry
metric_registry = MetricRegistry(MetricType)

# Register all metrics with their associated functions
metric_registry.register_metric(
    MetricType.CCT,
    stats_processor=_process_numeric_metric,
    score_func=score_cct_by_filename,
    use_case_template="content_{version}",
    valid_formats={"v1": ["v1", "plain_text_v1", "unstructured_json_v1"]},
    default_config_sample_format="v1",
    default_config_ground_truth_format="v1",
)

metric_registry.register_metric(
    MetricType.PERCENT_TOKENS_FOUND,
    stats_processor=_process_numeric_metric,
    score_func=score_tokens_found_by_filename,
    use_case_template="content_{version}",
    valid_formats={"v1": ["v1", "plain_text_v1", "unstructured_json_v1"]},
    default_config_sample_format="v1",
    default_config_ground_truth_format="v1",
)

metric_registry.register_metric(
    MetricType.PERCENT_TOKENS_ADDED,
    stats_processor=_process_numeric_metric,
    score_func=score_tokens_added_by_filename,
    use_case_template="content_{version}",
    valid_formats={"v1": ["v1", "plain_text_v1", "unstructured_json_v1"]},
    default_config_sample_format="v1",
    default_config_ground_truth_format="v1",
)

metric_registry.register_metric(
    MetricType.TABLES,
    stats_processor=_process_basemodel_metric,
    score_func=score_tables_by_filename,
    use_case_template="table_{version}",
    valid_formats={"v1": ["html", "cells", "text"]},
    default_config_sample_format="html",
    default_config_ground_truth_format="text",
)

metric_registry.register_metric(
    MetricType.ELEMENT_ALIGNMENT,
    stats_processor=_process_numeric_metric,
    score_func=score_element_alignment_by_filename,
    use_case_template="structure_{version}",
    valid_formats={"v1": ["v1", "plain_text_v1", "unstructured_json_v1"]},
    default_config_sample_format="v1",
    default_config_ground_truth_format="v1",
)

metric_registry.register_metric(
    MetricType.ELEMENT_CONSISTENCY,
    stats_processor=_process_matrix_metric,
    score_func=score_element_consistency_by_filename,
    use_case_template="element_consistency_{version}",
    valid_formats={"v1": ["v1", "plain_text_v1", "unstructured_json_v1"]},
    default_config_sample_format="v1",
    default_config_ground_truth_format="v1",
)

metric_registry.register_metric(
    MetricType.ADJUSTED_CCT,
    stats_processor=_process_numeric_metric,
    score_func=score_cct_adjustment_by_filename,
    use_case_template="content_{version}",
    valid_formats={"v1": ["v1", "unstructured_json_v1"]},
    default_config_sample_format="v1",
    default_config_ground_truth_format="v1",
)

# Export the configured dictionaries
STATS_PROCESSORS = metric_registry.get_stats_processors()
SCORE_FUNCS = metric_registry.get_score_funcs()
METRIC_TO_USE_CASE_MAP = metric_registry.get_metric_to_use_case_map()
VALID_FORMATS = metric_registry.get_valid_formats()
CONFIG_SAMPLE_FORMATS = metric_registry.get_default_config_sample_formats()
CONFIG_GROUND_TRUTH_FORMATS = metric_registry.get_default_config_ground_truth_formats()
