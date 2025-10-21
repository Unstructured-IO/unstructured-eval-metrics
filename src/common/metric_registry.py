"""
Metric Registry for ensuring alignment across metric-related components.

This module provides a centralized registry that ensures all metric-related
dictionaries (STATS_PROCESSORS, SCORE_FUNCS, METRIC_TO_USE_CASE_MAP) remain
aligned when new metrics are added.
"""

from typing import Any, Callable, Dict, List, Set


class MetricRegistry:
    """
    Central registry for managing metric types and their associated functions.

    This registry ensures that all metric-related functions and mappings stay synchronized
    when new metrics are added to the system.
    """

    def __init__(self, metric_enum_class: type):

        self.metric_enum_class = metric_enum_class
        self._stats_processors: Dict[Any, Callable] = {}
        self._score_funcs: Dict[Any, Callable] = {}
        self._metric_to_use_case_map: Dict[Any, str] = {}
        self._valid_formats: Dict[Any, Dict[Any, List[str]]] = {}
        self._registered_metrics: Set[Any] = set()
        self._default_config_sample_formats: Dict[Any, str] = {}
        self._default_config_ground_truth_formats: Dict[Any, str] = {}

    def register_metric(
        self,
        metric_type: Any,
        stats_processor: Callable,
        score_func: Callable,
        use_case_template: str,
        valid_formats: Dict[str, List[str]] = {},
        default_config_sample_format: str = "v1",
        default_config_ground_truth_format: str = "v1",
    ) -> None:
        """
        Register a metric with all its associated functions.

        Args:
            metric_type: The metric type from the enum
            stats_processor: Function to process/aggregate metric statistics
            score_func: Function to calculate the metric score
            use_case_template: Template string for use case mapping
            valid_formats: Dictionary mapping version strings to valid format lists
            default_config_sample_format: Default format for sample data
            default_config_ground_truth_format: Default format for ground truth data
        """
        # Validate metric type belongs to the enum
        if not isinstance(metric_type, self.metric_enum_class):
            raise ValueError(
                f"Metric type {metric_type} is not a valid {self.metric_enum_class.__name__}"
            )

        # Check if already registered
        if metric_type in self._registered_metrics:
            raise ValueError(f"Metric type {metric_type} is already registered")

        # Register all components
        self._stats_processors[metric_type] = stats_processor
        self._score_funcs[metric_type] = score_func
        self._metric_to_use_case_map[metric_type] = use_case_template
        self._registered_metrics.add(metric_type)

        # Register valid formats if provided
        if valid_formats:
            for version, formats in valid_formats.items():
                if version not in self._valid_formats:
                    self._valid_formats[version] = {}
                self._valid_formats[version][metric_type] = formats

        # Register default formats if provided
        if default_config_sample_format:
            self._default_config_sample_formats[metric_type] = default_config_sample_format
        if default_config_ground_truth_format:
            self._default_config_ground_truth_formats[metric_type] = (
                default_config_ground_truth_format
            )

    def validate_completeness(self) -> None:
        """
        Validate that all enum values have been registered.
        """
        enum_values = set(self.metric_enum_class)
        missing_metrics = enum_values - self._registered_metrics

        if missing_metrics:
            raise ValueError(f"Missing registrations for metrics: {missing_metrics}")

    def get_stats_processors(self) -> Dict[Any, Callable]:
        """Get the stats processors dictionary."""
        self.validate_completeness()
        return self._stats_processors.copy()

    def get_score_funcs(self) -> Dict[Any, Callable]:
        """Get the score functions dictionary."""
        self.validate_completeness()
        return self._score_funcs.copy()

    def get_metric_to_use_case_map(self) -> Dict[Any, str]:
        """Get the metric to use case mapping dictionary."""
        self.validate_completeness()
        return self._metric_to_use_case_map.copy()

    def get_registered_metrics(self) -> Set[Any]:
        """Get all registered metric types."""
        return self._registered_metrics.copy()

    def get_valid_formats(self) -> Dict[Any, Dict[Any, List[str]]]:
        """Get the valid formats configuration."""
        self.validate_completeness()
        return self._valid_formats.copy()

    def get_default_config_sample_formats(self) -> Dict[Any, str]:
        """Get the default sample formats."""
        return self._default_config_sample_formats.copy()

    def get_default_config_ground_truth_formats(self) -> Dict[Any, str]:
        """Get the default ground truth formats."""
        return self._default_config_ground_truth_formats.copy()
