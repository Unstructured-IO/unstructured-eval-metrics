"""
Core metric types and enums.
"""

from enum import StrEnum


class MetricType(StrEnum):
    CCT = "cct"
    ADJUSTED_CCT = "adjusted_cct"
    PERCENT_TOKENS_FOUND = "percent_tokens_found"
    PERCENT_TOKENS_ADDED = "percent_tokens_added"
    TABLES = "tables"
    ELEMENT_ALIGNMENT = "element_alignment"
    ELEMENT_CONSISTENCY = "element_consistency"
