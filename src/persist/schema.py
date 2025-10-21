import datetime
from typing import List, Optional

from pydantic import BaseModel

# Define the schema structure for reference and validation

class MetricDataRecord(BaseModel):
    uuid: str
    dataset: str
    metric_name: str
    context_version: str
    context_sub_sub_label: Optional[str] = None
    output_identifier: Optional[str] = None
    source_filename: Optional[str] = None
    prediction_filename: Optional[bytes] = None
    ground_truth_filename: Optional[bytes] = None
    created_at: datetime.datetime
    metric_value: Optional[float] = None
    metric_nan: Optional[str] = None
    metric_json: Optional[str] = None
    hour: str


class PartitionInfo(BaseModel):
    context_type: str
    use_case: str
    context_label: str
    context_sub_label: str
    year: str
    month: str
    day: str


class MetricScore(BaseModel):
    data: MetricDataRecord
    partition: PartitionInfo


# Define column names for easy access
DATA_COLUMNS: List[str] = list(MetricDataRecord.model_fields.keys())
PARTITION_COLUMNS: List[str] = list(PartitionInfo.model_fields.keys())
ALL_COLUMNS: List[str] = DATA_COLUMNS + PARTITION_COLUMNS
