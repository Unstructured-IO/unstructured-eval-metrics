import json
import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from processing.table_processing import (
    TableAlignment,
    extract_and_convert_tables,
)

logger = logging.getLogger(__name__)


class TablesScoreV1(BaseModel):
    """
    The score for the table detection (F) and cell level alignment.
    """

    overall: float = Field(
        default=0,
        description="The average score of the detection f, cell level content acc,\
            and cell level index acc",
    )
    detection_f: float = Field(default=0, description="The F score for the table detection.")
    detection_f_beta: float = Field(default=1, description="The F score beta.")
    cell_level_content_acc: float = Field(
        default=0, description="The accuracy of the content matching (both row and column)."
    )
    shifted_cell_content_acc: float = Field(
        default=0,
        description=(
            "The adjusted accuracy of the content matching (both row and column) "
            "by shifting the indices."
        ),
    )
    cell_level_index_acc: float = Field(
        default=0, description="The accuracy of the index matching (both row and column)."
    )
    table_teds: float = Field(default=0, description="The table edit distance score.")
    table_teds_corrected: float = Field(
        default=0, description="The corrected table edit distance score."
    )

    @model_validator(mode="after")
    def validate_scores(self):
        if (
            self.overall < 0
            or self.detection_f < 0
            or self.detection_f_beta < 0
            or self.cell_level_content_acc < 0
            or self.shifted_cell_content_acc < 0
            or self.cell_level_index_acc < 0
            or self.table_teds < 0
            or self.table_teds_corrected < 0
        ):
            raise ValueError("Scores cannot be negative")
        return self

    @staticmethod
    def get_scores_with_values(val: float) -> "TablesScoreV1":
        return TablesScoreV1(
            overall=val,
            detection_f=val,
            detection_f_beta=val,
            cell_level_content_acc=val,
            shifted_cell_content_acc=val,
            cell_level_index_acc=val,
            table_teds=val,
            table_teds_corrected=val,
        )


class TablesScore(BaseModel):
    version: str = Field(default="v1")
    scores: Optional[TablesScoreV1] = None

    @model_validator(mode="after")
    def validate_version_and_scores(self):
        if self.version == "v1":
            if self.scores is None:
                self.scores = TablesScoreV1()  # Set a default TablesScoreV1 instance
            elif not isinstance(self.scores, TablesScoreV1):
                raise ValueError("When scores_version is 'v1', scores must be TablesScoreV1")
        else:
            raise ValueError(f"Invalid version: {self.version}")
        return self

    def get_scores_version(self, version: str) -> Optional[BaseModel]:
        """
        Return scores as TablesScoreV1 if version is v1, otherwise raise ValueError
        """
        if version == "v1" and isinstance(self.scores, TablesScoreV1):
            return self.scores
        else:
            raise ValueError(f"Invalid version: {self.version}")

    def get_scores_with_values(self, val: float) -> "TablesScore":
        """
        Return a TablesScore with scores with values
        """
        self.scores = self.get_scores_version(self.version).get_scores_with_values(val)
        return self


def calculate_table_detection_metrics(
    matched_indices: list[int], ground_truth_tables_number: int, beta: float = 1.0
) -> tuple[float, float, float]:
    """
    Calculate the table detection metrics: recall, precision, and f score.
    Args:
        matched_indices:
            List of indices indicating matches between predicted and ground truth tables
            For example: matched_indices[i] = j means that the
            i-th predicted table is matched with the j-th ground truth table.
        ground_truth_tables_number: the number of ground truth tables.
        beta: the beta value for F-score calculation, which indicates the recall is consider
            beta times as precision, default is 1.0 (standard F score).

    Returns:
        Tuple of recall, precision, and f scores
    """
    predicted_tables_number = len(matched_indices)

    matched_set = set(matched_indices)
    if -1 in matched_set:
        matched_set.remove(-1)

    true_positive = len(matched_set)
    false_positive = predicted_tables_number - true_positive
    positive = ground_truth_tables_number

    recall = true_positive / positive if positive > 0 else 0
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0
    )
    f = (
        (1 + (beta**2)) * precision * recall / (((beta**2) * precision) + recall)
        if precision + recall > 0
        else 0
    )

    return recall, precision, f


def score_tables(
    sample_table: List[dict],
    ground_truth: List[dict],
    table_algo: Optional[str] = "v1",
    sample_format: Literal["html", "cells", "text"] = "html",
    ground_truth_format: Literal["html", "cells", "text"] = "text",
    merged_consecutive_tables: bool = False,
    exclude_elements: Optional[List[str]] = None,
    f_beta: float = 1.0,
) -> TablesScore:
    """
    Compare a sample vs. ground truth using a specified table algorithm, returning the table scores.

    Args:
        sample_table (List[dict]): The sample input (list of dicts).
        ground_truth (List[dict]): The ground truth input (list of dicts).
        table_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_table.
        ground_truth_format (str, optional): The format of ground_truth
        (same options as sample_format).
        merged_consecutive_tables (bool, optional): Whether to merge consecutive tables.
        f_score_beta (float, optional): The beta value for F-score calculation, default is 1.0
                                        (standard F1 score).
    Returns:
        TablesScore: The computed table scores object.
    """

    ground_truth_table_data = extract_and_convert_tables(
        input_tables=ground_truth,
        table_format=ground_truth_format,
        merged_consecutive_tables=merged_consecutive_tables,
        exclude_elements=exclude_elements,
    )
    predicted_table_data = extract_and_convert_tables(
        input_tables=sample_table,
        table_format=sample_format,
        merged_consecutive_tables=merged_consecutive_tables,
        exclude_elements=exclude_elements,
    )

    is_table_in_gt = bool(ground_truth_table_data)
    is_table_predicted = bool(predicted_table_data)

    overall_score = float("nan")
    detection_f = float("nan")
    content_acc = float("nan")
    index_acc = float("nan")
    shifted_cell_content_acc = float("nan")
    table_teds = float("nan")
    table_teds_corrected = float("nan")

    # We can't score the table content and index accuracy if there is no table in gt
    if not is_table_in_gt:
        # detection_f should be undefined, but we will normalize it to 0 or 1
        if is_table_predicted:
            # normalize the detection_f to 0 if incorrect prediction exists
            detection_f = 0
        else:
            # treat this as a perfect case, assign 1 to detection_f
            detection_f = 1
    else:
        if not is_table_predicted:
            detection_f = 0
        else:
            matched_indices = TableAlignment.get_table_level_alignment(
                predicted_table_data,
                ground_truth_table_data,
            )

            _, _, detection_f = calculate_table_detection_metrics(
                matched_indices=matched_indices,
                ground_truth_tables_number=len(ground_truth_table_data),
                beta=f_beta,
            )
            detection_f = round(detection_f, 2)

            metrics = TableAlignment.get_cell_level_alignment(
                predicted_table_data, ground_truth_table_data, matched_indices
            )

            col_index_acc = metrics["col_index_acc"]
            row_index_acc = metrics["row_index_acc"]
            col_content_acc = metrics["col_content_acc"]
            row_content_acc = metrics["row_content_acc"]
            shifted_col_content_acc = metrics["shifted_col_content_acc"]
            shifted_row_content_acc = metrics["shifted_row_content_acc"]
            table_teds = metrics["table_teds"]
            table_teds_corrected = metrics["table_teds_corrected"]

            # Average the index accuracy of the columns and rows
            index_acc = round((col_index_acc + row_index_acc) / 2, 2)
            # Average the content accuracy of the columns and rows
            content_acc = round((col_content_acc + row_content_acc) / 2, 2)
            # Average the adjusted content accuracy of the columns and rows by shifting the indices
            shifted_cell_content_acc = round(
                (shifted_col_content_acc + shifted_row_content_acc) / 2, 2
            )

            overall_score = round((detection_f + index_acc + content_acc) / 3, 2)

    return TablesScore(
        version=table_algo,
        scores=TablesScoreV1(
            overall=overall_score,
            detection_f=detection_f,
            detection_f_beta=f_beta,
            cell_level_content_acc=content_acc,
            shifted_cell_content_acc=shifted_cell_content_acc,
            cell_level_index_acc=index_acc,
            table_teds=table_teds,
            table_teds_corrected=table_teds_corrected,
        ),
    )


def score_tables_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    table_algo: Optional[str] = "v1",
    sample_format: Literal["html", "cells", "text"] = "html",
    ground_truth_format: Literal["html", "cells", "text"] = "text",
    merged_consecutive_tables: bool = False,
    f_beta: float = 1.0,
) -> TablesScore:
    """
    Compare a sample file vs. ground truth file using a specified table algorithm,
    returning the table scores.

    Args:
        sample_filename (str): Path to the sample file.
        ground_truth_filename (str): Path to the ground truth file.
        table_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str): Format for sample file input.
        ground_truth_format (str): Format for ground truth file input.
        merged_consecutive_tables (bool, optional): Whether to merge consecutive tables.
        f_beta (float, optional): The beta value for F-score calculation, default is 1.0
                                  (standard F1 score).

    Returns:
        TablesScore: The computed table scores object.
    """

    with open(sample_filename) as f:
        prediction = json.load(f)
    with open(ground_truth_filename) as gt:
        ground_truth = json.load(gt)

    table_scores = score_tables(
        sample_table=prediction,
        ground_truth=ground_truth,
        table_algo=table_algo,
        sample_format=sample_format,
        ground_truth_format=ground_truth_format,
        merged_consecutive_tables=merged_consecutive_tables,
        f_beta=f_beta,
    )

    return table_scores
