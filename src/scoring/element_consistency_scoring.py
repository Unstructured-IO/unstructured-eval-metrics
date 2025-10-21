import logging
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from processing.structure_processing import parse_structure
from processing.text_processing import extract_content_from_file

from .content_scoring import calculate_edit_distance
from .structure_scoring import CATEGORY_MAPPINGS

logger = logging.getLogger(__name__)

MAPPINGS = {str(elem): key for key, value in CATEGORY_MAPPINGS.items() for elem in value}


class ElementConsistencyScoreV1(BaseModel):
    """
    The score for the table detection (F1) and cell level alignment.
    """

    detection_f1: float = Field(default=0, description="The F1 score for element detection.")

    @model_validator(mode="after")
    def validate_scores(self):
        if self.detection_f1 < 0:
            raise ValueError("Scores cannot be negative")
        return self

    @staticmethod
    def get_scores_with_values(val: float) -> "ElementConsistencyScoreV1":
        return ElementConsistencyScoreV1(detection_f1=val)


class ElementConsistencyScore(BaseModel):
    version: str = Field(default="v1")
    scores: Optional[ElementConsistencyScoreV1] = None
    matrix: Optional[Dict[Tuple[str, str], int]] = Field(
        default=None, exclude=True, description="Confusion matrix for element alignment"
    )

    @model_validator(mode="after")
    def validate_version_and_scores(self):
        if self.version == "v1":
            if self.scores is None:
                self.scores = (
                    ElementConsistencyScoreV1()
                )  # Set a default ElementConsistencyScoreV1 instance
            elif not isinstance(self.scores, ElementConsistencyScoreV1):
                raise ValueError(
                    "When scores_version is 'v1', scores must be ElementConsistencyScoreV1"
                )
        else:
            raise ValueError(f"Invalid version: {self.version}")
        return self

    def get_scores_version(self, version: str) -> Optional[BaseModel]:
        """
        Return scores as ElementConsistencyScoreV1 if version is v1, otherwise raise ValueError
        """
        if version == "v1" and isinstance(self.scores, ElementConsistencyScoreV1):
            return self.scores
        else:
            raise ValueError(f"Invalid version: {self.version}")

    def get_scores_with_values(self, val: float) -> "ElementConsistencyScore":
        """
        Return a ElementConsistencyScore with scores with values
        """
        self.scores = self.get_scores_version(self.version).get_scores_with_values(val)
        return self


def align_elements(prediction: List[dict], ground_truth: List[dict]) -> List[Tuple[dict, dict]]:
    """
    Align elements from prediction and ground truth based on their types.

    Args:
        prediction (List[dict]): The predicted elements.
        ground_truth (List[dict]): The ground truth elements.

    Returns:
        List[Tuple[dict, dict]]: Aligned prediction and ground truth elements.
    """

    gt_index = 0
    aligned_prediction = []

    for i in range(len(prediction)):
        pred = prediction[i]

        gt_found = None
        for g in range(gt_index, len(ground_truth)):
            gt = ground_truth[g]

            if ("text" in pred and "text" in gt) and (
                calculate_edit_distance(
                    output=str(pred["text"]), source=str(gt["text"]), return_as="score"
                )
                > 0.8
            ):
                for g_nf in range(gt_index, g):
                    aligned_prediction.append(({"type": "nomatch", "text": ""}, ground_truth[g_nf]))

                gt_found = gt
                gt_index = g + 1
                break

        if not gt_found:
            aligned_prediction.append((pred, {"type": "nomatch", "text": ""}))
        else:
            aligned_prediction.append((pred, gt_found))

    return aligned_prediction


def calculate_f1_from_confusion_matrix(matrix):
    """
    Calculate F1 score from a confusion matrix.

    Args:
        matrix (Dict[Tuple[str, str], int]): Confusion matrix with (predicted, actual) keys

    Returns:
        float: F1 detection score
    """
    true_positives = sum(
        count
        for (pred_type, actual_type), count in matrix.items()
        if pred_type == actual_type and pred_type != "nomatch"
    )

    total_predictions = sum(
        count for (pred_type, _), count in matrix.items() if pred_type != "nomatch"
    )

    total_actual = sum(
        count for (_, actual_type), count in matrix.items() if actual_type != "nomatch"
    )

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1


def map_elements(elements: List[dict], mappings: Dict[str, str]) -> List[dict]:
    """
    Map elements to their corresponding types based on the provided mappings.

    Args:
        elements (List[dict]): List of elements to be mapped.
        mappings (Dict[str, str]): Dictionary mapping element values to types.

    Returns:
        List[dict]: List of elements with their types updated according to the mappings.
    """
    for elem in elements:
        if "type" in elem and elem["type"] in mappings:
            elem["type"] = mappings[elem["type"]]
    return elements


def score_element_consistency(
    sample_data: List[dict],
    ground_truth_data: List[dict],
    element_consistency_algo: Optional[str] = "v1",
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> ElementConsistencyScore:
    """
    Compare a sample vs. ground truth using a specified element consistency algorithm,
    returning the element consistency scores.

    Args:
        sample_elements (List[dict]): The sample input (list of dicts).
        ground_truth_elements (List[dict]): The ground truth input (list of dicts).
        element_consistency_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_elements.
        ground_truth_format (str, optional): The format of ground_truth_elements.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
    Returns:
        ElementConsistencyScore: The computed element consistency scores object.
    """

    sample_elements = parse_structure(sample_data, sample_format)
    ground_truth_elements = parse_structure(ground_truth_data, ground_truth_format)

    sample_elements = map_elements(sample_elements, MAPPINGS)
    ground_truth_elements = map_elements(ground_truth_elements, MAPPINGS)

    aligned_elements = align_elements(
        prediction=sample_elements, ground_truth=ground_truth_elements
    )

    confusion_matrix = {}
    for pred, gt in aligned_elements:
        pred_type = pred["type"] if "type" in pred else "nomatch"
        gt_type = gt["type"] if "type" in gt else "nomatch"

        if exclude_elements:
            if pred_type not in exclude_elements and gt_type not in exclude_elements:
                confusion_matrix[(pred_type, gt_type)] = (
                    confusion_matrix.get((pred_type, gt_type), 0) + 1
                )
        else:
            confusion_matrix[(pred_type, gt_type)] = (
                confusion_matrix.get((pred_type, gt_type), 0) + 1
            )

    detection_f1 = calculate_f1_from_confusion_matrix(confusion_matrix)

    # Create the ElementConsistencyScore object
    element_consistency_score = ElementConsistencyScore(
        version=element_consistency_algo,
        matrix=confusion_matrix,
        scores=ElementConsistencyScoreV1(
            detection_f1=round(detection_f1, 2),
        ),
    )
    return element_consistency_score


def score_element_consistency_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    element_consistency_algo: Optional[str] = "v1",
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> ElementConsistencyScore:
    """
    Compare a sample file vs. ground truth file using a specified table algorithm,
    returning the table scores.

    Args:
        sample_filename (str): Path to the sample file.
        ground_truth_filename (str): Path to the ground truth file.
        element_consistency_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_elements.
        ground_truth_format (str, optional): The format of ground_truth_elements.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation

    Returns:
        ElementConsistencyScore: The computed element consistency scores object.
    """
    try:
        with open(sample_filename, encoding="utf-8") as f_sample:
            raw_sample = f_sample.read()
    except Exception as e:
        raise ValueError(f"Error reading sample file {sample_filename}: {e}")

    try:
        with open(ground_truth_filename, encoding="utf-8") as f_gt:
            raw_gt = f_gt.read()
    except Exception as e:
        raise ValueError(f"Error reading ground truth file {ground_truth_filename}: {e}")

    sample_data = extract_content_from_file(raw_sample)
    gt_data = extract_content_from_file(raw_gt)

    element_consistency_scores = score_element_consistency(
        sample_data=sample_data,
        ground_truth_data=gt_data,
        sample_format=sample_format,
        ground_truth_format=ground_truth_format,
        element_consistency_algo=element_consistency_algo,
        exclude_elements=exclude_elements,
    )

    return element_consistency_scores
