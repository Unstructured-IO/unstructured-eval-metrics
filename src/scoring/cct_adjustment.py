from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from rapidfuzz.distance import Levenshtein

from processing.structure_processing import parse_structure
from processing.text_processing import _clean_newlines, extract_text, prepare_str
from scoring.content_scoring import (
    calculate_percent_tokens_added,
    calculate_percent_tokens_found,
    extract_content_from_file,
    score_cct,
)

logger = getLogger(__name__)


@dataclass(frozen=True)
class AlignmentConfig:
    """
    Configuration parameters for CCT alignment adjustment.

    Attributes:
        token_found_threshold: Minimum threshold for token matching ratio (default: 0.9)
        cct_improvement_threshold: Minimum gap between token found and cct score (default: 0.15)
        fuzzy_threshold: Minimum similarity threshold for fuzzy matching (default: 0.8)
    """

    token_found_threshold: float = 0.9
    cct_improvement_threshold: float = 0.15
    fuzzy_threshold: float = 0.95


@dataclass(frozen=True)
class BasicMetrics:
    """
    Basic OCR evaluation metrics container.
    """

    percent_tokens_found: float
    percent_tokens_added: float
    original_cct_score: float
    predicted_table_count: int
    ground_truth_table_count: int


def count_words(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(text.split())


def extract_table_cells(table_html: str) -> List[str]:
    """
    Extract individual text segments from table cells.
    """
    if not table_html:
        return []

    soup = BeautifulSoup(table_html, "html.parser")
    cells = []

    for cell in soup.find_all(["td", "th"]):
        if cell.find("table"):  # Skip nested tables
            continue

        # Get all text nodes and filter out empty ones
        text_parts = [text.strip() for text in cell.stripped_strings if text.strip()]
        cells.extend(text_parts)  # Add each part separately

    return cells


@dataclass(frozen=True)
class FuzzyMatch:
    start: int
    end: int
    text: str
    gt_text: str
    similarity: float


def find_fuzzy_cell_matches(
    cell_text: str, gt_text: str, fuzzy_threshold: float = 0.8
) -> List[Dict]:
    """
    Find exact and fuzzy matches for cell text in ground truth text.

    First attempts exact matching, then falls back to fuzzy matching with
    a sliding window approach if no exact matches are found.

    Args:
        cell_text: Text content from table cell
        gt_text: Ground truth text to search in
        fuzzy_threshold: Minimum similarity threshold for fuzzy matching

    Returns:
        List of match dictionaries containing start, end, text, gt_text, and similarity
    """
    matches = []
    if not cell_text:
        return matches

    cell_text_lower = cell_text.lower().strip()
    gt_text_lower = gt_text.lower()

    # Try exact match first
    start_idx = 0
    while True:
        idx = gt_text_lower.find(cell_text_lower, start_idx)
        if idx == -1:
            break

        matches.append(
            FuzzyMatch(
                start=idx,
                end=idx + len(cell_text_lower),
                text=cell_text,
                gt_text=gt_text[idx : idx + len(cell_text_lower)],
                similarity=1.0,
            )
        )
        start_idx = idx + 1

    # If no exact matches, try fuzzy matching
    if not matches:
        window_size = len(cell_text_lower)
        best_similarity = 0.0

        for i in range(len(gt_text_lower) - window_size + 1):
            window = gt_text_lower[i : i + window_size]
            distance = Levenshtein.distance(cell_text_lower, window)
            similarity = 1 - (distance / max(len(cell_text_lower), len(window)))

            if similarity >= fuzzy_threshold and similarity > best_similarity:
                best_similarity = similarity
                matches.append(
                    FuzzyMatch(
                        start=i,
                        end=i + window_size,
                        text=cell_text,
                        gt_text=gt_text[i : i + window_size],
                        similarity=similarity,
                    )
                )

    # Remove overlapping matches, keeping highest similarity
    filtered_matches = []
    matches.sort(key=lambda x: x.similarity, reverse=True)

    for match in matches:
        is_overlapping = any(
            match.start < existing.end and match.end > existing.start
            for existing in filtered_matches
        )
        if not is_overlapping:
            filtered_matches.append(match)

    return filtered_matches


def find_fuzzy_text_match(pred_text: str, gt_text: str, threshold: float = 0.8) -> Optional[Dict]:
    """
    Find the best fuzzy match for predicted text in ground truth using sliding window.

    Args:
        pred_text: Predicted text to match
        gt_text: Ground truth text to search in
        threshold: Minimum similarity threshold for matching

    Returns:
        Dictionary with match details or None if no match found above threshold
    """
    if not pred_text or not gt_text:
        return None

    pred_text_lower = pred_text.lower().strip()
    gt_text_lower = gt_text.lower()

    # Try exact match first
    exact_pos = gt_text_lower.find(pred_text_lower)
    if exact_pos != -1:
        return FuzzyMatch(
            start=exact_pos,
            end=exact_pos + len(pred_text_lower),
            text=pred_text,
            gt_text=gt_text[exact_pos : exact_pos + len(pred_text_lower)],
            similarity=1.0,
        )

    # Fuzzy matching with sliding window
    best_similarity = 0.0
    best_match = None
    window_size = len(pred_text_lower)

    for i in range(len(gt_text_lower) - window_size + 1):
        window = gt_text_lower[i : i + window_size]
        distance = Levenshtein.distance(pred_text_lower, window)
        similarity = 1 - (distance / max(len(pred_text_lower), len(window)))

        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_match = FuzzyMatch(
                start=i,
                end=i + window_size,
                text=pred_text,
                gt_text=gt_text[i : i + window_size],
                similarity=similarity,
            )

    return best_match


def calculate_basic_metrics(
    sample_data: List[Dict],
    gt_data: List[Dict],
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
) -> BasicMetrics:
    """
    Calculate basic OCR evaluation metrics from sample and ground truth data.

    Args:
        sample_data: List of prediction dictionaries
        gt_data: List of ground truth dictionaries
        sample_format: Format identifier for sample data
        ground_truth_format: Format identifier for ground truth data

    Returns:
        BasicMetrics object containing fundamental evaluation metrics
    """
    pred_text = extract_text(sample_data, sample_format)
    gt_text = extract_text(gt_data, ground_truth_format)

    percent_found = calculate_percent_tokens_found(pred_text, gt_text)
    percent_added = calculate_percent_tokens_added(pred_text, gt_text)
    original_cct = score_cct(pred_text, gt_text)

    predicted_table_count = sum(1 for d in sample_data if d.get("type", "").lower() == "table")
    ground_truth_table_count = sum(1 for d in gt_data if d.get("type", "").lower() == "table")

    return BasicMetrics(
        percent_tokens_found=percent_found,
        percent_tokens_added=percent_added,
        original_cct_score=original_cct,
        predicted_table_count=predicted_table_count,
        ground_truth_table_count=ground_truth_table_count,
    )


def should_apply_adjustment(
    basic_metrics: BasicMetrics, config: AlignmentConfig
) -> Tuple[bool, str]:
    """
    Determine if cct adjustment should be applied based on evaluation metrics.
    Returns:
        Tuple of (should_apply, reason) where should_apply is boolean and
        reason is descriptive string
    """
    table_mismatch = (
        abs(basic_metrics.predicted_table_count - basic_metrics.ground_truth_table_count) > 1
    )
    token_cct_gap = basic_metrics.percent_tokens_found - basic_metrics.original_cct_score
    high_token_found = basic_metrics.percent_tokens_found > config.token_found_threshold
    significant_gap = token_cct_gap > config.cct_improvement_threshold

    if table_mismatch and high_token_found and significant_gap:
        return True, "Table structure mismatch with high token overlap"
    elif high_token_found and table_mismatch:
        return True, "High token overlap suggests reading path misalignment"
    elif table_mismatch:
        return True, "Table structure mismatch"
    else:
        return False, "No adjustment is required"


def calculate_word_weighted_alignment_scores(
    predictions: List[Dict],
    gt_data: List[Dict],
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
    config: AlignmentConfig = None,
) -> float:
    """
    Calculate word-weighted alignment scores for both table and non-table content.

    This function evaluates how well predicted content aligns with ground truth
    by calculating separate scores for table cells and non-table elements, then
    combining them into a weighted overall score.

    Args:
        predictions: List of prediction dictionaries
        gt_data: List of ground truth dictionaries
        sample_format: Format identifier for sample data
        ground_truth_format: Format identifier for ground truth data
        config: Configuration parameters for alignment thresholds

    Returns:
        adjusted cct score between 0.0 and 1.0
    """
    gt_text = extract_text(gt_data, ground_truth_format)
    gt_text = prepare_str(gt_text, True)

    # Initialize counters
    table_word_count = 0
    table_words_matched = 0
    non_table_word_count = 0
    non_table_words_matched = 0
    table_cells_discovered = 0
    table_cells_total = 0
    fuzzy_matches_found = 0
    fuzzy_matches_total = 0

    # Process each prediction element
    for pred in predictions:
        if pred.get("type") == "Table":
            table_html = pred.get("metadata", {}).get("text_as_html", "")
            if not table_html:
                continue

            table_cells = extract_table_cells(table_html)
            table_cells_total += len(table_cells)

            for cell_text in table_cells:
                cell_text = _clean_newlines(cell_text)
                if not cell_text.strip():
                    continue

                cell_word_count = count_words(cell_text)
                table_word_count += cell_word_count

                # Check if this cell can be found in ground truth
                matches = find_fuzzy_cell_matches(cell_text, gt_text, config.fuzzy_threshold)

                if matches:
                    table_cells_discovered += 1
                    table_words_matched += cell_word_count

        else:
            # Handle non-table elements
            pred_element_text = pred.get("text", "").strip()
            pred_element_text = prepare_str(_clean_newlines(pred_element_text), True)
            if not pred_element_text:
                continue

            element_word_count = count_words(pred_element_text)
            non_table_word_count += element_word_count
            fuzzy_matches_total += 1

            # Check if this element can be fuzzy matched
            best_match = find_fuzzy_text_match(pred_element_text, gt_text, config.fuzzy_threshold)
            if best_match is not None:
                fuzzy_matches_found += 1
                # Weight by similarity score
                non_table_words_matched += int(element_word_count * best_match.similarity)

    # Calculate combined score weighted by total word count
    total_words = table_word_count + non_table_word_count
    if total_words > 0:
        combined_score = (table_words_matched + non_table_words_matched) / total_words
    else:
        combined_score = 0.0

    return combined_score


def calculate_cct_adjustment_score(
    sample_data: List[Dict],
    gt_data: List[Dict],
    cct_algo: str = "v1",
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
) -> float:
    """
    Calculate CCT adjustment score with alignment metrics.

    This is the core function that combines basic metrics with alignment analysis
    to produce an adjusted CCT score that better reflects content quality.

    Args:
        sample_string (Union[str, List[dict]]): The sample input (plain text or list of dicts).
        ground_truth (Union[str, List[dict]]): The ground truth input (plain text or list of dicts).
        cct_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_string.
        ground_truth_format (str, optional): The format of ground_truth.
    Returns:
        float: The adjusted cct score.
    """
    # Calculate basic metrics
    basic_metrics = calculate_basic_metrics(
        sample_data, gt_data, sample_format, ground_truth_format
    )

    config = AlignmentConfig()

    # Determine if adjustment is needed
    adjustment_needed, reason = should_apply_adjustment(basic_metrics, config)

    adjusted_cct_score = basic_metrics.original_cct_score

    # Calculate alignment scores if adjustment is needed
    if adjustment_needed:
        alignment_score = calculate_word_weighted_alignment_scores(
            sample_data, gt_data, sample_format, ground_truth_format, config
        )

        # Use alignment score if it's higher than original CCT score
        if alignment_score > basic_metrics.original_cct_score:
            adjusted_cct_score = alignment_score

    return round(adjusted_cct_score, 2)


def score_cct_adjustment(
    sample_data: Union[str, List[dict]],
    ground_truth: Union[str, List[dict]],
    cct_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Compare a sample vs. ground truth using a specified CCT algorithm, returning the edit distance.

    Args:
        sample_string (Union[str, List[dict]]): The sample input (plain text or list of dicts).
        ground_truth (Union[str, List[dict]]): The ground truth input (plain text or list of dicts).
        cct_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_string. Accepts:
            - "v1" (auto-detect: string → "plain_text_v1", list → "unstructured_json_v1")
            - "plain_text_v1"
            - "unstructured_json_v1"
        ground_truth_format (str, optional): The format of ground_truth
        (same options as sample_format).
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
    Returns:
        float: The adjusted edit distance.
    """
    sample_elements = parse_structure(sample_data, sample_format)
    gt_elements = parse_structure(ground_truth, ground_truth_format)

    return calculate_cct_adjustment_score(
        sample_elements, gt_elements, cct_algo, sample_format, ground_truth_format
    )


def score_cct_adjustment_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    cct_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Score CCT adjustment by reading from files.

    This convenience function reads sample and ground truth data from files
    and calculates the adjusted CCT score.

    Args:
        sample_filename (str): Path to the sample file.
        ground_truth_filename (str): Path to the ground truth file.
        cct_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): Format for sample file input.
        ground_truth_format (str, optional): Format for ground truth file input.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
    Returns:
        float: The adjusted edit distance.
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

    return score_cct_adjustment(
        sample_data,
        gt_data,
        cct_algo=cct_algo,
        sample_format=sample_format,
        ground_truth_format=ground_truth_format,
        exclude_elements=exclude_elements,
    )
