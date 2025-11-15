import logging
from typing import List, Optional, Tuple, Union

from rapidfuzz.distance import Levenshtein

from processing.text_processing import (
    bag_of_tokens,
    extract_content_from_file,
    extract_text,
    prepare_str,
)

logger = logging.getLogger(__name__)


# Core content metrics
def calculate_edit_distance(
    output: Optional[str],
    source: Optional[str],
    weights: Tuple[int, int, int] = (1, 1, 1),
    return_as: str = "distance",
    standardize_whitespaces: bool = True,
    short_string_comparison: bool = False,
) -> float:
    """
    Calculates edit distance using Levenshtein distance between two strings.

    Args:
        output (str): The target string to be compared.
        source (str): The reference string against which 'output' is compared.
        weights (Tuple[int, int, int], optional): A tuple containing weights
            for insertion, deletion, and substitution operations. Default is (1, 1, 1).
        return_as (str, optional): The type of result to return, one of
            ["score", "distance"]. Default is "distance".
        standardize_whitespaces bool: If True, removes extra whitespace. Default is True.

    Returns:
        float: The calculated edit distance or similarity score.
    """
    return_types = ["score", "distance"]
    if return_as not in return_types:
        raise ValueError(f"Invalid return value type. Expected one of: {return_types}")

    output = prepare_str(output, standardize_whitespaces)
    source = prepare_str(source, standardize_whitespaces)
    distance = Levenshtein.distance(output, source, weights=weights)

    # Avoid division by zero
    # if short string comparison, use the length of the longer string as the denominator
    if short_string_comparison:
        source_char_len = max(len(source), len(output), 1.0)
        source_char_len += 1  # add 1 to make the score more smooth
    else:
        source_char_len = max(len(source), 1.0)

    bounded_percentage_distance = min(max(distance / source_char_len, 0.0), 1.0)

    if return_as == "score":
        return 1 - bounded_percentage_distance
    return distance


# Higher-level scoring functions
def score_cct(
    sample_string: Union[str, List[dict]],
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
        float: The computed edit distance.
    """
    sample_text = extract_text(sample_string, sample_format, exclude_elements)
    gt_text = extract_text(ground_truth, ground_truth_format, exclude_elements)

    distance = calculate_edit_distance(sample_text, gt_text, return_as="score")
    distance = round(distance, 2)
    return distance


def score_cct_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    cct_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Reads content from two files and compares them using `score_cct`.

    Args:
        sample_filename (str): Path to the sample file.
        ground_truth_filename (str): Path to the ground truth file.
        cct_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): Format for sample file input.
        ground_truth_format (str, optional): Format for ground truth file input.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
    Returns:
        float: The computed edit distance.
    """
    with open(sample_filename, encoding="utf-8") as f_sample:
        raw_sample = f_sample.read()

    with open(ground_truth_filename, encoding="utf-8") as f_gt:
        raw_gt = f_gt.read()

    sample_data = extract_content_from_file(raw_sample)
    gt_data = extract_content_from_file(raw_gt)

    return score_cct(
        sample_data,
        gt_data,
        cct_algo=cct_algo,
        sample_format=sample_format,
        ground_truth_format=ground_truth_format,
        exclude_elements=exclude_elements,
    )


def calculate_percent_tokens_found(
    output: Optional[str],
    source: Optional[str],
) -> float:
    """
    Creates the bag of tokens (BOT) found in each input text and their frequencies,
    then compares the output BOT against the source BOT to calculate the % of tokens
    from the source text found in the output text.

    If the output text contains all tokens from the source text and then some extra,
    result will be 100% tokens found.

    Args:
        output (str): The target string to be compared.
        source (str): The reference string against which 'output' is compared.

    Returns:
        float: The percentage of tokens found represented as a decimal between 0 and 1.
    """
    output = prepare_str(output)
    source = prepare_str(source)
    output_bot = bag_of_tokens(output)
    source_bot = bag_of_tokens(source)

    # get total words in source bow while counting missing words

    total_source_token_count = sum(source_bot.values())

    # calculate percent missing text
    if total_source_token_count == 0:
        return 0  # nothing missing because nothing in source document

    total_missing_token_count = sum(
        [max(v - output_bot.get(k, 0), 0) for k, v in source_bot.items()]
    )

    score_tokens_found = 1 - round(total_missing_token_count / total_source_token_count, 3)

    # Add clamping as safety net to ensure that any anomalies in the
    # computed result don’t break the [0, 1] bounds.
    return max(score_tokens_found, 0)


def score_tokens_found(
    sample_string: Union[str, List[dict]],
    ground_truth: Union[str, List[dict]],
    tokens_found_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Compare a sample vs. ground truth using a specified algorithm,
    returning the percentage of words found.

    Args:
        sample_string (Union[str, List[dict]]): The sample input (plain text or list of dicts).
        ground_truth (Union[str, List[dict]]): The ground truth input (plain text or list of dicts).
        tokens_found_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_string. Accepts:
            - "v1" (auto-detect: string → "plain_text_v1", list → "unstructured_json_v1")
            - "plain_text_v1"
            - "unstructured_json_v1"
        ground_truth_format (str, optional): The format of ground_truth
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation

    Returns:
        float: The computed percentage of words found.
    """
    sample_text = extract_text(sample_string, sample_format, exclude_elements)
    gt_text = extract_text(ground_truth, ground_truth_format, exclude_elements)

    percent_tokens_found = calculate_percent_tokens_found(sample_text, gt_text)
    return round(percent_tokens_found, 2)


def score_tokens_found_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    tokens_found_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Reads content from two files and compares them using `score_words_found`.

    Args:
        sample_filename (str): Path to the sample file.
        ground_truth_filename (str): Path to the ground truth file.
        tokens_found_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): Format for sample file input.
        ground_truth_format (str, optional): Format for ground truth file input.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation

    Returns:
        float: The computed percentage of words found.
    """
    with open(sample_filename, encoding="utf-8") as f_sample:
        raw_sample = f_sample.read()

    with open(ground_truth_filename, encoding="utf-8") as f_gt:
        raw_gt = f_gt.read()

    sample_data = extract_content_from_file(raw_sample)
    gt_data = extract_content_from_file(raw_gt)

    return score_tokens_found(
        sample_data,
        gt_data,
        tokens_found_algo=tokens_found_algo,
        sample_format=sample_format,
        ground_truth_format=ground_truth_format,
        exclude_elements=exclude_elements,
    )


def calculate_percent_tokens_added(
    output: Optional[str],
    source: Optional[str],
) -> float:
    """
    Creates the bag of tokens (BOT) added in each input text and their frequencies,
    then compares the output BOT against the source BOT to calculate the % of tokens
    from the source text added by the output text.

    If the output text does not contains tokens not present in the source
    text, result will be 0% tokens added.

    Args:
        output (str): The target string to be compared.
        source (str): The reference string against which 'output' is compared.

    Returns:
        float: The percentage of tokens added represented as a decimal between 0 and 1.
    """
    output = prepare_str(output)
    source = prepare_str(source)
    output_bot = bag_of_tokens(output)
    source_bot = bag_of_tokens(source)

    # get total words in source bow while counting missing words

    total_output_token_count = sum(output_bot.values())

    # calculate percent added text
    if total_output_token_count == 0:
        return 0

    total_added_token_count = sum([max(v - source_bot.get(k, 0), 0) for k, v in output_bot.items()])

    score_tokens_added = round(total_added_token_count / total_output_token_count, 3)

    # Add clamping as safety net to ensure that any anomalies in the
    # computed result don’t break the [0, 1] bounds.
    return max(score_tokens_added, 0)


def score_tokens_added(
    sample_string: Union[str, List[dict]],
    ground_truth: Union[str, List[dict]],
    tokens_added_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Compare a sample vs. ground truth using a specified algorithm,
    returning the percentage of tokens added.

    Args:
        sample_string (Union[str, List[dict]]): The sample input (plain text or list of dicts).
        ground_truth (Union[str, List[dict]]): The ground truth input (plain text or list of dicts).
        tokens_added_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): The format of sample_string. Accepts:
            - "v1" (auto-detect: string → "plain_text_v1", list → "unstructured_json_v1")
            - "plain_text_v1"
            - "unstructured_json_v1"
        ground_truth_format (str, optional): The format of ground_truth
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation

    Returns:
        float: The computed percentage of words added.
    """
    sample_text = extract_text(sample_string, sample_format, exclude_elements)
    gt_text = extract_text(ground_truth, ground_truth_format, exclude_elements)

    percent_tokens_added = calculate_percent_tokens_added(sample_text, gt_text)
    return round(percent_tokens_added, 2)


def score_tokens_added_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    tokens_added_algo: Optional[str] = "v1",
    sample_format: Optional[str] = "v1",
    ground_truth_format: Optional[str] = "v1",
    exclude_elements: Optional[List[str]] = None,
) -> float:
    """
    Reads content from two files and compares them using `score_words_added`.

    Args:
        sample_filename (str): Path to the sample file.
        ground_truth_filename (str): Path to the ground truth file.
        tokens_added_algo (str, optional): The algorithm version. Defaults to "v1".
        sample_format (str, optional): Format for sample file input.
        ground_truth_format (str, optional): Format for ground truth file input.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation

    Returns:
        float: The computed percentage of words added.
    """
    with open(sample_filename, encoding="utf-8") as f_sample:
        raw_sample = f_sample.read()

    with open(ground_truth_filename, encoding="utf-8") as f_gt:
        raw_gt = f_gt.read()

    sample_data = extract_content_from_file(raw_sample)
    gt_data = extract_content_from_file(raw_gt)

    return score_tokens_added(
        sample_data,
        gt_data,
        tokens_added_algo=tokens_added_algo,
        sample_format=sample_format,
        ground_truth_format=ground_truth_format,
        exclude_elements=exclude_elements,
    )
