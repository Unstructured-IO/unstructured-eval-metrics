from enum import StrEnum
from itertools import groupby
from typing import List, Union

from processing.structure_processing import parse_elements
from processing.text_processing import extract_content_from_file
from scoring.content_scoring import calculate_edit_distance


class ElementType(StrEnum):
    """Enum for document element types."""

    TITLE = "Title"
    TEXT = "Text"
    UNCATEGORIZED_TEXT = "UncategorizedText"
    NARRATIVE_TEXT = "NarrativeText"
    BULLETED_TEXT = "BulletedText"
    PARAGRAPH = "Paragraph"
    ABSTRACT = "Abstract"
    THREADING = "Threading"
    FORM = "Form"
    FIELD_NAME = "FieldName"
    VALUE = "Value"
    LINK = "Link"
    COMPOSITE_ELEMENT = "CompositeElement"
    IMAGE = "Image"
    PICTURE = "Picture"
    FIGURE_CAPTION = "FigureCaption"
    FIGURE = "Figure"
    CAPTION = "Caption"
    LIST = "List"
    LIST_ITEM = "ListItem"
    CHECKED = "Checked"
    UNCHECKED = "Unchecked"
    CHECK_BOX_CHECKED = "CheckBoxChecked"
    CHECK_BOX_UNCHECKED = "CheckBoxUnchecked"
    RADIO_BUTTON_CHECKED = "RadioButtonChecked"
    RADIO_BUTTON_UNCHECKED = "RadioButtonUnchecked"
    ADDRESS = "Address"
    EMAIL_ADDRESS = "EmailAddress"
    PAGE_BREAK = "PageBreak"
    FORMULA = "Formula"
    TABLE = "Table"
    HEADER = "Header"
    HEADLINE = "Headline"
    SUB_HEADLINE = "Subheadline"
    PAGE_HEADER = "PageHeader"
    SECTION_HEADER = "SectionHeader"
    FOOTER = "Footer"
    FOOTNOTE = "Footnote"
    PAGE_FOOTER = "PageFooter"
    PAGE_NUMBER = "PageNumber"
    CODE_SNIPPET = "CodeSnippet"
    FORM_KEYS_VALUES = "FormKeysValues"
    TABLE_CAPTION = "TableCaption"
    SUB_TITLE = "SubTitle"
    OTHER = "Other"


class CategoryID(StrEnum):
    """Enum for structural category IDs."""

    TITLE = "1"
    FIGURE = "2"
    TABLE = "3"


# Define category mappings for element types to structural categories
# FORMS and CAPTIONS are removed as we observe that we don't annotate them
# in the predictions

# TODO:Sub-Title is added for confusion matrix now, we need to refactor this part.

CATEGORY_MAPPINGS = {
    "TITLE": {ElementType.TITLE, ElementType.SUB_TITLE, "Sub-Title"},
    "FIGURE": {ElementType.FIGURE, ElementType.IMAGE, ElementType.PICTURE},
    "TABLE": {ElementType.TABLE},
}


def get_structural_category(element_type: str) -> str:
    """
    Map element types to major structural categories.

    Args:
        element_type: The original element type string

    Returns:
        The major structural category name, or empty string if no match
    """
    element_type = element_type.title().replace("-", "").replace(" ", "")
    for category, types in CATEGORY_MAPPINGS.items():
        values = [str(elem_type) for elem_type in types]
        if element_type in values:
            return category

    return str(ElementType.OTHER)


def _flatten_categories(categories: List[str]) -> List[str]:
    """
    Remove consecutive duplicate categories from a list.

    Args:
        categories: List of category strings

    Returns:
        List with consecutive duplicates removed
    """
    if not categories:
        return []
    return [key for key, _ in groupby(categories)]


# Data transformation functions
def transform_to_structural_categories(elements: List[str]) -> List[str]:
    """
    Transform element types to structural categories.

    Args:
        elements: List of element type strings

    Returns:
        List of corresponding structural category names
    """
    return [get_structural_category(elem) for elem in elements]


def transform_to_category_ids(categories: List[str]) -> List[str]:
    """
    Transform categories to category IDs for edit distance calculation.

    Converts category names to single-digit numeric IDs to ensure the edit distance
    computation is based on structural differences rather than being skewed by the
    length of category names.

    Filters out empty categories.
    """
    return [
        CategoryID[elem].value
        for elem in categories
        if elem != "" and elem in CategoryID.__members__
    ]


def prepare_categories_for_scoring(categories: List[str]) -> str:
    """
    Prepare categories for CCT scoring by joining with spaces.

    Args:
        categories: List of category strings

    Returns:
        Space-separated string of categories
    """
    return " ".join(categories)


# Core scoring functions
def compute_structure_score(
    pred_categories: List[str],
    true_categories: List[str],
    ele_align_algo: str = "v1",
    short_string_threshold: int = 5,
) -> float:
    """
    Compute structure score using CCT algorithm.

    Args:
        pred_categories: Predicted category sequence
        true_categories: Ground truth category sequence
        ele_align_algo: Algorithm version for element alignment

    Returns:
        score: The computed edit distance.
    """
    pred_categories_cct = prepare_categories_for_scoring(pred_categories)
    true_categories_cct = prepare_categories_for_scoring(true_categories)

    short_string_flag = False

    both_short_string_cond = (
        len(pred_categories_cct) <= short_string_threshold
        and len(true_categories_cct) <= short_string_threshold
    )

    # If one of the strings is empty, we don't use short string comparison,
    # otherwise it will result in higher score
    either_empty_cond = len(pred_categories_cct) == 0 or len(true_categories_cct) == 0

    if both_short_string_cond and not either_empty_cond:
        short_string_flag = True

    return round(
        calculate_edit_distance(
            pred_categories_cct,
            true_categories_cct,
            return_as="score",
            short_string_comparison=short_string_flag,
        ),
        2,
    )


def score_element_alignment(
    sample: Union[str, List[dict]],
    ground_truth: Union[str, List[dict]],
    ele_align_algo: str = "v1",
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
) -> float:
    """
    Score document structure understanding by comparing predicted vs ground truth element types.

    The score is computed by:
    1. Parsing the sample and ground truth data into element types
    2. Keeping only the element types that are in the CATEGORY_MAPPINGS,
       which are the major structural categories
    3. Transforming the element types to structural categories
    4. Converting the structural categories to category IDs
    5. Computing the edit distance between the predicted and ground truth structural categories

    Args:
        sample: Sample data (predicted elements)
        ground_truth: Ground truth data (true elements)
        ele_align_algo: Algorithm version for element alignment
        sample_format: Format version for sample data
        ground_truth_format: Format version for ground truth data

    Returns:
        score: The computed edit distance.
    """
    parsed_sample = parse_elements(sample, sample_format)
    parsed_gt = parse_elements(ground_truth, ground_truth_format)

    # Transform to structural categories
    pred_categories = transform_to_structural_categories(parsed_sample)
    true_categories = transform_to_structural_categories(parsed_gt)

    # Convert to category IDs
    pred_category_ids = transform_to_category_ids(pred_categories)
    true_category_ids = transform_to_category_ids(true_categories)

    if len(pred_category_ids) == 0 and len(true_category_ids) == 0:
        return 1.0

    # Compute score
    return compute_structure_score(pred_category_ids, true_category_ids, ele_align_algo)


def score_element_alignment_by_filename(
    sample_filename: str,
    ground_truth_filename: str,
    ele_align_algo: str = "v1",
    sample_format: str = "v1",
    ground_truth_format: str = "v1",
) -> float:
    """
    Score element alignment by loading data from files.

    Args:
        sample_filename: Path to sample data file
        ground_truth_filename: Path to ground truth data file
        ele_align_algo: Algorithm version for element alignment
        sample_format: Format version for sample data
        ground_truth_format: Format version for ground truth data

    Returns:
        score: The computed edit distance.
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

    return score_element_alignment(
        sample_data, gt_data, ele_align_algo, sample_format, ground_truth_format
    )
