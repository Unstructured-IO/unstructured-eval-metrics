import json
import logging
import re
from typing import Dict, Final, List, Literal, Optional, Set, Union

from utils.tokenizer import TiktokenCache

logger = logging.getLogger(__name__)

InputFormat = Literal["v1", "plain_text_v1", "unstructured_json_v1", "unstructured_plain_text_v1"]

UNICODE_BULLETS: Final[List[str]] = [
    "\u0095",
    "\u2022",
    "\u2023",
    "\u2043",
    "\u3164",
    "\u204C",
    "\u204D",
    "\u2219",
    "\u25CB",
    "\u25CF",
    "\u25D8",
    "\u25E6",
    "\u2619",
    "\u2765",
    "\u2767",
    "\u29BE",
    "\u29BF",
    "\u002D",
    "",
    r"\*",
    "\x95",
    "·",
]
BULLETS_PATTERN = "|".join(UNICODE_BULLETS)
UNICODE_BULLETS_RE = re.compile(f"(?:{BULLETS_PATTERN})(?!{BULLETS_PATTERN})")
BOLD_PATTERN = r"(\*{1,2})(.*?)\1"
BOLD_PATTERN_RE = re.compile(BOLD_PATTERN)
NEWLINE_PATTERN = r"\n+"
NEWLINE_PATTERN_RE = re.compile(NEWLINE_PATTERN)


def prepare_str(string: Optional[str], standardize_whitespaces: bool = False) -> str:
    if not string:
        return ""
    if standardize_whitespaces:
        return " ".join(string.split())
    return str(string)  # type: ignore


def extract_content_from_file(
    raw_data: str,
) -> Union[str, List[dict]]:
    """
    Extracts content from a file. Parses JSON if possible, otherwise returns the raw data.

    Args:
        raw_data (str): The raw data to process.

    Returns:
        Union[str, List[dict]]: The extracted content.
    """
    try:
        data = json.loads(raw_data)
        return data
    except json.JSONDecodeError:
        return raw_data


def remove_alt_text_from_image_text_field(data_list: List[dict]) -> None:
    """
    Extract alt text from text_as_html and remove the first occurrence from the text field
    for image type dictionaries. Alt text includes non-OCR text that describes the image,
    so it should not be included in text content extraction.

    Args:
        data_list: List of dictionaries to process

    Returns:
        List of dictionaries with first occurrence of alt text removed from text field
    """
    for item in data_list:
        # Check if this is an image type
        if item.get("type") == "Image":
            # Check if we have both text and text_as_html
            if "text" in item and "metadata" in item and "text_as_html" in item["metadata"]:
                html = item["metadata"]["text_as_html"]

                # Extract the alt text from the img tag
                alt_match = re.search(r'alt="([^"]*)"', html)

                if alt_match:
                    alt_text = alt_match.group(1)
                    # Remove only the first occurrence of the alt text
                    item["text"] = item["text"].replace(alt_text, "", 1).strip()


def extract_text(
    input_data: Union[str, List[dict]],
    input_format: InputFormat,
    exclude_elements: Optional[List[str]] = [],
) -> str:
    """
    Extract text content from the input data based on its format.

    Args:
        input_data (Union[str, List[dict]]): Either a plain text string or a list of dicts.
        input_format (InputFormat): The format of the input data.
        exclude_elements (Optional[List[str]]): The elements to exclude from the evaluation
    Returns:
        str: The extracted text.
    """
    if input_format == "v1":
        if isinstance(input_data, list):
            input_format = "unstructured_json_v1"
        elif is_unstructured_plain_text(input_data):
            input_format = "unstructured_plain_text_v1"
        else:
            input_format = "plain_text_v1"

    if input_format == "unstructured_json_v1":
        if not isinstance(input_data, list):
            raise ValueError(
                "Invalid unstructured elements format:\
                expected a list of unstructured elements."
            )
        remove_alt_text_from_image_text_field(input_data)
        texts = []
        for element in input_data:
            if not isinstance(element, dict) or "text" not in element:
                raise ValueError("Each unstructured element must be a dict with a 'text' key.")
            if exclude_elements and element.get("type") in exclude_elements:
                continue
            texts.append(str(element["text"]))
        return " ".join(texts)
    elif input_format == "unstructured_plain_text_v1":
        if not isinstance(input_data, str):
            raise ValueError("Invalid unstructured plain text format: expected a string.")
        return _process_unstructured_plain_text(input_data, exclude_elements)
    elif input_format == "plain_text_v1":
        if not isinstance(input_data, str):
            raise ValueError("Invalid plain text format: expected a string.")
        return input_data


def is_unstructured_plain_text(text: str) -> bool:
    """
    Check if the text is in unstructured plain text format by looking for format markers.
    """
    return "Unstructured Plain Text Format" in text and "---" in text


TAG_PATTERN = re.compile(r"^-+ Unstructured (.*) (Begin|End)$")


def convert_unstructured_plaintext_to_type_text_pairs(
    input_data: str, flatten_all: bool = True, flatten_types: Set[str] = {}
) -> List[dict]:
    """
    Converts unstructured plaintext with tags to a list of type-text pairs.

    For normal processing: nested elements are treated as text content of their parent.
    For flattened types: the parent container is removed and only direct children are kept.

    Args:
        input_data: The input text with unstructured tags
        flatten_all: If True, all elements are flattened.
        If False, only the specified types are flattened. Default is True.
        flatten_types: Set of element types to flatten (e.g., {"Form"})
                      If None, no flattening is performed

    Returns:
        List of dictionaries with "type" and "text" keys

    Example:
        Normal: Form containing Sub-Title -> [{"type": "Form", "text": "...all content..."}]
        Flattened: Form containing Sub-Title -> [{"type": "Sub-Title", "text": "..."}]
    """

    if not input_data.strip():
        return []

    lines = input_data.splitlines()
    result = []

    # State tracking
    text_buffer = []  # Accumulates text lines
    element_type = "Text"  # Current element type being processed
    nesting_depth = 0  # How deep we are in nested tags

    # Flattening state (only used when inside a flattened element)
    is_flattening = False
    flattened_children = []  # Stores child elements when flattening
    child_type = None  # Current child element type
    child_text = []  # Current child element text

    def _finalize_text_element(element_type: str, text_lines: List[str]) -> None:
        """Helper to add a completed element to results"""
        text_content = "\n".join(text_lines).strip()
        if text_content:
            if is_flattening:
                flattened_children.append({"type": element_type, "text": text_content})
            else:
                result.append({"type": element_type, "text": text_content})

    try:
        for line_num, line in enumerate(lines, 1):
            # Skip header lines
            if "Unstructured Plain Text Format" in line:
                continue

            # Check if this line is a tag
            tag_match = TAG_PATTERN.match(line)

            if tag_match:
                tag_type = tag_match.group(1).strip()
                tag_action = tag_match.group(2)  # "Begin" or "End"

                if tag_action == "Begin":
                    if nesting_depth == 0:
                        # Starting a new top-level element

                        # First, save any accumulated text as a "Text" element
                        if text_buffer:
                            _finalize_text_element("Text", text_buffer)
                            text_buffer = []

                        # Set up the new element
                        element_type = tag_type

                        # Check if we should flatten this element type
                        if flatten_all or tag_type in flatten_types:
                            is_flattening = True
                            flattened_children = []
                        else:
                            is_flattening = False

                    elif is_flattening and nesting_depth == 1:
                        # Starting a direct child of a flattened element

                        # Save any loose text in the parent as a separate element
                        if text_buffer:
                            _finalize_text_element(element_type, text_buffer)
                            text_buffer = []

                        # Start collecting the child element
                        child_type = tag_type
                        child_text = []

                    nesting_depth += 1

                elif tag_action == "End":
                    if nesting_depth > 0:
                        nesting_depth -= 1

                        if nesting_depth == 0:
                            # Finished processing the top-level element

                            if is_flattening:
                                # Finalize any current child
                                if child_type and child_text:
                                    _finalize_text_element(child_type, child_text)

                                # Add any remaining loose text in the parent
                                if text_buffer:
                                    _finalize_text_element(element_type, text_buffer)

                                # Add all flattened children to the main result
                                result.extend(flattened_children)

                                # Reset flattening state
                                is_flattening = False
                                flattened_children = []
                                child_type = None
                                child_text = []
                            else:
                                # Normal processing: add the complete element
                                _finalize_text_element(element_type, text_buffer)

                            # Reset for next element
                            text_buffer = []
                            element_type = "Text"

                        elif is_flattening and nesting_depth == 1:
                            # Finished a direct child of a flattened element
                            if child_type and child_text:
                                _finalize_text_element(child_type, child_text)

                            # Reset child tracking
                            child_type = None
                            child_text = []
                    else:
                        # Orphaned End tag - treat as regular text
                        logger.warning(
                            f"Warning: Line {line_num}: "
                            f"Orphaned End tag '{line.strip()}' treated as text"
                        )
                        if is_flattening and child_type:
                            child_text.append(line)
                        else:
                            text_buffer.append(line)

            else:
                # Regular text line - add to appropriate buffer
                if is_flattening and child_type:
                    child_text.append(line)
                else:
                    text_buffer.append(line)

        # Handle any remaining content at end of input
        if is_flattening:
            # Finalize any current child
            if child_type and child_text:
                _finalize_text_element(child_type, child_text)

            # Add any remaining loose text
            if text_buffer:
                _finalize_text_element(element_type, text_buffer)

            # Add all flattened children to result
            result.extend(flattened_children)
        elif text_buffer:
            # Add any remaining text as final element
            _finalize_text_element(element_type, text_buffer)

        # Warn about unclosed tags
        if nesting_depth > 0:
            logger.warning(f"Warning: End of input reached with {nesting_depth} unclosed tags")

    except Exception as e:
        print(f"Error during processing: {e}")
        return []

    return result


def _process_unstructured_plain_text(
    text: str,
    exclude_elements: Optional[List[str]] = [],
) -> str:
    """
    Process unstructured plain text format and extract content based on element types.

    Args:
        text (str): The unstructured plain text content.
        exclude_elements (Optional[List[str]]): Element types to exclude.

    Returns:
        str: The processed text with specified elements excluded.
    """
    type_text_pairs = convert_unstructured_plaintext_to_type_text_pairs(text)
    texts = []
    for pair in type_text_pairs:
        if "type" in pair and "text" in pair:
            if exclude_elements and pair["type"] in exclude_elements:
                continue
            texts.append(pair["text"])
    return " ".join(texts)


def _clean_bullets_and_bold(text: str) -> str:
    """
    Cleans unicode bullets and bold text from a section of text.
    """
    cleaned_text = BOLD_PATTERN_RE.sub(r"\2", text)
    cleaned_text = UNICODE_BULLETS_RE.sub("", cleaned_text)
    return cleaned_text.strip()


def _clean_newlines(text: str) -> str:
    """
    Cleans newlines from a section of text.
    """
    return NEWLINE_PATTERN_RE.sub(" ", text)


def bag_of_tokens(text: str) -> Dict[str, int]:
    """
    Outputs the bag of tokens (BOT) found in the input text and their frequencies.

    Args:
        text (str): The input text to process.

    Returns:
        Dict[str, int]: A dictionary containing the BOT and their frequencies.
    """
    bot: Dict[str, int] = {}

    # Clean up the most common style inconsistencies in generated text.
    cleaned_text = _clean_bullets_and_bold(text)
    # Clean up newlines
    cleaned_text = _clean_newlines(cleaned_text)

    tokenizer = TiktokenCache()
    token_ids = tokenizer.encode(cleaned_text)
    tokens = tokenizer.decode_tokens(tuple(token_ids))

    for token in tokens:
        try:
            # Decode the token and strip whitespace because sometimes extra whitespace is generated
            token = token.decode("utf-8").strip()
            # Skip empty tokens
            if token:
                bot[token] = bot.get(token, 0) + 1

        except UnicodeDecodeError:
            bot[token] = bot.get(token, 0) + 1

    return bot
