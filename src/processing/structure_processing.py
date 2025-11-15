from typing import List, Union

from processing.text_processing import (
    convert_unstructured_plaintext_to_type_text_pairs,
    is_unstructured_plain_text,
)


def parse_elements(input_data: Union[str, List[dict]], input_format: str) -> List[str]:
    """
    Parse element types from input data based on specified format.

    Args:
        input_data: Either a string (for text formats) or list of dicts (for JSON)
        input_format: Format specification ('v1', 'unstructured_plain_text_v1', etc.)

    Returns:
        List of element types extracted from the input
    """
    return _parse_element_from_json_format(parse_structure(input_data, input_format))


def parse_structure(input_data: Union[str, List[dict]], input_format: str) -> List[dict]:
    """
    Extract elements from input data based on specified format.

    Args:
        input_data: Either a string (for text formats) or list of dicts (for JSON)
        input_format: Format specification ('v1', 'unstructured_plain_text_v1', etc.)

    Returns:
        List of elements extracted from the input
    """
    if input_format == "v1":
        if isinstance(input_data, str):
            if not is_unstructured_plain_text(input_data):
                raise ValueError(
                    "Invalid unstructured plain text format: "
                    "expected a string with Unstructured format markers."
                )
            return convert_unstructured_plaintext_to_type_text_pairs(input_data)
        elif isinstance(input_data, list):
            if not all(isinstance(elem, dict) for elem in input_data):
                raise ValueError("Invalid JSON format: expected a list of dictionaries.")
            return input_data
    elif input_format == "unstructured_plain_text_v1":
        if not isinstance(input_data, str):
            raise ValueError("Invalid unstructured plain text format: expected a string.")
        if not is_unstructured_plain_text(input_data):
            raise ValueError(
                "Invalid unstructured plain text format: "
                "expected a string with Unstructured format markers."
            )
        return convert_unstructured_plaintext_to_type_text_pairs(input_data)
    elif input_format == "plain_text_v1":
        if not isinstance(input_data, str):
            raise ValueError("Invalid plain text format: expected a string.")
        # For plain text, create a single element with type "text"
        return [{"type": "text", "text": input_data}]
    else:
        raise ValueError(f"Invalid input format: {input_format}")


def _convert_unstructured_plaintext_to_elements(input_data: str) -> List[str]:
    """
    Convert unstructured plain text to elements.
    """
    type_text_pairs = convert_unstructured_plaintext_to_type_text_pairs(input_data)
    return [pair["type"] for pair in type_text_pairs]


def _parse_element_from_json_format(data: List[dict], key: str = "type") -> List[str]:
    """
    Parse element types from JSON format data.

    Args:
        data: List of dictionaries containing element information
        key: Key to extract element type from each dictionary

    Returns:
        List of element types

    Raises:
        ValueError: If the specified key is not found in any element
    """
    elements = []
    for elem in data:
        if key not in elem:
            raise ValueError(f"Key {key} not found in {elem}")
        else:
            elements.append(elem[key])
    return elements
