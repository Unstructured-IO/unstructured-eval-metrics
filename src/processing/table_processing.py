import difflib
import logging
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Optional

import distance
import numpy as np
import pandas as pd
from apted import APTED, Config
from apted.helpers import Tree
from bs4 import BeautifulSoup

from scoring.content_scoring import calculate_edit_distance

logger = logging.getLogger(__name__)


def group_contents(
    df: pd.DataFrame, group_col: str, content_col: str, new_col_name: str
) -> pd.DataFrame:
    """
    Group DataFrame by a column and join content into a single string.

    Args:
        df: DataFrame to group
        group_col: Column to group by
        content_col: Column containing content to join
        new_col_name: Name for the new column containing joined content

    Returns:
        Grouped DataFrame with joined content
    """
    grouped = (
        df.groupby(group_col)[content_col].apply(lambda x: " ".join(map(str, x))).reset_index()
    )
    grouped.rename(columns={content_col: new_col_name}, inplace=True)
    return grouped


def apply_shift_to_grouped_df(df: pd.DataFrame, group_col: str, shift: int) -> pd.DataFrame:
    """
    Apply a shift to the group column values.

    Args:
        df: DataFrame to apply shift to
        group_col: Column to shift values in
        shift: Amount to shift by

    Returns:
        DataFrame with shifted group column values
    """
    if shift == 0:
        return df.copy()

    shifted_df = df.copy()
    try:
        # Convert group_col values to int, add shift, then filter
        shifted_df[group_col] = shifted_df[group_col].astype(int) + shift
        return shifted_df
    except Exception as e:
        logger.warning(f"Error applying shift to {group_col} with shift {shift}: {e}")
        return df.copy()


def check_first_row_score(
    actual_grouped: pd.DataFrame,
    pred_grouped: pd.DataFrame,
    merge_col_name: str,
    cutoff: Optional[float] = 0.8,
) -> bool:
    """
    Check if the first row has high score matches to determine if we should try shifts.

    Args:
        actual_grouped: Grouped actual DataFrame
        pred_grouped: Grouped predicted DataFrame
        merge_col_name: Name of the merged column
        cutoff: Optional. Score threshold for determining if shifts should be tried

    Returns:
        True if shifts should be tried, False otherwise
    """
    # Filter for first row (row_index 0)
    first_row_actual = actual_grouped.head(1)
    first_row_pred = pred_grouped.head(1)

    first_row_score = calculate_edit_distance(
        str(first_row_actual[f"{merge_col_name}_df1"]),
        str(first_row_pred[f"{merge_col_name}_df2"]),
        return_as="score",
    )
    should_try_shifts = first_row_score < cutoff
    return should_try_shifts


def calculate_shift_scores(
    actual_grouped: pd.DataFrame,
    pred_grouped: pd.DataFrame,
    group_col: str,
    merge_col_name: str,
    shifts_to_try: List[int],
) -> List[float]:
    """
    Calculate scores for different shifts and return the list of scores.

    Args:
        actual_grouped: Grouped actual DataFrame
        pred_grouped: Grouped predicted DataFrame
        group_col: Column to group by
        merge_col_name: Name of the merged column
        shifts_to_try: List of shifts to try

    Returns:
        List of scores for each shift
    """
    scores = []
    for shift in shifts_to_try:
        # Apply shift to actual_grouped
        pred_shifted = apply_shift_to_grouped_df(pred_grouped, group_col, shift)

        actual_grouped[group_col] = actual_grouped[group_col].astype(str)
        pred_shifted[group_col] = pred_shifted[group_col].astype(str)

        # Merge the two grouped DataFrames on the grouping column.
        merged = pd.merge(actual_grouped, pred_shifted, on=group_col, how="left")

        # Replace missing values in prediction column with empty string
        merged[f"{merge_col_name}_df2"] = merged[f"{merge_col_name}_df2"].fillna("")

        # Compute the edit distance for each merged row.
        merged[f"{group_col}_edit_distance"] = merged.apply(
            lambda row: calculate_edit_distance(
                str(row[f"{merge_col_name}_df1"]),
                str(row[f"{merge_col_name}_df2"]),
                return_as="score",
            ),
            axis=1,
        )
        if merged.empty:
            score = 0.0
        else:
            score = merged[f"{group_col}_edit_distance"].mean()

        # Calculate average edit distance for this shift
        scores.append(score)

    return scores


EMPTY_CELL = MappingProxyType(
    {
        "row_index": "",
        "col_index": "",
        "content": "",
    }
)


# Refactoring of TEDS (Tree-Edit-Distance-based Similarity)
# https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src
class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = []

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = (
                f'"tag": {self.tag}, "colspan": {self.colspan}, '
                + f'"rowspan": {self.rowspan}, "text": {self.content}'
            )
        else:
            result = f'"tag": {self.tag}'

        result += "".join(child.bracket() for child in self.children)

        return f"{{ {result} }}"


class TEDSCustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value from a collection of sequences"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1 using normalised levenshtein distance"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees and returns a distance score from 0 to 1"""
        # If tags, colspan or rowspan are different, consider them different
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0

        # For a cell, compare the content using a normalised levenshtein distance
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance based Similarity"""

    def __init__(self):
        self.__tokens__ = []

    def _assign_cells_to_rows(self, cells):
        rows = {}

        for cell in cells:
            row_index = cell.get("row_index")
            if row_index not in rows:
                rows[row_index] = []
            rows[row_index].append(cell)

        # sort row cells
        for row in rows.values():
            row.sort(key=lambda x: x["col_index"])

        return rows

    def _get_tree_from_cell(self, cells):
        """Converts a table cell to a tree structure"""
        # Create a root node
        root = TableTree("table")
        n_nodes = 1

        rows = self._assign_cells_to_rows(cells)

        for _, row in sorted(rows.items(), key=lambda x: x[0]):
            tree = TableTree("tr")
            n_nodes += 1

            for cell in row:
                cell_tree = TableTree(
                    "td",
                    colspan=cell.get("colspan"),
                    rowspan=cell.get("rowspan"),
                    content=cell.get("content"),
                )
                tree.children.append(cell_tree)
                n_nodes += 1

            root.children.append(tree)

        return root, n_nodes

    def _get_max_distance(self, tree_pred, tree_gt):
        """Calculate the max distance between two trees"""
        """Trees, based on the cells data structure, will have a root node,
        rows nodes and there are column cells per row. This function tries
        to count the maximum number of possible changes that is used to
        normalise the distance.

        The tree algorithm has three operations, insertion, deletion and renaming
        (change the value of the node). For our trees, there does not seem to be
        a chance for deletion, and only insertion and renaming will apply. This
        means that most operations will remain based on the structure that we
        have for tables, but for additional rows we need to add the row node in
        addition to the column cells for that row. To find the number of additional
        rows, the difference between the number of rows is needed."""
        if len(tree_pred.children) > len(tree_gt.children):
            tree1 = tree_pred
            tree2 = tree_gt
        else:
            tree1 = tree_gt
            tree2 = tree_pred

        total_cells = 0

        for row_id in range(len(tree1.children)):
            if len(tree2.children) > row_id:
                tree_2_cols = len(tree2.children[row_id].children)
            else:
                tree_2_cols = 0

            total_cells += max(len(tree1.children[row_id].children), tree_2_cols)

        # td tags and potential tr tags to add
        return total_cells + (len(tree1.children) - len(tree2.children))

    def score(self, pred_cells, gt_cells):
        """Computes TEDS score between prediction and ground truth table"""
        if not pred_cells or not gt_cells:
            return 0.0

        tree_pred, n_nodes_pred = self._get_tree_from_cell(pred_cells)
        tree_gt, n_nodes_gt = self._get_tree_from_cell(gt_cells)

        max_distance = max(n_nodes_pred, n_nodes_gt)

        max_distance_corrected = self._get_max_distance(tree_pred, tree_gt)

        distance = APTED(tree_pred, tree_gt, TEDSCustomConfig()).compute_edit_distance()

        return max(1.0 - (float(distance) / max_distance), 0.0), 1.0 - (
            float(distance) / max_distance_corrected
        )


def _extract_cells_from_text_as_html(element: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    """
    Extract and parse cells from "text_as_html" field in Element structure

    Args:
        element: Example element:
        {
            "type": "Table",
            "metadata": {
                "text_as_html": "<table>
                                    <thead>
                                        <tr>
                                            <th>Month A.</th>
                                        </tr>
                                    </thead>
                                    </tbody>
                                        <tr>
                                            <td>22</td>
                                        </tr>
                                    </tbody>
                                </table>"
            }
        }

    Returns:
        List of extracted cells in a format:
        [
            {
                "row_index": 0,
                "col_index": 0,
                "content": "Month A.",
            },
            ...,
        ]
    """
    html_content = element.get("metadata", {}).get("text_as_html")
    if not html_content:
        return None

    converted_cells = None
    try:
        converted_cells = _convert_table_from_html(html_content)
    except (AttributeError, ValueError) as e:
        logger.warning(f"Error parsing HTML table structure: {e}")
    except BeautifulSoup.ParserError as e:
        logger.warning(f"HTML parsing failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error converting HTML table data: {e}")

    return converted_cells


def _extract_cells_from_table_as_cells(element: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and parse grid cells from "table_as_cells" field in Element structure

    Args:
        element: Example element:
        {
            "type": "Table",
            "metadata": {
                "table_as_cells": [{"x": 0, "y": 0, "w": 1, "h": 1, "content": "Month A."},
                                   {"x": 0, "y": 1, "w": 1, "h": 1, "content": "22"}]
            }
        }

    Returns:
        List of extracted cells in a format:
        [
            {
                "row_index": 0,
                "col_index": 0,
                "content": "Month A.",
            },
            ...,
        ]
    """
    grid_cells = element.get("metadata", {}).get("table_as_cells", [])
    if not grid_cells:
        return None
    converted_cells = None
    if grid_cells:
        converted_cells = _convert_table_from_grid_cells(grid_cells)
    return converted_cells


def _extract_cells_from_text(element: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and parse grid cells from "text" field in Element structure

    Note:
    - Unstructured's ground truth data is stored in this (Deckerd) format.

    Args:
        element: Example element:
        {
            "type": "Table",
            "text": [
                {"x": 0, "y": 0, "w": 1, "h": 1, "content": "Month A."},
                {"x": 0, "y": 1, "w": 1, "h": 1, "content": "22"}
            ]
        }

    Returns:
        List of extracted cells in a format:
        [
            {
                "row_index": 0,
                "col_index": 0,
                "content": "Month A.",
            },
            ...,
        ]
    """
    grid_cells = element.get("text", "")
    if not grid_cells or not isinstance(grid_cells, list):
        return None

    converted_cells = None
    try:
        converted_cells = _convert_table_from_grid_cells(grid_cells)
    except KeyError as e:
        logger.warning(f"Missing required cell attribute {e} in grid cells")
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid cell data format in grid cells: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error converting grid cell data: {e}")

    return converted_cells


def _convert_table_from_html(content: str) -> List[Dict[str, Any]]:
    """Convert html format to table structure. As a middle step it converts
    html to the Deckerd format as it's more convenient to work with.

    Args:
        content: The html content with a table to extract.

    Returns:
        A list of dictionaries where each dictionary represents a cell in the table.
    """
    grid_cells = _html_table_to_grid_cells(content)
    return _convert_table_from_grid_cells(grid_cells)


def _convert_table_from_grid_cells(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert list of cells format to table structure.

    Args:
      content: The Deckard formatted content with a table to extract.

    Returns:
      A list of dictionaries where each dictionary represents a cell in the table.
    """
    table_data = []
    for table in content:
        try:
            cell_data = {
                "row_index": table["y"],
                "col_index": table["x"],
                "colspan": table["w"],
                "rowspan": table["h"],
                "content": table["content"],
            }
        except KeyError:
            cell_data = EMPTY_CELL
        except TypeError:
            cell_data = EMPTY_CELL
        table_data.append(cell_data)
    return table_data


def _sort_table_cells(table_data: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    return sorted(table_data, key=lambda cell: (cell["row_index"], cell["col_index"]))


def _move_cells_for_spanned_cells(cells: List[Dict[str, Any]]):
    """Move cells to the right if spanned cells have an influence on the rendering.

    Args:
        cells: List of cells in the table in Deckerd format.

    Returns:
        List of cells in the table in Deckerd format with cells moved to the right if spanned.
    """
    sorted_cells = sorted(cells, key=lambda x: (x["y"], x["x"]))
    cells_occupied_by_spanned = set()
    for cell in sorted_cells:
        if cell["w"] > 1 or cell["h"] > 1:
            for i in range(cell["y"], cell["y"] + cell["h"]):
                for j in range(cell["x"], cell["x"] + cell["w"]):
                    if (i, j) != (cell["y"], cell["x"]):
                        cells_occupied_by_spanned.add((i, j))
        while (cell["y"], cell["x"]) in cells_occupied_by_spanned:
            cell_y, cell_x = cell["y"], cell["x"]
            cells_to_the_right = [c for c in sorted_cells if c["y"] == cell_y and c["x"] >= cell_x]
            for cell_to_move in cells_to_the_right:
                cell_to_move["x"] += 1
            cells_occupied_by_spanned.remove((cell_y, cell_x))
    return sorted_cells


def _html_table_to_grid_cells(content: str) -> List[Dict[str, Any]]:
    """Convert html format to grid cells table structure.

    Args:
        content: The html content with a table to extract.

    Returns:
        A list of dictionaries where each dictionary represents a grid cell in the table.

    Note:
        - grid cells are used as a middle step to handle row and
          col spans in converting the html.
    """

    soup = BeautifulSoup(content, "html.parser")
    table = soup.find("table")
    rows = table.find_all(["tr"])
    table_data = []

    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        for j, cell_data in enumerate(cells):
            cell = {
                "y": i,
                "x": j,
                "w": int(cell_data.attrs.get("colspan", 1)),
                "h": int(cell_data.attrs.get("rowspan", 1)),
                "content": cell_data.text,
            }
            table_data.append(cell)
    return _move_cells_for_spanned_cells(table_data)


TABLE_FORMAT_TO_EXTRACTION_STRATEGIES = {
    "html": _extract_cells_from_text_as_html,
    "cells": _extract_cells_from_table_as_cells,
    "text": _extract_cells_from_text,
}


def extract_and_convert_tables(
    input_tables: List[Dict[str, Any]],
    table_format: Literal["html", "cells"],
    merged_consecutive_tables: bool = False,
    exclude_elements: Optional[List[str]] = None,
) -> List[List[Dict[str, Any]]]:
    """Extract and convert table data to a structured format based on the table format.

    Args:
        input_tables: List of table elements from the file.
        table_format: String indicating the input format, either 'cells' or 'html' or 'text'.
                    'cells' refers to the grid cells in the 'table_as_cells' field.
                    'html' refers to the HTML representation of the table extracted
                    from the 'text_as_html' field.
                    'text' refers to the text representation of the table extracted
        merged_consecutive_tables: Feature flag to indicate whether to merge consecutive tables.
    Returns:
        A list of tables, where each table is represented as a list of indexed cell dictionaries.

    """

    def _tables_have_matching_columns(table1, table2):
        """Check if two tables have the same column count."""
        if not table1 or not table2:
            return False
        col_count1 = max(cell["col_index"] for cell in table1)
        col_count2 = max(cell["col_index"] for cell in table2)
        return col_count1 == col_count2

    def _merge_table(base_table, new_table):
        """Merge new_table into base_table with updated row indices."""
        if not base_table:
            return new_table.copy()

        # Calculate row offset based on the max row index in base_table
        row_offset = max(cell["row_index"] for cell in base_table) + 1

        # Create merged table with adjusted row indices
        merged_table = base_table.copy()
        for cell in new_table:
            adjusted_cell = cell.copy()
            adjusted_cell["row_index"] += row_offset
            merged_table.append(adjusted_cell)

        return merged_table

    def _get_extraction_and_fallback_fn(table_format):

        if table_format not in TABLE_FORMAT_TO_EXTRACTION_STRATEGIES:
            raise ValueError(
                f"table_format {table_format} is not valid. "
                'Valid formats are "html", "cells", and "text"'
            )

        extract_cells_fn = TABLE_FORMAT_TO_EXTRACTION_STRATEGIES[table_format]
        fallback_extract_cells_fn = (
            TABLE_FORMAT_TO_EXTRACTION_STRATEGIES["cells"]
            if table_format == "cells"
            else TABLE_FORMAT_TO_EXTRACTION_STRATEGIES["html"]
        )
        return extract_cells_fn, fallback_extract_cells_fn

    extract_fn, fallback_extract_fn = _get_extraction_and_fallback_fn(table_format)

    table_data = []
    previous_was_table = False

    for element in input_tables:
        if element.get("type") == "Table":
            extracted_cells = extract_fn(element)
            if not extracted_cells:
                extracted_cells = fallback_extract_fn(element)
            if not extracted_cells:
                previous_was_table = False
                continue

            sorted_cells = _sort_table_cells(extracted_cells)

            if not merged_consecutive_tables:
                table_data.append(sorted_cells)
            else:
                if previous_was_table:
                    # Previous element was a table - try to merge with the most recent table
                    previous_table = table_data[-1]
                    if _tables_have_matching_columns(previous_table, sorted_cells):
                        # Merge and replace the last table in table_data
                        merged_table = _merge_table(previous_table, sorted_cells)
                        table_data[-1] = merged_table
                    else:
                        # Can't merge, add as a new table
                        table_data.append(sorted_cells)
                else:
                    # Previous element wasn't a table, add as a new table
                    table_data.append(sorted_cells)

            previous_was_table = True
        else:
            previous_was_table = False

    return table_data


class TableAlignment:
    def __init__(self, cutoff: float = 0.8):
        self.cutoff = cutoff

    @staticmethod
    def _compare_contents_as_df(
        actual_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        cutoff: Optional[float] = 0.8,
        adjust_by_shifts: Optional[bool] = False,
        shifts_to_try: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compare two DataFrames where each row contains (row_index, col_index, value).

        We noticed that when there are shifts in the content, such as one row or column being displaced,
        it results in very low cell_level_content_acc score (e.g., close to 0)
        adjust_by_shifts and shifts_to_try are used to adjust the indices by [-3, -2, -1, 0, 1, 2, 3]
        and find the max score. The new metric aligns with TEDS better.
        This adjustment is required because around 10–20% of tables experience this issue.

        Parameters:
        actual_df, pred_df (pandas.DataFrame): DataFrames with columns [row_index, col_index, value]
        cutoff: The cutoff value for the close matches.
        adjust_by_shifts: Whether to adjust the content accuracy by shifting the indices.
        shifts_to_try: The shifts to try to adjust the content accuracy.

        Returns:
        dict: Contains row-wise and column-wise differences
        """  # noqa: E501

        def _group_and_merge_edit_distance(
            actual_df: pd.DataFrame,
            pred_df: pd.DataFrame,
            group_col: str,
            content_col: str = "content",
            merge_col_name: str = "full_content",
            cutoff: Optional[float] = 0.8,
            adjust_by_shifts: Optional[bool] = False,
            shifts_to_try: Optional[List[int]] = None,
        ) -> float:
            """
            Group both DataFrames by a given column, merge the 'content' strings,
            and compute the average edit distance.
            """
            actual_grouped = group_contents(
                actual_df, group_col, content_col, f"{merge_col_name}_df1"
            )
            pred_grouped = group_contents(pred_df, group_col, content_col, f"{merge_col_name}_df2")

            # Check if first row has high score matches
            should_try_shifts = adjust_by_shifts and check_first_row_score(
                actual_grouped, pred_grouped, merge_col_name, cutoff
            )

            if should_try_shifts:
                shifts_to_try = shifts_to_try
            else:
                shifts_to_try = [0]

            scores = calculate_shift_scores(
                actual_grouped, pred_grouped, group_col, merge_col_name, shifts_to_try
            )

            return max(scores)

        average_row_score = _group_and_merge_edit_distance(
            actual_df,
            pred_df,
            group_col="row_index",
            merge_col_name="full_content",
            cutoff=cutoff,
            adjust_by_shifts=adjust_by_shifts,
            shifts_to_try=shifts_to_try,
        )

        average_col_score = _group_and_merge_edit_distance(
            actual_df,
            pred_df,
            group_col="col_index",
            merge_col_name="full_content",
            cutoff=cutoff,
            adjust_by_shifts=adjust_by_shifts,
            shifts_to_try=shifts_to_try,
        )

        return {
            "by_col_score": round(average_col_score, 2),
            "by_row_score": round(average_row_score, 2),
        }

    @staticmethod
    def get_table_level_alignment(
        predicted_table_data: List[List[Dict[str, Any]]],
        ground_truth_table_data: List[List[Dict[str, Any]]],
    ) -> List[int]:
        """Compares predicted table data with ground truth data to find the best
        matching table index for each predicted table.

        Args:
          predicted_table_data: A list of predicted tables.
          ground_truth_table_data: A list of ground truth tables.

        Returns:
          A list of indices indicating the best match in the ground truth for
          each predicted table.

        """

        def _get_content_in_tables(table_data: List[List[Dict[str, Any]]]) -> List[str]:
            # Replace below docstring with google-style docstring
            """Extracts and concatenates the content of cells from each table in a list of tables.

            Args:
            table_data: A list of tables, each table being a list of cell data dictionaries.

            Returns:
            List of strings where each string represents the concatenated content of one table.
            """
            return [" ".join([d["content"] for d in td if "content" in d]) for td in table_data]

        ground_truth_texts = _get_content_in_tables(ground_truth_table_data)

        matched_indices = []
        for td in predicted_table_data:
            reference = _get_content_in_tables([td])[0]
            matches = difflib.get_close_matches(reference, ground_truth_texts, cutoff=0.1, n=1)
            matched_indices.append(ground_truth_texts.index(matches[0]) if matches else -1)

        return matched_indices

    @staticmethod
    def get_cell_level_alignment(
        predicted_table_data: List[List[Dict[str, Any]]],
        ground_truth_table_data: List[List[Dict[str, Any]]],
        matched_indices: List[int],
        cutoff: Optional[float] = 0.8,
        shifts_to_try: Optional[List[int]] = [-3, -2, -1, 0, 1, 2, 3],
    ) -> Dict[str, float]:
        """
        Rule-based algorithm to examine the alignment at the cell level.
        This algorithm provides a transparent way to understand the alignment
        of the predicted tables with the ground truth tables.

        Args:
          predicted_table_data: A list of predicted tables.
          ground_truth_table_data: A list of ground truth tables.
          matched_indices: Indices of the best matching ground truth table for each predicted table.
          shifts_to_try: The shifts to try to adjust the content accuracy.

        Returns:
          A dictionary with column and row alignment accuracies and graphical-based metrics.
        """

        def _add_zero_metrics(metrics: Dict[str, List[float]]) -> None:
            """Add zero values to all metrics for unmatched tables."""
            for key in metrics:
                metrics[key].append(0)

        def _calculate_content_accuracy(
            predicted_table: List[Dict[str, Any]],
            ground_truth_table: List[Dict[str, Any]],
            cutoff: Optional[float] = 0.8,
            adjust_by_shifts: Optional[bool] = False,
            shifts_to_try: Optional[List[int]] = None,
        ) -> Dict[str, float]:
            """
            Calculate content accuracy using dataframe comparison.
            Shifts can be enabled to adjust the indices to align the content.
            """
            predict_table_df = _zip_to_dataframe(predicted_table)
            ground_truth_table_df = _zip_to_dataframe(ground_truth_table)

            return TableAlignment._compare_contents_as_df(
                ground_truth_table_df.fillna(""),
                predict_table_df.fillna(""),
                cutoff=cutoff,
                adjust_by_shifts=adjust_by_shifts,
                shifts_to_try=shifts_to_try,
            )

        def _calculate_index_accuracy(
            predicted_table: List[Dict[str, Any]],
            ground_truth_table: List[Dict[str, Any]],
            cutoff: float,
        ) -> Dict[str, float]:
            """
            Calculate row and column index accuracy between predicted and ground truth tables.
            """
            ground_truth_contents = [item["content"].lower() for item in ground_truth_table]
            used_indices = set()
            matches = []

            # Find content matches between predicted and ground truth cells
            for pred_cell in predicted_table:
                content = pred_cell["content"].lower()
                row_idx = pred_cell["row_index"]
                col_idx = pred_cell["col_index"]

                close_matches = difflib.get_close_matches(
                    content,
                    ground_truth_contents,
                    cutoff=cutoff,
                    n=1,
                )

                if not close_matches:
                    continue

                # Find matching indices that haven't been used yet
                matching_gt_indices = [
                    i
                    for i, gt_content in enumerate(ground_truth_contents)
                    if gt_content == close_matches[0] and i not in used_indices
                ]

                # If all potential matches have been used, reset and find the first match
                if not matching_gt_indices:
                    used_indices.clear()
                    matching_gt_indices = [
                        i
                        for i, gt_content in enumerate(ground_truth_contents)
                        if gt_content == close_matches[0]
                    ]

                if matching_gt_indices:
                    match_idx = matching_gt_indices[0]
                    used_indices.add(match_idx)
                    gt_cell = ground_truth_table[match_idx]
                    matches.append(
                        ((row_idx, col_idx), (gt_cell["row_index"], gt_cell["col_index"]))
                    )

            # Calculate accuracy metrics
            aligned_rows = sum(1 for pred, gt in matches if pred[0] == gt[0])
            aligned_cols = sum(1 for pred, gt in matches if pred[1] == gt[1])
            total_matches = len(matches)

            if total_matches == 0:
                return {"row_accuracy": 0, "col_accuracy": 0}

            # Clamp the score to [0, 1]
            row_accuracy = max(0, min(aligned_rows / total_matches, 1))
            col_accuracy = max(0, min(aligned_cols / total_matches, 1))
            return {"row_accuracy": round(row_accuracy, 2), "col_accuracy": round(col_accuracy, 2)}

        def _zip_to_dataframe(table_data: List[Dict[str, Any]]) -> pd.DataFrame:
            df = pd.DataFrame(table_data, columns=["row_index", "col_index", "content"])
            df = df.set_index("row_index")
            df["col_index"] = df["col_index"].astype(str)
            return df

        metrics = {
            "col_content_acc": [],
            "row_content_acc": [],
            "shifted_col_content_acc": [],
            "shifted_row_content_acc": [],
            "col_index_acc": [],
            "row_index_acc": [],
            "table_teds": [],
            "table_teds_corrected": [],
        }

        # Process matched tables
        for idx, predicted_table in zip(matched_indices, predicted_table_data):
            if idx == -1:
                # Add zeros for unmatched tables
                _add_zero_metrics(metrics)
                continue

            ground_truth_table = ground_truth_table_data[idx]

            # Calculate content accuracy using dataframes
            content_metrics = _calculate_content_accuracy(predicted_table, ground_truth_table)
            # Calculate adjusted content accuracy by shifting the indices
            adjust_content_metrics = _calculate_content_accuracy(
                predicted_table,
                ground_truth_table,
                cutoff=cutoff,
                adjust_by_shifts=True,
                shifts_to_try=shifts_to_try,
            )

            metrics["col_content_acc"].append(content_metrics["by_col_score"])
            metrics["row_content_acc"].append(content_metrics["by_row_score"])
            metrics["shifted_col_content_acc"].append(adjust_content_metrics["by_col_score"])
            metrics["shifted_row_content_acc"].append(adjust_content_metrics["by_row_score"])

            # Calculate index accuracy
            index_metrics = _calculate_index_accuracy(predicted_table, ground_truth_table, cutoff)
            metrics["col_index_acc"].append(index_metrics["col_accuracy"])
            metrics["row_index_acc"].append(index_metrics["row_accuracy"])

            # Calculate teds accuracy
            teds_score, teds_score_corrected = TEDS().score(predicted_table, ground_truth_table)
            teds_score = round(teds_score, 2)
            teds_score_corrected = round(teds_score_corrected, 2)

            metrics["table_teds"].append(teds_score)
            metrics["table_teds_corrected"].append(teds_score_corrected)

        # Handle ground truth tables with no matches
        unmatched_gt_count = len(ground_truth_table_data) - len(set(matched_indices) - {-1})
        for _ in range(unmatched_gt_count):
            _add_zero_metrics(metrics)

        # Calculate final averages
        return {
            "col_index_acc": round(np.mean(metrics["col_index_acc"]), 2),
            "row_index_acc": round(np.mean(metrics["row_index_acc"]), 2),
            "col_content_acc": round(np.mean(metrics["col_content_acc"]), 2),
            "row_content_acc": round(np.mean(metrics["row_content_acc"]), 2),
            "shifted_col_content_acc": round(np.mean(metrics["shifted_col_content_acc"]), 2),
            "shifted_row_content_acc": round(np.mean(metrics["shifted_row_content_acc"]), 2),
            "table_teds": round(np.mean(metrics["table_teds"]), 2),
            "table_teds_corrected": round(np.mean(metrics["table_teds_corrected"]), 2),
        }
