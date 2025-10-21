# unstructured-eval-metrics

A functional programming-based framework designed to evaluate and benchmark the performance of document parsing systems such as Unstructured and Visual Language Models (VLMs) using a comprehensive suite of metrics.


## Installation

### Prerequisites

- Python 3.11 or higher
- uv: https://docs.astral.sh/uv/getting-started/installation/

### Setup

1. Install dependencies using `uv`:
```bash
make install
```
This will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

2. Performs a linting check
```bash
make check
```
3. Automatically fixes linting issues
```bash
make tidy
```

## Packages

### Scoring

The `scoring` package implements performance metrics:

- `aggregate_scoring.py`: Generates aggregated scores across multiple datapoints
- `cct_adjustment.py`: Adjusts CCT (Clean Concatenated Text) metrics for fair comparison
- `content_scoring.py`: Measures OCR quality and reading order accuracy
- `element_consistency_scoring.py`: Evaluates consistency of element detection and classification
- `structure_scoring.py`: Assesses document structure and element alignment
- `table_scoring.py`: Evaluates table structure and content extraction accuracy

### Processing

The `processing` package contains data transformation functions:

- `aggregate.py`: Functions for constructing datapoints and aggregating data across documents
- `structure_processing.py`: Document structure processing and element extraction
- `table_processing.py`: Extracting, parsing, and normalizing table structures
- `text_processing.py`: Text extraction, normalization, and tokenization

### Evaluation

The `evaluation` package provides the evaluation orchestration:

- `config_model.py`: Pydantic-based configuration models for validation
- `generate_agg_eval_metrics.py`: Generates aggregate metrics across datasets
- `generate_doc_eval_metrics.py`: Produces document-level evaluation metrics

## Usage

### 📊 Generate Evaluation Metrics

#### To generate document-level evaluation metrics:

```bash
PYTHONPATH=./src python3 -m evaluation.generate_doc_eval_metrics \
  --config configs/all_metrics_config.json \
  --output_csv reports/all_metrics_report.csv
```

#### To generate aggregate evaluation metrics:

```bash
PYTHONPATH=./src python3 -m evaluation.generate_agg_eval_metrics \
  --config configs/all_metrics_config.json \
  --output_csv reports/all_agg_metrics_report.csv
```

### Configuration

Evaluations are configured using JSON files:

```json
{
  "storage": {
      "store_metrics": false,  // Storage is used to store metrics in S3. It is set to false by default.
  },
  "file_naming": {
      "type_of_run": "ci",
      "run_identifier": "simple-data"
  },
  "formats": {
      "sample_formats": {
          "cct": "v1",
          "adjusted_cct": "v1",
          "percent_tokens_found": "v1",
          "percent_tokens_added": "v1",
          "tables": "html",
          "element_alignment": "v1",
          "element_consistency": "v1"
      },
      "ground_truth_formats": {
          "cct": "v1",
          "adjusted_cct": "v1",
          "percent_tokens_found": "v1",
          "percent_tokens_added": "v1",
          "tables": "text",
          "element_alignment": "v1",
          "element_consistency": "v1"
      }
  },
  "datasets": [
      {
          "name": "simple-data",
          "metrics": [
              "cct",
              "adjusted_cct",
              "percent_tokens_found",
              "percent_tokens_added",
              "element_alignment",
              "element_consistency"
          ],
          "to_evaluate_dir": "./simple-data/od-first-prediction",
          "ground_truth_dir": "./simple-data/cct-gt"
      },
      {
          "name": "simple-data",
          "metrics": [
              "tables"
          ],
          "to_evaluate_dir": "./simple-data/od-first-prediction",
          "ground_truth_dir": "./simple-data/table-gt"
      }
  ]
}
```


## License

This project is licensed under the [CC-BY-NC-SA 4.0](./LICENSE).