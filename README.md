# Unstructured Evaluation Metrics (SCORE Framework)

A comprehensive evaluation framework designed to benchmark the performance of document 
parsing systems, such as [Unstructured](https://unstructured.io/), that leverage Visual Language Models (VLMs).
[SCORE (Structural and Content Robust Evaluation)](https://arxiv.org/pdf/2509.19345) is a new, 
**interpretation-agnostic** framework that addresses the limitations of applying traditional, 
deterministic metrics (like CER, WER, or TEDS) to generative document parsing. This repository 
illustrates how SCORE can be applied in practice to measure the quality of the outputs of generative document parsers.

| [SCORE paper](https://arxiv.org/pdf/2509.19345) | [Blog: "Benchmarking Document Parsing (and What Actually Matters)"](https://unstructured.io/blog/benchmarking-document-parsing-and-what-actually-matters) |

## Overview

Document parsing is the process of extracting structured information from unstructured documents. 
The emergence of multi-modal generative systems (like GPT-5 Mini or Gemini 2.5 Flash) has made traditional evaluation 
inadequate, as these systems often produce semantically correct yet structurally divergent outputs. 
Conventional metrics misclassify this diversity as an error, penalizing valid interpretations.
The SCORE framework is designed to embrace this representational diversity while enforcing semantic rigor by advancing 
evaluation in four critical dimensions:

1. **Adjusted Edit Distance:** Robust content fidelity that tolerates structural reorganization.  
2. **Token-Level Diagnostics:** Separation of content omissions from hallucinations.  
3. **Table Evaluation:** Semantic alignment with spatial tolerance for structural variations.  
4. **Hierarchy-Aware Consistency:** Assessment of document structure understanding.

This repository provides:

* A collection of document examples with labeled ground truth
* A suite of evaluation metrics for assessing parsing quality
* Scripts for calculating and aggregating these metrics
* A framework for benchmarking document parsing systems

The SCORE framework goes beyond simple text extraction metrics to evaluate structural understanding, element classification, 
table extraction, content organization, and hallucinations.

## Repository Structure

```
unstructured-eval-metrics/
├── configs/                     # Configuration files for evaluation
│   └── all_metrics_config.json  # Example configuration for all metrics
├── simple-data/                 # Example documents and ground truth
│   ├── cct-gt/                  # Ground truth for text content evaluation
│   ├── od-first-prediction/     # Example prediction outputs
│   ├── src/                     # Example PDFs
│   └── table-gt/                # Ground truth for table evaluation
└── src/                         # Source code
    ├── common/                  # Common utilities
    ├── evaluation/              # Evaluation orchestration
    ├── persist/                 # Data persistence utilities
    ├── processing/              # Data transformation functions
    ├── scoring/                 # Performance metrics implementation
    └── utils/                   # General utility functions
```

## Data 

The Unstructured internal evaluation dataset, designed to measure performance on real-world document complexity, includes diverse 
formats such as scanned invoices, technical manuals, and financial reports—cases that differentiate production-grade systems 
from research prototypes. However, a significant portion of the internal evaluation dataset consists of proprietary enterprise 
documents and cannot be publicly distributed.

To ensure transparency and reproducibility, we provide a few representative labeled examples in the `simple-data/` directory: 
* The original PDF files can be found under `simple-data/src`.
* The sample Unstructured outputs are located under `simple-data/od-first-prediction`.
* Ground truth labels are under `simple-data/cct-gt` and `simple-data/table-gt`. 

You can use these examples to:
* Understand the evaluation methodology and data formats used to calculate SCORE metrics.
* Label your own documents following the same conventions.
* Run your own benchmarks and compare results using your data.

This approach allows you to replicate SCORE evaluation process and assess performance on datasets that reflect your own 
domain.

### Ground Truth Format

The repository uses the following types of ground truth data:

1. **Text Content Ground Truth** (in `simple-data/cct-gt` directory):
   - Files use the format: `{original_filename}__uns-plaintext-v1.0.0__{id}.txt`
   - Content is structured with markers for different document elements, enabling evaluation against a clean concatenated text representation (CCT).
     ```
     --------------------------------------------------- Unstructured Plain Text Format 1

     --------------------------------------------------- Unstructured Title Begin
     DOCUMENT TITLE
     --------------------------------------------------- Unstructured Title End

     --------------------------------------------------- Unstructured NarrativeText Begin
     Document content...
     --------------------------------------------------- Unstructured NarrativeText End
     ```

2. **Table Ground Truth** (in `simple-data/table-gt` directory):
   - Files use the format: `{original_filename}__uns-table-json-v1.0.0__{id}.json`
   - Tables are represented as JSON with cell coordinates and content, serving as the ground truth for our format-agnostic table evaluation
     ```json
     [
       {
         "type": "Table",
         "text": [
           {
             "id": "cell-id",
             "x": 0,
             "y": 0,
             "w": 1,
             "h": 1,
             "content": "Cell content"
           },
           ...
         ]
       }
     ]
     ```

### Creating Your Own Ground Truth

To label your own documents:

1. Create a directory for your document set
2. For text content:
   - Create plaintext files with the appropriate markers for document elements
   - Name files according to the convention: `{original_filename}__uns-plaintext-v1.0.0__{id}.txt`
3. For tables:
   - Create JSON files with table structure and cell content
   - Name files according to the convention: `{original_filename}__uns-table-json-v1.0.0__{id}.json`

## Metrics

The SCORE framework includes a multi-dimensional suite of metrics to characterize performance beyond a single score.

### Content Fidelity Metrics

These metrics focus on the accuracy of the extracted text content, adjusted to be robust against structural differences.

| Metric | Definition & Formula                                                         | Context  ||
| :---- |:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :---- |
| **CCT** (Clean Concatenated Text) | Based on Normalized Edit Distance (NED)                                      | Serves as the **traditional baseline**, measuring raw string-level similarity. It is known to fail when content is semantically equivalent but structurally reorganized.                                                                                     |  |
| **Adjusted CCT**(Adj. NED) |  $NED\_{adj}(s,g)=max(NED(s,g),\\frac{\\sum\_{k\\in K}W\_{k}}{W\_{total}})$  | Corrects the CCT/NED score by applying word-weighted fuzzy alignment across different elements (tables, paragraphs), recognizing semantic equivalence despite structural variation. This recovers equivalence between alternative but valid interpretations. |
| **Percent Tokens Found**(TokensFound) | $TokensFound(s,g)=\\frac{\\sum\_{t}min(freq\_{s}(t),freq\_{g}(t))}{\\sum\_{t}freq\_{g}(t)}$                                                                                                                                                                  | Measures the **recall** of reference tokens preserved, independent of ordering. It is a direct indicator of **content omission/loss**. |
| **Percent Tokens Added**(TokensAdded) | $TokensAdded(s,g)=\\frac{\\sum\_{t}max(0,freq\_{s}(t)-freq\_{g}(t))}{\\sum\_{t}freq\_{s}(t)}$                                                                                                                                                                | Measures the proportion of spurious tokens generated. It serves as a direct indicator of the model's **hallucination level**. |


### Structural Hierarchy Metrics
These metrics assess the system's ability to capture the organizational structure of the document.

| Metric | Context |
| :---- | :---- |
| **Element Alignment**(F1-based) | Measures how well predicted elements align with ground truth. It is largely a **reality-grounded metric** that captures how systems map heterogeneous labels into a coherent hierarchy. |
| **Element Consistency**(F1-based) | Evaluates whether the system assigns consistent labels to functionally similar elements across the document using a Confusion Matrix framework. Element labels ("title," "sub-heading") are mapped to **functional categories** ("TITLE") for a semantic-level comparison. |

### Table Structure and Content Metrics

The framework employs a **format-agnostic representation** where all structured outputs (HTML, JSON, etc.) are mapped into equivalent tuples like $(row=0, col=0, \\text{"Q1"})$ for fair comparison.

| Metric | Context                                                                                                                                                                                                                                   |
| :---- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Detection F1** | Recasts table detection as a **content-based classification** problem, not a bounding-box matching task. This is critical for VLMs, which lack explicit object detection modules.                                                         |
| **Content Accuracy** | Measures the accuracy of the extracted cell text/content (text recognition).                                                                                                                                                              |
| **Index Accuracy** | Measures the accuracy of the spatial relationships (cell index/position). Evaluation incorporates **spatial tolerance** for positional shifts, ensuring systems aren't penalized for legitimate variations like merged vs. split headers. |
| **TEDS** | Tree Edit Distance-based Similarity. This traditional metric is integrated to evaluate **hierarchical table structures**.                                                                                                                 |

## Benchmarking Document Parsing

To benchmark a document parsing system:

1. Process your documents with the parsing system to generate outputs
2. Ensure outputs are in the expected format (JSON for structured parsing)
3. Update the configuration file specifying:
   - Metrics to calculate
   - Directories for predictions and ground truth
   - Format specifications
4. Run the evaluation scripts to calculate metrics
5. Analyze the results to identify strengths and weaknesses

## Running the Scripts

### Installation

#### Prerequisites

- Python 3.11 or higher
- uv: https://docs.astral.sh/uv/getting-started/installation/

#### Setup

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

### Generating Evaluation Metrics

#### Document-level Metrics

To generate metrics for individual documents:

```bash
PYTHONPATH=./src python3 -m evaluation.generate_doc_eval_metrics \
  --config configs/all_metrics_config.json \
  --output_csv reports/all_metrics_report.csv
```

#### Aggregate Metrics

To generate aggregate metrics across all documents:

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
      "store_metrics": false  // Storage is used to store metrics in S3. It is set to false by default.
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

### Results

After running the evaluation scripts, results will be saved to the specified CSV file. The results include:
- Document-level metrics for each document and metric type
- Aggregate metrics across all documents in the dataset

These metrics can be used to:

* Compare different document parsing systems  
* Track improvements in parsing quality over time  
* Identify specific areas for improvement, distinguishing between **hallucination, omission, and structural variation**.
* 
## License

This project is licensed under the [CC-BY-NC-SA 4.0](./LICENSE).
