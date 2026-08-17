# AI Literature Screening Pipeline
An LLM-automated screening pipeline for a scoping review in AI in education (AIED) <br>
The pipeline consists of four steps: <br>
1) Screening: Literature abstracts are screened for inclusion and exclusion criteria
2) Use-Case-labeling: Use-case labels based on the AI's educational function
3) Clustering: Use-case labels are clustered around the given seed taxonomy and LLM created novel clusters
4) Meta-data Extraction: Extracting metadata of AIED landscape and pipeline output
---

## 1. Getting started

### 1.1 Prerequisites

- Python 3.9 or later
- OpenAI compatible API key

### 1.2 Installation

Clone the repository and install dependencies:

```bash
pip3 install -r requirements.txt
```
Set endpoint and model in `config/llm_config.json` to an OpenAI compatible provider.<br>
```json
{
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "model": "gpt-4o-mini",
  "parameters": {
    "temperature": 0.2,
    "top_p": 0.9
  },
  "retry": {
    "max_retries": 3,
    "backoff_factor": 2
  },
  "timeout_seconds": 120
}
```
Set the API key, either as a shell environment variable:
```bash
export KICONNECT_API_KEY=your_key_here
```
or in a `.env` file in the project root:
```
KICONNECT_API_KEY=your_key_here
```
### 1.3 Data-Input
The pipeline expects a CSV in `data/input/`. It needs at least these columns:

| Column | Required | Description |
|---|---|---|
| `Title` | Yes | Paper title |
| `Abstract` | Yes | Paper abstract |
| `Year` | No | Publication year |
| `Authors` | No | Author list |
| `Journal` | No | Journal or conference name |
| `Publication Type` | No | e.g. journal article, conference paper |

Rows with an empty `Title` or `Abstract` are skipped automatically.<br>
The delimiter (comma or semicolon) is detected automatically.

### 1.4 Data-Output
| File                            | Produced by | Description                                                                      |
|---------------------------------|-------------|----------------------------------------------------------------------------------|
| `included.csv` / `excluded.csv` | Step 1      | Screening decisions                                                              |
| `labeled.csv`                   | Step 2      | Papers with raw use-case labels                                                  |
| `taxonomy_mapping.json`         | Step 3      | Raw label to cluster mapping                                                     |
| `corpus_final.csv`              | Step 3      | Raw pipeline output                                                              |
| `metadata.csv`                  | Step 4      | Per-paper educational level, subject domain, AI technology, empirical validation |
| `corpus_final_corrected.csv`    | Script      | Manual adjusted output file                                                      |

---

## 2. How to run

### 2.1 The pipeline

```bash
python3 scripts/pipeline.py --input-file data/input/your_InputFile.csv
```

| Step | What it does | Output |
|---|---|---|
| 1. Screening | INCLUDE/EXCLUDE decision + fit score against `screening_prompt.txt` | `included.csv`, `excluded.csv` |
| 2. Use-case labeling | Assigns one or more `snake_case` use-case labels per paper | `labeled.csv` |
| 3. Clustering | Maps every unique label to a seed or novel taxonomy cluster | `taxonomy_mapping.json`, `corpus_final.csv` |
| 4. Metadata extraction | Extracts educational level, subject domain, AI technology, and empirical validation type | `metadata.csv` |

`--stop-after` defaults to `3`. Pass `--stop-after 4` explicitly to run the metadata extraction, otherwise the pipeline stops after clustering to allow for manual reviews of the output file.

Other flags:

| Flag | Default | Description |
|---|---|---|
| `--batch-size` | 5 | Papers per API call (Steps 1, 2, 4) |
| `--label-batch-size` | 20 | Labels per API call (Step 3) |
| `--limit` | none | Cap input size, for test runs |
| `--resume` | off | Resume an interrupted step from checkpoint |
| `--fresh` | off | Delete all outputs and checkpoints, rerun from Step 1 |

### 2.2 Manual adjustments to final output
```bash
python3 scripts/apply_corrections.py
```
### 2.3 Meta-data analysis
Reads `corpus_final_corrected.csv` and `metadata.csv`, and writes every figure and table listed in [Project structure](#3-project-structure) to `data/output/tables/` and `data/output/charts/`. It also runs fine without `metadata.csv`; the metadata-dependent figures are just skipped.

```bash
python3 scripts/analysis.py
```
---
### 2.4 Restart and resume
The pipeline creates checkpoints as a safety mechanism.
This only works for Steps 1, 2, and 4, since Step 3 clusters all labels in one pass and only writes `taxonomy_mapping.json` at the end, so there is no partial progress to resume from.
Finished steps get skipped automatically once their output exists.
```bash
python3 scripts/pipeline.py --input-file data/input/your_file.csv --resume
```
`--fresh` deletes all existing outputs and checkpoints and reruns everything from Step 1.

For example:
To rerun from Step 2:  delete `labeled.csv`, `taxonomy_mapping.json`, `corpus_final.csv`, `corpus_final.json`, and `metadata.csv`, but keep `included.csv` and `excluded.csv`.

---

## 3. Project structure

```
ai_screening_pipeline/
├── config/
│   └── llm_config.json              # LLM endpoint, model, and retry settings
├── data/
│   ├── input/                       # Input CSVs
│   ├── output/                      # All pipeline and analysis outputs
│   │   ├── tables/                  # CSV tables from analysis.py
│   │   └── charts/                  # SVG/PNG charts from analysis.py
│   └── checkpoints/                 # Resume state for interrupted steps
├── prompts/
│   ├── screening_prompt.txt
│   ├── use_case_labeling_prompt.txt
│   ├── clustering_prompt.txt
│   └── metadata_prompt.txt
├── scripts/
│   ├── pipeline.py                  # Main pipeline (Steps 1-4)
│   ├── apply_corrections.py         # Manual cluster corrections; also exports LABEL_CORRECTIONS as an importable dict
│   ├── analysis.py                  # All figures and tables for the thesis
│   ├── compare_pipeline_output.py   # Raw vs. corrected corpus comparison report
│   └── validate.py                  # Step 1 validation against a gold dataset
├── utils/
│   ├── llm_client.py                # LLM connection and retry handler
│   └── checkpoint.py                # Checkpoint save/load/clear
├── .env                              # API key (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```
## 4. How to reproduce the analysis
Clone the repository, then set the endpoint, model and API key as described in [1.2 Installation](#12-installation) and install the dependencies with `pip3 install -r requirements.txt`. After that, run the three scripts in this order:

```bash
python3 scripts/pipeline.py --input-file data/input/corpus_input.csv --stop-after 4
python3 scripts/apply_corrections.py
python3 scripts/analysis.py
```

The pipeline writes `corpus_final.csv` and `metadata.csv`, `apply_corrections.py` turns the raw corpus into `corpus_final_corrected.csv`, and `analysis.py` reads both to write the figures and tables.

## 5. Citation

Developed for the master thesis:

> Yilmaz, I. (2026). *AI Affordances in Education: A Scoping Review on Use Cases, Constraints, and Alignment with Educational Purpose*. University of Cologne.

LLM inference was provided through KI:Connect NRW, documented at the [University of Cologne ITCC service page](https://itcc.uni-koeln.de/services/software/kuenstliche-intelligenz).
