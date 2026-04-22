# LLM Annotation Analysis Pipeline

A statistical analysis pipeline for comparing Large Language Model (LLM) annotations with human expert annotations of German-language colonoscopy reports. This pipeline evaluates annotation agreement, error rates, and classification performance across multiple LLM models and sizes.

## Overview

This project assesses whether LLMs (Gemma, DeepSeek, GPT, Mistral) can reliably extract structured information from clinical colonoscopy reports as a complement to human expert annotations. The analysis covers five key medical variables:

- **Durchm1 / Durchm2** — Lesion diameter (measurements 1 & 2)
- **Abtragungsart** — Resection type/method
- **mehrere_Polypen** — Multiple polyps (yes/no)
- **Lage** — Polyp location

## Directory Structure

```
analysis_pipeline/
├── analysis/                        # Main blinded comparison analysis
│   ├── main.py                      # Entry point — runs all statistical tests
│   ├── config.py                    # File paths, bootstrap settings, label mappings
│   ├── stats_utils.py               # Weighted AUC, bootstrap CIs, operating point analysis
│   └── plotting.py                  # Publication-quality figure generation
│
├── inter_rater_agreement/           # Cohen's κ and Fleiss' κ analysis
│   └── inter_rater_agreement.py
│
├── location/                        # Polyp location distribution analysis
│   └── location_statistics.py
│
├── evaluate_errors_gemma/           # LLM error type breakdown
│   └── plot_evaluate_errors.py
│
└── model_size_comparison/           # Comparison across model sizes
    ├── plot_match_gemma.py          # Gemma 3 (1B, 4B, 12B, 27B)
    └── location_match_deepseek.py   # DeepSeek-R1 (1.5B, 7B, 14B, 32B, 70B)
```

## Installation

Requires Python 3.6+.

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib openpyxl
```

## Input Data

All input files are Excel (`.xlsx`) and should be placed in the parent directory (`../`). Due to patient privacy regulations, the original clinical data cannot be shared.

| File | Used by | Description |
|------|---------|-------------|
| `blinded_annotations.xlsx` | `analysis/main.py` | Per-case blinded annotations with LLM scores and voter correctness labels |
| `population_data.xlsx` | `analysis/main.py` | Population-level match data for extrapolated error estimation |
| `annotation_comparison.xlsx` | `inter_rater_agreement.py`, `location_statistics.py` | Per-case annotations from all raters (human + LLMs) |
| `error_analysis_data.xlsx` | `plot_evaluate_errors.py` | Gemma error type labels per case |
| `gemma3_results_1B/4B/12B/27B.xlsx` | `plot_match_gemma.py` | Match results for each Gemma 3 model size |
| `deepseek_results_1.5B/7B/14B/32B/70B.xlsx` | `location_match_deepseek.py` | Match results for each DeepSeek-R1 model size |

### Key columns in `blinded_annotations.xlsx`

| Column | Values | Description |
|--------|--------|-------------|
| `id` | integer | Case identifier |
| `variable` | Durchm1, Durchm2, Lage, Abtragungsart, mehrere_Polypen | Annotation variable |
| `value` | 0–4 | LLM agreement score (used for stratified sampling) |
| `voter_correct` | A / B / beide / beide falsch / unklar / nicht entscheidbar | Blinded comparison outcome (German labels) |
| `gemma_out` | 0 / 1 | Binary Gemma prediction |

## Usage

### Main blinded comparison analysis

Runs all statistical tests and generates all output figures.

```bash
cd analysis_pipeline/analysis
python main.py
```

**What it does:**
- Loads blinded annotations and population prevalence data
- Computes prevalence weights for stratified sampling correction
- Defines strict and broad outcome variables
- Generates descriptive statistics and crosstabulations
- Runs bootstrap AUC-ROC and AUC-PR analysis (N=2000, seed=42) with 95% CIs
- Performs Gemma point estimate analysis
- Extrapolates population-level error estimates with bootstrap CIs
- Saves all figures as PNG (300 DPI) and SVG

### Inter-rater agreement

```bash
cd analysis_pipeline/inter_rater_agreement
python inter_rater_agreement.py
```

Computes pairwise Cohen's κ and Fleiss' κ for all raters. Outputs 3 SVG plots (κ heatmaps, % agreement, Fleiss' κ by variable).

### Location distribution

```bash
cd analysis_pipeline/location
python location_statistics.py
```

Plots polyp location distribution per rater and model. Outputs `fig1_location_distribution.png/.svg`.

### LLM error analysis

```bash
cd analysis_pipeline/evaluate_errors_gemma
python plot_evaluate_errors.py
```

Generates a horizontal lollipop chart of Gemma error types. Outputs `error_type_counts.png/.svg`.

### Model size comparison

```bash
cd analysis_pipeline/model_size_comparison
python plot_match_gemma.py          # Gemma 3: 1B, 4B, 12B, 27B
python location_match_deepseek.py   # DeepSeek-R1: 1.5B, 7B, 14B, 32B, 70B
```

Generates bar charts of match rates across model sizes. Outputs `match_rates_gemma3_comparison.png/.svg` and `match_rates_deepseek_comparison.png/.svg`.

## Configuration

Edit `analysis/config.py` to adjust file paths, bootstrap parameters, or output directory:

```python
FILE_BLINDED = "../blinded_annotations.xlsx"
FILE_PREVALENCE = "../population_data.xlsx"
OUTPUT_DIR = "."
N_BOOTSTRAP = 2000
BOOTSTRAP_ALPHA = 0.05   # 95% CIs
RANDOM_SEED = 42
```

## Statistical Methods

### Sampling design

Cases were selected using **stratified sampling** by (value (=LLM agreement score), variable(location, diameter, resection type, multiple polyps y/n)):
- Values 0–2: up to 20 cases per variable
- Values 3–4: up to 10 cases per variable

### Prevalence weighting

To correct for the stratified design, each stratum is assigned a weight:

```
weight = N_population / n_sample
```

Weights are used throughout AUC computation and bootstrap resampling.

### Outcome definitions

**Strict definition** (excludes `unclear` and `not decidable`):
- `llm_incorrect = 1` — human was right (voter_correct = "Human" or "both wrong")
- `human_incorrect = 1` — LLM was right (voter_correct = "Gemma" or "both wrong")

**Broad definition** (treats ambiguous cases as errors):
- `human_incorrect_broad = 1` — includes "Gemma", "both wrong", "unclear", "not decidable"

### Statistical tests

| Method | Purpose |
|--------|---------|
| Weighted AUC-ROC / AUC-PR | Primary classification performance metric |
| Bootstrap CIs (N=2000) | Uncertainty quantification for all AUC estimates |
| Bootstrap permutation test | AUC comparison between models |
| Cohen's κ | Pairwise inter-rater agreement |
| Fleiss' κ | Multi-rater agreement across all annotators |
| Extrapolated error estimation | Observed error rate × population count with bootstrap CIs |

## Output Figures

| Script | Output files | Description |
|--------|-------------|-------------|
| `main.py` | 6 figures (PNG + SVG) | Agreement distribution, prevalence, error rates, Gemma analysis, ROC/PR curves, extrapolated errors |
| `inter_rater_agreement.py` | 3 SVG plots | Cohen's κ heatmaps, % agreement, Fleiss' κ |
| `location_statistics.py` | `fig1_location_distribution.png/.svg` | Location distribution per rater |
| `plot_evaluate_errors.py` | `error_type_counts.png/.svg` | Gemma error type lollipop chart |
| `plot_match_gemma.py` | `match_rates_gemma3_comparison.png/.svg` | Match rates by Gemma 3 model size |
| `location_match_deepseek.py` | `match_rates_deepseek_comparison.png/.svg` | Match rates by DeepSeek-R1 model size |

All figures are saved at 300 DPI (PNG) and as vector graphics (SVG) suitable for publication.
