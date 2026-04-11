# Inference Pipeline

This pipeline processes German-language colonoscopy reports to extract, normalize, and validate information about removed lesions using a local LLM (Gemma 3 27B via [Ollama](https://ollama.com/)).

## Pipeline Stages

```
input_data.xlsx
    │
    ▼
1. LLM_API.py            → llm_out_stage1.xlsx
2. llm_json.py           → llm_out_stage2.xlsx
3. llm_transform.py      → llm_out_stage3.xlsx
4. llm_location_match.py → llm_out_stage4.xlsx
```

| Stage | Script | Task |
|-------|--------|------|
| 1 | `LLM_API.py` | LLM inference — extracts removed lesions from free-text reports |
| 2 | `llm_json.py` | JSON parsing — extracts structured answer field; falls back to LLM repair |
| 3 | `llm_transform.py` | Normalization — extracts sizes, converts units (cm→mm), adds validation columns |
| 4 | `llm_location_match.py` | Location mapping — maps free-text locations to canonical anatomical categories |

## Requirements

```bash
pip install openai pandas openpyxl
```

A running [Ollama](https://ollama.com/) instance with the `gemma3:27b` model is required:

```bash
ollama pull gemma3:27b
ollama serve
```

## Configuration

By default, the pipeline connects to a local Ollama instance at `http://localhost:11434/v1`. Override via environment variables:

```bash
export OLLAMA_BASE_URL="http://<your-ollama-host>:11434/v1"  # required
export OLLAMA_MODEL="<model-name>"                           # required; e.g. gemma3:27b
export OLLAMA_API_KEY="dummy-key"                            # optional; Ollama does not require authentication
```

## Input Data Format

Place your input file as `input_data.xlsx` in this directory. The file must contain at least the following columns:

| Column | Description |
|--------|-------------|
| `report` | Free-text colonoscopy report (German) |
| `Lage` | Reference anatomical location (for validation) |
| `Distanz_a.a.` | Distance from anal verge (for validation) |
| `Durchmesser` | Reference lesion diameter (for validation) |
| `Abtragungsstatus` | Reference resection method (for validation) |
| `mehrere_Polypen` | Reference flag for multiple polyps (for validation) |

> **Note:** The original dataset used in the associated publication cannot be shared due to patient privacy regulations.

## Usage

Run the four stages sequentially:

```bash
python LLM_API.py
python llm_json.py
python llm_transform.py
python llm_location_match.py
```

Each stage saves progress every 10 rows and can be resumed after interruption.

## Output

The final output `llm_out_stage4.xlsx` extends the input with the following columns:

| Column | Description |
|--------|-------------|
| `llm_out` | Raw LLM response |
| `answer` | Parsed structured answer |
| `location_out` | Extracted lesion location |
| `size_out` | Extracted lesion size |
| `en_bloc_out` | Extracted resection method |
| `num1_mm`, `num2_mm` | Normalized sizes in mm |
| `match_Durchm1`, `match_Durchm2` | Size match against reference |
| `match_Abtragungsart` | Resection method match against reference |
| `match_mehrere_Polypen` | Multiple polyps flag match |
| `location_mapped` | Canonical anatomical location |
| `location_distance_match` | Location/distance match against reference |
| `location_distance_match_reason` | Explanation for match/mismatch |
