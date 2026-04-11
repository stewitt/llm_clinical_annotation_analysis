# llm_clinical_annotation_analysis
A research codebase evaluating how well LLMs extract structured clinical data from German-language colonoscopy reports compared to human expert annotations.

Two main pipelines:

Inference pipeline: Runs free-text colonoscopy reports through local LLMs (via Ollama), extracts structured lesion data (diameter, location, resection type), normalizes and maps outputs to canonical categories

Analysis pipeline: Computes error distribution, raw human-llm agreement, inter-rater agreement (Cohen's κ, Fleiss' κ), location match statistics, and cross-model size comparisons

Tech stack: Python, pandas, scikit-learn, Ollama API, Excel I/O, matplotlib (publication figures with bootstrap CIs)
