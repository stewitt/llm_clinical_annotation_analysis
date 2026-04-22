# llm_clinical_annotation_analysis
A research codebase for detecting human annotation errors in structured data extraction from clinical free-text reports using multi-LLM disagreement as a quality control signal. In the corresponding paper, the framework is applied to German-language colonoscopy reports.

Two main pipelines:

Inference pipeline: Runs free-text colonoscopy reports through local LLMs (via Ollama), extracts structured lesion data (diameter, location, resection type), normalizes and maps outputs to canonical categories

Analysis pipeline: Computes error distribution, raw human-llm agreement, inter-rater agreement (Cohen's κ, Fleiss' κ), location match statistics, and cross-model size comparisons


