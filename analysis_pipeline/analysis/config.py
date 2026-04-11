"""
Configuration constants for blinded annotation analysis.

Paths, bootstrap parameters, label mappings, and column definitions.
"""

# ============================================================
# FILE PATHS
# ============================================================
FILE_BLINDED = "../blinded_annotations.xlsx"
FILE_PREVALENCE = "../population_data.xlsx"
OUTPUT_DIR = "."

# ============================================================
# BOOTSTRAP CONFIG
# ============================================================
N_BOOTSTRAP = 2000
BOOTSTRAP_ALPHA = 0.05   # -> 95% CI
RANDOM_SEED = 42

# ============================================================
# LABEL MAPPING
# Keys are the raw values as stored in the source Excel file;
# values are the English display labels used throughout the analysis.
# ============================================================
VOTER_LABEL_MAP = {
    "A": "Human",
    "B": "Gemma",
    "beide": "both correct",          # German: "both"
    "beide falsch": "both wrong",     # German: "both wrong"
    "unklar": "unclear",              # German: "unclear"
    "nicht entscheidbar": "not decidable",  # German: "not decidable"
}

# ============================================================
# PREVALENCE COLUMN MAPPING
# Keys are column names as they appear in the source Excel file (German);
# values are the internal variable identifiers used in this codebase.
# German originals: Durchm = Durchmesser (Diameter), Abtragungsart = Resection Type,
# mehrere_Polypen = Multiple Polyps, Lage = Location
# ============================================================
MATCH_COLS_MAP = {
    "match_Durchm1": "Durchm1",
    "match_Durchm2": "Durchm2",
    "match_Abtragungsart": "Abtragungsart",
    "match_mehrere_Polypen": "mehrere_Polypen",
    "match_Lage": "Lage",
}

# Human-readable English labels for variable names used in plots
VAR_LABEL_MAP = {
    "Durchm1": "Diameter 1",
    "Durchm2": "Diameter 2",
    "Abtragungsart": "Resection Type",
    "mehrere_Polypen": "Multiple Polyps",
    "Lage": "Location",
    "Combined": "Combined",
}

# Display order of variables in plots
# (variable names match source Excel column names)
VAR_ORDER = ["Lage", "Durchm1", "Durchm2", "Abtragungsart", "mehrere_Polypen"]

# ============================================================
# SHARED PLOT COLORS
# ============================================================
COLORS_CAT = {
    "Human": "#4CAF50", "Gemma": "#E53935", "both correct": "#1E88E5",
    "both wrong": "#FB8C00", "unclear": "#9E9E9E"
}
CATEGORIES = ["Human", "Gemma", "both correct", "both wrong", "unclear"]

# Palette for per-variable bars — muted, publication-friendly
VAR_COLORS = ["#5B8FA8", "#A8786B", "#7BA07E", "#C4A45A", "#8B7EAA", "#6BA3A0"]
