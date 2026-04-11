import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- Toggle: use dist_ref-adjusted location matching (True) or pre-computed column (False) ---
USE_DIST_REF_LOCATION = False

# --- Global font sizes ---
plt.rcParams.update({
    "font.size":        13,
    "axes.labelsize":   14,
    "axes.titlesize":   15,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
})

# --- Dist-ref location matching helpers ---
# Location values as stored in the source data (mix of German and Latin anatomical terms).
# English equivalents: Zökum=Cecum, rechte/linke Flexur=Right/Left Flexure,
# Sigma=Sigmoid Colon, Rektum=Rectum, unklar=unclear, siehe_Distanz=see distance column
VALID_LOCATIONS_LOWER = {v.lower(): v for v in [
    "Zökum-linke Flexur", "unklar", "rechte Flexur", "linke Flexur",
    "siehe_Distanz", "Zökum", "Colon ascendens", "Colon transversum",
    "Colon descendens", "Sigma", "Rektum",
]}

def extract_numbers(text):
    if not isinstance(text, str):
        return set()
    text = text.replace("–", "-").replace("—", "-")
    return {float(m.replace(",", ".")) for m in re.findall(r"\d+[.,]?\d*", text)}

def normalize_location_domain(raw):
    if isinstance(raw, (float, int)) and not np.isnan(raw):
        return f"dist_{raw}"
    if not isinstance(raw, str) or raw == "nan":
        return "unklar"
    text = raw.strip()
    if text == "":
        return "unklar"
    text_lower = text.lower()
    for key, canonical in VALID_LOCATIONS_LOWER.items():
        if key in text_lower:
            return canonical
    nums = extract_numbers(text_lower)
    if nums:
        return f"dist_{sorted(nums)[0]}"
    return text

def compute_location_match_dist_ref(df):
    """Recalculate location match using dist_ref adjustment logic.

    Column names are German as stored in the source data:
    Lage = Location, Distanz_a.a. = Distance, siehe_distanz = see distance column.
    """
    human_loc = df["Lage"].astype(str).str.strip()
    mask_dist = human_loc.str.lower() == "siehe_distanz"
    human_loc = human_loc.copy()
    human_loc[mask_dist] = df.loc[mask_dist, "Distanz_a.a."].astype(str).str.strip()
    human_norm = human_loc.apply(normalize_location_domain)
    dist_ref = df["Distanz_a.a."].apply(normalize_location_domain)
    llm_norm = df["location_mapped"].apply(normalize_location_domain)
    human_adj = np.where(llm_norm == dist_ref, dist_ref, human_norm)
    return (llm_norm == human_adj).astype(int)

# --- Config ---
FILES = {
    "1B":  "gemma3_results_1b.xlsx",
    "4B":  "gemma3_results_4b.xlsx",
    "12B": "gemma3_results_12b.xlsx",
    "27B": "gemma3_results_27b.xlsx",
}
COLS = [
    "location_distance_match",
    "match_Durchm1",
    "match_Durchm2",
    "match_Abtragungsart",
    "match_mehrere_Polypen",
]
COL_LABELS = [
    "Location",
    "Diameter 1",
    "Diameter 2",
    "Resection status",
    "Multiple polyps",
]
# --- Read & compute stats per model ---
results = {}  # model -> (pct_series, se_series)
for label, path in FILES.items():
    df = pd.read_excel(path)
    df = df[df.index >= 30].reset_index(drop=True)
    for col in COLS:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {path}. Available: {df.columns.tolist()}")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(bool).astype(int)
    if USE_DIST_REF_LOCATION:
        df["location_distance_match"] = compute_location_match_dist_ref(df)
    n = len(df)
    p = df[COLS].mean()
    pct = p * 100
    # 95% CI half-width for a proportion: 1.96 * sqrt(p*(1-p)/n) * 100
    ci95 = 1.96 * np.sqrt(p * (1 - p) / n) * 100
    results[label] = (pct, ci95)
# --- Plot grouped bar chart ---
models = list(FILES.keys())
n_cols = len(COLS)
n_models = len(models)
x = np.arange(n_cols)
width = 0.18
colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]



fig, ax = plt.subplots(figsize=(13, 6))
for i, model in enumerate(models):
    pct, ci95 = results[model]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, pct.values, width, yerr=ci95.values,
                  label=f"Gemma3-{model}", color=colors[i], edgecolor="white",
                  capsize=3, error_kw={"lw": 1.2})
    # value labels
    for bar, val in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(COL_LABELS, rotation=35, ha="right")
ax.set_ylabel("Match Rate (%)")
ax.set_title("Match Rates by Gemma3 Model Size", fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#ax.text(-0.07, 1.05, "a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

plt.tight_layout()
plt.savefig("match_rates_gemma3_comparison.png", dpi=150)
plt.savefig("match_rates_gemma3_comparison.svg", format="svg")
plt.show()
print("\nSaved: match_rates_gemma3_comparison.png")
print("Saved: match_rates_gemma3_comparison.svg")
