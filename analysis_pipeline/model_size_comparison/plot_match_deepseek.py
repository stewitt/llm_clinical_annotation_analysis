import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- Toggle: use dist_ref-adjusted location matching (True) or pre-computed column (False) ---
USE_DIST_REF_LOCATION = False

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
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
    human_loc = df["Lage"].astype(str).str.strip().copy()
    mask_dist = human_loc.str.lower() == "siehe_distanz"
    human_loc[mask_dist] = df.loc[mask_dist, "Distanz_a.a."].astype(str).str.strip()
    human_norm = human_loc.apply(normalize_location_domain)
    dist_ref = df["Distanz_a.a."].apply(normalize_location_domain)
    llm_norm = df["location_mapped"].apply(normalize_location_domain)
    human_adj = np.where(llm_norm == dist_ref, dist_ref, human_norm)
    return (llm_norm == human_adj).astype(int)

# --- Config ---
FILES = {
    "1.5B": "deepseek_results_1_5b.xlsx",
    "7B":   "deepseek_results_7b.xlsx",
    "14B":  "deepseek_results_14b.xlsx",
    "32B":  "deepseek_results_32b.xlsx",
    "70B":  "deepseek_results_70b.xlsx",
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
results = {}
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
    ci95 = 1.96 * np.sqrt(p * (1 - p) / n) * 100
    results[label] = (pct, ci95)

# --- Plot grouped bar chart ---
models = list(FILES.keys())
n_cols = len(COLS)
n_models = len(models)
x = np.arange(n_cols)
width = 0.15
colors = ["#a6cee3", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]

fig, ax = plt.subplots(figsize=(14, 6))
for i, model in enumerate(models):
    pct, ci95 = results[model]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, pct.values, width, yerr=ci95.values,
                  label=f"DeepSeek-R1-{model}", color=colors[i], edgecolor="white",
                  capsize=3, error_kw={"lw": 1.2})
    for bar, val in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(COL_LABELS, rotation=35, ha="right", fontsize=13)
ax.set_ylabel("Match Rate (%)", fontsize=14)
ax.set_title("Match Rates by DeepSeek-R1 Model Size", fontsize=15, fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("match_rates_deepseek_comparison.png", dpi=150)
plt.savefig("match_rates_deepseek_comparison.svg", format="svg")
plt.show()
print("\nSaved: match_rates_deepseek_comparison.png")
print("Saved: match_rates_deepseek_comparison.svg")
