#!/usr/bin/env python3
"""
Statistical analysis of blinded human vs LLM annotation comparison.

Inputs:
  1. blinded_annotations.xlsx
     - voter_correct: blinded adjudication vote (A, B, both correct, both wrong, unclear, not decidable)
     - value: 0-4 = how many of 4 LLMs matched human annotation (higher = more agreement)
     - gemma_out: 1 if gemma matched human, 0 if not
     - variable: annotation category (Diameter1, Diameter2, Location, ResectionType, MultiplePolyps)

  2. population_data.xlsx  (for prevalence calculation)
     - columns: match_Diameter1, match_Diameter2, match_ResectionType, match_MultiplePolyps, match_Location
     - values 0-4: how many LLMs matched human annotations

Sampling design (stratified by value x variable):
  - value 0,1,2: up to 20 per variable (5 variables -> up to 100 per value)
  - value 3,4:   up to 10 per variable (5 variables -> up to 50 per value)
  - If a cell had fewer than the quota, all available were used.

Prevalence adjustment uses per-(value, variable) importance weights:
  weight(v, var) = N_population(v, var) / n_sample(v, var)

Usage:
    python main.py

Adjust FILE_BLINDED and FILE_PREVALENCE in config.py if needed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


from config import (
    FILE_BLINDED, FILE_PREVALENCE, OUTPUT_DIR,
    N_BOOTSTRAP, BOOTSTRAP_ALPHA, RANDOM_SEED,
    VOTER_LABEL_MAP, MATCH_COLS_MAP,
)
from stats_utils import (
    bootstrap_auc, bootstrap_weighted_auc,
    bootstrap_gemma_operating_point,
    weighted_auc_roc, weighted_auc_pr,
)
from plotting import (
    plot_figure_a, plot_figure_b, plot_figure_c,
    plot_figure_d, plot_extrapolated_errors_combined,
    plot_figure_agreement_vs_human,
)


# ============================================================
# 1. LOAD AND CLEAN BLINDED DATA
# ============================================================
print("=" * 80)
print("1. LOADING DATA")
print("=" * 80)

df = pd.read_excel(FILE_BLINDED)
df = df.dropna(subset=["id"])
df = df[df["id"] >= 30]
df["voter_correct"] = df["voter_correct"].astype(str).str.strip()
df["voter_correct"] = df["voter_correct"].map(VOTER_LABEL_MAP).fillna(df["voter_correct"])
df["value"] = df["value"].astype(int)
df["gemma_out"] = df["gemma_out"].astype(int)

print(f"Blinded data: {len(df)} rows after cleaning")
print(f"Variables: {df['variable'].dropna().unique().tolist()}")
print(f"voter_correct categories: {df['voter_correct'].value_counts().to_dict()}")
print(f"value distribution: {df['value'].value_counts().sort_index().to_dict()}")
print()


# ============================================================
# 2. LOAD PREVALENCE DATA & COMPUTE PER-(VALUE, VARIABLE) WEIGHTS
# ============================================================
print("=" * 80)
print("2. PREVALENCE FROM population_data.xlsx")
print("=" * 80)

try:
    df_prev = pd.read_excel(FILE_PREVALENCE)
    df_prev = df_prev[df_prev["index"] >= 30]
    print(f"Prevalence file loaded: {len(df_prev)} rows (index >= 30)")

    pop_counts = {}
    prevalence_by_var = {}

    for match_col, var_name in MATCH_COLS_MAP.items():
        if match_col not in df_prev.columns:
            print(f"  WARNING: column {match_col} not found")
            continue
        col_data = df_prev[match_col].dropna().astype(int)
        counts = col_data.value_counts().sort_index()
        total = counts.sum()
        prev = (counts / total).to_dict()
        prevalence_by_var[var_name] = {
            "counts": counts.to_dict(), "total": total, "prevalence": prev
        }
        print(f"\n  {var_name} (n={total}):")
        for v in sorted(counts.index):
            pop_counts[(v, var_name)] = counts[v]
            print(f"    value={v}: {counts[v]:5d}  ({prev[v]*100:.1f}%)")

    all_values = []
    for match_col in MATCH_COLS_MAP.keys():
        if match_col in df_prev.columns:
            all_values.extend(df_prev[match_col].dropna().astype(int).tolist())
    all_values = pd.Series(all_values)
    overall_prev = (all_values.value_counts().sort_index() / len(all_values)).to_dict()
    overall_counts = all_values.value_counts().sort_index().to_dict()
    print(f"\n  OVERALL (n={len(all_values)}):")
    for v in sorted(overall_prev.keys()):
        print(f"    value={v}: {overall_counts.get(v, 0):5d}  ({overall_prev[v]*100:.1f}%)")

    prevalence_by_var["Combined"] = {
        "counts": overall_counts, "total": len(all_values), "prevalence": overall_prev
    }

    # --- Per-LLM individual match rates ---
    # Column keys are German source names: Durchm1/Durchm2 = Diameter 1/2,
    # Abtragungsart = Resection Type, mehrere_Polypen = Multiple Polyps, Lage = Location
    llm_match_cols = {
        "Deepseek": {
            "Durchm1":        "match_Durchm1_deepseek",
            "Durchm2":        "match_Durchm2_deepseek",
            "Abtragungsart":  "match_Abtragungsart_deepseek",
            "mehrere_Polypen":"match_mehrere_Polypen_deepseek",
            "Lage":           "location_distance_match_deepseek",
        },
        "Gemma": {
            "Durchm1":        "match_Durchm1_gemma",
            "Durchm2":        "match_Durchm2_gemma",
            "Abtragungsart":  "match_Abtragungsart_gemma",
            "mehrere_Polypen":"match_mehrere_Polypen_gemma",
            "Lage":           "location_distance_match_gemma",
        },
        "GPT": {
            "Durchm1":        "match_Durchm1_pgt",
            "Durchm2":        "match_Durchm2_pgt",
            "Abtragungsart":  "match_Abtragungsart_pgt",
            "mehrere_Polypen":"match_mehrere_Polypen_pgt",
            "Lage":           "location_distance_match_pgt",
        },
        "Mistral": {
            "Durchm1":        "match_Durchm1_mistral",
            "Durchm2":        "match_Durchm2_mistral",
            "Abtragungsart":  "match_Abtragungsart_mistral",
            "mehrere_Polypen":"match_mehrere_Polypen_mistral",
            "Lage":           "location_distance_match_mistral",
        },
    }

    # Variable names match source Excel column names (German); see VAR_LABEL_MAP in config.py for English labels
    var_order_llm = ["Durchm1", "Durchm2", "Abtragungsart", "mehrere_Polypen", "Lage"]
    print(f"\n--- Individual LLM match rates vs. human (n={len(df_prev)} rows) ---")
    header = f"  {'Variable':>20s}" + "".join(f"  {m:>10s}" for m in llm_match_cols)
    print(header)
    print("  " + "-" * (22 + 12 * len(llm_match_cols)))
    for var in var_order_llm:
        row_str = f"  {var:>20s}"
        for model, cols in llm_match_cols.items():
            col = cols.get(var)
            if col and col in df_prev.columns:
                valid = df_prev[col].dropna()
                valid = valid[valid.isin([0, 1])]
                rate = valid.mean() if len(valid) > 0 else float("nan")
                row_str += f"  {rate*100:>9.1f}%"
            else:
                row_str += f"  {'N/A':>10s}"
        print(row_str)
    # Overall (mean across all variables per LLM)
    row_str = f"  {'OVERALL':>20s}"
    for model, cols in llm_match_cols.items():
        vals = []
        for var, col in cols.items():
            if col in df_prev.columns:
                valid = df_prev[col].dropna()
                valid = valid[valid.isin([0, 1])]
                vals.extend(valid.tolist())
        rate = np.mean(vals) if vals else float("nan")
        row_str += f"  {rate*100:>9.1f}%"
    print(row_str)

    sample_counts = df.groupby(["value", "variable"]).size().to_dict()

    cell_weights = {}
    print("\n--- Per-(value, variable) importance weights ---")
    print(f"  {'variable':>20s}  {'value':>5s}  {'N_pop':>6s}  {'n_samp':>6s}  {'weight':>10s}")
    print("  " + "-" * 55)

    all_vars = sorted(set(v for _, v in pop_counts.keys()))
    for var_name in all_vars:
        for v in range(5):
            n_pop = pop_counts.get((v, var_name), 0)
            n_samp = sample_counts.get((v, var_name), 0)
            if n_samp > 0 and n_pop > 0:
                w = n_pop / n_samp
            else:
                w = 0.0
            cell_weights[(v, var_name)] = w
            if n_samp > 0 or n_pop > 0:
                print(f"  {var_name:>20s}  {v:>5d}  {n_pop:>6d}  {n_samp:>6d}  {w:>10.4f}")

    PREVALENCE_AVAILABLE = True

except FileNotFoundError:
    print(f"  WARNING: {FILE_PREVALENCE} not found. Skipping prevalence-based analyses.")
    PREVALENCE_AVAILABLE = False
    prevalence_by_var = {}
    overall_prev = {}
    cell_weights = {}
    pop_counts = {}
    all_vars = []

print()


# ============================================================
# Helper: assign weight to each row based on (value, variable)
# ============================================================
def assign_weights(dataframe):
    dataframe = dataframe.copy()
    dataframe["weight"] = dataframe.apply(
        lambda r: cell_weights.get((int(r["value"]), r["variable"]), 1.0), axis=1
    )
    return dataframe


# ============================================================
# 3. DEFINE OUTCOMES
# ============================================================
print("=" * 80)
print("3. OUTCOME DEFINITIONS")
print("=" * 80)

# Strict definition: exclude unclear / not decidable
df["human_incorrect"] = df["voter_correct"].map({
    "Human": 0, "Gemma": 1, "both correct": 0, "both wrong": 1, "unclear": np.nan
})
df["llm_incorrect"] = df["voter_correct"].map({
    "Human": 1, "Gemma": 0, "both correct": 0, "both wrong": 1, "unclear": np.nan
})

# Broad definition: unclear + not decidable -> human_incorrect = 1
df["human_incorrect_broad"] = df["voter_correct"].map({
    "Human": 0, "Gemma": 1, "both correct": 0, "both wrong": 1,
    "unclear": 1, "not decidable": 1,
})

df_binary = df.dropna(subset=["human_incorrect"]).copy()
df_binary_broad = df.dropna(subset=["human_incorrect_broad"]).copy()

if PREVALENCE_AVAILABLE:
    df = assign_weights(df)
    df_binary = assign_weights(df_binary)
    df_binary_broad = assign_weights(df_binary_broad)

print(f"Resolved cases (strict): n={len(df_binary)}")
print(f"  human_incorrect rate = {df_binary['human_incorrect'].mean():.3f}  "
      f"(Gemma correct + both wrong)")
print(f"  llm_incorrect rate   = {df_binary['llm_incorrect'].mean():.3f}  "
      f"(Human correct + both wrong)")
print(f"  Excluded (unclear): n={len(df) - len(df_binary)}")

print(f"\nResolved cases (broad = strict + unclear): n={len(df_binary_broad)}")
print(f"  human_incorrect_broad rate = {df_binary_broad['human_incorrect_broad'].mean():.3f}  "
      f"(Gemma correct + both wrong + unclear)")
print()


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
print("=" * 80)
print("4. DESCRIPTIVE: value vs voter_correct")
print("=" * 80)

ct = pd.crosstab(df["voter_correct"], df["value"], margins=True)
print("\nCrosstab (voter_correct x value):")
print(ct)

print("\n--- Human incorrect rate by value (Gemma correct + both wrong) ---")
for v in sorted(df_binary["value"].unique()):
    sub = df_binary[df_binary["value"] == v]
    rate = sub["human_incorrect"].mean()
    n = len(sub)
    print(f"  value={v}: {rate:.3f}  (n={n}, {int(sub['human_incorrect'].sum())} errors)")

print("\n--- LLM incorrect rate by value (Human correct + both wrong) ---")
for v in sorted(df_binary["value"].unique()):
    sub = df_binary[df_binary["value"] == v]
    rate = sub["llm_incorrect"].mean()
    n = len(sub)
    print(f"  value={v}: {rate:.3f}  (n={n}, {int(sub['llm_incorrect'].sum())} errors)")

print("\n--- Per variable ---")
for var in sorted(df_binary["variable"].dropna().unique()):
    print(f"\n  {var}:")
    sub_var = df_binary[df_binary["variable"] == var]
    for v in sorted(sub_var["value"].unique()):
        sub = sub_var[sub_var["value"] == v]
        h_rate = sub["human_incorrect"].mean()
        l_rate = sub["llm_incorrect"].mean()
        print(f"    value={v}: human_err={h_rate:.3f}  llm_err={l_rate:.3f}  (n={len(sub)})")

print()


# ============================================================
# 5. GEMMA_OUT ANALYSIS
# ============================================================
print("=" * 80)
print("5. GEMMA_OUT vs voter_correct")
print("=" * 80)

print("\nCrosstab (voter_correct x gemma_out):")
print(pd.crosstab(df["voter_correct"], df["gemma_out"], margins=True))

ct_gemma = pd.crosstab(df_binary["gemma_out"], df_binary["human_incorrect"])
print(f"\nGemma_out vs human_incorrect (Gemma correct + both wrong):")
print(ct_gemma)

ct_gemma_l = pd.crosstab(df_binary["gemma_out"], df_binary["llm_incorrect"])
print(f"\nGemma_out vs llm_incorrect (Human correct + both wrong):")
print(ct_gemma_l)

print("\n--- When gemma != human (gemma_out=0), who was right? ---")
sub_g0_console = df[df["gemma_out"] == 0].copy()
print("Raw counts:")
vc_g0 = sub_g0_console["voter_correct"].value_counts()
total_g0 = len(sub_g0_console)
for cat, cnt in vc_g0.items():
    print(f"  {cat:20s}: {cnt:4d}  ({cnt/total_g0*100:5.1f}%)")

if PREVALENCE_AVAILABLE:
    print("\nPrevalence-adjusted proportions (gemma_out=0):")
    total_w = sub_g0_console["weight"].sum()
    for cat in ["Human", "Gemma", "both correct", "both wrong", "unclear", "not decidable"]:
        w_cat = sub_g0_console.loc[sub_g0_console["voter_correct"] == cat, "weight"].sum()
        cnt = vc_g0.get(cat, 0)
        if w_cat > 0:
            print(f"  {cat:20s}: {cnt:4d}  ({cnt/total_g0*100:5.1f}% raw)  {w_cat/total_w*100:5.1f}% prev-adj  (weighted n={w_cat:.1f})")

print("\n--- Detailed breakdown: gemma_out=0, by value and variable ---")
sub_g0_detail = df[df["gemma_out"] == 0].copy()
print(f"Total rows with gemma_out=0: {len(sub_g0_detail)}")

print(f"\n{'variable':>20s}  {'value':>5s}  {'n_samp':>6s}  {'N_pop':>6s}  "
      f"{'weight':>8s}  {'Human':>3s}  {'Gemma':>3s}  {'both':>5s}  {'b.wrg':>5s}  {'uncl':>4s}")
print("-" * 95)

for var in sorted(sub_g0_detail["variable"].dropna().unique()):
    for v in sorted(sub_g0_detail["value"].unique()):
        sv = sub_g0_detail[(sub_g0_detail["variable"] == var) & (sub_g0_detail["value"] == v)]
        if len(sv) == 0:
            continue
        n = len(sv)
        vc = sv["voter_correct"].value_counts()
        a = vc.get("Human", 0)
        b = vc.get("Gemma", 0)
        both = vc.get("both correct", 0)
        bw = vc.get("both wrong", 0)
        unk = vc.get("unclear", 0) + vc.get("not decidable", 0)
        w = cell_weights.get((v, var), np.nan)
        n_pop = pop_counts.get((v, var), 0)
        print(f"{var:>20s}  {v:>5d}  {n:>6d}  {n_pop:>6d}  "
              f"{w:>8.2f}  {a:>3d}  {b:>3d}  {both:>5d}  {bw:>5d}  {unk:>4d}")

if PREVALENCE_AVAILABLE:
    print(f"\n{'(all vars)':>20s}  {'value':>5s}  {'n_samp':>6s}  {'N_pop':>6s}  "
          f"{'Human':>3s}  {'Gemma':>3s}  {'both':>5s}  {'b.wrg':>5s}  {'uncl':>4s}")
    print("-" * 75)
    for v in sorted(sub_g0_detail["value"].unique()):
        sv = sub_g0_detail[sub_g0_detail["value"] == v]
        n = len(sv)
        vc = sv["voter_correct"].value_counts()
        a = vc.get("Human", 0)
        b = vc.get("Gemma", 0)
        both = vc.get("both correct", 0)
        bw = vc.get("both wrong", 0)
        unk = vc.get("unclear", 0) + vc.get("not decidable", 0)
        n_pop_v = sum(pop_counts.get((v, var), 0) for var in all_vars)
        print(f"{'(all)':>20s}  {v:>5d}  {n:>6d}  {n_pop_v:>6d}  "
              f"{a:>3d}  {b:>3d}  {both:>5d}  {bw:>5d}  {unk:>4d}")

# ---- gemma_out == 1: when gemma agreed with human ----
print("\n--- When gemma == human (gemma_out=1), voter_correct distribution ---")
sub_g1_console = df[df["gemma_out"] == 1].copy()
print("Raw counts:")
vc_g1 = sub_g1_console["voter_correct"].value_counts()
total_g1 = len(sub_g1_console)
for cat, cnt in vc_g1.items():
    print(f"  {cat:20s}: {cnt:4d}  ({cnt/total_g1*100:5.1f}%)")

if PREVALENCE_AVAILABLE:
    print("\nPrevalence-adjusted proportions (gemma_out=1):")
    total_w1 = sub_g1_console["weight"].sum()
    for cat in ["Human", "Gemma", "both correct", "both wrong", "unclear", "not decidable"]:
        w_cat = sub_g1_console.loc[sub_g1_console["voter_correct"] == cat, "weight"].sum()
        cnt = vc_g1.get(cat, 0)
        if w_cat > 0:
            print(f"  {cat:20s}: {cnt:4d}  ({cnt/total_g1*100:5.1f}% raw)  {w_cat/total_w1*100:5.1f}% prev-adj  (weighted n={w_cat:.1f})")

print()
print("\n--- Detailed breakdown: gemma_out=1, by value and variable ---")
sub_g1_detail = df[df["gemma_out"] == 1].copy()
print(f"Total rows with gemma_out=1: {len(sub_g1_detail)}")

print(f"\n{'variable':>20s}  {'value':>5s}  {'n_samp':>6s}  {'N_pop':>6s}  "
      f"{'weight':>8s}  {'Human':>3s}  {'Gemma':>3s}  {'both':>5s}  {'b.wrg':>5s}  {'uncl':>4s}")
print("-" * 95)

for var in sorted(sub_g1_detail["variable"].dropna().unique()):
    for v in sorted(sub_g1_detail["value"].unique()):
        sv = sub_g1_detail[(sub_g1_detail["variable"] == var) & (sub_g1_detail["value"] == v)]
        if len(sv) == 0:
            continue
        n = len(sv)
        vc = sv["voter_correct"].value_counts()
        a = vc.get("Human", 0)
        b = vc.get("Gemma", 0)
        both = vc.get("both correct", 0)
        bw = vc.get("both wrong", 0)
        unk = vc.get("unclear", 0) + vc.get("not decidable", 0)
        w = cell_weights.get((v, var), np.nan)
        n_pop = pop_counts.get((v, var), 0)
        print(f"{var:>20s}  {v:>5d}  {n:>6d}  {n_pop:>6d}  "
              f"{w:>8.2f}  {a:>3d}  {b:>3d}  {both:>5d}  {bw:>5d}  {unk:>4d}")

if PREVALENCE_AVAILABLE:
    print(f"\n{'(all vars)':>20s}  {'value':>5s}  {'n_samp':>6s}  {'N_pop':>6s}  "
          f"{'Human':>3s}  {'Gemma':>3s}  {'both':>5s}  {'b.wrg':>5s}  {'uncl':>4s}")
    print("-" * 75)
    for v in sorted(sub_g1_detail["value"].unique()):
        sv = sub_g1_detail[sub_g1_detail["value"] == v]
        n = len(sv)
        vc = sv["voter_correct"].value_counts()
        a = vc.get("Human", 0)
        b = vc.get("Gemma", 0)
        both = vc.get("both correct", 0)
        bw = vc.get("both wrong", 0)
        unk = vc.get("unclear", 0) + vc.get("not decidable", 0)
        n_pop_v = sum(pop_counts.get((v, var), 0) for var in all_vars)
        print(f"{'(all)':>20s}  {v:>5d}  {n:>6d}  {n_pop_v:>6d}  "
              f"{a:>3d}  {b:>3d}  {both:>5d}  {bw:>5d}  {unk:>4d}")

print()


# ============================================================
# 6. CLASSIFICATION COMPARISON
# ============================================================
print("=" * 80)
print("6. CLASSIFICATION: value (with bootstrap 95% CI), gemma_out (point estimate)")
print("=" * 80)

y_true = df_binary["human_incorrect"].values.astype(int)
val_score = df_binary["value"].values.astype(float)
gemma_score = df_binary["gemma_out"].values.astype(float)


def compute_metrics_with_ci(y_true, scores, label, higher_means_positive=True):
    if len(np.unique(y_true)) < 2:
        print(f"  {label}: Only one class, skipping.")
        return None
    s = scores if higher_means_positive else -scores
    auc_roc = roc_auc_score(y_true, s)
    auc_pr = average_precision_score(y_true, s)
    prevalence = y_true.mean()
    boot = bootstrap_auc(y_true, scores, higher_means_positive=higher_means_positive)
    print(f"  {label}:")
    print(f"    AUC-ROC = {auc_roc:.4f}  95% CI [{boot['ci_roc'][0]:.4f}, {boot['ci_roc'][1]:.4f}]")
    print(f"    AUC-PR  = {auc_pr:.4f}  95% CI [{boot['ci_pr'][0]:.4f}, {boot['ci_pr'][1]:.4f}]"
          f"  (prevalence = {prevalence:.4f})")
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "prevalence": prevalence,
            "ci_roc": boot["ci_roc"], "ci_pr": boot["ci_pr"]}


def compute_metrics_point(y_true, scores, label, higher_means_positive=True):
    if len(np.unique(y_true)) < 2:
        print(f"  {label}: Only one class, skipping.")
        return None
    s = scores if higher_means_positive else -scores
    auc_roc = roc_auc_score(y_true, s)
    auc_pr = average_precision_score(y_true, s)
    prevalence = y_true.mean()
    print(f"  {label}:")
    print(f"    AUC-ROC = {auc_roc:.4f}  (point estimate only — binary predictor)")
    print(f"    AUC-PR  = {auc_pr:.4f}  (point estimate only — binary predictor)"
          f"  (prevalence = {prevalence:.4f})")
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "prevalence": prevalence}


print(f"\n--- Predicting human error (Gemma correct + both wrong) ---")
print(f"  Prevalence of human error: {y_true.mean():.4f} (n={len(y_true)})")
print(f"  Bootstrap: n_boot={N_BOOTSTRAP}, stratified resampling, seed={RANDOM_SEED}")
m_val = compute_metrics_with_ci(y_true, val_score, "value (higher=more LLM agreement)", higher_means_positive=False)
m_gem = compute_metrics_point(y_true, gemma_score, "gemma_out (1=match human)", higher_means_positive=False)

print("\n  NOTE: gemma_out is binary (0/1), so its ROC 'curve' is just 3 points.")

print("\n--- AUC-ROC by variable (with bootstrap 95% CI for value-based) ---")
for var in sorted(df_binary["variable"].dropna().unique()):
    sub = df_binary[df_binary["variable"] == var]
    y_v = sub["human_incorrect"].values.astype(int)
    if len(np.unique(y_v)) < 2:
        print(f"  {var}: only one class, skipping")
        continue
    sv = sub["value"].values.astype(float)
    sg = sub["gemma_out"].values.astype(float)
    boot_v = bootstrap_auc(y_v, sv, higher_means_positive=False)
    auc_g = roc_auc_score(y_v, -sg)
    print(f"  {var} (n={len(sub)}, prev={y_v.mean():.3f}):")
    print(f"    AUC_val={boot_v['point_roc']:.3f} [{boot_v['ci_roc'][0]:.3f}, {boot_v['ci_roc'][1]:.3f}]"
          f"    AUC_gemma={auc_g:.3f} (point est.)")

print()


# ============================================================
# 7. PREVALENCE-ADJUSTED AUC
# ============================================================
print("=" * 80)
print("7. PREVALENCE-ADJUSTED AUC-ROC AND AUC-PR (with bootstrap 95% CI)")
print("=" * 80)

if PREVALENCE_AVAILABLE:
    print("\nUsing per-(value, variable) importance weights:")
    print("  weight(v, var) = N_population(v, var) / n_sample(v, var)")
    print(f"  Bootstrap: n_boot={N_BOOTSTRAP}, stratified resampling, seed={RANDOM_SEED}")

    weight_summary = df_binary.groupby(["variable", "value"]).agg(
        n_sample=("weight", "size"),
        weight=("weight", "first")
    ).reset_index()
    print(f"\n  {'variable':>20s}  {'value':>5s}  {'n_samp':>6s}  {'weight':>10s}")
    print("  " + "-" * 45)
    for _, row in weight_summary.iterrows():
        print(f"  {row['variable']:>20s}  {int(row['value']):>5d}  "
              f"{int(row['n_sample']):>6d}  {row['weight']:>10.4f}")

    def report_weighted_with_ci(y_true, scores, weights, label):
        boot = bootstrap_weighted_auc(y_true, scores, weights)
        adj_prev = (y_true * weights).sum() / weights.sum()
        print(f"  {label}:")
        print(f"    Weighted AUC-ROC = {boot['point_roc']:.4f}  "
              f"95% CI [{boot['ci_roc'][0]:.4f}, {boot['ci_roc'][1]:.4f}]")
        print(f"    Weighted AUC-PR  = {boot['point_pr']:.4f}  "
              f"95% CI [{boot['ci_pr'][0]:.4f}, {boot['ci_pr'][1]:.4f}]"
              f"  (adj. prevalence = {adj_prev:.4f})")
        return {"w_auc_roc": boot["point_roc"], "w_auc_pr": boot["point_pr"],
                "adj_prev": adj_prev,
                "ci_roc": boot["ci_roc"], "ci_pr": boot["ci_pr"]}

    def report_weighted_point(y_true, scores, weights, label):
        w_roc = weighted_auc_roc(y_true, scores, weights)
        w_pr = weighted_auc_pr(y_true, scores, weights)
        adj_prev = (y_true * weights).sum() / weights.sum()
        print(f"  {label}:")
        print(f"    Weighted AUC-ROC = {w_roc:.4f}  (point estimate only — binary predictor)")
        print(f"    Weighted AUC-PR  = {w_pr:.4f}  (point estimate only — binary predictor)"
              f"  (adj. prevalence = {adj_prev:.4f})")
        return {"w_auc_roc": w_roc, "w_auc_pr": w_pr, "adj_prev": adj_prev}

    print("\n--- Prevalence-adjusted (human error = Gemma correct + both wrong) ---")
    y_s = df_binary["human_incorrect"].values.astype(int)
    w_s = df_binary["weight"].values
    m_val_w = report_weighted_with_ci(y_s, -df_binary["value"].values.astype(float), w_s, "value-based")
    m_gem_w = report_weighted_point(y_s, -df_binary["gemma_out"].values.astype(float), w_s, "gemma_out-based")

    print("\n--- Prevalence-adjusted AUC-ROC by variable ---")
    for var in sorted(df_binary["variable"].dropna().unique()):
        sub = df_binary[df_binary["variable"] == var]
        y_v = sub["human_incorrect"].values.astype(int)
        if len(np.unique(y_v)) < 2:
            print(f"  {var}: only one class, skipping")
            continue
        var_weights = sub["weight"].values
        sv_v = -sub["value"].values.astype(float)
        sg_v = -sub["gemma_out"].values.astype(float)
        boot_v = bootstrap_weighted_auc(y_v, sv_v, var_weights)
        w_roc_g = weighted_auc_roc(y_v, sg_v, var_weights)
        print(f"  {var} (n={len(sub)}):")
        print(f"    wAUC_val={boot_v['point_roc']:.3f} "
              f"[{boot_v['ci_roc'][0]:.3f}, {boot_v['ci_roc'][1]:.3f}]"
              f"    wAUC_gem={w_roc_g:.3f} (point est.)")
else:
    print("Skipping prevalence-adjusted analysis (no prevalence file).")

print()


# ============================================================
# 8. FIGURES
# ============================================================
print("=" * 80)
print("8. GENERATING FIGURES")
print("=" * 80)

# Pre-compute bootstrap CI for gemma operating point (needed for figures)
print("  Computing bootstrap CI for gemma operating point + AUC...")
if PREVALENCE_AVAILABLE:
    g_boot_ci = bootstrap_gemma_operating_point(
        df_binary["human_incorrect"].values.astype(int),
        df_binary["gemma_out"].values,
        weights=df_binary["weight"].values,
        n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
else:
    g_boot_ci = bootstrap_gemma_operating_point(
        y_true, df_binary["gemma_out"].values,
        weights=None,
        n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

plot_figure_a(df, prevalence_by_var, PREVALENCE_AVAILABLE)
plot_figure_b(df, df_binary)
plot_figure_c(df, df_binary, PREVALENCE_AVAILABLE, g_boot_ci, cell_weights)
auc_strict_roc, ap_strict, auc_lib_roc, ap_lib = plot_figure_d(
    df_binary, df_binary_broad, PREVALENCE_AVAILABLE)

# ---- Figure G: Pct Agreement LLM vs Human per category ----
if PREVALENCE_AVAILABLE:
    agree_rows = []
    for llm_name, var_cols in llm_match_cols.items():
        for var_name, col in var_cols.items():
            if col not in df_prev.columns:
                continue
            valid = df_prev[col].dropna()
            valid = valid[valid.isin([0, 1])]
            n = len(valid)
            if n == 0:
                continue
            p = valid.mean()
            se = np.sqrt(p * (1 - p) / n) * 100
            pct = p * 100
            agree_rows.append({
                'llm': llm_name,
                'variable': var_name,
                'pct_agree': pct,
                'pct_se': se,
                'pct_ci_low':  max(0.0,   pct - 1.96 * se),
                'pct_ci_high': min(100.0, pct + 1.96 * se),
                'n': n,
            })
    if agree_rows:
        res_df_agree = pd.DataFrame(agree_rows)
        plot_figure_agreement_vs_human(res_df_agree)


# ============================================================
# 9. EXTRAPOLATED ERROR ESTIMATION
# ============================================================
print()
print("=" * 80)
print("9. EXTRAPOLATED ERROR ESTIMATION")
print("=" * 80)
print("""
  Method: For each (value, variable) cell, the observed human error rate
  from the blinded sample is applied to the full population count N_pop(v, var)
  to estimate the total number of probable human annotation errors.
  Bootstrap 95% CIs are computed by resampling within each cell.
""")

if PREVALENCE_AVAILABLE:
    rng_extrap = np.random.RandomState(RANDOM_SEED)
    N_BOOT_EXTRAP = N_BOOTSTRAP

    all_vars_extrap = sorted(set(v for _, v in pop_counts.keys()))
    val_range_extrap = sorted(set(v for v, _ in pop_counts.keys()))

    # ---- STRICT DEFINITION ----
    cell_results = []
    print(f"  {'variable':>20s}  {'val':>3s}  {'n_samp':>6s}  {'N_pop':>6s}  "
          f"{'err_rate':>8s}  {'est_errors':>10s}  {'95% CI':>20s}")
    print("  " + "-" * 80)

    for var_name in all_vars_extrap:
        for v in val_range_extrap:
            n_pop = pop_counts.get((v, var_name), 0)
            if n_pop == 0:
                continue
            mask = (df_binary["value"] == v) & (df_binary["variable"] == var_name)
            sub_cell = df_binary[mask]
            n_samp = len(sub_cell)

            if n_samp == 0:
                cell_results.append({
                    "variable": var_name, "value": v,
                    "n_samp": 0, "n_pop": n_pop,
                    "err_rate": np.nan, "est_errors": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan,
                })
                print(f"  {var_name:>20s}  {v:>3d}  {0:>6d}  {n_pop:>6d}  "
                      f"{'N/A':>8s}  {'N/A':>10s}  {'N/A':>20s}")
                continue

            y_cell = sub_cell["human_incorrect"].values.astype(int)
            err_rate = y_cell.mean()
            est_errors = err_rate * n_pop

            boot_errors = np.empty(N_BOOT_EXTRAP)
            for b in range(N_BOOT_EXTRAP):
                idx_b = rng_extrap.choice(n_samp, size=n_samp, replace=True)
                rate_b = y_cell[idx_b].mean()
                boot_errors[b] = rate_b * n_pop

            ci_lo = np.percentile(boot_errors, BOOTSTRAP_ALPHA / 2 * 100)
            ci_hi = np.percentile(boot_errors, (1 - BOOTSTRAP_ALPHA / 2) * 100)

            cell_results.append({
                "variable": var_name, "value": v,
                "n_samp": n_samp, "n_pop": n_pop,
                "err_rate": err_rate, "est_errors": est_errors,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

            print(f"  {var_name:>20s}  {v:>3d}  {n_samp:>6d}  {n_pop:>6d}  "
                  f"{err_rate:>8.4f}  {est_errors:>10.1f}  "
                  f"[{ci_lo:>8.1f}, {ci_hi:>8.1f}]")

    df_extrap = pd.DataFrame(cell_results)

    # Per-variable totals (strict)
    print(f"\n--- Per-variable extrapolated error totals ---")
    print(f"  {'variable':>20s}  {'N_pop':>6s}  {'est_errors':>10s}  {'95% CI':>20s}  {'err_rate':>8s}")
    print("  " + "-" * 70)

    rng_joint = np.random.RandomState(RANDOM_SEED)
    cell_boot_arrays = {}
    for _, row in df_extrap.iterrows():
        var_name = row["variable"]
        v = row["value"]
        n_pop = row["n_pop"]
        mask = (df_binary["value"] == v) & (df_binary["variable"] == var_name)
        sub_cell = df_binary[mask]
        n_samp = len(sub_cell)

        if n_samp == 0:
            cell_boot_arrays[(v, var_name)] = np.full(N_BOOT_EXTRAP, np.nan)
            continue

        y_cell = sub_cell["human_incorrect"].values.astype(int)
        boots = np.empty(N_BOOT_EXTRAP)
        for b in range(N_BOOT_EXTRAP):
            idx_b = rng_joint.choice(n_samp, size=n_samp, replace=True)
            boots[b] = y_cell[idx_b].mean() * n_pop
        cell_boot_arrays[(v, var_name)] = boots

    var_totals = []
    for var_name in all_vars_extrap:
        sub_v = df_extrap[df_extrap["variable"] == var_name]
        total_pop = sub_v["n_pop"].sum()
        total_est = sub_v["est_errors"].sum()

        var_boots = np.zeros(N_BOOT_EXTRAP)
        any_valid = False
        for v in val_range_extrap:
            arr = cell_boot_arrays.get((v, var_name))
            if arr is not None and not np.all(np.isnan(arr)):
                var_boots += np.nan_to_num(arr, nan=0.0)
                any_valid = True

        if any_valid:
            ci_lo_v = np.percentile(var_boots, BOOTSTRAP_ALPHA / 2 * 100)
            ci_hi_v = np.percentile(var_boots, (1 - BOOTSTRAP_ALPHA / 2) * 100)
        else:
            ci_lo_v = np.nan
            ci_hi_v = np.nan

        overall_rate = total_est / total_pop if total_pop > 0 else np.nan
        var_totals.append({
            "variable": var_name, "n_pop": total_pop,
            "est_errors": total_est, "ci_lo": ci_lo_v, "ci_hi": ci_hi_v,
            "err_rate": overall_rate
        })
        print(f"  {var_name:>20s}  {total_pop:>6.0f}  {total_est:>10.1f}  "
              f"[{ci_lo_v:>8.1f}, {ci_hi_v:>8.1f}]  {overall_rate:>8.4f}")

    # Overall total (strict)
    overall_boots = np.zeros(N_BOOT_EXTRAP)
    for key, arr in cell_boot_arrays.items():
        if not np.all(np.isnan(arr)):
            overall_boots += np.nan_to_num(arr, nan=0.0)

    total_pop_all = df_extrap["n_pop"].sum()
    total_est_all = df_extrap["est_errors"].sum()
    ci_lo_all = np.percentile(overall_boots, BOOTSTRAP_ALPHA / 2 * 100)
    ci_hi_all = np.percentile(overall_boots, (1 - BOOTSTRAP_ALPHA / 2) * 100)
    overall_rate_all = total_est_all / total_pop_all if total_pop_all > 0 else np.nan

    print(f"\n  {'OVERALL':>20s}  {total_pop_all:>6.0f}  {total_est_all:>10.1f}  "
          f"[{ci_lo_all:>8.1f}, {ci_hi_all:>8.1f}]  {overall_rate_all:>8.4f}")
    print(f"\n  Interpretation: Of {total_pop_all:.0f} total annotations in the population,")
    print(f"  an estimated {total_est_all:.0f} (95% CI: {ci_lo_all:.0f}\u2013{ci_hi_all:.0f}) are")
    print(f"  probable human annotation errors (= Gemma correct + both wrong).")
    print(f"  Overall estimated error rate: {overall_rate_all*100:.2f}%")

    # Per agreement level (strict)
    print(f"\n--- Per agreement level (all variables combined) ---")
    print(f"  {'value':>5s}  {'N_pop':>8s}  {'est_errors':>10s}  {'95% CI':>20s}  {'err_rate':>8s}")
    print("  " + "-" * 60)

    for v in val_range_extrap:
        sub_val = df_extrap[df_extrap["value"] == v]
        n_pop_v = sub_val["n_pop"].sum()
        est_v = sub_val["est_errors"].sum()

        val_boots = np.zeros(N_BOOT_EXTRAP)
        for var_name in all_vars_extrap:
            arr = cell_boot_arrays.get((v, var_name))
            if arr is not None and not np.all(np.isnan(arr)):
                val_boots += np.nan_to_num(arr, nan=0.0)

        ci_lo_val = np.percentile(val_boots, BOOTSTRAP_ALPHA / 2 * 100)
        ci_hi_val = np.percentile(val_boots, (1 - BOOTSTRAP_ALPHA / 2) * 100)
        rate_v = est_v / n_pop_v if n_pop_v > 0 else np.nan

        print(f"  {v:>5d}  {n_pop_v:>8.0f}  {est_v:>10.1f}  "
              f"[{ci_lo_val:>8.1f}, {ci_hi_val:>8.1f}]  {rate_v:>8.4f}")

        # Per-variable breakdown for this agreement level
        hdr = f"  {'Variable':>20s}  {'n_samp':>6s}  {'errors':>6s}  {'err_rate':>8s}  {'n_pop':>6s}  {'est_errors':>10s}"
        print(hdr)
        print("  " + "-" * 65)
        tot_nsamp, tot_errors, tot_npop, tot_est = 0, 0, 0, 0.0
        for var_name in all_vars_extrap:
            row = df_extrap[(df_extrap["value"] == v) & (df_extrap["variable"] == var_name)]
            if row.empty:
                continue
            r = row.iloc[0]
            n_s = int(r["n_samp"])
            n_p = int(r["n_pop"])
            est = r["est_errors"]
            if np.isnan(est):
                continue
            n_err = round(r["err_rate"] * n_s) if not np.isnan(r["err_rate"]) else 0
            rate_str = f"{r['err_rate']*100:.1f}%" if not np.isnan(r["err_rate"]) else "N/A"
            print(f"  {var_name:>20s}  {n_s:>6d}  {n_err:>6d}  {rate_str:>8s}  {n_p:>6d}  {est:>10.1f}")
            tot_nsamp += n_s; tot_errors += n_err; tot_npop += n_p; tot_est += est
        tot_rate_str = f"{tot_errors/tot_nsamp*100:.1f}%" if tot_nsamp > 0 else "N/A"
        print("  " + "-" * 65)
        print(f"  {'Total':>20s}  {tot_nsamp:>6d}  {tot_errors:>6d}  {tot_rate_str:>8s}  {tot_npop:>6d}  {tot_est:>10.1f}")
        print()

    # ---- LIBERAL DEFINITION ----
    print(f"\n--- Extrapolation under LIBERAL definition ---")
    print(f"  (broad = strict + unclear; human error = Gemma correct + both wrong + unclear)")

    cell_results_lib = []
    rng_lib = np.random.RandomState(RANDOM_SEED)

    print(f"\n  {'variable':>20s}  {'val':>3s}  {'n_samp':>6s}  {'N_pop':>6s}  "
          f"{'err_rate':>8s}  {'est_errors':>10s}  {'95% CI':>20s}")
    print("  " + "-" * 80)

    for var_name in all_vars_extrap:
        for v in val_range_extrap:
            n_pop = pop_counts.get((v, var_name), 0)
            if n_pop == 0:
                continue
            mask = (df_binary_broad["value"] == v) & (df_binary_broad["variable"] == var_name)
            sub_cell = df_binary_broad[mask]
            n_samp = len(sub_cell)

            if n_samp == 0:
                cell_results_lib.append({
                    "variable": var_name, "value": v,
                    "n_samp": 0, "n_pop": n_pop,
                    "err_rate": np.nan, "est_errors": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan,
                })
                print(f"  {var_name:>20s}  {v:>3d}  {0:>6d}  {n_pop:>6d}  "
                      f"{'N/A':>8s}  {'N/A':>10s}  {'N/A':>20s}")
                continue

            y_cell = sub_cell["human_incorrect_broad"].values.astype(int)
            err_rate = y_cell.mean()
            est_errors = err_rate * n_pop

            boot_errors = np.empty(N_BOOT_EXTRAP)
            for b in range(N_BOOT_EXTRAP):
                idx_b = rng_lib.choice(n_samp, size=n_samp, replace=True)
                boot_errors[b] = y_cell[idx_b].mean() * n_pop

            ci_lo = np.percentile(boot_errors, BOOTSTRAP_ALPHA / 2 * 100)
            ci_hi = np.percentile(boot_errors, (1 - BOOTSTRAP_ALPHA / 2) * 100)

            cell_results_lib.append({
                "variable": var_name, "value": v,
                "n_samp": n_samp, "n_pop": n_pop,
                "err_rate": err_rate, "est_errors": est_errors,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

            print(f"  {var_name:>20s}  {v:>3d}  {n_samp:>6d}  {n_pop:>6d}  "
                  f"{err_rate:>8.4f}  {est_errors:>10.1f}  "
                  f"[{ci_lo:>8.1f}, {ci_hi:>8.1f}]")

    df_extrap_lib = pd.DataFrame(cell_results_lib)

    # Per-variable totals (broad)
    print(f"\n--- Per-variable extrapolated error totals (broad) ---")
    print(f"  {'variable':>20s}  {'N_pop':>6s}  {'est_errors':>10s}  {'95% CI':>20s}  {'err_rate':>8s}")
    print("  " + "-" * 70)

    rng_joint_lib = np.random.RandomState(RANDOM_SEED)
    cell_boot_arrays_lib = {}
    for _, row in df_extrap_lib.iterrows():
        var_name = row["variable"]
        v = row["value"]
        n_pop = row["n_pop"]
        mask = (df_binary_broad["value"] == v) & (df_binary_broad["variable"] == var_name)
        sub_cell = df_binary_broad[mask]
        n_samp = len(sub_cell)
        if n_samp == 0:
            cell_boot_arrays_lib[(v, var_name)] = np.full(N_BOOT_EXTRAP, np.nan)
            continue
        y_cell = sub_cell["human_incorrect_broad"].values.astype(int)
        boots = np.empty(N_BOOT_EXTRAP)
        for b in range(N_BOOT_EXTRAP):
            idx_b = rng_joint_lib.choice(n_samp, size=n_samp, replace=True)
            boots[b] = y_cell[idx_b].mean() * n_pop
        cell_boot_arrays_lib[(v, var_name)] = boots

    var_totals_lib = []
    for var_name in all_vars_extrap:
        sub_v = df_extrap_lib[df_extrap_lib["variable"] == var_name]
        total_pop = sub_v["n_pop"].sum()
        total_est = sub_v["est_errors"].sum()
        var_boots = np.zeros(N_BOOT_EXTRAP)
        any_valid = False
        for v in val_range_extrap:
            arr = cell_boot_arrays_lib.get((v, var_name))
            if arr is not None and not np.all(np.isnan(arr)):
                var_boots += np.nan_to_num(arr, nan=0.0)
                any_valid = True
        if any_valid:
            ci_lo_v = np.percentile(var_boots, BOOTSTRAP_ALPHA / 2 * 100)
            ci_hi_v = np.percentile(var_boots, (1 - BOOTSTRAP_ALPHA / 2) * 100)
        else:
            ci_lo_v = np.nan
            ci_hi_v = np.nan
        overall_rate = total_est / total_pop if total_pop > 0 else np.nan
        var_totals_lib.append({
            "variable": var_name, "n_pop": total_pop,
            "est_errors": total_est, "ci_lo": ci_lo_v, "ci_hi": ci_hi_v,
            "err_rate": overall_rate
        })
        print(f"  {var_name:>20s}  {total_pop:>6.0f}  {total_est:>10.1f}  "
              f"[{ci_lo_v:>8.1f}, {ci_hi_v:>8.1f}]  {overall_rate:>8.4f}")

    # Overall totals (broad)
    overall_boots_lib = np.zeros(N_BOOT_EXTRAP)
    for key, arr in cell_boot_arrays_lib.items():
        if not np.all(np.isnan(arr)):
            overall_boots_lib += np.nan_to_num(arr, nan=0.0)

    total_pop_all_lib = df_extrap_lib["n_pop"].sum()
    total_est_all_lib = df_extrap_lib["est_errors"].sum()
    ci_lo_all_lib = np.percentile(overall_boots_lib, BOOTSTRAP_ALPHA / 2 * 100)
    ci_hi_all_lib = np.percentile(overall_boots_lib, (1 - BOOTSTRAP_ALPHA / 2) * 100)
    overall_rate_all_lib = total_est_all_lib / total_pop_all_lib if total_pop_all_lib > 0 else np.nan

    print(f"\n  {'OVERALL':>20s}  {total_pop_all_lib:>6.0f}  {total_est_all_lib:>10.1f}  "
          f"[{ci_lo_all_lib:>8.1f}, {ci_hi_all_lib:>8.1f}]  {overall_rate_all_lib:>8.4f}")
    print(f"\n  Interpretation (broad): Of {total_pop_all_lib:.0f} total annotations,")
    print(f"  an estimated {total_est_all_lib:.0f} (95% CI: {ci_lo_all_lib:.0f}\u2013{ci_hi_all_lib:.0f}) are")
    print(f"  probable human annotation errors.")
    print(f"  Overall estimated error rate (broad): {overall_rate_all_lib*100:.2f}%")

    # Per agreement level (broad)
    print(f"\n--- Per agreement level, broad (all variables combined) ---")
    print(f"  {'value':>5s}  {'N_pop':>8s}  {'est_errors':>10s}  {'95% CI':>20s}  {'err_rate':>8s}")
    print("  " + "-" * 60)

    for v in val_range_extrap:
        sub_val = df_extrap_lib[df_extrap_lib["value"] == v]
        n_pop_v = sub_val["n_pop"].sum()
        est_v = sub_val["est_errors"].sum()
        val_boots = np.zeros(N_BOOT_EXTRAP)
        for var_name in all_vars_extrap:
            arr = cell_boot_arrays_lib.get((v, var_name))
            if arr is not None and not np.all(np.isnan(arr)):
                val_boots += np.nan_to_num(arr, nan=0.0)
        ci_lo_val = np.percentile(val_boots, BOOTSTRAP_ALPHA / 2 * 100)
        ci_hi_val = np.percentile(val_boots, (1 - BOOTSTRAP_ALPHA / 2) * 100)
        rate_v = est_v / n_pop_v if n_pop_v > 0 else np.nan
        print(f"  {v:>5d}  {n_pop_v:>8.0f}  {est_v:>10.1f}  "
              f"[{ci_lo_val:>8.1f}, {ci_hi_val:>8.1f}]  {rate_v:>8.4f}")

        # Per-variable breakdown for this agreement level (broad)
        hdr = f"  {'Variable':>20s}  {'n_samp':>6s}  {'errors':>6s}  {'err_rate':>8s}  {'n_pop':>6s}  {'est_errors':>10s}"
        print(hdr)
        print("  " + "-" * 65)
        tot_nsamp, tot_errors, tot_npop, tot_est = 0, 0, 0, 0.0
        for var_name in all_vars_extrap:
            row = df_extrap_lib[(df_extrap_lib["value"] == v) & (df_extrap_lib["variable"] == var_name)]
            if row.empty:
                continue
            r = row.iloc[0]
            n_s = int(r["n_samp"])
            n_p = int(r["n_pop"])
            est = r["est_errors"]
            if np.isnan(est):
                continue
            n_err = round(r["err_rate"] * n_s) if not np.isnan(r["err_rate"]) else 0
            rate_str = f"{r['err_rate']*100:.1f}%" if not np.isnan(r["err_rate"]) else "N/A"
            print(f"  {var_name:>20s}  {n_s:>6d}  {n_err:>6d}  {rate_str:>8s}  {n_p:>6d}  {est:>10.1f}")
            tot_nsamp += n_s; tot_errors += n_err; tot_npop += n_p; tot_est += est
        tot_rate_str = f"{tot_errors/tot_nsamp*100:.1f}%" if tot_nsamp > 0 else "N/A"
        print("  " + "-" * 65)
        print(f"  {'Total':>20s}  {tot_nsamp:>6d}  {tot_errors:>6d}  {tot_rate_str:>8s}  {tot_npop:>6d}  {tot_est:>10.1f}")
        print()

    # ---- PLOT COMBINED FIGURE E+F ----
    print(f"\n  Generating combined extrapolated errors figure (strict + broad)...")
    plot_extrapolated_errors_combined(
        df_ext_strict=df_extrap,
        cell_boots_strict=cell_boot_arrays,
        var_tots_strict=var_totals,
        total_est_strict=total_est_all,
        total_pop_strict=total_pop_all,
        ci_lo_strict=ci_lo_all,
        ci_hi_strict=ci_hi_all,
        overall_rate_strict=overall_rate_all,
        df_ext_lib=df_extrap_lib,
        cell_boots_lib=cell_boot_arrays_lib,
        var_tots_lib=var_totals_lib,
        total_est_lib=total_est_all_lib,
        total_pop_lib=total_pop_all_lib,
        ci_lo_lib=ci_lo_all_lib,
        ci_hi_lib=ci_hi_all_lib,
        overall_rate_lib=overall_rate_all_lib,
        val_range_extrap=val_range_extrap,
        all_vars_extrap=all_vars_extrap,
    )


else:
    print("  Skipping extrapolation (no prevalence file available).")
    print("  Re-run with population_data.xlsx to enable this section.")

print()


# ============================================================
# 10. SUMMARY
# ============================================================
print()
print("=" * 80)
print("10. SUMMARY")
print("=" * 80)
print(f"""
Key findings:
  - LLM agreement score (0-4) is kept as-is: higher = more LLMs agree with human annotator
  - gemma_out is binary (0/1): 1 = gemma matched human
  - Lower agreement score and gemma_out=0 predict human annotation errors

Outcome definitions:
  human_incorrect (strict)  = Gemma correct + both wrong  (human was wrong)
  human_incorrect (broad) = Strict + Unclear  (i.e. Gemma correct + both wrong + unclear)
  human_correct   = Human correct + both correct (human was right)
  llm_incorrect   = Human correct + both wrong  (LLM was wrong)
  llm_correct     = Gemma correct + both correct (LLM was right)
  unclear (= "unclear" + "not decidable") -> excluded from strict, counted as error in broad

Bootstrap confidence intervals:
  - Method: stratified non-parametric bootstrap ({N_BOOTSTRAP} replicates, seed={RANDOM_SEED})
  - Stratification: resampled within positive/negative class separately
  - CI: percentile method at {(1 - BOOTSTRAP_ALPHA) * 100:.0f}% level

Prevalence adjustment:
  Per-(value, variable) importance weights:
    weight(v, var) = N_population(v, var) / n_sample(v, var)

Figures saved:
  figure_agreement_distribution.png/.svg   — Plots 4, 8 (agreement distributions)
  figure_prevalence_distribution.png/.svg  — Plot 7 (prevalence by annotation category)
  figure_error_rates.png/.svg              — Plots 1, 2, 3 (error rates & adjudication vote breakdown)
  figure_gemma.png/.svg                    — Plots 5, 6, 9 (gemma analysis, ROC, PR curves)
  figure_sensitivity_roc_pr.png/.svg       — Figure D: ROC & PR strict vs broad outcome
  figure_extrapolated_errors_combined.png/.svg — Figures E+F: extrapolated errors (strict top, broad bottom)
""")

print("Done.")
