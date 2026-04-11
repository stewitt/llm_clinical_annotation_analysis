import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Extraction & Normalization Logic
# ---------------------------------------------------------
def extract_numbers(text):
    if not isinstance(text, str):
        return set()
    text = text.replace("–", "-").replace("—", "-")
    matches = re.findall(r"\d+[.,]?\d*", text)
    nums = {float(m.replace(",", ".")) for m in matches}
    return nums

# Location values as stored in the source data (mix of German and Latin anatomical terms).
# English equivalents: Zökum=Cecum, rechte/linke Flexur=Right/Left Flexure,
# Sigma=Sigmoid Colon, Rektum=Rectum, unklar=unclear, siehe_Distanz=see distance column
VALID_LOCATIONS = [
    "Zökum-linke Flexur", "unklar", "rechte Flexur", "linke Flexur",
    "siehe_Distanz", "Zökum", "Colon ascendens", "Colon transversum",
    "Colon descendens", "Sigma", "Rektum",
]
VALID_LOCATIONS_LOWER = {v.lower(): v for v in VALID_LOCATIONS}

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

# ---------------------------------------------------------
# 2. Enhanced Statistical Functions
# ---------------------------------------------------------
def compute_metrics_with_errors(a, b, n_total=None):
    n = len(a)
    if n == 0:
        return [0] * 8
    if n_total is None:
        n_total = n

    kappa = cohen_kappa_score(a, b)

    # pct_agree uses n_total so NaN-excluded rows count as mismatches
    a = np.asarray(a)
    b = np.asarray(b)
    matches = np.sum(a == b)
    po = matches / n_total
    categories = sorted(set(a) | set(b))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    P = np.zeros((k, k))
    for ai, bi in zip(a, b):
        P[cat_idx[ai], cat_idx[bi]] += 1
    P /= n

    p_row = P.sum(axis=1)   # rater-a marginals
    p_col = P.sum(axis=0)   # rater-b marginals
    pe = float(np.dot(p_row, p_col))

    # Full asymptotic variance (Fleiss, Cohen & Everitt, 1969)
    # Var(κ) = (A + B - C) / [n * (1-pe)^4]
    term_A = sum(
        P[i, i] * (1 - (p_row[i] + p_col[i]) * (1 - kappa)) ** 2
        for i in range(k)
    )
    term_B = (1 - kappa) ** 2 * sum(
        P[i, j] * (p_row[i] + p_col[j]) ** 2
        for i in range(k) for j in range(k) if i != j
    )
    term_C = (kappa * pe - sum(P[i, i] * (p_row[i] + p_col[i]) for i in range(k))) ** 2

    denom4 = (1 - pe) ** 4
    var_kappa = (term_A + term_B - term_C) / (n * denom4) if denom4 != 0 else 0
    se_kappa = np.sqrt(max(var_kappa, 0))
    k_ci_low = kappa - 1.96 * se_kappa
    k_ci_high = kappa + 1.96 * se_kappa

    pct_agree = po * 100
    se_pct = np.sqrt((po * (1 - po)) / n_total) * 100
    p_ci_low = max(0, pct_agree - 1.96 * se_pct)
    p_ci_high = min(100, pct_agree + 1.96 * se_pct)

    return (kappa, se_kappa, k_ci_low, k_ci_high,
            pct_agree, se_pct, p_ci_low, p_ci_high)

# ---------------------------------------------------------
# 3. Data Loading & Preparation
# ---------------------------------------------------------
df = pd.read_excel('../inter_rater_data.xlsx')
df = df[df['index'] >= 30].reset_index(drop=True)
df = df.fillna("nan")

# Source column names are German as they appear in the original data file.
# Abtragungsstatus = Resection Status, mehrere_Polypen = Multiple Polyps,
# Durchm1_mm/Durchm2_mm = Diameter 1/2 (mm), Lage = Location,
# Distanz_a.a. = Distance (used when location is recorded as a numeric distance),
# siehe_Distanz = "see distance" (flag indicating distance column should be used instead),
# ja/nein = yes/no
df['en_bloc_out_human'] = df['Abtragungsstatus']
df['multiple_polyps_human'] = (
    df['mehrere_Polypen']
    .astype(str).str.strip().str.lower()
    .map({'ja': 1, 'nein': 0, 'nan': 2})
)
df['num1_mm_human'] = df['Durchm1_mm']
df['num2_mm_human'] = df['Durchm2_mm']
df['location_mapped_human'] = df['Lage'].astype(str).str.strip()

mask = df['location_mapped_human'].str.lower() == "siehe_distanz"
df.loc[mask, 'location_mapped_human'] = df.loc[mask, 'Distanz_a.a.'].astype(str).str.strip()

# ---------------------------------------------------------
# 4. Pairwise Analysis
# ---------------------------------------------------------
base_cols = ['en_bloc_out', 'multiple_polyps', 'num1_mm', 'num2_mm']
suffixes = ['deepseek', 'gemma', 'gpt', 'mistral', 'human']
results = []

for base in base_cols + ['location_mapped']:
    for suf_a, suf_b in combinations(suffixes, 2):
        col_a = f"{base}_{suf_a}"
        col_b = f"{base}_{suf_b}"

        if col_a not in df.columns or col_b not in df.columns:
            continue

        mask = (df[col_a] != "nan") & (df[col_b] != "nan")
        n = mask.sum()
        if n == 0:
            continue

        a = df.loc[mask, col_a]
        b = df.loc[mask, col_b]

        if base == 'location_mapped':
            a = a.apply(normalize_location_domain)
            b = b.apply(normalize_location_domain)
            dist_ref = df.loc[mask, "Distanz_a.a."].apply(normalize_location_domain)
            if suf_a == 'human':
                a = np.where(b == dist_ref, dist_ref, a)
            if suf_b == 'human':
                b = np.where(a == dist_ref, dist_ref, b)

        n_total = len(df) if ('human' in (suf_a, suf_b)) else n
        metrics = compute_metrics_with_errors(a, b, n_total=n_total)

        results.append({
            'variable': base,
            'rater_1': suf_a,
            'rater_2': suf_b,
            'kappa': metrics[0],
            'k_se': metrics[1],
            'k_ci_low': metrics[2],
            'k_ci_high': metrics[3],
            'pct_agree': metrics[4],
            'pct_se': metrics[5],
            'pct_ci_low': metrics[6],
            'pct_ci_high': metrics[7],
            'n': n
        })

res_df = pd.DataFrame(results)

# --- Combined pairwise Cohen's κ (macro-average across variables per pair) ---
combined_cohen_rows = []
for suf_a, suf_b in combinations(suffixes, 2):
    sub = res_df[(res_df['rater_1'] == suf_a) & (res_df['rater_2'] == suf_b)]
    if len(sub) == 0:
        continue
    avg_k = sub['kappa'].mean()
    pooled_se = np.sqrt((sub['k_se'] ** 2).mean())
    avg_pct = sub['pct_agree'].mean()
    pooled_pct_se = np.sqrt((sub['pct_se'] ** 2).mean())
    combined_cohen_rows.append({
        'variable': 'Combined',
        'rater_1': suf_a, 'rater_2': suf_b,
        'kappa': avg_k,
        'k_se': pooled_se,
        'k_ci_low': avg_k - 1.96 * pooled_se,
        'k_ci_high': avg_k + 1.96 * pooled_se,
        'pct_agree': avg_pct,
        'pct_se': pooled_pct_se,
        'pct_ci_low': max(0, avg_pct - 1.96 * pooled_pct_se),
        'pct_ci_high': min(100, avg_pct + 1.96 * pooled_pct_se),
        'n': np.nan
    })

res_df = pd.concat([res_df, pd.DataFrame(combined_cohen_rows)], ignore_index=True)

# ---------------------------------------------------------
# 5. Fleiss' Kappa
# ---------------------------------------------------------
def run_fleiss(rater_df):
    arr, _ = aggregate_raters(rater_df.values)
    kappa = fleiss_kappa(arr, method='fleiss')
    n, m = rater_df.shape
    se = np.sqrt(2 / (n * m * (m - 1)))
    return kappa, se, kappa - 1.96 * se, kappa + 1.96 * se

fleiss_results = []
llm_suffixes = ['deepseek', 'gemma', 'gpt', 'mistral']

for base in base_cols + ['location_mapped']:
    groups = [("4 LLMs", llm_suffixes), ("All 5 Raters", suffixes)]
    for group_name, subs in groups:
        cols = [f"{base}_{s}" for s in subs]
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) < 2:
            continue

        temp_df = df[valid_cols].copy()
        if base == 'location_mapped':
            for c in valid_cols:
                temp_df[c] = temp_df[c].apply(normalize_location_domain)

        try:
            k, se, lo, hi = run_fleiss(temp_df)
            fleiss_results.append({
                'variable': base, 'group': group_name, 'raters': len(valid_cols),
                'f_kappa': k, 'se': se, 'ci_low': lo, 'ci_high': hi
            })
        except:
            pass

fleiss_df = pd.DataFrame(fleiss_results)

# --- Combined Fleiss ---
combined_fleiss = []
for group_name in ["4 LLMs", "All 5 Raters"]:
    sub = fleiss_df[fleiss_df['group'] == group_name]
    if len(sub) == 0:
        continue
    avg_k = sub['f_kappa'].mean()
    pooled_se = np.sqrt((sub['se'] ** 2).mean())
    combined_fleiss.append({
        'variable': 'Combined', 'group': group_name, 'raters': sub['raters'].iloc[0],
        'f_kappa': avg_k, 'se': pooled_se,
        'ci_low': avg_k - 1.96 * pooled_se, 'ci_high': avg_k + 1.96 * pooled_se
    })

fleiss_df = pd.concat([fleiss_df, pd.DataFrame(combined_fleiss)], ignore_index=True)

# ---------------------------------------------------------
# 6. Print Tables
# ---------------------------------------------------------
print("\n" + "=" * 90)
print("PAIRWISE ANALYSIS (COHEN'S KAPPA & PERCENTAGE AGREEMENT WITH ERROR)")
print("=" * 90)
out_df = res_df.copy()
out_df['kappa'] = out_df['kappa'].round(4)
out_df['k_se'] = out_df['k_se'].round(4)
out_df['k_ci'] = out_df.apply(lambda r: f"[{r['k_ci_low']:.2f}, {r['k_ci_high']:.2f}]", axis=1)
out_df['pct_agree'] = out_df['pct_agree'].round(1)
out_df['pct_se'] = out_df['pct_se'].round(2)
out_df['pct_ci'] = out_df.apply(lambda r: f"[{r['pct_ci_low']:.1f}, {r['pct_ci_high']:.1f}]", axis=1)
print(out_df[['variable', 'rater_1', 'rater_2', 'kappa', 'k_se', 'k_ci',
              'pct_agree', 'pct_se', 'pct_ci', 'n']].to_string(index=False))

print("\n" + "=" * 90)
print("FLEISS' KAPPA SUMMARY (incl. Combined)")
print("=" * 90)
fl_out = fleiss_df.copy()
fl_out['f_kappa'] = fl_out['f_kappa'].round(4)
fl_out['se'] = fl_out['se'].round(4)
fl_out['ci'] = fl_out.apply(lambda r: f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]", axis=1)
print(fl_out[['variable', 'group', 'raters', 'f_kappa', 'se', 'ci']].to_string(index=False))

# ---------------------------------------------------------
# 7. PLOTS
# ---------------------------------------------------------

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

VAR_LABELS = {
    'en_bloc_out': 'Resection Status',
    'multiple_polyps': 'Multiple Polyps',
    'num1_mm': 'Diameter 1 (mm)',
    'num2_mm': 'Diameter 2 (mm)',
    'location_mapped': 'Location',
    'Combined': 'Combined'
}

RATER_LABELS = {
    'deepseek': 'DeepSeek',
    'gemma': 'Gemma',
    'gpt': 'GPT-OSS',
    'mistral': 'Mistral',
    'human': 'Human'
}

all_raters = ['gemma', 'deepseek', 'gpt', 'mistral', 'human']
var_list = ['location_mapped', 'num1_mm', 'num2_mm', 'en_bloc_out', 'multiple_polyps']

# ============================================================
# PLOT 1: Cohen's Kappa – Lower-Triangle Heatmaps, 2 rows
# ============================================================

# Colormap tuned for the 0.3–1.0 range
cmap = mcolors.LinearSegmentedColormap.from_list(
    'kappa_cmap',
    [(0.0, '#d73027'),    # red        (κ = 0.30)
     (0.2, '#fc8d59'),    # orange     (κ = 0.44)
     (0.4, '#fee08b'),    # yellow     (κ = 0.58)
     (0.6, '#d9ef8b'),    # lime       (κ = 0.72)
     (0.8, '#66bd63'),    # green      (κ = 0.86)
     (1.0, '#1a9850')],   # dark green (κ = 1.00)
)

fig1, axes_flat = plt.subplots(2, 3, figsize=(14, 9))
ax_list = [axes_flat[0, 0], axes_flat[0, 1], axes_flat[0, 2],
           axes_flat[1, 0], axes_flat[1, 1], axes_flat[1, 2]]

nr = len(all_raters)
var_list_heatmap = var_list + ['Combined']

for ax, var in zip(ax_list, var_list_heatmap):
    # Build symmetric kappa and SE matrices
    mat = pd.DataFrame(np.nan, index=all_raters, columns=all_raters)
    mat_se = pd.DataFrame(np.nan, index=all_raters, columns=all_raters)
    for _, row in res_df[res_df['variable'] == var].iterrows():
        r1, r2 = row['rater_1'], row['rater_2']
        mat.loc[r1, r2] = row['kappa']
        mat.loc[r2, r1] = row['kappa']
        mat_se.loc[r1, r2] = row['k_se']
        mat_se.loc[r2, r1] = row['k_se']
    for r in all_raters:
        mat.loc[r, r] = 1.0
        mat_se.loc[r, r] = 0.0

    data = mat.values.astype(float)
    data_se = mat_se.values.astype(float)

    # Build display array: lower triangle + diagonal, upper = NaN
    display = data.copy()
    for i in range(nr):
        for j in range(nr):
            if j > i:
                display[i, j] = np.nan

    # Plot with masked NaNs
    masked = np.ma.masked_invalid(display)
    im = ax.pcolormesh(np.arange(nr + 1) - 0.5, np.arange(nr + 1) - 0.5,
                       masked, cmap=cmap, vmin=0.3, vmax=1.0,
                       edgecolors='white', linewidth=1.5)

    # Annotate lower triangle + diagonal – kappa value with ±SE
    for i in range(nr):
        for j in range(nr):
            if j > i:
                continue
            val = data[i, j]
            if np.isnan(val):
                continue
            se = data_se[i, j]
            if i == j or np.isnan(se):
                label = f"{val:.2f}"
            else:
                label = f"{val:.2f}\n±{se:.2f}"
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='black', linespacing=1.3)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    tick_labels = [RATER_LABELS[r] for r in all_raters]
    ax.set_xticks(range(nr))
    ax.set_xticklabels(tick_labels, fontsize=12, rotation=45, ha='right')
    ax.set_yticks(range(nr))
    ax.set_yticklabels(tick_labels, fontsize=12)
    ax.set_xlim(-0.5, nr - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_title(VAR_LABELS.get(var, var), fontsize=15, fontweight='bold', pad=10)
    ax.tick_params(length=0)

    # Hide upper-triangle background
    ax.set_facecolor('white')

fig1.suptitle("Pairwise Cohen's κ", fontsize=18, fontweight='bold', y=0.98)

# Shared colorbar on the right
cbar_ax = fig1.add_axes([0.96, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.3, vmax=1.0))
sm.set_array([])
fig1.colorbar(sm, cax=cbar_ax, label="Cohen's κ")

fig1.tight_layout(rect=[0, 0, 0.95, 0.95])
fig1.savefig('plot1_kappa_heatmaps.svg', bbox_inches='tight')
print("\nSaved: plot1_kappa_heatmaps.svg")

# ============================================================
# PLOT 2: Pct Agreement – LLM vs Human only + Combined
# ============================================================
human_df = res_df[
    ((res_df['rater_1'] == 'human') | (res_df['rater_2'] == 'human'))
    & (res_df['variable'] != 'Combined')
].copy()

human_df['llm'] = human_df.apply(
    lambda r: r['rater_1'] if r['rater_2'] == 'human' else r['rater_2'], axis=1
)

# Combined (macro-average across variables) per LLM
combined_rows = []
for llm in ['deepseek', 'gemma', 'gpt', 'mistral']:
    sub = human_df[human_df['llm'] == llm]
    if len(sub) == 0:
        continue
    avg_pct = sub['pct_agree'].mean()
    pooled_se = np.sqrt((sub['pct_se'] ** 2).mean())
    combined_rows.append({
        'variable': 'Combined',
        'rater_1': llm, 'rater_2': 'human',
        'kappa': np.nan, 'k_se': np.nan, 'k_ci_low': np.nan, 'k_ci_high': np.nan,
        'pct_agree': avg_pct, 'pct_se': pooled_se,
        'pct_ci_low': max(0, avg_pct - 1.96 * pooled_se),
        'pct_ci_high': min(100, avg_pct + 1.96 * pooled_se),
        'n': np.nan, 'llm': llm
    })

human_df = pd.concat([human_df, pd.DataFrame(combined_rows)], ignore_index=True)

llm_order = ['gemma', 'deepseek', 'gpt', 'mistral']
llm_colors = {
    'deepseek': '#1ABC9C',
    'gemma': '#3498DB',
    'gpt': '#9B59B6',
    'mistral': '#E67E22'
}

var_list_ext = var_list + ['Combined']

fig2, ax2 = plt.subplots(figsize=(14, 5.5))

n_llms = len(llm_order)
n_vars_plot = len(var_list_ext)
bar_width = 0.18
x_base = np.arange(n_vars_plot)

for i, llm in enumerate(llm_order):
    sub = human_df[human_df['llm'] == llm].set_index('variable').reindex(var_list_ext)
    offset = (i - (n_llms - 1) / 2) * bar_width
    vals = sub['pct_agree'].values
    errs = sub['pct_se'].values * 1.96
    bars = ax2.bar(x_base + offset, vals, bar_width, yerr=errs, capsize=3,
                   label=RATER_LABELS[llm], color=llm_colors[llm],
                   edgecolor='white', linewidth=0.5, alpha=0.88)
    for bar, v, e in zip(bars, vals, errs):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + e + 1.0,
                     f"{v:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

sep_x = n_vars_plot - 1.5
ax2.axvline(sep_x, color='grey', ls='--', lw=0.8, alpha=0.5)

ax2.set_xticks(x_base)
ax2.set_xticklabels([VAR_LABELS.get(v, v) for v in var_list_ext], fontsize=12)
ax2.set_ylabel("Agreement (%)", fontsize=14)
ax2.set_ylim(0, 109)
ax2.set_title("Percentage Agreement: LLM vs Human (with 95% CI)", fontsize=15, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig2.tight_layout()
fig2.savefig('plot2_pct_agreement_vs_human.svg', bbox_inches='tight')
print("Saved: plot2_pct_agreement_vs_human.svg")

# ============================================================
# PLOT 3: Fleiss' Kappa (incl. Combined)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 5))

group_colors = {'4 LLMs': '#2ECC71', 'All 5 Raters': '#9B59B6'}

groups = fleiss_df['group'].unique()
n_groups = len(groups)
bar_width = 0.35
var_order = var_list + ['Combined']

x = np.arange(len(var_order))

ax3.axhline(0.61, color='grey', ls='--', lw=0.7, alpha=0.5, zorder=0)
ax3.axhline(0.81, color='grey', ls=':', lw=0.7, alpha=0.5, zorder=0)

for i, grp in enumerate(groups):
    sub = fleiss_df[fleiss_df['group'] == grp].set_index('variable').reindex(var_order)
    offset = (i - (n_groups - 1) / 2) * bar_width
    yerr_low = sub['f_kappa'] - sub['ci_low']
    yerr_high = sub['ci_high'] - sub['f_kappa']
    ax3.bar(x + offset, sub['f_kappa'], bar_width,
            yerr=[yerr_low, yerr_high], capsize=4,
            label=grp, color=group_colors[grp], edgecolor='white', alpha=0.85)

sep_x = len(var_order) - 1.5
ax3.axvline(sep_x, color='grey', ls='--', lw=0.8, alpha=0.5)

ax3.set_xticks(x)
ax3.set_xticklabels([VAR_LABELS.get(v, v) for v in var_order], fontsize=12)
ax3.set_ylabel("Fleiss' κ", fontsize=14)
ax3.set_title("Fleiss' Kappa by Variable (incl. Combined) with 95% CI", fontsize=15, fontweight='bold')
ax3.legend(fontsize=11, loc='upper left')
ax3.set_ylim(0, 1.05)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
fig3.tight_layout()
fig3.savefig('plot3_fleiss_kappa.svg', bbox_inches='tight')
print("Saved: plot3_fleiss_kappa.svg")

plt.show()
print("\nDone. All 3 plots saved.")
