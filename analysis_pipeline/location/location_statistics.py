import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patheffects
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────
# STYLE CONFIGURATION — Publication-quality medical journal look
# ─────────────────────────────────────────────────────────────

# Use a clean base
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#444444',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.edgecolor': '#cccccc',
    'legend.fancybox': True,
    'legend.shadow': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.dpi': 150,
})

# ── Colour palette: 5 distinct, colourblind-friendly, modern medical ──
PALETTE = {
    'Human Annotations': '#2D3142',   # Dark slate (reference)
    'Gemma':             '#4C9F70',   # Muted emerald
    'Deepseek':          '#D4763C',   # Warm terracotta
    'GPT':               '#4A7FB5',   # Soft steel-blue
    'Mistral':           '#9B5DE5',   # Muted violet
}
HATCH_PATTERNS = {
    'Human Annotations': '',
    'Gemma':             '',
    'Deepseek':          '',
    'GPT':               '',
    'Mistral':           '',
}

RATER_ORDER = ['Human Annotations', 'Gemma', 'Deepseek', 'GPT', 'Mistral']

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_excel('../inter_rater_data.xlsx')
df = df[df["index"] >= 30]
# Location values as stored in the source data (mix of German and Latin anatomical terms).
# English equivalents: Zökum=Cecum, rechte/linke Flexur=Right/Left Flexure,
# Sigma=Sigmoid Colon, Rektum=Rectum
allowed_lage = ["Zökum", "Colon ascendens", "rechte Flexur", "Colon transversum",
                "linke Flexur", "Colon descendens", "Sigma", "Rektum"]

raters = {
    "Human Annotations": {"loc": "Lage"},
    "Gemma":        {"loc": "location_mapped_gemma"},
    "Deepseek":     {"loc": "location_mapped_deepseek"},
    "GPT":          {"loc": "location_mapped_gpt"},
    "Mistral":      {"loc": "location_mapped_mistral"},
}

Z_SCORE = 1.96
results_dist_lage = []

# ─────────────────────────────────────────────────────────────
# 2. DATA PROCESSING
# ─────────────────────────────────────────────────────────────
for rater_name, cols in raters.items():
    # A. Distribution: Location
    v_lage = df[df[cols['loc']].isin(allowed_lage)][cols['loc']]
    c_lage = v_lage.value_counts()
    t_lage = len(v_lage)
    for loc in allowed_lage:
        p = (c_lage.get(loc, 0) / t_lage) if t_lage > 0 else 0
        ci = Z_SCORE * np.sqrt(p * (1 - p) / t_lage) * 100 if t_lage > 0 else 0
        results_dist_lage.append({'Rater': rater_name, 'Lage': loc, 'Pct': p * 100, 'CI_Pct': ci})

df_dist_l = pd.DataFrame(results_dist_lage)

# ─────────────────────────────────────────────────────────────
# 3. BEAUTIFUL PLOTTING ENGINE
# ─────────────────────────────────────────────────────────────

def add_subtle_grid(ax, axis='y'):
    ax.grid(axis=axis, color='#E0E0E0', linewidth=0.5, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def add_value_labels(ax, bars, fmt='{:.1f}', fontsize=9, offset=2, errors=None,
                     rotation=0):
    """Add value labels on top of each bar (above error caps when errors given)."""
    for idx, bar in enumerate(bars):
        h = bar.get_height()
        if h > 0 and not np.isnan(h):
            top = h + (errors[idx] if errors is not None else 0)
            ha = 'center'
            ax.annotate(fmt.format(h),
                        xy=(bar.get_x() + bar.get_width() / 2, top),
                        xytext=(0, offset),
                        textcoords='offset points',
                        ha=ha, va='bottom',
                        fontsize=fontsize, color='#555555',
                        fontweight='medium',
                        rotation=rotation)


def plot_comparison(data, x_col, y_col, categories, title, ylabel,
                    err_col=None, figsize=(13, 6.5), show_values=True,
                    value_fmt='{:.1f}', subtitle=None,
                    ylim_pad=1.15, category_labels=None,
                    value_rotation=0):
    """
    Publication-quality grouped bar chart.
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_cats = len(categories)
    n_raters = len(RATER_ORDER)
    total_group_width = 0.78
    bar_width = total_group_width / n_raters
    x = np.arange(n_cats)

    all_bars = []
    for i, rater in enumerate(RATER_ORDER):
        subset = data[data['Rater'] == rater].set_index(x_col).reindex(categories)
        vals = subset[y_col].fillna(0).values
        err  = subset[err_col].fillna(0).values if err_col else None
        offset = x + (i - n_raters / 2 + 0.5) * bar_width

        bars = ax.bar(
            offset, vals, bar_width,
            yerr=err,
            label=rater,
            color=PALETTE[rater],
            edgecolor='white',
            linewidth=0.6,
            capsize=2.5,
            error_kw={'elinewidth': 1.0, 'capthick': 0.8, 'color': '#555555', 'alpha': 0.7},
            zorder=3,
            alpha=0.92,
        )
        all_bars.append(bars)

        if show_values:
            add_value_labels(ax, bars, fmt=value_fmt, fontsize=9, offset=3,
                             errors=err, rotation=value_rotation)

    # Styling
    add_subtle_grid(ax)

    # Title
    ax.set_title(title, pad=20, loc='left', fontsize=15, fontweight='bold', color='#2D3142')

    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=12, color='#777777', style='italic', va='bottom')

    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xticks(x)
    display_labels = category_labels if category_labels is not None else categories
    ax.set_xticklabels(display_labels, rotation=30 if n_cats > 4 else 0,
                       ha='right' if n_cats > 4 else 'center',
                       fontsize=12)

    # Y-axis: add a bit of headroom for value labels
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * ylim_pad)

    # Legend — outside, neat
    leg = ax.legend(
        loc='upper left', bbox_to_anchor=(1.01, 1),
        borderaxespad=0, frameon=True,
        facecolor='white', edgecolor='#dddddd',
        title='Rater / Model', title_fontproperties={'weight': 'bold', 'size': 11},
        handlelength=1.2, handleheight=0.9,
    )
    leg.get_frame().set_linewidth(0.6)

    # Thin bottom spine accent
    ax.spines['bottom'].set_color('#999999')
    ax.spines['left'].set_color('#999999')

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 4. GENERATE PLOT
# ─────────────────────────────────────────────────────────────

lage_english = ["Cecum", "Ascending Colon", "Right Flexure", "Transverse Colon",
                "Left Flexure", "Descending Colon", "Sigmoid Colon", "Rectum"]

fig1 = plot_comparison(
    df_dist_l, 'Lage', 'Pct', allowed_lage,
    "Distribution of Polyp Locations",
    "Proportion (%)",
    err_col="CI_Pct",
    subtitle="Percentage of polyps per anatomical location  ·  Error bars: 95 % CI",
    figsize=(14, 6.5),
    category_labels=lage_english,
    value_rotation=90,
)
fig1.savefig('fig1_location_distribution.svg')

print("fig1_location_distribution.svg saved.")
plt.show()
