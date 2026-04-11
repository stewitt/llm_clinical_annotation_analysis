"""
Plotting functions for blinded annotation analysis.

Figures:
  A — Distribution plots (agreement per category, prevalence, overall)
  B — Error rate plots (human vs LLM error, adjudication vote stacked bars)
  C — Gemma analysis (who was right when gemma != human)
  D — Sensitivity analysis ROC & PR (strict vs broad)
  E/F — Extrapolated errors: strict (left column) + broad (right column), combined 6-panel figure
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR, N_BOOTSTRAP, BOOTSTRAP_ALPHA, RANDOM_SEED,
    COLORS_CAT, CATEGORIES, VAR_LABEL_MAP, VAR_ORDER, VAR_COLORS,
)

def _order_vars(names):
    """Return names in VAR_ORDER, with any unknowns appended alphabetically."""
    ordered = [v for v in VAR_ORDER if v in names]
    extras = sorted(v for v in names if v not in VAR_ORDER)
    return ordered + extras

# ---------------------------------------------------------------------------
# Publication-ready global style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.85',
    'legend.fancybox': True,
    'legend.shadow': False,
    'lines.linewidth': 2.0,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.which': 'major',
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'grid.color': '#cccccc',
    'axes.axisbelow': True,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'savefig.facecolor': 'white',
    'figure.dpi': 100,
    # Embed fonts as TrueType (Type 42) — required by most journals
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

from stats_utils import (
    bootstrap_auc, bootstrap_weighted_auc,
    bootstrap_roc_curves, bootstrap_pr_curves,
    bootstrap_gemma_operating_point,
    weighted_auc_roc, weighted_auc_pr,
    weighted_roc_curve, weighted_pr_curve,
    gemma_operating_point,
)


# ============================================================
# FIGURE A: Distribution plots (4, 7, 8)
# ============================================================

def plot_figure_a(df, prevalence_by_var, PREVALENCE_AVAILABLE):
    """Plot 4 (agreement per variable), 7 (prevalence), 8 (overall distribution)."""
    print("  Generating Figure A (plots 4, 7, 8)...")
    fig_a, axes_a = plt.subplots(1, 3, figsize=(18, 6.5))

    # --- Plot 4: Value distribution per annotation category ---
    ax = axes_a[0]
    var_names_plot7 = _order_vars(df["variable"].dropna().unique())
    n_vars7 = len(var_names_plot7)
    val_range7 = sorted(df["value"].unique())
    bar_width7 = 0.15
    x7 = np.arange(len(val_range7))

    for i, var_name in enumerate(var_names_plot7):
        sub_v = df[df["variable"] == var_name]
        val_counts = sub_v["value"].value_counts().sort_index()
        heights = [val_counts.get(v, 0) for v in val_range7]
        offset = (i - n_vars7 / 2 + 0.5) * bar_width7
        ax.bar(x7 + offset, heights, bar_width7,
               label=VAR_LABEL_MAP.get(var_name, var_name),
               color=VAR_COLORS[i % len(VAR_COLORS)], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x7)
    ax.set_xticklabels([str(v) for v in val_range7])
    ax.set_xlabel("LLM Agreement Score (No. of LLMs Matching Human)")
    ax.set_ylabel("Count")
    ax.set_title("LLM Agreement Distribution per Annotation\nCategory (Nominal Counts)")
    ax.legend(fontsize=11, loc="upper right")

    # --- Plot 7: Prevalence distribution grouped by annotation category ---
    ax = axes_a[1]
    if PREVALENCE_AVAILABLE:
        var_names = _order_vars(prevalence_by_var.keys())
        n_vars = len(var_names)
        val_range = list(range(5))
        n_vals = len(val_range)
        bar_width = 0.15
        x6 = np.arange(n_vars)

        val_colors = plt.cm.YlGnBu(np.linspace(0.2, 0.85, n_vals))
        for j, v in enumerate(val_range):
            heights = [prevalence_by_var[var]["prevalence"].get(v, 0) for var in var_names]
            offset = (j - n_vals / 2 + 0.5) * bar_width
            ax.bar(x6 + offset, heights, bar_width, label=f"# LLM match = {v}",
                   color=val_colors[j], alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x6)
        ax.set_xticklabels([VAR_LABEL_MAP.get(v, v) for v in var_names],
                           fontsize=11, rotation=30, ha="right")
        ax.set_ylabel("Proportion (Population Prevalence)")
        ax.set_title("Population Prevalence Grouped\nby Annotation Category")
        ax.legend(fontsize=11, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No prevalence file\navailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("Prevalence by Subcategory (Unavailable)")

    # --- Plot 8: Overall value distribution ---
    ax = axes_a[2]
    val_range_all = sorted(df["value"].unique())
    val_counts_all = df["value"].value_counts().sort_index()
    bar_heights = [val_counts_all.get(v, 0) for v in val_range_all]
    bar_colors = plt.cm.YlGnBu(np.linspace(0.2, 0.85, len(val_range_all)))
    ax.bar(val_range_all, bar_heights, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.5)
    for v, h in zip(val_range_all, bar_heights):
        ax.text(v, h + max(bar_heights) * 0.02, str(h), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_xticks(val_range_all)
    ax.set_xlabel("LLM Agreement Score (No. of LLMs Matching Human)")
    ax.set_ylabel("Number of Entries")
    ax.set_title("Overall LLM Agreement Distribution\n(All Categories Combined)")

    # --- Save Figure A-left/right: Plot 4 (left) and Plot 8 (right) ---
    fig_lr, (ax_lr0, ax_lr1) = plt.subplots(1, 2, figsize=(13, 6.5))

    # Copy plot 4 to ax_lr0
    var_names_plot7 = _order_vars(df["variable"].dropna().unique())
    n_vars7 = len(var_names_plot7)
    val_range7 = sorted(df["value"].unique())
    bar_width7 = 0.15
    x7 = np.arange(len(val_range7))
    for i, var_name in enumerate(var_names_plot7):
        sub_v = df[df["variable"] == var_name]
        val_counts = sub_v["value"].value_counts().sort_index()
        heights = [val_counts.get(v, 0) for v in val_range7]
        offset = (i - n_vars7 / 2 + 0.5) * bar_width7
        ax_lr0.bar(x7 + offset, heights, bar_width7,
                   label=VAR_LABEL_MAP.get(var_name, var_name),
                   color=VAR_COLORS[i % len(VAR_COLORS)], alpha=0.85, edgecolor="white", linewidth=0.5)
    _cap_labels7 = [
        f"{v}\n(cap: {'100' if v <= 2 else '50'})" for v in val_range7
    ]
    ax_lr0.set_xticks(x7)
    ax_lr0.set_xticklabels(_cap_labels7, fontsize=11)
    ax_lr0.set_xlabel("# LLMs Matching Human (Stratified Sample)")
    ax_lr0.set_ylabel("Number of Cases in Sample")
    ax_lr0.set_title("Stratified Sample: Cases per Agreement Score\nby Annotation Category")
    ax_lr0.legend(fontsize=11, loc="upper right")

    # Copy plot 8 to ax_lr1
    val_range_all = sorted(df["value"].unique())
    val_counts_all = df["value"].value_counts().sort_index()
    bar_heights = [val_counts_all.get(v, 0) for v in val_range_all]
    bar_colors = plt.cm.YlGnBu(np.linspace(0.2, 0.85, len(val_range_all)))
    ax_lr1.bar(val_range_all, bar_heights, color=bar_colors, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    for v, h in zip(val_range_all, bar_heights):
        ax_lr1.text(v, h + max(bar_heights) * 0.02, str(h), ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
    _cap_labels_all = [
        f"{v}\n(cap: {'100' if v <= 2 else '50'})" for v in val_range_all
    ]
    ax_lr1.set_xticks(val_range_all)
    ax_lr1.set_xticklabels(_cap_labels_all, fontsize=11)
    ax_lr1.set_xlabel("# LLMs Matching Human (Stratified Sample)")
    ax_lr1.set_ylabel("Number of Cases in Sample")
    ax_lr1.set_title("Stratified Sample: Cases per Agreement Score\n(All Categories Combined)")

    for ax, lbl in zip([ax_lr0, ax_lr1], ["a)", "b)"]):
        ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="bottom", ha="left")

    fig_lr.tight_layout()
    fig_lr_path = f"{OUTPUT_DIR}/figure_agreement_distribution.png"
    fig_lr.savefig(fig_lr_path, dpi=300, bbox_inches="tight")
    fig_lr.savefig(f"{OUTPUT_DIR}/figure_agreement_distribution.svg", bbox_inches="tight")
    plt.close(fig_lr)
    print(f"  Saved: {fig_lr_path} + .svg")

    # --- Save Figure A-middle: Plot 7 (prevalence) ---
    fig_mid, ax_mid = plt.subplots(1, 1, figsize=(8.5, 6.5))
    if PREVALENCE_AVAILABLE:
        var_names = _order_vars(prevalence_by_var.keys())
        n_vars = len(var_names)
        val_range = list(range(5))
        n_vals = len(val_range)
        bar_width = 0.15
        x6 = np.arange(n_vars)
        val_colors = plt.cm.YlGnBu(np.linspace(0.2, 0.85, n_vals))
        for j, v in enumerate(val_range):
            heights = [prevalence_by_var[var]["prevalence"].get(v, 0) for var in var_names]
            offset = (j - n_vals / 2 + 0.5) * bar_width
            ax_mid.bar(x6 + offset, heights, bar_width, label=f"# LLM matching ({v})",
                       color=val_colors[j], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_mid.set_xticks(x6)
        ax_mid.set_xticklabels([VAR_LABEL_MAP.get(v, v) for v in var_names],
                               fontsize=11, rotation=30, ha="right")
        ax_mid.set_ylabel("Proportion (Population Prevalence)")
        ax_mid.set_title("Population Prevalence Grouped\nby Annotation Category")
        ax_mid.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1),
                      borderaxespad=0)
    else:
        ax_mid.text(0.5, 0.5, "No prevalence file\navailable", ha="center", va="center",
                    transform=ax_mid.transAxes, fontsize=12, color="gray")
        ax_mid.set_title("Prevalence by Subcategory (Unavailable)")

    fig_mid.tight_layout()
    fig_mid_path = f"{OUTPUT_DIR}/figure_prevalence_distribution.png"
    fig_mid.savefig(fig_mid_path, dpi=300, bbox_inches="tight")
    fig_mid.savefig(f"{OUTPUT_DIR}/figure_prevalence_distribution.svg", bbox_inches="tight")
    plt.close(fig_mid)
    print(f"  Saved: {fig_mid_path} + .svg")

    plt.close(fig_a)


# ============================================================
# FIGURE B: Error rates & adjudication vote (1, 2, 3)
# ============================================================

def plot_figure_b(df, df_binary):
    """Plot 1 (error rates), 2 (adjudication vote absolute), 3 (adjudication vote normalized)."""
    print("  Generating Figure B (plots 1, 2, 3)...")
    fig_b, axes_b = plt.subplots(1, 3, figsize=(19, 7))

    values_sorted = sorted(df_binary["value"].unique())
    x = np.arange(len(values_sorted))
    categories = CATEGORIES
    colors_cat = COLORS_CAT
    cat_display = {"unclear": "Indeterminate", "Human": "Human correct", "Gemma": "Gemma correct"}

    # Softer, more elegant color palette for error rate lines
    _col_human = "#C62828"   # deep red
    _col_llm   = "#1565C0"   # deep blue
    _col_unclear = "#78909C"  # blue-grey

    # --- Plot 1: Human vs LLM error by value with error bars ---
    ax = axes_b[0]
    rates_human_err, rates_llm_err, rates_unclear = [], [], []
    ci_human, ci_llm, ci_unclear = [], [], []
    ns_plot1 = []

    for v in values_sorted:
        sub = df[df["value"] == v]
        n_all = len(sub)
        sub_res = df_binary[df_binary["value"] == v]
        n_res = len(sub_res)
        ns_plot1.append(n_all)
        r_h = sub_res["human_incorrect"].sum() / n_all
        r_l = sub_res["llm_incorrect"].sum() / n_all
        n_unclear = len(sub[sub["voter_correct"].isin(["unclear", "not decidable"])])
        r_u = n_unclear / n_all
        rates_human_err.append(r_h)
        rates_llm_err.append(r_l)
        rates_unclear.append(r_u)
        z = 1.96
        z2 = z * z
        for r, n, ci_list in [(r_h, n_res, ci_human), (r_l, n_res, ci_llm),
                               (r_u, n_all, ci_unclear)]:
            if n > 0:
                denom = 1 + z2 / n
                center = (r + z2 / (2 * n)) / denom
                half = z * np.sqrt(r * (1 - r) / n + z2 / (4 * n * n)) / denom
                ci_list.append([center - max(center - half, 0),
                                 min(center + half, 1) - center])
            else:
                ci_list.append([0, 0])

    # Shaded CI bands instead of bare error bars
    vs = np.array(values_sorted)
    for rates, ci, col, label, marker in [
        (rates_human_err, ci_human, _col_human, "Human error", "o"),
        (rates_llm_err, ci_llm, _col_llm, "LLM error", "s"),
    ]:
        rates_arr = np.array(rates)
        ci_arr = np.array(ci)
        ax.fill_between(vs, rates_arr - ci_arr[:, 0], rates_arr + ci_arr[:, 1],
                        color=col, alpha=0.10)
        ax.plot(vs, rates_arr, color=col, lw=2.2, zorder=4)
        ax.scatter(vs, rates_arr, color=col, marker=marker, s=55, zorder=5,
                   edgecolors="white", linewidths=0.8, label=label)

    # Unclear as dashed with lighter treatment
    rates_unc_arr = np.array(rates_unclear)
    ci_unc_arr = np.array(ci_unclear)
    ax.fill_between(vs, rates_unc_arr - ci_unc_arr[:, 0], rates_unc_arr + ci_unc_arr[:, 1],
                    color=_col_unclear, alpha=0.08)
    ax.plot(vs, rates_unc_arr, color=_col_unclear, lw=1.5, ls="--", alpha=0.7, zorder=4)
    ax.scatter(vs, rates_unc_arr, color=_col_unclear, marker="D", s=35, zorder=5,
               edgecolors="white", linewidths=0.6, label="Indeterminate", alpha=0.8)

    ax.set_xticks(values_sorted)
    ax.set_xlabel("LLM Agreement Score (No. of LLMs Matching Human)")
    ax.set_ylabel("Rate")
    ax.set_title("Human vs LLM Error Rate by LLM Agreement\n(shaded 95% CI)")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    for i, v in enumerate(values_sorted):
        ymax = max(rates_human_err[i] + ci_human[i][1], rates_llm_err[i] + ci_llm[i][1],
                   rates_unclear[i] + ci_unclear[i][1])
        ax.text(v, ymax + 0.015, f"n={ns_plot1[i]}", ha="center", va="bottom",
                fontsize=11, color="0.45", fontstyle="italic")

    # --- Plot 2: Expert vote distribution by value (absolute) ---
    ax = axes_b[1]
    bar_w = 0.62
    bottoms = np.zeros(len(values_sorted))
    for cat in categories:
        counts = np.array([len(df[(df["value"] == v) & (df["voter_correct"] == cat)]) for v in values_sorted])
        ax.bar(x, counts, width=bar_w, bottom=bottoms, label=cat_display.get(cat, cat),
               color=colors_cat[cat], alpha=0.88, edgecolor="white", linewidth=0.6)
        # Add count labels inside segments > 8
        for i, (c, bot) in enumerate(zip(counts, bottoms)):
            if c > 8:
                ax.text(i, bot + c / 2, str(c), ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        bottoms += counts
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values_sorted])
    ax.set_xlabel("LLM Agreement Score (No. of LLMs Matching Human)")
    ax.set_ylabel("Count")
    ax.set_title("Adjudication Vote Distribution by LLM Agreement\n(Absolute)")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

    # --- Plot 3: Expert vote distribution by value (normalized to 100%) ---
    ax = axes_b[2]
    bottoms_pct = np.zeros(len(values_sorted))
    totals_by_value = [len(df[df["value"] == v]) for v in values_sorted]
    for cat in categories:
        counts = [len(df[(df["value"] == v) & (df["voter_correct"] == cat)]) for v in values_sorted]
        pcts = np.array([c / t * 100 if t > 0 else 0 for c, t in zip(counts, totals_by_value)])
        ax.bar(x, pcts, width=bar_w, bottom=bottoms_pct, label=cat_display.get(cat, cat),
               color=colors_cat[cat], alpha=0.88, edgecolor="white", linewidth=0.6)
        for i, (pct, bot) in enumerate(zip(pcts, bottoms_pct)):
            if pct > 8:
                ax.text(i, bot + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
        bottoms_pct += pcts
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values_sorted])
    ax.set_xlabel("LLM Agreement Score (No. of LLMs Matching Human)")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_title("Adjudication Vote Distribution by LLM Agreement\n(Normalized to 100%)")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

    for ax, lbl in zip(axes_b, ["a)", "b)", "c)"]):
        ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="bottom", ha="left")

    fig_b.tight_layout(w_pad=3.5)
    fig_b_path = f"{OUTPUT_DIR}/figure_error_rates.png"
    fig_b.savefig(fig_b_path, dpi=300, bbox_inches="tight")
    fig_b.savefig(f"{OUTPUT_DIR}/figure_error_rates.svg", bbox_inches="tight")
    plt.close(fig_b)
    print(f"  Saved: {fig_b_path} + .svg")


# ============================================================
# FIGURE C: Gemma analysis (plot 5, + commented-out 6 & 9)
# ============================================================

def plot_figure_c(df, df_binary, PREVALENCE_AVAILABLE, g_boot_ci, cell_weights):
    """Plot 5a (gemma!=human) and 5b (gemma==human): who was right in each case."""
    print("  Generating Figure C (plots 5a, 5b)...")
    nrows = 2 if PREVALENCE_AVAILABLE else 1
    fig_c, axes_c = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows))
    if nrows == 1:
        axes_c = axes_c[np.newaxis, :]
    colors_cat = COLORS_CAT

    vote_cats = ["Human", "Gemma", "both correct", "both wrong", "unclear"]
    vote_labels = [
        "Human correct",
        "Gemma correct",
        "Both correct",
        "Both wrong",
        "Indeterminate"
    ]
    z_ci = 1.96

    def _compute_vote_stats(sub, force_raw=False):
        if PREVALENCE_AVAILABLE and not force_raw:
            vote_weighted = []
            for c in vote_cats:
                mask = sub["voter_correct"] == c
                vote_weighted.append(sub.loc[mask, "weight"].sum())
            total_w = sum(vote_weighted)
            pcts = [w / total_w * 100 if total_w > 0 else 0 for w in vote_weighted]
            adj = " (prevalence-adjusted)"
            w = sub["weight"].values
            n_eff = (w.sum() ** 2) / (w ** 2).sum() if (w ** 2).sum() > 0 else len(sub)
        else:
            counts = [len(sub[sub["voter_correct"] == c]) for c in vote_cats]
            total_r = sum(counts)
            pcts = [c / total_r * 100 if total_r > 0 else 0 for c in counts]
            adj = " (raw)" if PREVALENCE_AVAILABLE else " (raw, no prevalence file)"
            n_eff = len(sub)
        ci_lo, ci_hi = [], []
        for pct in pcts:
            p_hat = pct / 100.0
            if n_eff > 0 and 0 < p_hat < 1:
                denom = 1 + z_ci ** 2 / n_eff
                center = (p_hat + z_ci ** 2 / (2 * n_eff)) / denom
                margin = z_ci * np.sqrt((p_hat * (1 - p_hat) + z_ci ** 2 / (4 * n_eff)) / n_eff) / denom
                lo = max(0, center - margin) * 100
                hi = min(1, center + margin) * 100
            else:
                lo = pct
                hi = pct
            ci_lo.append(pct - lo)
            ci_hi.append(hi - pct)
        return pcts, ci_lo, ci_hi, adj, n_eff

    def _draw_vote_panel(ax, sub, title, force_raw=False):
        pcts, ci_lo, ci_hi, adj_label, n_eff = _compute_vote_stats(sub, force_raw=force_raw)
        vote_colors = [colors_cat[c] for c in vote_cats]
        y_pos = np.arange(len(vote_cats))

        # Lollipop / dot-and-whisker style
        for i in range(len(vote_cats)):
            # Stem line from 0 to dot
            ax.plot([0, pcts[i]], [y_pos[i], y_pos[i]],
                    color=vote_colors[i], lw=2.5, solid_capstyle="round")
            # CI whisker
            ax.errorbar(pcts[i], y_pos[i],
                        xerr=[[ci_lo[i]], [ci_hi[i]]],
                        fmt="none", ecolor="0.3", lw=1.2, capsize=4, capthick=1.2)
            # Dot
            ax.plot(pcts[i], y_pos[i], "o",
                    color=vote_colors[i], markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(vote_labels, fontsize=12)
        ax.set_xlabel("Percent (%)")
        adj_suffix = "Prevalence-Adjusted Proportions" if "adjusted" in adj_label else "Raw Proportions"
        if "adjusted" in adj_label:
            ax.set_title(f"{title}\n{adj_suffix}")
        else:
            ax.set_title(f"{title}\nn={len(sub)}, {adj_suffix}")
        for i, pct in enumerate(pcts):
            # Ensure label never overlaps error bar cap — use at least 2.5 offset
            x_offset = max(ci_hi[i] + 1.5, 2.5)
            ax.text(pct + x_offset, i, f"{pct:.1f}%", va="center", fontsize=12,
                    color="0.25")
        max_x = max(p + max(e, 2.5) for p, e in zip(pcts, ci_hi))
        ax.set_xlim(0, max_x + 12)
        ax.set_ylim(-0.5, len(vote_cats) - 0.5)
        ax.invert_yaxis()

    sub_neq = df[df["gemma_out"] == 0].copy()
    sub_eq  = df[df["gemma_out"] == 1].copy()

    if PREVALENCE_AVAILABLE:
        # --- Row 0: raw unadjusted ---
        _draw_vote_panel(axes_c[0, 0], sub_neq, "Gemma\u2013Human Non-Matching Cases", force_raw=True)
        _draw_vote_panel(axes_c[0, 1], sub_eq,  "Gemma\u2013Human Matching Cases", force_raw=True)
        # --- Row 1: prevalence-adjusted ---
        _draw_vote_panel(axes_c[1, 0], sub_neq, "Gemma\u2013Human Non-Matching Cases")
        _draw_vote_panel(axes_c[1, 1], sub_eq,  "Gemma\u2013Human Matching Cases")
        labels = ["a)", "b)", "c)", "d)"]
        for ax, lbl in zip(axes_c.flat, labels):
            ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="bottom", ha="left")
    else:
        _draw_vote_panel(axes_c[0, 0], sub_neq, "Gemma\u2013Human Non-Matching Cases")
        _draw_vote_panel(axes_c[0, 1], sub_eq,  "Gemma\u2013Human Matching Cases")
        for ax, lbl in zip(axes_c.flat, ["a)", "b)"]):
            ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="bottom", ha="left")

    fig_c.tight_layout()
    fig_c_path = f"{OUTPUT_DIR}/figure_gemma.png"
    fig_c.savefig(fig_c_path, dpi=300, bbox_inches="tight")
    fig_c.savefig(f"{OUTPUT_DIR}/figure_gemma.svg", bbox_inches="tight")
    plt.close(fig_c)
    print(f"  Saved: {fig_c_path} + .svg")


# ============================================================
# FIGURE D: Sensitivity analysis — ROC & PR, strict vs broad
# ============================================================

def plot_figure_d(df_binary, df_binary_broad, PREVALENCE_AVAILABLE):
    """4-panel figure: strict ROC, strict PR, broad ROC, broad PR."""
    print()
    print("=" * 80)
    print("9b. FIGURE D: SENSITIVITY ANALYSIS (broad = strict + unclear)")
    print("=" * 80)

    y_strict = df_binary["human_incorrect"].values.astype(int)
    val_strict = df_binary["value"].values.astype(float)
    gemma_strict = df_binary["gemma_out"].values.astype(int)

    y_broad = df_binary_broad["human_incorrect_broad"].values.astype(int)
    val_broad = df_binary_broad["value"].values.astype(float)
    gemma_broad = df_binary_broad["gemma_out"].values.astype(int)

    print(f"  Strict:  n={len(y_strict)},  prev={y_strict.mean():.4f}")
    print(f"  Broad: n={len(y_broad)}, prev={y_broad.mean():.4f}")

    # Bootstrap CI for gemma operating point
    print("  Computing bootstrap CI for gemma (strict definition, for Figure D)...")
    if PREVALENCE_AVAILABLE:
        w_strict_d = df_binary["weight"].values
        g_boot_ci_strict = bootstrap_gemma_operating_point(
            y_strict, gemma_strict, weights=w_strict_d,
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
    else:
        w_strict_d = None
        g_boot_ci_strict = bootstrap_gemma_operating_point(
            y_strict, gemma_strict, weights=None,
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

    print("  Computing bootstrap CI for gemma (broad definition)...")
    if PREVALENCE_AVAILABLE:
        w_broad = df_binary_broad["weight"].values
        g_boot_ci_lib = bootstrap_gemma_operating_point(
            y_broad, gemma_broad, weights=w_broad,
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
    else:
        w_broad = None
        g_boot_ci_lib = bootstrap_gemma_operating_point(
            y_broad, gemma_broad, weights=None,
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

    fig_d, axes_d = plt.subplots(2, 2, figsize=(14, 12))

    # --- Helper for a single ROC panel ---
    def _plot_roc_panel(ax, y, val_arr, gemma_arr, w, g_boot_ci, color, definition_label):
        from sklearn.metrics import roc_curve as sk_roc_curve, roc_auc_score as sk_roc_auc
        s_val = -val_arr.astype(float)

        if w is not None:
            fpr_arr, tpr_arr = weighted_roc_curve(y, s_val, w)
            auc_val = weighted_auc_roc(y, s_val, w)
            boot_roc = bootstrap_weighted_auc(y, s_val, w)
            fpr_grid, _, tpr_lo, tpr_hi = bootstrap_roc_curves(
                y, s_val, weights=w, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
            g_op = gemma_operating_point(y, gemma_arr, weights=w)
            prefix = "Prevalence-adjusted"
            auc_label = "AUC-ROC"
        else:
            fpr_arr, tpr_arr, _ = sk_roc_curve(y, s_val)
            auc_val = sk_roc_auc(y, s_val)
            boot_roc = bootstrap_auc(y, val_arr, higher_means_positive=False)
            fpr_grid, _, tpr_lo, tpr_hi = bootstrap_roc_curves(
                y, s_val, weights=None, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
            g_op = gemma_operating_point(y, gemma_arr)
            prefix = "Raw"
            auc_label = "AUC"

        ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color=color, alpha=0.15, label="_nolegend_")
        ax.plot(fpr_arr, tpr_arr, color=color, lw=2,
                label=(f"LLM agreement (4 LLMs) {auc_label}={auc_val:.3f}\n"
                       f"  95% CI [{boot_roc['ci_roc'][0]:.3f}, {boot_roc['ci_roc'][1]:.3f}]"))

        gemma_roc_auc = np.trapz([0, g_op["tpr"], 1], [0, g_op["fpr"], 1])
        ax.plot([0, g_op["fpr"], 1], [0, g_op["tpr"], 1],
                color="#8E24AA", lw=1.5, ls="--", alpha=0.6, label="_nolegend_")
        ax.fill_between([0, g_op["fpr"], 1], 0, [0, g_op["tpr"], 1],
                        color="#8E24AA", alpha=0.06, label="_nolegend_")

        tpr_err_lo = g_op["tpr"] - g_boot_ci["tpr_ci"][0]
        tpr_err_hi = g_boot_ci["tpr_ci"][1] - g_op["tpr"]
        ax.errorbar(g_op["fpr"], g_op["tpr"],
                    yerr=[[tpr_err_lo], [tpr_err_hi]],
                    fmt="D", color="#8E24AA", markersize=10,
                    markeredgecolor="white", markeredgewidth=1.0,
                    capsize=5, capthick=1.5, ecolor="#8E24AA", elinewidth=1.5,
                    zorder=5,
                    label=(f"Gemma only {auc_label}={gemma_roc_auc:.3f}\n"
                           f"  95% CI [{g_boot_ci['auc_roc_ci'][0]:.3f}, "
                           f"{g_boot_ci['auc_roc_ci'][1]:.3f}]"))

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        if w is not None:
            ax.set_title(f"{prefix} ROC — {definition_label}")
        else:
            ax.set_title(f"{prefix} ROC — {definition_label}\n(n={len(y)})")
        ax.legend(fontsize=11, loc="lower right")
        return auc_val

    # --- Helper for a single PR panel ---
    def _plot_pr_panel(ax, y, val_arr, gemma_arr, w, g_boot_ci, color, definition_label):
        from sklearn.metrics import precision_recall_curve as sk_prc, average_precision_score as sk_ap
        s_val = -val_arr.astype(float)

        if w is not None:
            rec_arr, prec_arr = weighted_pr_curve(y, s_val, w)
            ap_val = weighted_auc_pr(y, s_val, w)
            boot_pr = bootstrap_weighted_auc(y, s_val, w)
            prev = (y * w).sum() / w.sum()
            rec_grid, _, prec_lo, prec_hi = bootstrap_pr_curves(
                y, s_val, weights=w, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
            g_op = gemma_operating_point(y, gemma_arr, weights=w)
            prefix = "Prevalence-adjusted"
            ap_label = "AUC-PR"
        else:
            prec_arr, rec_arr, _ = sk_prc(y, s_val)
            ap_val = sk_ap(y, s_val)
            boot_pr = bootstrap_auc(y, val_arr, higher_means_positive=False)
            prev = y.mean()
            rec_grid, _, prec_lo, prec_hi = bootstrap_pr_curves(
                y, s_val, weights=None, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
            g_op = gemma_operating_point(y, gemma_arr)
            prefix = "Raw"
            ap_label = "AP"

        # Extend shaded CI region to recall=0
        if rec_grid[0] > 0:
            rec_grid = np.concatenate([[0], rec_grid])
            prec_lo = np.concatenate([[prec_lo[0]], prec_lo])
            prec_hi = np.concatenate([[prec_hi[0]], prec_hi])
        ax.fill_between(rec_grid, prec_lo, prec_hi, color=color, alpha=0.15, label="_nolegend_")
        ax.plot(rec_arr, prec_arr, color=color, lw=2,
                label=(f"LLM agreement (4 LLMs) {ap_label}={ap_val:.3f}\n"
                       f"  95% CI [{boot_pr['ci_pr'][0]:.3f}, {boot_pr['ci_pr'][1]:.3f}]"))

        gemma_pr_rec = [0, g_op["recall"], 1]
        gemma_pr_prec = [1.0, g_op["precision"], prev]
        gemma_auc_pr = np.trapz(gemma_pr_prec, gemma_pr_rec)

        ax.plot(gemma_pr_rec, gemma_pr_prec,
                color="#8E24AA", lw=1.5, ls="--", alpha=0.6, label="_nolegend_")
        ax.fill_between(gemma_pr_rec, prev, gemma_pr_prec,
                        where=[p >= prev for p in gemma_pr_prec],
                        color="#8E24AA", alpha=0.06, label="_nolegend_",
                        interpolate=True)

        prec_err_lo = g_op["precision"] - g_boot_ci["prec_ci"][0]
        prec_err_hi = g_boot_ci["prec_ci"][1] - g_op["precision"]
        ax.errorbar(g_op["recall"], g_op["precision"],
                    yerr=[[prec_err_lo], [prec_err_hi]],
                    fmt="D", color="#8E24AA", markersize=10,
                    markeredgecolor="white", markeredgewidth=1.0,
                    capsize=5, capthick=1.5, ecolor="#8E24AA", elinewidth=1.5,
                    zorder=5,
                    label=(f"Gemma only {ap_label}={gemma_auc_pr:.3f}\n"
                           f"  95% CI [{g_boot_ci['auc_pr_ci'][0]:.3f}, "
                           f"{g_boot_ci['auc_pr_ci'][1]:.3f}]"))

        ax.axhline(prev, color="k", ls="--", alpha=0.3,
                   label=f"baseline (prev={prev:.3f})")
        ax.set_xlim(left=0)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        if w is not None:
            ax.set_title(f"{prefix} PR — {definition_label}")
        else:
            ax.set_title(f"{prefix} PR — {definition_label}\n(n={len(y)})")
        ax.legend(fontsize=11, loc="lower left")
        return ap_val

    # Top-left: Strict ROC
    auc_strict_roc = _plot_roc_panel(
        axes_d[0, 0], y_strict, val_strict, gemma_strict,
        w_strict_d if PREVALENCE_AVAILABLE else None,
        g_boot_ci_strict, "#1E88E5",
        "Strict (human error)")

    # Top-right: Strict PR
    ap_strict = _plot_pr_panel(
        axes_d[0, 1], y_strict, val_strict, gemma_strict,
        w_strict_d if PREVALENCE_AVAILABLE else None,
        g_boot_ci_strict, "#1E88E5",
        "Strict (human error)")

    # Bottom-left: Broad ROC
    auc_lib_roc = _plot_roc_panel(
        axes_d[1, 0], y_broad, val_broad, gemma_broad,
        w_broad if PREVALENCE_AVAILABLE else None,
        g_boot_ci_lib, "#FB8C00",
        "Broad (human error + indeterminate)")

    # Bottom-right: Broad PR
    ap_lib = _plot_pr_panel(
        axes_d[1, 1], y_broad, val_broad, gemma_broad,
        w_broad if PREVALENCE_AVAILABLE else None,
        g_boot_ci_lib, "#FB8C00",
        "Broad (human error + indeterminate)")

    for ax, lbl in zip(axes_d.flat, ["a)", "b)", "c)", "d)"]):
        ax.text(-0.08, 1.02, lbl, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="bottom", ha="left")

    fig_d.tight_layout(h_pad=3.0, w_pad=3.0)
    fig_d_path = f"{OUTPUT_DIR}/figure_sensitivity_roc_pr.png"
    fig_d.savefig(fig_d_path, dpi=300, bbox_inches="tight")
    fig_d.savefig(f"{OUTPUT_DIR}/figure_sensitivity_roc_pr.svg", bbox_inches="tight")
    plt.close(fig_d)
    print(f"  Saved: {fig_d_path} + .svg")

    # Console summary
    print(f"\n--- Sensitivity analysis summary ---")
    print(f"  Strict:  n={len(y_strict)}, prev={y_strict.mean():.4f}")
    print(f"  Broad: n={len(y_broad)}, prev={y_broad.mean():.4f}")
    if PREVALENCE_AVAILABLE:
        print(f"  Strict  AUC-ROC={auc_strict_roc:.4f}, AUC-PR={ap_strict:.4f}")
        print(f"  Broad AUC-ROC={auc_lib_roc:.4f}, AUC-PR={ap_lib:.4f}")
    else:
        print(f"  Strict  AUC-ROC={auc_strict_roc:.4f}, AUC-PR={ap_strict:.4f}")
        print(f"  Broad AUC-ROC={auc_lib_roc:.4f}, AUC-PR={ap_lib:.4f}")

    return auc_strict_roc, ap_strict, auc_lib_roc, ap_lib


# ============================================================
# FIGURES E & F: Extrapolated error estimation plots
# ============================================================

def _compute_extrap_data(df_ext, cell_boots, total_est, total_pop,
                         ci_lo_tot, ci_hi_tot, overall_rate,
                         val_range_extrap, all_vars_extrap):
    """Compute per-agreement-level arrays needed for extrapolated error panels."""
    p_est, p_ci_lo, p_ci_hi = [], [], []
    p_n_pop, p_rate, p_rci_lo, p_rci_hi = [], [], [], []

    for v in val_range_extrap:
        sv = df_ext[df_ext["value"] == v]
        npv = sv["n_pop"].sum()
        estv = sv["est_errors"].sum()

        vb = np.zeros(len(next(iter(cell_boots.values()))))
        for vn in all_vars_extrap:
            arr = cell_boots.get((v, vn))
            if arr is not None and not np.all(np.isnan(arr)):
                vb += np.nan_to_num(arr, nan=0.0)

        clo = np.percentile(vb, BOOTSTRAP_ALPHA / 2 * 100)
        chi = np.percentile(vb, (1 - BOOTSTRAP_ALPHA / 2) * 100)
        rv = estv / npv if npv > 0 else 0.0

        p_est.append(estv)
        p_ci_lo.append(estv - clo)
        p_ci_hi.append(chi - estv)
        p_n_pop.append(npv)
        p_rate.append(rv)
        p_rci_lo.append(rv - clo / npv if npv > 0 else 0.0)
        p_rci_hi.append(chi / npv - rv if npv > 0 else 0.0)

    p_labels = [str(v) for v in val_range_extrap] + ["Total"]
    p_est.append(total_est)
    p_ci_lo.append(total_est - ci_lo_tot)
    p_ci_hi.append(ci_hi_tot - total_est)
    p_n_pop.append(total_pop)
    p_rate.append(overall_rate)
    tot_rate_ci_lo = ci_lo_tot / total_pop if total_pop > 0 else 0.0
    tot_rate_ci_hi = ci_hi_tot / total_pop if total_pop > 0 else 0.0
    p_rci_lo.append(overall_rate - tot_rate_ci_lo)
    p_rci_hi.append(tot_rate_ci_hi - overall_rate)

    return p_labels, p_est, p_ci_lo, p_ci_hi, p_n_pop, p_rate, p_rci_lo, p_rci_hi


def _draw_extrap_row(axes, p_labels, p_est, p_ci_lo, p_ci_hi, p_n_pop,
                     p_rate, p_rci_lo, p_rci_hi,
                     var_tots, total_est, total_pop, definition_label, sublabel_offset):
    """Draw 3 extrapolated-error panels into axes[0..2], sublabels starting at sublabel_offset."""
    _LABELS = "abcdef"
    n_bars = len(p_labels)
    x_pos = np.arange(n_bars)
    bcols_vals = plt.cm.YlGnBu(np.linspace(0.2, 0.8, n_bars - 1))
    bcols = list(bcols_vals) + [(0.35, 0.35, 0.35, 0.9)]

    # --- Panel: Absolute errors ---
    ax = axes[0]
    ax.text(-0.08, 1.02, f"{_LABELS[sublabel_offset]})",
            transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
    ax.bar(x_pos, p_est, color=bcols, alpha=0.85, edgecolor="white", linewidth=0.5,
           yerr=[p_ci_lo, p_ci_hi], capsize=5,
           error_kw={"ecolor": "black", "lw": 1.2, "capthick": 1.2})
    max_est = max(p_est) if max(p_est) > 0 else 1
    for i in range(n_bars):
        ax.text(x_pos[i] - 0.25, p_est[i] + p_ci_hi[i] + max_est * 0.02,
                f"{p_est[i]:.0f}\n(N={p_n_pop[i]:.0f})", ha="left", va="bottom",
                fontsize=12, fontweight="bold")
    top_abs = max(p_est[i] + p_ci_hi[i] for i in range(n_bars))
    ax.set_ylim(0, top_abs * 1.55)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(p_labels)
    ax.set_xlabel("LLM Agreement Score")
    ax.set_ylabel("Estimated Number of Errors")
    ax.set_title(f"Extrapolated Annotation Errors\nper Agreement Level — {definition_label}")
    ax.axvline(x_pos[-1] - 0.5, color="gray", ls=":", lw=1, alpha=0.6)

    # --- Panel: Error rate ---
    ax = axes[1]
    ax.text(-0.08, 1.02, f"{_LABELS[sublabel_offset + 1]})",
            transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
    rate_pcts = [r * 100 for r in p_rate]
    rate_ci_lo_pcts = [r * 100 for r in p_rci_lo]
    rate_ci_hi_pcts = [r * 100 for r in p_rci_hi]
    ax.bar(x_pos, rate_pcts, color=bcols, alpha=0.85, edgecolor="white", linewidth=0.5,
           yerr=[rate_ci_lo_pcts, rate_ci_hi_pcts], capsize=5,
           error_kw={"ecolor": "black", "lw": 1.2, "capthick": 1.2})
    for i in range(n_bars):
        ypos = rate_pcts[i] + rate_ci_hi_pcts[i]
        ax.text(x_pos[i] - 0.25, ypos + 1.0, f"{rate_pcts[i]:.1f}%", ha="left", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(p_labels)
    ax.set_xlabel("LLM Agreement Score")
    ax.set_ylabel("Estimated Error Rate (%)")
    ax.set_title(f"Extrapolated Annotation Error Rate\nper Agreement Level — {definition_label}")
    max_rate_top = max(rp + rc for rp, rc in zip(rate_pcts, rate_ci_hi_pcts))
    ax.set_ylim(0, min(110, max_rate_top + 20))
    ax.axvline(x_pos[-1] - 0.5, color="gray", ls=":", lw=1, alpha=0.6)

    # --- Panel: Stacked bar per variable ---
    ax = axes[2]
    ax.text(-0.08, 1.02, f"{_LABELS[sublabel_offset + 2]})",
            transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
    df_vt = pd.DataFrame(var_tots)
    ordered_vars = _order_vars(df_vt["variable"].tolist())
    df_vt = df_vt.set_index("variable").reindex(ordered_vars).reset_index()
    vnames = df_vt["variable"].tolist() + ["Total"]
    est_err = np.append(df_vt["est_errors"].values, total_est)
    npop_arr = np.append(df_vt["n_pop"].values, total_pop)
    est_corr = npop_arr - est_err
    xe = np.arange(len(vnames))
    bw = 0.5
    corr_colors = ["#43A047"] * (len(vnames) - 1) + ["#2E7D32"]
    err_colors  = ["#E53935"] * (len(vnames) - 1) + ["#C62828"]
    ax.bar(xe, est_corr, bw, label="Estimated correct", color=corr_colors, alpha=0.8)
    ax.bar(xe, est_err,  bw, bottom=est_corr, label="Estimated errors", color=err_colors, alpha=0.8)
    max_npop = max(npop_arr) if len(npop_arr) > 0 else 1
    for i, (vn, errs, npop) in enumerate(zip(vnames, est_err, npop_arr)):
        rpct = errs / npop * 100 if npop > 0 else 0
        ax.text(i - 0.25, npop + max_npop * 0.01,
                f"{rpct:.1f}%\n({errs:.0f}/{npop:.0f})",
                ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_npop * 1.45)
    ax.set_xticks(xe)
    ax.set_xticklabels([VAR_LABEL_MAP.get(n, n) for n in vnames],
                       fontsize=11, rotation=30, ha="right")
    ax.set_ylabel("Number of Annotations")
    ax.set_title(f"Estimated Correct vs Erroneous Annotations\nper Variable — {definition_label}")
    ax.legend(fontsize=12, loc="upper right")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(xe[-1] - 0.5, color="gray", ls=":", lw=1, alpha=0.6)


def plot_extrapolated_errors_combined(
        df_ext_strict, cell_boots_strict, var_tots_strict,
        total_est_strict, total_pop_strict, ci_lo_strict, ci_hi_strict, overall_rate_strict,
        df_ext_lib, cell_boots_lib, var_tots_lib,
        total_est_lib, total_pop_lib, ci_lo_lib, ci_hi_lib, overall_rate_lib,
        val_range_extrap, all_vars_extrap):
    """
    Combined 3×2 figure: strict (left column, panels a–c) and broad (right column, panels d–f).
      Row 0: a) strict absolute errors      d) broad absolute errors
      Row 1: b) strict error rate           e) broad error rate
      Row 2: c) strict stacked bar          f) broad stacked bar
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    strict_data = _compute_extrap_data(
        df_ext_strict, cell_boots_strict,
        total_est_strict, total_pop_strict, ci_lo_strict, ci_hi_strict, overall_rate_strict,
        val_range_extrap, all_vars_extrap)

    lib_data = _compute_extrap_data(
        df_ext_lib, cell_boots_lib,
        total_est_lib, total_pop_lib, ci_lo_lib, ci_hi_lib, overall_rate_lib,
        val_range_extrap, all_vars_extrap)

    # Left column = strict (a, b, c); right column = broad (d, e, f)
    _draw_extrap_row([axes[0, 0], axes[1, 0], axes[2, 0]], *strict_data,
                     var_tots_strict, total_est_strict, total_pop_strict,
                     "Strict", sublabel_offset=0)
    _draw_extrap_row([axes[0, 1], axes[1, 1], axes[2, 1]], *lib_data,
                     var_tots_lib, total_est_lib, total_pop_lib,
                     "Broad (= Strict + Indeterminate)", sublabel_offset=3)

    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    output_name = "figure_extrapolated_errors_combined"
    fig_path_png = f"{OUTPUT_DIR}/{output_name}.png"
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/{output_name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path_png} + .svg")


# ============================================================
# FIGURE G: Percentage Agreement – LLM vs Human per category
# ============================================================

def plot_figure_agreement_vs_human(res_df):
    """
    Bar chart of percentage agreement between each LLM and human annotators,
    per annotation category plus a macro-average Combined bar.

    Parameters
    ----------
    res_df : pd.DataFrame
        Must contain columns: llm, variable, pct_agree, pct_se, n
        One row per (llm, variable) pair.  A 'Combined' variable row
        (macro-average across categories) is added automatically.
    """
    print("  Generating Figure G (pct agreement LLM vs Human)...")

    llm_order = ['Gemma', 'Deepseek', 'GPT', 'Mistral']
    llm_colors = {
        'Deepseek': '#1ABC9C',
        'Gemma':    '#3498DB',
        'GPT':      '#9B59B6',
        'Mistral':  '#E67E22',
    }
    rater_labels = {
        'Deepseek': 'DeepSeek',
        'Gemma':    'Gemma',
        'GPT':      'GPT-4',
        'Mistral':  'Mistral',
    }

    var_order = [v for v in VAR_ORDER if v in res_df['variable'].unique()]

    # Build Combined (macro-average across variables) per LLM
    combined_rows = []
    for llm in llm_order:
        sub = res_df[res_df['llm'] == llm]
        if len(sub) == 0:
            continue
        avg_pct = sub['pct_agree'].mean()
        pooled_se = np.sqrt((sub['pct_se'] ** 2).mean())
        combined_rows.append({
            'llm': llm,
            'variable': 'Combined',
            'pct_agree': avg_pct,
            'pct_se': pooled_se,
            'pct_ci_low':  max(0.0,   avg_pct - 1.96 * pooled_se),
            'pct_ci_high': min(100.0, avg_pct + 1.96 * pooled_se),
            'n': np.nan,
        })

    plot_df = pd.concat(
        [res_df, pd.DataFrame(combined_rows)], ignore_index=True
    )

    var_list_ext = var_order + ['Combined']
    n_llms = len(llm_order)
    n_vars = len(var_list_ext)
    bar_width = 0.18
    x_base = np.arange(n_vars)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, llm in enumerate(llm_order):
        sub = (plot_df[plot_df['llm'] == llm]
               .set_index('variable')
               .reindex(var_list_ext))
        offset = (i - (n_llms - 1) / 2) * bar_width
        vals = sub['pct_agree'].values
        errs = sub['pct_se'].values * 1.96

        bars = ax.bar(
            x_base + offset, vals, bar_width,
            yerr=errs, capsize=3,
            label=rater_labels[llm],
            color=llm_colors[llm],
            edgecolor='white', linewidth=0.5, alpha=0.88,
        )
        for bar, v, e in zip(bars, vals, errs):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + e + 1.0,
                    f"{v:.1f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                )

    # Separator before Combined column
    ax.axvline(n_vars - 1.5, color='grey', ls='--', lw=0.8, alpha=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [VAR_LABEL_MAP.get(v, v) for v in var_list_ext], fontsize=12
    )
    ax.set_ylabel("Agreement (%)", fontsize=14)
    ax.set_ylim(0, 109)
    ax.set_title(
        "Percentage Agreement: LLM vs Human (with 95% CI)",
        fontsize=15, fontweight='bold',
    )
    ax.legend(fontsize=11, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    output_name = "figure_agreement_vs_human"
    fig_path_png = f"{OUTPUT_DIR}/{output_name}.png"
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/{output_name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path_png} + .svg")
