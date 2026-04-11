"""
Statistical utility functions for blinded annotation analysis.

Includes:
  - Bootstrap CI for AUC-ROC / AUC-PR (unweighted and weighted)
  - Bootstrap CI bands for ROC and PR curves
  - Bootstrap CI for gemma binary operating point
  - Weighted AUC-ROC, AUC-PR, and curve functions
  - Bootstrap AUC comparison test
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve,
)

from config import N_BOOTSTRAP, BOOTSTRAP_ALPHA, RANDOM_SEED


# ============================================================
# WEIGHTED AUC / CURVE FUNCTIONS
# ============================================================

def weighted_auc_roc(y_true, scores, weights):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    w_sorted = weights[order]
    s_sorted = scores[order]

    total_pos = (y_sorted * w_sorted).sum()
    total_neg = ((1 - y_sorted) * w_sorted).sum()
    if total_pos == 0 or total_neg == 0:
        return np.nan

    tpr_list, fpr_list = [0.0], [0.0]
    cum_tp, cum_fp = 0.0, 0.0
    prev_score = None

    for i in range(len(y_sorted)):
        if prev_score is not None and s_sorted[i] != prev_score:
            tpr_list.append(cum_tp / total_pos)
            fpr_list.append(cum_fp / total_neg)
        if y_sorted[i] == 1:
            cum_tp += w_sorted[i]
        else:
            cum_fp += w_sorted[i]
        prev_score = s_sorted[i]

    tpr_list.append(cum_tp / total_pos)
    fpr_list.append(cum_fp / total_neg)
    return np.trapz(tpr_list, fpr_list)


def weighted_auc_pr(y_true, scores, weights):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    w_sorted = weights[order]
    s_sorted = scores[order]

    total_pos = (y_sorted * w_sorted).sum()
    if total_pos == 0:
        return np.nan

    cum_tp, cum_total = 0.0, 0.0
    prec_list, rec_list = [1.0], [0.0]
    prev_score = None

    for i in range(len(y_sorted)):
        if prev_score is not None and s_sorted[i] != prev_score and cum_total > 0:
            prec_list.append(cum_tp / cum_total)
            rec_list.append(cum_tp / total_pos)
        if y_sorted[i] == 1:
            cum_tp += w_sorted[i]
        cum_total += w_sorted[i]
        prev_score = s_sorted[i]

    if cum_total > 0:
        prec_list.append(cum_tp / cum_total)
        rec_list.append(cum_tp / total_pos)
    return abs(np.trapz(prec_list, rec_list))


def weighted_roc_curve(y_true, scores, weights):
    order = np.argsort(-scores)
    y_s = y_true[order]
    w_s = weights[order]
    sc_s = scores[order]
    total_pos = (y_s * w_s).sum()
    total_neg = ((1 - y_s) * w_s).sum()
    if total_pos == 0 or total_neg == 0:
        return np.array([0, 1]), np.array([0, 1])
    tpr_l, fpr_l = [0.0], [0.0]
    cum_tp, cum_fp = 0.0, 0.0
    prev_sc = None
    for i in range(len(y_s)):
        if prev_sc is not None and sc_s[i] != prev_sc:
            tpr_l.append(cum_tp / total_pos)
            fpr_l.append(cum_fp / total_neg)
        if y_s[i] == 1:
            cum_tp += w_s[i]
        else:
            cum_fp += w_s[i]
        prev_sc = sc_s[i]
    tpr_l.append(cum_tp / total_pos)
    fpr_l.append(cum_fp / total_neg)
    return np.array(fpr_l), np.array(tpr_l)


def weighted_pr_curve(y_true, scores, weights):
    order = np.argsort(-scores)
    y_s = y_true[order]
    w_s = weights[order]
    sc_s = scores[order]
    total_pos = (y_s * w_s).sum()
    if total_pos == 0:
        return np.array([0, 1]), np.array([1, 0])
    cum_tp, cum_total = 0.0, 0.0
    prec_l, rec_l = [1.0], [0.0]
    prev_sc = None
    for i in range(len(y_s)):
        if prev_sc is not None and sc_s[i] != prev_sc and cum_total > 0:
            prec_l.append(cum_tp / cum_total)
            rec_l.append(cum_tp / total_pos)
        if y_s[i] == 1:
            cum_tp += w_s[i]
        cum_total += w_s[i]
        prev_sc = sc_s[i]
    if cum_total > 0:
        prec_l.append(cum_tp / cum_total)
        rec_l.append(cum_tp / total_pos)
    return np.array(rec_l), np.array(prec_l)


# ============================================================
# BOOTSTRAP: AUC (UNWEIGHTED)
# ============================================================

def bootstrap_auc(y_true, scores, n_boot=N_BOOTSTRAP, alpha=BOOTSTRAP_ALPHA,
                  higher_means_positive=True, seed=RANDOM_SEED):
    """
    Non-parametric bootstrap CI for AUC-ROC and AUC-PR (unweighted).
    Stratified by outcome to guarantee both classes in every replicate.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if not higher_means_positive:
        s = -s

    if len(np.unique(y)) < 2:
        return {"point_roc": np.nan, "ci_roc": (np.nan, np.nan),
                "point_pr": np.nan, "ci_pr": (np.nan, np.nan)}

    point_roc = roc_auc_score(y, s)
    point_pr = average_precision_score(y, s)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    roc_boots = np.empty(n_boot)
    pr_boots = np.empty(n_boot)

    for b in range(n_boot):
        sel_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sel_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        y_b = y[sel]
        s_b = s[sel]
        roc_boots[b] = roc_auc_score(y_b, s_b)
        pr_boots[b] = average_precision_score(y_b, s_b)

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_roc = (np.percentile(roc_boots, lo), np.percentile(roc_boots, hi))
    ci_pr = (np.percentile(pr_boots, lo), np.percentile(pr_boots, hi))

    return {"point_roc": point_roc, "ci_roc": ci_roc,
            "point_pr": point_pr, "ci_pr": ci_pr}


# ============================================================
# BOOTSTRAP: AUC (WEIGHTED)
# ============================================================

def bootstrap_weighted_auc(y_true, scores, weights,
                           n_boot=N_BOOTSTRAP, alpha=BOOTSTRAP_ALPHA,
                           seed=RANDOM_SEED):
    """
    Non-parametric bootstrap CI for weighted AUC-ROC and AUC-PR.
    Each replicate resamples rows (stratified by outcome) and carries
    their importance weights along.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)

    if len(np.unique(y)) < 2:
        return {"point_roc": np.nan, "ci_roc": (np.nan, np.nan),
                "point_pr": np.nan, "ci_pr": (np.nan, np.nan)}

    point_roc = weighted_auc_roc(y, s, w)
    point_pr = weighted_auc_pr(y, s, w)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    roc_boots = np.empty(n_boot)
    pr_boots = np.empty(n_boot)

    for b in range(n_boot):
        sel_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sel_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        roc_boots[b] = weighted_auc_roc(y[sel], s[sel], w[sel])
        pr_boots[b] = weighted_auc_pr(y[sel], s[sel], w[sel])

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_roc = (np.percentile(roc_boots, lo), np.percentile(roc_boots, hi))
    ci_pr = (np.percentile(pr_boots, lo), np.percentile(pr_boots, hi))

    return {"point_roc": point_roc, "ci_roc": ci_roc,
            "point_pr": point_pr, "ci_pr": ci_pr}


# ============================================================
# BOOTSTRAP: ROC / PR CURVE CI BANDS
# ============================================================

def bootstrap_roc_curves(y_true, scores, weights=None,
                         n_boot=N_BOOTSTRAP, alpha=BOOTSTRAP_ALPHA,
                         n_grid=200, seed=RANDOM_SEED):
    """Bootstrap pointwise CI bands for ROC curve (optionally weighted)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(y))

    fpr_grid = np.linspace(0, 1, n_grid)
    tpr_matrix = np.empty((n_boot, n_grid))

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    for b in range(n_boot):
        sel_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sel_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        if weights is not None:
            fpr_b, tpr_b = weighted_roc_curve(y[sel], s[sel], w[sel])
        else:
            fpr_b, tpr_b, _ = roc_curve(y[sel], s[sel])
        tpr_matrix[b, :] = np.interp(fpr_grid, fpr_b, tpr_b)

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    tpr_median = np.median(tpr_matrix, axis=0)
    tpr_lo = np.percentile(tpr_matrix, lo, axis=0)
    tpr_hi = np.percentile(tpr_matrix, hi, axis=0)

    return fpr_grid, tpr_median, tpr_lo, tpr_hi


def bootstrap_pr_curves(y_true, scores, weights=None,
                        n_boot=N_BOOTSTRAP, alpha=BOOTSTRAP_ALPHA,
                        n_grid=200, seed=RANDOM_SEED):
    """Bootstrap pointwise CI bands for PR curve (optionally weighted)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(y))

    rec_grid = np.linspace(0, 1, n_grid)
    prec_matrix = np.empty((n_boot, n_grid))

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    for b in range(n_boot):
        sel_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sel_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        if weights is not None:
            rec_b, prec_b = weighted_pr_curve(y[sel], s[sel], w[sel])
        else:
            prec_b, rec_b, _ = precision_recall_curve(y[sel], s[sel])
        sort_idx = np.argsort(rec_b)
        prec_matrix[b, :] = np.interp(rec_grid, rec_b[sort_idx], prec_b[sort_idx])

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    prec_median = np.median(prec_matrix, axis=0)
    prec_lo = np.percentile(prec_matrix, lo, axis=0)
    prec_hi = np.percentile(prec_matrix, hi, axis=0)

    return rec_grid, prec_median, prec_lo, prec_hi


# ============================================================
# BOOTSTRAP: GEMMA OPERATING POINT
# ============================================================

def bootstrap_gemma_operating_point(y_true, gemma_vals, weights=None,
                                    n_boot=N_BOOTSTRAP, alpha=BOOTSTRAP_ALPHA,
                                    seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    g = np.asarray(gemma_vals, dtype=int)
    w = np.asarray(weights, dtype=float) if weights is not None else None

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    fpr_boots = np.empty(n_boot)
    tpr_boots = np.empty(n_boot)
    prec_boots = np.empty(n_boot)
    rec_boots = np.empty(n_boot)
    auc_roc_boots = np.empty(n_boot)
    auc_pr_boots = np.empty(n_boot)

    for b in range(n_boot):
        sel_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sel_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        y_b = y[sel]
        g_b = g[sel]
        pred_pos = (g_b == 0).astype(int)

        if w is not None:
            w_b = w[sel]
            tp = ((pred_pos == 1) & (y_b == 1)).astype(float) @ w_b
            fp = ((pred_pos == 1) & (y_b == 0)).astype(float) @ w_b
            fn = ((pred_pos == 0) & (y_b == 1)).astype(float) @ w_b
            tn = ((pred_pos == 0) & (y_b == 0)).astype(float) @ w_b
        else:
            tp = ((pred_pos == 1) & (y_b == 1)).sum()
            fp = ((pred_pos == 1) & (y_b == 0)).sum()
            fn = ((pred_pos == 0) & (y_b == 1)).sum()
            tn = ((pred_pos == 0) & (y_b == 0)).sum()

        tpr_b = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_b = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec_b = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_b = tpr_b

        tpr_boots[b] = tpr_b
        fpr_boots[b] = fpr_b
        prec_boots[b] = prec_b
        rec_boots[b] = rec_b

        auc_roc_boots[b] = np.trapz([0, tpr_b, 1], [0, fpr_b, 1])

        if w is not None:
            prev_b = (y_b.astype(float) @ w_b) / w_b.sum()
        else:
            prev_b = y_b.mean()
        pr_rec = [0.0, rec_b, 1.0]
        pr_prec = [1.0, prec_b, prev_b]
        auc_pr_boots[b] = np.trapz(pr_prec, pr_rec)

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    return {
        "fpr_ci": (np.percentile(fpr_boots, lo), np.percentile(fpr_boots, hi)),
        "tpr_ci": (np.percentile(tpr_boots, lo), np.percentile(tpr_boots, hi)),
        "prec_ci": (np.percentile(prec_boots, lo), np.percentile(prec_boots, hi)),
        "rec_ci": (np.percentile(rec_boots, lo), np.percentile(rec_boots, hi)),
        "auc_roc_ci": (np.percentile(auc_roc_boots, lo), np.percentile(auc_roc_boots, hi)),
        "auc_roc_point": np.median(auc_roc_boots),
        "auc_pr_ci": (np.percentile(auc_pr_boots, lo), np.percentile(auc_pr_boots, hi)),
        "auc_pr_point": np.median(auc_pr_boots),
    }


# ============================================================
# BOOTSTRAP SIGNIFICANCE TESTS FOR AUC COMPARISON
# ============================================================


def bootstrap_compare_auc(y_true, scores_a, scores_b,
                          metric="roc", weights=None,
                          n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    w = np.asarray(weights, dtype=float) if weights is not None else None

    if len(np.unique(y)) < 2:
        return {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "se_diff": np.nan}

    if metric == "roc":
        if w is not None:
            fn = lambda y_, s_, w_: weighted_auc_roc(y_, s_, w_)
        else:
            fn = lambda y_, s_, w_: roc_auc_score(y_, s_)
    else:
        if w is not None:
            fn = lambda y_, s_, w_: weighted_auc_pr(y_, s_, w_)
        else:
            fn = lambda y_, s_, w_: average_precision_score(y_, s_)

    w_arr = w if w is not None else np.ones(len(y))

    point_a = fn(y, sa, w_arr)
    point_b = fn(y, sb, w_arr)
    point_diff = point_a - point_b

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sel_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sel_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        auc_a_b = fn(y[sel], sa[sel], w_arr[sel])
        auc_b_b = fn(y[sel], sb[sel], w_arr[sel])
        diffs[b] = auc_a_b - auc_b_b

    se_diff = np.std(diffs, ddof=1)
    ci_lo = np.percentile(diffs, BOOTSTRAP_ALPHA / 2 * 100)
    ci_hi = np.percentile(diffs, (1 - BOOTSTRAP_ALPHA / 2) * 100)

    if point_diff >= 0:
        p_value = 2 * np.mean(diffs <= 0)
    else:
        p_value = 2 * np.mean(diffs >= 0)
    p_value = min(p_value, 1.0)

    return {"diff": point_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_value": p_value, "se_diff": se_diff}


# ============================================================
# GEMMA OPERATING POINT (single point estimate)
# ============================================================

def gemma_operating_point(y_true, gemma_scores, weights=None):
    """Compute FPR, TPR, precision, recall for gemma binary predictor."""
    y = np.asarray(y_true, dtype=int)
    g = np.asarray(gemma_scores, dtype=int)
    pred_pos = (g == 0).astype(int)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        tp = ((pred_pos == 1) & (y == 1)).astype(float) @ w
        fp = ((pred_pos == 1) & (y == 0)).astype(float) @ w
        fn = ((pred_pos == 0) & (y == 1)).astype(float) @ w
        tn = ((pred_pos == 0) & (y == 0)).astype(float) @ w
    else:
        tp = ((pred_pos == 1) & (y == 1)).sum()
        fp = ((pred_pos == 1) & (y == 0)).sum()
        fn = ((pred_pos == 0) & (y == 1)).sum()
        tn = ((pred_pos == 0) & (y == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
