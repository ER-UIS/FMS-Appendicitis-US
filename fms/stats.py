"""Statistical procedures used in the paper (eMethods 2 in Supplement 1).

All functions are pure numpy/scipy and operate on boolean or numeric arrays.

* :func:`wilson` ............ Wilson score 95% CI for a proportion
* :func:`mcnemar` ........... patient-matched McNemar test (exact binomial when
  the discordant total is <= 1000, continuity-corrected chi-square otherwise)
* :func:`newcombe_paired` ... Newcombe (method 10) CI for a paired difference
  of proportions
* :func:`durkalski` ......... cluster-adjusted McNemar test (Durkalski 2003)
  for feature-level comparisons with several assessments per patient
* :func:`cohen_kappa`, :func:`fleiss_kappa` ... inter-reader agreement
* :func:`paired_metric_tests` ... paired t and Wilcoxon tests across features
* :func:`holm` .............. Holm-Bonferroni adjustment
* :func:`binary_metrics` .... accuracy, precision, recall, F1, AUC and the
  comprehensive index (0.3 accuracy + 0.4 F1 + 0.3 AUC)
* :func:`bootstrap_ci` ...... percentile bootstrap CI (patient-level resampling)
"""
from __future__ import annotations
from math import sqrt
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, chi2, friedmanchisquare, ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score

from .features import WEIGHTS

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for k successes in n trials (returns proportions)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar(a, b) -> Dict[str, float]:
    """Patient-matched McNemar test for two paired binary outcomes.

    `a` and `b` are boolean arrays of equal length (eg, "substandard report"
    under two strategies for the same patients).  Returns the discordant counts
    b (a & ~b) and c (~a & b), the continuity-corrected statistic and the
    P value: exact 2-sided binomial if b + c <= 1000, chi-square otherwise.
    """
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    nb = int((a & ~b).sum())
    nc = int((~a & b).sum())
    if nb + nc == 0:
        return {"b": nb, "c": nc, "statistic": 0.0, "p": float("nan"), "method": "none"}
    stat = (abs(nb - nc) - 1) ** 2 / (nb + nc)
    if nb + nc <= 1000:
        p = binomtest(min(nb, nc), nb + nc, 0.5).pvalue
        method = "exact binomial"
    else:
        p = chi2.sf(stat, 1)
        method = "continuity-corrected chi-square"
    return {"b": nb, "c": nc, "statistic": stat, "p": float(p), "method": method}


def newcombe_paired(x, y, z: float = Z95) -> Dict[str, float]:
    """Newcombe (1998, method 10) CI for p(x) - p(y) with paired binary data.

    Returns the difference and CI in percentage points.
    """
    x = np.asarray(x, bool)
    y = np.asarray(y, bool)
    n = len(x)
    a = int((x & y).sum()); b = int((x & ~y).sum()); c = int((~x & y).sum()); d = int((~x & ~y).sum())
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2, 0)
        phi = num / sqrt(den)
    dl = sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return {"diff": 100 * (p1 - p2), "lo": 100 * (p1 - p2 - dl), "hi": 100 * (p1 - p2 + du), "b": b, "c": c}


def durkalski(correct_a, correct_b) -> Dict[str, float]:
    """Cluster-adjusted McNemar test (Durkalski et al, Stat Med 2003).

    `correct_a`, `correct_b`: boolean matrices (patients x assessments), eg
    9 feature-level correctness indicators per patient under two methods.
    """
    ca = np.asarray(correct_a, bool)
    cb = np.asarray(correct_b, bool)
    b = (ca & ~cb).sum(1)
    c = (~ca & cb).sum(1)
    den = ((b - c) ** 2).sum()
    stat = (b.sum() - c.sum()) ** 2 / den if den > 0 else float("nan")
    return {"statistic": float(stat), "p": float(chi2.sf(stat, 1)) if den > 0 else float("nan"),
            "b": int(b.sum()), "c": int(c.sum())}


def cohen_kappa(a, b) -> Tuple[float, float]:
    """Cohen kappa and observed agreement for two raters (categorical labels)."""
    a = np.asarray(a); b = np.asarray(b)
    po = float((a == b).mean())
    cats = np.union1d(a, b)
    pe = sum(float((a == k).mean() * (b == k).mean()) for k in cats)
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po


def fleiss_kappa(mat) -> Tuple[float, float]:
    """Fleiss kappa for an n x k matrix of category labels from k raters."""
    mat = np.asarray(mat)
    cats = np.unique(mat)
    n, k = mat.shape
    cnt = np.stack([(mat == c).sum(1) for c in cats], 1)
    Pi = ((cnt ** 2).sum(1) - k) / (k * (k - 1))
    Pbar = Pi.mean()
    pj = cnt.sum(0) / (n * k)
    Pe = (pj ** 2).sum()
    return ((Pbar - Pe) / (1 - Pe) if Pe < 1 else float("nan")), float(Pbar)


def paired_metric_tests(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Paired t test and Wilcoxon signed-rank test across features (n = 9)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    t = ttest_rel(x, y)
    try:
        w = wilcoxon(x, y)
        pw = float(w.pvalue)
    except ValueError:  # all differences zero
        pw = float("nan")
    return {"mean_x": float(x.mean()), "mean_y": float(y.mean()), "t": float(t.statistic), "p_t": float(t.pvalue), "p_wilcoxon": pw}


def friedman(*groups) -> Dict[str, float]:
    """Friedman test across k related samples (eg, 3 early-stopping rules x 9 features)."""
    r = friedmanchisquare(*groups)
    return {"statistic": float(r.statistic), "p": float(r.pvalue)}


def holm(pvals: Sequence[float]) -> list:
    """Holm-Bonferroni adjusted P values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def binary_metrics(y, pred, p=None) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, AUC and comprehensive index.

    `y`, `pred`: boolean arrays; `p`: positive-class probabilities (for AUC;
    if None, `pred` is used and AUC degenerates to balanced accuracy).
    """
    y = np.asarray(y, bool); pred = np.asarray(pred, bool)
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())
    n = len(y)
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    score = np.asarray(p, float) if p is not None else pred.astype(float)
    auc = float(roc_auc_score(y, score)) if 0 < y.sum() < n else float("nan")
    comp = WEIGHTS["accuracy"] * acc + WEIGHTS["f1"] * f1 + WEIGHTS["auc"] * (auc if auc == auc else 0.0)
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "comprehensive_index": comp,
            "sensitivity": rec, "specificity": tn / (tn + fp) if tn + fp else float("nan")}


def comprehensive_index(acc: float, f1: float, auc: float) -> float:
    return WEIGHTS["accuracy"] * acc + WEIGHTS["f1"] * f1 + WEIGHTS["auc"] * auc


def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, B: int = 2000, seed: int = 2026,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of `stat(indices)` over patient-level resamples.

    `stat` receives an integer index array (with replacement) of length n.
    """
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(B)], float)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def format_p(p: float) -> str:
    """JAMA-style P value formatting."""
    if p != p:
        return "NA"
    if p < 0.001:
        return "<.001"
    if p > 0.99:
        return ">.99"
    s = f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
    return s[1:]
