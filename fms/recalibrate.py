"""Post hoc threshold recalibration (eMethods 4; eTable 20).

For a deployed checkpoint, the decision threshold is chosen on the
model-selection cohort as the value on a 0.01-0.99 grid maximizing the
comprehensive index; where several thresholds tie, the one closest to 0.5 is
taken.  The locked threshold is then applied to the test cohorts.
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .stats import binary_metrics, comprehensive_index

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def pick_threshold(y, p, grid=GRID) -> Tuple[float, float]:
    """Threshold maximizing the comprehensive index on (y, p); returns (t, index)."""
    y = np.asarray(y, bool); p = np.asarray(p, float)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else 0.0
    vals = []
    for t in grid:
        m = binary_metrics(y, p >= t)
        vals.append(comprehensive_index(m["accuracy"], m["f1"], auc))
    best = max(vals)
    cands = [t for t, v in zip(grid, vals) if abs(v - best) < 1e-9]
    if any(abs(t - 0.5) < 1e-9 for t in cands):
        return 0.5, best
    return float(min(cands, key=lambda t: abs(t - 0.5))), best


def recalibrate(selection: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-feature recalibration table.

    `selection` and `test`: code -> prediction DataFrame (columns y, p) of the
    deployed checkpoint on the selection cohort and on a test cohort.
    """
    rows = []
    for code, s in selection.items():
        t, idx = pick_threshold(s["y"].values, s["p"].values)
        te = test[code]
        m05 = binary_metrics(te["y"].values, te["p"].values >= 0.5, te["p"].values)
        mt = binary_metrics(te["y"].values, te["p"].values >= t, te["p"].values)
        rows.append({"feature": code, "threshold": t, "selection_index_at_0.5": pick_index(s, 0.5),
                     "selection_index_recalibrated": idx, "test_accuracy_at_0.5": m05["accuracy"],
                     "test_accuracy_recalibrated": mt["accuracy"],
                     "saturated_fraction": float(((s["p"] <= 0.05) | (s["p"] >= 0.95)).mean())})
    return pd.DataFrame(rows).set_index("feature")


def pick_index(s: pd.DataFrame, t: float) -> float:
    from sklearn.metrics import roc_auc_score
    y = s["y"].values.astype(bool); p = s["p"].values
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else 0.0
    m = binary_metrics(y, p >= t)
    return comprehensive_index(m["accuracy"], m["f1"], auc)


def apply_thresholds(test: Dict[str, pd.DataFrame], thresholds: Dict[str, float]) -> pd.DataFrame:
    """Boolean structured report from probabilities and per-feature thresholds."""
    return pd.DataFrame({c: (test[c]["p"].values >= thresholds[c]) for c in thresholds},
                        index=next(iter(test.values())).index)


def test_set_optimal(test: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Overfitted ceiling: per-feature threshold maximizing accuracy on the test
    cohort itself (not a deployable rule; reported only as a bound)."""
    out = {}
    for c, df in test.items():
        y = df["y"].values.astype(bool); p = df["p"].values
        accs = [(float(((p >= t) == y).mean()), t) for t in GRID]
        best = max(a for a, _ in accs)
        out[c] = float(min((t for a, t in accs if abs(a - best) < 1e-12), key=lambda t: abs(t - 0.5)))
    return out
