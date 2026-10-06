"""Selection-set bootstrap (eMethods 4; eTables 18-19; eFigure 7).

The model-selection cohort (n = 184) is resampled with replacement B times;
in each replicate the FMO rule is re-applied to every saved checkpoint of a
feature.  The *top-1 rate* is the proportion of replicates in which the
deployed checkpoint is selected again; the *reachable set* comprises the
checkpoints selected in at least `reach` of replicates.  Re-testing every
reachable checkpoint on the test cohorts bounds the optimism of the headline
estimate (a sensitivity analysis, not a formal optimism correction).
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from .stats import binary_metrics


def _index_matrix(preds: Dict[int, pd.DataFrame], threshold: float = 0.5):
    eps = sorted(preds)
    ref = preds[eps[0]]
    y = ref["y"].values.astype(bool)
    P = np.column_stack([preds[e].loc[ref.index, "p"].values for e in eps])
    return eps, y, P


def _comp(yb, pb, threshold=0.5):
    m = binary_metrics(yb, pb >= threshold, pb)
    return m["comprehensive_index"]


def feature_bootstrap(preds: Dict[int, pd.DataFrame], B: int = 1000, seed: int = 20260930,
                      reach: float = 0.01, threshold: float = 0.5) -> dict:
    """Bootstrap re-selection for one feature.

    `preds`: epoch -> prediction DataFrame on the selection cohort (y, p).
    Returns the deployed (full-sample) FMO epoch, selection frequencies,
    top-1 rate, reachable set and coverage.
    """
    eps, y, P = _index_matrix(preds, threshold)
    n = len(y)
    full = np.array([_comp(y, P[:, j], threshold) for j in range(len(eps))])
    deployed = eps[int(np.argmax(full))]
    rng = np.random.default_rng(seed)
    chosen = np.empty(B, int)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() in (0, n):  # degenerate resample: fall back to accuracy only
            scores = [float(((P[idx, j] >= threshold) == yb).mean()) for j in range(len(eps))]
        else:
            scores = [_comp(yb, P[idx, j], threshold) for j in range(len(eps))]
        chosen[b] = eps[int(np.argmax(scores))]
    freq = pd.Series(chosen).value_counts(normalize=True).sort_index()
    reachable = [int(e) for e, f in freq.items() if f >= reach]
    return {"deployed_epoch": int(deployed), "top1_rate": float(freq.get(deployed, 0.0)),
            "selection_frequency": {int(k): float(v) for k, v in freq.items()},
            "reachable": reachable, "coverage": float(sum(freq[e] for e in reachable)),
            "choices": chosen}


def joint_bootstrap(preds_by_feature: Dict[str, Dict[int, pd.DataFrame]], B: int = 1000,
                    seed: int = 20260930, threshold: float = 0.5) -> pd.DataFrame:
    """Joint re-selection of all nine checkpoints per replicate (eTable 19).

    All features share the same patient resample in each replicate.  Returns a
    DataFrame (B x features) of selected epochs.
    """
    codes = list(preds_by_feature)
    mats = {c: _index_matrix(preds_by_feature[c], threshold) for c in codes}
    n = len(next(iter(mats.values()))[1])
    rng = np.random.default_rng(seed)
    out = np.empty((B, len(codes)), int)
    for b in range(B):
        idx = rng.integers(0, n, n)
        for j, c in enumerate(codes):
            eps, y, P = mats[c]
            yb = y[idx]
            scores = [_comp(yb, P[idx, k], threshold) if 0 < yb.sum() < n else float(((P[idx, k] >= threshold) == yb).mean())
                      for k in range(len(eps))]
            out[b, j] = eps[int(np.argmax(scores))]
    return pd.DataFrame(out, columns=codes)


def whole_report_sensitivity(choices: pd.DataFrame, rate_lookup) -> np.ndarray:
    """Substandard-report rate of each replicate's configuration.

    `rate_lookup(config: dict code->epoch) -> float or None` returns the rate of
    a configuration (None if a checkpoint was not re-tested).  Replicates with
    an unavailable checkpoint are excluded (reported as NaN).
    """
    rates = []
    for _, row in choices.iterrows():
        r = rate_lookup(row.to_dict())
        rates.append(np.nan if r is None else r)
    return np.asarray(rates, float)
