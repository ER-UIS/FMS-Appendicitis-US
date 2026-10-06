"""Checkpoint selection strategies.

* **FMO** (feature-specific model optimization): after training, score every
  saved snapshot of a feature classifier on the expert-verified model-selection
  cohort (n = 184) and deploy the one maximizing the comprehensive index
  0.3 x accuracy + 0.4 x F1 + 0.3 x AUC (ties -> earliest epoch).
* **ESO** (early-stopping optimization): the snapshot fixed *during* training
  by early stopping on the same cohort (accuracy, patience 10); see
  :mod:`fms.train` for the callbacks.  Given a table of per-epoch validation
  accuracies, :func:`eso_epoch` reproduces the rule offline.
* **NO** (no optimization): the final-epoch snapshot.

:func:`monte_carlo_weights` is the weight-sensitivity analysis of eTable 2.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import read_checkpoint_dir
from .stats import binary_metrics


def score_checkpoints(preds: Dict[int, pd.DataFrame], threshold: float = 0.5) -> pd.DataFrame:
    """Metrics of every checkpoint on one cohort; index = epoch."""
    rows = {}
    for ep, df in sorted(preds.items()):
        y = df["y"].astype(bool).values
        pred = (df["p"].values >= threshold) if "p" in df else df["pred"].astype(bool).values
        rows[ep] = binary_metrics(y, pred, df["p"].values if "p" in df else None)
    out = pd.DataFrame(rows).T
    out.index.name = "epoch"
    return out


def fmo_epoch(scores: pd.DataFrame) -> int:
    """Epoch maximizing the comprehensive index (earliest epoch on ties)."""
    best = scores["comprehensive_index"].max()
    return int(min(e for e, v in scores["comprehensive_index"].items() if abs(v - best) < 1e-12))


def no_epoch(scores: pd.DataFrame) -> int:
    """Final epoch."""
    return int(max(scores.index))


def eso_epoch(val_accuracy: Sequence[float], patience: int = 10, min_delta: float = 0.0) -> int:
    """Epoch (1-based) retained by static early stopping on validation accuracy.

    Mirrors ``keras.callbacks.EarlyStopping(monitor='val_accuracy',
    patience=patience, restore_best_weights=True)``: training stops after
    `patience` epochs without improvement and the best epoch is kept.  If the
    run ends before patience is exhausted, the best epoch so far is returned.
    """
    best, best_ep, wait = -np.inf, 1, 0
    for i, v in enumerate(val_accuracy, start=1):
        if v > best + min_delta:
            best, best_ep, wait = v, i, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_ep


def eso_dynamic_epoch(val_accuracy: Sequence[float], initial_threshold: float = 0.9, decay: float = 0.98,
                      patience: int = 10) -> int:
    """ESO2 of eTable 3: dynamic-threshold rule.  The threshold decays by
    `decay` each epoch; training stops when the metric has failed the threshold
    for `patience` consecutive epochs; the best epoch so far is kept."""
    thr, fails, best, best_ep = initial_threshold, 0, -np.inf, 1
    for i, v in enumerate(val_accuracy, start=1):
        if v > best:
            best, best_ep = v, i
        fails = 0 if v >= thr else fails + 1
        if fails >= patience:
            break
        thr *= decay
    return best_ep


def eso_target_epoch(val_accuracy: Sequence[float], target: float = 0.95) -> int:
    """ESO3 of eTable 3: absolute-target rule; stops at the first epoch reaching
    `target` (or returns the best epoch if never reached)."""
    for i, v in enumerate(val_accuracy, start=1):
        if v >= target:
            return i
    return int(np.argmax(val_accuracy) + 1)


def select_all(checkpoint_dirs: Dict[str, str], threshold: float = 0.5) -> pd.DataFrame:
    """FMO / NO selection for every feature from directories of prediction tables.

    `checkpoint_dirs`: code -> directory with one prediction table per epoch
    (written by ``scripts/02_predict_checkpoints.py``).  Returns a table with
    the chosen epochs and their selection-cohort metrics.
    """
    rows = []
    for code, d in checkpoint_dirs.items():
        sc = score_checkpoints(read_checkpoint_dir(d, code), threshold)
        f = fmo_epoch(sc); n = no_epoch(sc)
        rows.append({"feature": code, "FMO_epoch": f, "NO_epoch": n, "n_checkpoints": len(sc),
                     **{f"FMO_{k}": sc.loc[f, k] for k in ("accuracy", "precision", "recall", "f1", "auc", "comprehensive_index")}})
    return pd.DataFrame(rows).set_index("feature")


def monte_carlo_weights(scores: pd.DataFrame, n_iter: int = 10000, seed: int = 2026,
                        concentration: float = 10.0) -> Dict[str, float]:
    """Weight-sensitivity analysis of the FMO choice (eTable 2, last column).

    Weights (accuracy, F1, AUC) are drawn from a Dirichlet distribution centred
    on (0.3, 0.4, 0.3); the proportion of draws in which the deployed FMO
    checkpoint remains the top-1 choice is the *weight-consistency* rate.
    """
    rng = np.random.default_rng(seed)
    base = np.array([0.3, 0.4, 0.3])
    M = scores[["accuracy", "f1", "auc"]].fillna(0).values
    fmo = fmo_epoch(scores)
    eps = list(scores.index)
    W = rng.dirichlet(base * concentration, size=n_iter)
    idx = np.argmax(M @ W.T, axis=0)  # best checkpoint per draw
    top1 = float(np.mean([eps[i] == fmo for i in idx]))
    lo, hi = np.percentile(np.max(M @ W.T, axis=0), [2.5, 97.5])
    return {"fmo_epoch": fmo, "top1_rate": top1, "index_ci_lo": float(lo), "index_ci_hi": float(hi),
            "level": "High" if top1 >= 0.9 else ("Moderate" if top1 >= 0.6 else "Low")}
