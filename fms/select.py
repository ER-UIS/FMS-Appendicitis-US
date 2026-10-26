"""Checkpoint selection strategies.

* **FMO** (feature-specific model optimization): after training, score every
  saved snapshot of a feature classifier on the expert-verified model-selection
  cohort (n = 184) and deploy the one maximizing the comprehensive index
  0.3 x accuracy + 0.4 x F1 + 0.3 x AUC (ties -> earliest epoch).
* **ESO** (early-stopping optimization): early stopping on *training* accuracy
  (patience 10) within the same run -- the last snapshot before 10 consecutive
  epochs without improvement (:func:`eso_epoch`, applied to the snapshot
  epochs).  ESO2 and ESO3 of Table 4 and Supplementary Table 18 are separate runs with early stopping on
  validation accuracy (:func:`eso_val_epoch`, :func:`eso_dynamic_epoch`; see
  :mod:`fms.train` for the callbacks).
* **NO** (no optimization): the final-epoch snapshot.

:func:`monte_carlo_weights` is the weight-sensitivity analysis of Supplementary Table 2.
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


def eso_epoch(snapshot_epochs: Sequence[int], patience: int = 10) -> int:
    """ESO: epoch of the snapshot deployed by early stopping on *training* accuracy.

    The pool-producing run (``ERTrain.py``) saves a snapshot whenever training
    accuracy improves, so the gaps between snapshot epochs are runs of
    non-improving epochs.  ESO deploys the last snapshot before training
    accuracy failed to improve for `patience` consecutive epochs; if the run
    ended (training accuracy 100%) before that happened, the last snapshot is
    deployed and ESO coincides with NO.  Reproduces the nine deployed ESO
    epochs of Supplementary Table 2 from the snapshot pools (AODC 14, CC 46, CAE 22, AWC 57,
    GAC 25, FAC 31, PFCC 40, MSC 25, BFC 26).
    """
    eps = sorted(int(e) for e in snapshot_epochs)
    for cur, nxt in zip(eps, eps[1:]):
        if nxt - cur - 1 >= patience:
            return cur
    return eps[-1]


def eso_val_epoch(val_accuracy: Sequence[float], patience: int = 10, min_delta: float = 0.0) -> int:
    """Offline equivalent of the ESO2 run of Table 4 and Supplementary Table 18: ``keras.callbacks.EarlyStopping``
    on per-epoch validation accuracy (patience) with the best-val_accuracy epoch deployed."""
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
    """Offline equivalent of the ESO3 run of Table 4 and Supplementary Table 18 (dynamic-threshold
    callback): val_accuracy must exceed max(`initial_threshold`, best so far) to count as an
    improvement; training stops after `patience` consecutive non-improving epochs and the
    best-val_accuracy epoch is kept.  `decay` is accepted for completeness; in the original
    callback it only changed a printed value and never affected stopping."""
    best, best_ep, wait = initial_threshold, 1, 0
    seen_best = False
    for i, v in enumerate(val_accuracy, start=1):
        if v > best:
            best, best_ep, wait, seen_best = v, i, 0, True
        else:
            wait += 1
            if wait >= patience:
                break
    if not seen_best:  # never exceeded the floor: ModelCheckpoint would have kept the best epoch anyway
        best_ep = int(np.argmax(val_accuracy) + 1)
    return best_ep


def select_all(checkpoint_dirs: Dict[str, str], threshold: float = 0.5) -> pd.DataFrame:
    """FMO / ESO / NO selection for every feature from directories of prediction tables.

    `checkpoint_dirs`: code -> directory with one prediction table per epoch
    (written by ``scripts/02_predict_checkpoints.py``).  Returns a table with
    the chosen epochs and their selection-cohort metrics.
    """
    rows = []
    for code, d in checkpoint_dirs.items():
        sc = score_checkpoints(read_checkpoint_dir(d, code), threshold)
        f = fmo_epoch(sc); n = no_epoch(sc); e = eso_epoch(list(sc.index))
        rows.append({"feature": code, "FMO_epoch": f, "ESO_epoch": e, "NO_epoch": n, "n_checkpoints": len(sc),
                     **{f"FMO_{k}": sc.loc[f, k] for k in ("accuracy", "precision", "recall", "f1", "auc", "comprehensive_index")}})
    return pd.DataFrame(rows).set_index("feature")


def monte_carlo_weights(scores: pd.DataFrame, n_iter: int = 10000, seed: int = 2026,
                        concentration: float = 10.0) -> Dict[str, float]:
    """Weight-sensitivity analysis of the FMO choice (Supplementary Table 2, last column).

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
