"""Readers for the spreadsheets used by the pipeline.

Two spreadsheet layouts occur:

1. *Label tables* (reference labels per patient), e.g. ``Testdata_184.xlsx``,
   ``Test_data_3214.xlsx``: one row per patient, column ``App_No`` (patient /
   examination identifier) and one column per feature code (AODC ... BFC)
   holding the English class label.  Chinese column names (阑尾外径 ...) and
   Chinese labels are accepted and translated.

2. *Checkpoint prediction tables* written by :mod:`fms.predict` (one file per
   checkpoint): columns ``App_No``, ``<CODE>`` (reference label, if known),
   ``pred`` (predicted label), ``p_pos`` (positive-class probability of the
   max-pooled image), ``p_neg``, ``image`` (name of the image that produced
   the maximum) and ``epoch``.
"""
from __future__ import annotations
import os
import re
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .features import BY_CODE, BY_ZH, CODES, to_english

ID_COL = "App_No"
_ID_ALIASES = ("App_No", "app_no", "申请号", "ID", "id", "patient_id")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, translate Chinese feature columns, unify the id column."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for c in df.columns:
        if c in BY_ZH:
            ren[c] = BY_ZH[c].code
        elif c in _ID_ALIASES and c != ID_COL:
            ren[c] = ID_COL
        else:
            base = c.replace("_correct", "")
            if base in BY_ZH:
                ren[c] = BY_ZH[base].code
    df = df.rename(columns=ren)
    if ID_COL not in df.columns:
        raise KeyError(f"No patient identifier column found; expected one of {_ID_ALIASES}. Columns: {list(df.columns)}")
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    return df


def read_labels(path: str, codes: Iterable[str] = CODES) -> pd.DataFrame:
    """Read a label table; returns a DataFrame indexed by App_No with English labels."""
    df = _normalise_columns(pd.read_excel(path))
    df = df.set_index(ID_COL)
    out = pd.DataFrame(index=df.index)
    for code in codes:
        if code in df.columns:
            out[code] = df[code].astype(str).str.strip().map(lambda v, c=code: to_english(v, c))
    return out


def labels_to_bool(labels: pd.DataFrame) -> pd.DataFrame:
    """Convert English/Chinese labels to booleans (True = finding present)."""
    out = pd.DataFrame(index=labels.index)
    for code in labels.columns:
        f = BY_CODE[code]
        s = labels[code].astype(str).str.strip()
        out[code] = (s == f.positive) | (s == f.zh_positive)
    return out


_EPOCH_RE = re.compile(r"ex-(\d+)")


def epoch_from_name(name: str) -> Optional[int]:
    """Epoch number encoded in a checkpoint or result file name (``...ex-041...``)."""
    m = _EPOCH_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def read_predictions(path: str, code: str) -> pd.DataFrame:
    """Read one checkpoint prediction table.

    Returns a DataFrame indexed by App_No with columns ``y`` (reference label
    as bool, NaN if absent), ``pred`` (bool) and ``p`` (positive-class
    probability).  Accepts both the layout written by :mod:`fms.predict` and
    the legacy layout of the original study (Chinese column names, the two
    trailing columns being the class probabilities).
    """
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    f = BY_CODE[code]
    cols = list(raw.columns)
    if "p_pos" in cols:  # new layout
        df = _normalise_columns(raw).set_index(ID_COL)
        y = df[code].map(f.is_positive) if code in df.columns else pd.Series(np.nan, index=df.index)
        return pd.DataFrame({"y": y, "pred": df["pred"].map(f.is_positive), "p": df["p_pos"].astype(float)})
    # legacy layout: id, reference, predicted, ..., prob(class A), prob(class B)
    ids = raw[cols[0]].astype(str).str.strip()
    prob_cols = cols[-2:]
    pos_prob = [c for c in prob_cols if f.positive in c or f.zh_positive in c or not (c.startswith("无") or "连续" in c or c.startswith("No "))]
    pos_prob = pos_prob[0]
    return pd.DataFrame({"y": raw[cols[1]].map(f.is_positive).values, "pred": raw[cols[2]].map(f.is_positive).values,
                         "p": raw[pos_prob].astype(float).values}, index=ids)


def read_checkpoint_dir(directory: str, code: str) -> Dict[int, pd.DataFrame]:
    """Read every prediction table in a directory, keyed by epoch."""
    out = {}
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".xlsx") or name.startswith("~$"):
            continue
        ep = epoch_from_name(name)
        if ep is None:
            continue
        out[ep] = read_predictions(os.path.join(directory, name), code)
    if not out:
        raise FileNotFoundError(f"No prediction tables with an 'ex-NNN' epoch tag in {directory}")
    return out
