"""Feature definitions and label conventions (config/features.json)."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "features.json")


@dataclass(frozen=True)
class Feature:
    code: str
    name: str
    short: str
    positive: str
    negative: str
    zh_column: str
    zh_positive: str
    zh_negative: str
    shap_weight: float
    deployable: bool = True

    def is_positive(self, label) -> bool:
        """True if an English or Chinese label denotes the finding being present."""
        s = str(label).strip()
        return s in (self.positive, self.zh_positive)


def load_config(path: str = _CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_CFG = load_config()
FEATURES: List[Feature] = [
    Feature(code=d["code"], name=d["name"], short=d["short"], positive=d["positive"], negative=d["negative"],
            zh_column=d["zh_column"], zh_positive=d["zh_positive"], zh_negative=d["zh_negative"],
            shap_weight=float(d.get("shap_weight", 0.0)), deployable=bool(d.get("deployable", True)))
    for d in _CFG["features"]
]
CODES: List[str] = [f.code for f in FEATURES]
BY_CODE: Dict[str, Feature] = {f.code: f for f in FEATURES}
BY_ZH: Dict[str, Feature] = {f.zh_column: f for f in FEATURES}
DEPLOYABLE: List[str] = [f.code for f in FEATURES if f.deployable]
WEIGHTS = _CFG["comprehensive_index_weights"]          # accuracy / f1 / auc
THRESHOLDS = _CFG["report_thresholds"]                 # substandard / discarded
DEPLOYED = _CFG["deployed_checkpoints"]                # strategy -> code -> epoch
SEEDS = _CFG["seeds"]


def to_english(label: str, code: str) -> str:
    """Translate a Chinese class label of feature `code` into the English label."""
    f = BY_CODE[code]
    s = str(label).strip()
    if s == f.zh_positive:
        return f.positive
    if s == f.zh_negative:
        return f.negative
    return s  # already English (or unknown)


def positive_mask(series, code: str):
    """Boolean numpy array: label == positive class (English or Chinese)."""
    import numpy as np
    f = BY_CODE[code]
    s = series.astype(str).str.strip()
    return np.asarray((s == f.positive) | (s == f.zh_positive))
