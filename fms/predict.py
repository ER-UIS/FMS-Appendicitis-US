"""Checkpoint inference with image-to-patient aggregation (eMethods 1).

For a patient with images i = 1..m, each image is scored by the classifier and
the patient-level prediction is the class with the highest probability across
all images (max-pooling); the probability of that image is recorded.  One
prediction table per checkpoint is written as
``<out>/<CODE>_ex-<epoch>.xlsx`` with columns App_No, <CODE> (reference label
if available), pred, p_pos, p_neg, image, epoch.

Requires TensorFlow 2.x (Keras 2 API).
"""
from __future__ import annotations
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data import ID_COL, epoch_from_name, read_labels
from .features import BY_CODE

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _load_images(folder: str, image_size: int):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXT))
    arr = np.stack([img_to_array(load_img(os.path.join(folder, n), target_size=(image_size, image_size))) / 255.0 for n in names]) if names else None
    return names, arr


def predict_checkpoint(code: str, weights_path: str, class_json: str, image_root: str, ids: Iterable[str],
                       labels: Optional[pd.DataFrame] = None, image_size: int = 224, batch_size: int = 32) -> pd.DataFrame:
    """Patient-level predictions of one checkpoint for the patients in `ids`."""
    from .train import build_model
    with open(class_json, "r", encoding="utf-8") as f:
        classes = json.load(f)                       # index -> label
    f_ = BY_CODE[code]
    pos_idx = [int(k) for k, v in classes.items() if f_.is_positive(v)]
    if len(pos_idx) != 1:
        raise ValueError(f"cannot identify the positive class of {code} in {class_json}: {classes}")
    pos_idx = pos_idx[0]
    model = build_model(len(classes), image_size, weights=None)
    model.load_weights(weights_path)
    rows = []
    for pid in ids:
        folder = os.path.join(image_root, str(pid))
        if not os.path.isdir(folder):
            continue
        names, arr = _load_images(folder, image_size)
        if arr is None:
            continue
        probs = model.predict(arr, batch_size=batch_size, verbose=0)
        best = int(np.unravel_index(np.argmax(probs), probs.shape)[0])  # image with the highest class probability
        p_pos = float(probs[best, pos_idx])
        rows.append({ID_COL: str(pid), code: labels.loc[str(pid), code] if labels is not None and str(pid) in labels.index else None,
                     "pred": f_.positive if p_pos >= 0.5 else f_.negative, "p_pos": p_pos, "p_neg": 1.0 - p_pos,
                     "image": names[best], "epoch": epoch_from_name(weights_path)})
    return pd.DataFrame(rows)


def predict_all_checkpoints(code: str, models_dir: str, class_json: str, image_root: str, label_file: str,
                            out_dir: str, image_size: int = 224) -> List[str]:
    """Run every snapshot in `models_dir` on the patients of `label_file`."""
    labels = read_labels(label_file)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name in sorted(os.listdir(models_dir)):
        if not name.endswith(".h5"):
            continue
        ep = epoch_from_name(name)
        df = predict_checkpoint(code, os.path.join(models_dir, name), class_json, image_root, labels.index, labels, image_size)
        path = os.path.join(out_dir, f"{code}_ex-{ep:03d}.xlsx")
        df.to_excel(path, index=False)
        written.append(path)
    return written


def structured_report(checkpoints: Dict[str, str], class_jsons: Dict[str, str], image_root: str, ids: Iterable[str],
                      image_size: int = 224) -> pd.DataFrame:
    """Nine-feature structured report (boolean DataFrame) from deployed checkpoints.

    `checkpoints`: code -> weights path; `class_jsons`: code -> model_class.json.
    """
    ids = [str(i) for i in ids]
    cols = {}
    for code, w in checkpoints.items():
        df = predict_checkpoint(code, w, class_jsons[code], image_root, ids, None, image_size).set_index(ID_COL)
        cols[code] = df["p_pos"].reindex(ids) >= 0.5
    return pd.DataFrame(cols, index=ids)
