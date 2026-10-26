"""Training of one feature classifier (InceptionResNetV2, transfer learning).

Reproduces the training set-up of the study:

* ImageNet-pretrained InceptionResNetV2 backbone (224 x 224 x 3 input, global
  average pooling) with a 2-class softmax head; Adam (lr 1e-3, decay 1e-4),
  categorical cross-entropy; batch size 16; online augmentation (horizontal
  flip, 10% shifts); pixel rescaling 1/255.
* A *snapshot* is saved whenever training accuracy improves
  (``ModelCheckpoint(monitor='accuracy', save_best_only=True)``), producing the
  candidate checkpoint pool used by FMO.
* Each feature's annotations were split randomly by image 80:20 into training
  (``train/``) and validation (``test/``) portions.  The validation portion only
  monitors accuracy; it is the validation set of the two separate
  validation-based early-stopping runs (ESO2, ESO3 of Table 4 and Supplementary Table 18).  ESO itself
  uses training accuracy (patience 10) within the pool-producing run, and the
  expert-verified model-selection cohort (n = 184) is used only to score the
  snapshots for FMO (``scripts/02_predict_checkpoints.py``).

Directory layout expected under ``--data-root`` (default ``model_appendix``)::

    <root>/<CODE>/train/<class label>/*.jpg     training images
    <root>/<CODE>/test/<class label>/*.jpg      validation images (20% image-level split)
    <root>/<CODE>/models/                       snapshots (written)
    <root>/<CODE>/json/model_class.json         class index -> label (written)
    <root>/<CODE>/logs/                         training log (written)

Requires TensorFlow 2.x with the Keras 2 API (tensorflow < 2.16, or
``TF_USE_LEGACY_KERAS=1`` with tf-keras installed).
"""
from __future__ import annotations
import csv
import json
import os
import random
import time
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and TensorFlow seeds (determinism is still limited by GPU kernels)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def build_model(num_classes: int = 2, image_size: int = 224, weights: Optional[str] = "imagenet"):
    """InceptionResNetV2 backbone + softmax head.

    `weights`: 'imagenet', a path to an .h5 weight file, or None.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    base = tf.keras.applications.InceptionResNetV2(include_top=False, weights=weights if weights in ("imagenet", None) else None,
                                                   input_shape=(image_size, image_size, 3), pooling="avg")
    if weights not in ("imagenet", None):
        base.load_weights(weights, by_name=True, skip_mismatch=True)
    out = layers.Dense(num_classes, activation="softmax", use_bias=True)(base.output)
    return models.Model(inputs=base.input, outputs=out)


class EarlyStoppingAtAccuracy:
    """Stopping rule of the pool-producing run (``ERTrain.py``): stop at the first epoch
    whose *training* accuracy reaches `target` (1.0)."""

    def __new__(cls, target=1.0):
        import tensorflow as tf

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                v = (logs or {}).get("accuracy")
                if v is not None and v >= target:
                    self.model.stop_training = True
        return _CB()


class EarlyStoppingDynamic:
    """ESO3 of Table 4 and Supplementary Table 18 (the authors' ``EarlyStoppingDynamicThreshold``
    callback): early stopping on ``val_accuracy`` with `patience` epochs and an *improvement
    floor* -- ``best`` starts at `initial` (0.9), so an epoch counts as an improvement only
    when val_accuracy exceeds max(initial, best so far).  The original callback also printed
    a threshold multiplied by `decay` (0.98) after each non-improving epoch; that value
    never entered the stopping decision and is kept here only for the log line.  The
    deployed checkpoint of such a run is the epoch with the highest val_accuracy
    (``ModelCheckpoint(monitor='val_accuracy', save_best_only=True)``)."""

    def __new__(cls, initial=0.9, decay=0.98, patience=10):
        import tensorflow as tf

        class _CB(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.best, self.thr, self.wait, self.best_w = initial, initial, 0, None

            def on_epoch_end(self, epoch, logs=None):
                v = (logs or {}).get("val_accuracy")
                if v is None:
                    return
                if v > self.best:
                    self.best, self.wait, self.thr, self.best_w = v, 0, initial, self.model.get_weights()
                else:
                    self.wait += 1; self.thr *= decay
                    if self.wait >= patience:
                        self.model.stop_training = True
                        if self.best_w is not None:
                            self.model.set_weights(self.best_w)
        return _CB()


def train_feature(code: str, data_root: str = "model_appendix", epochs: int = 100, batch_size: int = 16,
                  image_size: int = 224, learning_rate: float = 1e-3, decay: float = 1e-4, seed: int = 2026,
                  augment: bool = True, early_stopping: str = "none", patience: int = 10,
                  pretrained: Optional[str] = "imagenet") -> str:
    """Train the classifier of feature `code`; returns the snapshot directory.

    `early_stopping`: 'none' (save all improving snapshots, stop when training
    accuracy reaches 100%; FMO, ESO and NO are then chosen offline from the
    pool, see :mod:`fms.select`, so a single run serves all three strategies as
    in the study), 'static' (ESO2 run of Table 4 and Supplementary Table 18: ``keras.callbacks.EarlyStopping``
    on val_accuracy, patience 10, best-val_accuracy checkpoint) or 'dynamic'
    (ESO3 run of Table 4 and Supplementary Table 18: the authors' dynamic-threshold callback, see
    :class:`EarlyStoppingDynamic`).
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    set_seed(seed)
    root = os.path.join(data_root, code)
    d_train, d_val = os.path.join(root, "train"), os.path.join(root, "test")
    d_models, d_json, d_logs = (os.path.join(root, s) for s in ("models", "json", "logs"))
    for d in (d_models, d_json, d_logs):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(d_train):
        raise FileNotFoundError(f"training directory not found: {d_train}")

    shift = 0.1 if augment else 0.0
    train_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=augment, height_shift_range=shift, width_shift_range=shift)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)
    train = train_gen.flow_from_directory(d_train, target_size=(image_size, image_size), batch_size=batch_size,
                                          class_mode="categorical", seed=seed)
    val = val_gen.flow_from_directory(d_val, target_size=(image_size, image_size), batch_size=batch_size,
                                      class_mode="categorical", shuffle=False) if os.path.isdir(d_val) else None
    classes = {str(v): k for k, v in train.class_indices.items()}
    with open(os.path.join(d_json, "model_class.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    model = build_model(len(classes), image_size, pretrained)
    try:
        opt = Adam(learning_rate=learning_rate, decay=decay)
    except TypeError:  # Keras 3 removed `decay`
        opt = Adam(learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, 1, decay))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt = ModelCheckpoint(os.path.join(d_models, "InceptionResNetV2-imagenet-transfer-ERmodel_ex-{epoch:03d}_acc-{accuracy:.4f}.h5"),
                           monitor="accuracy", save_best_only=True, save_weights_only=True, verbose=1)
    callbacks = [ckpt, CSVLogger(os.path.join(d_logs, f"train_{stamp}.csv"))]
    if early_stopping == "static":
        callbacks.append(EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1))
    elif early_stopping == "dynamic":
        callbacks.append(EarlyStoppingDynamic(patience=patience))
    elif early_stopping == "none":
        callbacks.append(EarlyStoppingAtAccuracy())
    else:
        raise ValueError("early_stopping must be none|static|dynamic")

    model.fit(train, epochs=epochs, validation_data=val, callbacks=callbacks, verbose=1)
    with open(os.path.join(d_logs, f"run_{stamp}.json"), "w") as f:
        json.dump({"feature": code, "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
                   "learning_rate": learning_rate, "decay": decay, "seed": seed, "augment": augment,
                   "early_stopping": early_stopping, "patience": patience, "pretrained": pretrained,
                   "n_train": train.samples, "n_val": val.samples if val else 0, "classes": classes,
                   "tensorflow": tf.__version__}, f, indent=2, ensure_ascii=False)
    return d_models
