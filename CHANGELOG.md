# Changelog

## 2.1.0 (2026-10-26)

Includes all changes of 2.0.1, which had not been published on GitHub (in particular the corrected ESO rule in
`fms.select.eso_epoch` and the corrected description of the patient-level rule).

- Added: `retraining/` — the five-seed retraining analysis (seeds 1–5; FMO, ESO, NO, ACC, ESOval and ESOdyn selected within each
  run; Supplementary Tables 20 and 27). Paths are set by `FMS_RETRAIN_DATA` / `FMS_ALGO_DIR`; the label readers accept the figshare
  CSV files; training folders may use the Chinese (original) or English (figshare) class names; without the vendored `KerasModels`
  the identical `keras.applications.InceptionResNetV2` is used.
- Added: `report_text/` — report-text feature extraction program (`report_text_extract.py`) and the annotation rules
  (`RULES_zh.md`) used for the report-text reference (Supplementary Table 21). No report text is included.
- Added: `reader_study/` — offline reader-study tool, case manifest and analysis script (Supplementary Table 25); reading-time
  quantiles are now rounded half up (eg, 23.95 → 24.0) as in the article.
- Added: `LICENSE` (MIT) in the repository root.
- Changed: `fms.data.read_labels` also reads the figshare CSV label files (`label_<F>` / `expert_<F>` columns, 0/1 coding).
- Changed: deployed weight files are available from the corresponding authors on request (not part of the figshare record);
  `config/deployed_checkpoints.json` and `notebooks/inference_demo.ipynb` updated accordingly.
- Documentation: README, CITATION.cff, docstrings and config comments now refer to the accompanying article (Tables 1–4,
  Supplementary Tables 1–27, Supplementary Figs. 1–4); analyses of earlier versions that the article does not report (threshold
  recalibration, report-quality score, second-reader alerts) are labelled as such. `config/features.json` records the retraining
  seeds (1–5) and the reader-study sampling seed (20261024). `data/Test_data_3214.xlsx` is described correctly as the report labels
  (original report revised with the stored images, pathology or operative findings and laboratory results).

## 2.0.1 (2026-10-10)

- Fixed: `fms.select.eso_epoch` now implements ESO as actually applied (early stopping on *training* accuracy, patience 10, within the pool-producing run; reproduces all nine deployed ESO epochs from the snapshot pools); the validation-accuracy variant is kept as `eso_val_epoch` (= ESO2 offline).
- Fixed: ESO2 and ESO3 now match the authors' original training scripts. ESO2 = `keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)` with the best-val_accuracy checkpoint (`--early-stopping static`); ESO3 = the authors' `EarlyStoppingDynamicThreshold` callback (val_accuracy, patience 10, improvement floor 0.9; the printed 0.98 decay never affected stopping) (`--early-stopping dynamic`, `fms.select.eso_dynamic_epoch`). The 2.0.0 'target' rule (val_accuracy >= 0.95) did not exist in the original code and was removed; the pool-producing run now stops at 100% training accuracy (`EarlyStoppingAtAccuracy`), as in `ERTrain.py`.
- Added: `notebooks/inference_demo.ipynb` (verify, load and run the nine deployed FMO checkpoints) and `config/deployed_checkpoints.json` (manifest with MD5 sums, epochs and class mappings).
- Fixed: the `patient_level_rule` text in `config/deployed_checkpoints.json` and the docstrings of `fms/predict.py` and `fms/data.py`, now state the rule actually used for all published results: the patient-level class is that of the single most confident image-level prediction (maximum softmax probability over all images and both classes); the earlier text (maximum positive-class probability over images) described a different rule. Code and results unchanged.
- Fixed: `fms/train.py` docstring now states that the `test/` folder of each feature is a random 80:20 image-level validation split of that feature's annotations (validation set of ESO2/ESO3), not the model-selection cohort, which is used only to score snapshots for FMO.
- Documentation: table, figure and appendix references renumbered. No code changes.

## 2.0.0 (2026-10-07)

Restructured from the June-2025 scripts (`FMS_appendix.py`, `ERTrain.py`,
`classification_module.py`, `classification_evaluator.py`, `multi_mfusiond.py`)
into a package (`fms/`) plus thin command-line scripts (`scripts/`).

### Fixed
- `ERTrain.py` did not run: an empty `if enhance_data:` block was a syntax error.
- `Testdata_184.xlsx` has a column named `" FAC"` (leading space); column names are now stripped.
- Legacy Keras 1-style calls (`fit_generator`, `Adam(lr=, decay=)`, `ModelCheckpoint(period=)`) replaced by the TensorFlow 2 / Keras 2 API.
- Fixed mappings in `classification_evaluator.py` (`thickened_prob`, ...) did not match the probability column names actually written by `classification_module.py`; labels and probabilities are now defined once in `config/features.json`.
- Unused imports (psutil, six, packaging, svm, RandomForest) and unreachable helper code removed.

### Added (analyses reported in the paper that the old scripts did not contain)
- ESO baselines: static (`EarlyStopping`, val_accuracy, patience 10), dynamic-threshold rule (see 2.0.1 for the corrected naming), applied to the same model-selection cohort used by FMO (`fms/train.py`, `fms/select.py`).
- Monte Carlo weight-sensitivity analysis of the FMO choice.
- Nine-feature discrepancy counts, substandard (≥4) and discarded (≥6) end points, 8-feature deployable configuration, Wilson CIs, patient-matched McNemar tests (exact / continuity-corrected), Newcombe paired CIs, Holm adjustment.
- Expert-reference analyses: pooled feature-level accuracy / sensitivity / specificity with patient-level bootstrap CIs, cluster-adjusted McNemar (Durkalski) test, Cohen and Fleiss κ, second-reader alert yield.
- Post hoc threshold recalibration with the test-set-optimal ceiling.
- Selection-set bootstrap: per-feature and joint re-selection, reachable sets, coverage.
- SHAP-weighted report-quality score.
- Deployed epochs, seeds and end-point definitions recorded in `config/features.json`; unit tests in `tests/`.
- Per-feature CI method documented: percentile bootstrap (1000 resamples) for accuracy, precision, recall, F1 and AUC, matching the authors' original evaluation script.

### Changed
- `tensorflow.keras.applications.InceptionResNetV2` replaces the vendored `KerasModels` copy (identical architecture; ImageNet weights downloaded by Keras).
- Prediction tables now carry explicit `pred`, `p_pos`, `p_neg`, `image`, `epoch` columns; the legacy layout is still readable (`fms.data.read_predictions`).
