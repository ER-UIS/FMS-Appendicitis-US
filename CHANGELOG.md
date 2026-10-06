# Changelog

## 2.0.0 (2026-10-07) — version accompanying the JAMA Network Open submission

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
- ESO baselines: static (`EarlyStopping`, val_accuracy, patience 10), dynamic-threshold (ESO2) and absolute-target (ESO3) rules, applied to the same model-selection cohort used by FMO (`fms/train.py`, `fms/select.py`).
- Monte Carlo weight-sensitivity analysis of the FMO choice (eTable 2).
- Nine-feature discrepancy counts, substandard (≥4) and discarded (≥6) end points, 8-feature deployable configuration, Wilson CIs, patient-matched McNemar tests (exact / continuity-corrected), Newcombe paired CIs, Holm adjustment (Table 2, eTables 9, 13).
- Expert-reference analyses: pooled feature-level accuracy / sensitivity / specificity with patient-level bootstrap CIs, cluster-adjusted McNemar (Durkalski) test, Cohen and Fleiss κ, second-reader alert yield (eTables 10, 15).
- Post hoc threshold recalibration with the test-set-optimal ceiling (eMethods 4, eTable 20).
- Selection-set bootstrap: per-feature and joint re-selection, reachable sets, coverage (eMethods 4, eTables 18-19).
- SHAP-weighted report-quality score (eMethods 2).
- Deployed epochs, seeds and end-point definitions recorded in `config/features.json`; unit tests in `tests/`.

### Changed
- `tensorflow.keras.applications.InceptionResNetV2` replaces the vendored `KerasModels` copy (identical architecture; ImageNet weights downloaded by Keras).
- Prediction tables now carry explicit `pred`, `p_pos`, `p_neg`, `image`, `epoch` columns; the legacy layout is still readable (`fms.data.read_predictions`).
