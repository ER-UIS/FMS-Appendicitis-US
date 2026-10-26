# Five-seed retraining (post hoc robustness analysis)

Code of the retraining analysis in Supplementary Tables 20 and 27 (Methods, "Retraining, ensembles, and selection-cohort
bootstrap"). The nine feature classifiers are retrained five times (seeds 1–5) under the settings of the deployed run;
every epoch is scored on the 184-patient model-selection cohort with the patient-level rule, so that all selection rules
can be applied within the same run: FMO, ESO, NO, accuracy-based selection (ACC), early stopping on selection-cohort
accuracy (ESOval, patience 10) and its dynamic-threshold variant (ESOdyn). The selected checkpoints are then run on the
internal test and external cohorts. The outputs of the published runs are in the figshare record
(https://doi.org/10.6084/m9.figshare.29222042, `04_retraining_five_seeds`).

| file | step |
|---|---|
| `ms_common.py` | shared settings: feature table, cohort paths, image cache, model (InceptionResNetV2 + Dense softmax), patient-level rule, metrics |
| `ms_train.py` | step 1: train 5 seeds × 9 features; per-epoch selection-cohort metrics; checkpoints kept only for the selected epochs (`--check-only` checks the data layout) |
| `ms_predict.py` | step 2: run the selected checkpoints on the internal and external cohorts (`--original` also re-scores the deployed checkpoints) |
| `ms_aggregate.py` | step 3: substandard-draft rates per seed and strategy, within-run paired comparisons, selected epochs (Supplementary Table 20) |
| `ms_uncertainty.py` | seed disagreement and majority-vote ensembles (Supplementary Table 20, Parts D–E) |
| `run_all.bat` | runs steps 1–2 on Windows |

## Data layout
Set `FMS_RETRAIN_DATA` to the data folder (default: `retraining/data`) and, if the original model definition is used,
`FMS_ALGO_DIR` to the folder that contains `KerasModels/` and `TestImage_all/` (default: the repository root; without
`KerasModels/` the identical `keras.applications.InceptionResNetV2` is used).

```
<FMS_RETRAIN_DATA>/
  train/<F>/<positive class>/, train/<F>/<negative class>/   training images per feature (Chinese class names as in the
                                                             original archive, or the English names of the figshare archive)
  selection/images/<App_No>/*.jpg                            model-selection cohort (184)
  external/images/<pseudonym>/*.jpg                          external cohort (not public; available on request)
  labels/model_selection_184_verified_labels.csv             figshare 01_labels (CSV) or the original Excel tables
  labels/internal_test_3214_report_labels.csv
  labels/external_544_expert_labels.csv
  labels/expert_majority_panelA_300.csv, labels/expert_majority_panelB_321.csv
  weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5   (optional; otherwise ~/.keras/models)
<FMS_ALGO_DIR>/TestImage_all/<App_No>/*.jpg                  internal test cohort (3214)
```

Note: the published retraining runs used the uncropped stored frames; the images on figshare are center-cropped
(500 × 450 pixels) to remove burned-in identifying information, so runs on the figshare images reproduce the procedure but
not the exact published values (Supplementary Table 12 compares cropped and uncropped input). Training settings: 224 × 224
input, Adam (learning rate 1e-3, decay 1e-4), batch 16, up to 200 epochs, stopping at 100% training accuracy as in the
original training script. GPU nondeterminism was not disabled, so reruns with the same seed can differ slightly.
