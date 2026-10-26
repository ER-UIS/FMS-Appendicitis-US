# FMS-Appendicitis-US

Code for the article **"Deep learning feature reports as a drafting assistant for single-operator appendiceal ultrasound"**.
Table, figure and Supplementary numbers below refer to that article (Tables 1–4, Figs. 1–5,
Supplementary Tables 1–27 and Supplementary Figs. 1–4).

A server-side system drafts the feature report of an appendiceal ultrasound (US) examination from the stored images, for the
physician to adopt, edit, or rewrite. Nine binary image classifiers (InceptionResNetV2, transfer learning) each read one
sonographic feature. The deployed checkpoint of each classifier is chosen by **feature-specific model optimization (FMO)**:
after training, every saved snapshot is scored on a verified model-selection cohort (n = 184) and the snapshot maximizing the
*comprehensive index* (0.3 × accuracy + 0.4 × F1 + 0.3 × AUC) is deployed. The comparators are early stopping on training
accuracy within the same run (**ESO**, patience 10), the final epoch (**NO**), two separate early-stopping runs on a held-out
20% split (**ESO2**, **ESO3**) and, in the retraining analysis, accuracy-based selection and early stopping on the selection
cohort (ACC, ESOval, ESOdyn). The nine feature outputs form a structured draft that is compared feature by feature with the
report labels or with a blinded expert majority.

Version 2.1.0 accompanies the article; see `CHANGELOG.md`.

## Repository layout

```
config/features.json        feature definitions, label conventions (EN/ZH), index weights, end-point thresholds,
                            deployed epochs, random seeds
config/deployed_checkpoints.json   manifest of the nine deployed FMO weight files (names, MD5, epochs, class mapping)
fms/                        importable package
  features.py               feature table, Chinese/English label translation
  data.py                   readers for label tables and checkpoint prediction tables
  train.py                  training with snapshot saving; ESO2 / ESO3 early-stopping runs          [TensorFlow]
  predict.py                checkpoint inference, patient-level aggregation (most confident image)   [TensorFlow]
  select.py                 FMO / ESO / NO selection, Monte Carlo weight sensitivity (Supplementary Table 2)
  report.py                 structured drafts, discrepancy counts, end points
  stats.py                  Wilson, McNemar, Newcombe, Durkalski, κ, paired tests, Holm, bootstrap
  bootstrap.py              selection-cohort bootstrap (Supplementary Table 7, Supplementary Fig. 3)
  recalibrate.py            threshold recalibration (earlier versions of this work; not reported in the accompanying article)
scripts/01_train.py ... 08_selection_bootstrap.py   command-line steps (see below)
retraining/                 five-seed retraining analysis (Supplementary Tables 20 and 27)
report_text/                report-text feature extraction program and annotation rules (Supplementary Table 21)
reader_study/               reader-study tool and analysis (pilot; Supplementary Table 25)
notebooks/inference_demo.ipynb   runnable demo with the nine deployed FMO checkpoints
data/                       Testdata_184.xlsx (model-selection cohort labels), Test_data_3214.xlsx (internal test cohort,
                            report labels), Test_data.xlsx (2-patient example)
docs/                       user manual of the 2025-06 version (kept for reference)
tests/                      unit tests for the analysis modules
```

## Data

The dataset is on figshare: https://doi.org/10.6084/m9.figshare.29222042 (version 2; file dictionary in its `README_data.md`).

| Content | Where |
|---|---|
| 10 445 feature-level training images; stored images of the model-selection (184) and internal test (3214) cohorts | figshare `FMS/` archives |
| Report labels and final diagnoses (3214), model-selection labels (184), pseudonymized external expert labels (544), blinded expert majority of panel A (300) and panel B (321) | figshare `01_labels`; the 3214 and 184 label tables are also in `data/` |
| Per-patient predictions of every selection strategy; snapshot predictions for the bootstrap; retraining outputs; reader-study records | figshare `02`–`04`, `07` |
| Uncropped stored frames, external-cohort images, individual (non-majority) expert readings, report text | on reasonable request, subject to institutional approval |

`data/Test_data_3214.xlsx` holds the **report labels** of the internal test cohort: labels compiled from the original report and
revised after review of the stored images, pathology or operative findings and laboratory results (identical to figshare
`01_labels/internal_test_3214_report_labels.csv`). `data/Testdata_184.xlsx` holds the model-selection cohort labels, compiled
from the same sources and additionally verified feature by feature against the stored images.

**Images.** All shared images are center-cropped to 500 × 450 pixels to remove burned-in identifying information. The deployed
classifiers were trained and evaluated on the uncropped frames, so runs on the shared images reproduce the procedures but can
differ from the published values (Supplementary Table 12 compares cropped and uncropped input). Images are expected as one folder
per examination, named by `App_No`; label tables may use the English labels of `config/features.json`, the original Chinese
labels, or 0/1 coding.

## Deployed model weights

The nine FMO-selected checkpoints (AODC epoch 14, CC 41, CAE 4, AWC 4, GAC 9, FAC 9, PFCC 9, MSC 46, BFC 4; Supplementary
Table 2) are **available from the corresponding authors on request**. They are Keras HDF5 weights-only files: rebuild the
architecture with `fms.train.build_model(2, 224, weights=None)` and call `load_weights`. `config/deployed_checkpoints.json` lists
file names, MD5 sums, epochs and the class-index mapping; `notebooks/inference_demo.ipynb` verifies the files, loads the models,
scores a patient folder and assembles the structured draft. The patient-level rule is that of the original pipeline: the class of
the single most confident image-level prediction across all images of the patient. Appendiceal wall condition (AWC) is withheld
from the drafts and is computed only to reproduce research metrics. Run on all 3214 internal test examinations, the weight files
reproduce the published FMO predictions exactly (28 926 feature-level predictions, 0 disagreements); a nine-feature draft took a
median of 1.8 s per examination on one Quadro P4000 GPU.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                              # analysis modules, no TensorFlow needed
```

The analysis modules need only numpy, pandas, scipy, scikit-learn and openpyxl. Training and inference need TensorFlow 2.10–2.15
(Keras 2 API); with TensorFlow ≥ 2.16 install `tf-keras` and set `TF_USE_LEGACY_KERAS=1`. The original study used Keras 2 with
Python 3.9 for training and the original analyses, and Python 3.11 for the recalculations. Per-feature 95% CIs (accuracy,
precision, recall, F1, AUC) are percentile bootstrap intervals with 1000 resamples of the test cohort (`fms.stats.bootstrap_ci`),
computed without a fixed seed, so repeated runs may differ in the last digit.

## Reproducing the pipeline

Paths below assume the layout of the 2025-06 release (`model_appendix/<CODE>/train`, `model_appendix/<CODE>/test`,
`TestImage_crop/<App_No>/`, `TestImage/<App_No>/`). Every script prints `--help`.

```bash
# 1. Train the nine classifiers; every improvement in training accuracy is saved as a snapshot.
python scripts/01_train.py --data-root model_appendix --epochs 100 --seed 2026
#    ESO2 / ESO3 runs (Table 4, Supplementary Tables 3 and 18): separate runs with early stopping on the 20% validation split
python scripts/01_train.py --early-stopping static --patience 10     # ESO2
python scripts/01_train.py --early-stopping dynamic --patience 10    # ESO3

# 2. Score every snapshot on the model-selection cohort (one table per checkpoint and feature)
for F in AODC CC CAE AWC GAC FAC PFCC MSC BFC; do
  python scripts/02_predict_checkpoints.py --feature $F --models model_appendix/$F/models \
     --class-json model_appendix/$F/json/model_class.json --images TestImage_crop \
     --labels data/Testdata_184.xlsx --out results/selection/$F
done

# 3. FMO / ESO / NO selection and the Monte Carlo weight-sensitivity analysis (Supplementary Table 2)
python scripts/03_select_checkpoints.py --pred-root results/selection --out results/selection_summary.xlsx

# 4. Structured drafts of the internal test cohort under each strategy
#    (deployed epochs from config/features.json; override with --epochs CODE=epoch)
python scripts/04_structured_reports.py --strategy FMO --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_FMO_3214.xlsx
python scripts/04_structured_reports.py --strategy ESO --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_ESO_3214.xlsx
python scripts/04_structured_reports.py --strategy NO  --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_NO_3214.xlsx

# 5. Table 4 (and Supplementary Table 4 with --deployable): discrepancy thresholds, substandard (>=4) and discarded (>=6) rates
python scripts/05_endpoints.py --reference data/Test_data_3214.xlsx \
   --reports FMO=results/reports_FMO_3214.xlsx,ESO=results/reports_ESO_3214.xlsx,NO=results/reports_NO_3214.xlsx --out results/table4.xlsx

# 6. Expert-reference cohorts (Table 2, Supplementary Tables 6, 13 and 17): rates against the panel majority, pooled feature-level
#    metrics with bootstrap CIs, cluster-adjusted McNemar test (figshare 01_labels/expert_majority_panelA_300.csv)
python scripts/06_expert_cohorts.py --expert expert_majority_panelA_300.csv --original data/Test_data_3214.xlsx \
   --reports FMO=results/reports_FMO_3214.xlsx,ESO=results/reports_ESO_3214.xlsx,NO=results/reports_NO_3214.xlsx,Labels=data/Test_data_3214.xlsx \
   --out results/expert_panelA.xlsx

# 7. Selection-cohort bootstrap (Supplementary Table 7, Supplementary Fig. 3)
python scripts/08_selection_bootstrap.py --pred-root results/selection --B 1000 --out results/selection_bootstrap.xlsx
```

The five-seed retraining (Supplementary Tables 20 and 27), the report-text extraction (Supplementary Table 21) and the reader
study (Supplementary Table 25) have their own folders and READMEs. `scripts/07_recalibration.py` (threshold recalibration) and
`fms.report.quality_score` / `fms.report.alerts` belong to earlier versions of this work and are not reported in the accompanying article.

### Mapping of article elements to code

| Article | Code |
|---|---|
| Methods, patient-level aggregation | `fms.predict.predict_checkpoint` |
| Comprehensive index, FMO / NO selection; weight sensitivity (Supplementary Table 2) | `fms.select` |
| ESO rules (Table 4; Supplementary Tables 3 and 18) | `fms.train` callbacks (`--early-stopping static` = ESO2, `dynamic` = ESO3); offline equivalents `fms.select.eso_epoch` (ESO: training accuracy, patience 10), `eso_val_epoch` (ESO2), `eso_dynamic_epoch` (ESO3) |
| Table 4; Supplementary Tables 4, 13 and 15 | `fms.report.threshold_table`, `endpoint_summary` |
| Mean per-feature metrics and paired tests (Supplementary Tables 2 and 26, Supplementary Fig. 2) | `fms.select.score_checkpoints`, `fms.stats.paired_metric_tests`, `friedman` |
| Table 2; Supplementary Tables 6, 13 and 17 | `scripts/06_expert_cohorts.py` (`fms.stats.durkalski`, `bootstrap_ci`, `cohen_kappa`, `fleiss_kappa`) |
| Supplementary Table 7, Supplementary Fig. 3 | `fms.bootstrap` |
| Supplementary Tables 20 and 27 | `retraining/` |
| Supplementary Table 21 (report-text columns) | `report_text/` |
| Supplementary Table 25 | `reader_study/` |

## Conventions

* **Positive class** = the finding present (eg, "Thickening", "Presence of fecal stones"); see `config/features.json`.
* **Discrepancy** = draft label ≠ reference label for a feature; **substandard draft** = ≥ 4 of 9 discrepancies (a study-defined
  discrepancy burden, not clinical acceptability); **discarded** = ≥ 6. The eight-feature deployable configuration excludes AWC.
* **Deployed epochs** and the seeds of the published analyses are stored in `config/features.json`. The deployed training run did
  not fix a random seed; the retraining runs used seeds 1–5; `scripts/01_train.py` fixes one (`--seed`, default 2026) for new runs,
  but GPU kernels may still introduce small run-to-run differences.
* `P` values: exact binomial McNemar when the discordant total is ≤ 1000, continuity-corrected χ² otherwise; Wilson CIs for
  proportions; Newcombe CIs for paired differences; cluster-adjusted McNemar (Durkalski) for feature-level comparisons.

## Licence and citation

MIT licence (`LICENSE`). Please cite the article and this repository (`CITATION.cff`).

## Contact

Corresponding author: Jian Li (erzying@foxmail.com); co-corresponding authors: Min Zhang (18163663960@163.com), Qinghai Peng
(pqh12079@aliyun.com) and Linyuan Jin (jinlinyuan-dr@foxmail.com). Weight files and data available on request: contact the
corresponding authors.
