# FMS-Appendicitis-US

Code for **"Feature-Specific Checkpoint Selection for Automated Quality Review of Appendiceal Ultrasound Reports: An Exploratory Diagnostic Study"** (submitted to *JAMA Network Open*).

Nine binary image classifiers (InceptionResNetV2, transfer learning) each detect one sonographic feature of the appendix. The deployed checkpoint of each classifier is chosen by **feature-specific model optimization (FMO)**: after training, every saved snapshot is scored on an expert-verified model-selection cohort (n = 184) and the snapshot maximizing the *comprehensive index* (0.3 × accuracy + 0.4 × F1 + 0.3 × AUC) is deployed. The nine feature outputs form a structured report that is compared, feature by feature, with the original clinical report or with a blinded expert majority. The comparators are per-feature early stopping (**ESO**, same cohort, patience 10) and the final epoch (**NO**).

Version 2.0.0 replaces the June-2025 scripts; see `CHANGELOG.md`.

## Repository layout

```
config/features.json        feature definitions, label conventions (EN/ZH), index weights,
                            end-point thresholds, deployed epochs, random seeds
fms/                        importable package
  features.py               feature table, Chinese/English label translation
  data.py                   readers for label tables and checkpoint prediction tables
  train.py                  training with snapshot saving; ESO / ESO2 / ESO3 callbacks      [TensorFlow]
  predict.py                checkpoint inference, image-to-patient max-pooling (eMethods 1)  [TensorFlow]
  select.py                 FMO / ESO / NO selection, Monte Carlo weight sensitivity (eTable 2)
  report.py                 structured reports, discrepancy counts, end points, alerts, quality score
  stats.py                  Wilson, McNemar, Newcombe, Durkalski, κ, paired tests, Holm, bootstrap
  recalibrate.py            threshold recalibration (eMethods 4, eTable 20)
  bootstrap.py              selection-set bootstrap (eMethods 4, eTables 18-19)
scripts/01_train.py ... 08_selection_bootstrap.py   command-line steps (see below)
data/                       Testdata_184.xlsx (model-selection cohort labels),
                            Test_data_3214.xlsx (internal test cohort, original-report labels),
                            Test_data.xlsx (2-patient example for the report generator)
docs/                       user manual of the 2025-06 version (kept for reference)
tests/                      unit tests for the analysis modules
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                              # analysis modules, no TensorFlow needed
```

The analysis modules need only numpy, pandas, scipy, scikit-learn and openpyxl. Training and inference need TensorFlow 2.10-2.15 (Keras 2 API). With TensorFlow ≥ 2.16, install `tf-keras` and set `TF_USE_LEGACY_KERAS=1`. The original study used Keras 2 with Python 3.9 (SciPy 1.7.0) for training and the original analyses, and Python 3.11 (SciPy 1.17.1) for the recalculations; MedCalc 20.014 was used for some confidence intervals (exact binomial for proportions, bootstrap for AUC).

## Data

| File | Content | Availability |
|---|---|---|
| Training images and feature-level annotations (10 445) | per-feature `train/<class>/` folders | Figshare, https://doi.org/10.6084/m9.figshare.29222042 |
| `data/Testdata_184.xlsx` | model-selection cohort: `App_No` + 9 feature labels (expert-verified) | this repository / Figshare |
| `data/Test_data_3214.xlsx` | internal test cohort: `App_No` + 9 feature labels from the original clinical reports | this repository / Figshare |
| Expert-majority labels (panel A, n = 300; panel B, n = 321) and external cohort (n = 544) | same layout | on reasonable request (institutional data-sharing agreements) |

Images are expected as one folder per examination, named by `App_No`, containing the diagnostic-quality images (`.jpg`). Label tables may use the English labels of `config/features.json` or the original Chinese labels; both are read.

## Reproducing the pipeline

Paths below assume the layout of the 2025-06 release (`model_appendix/<CODE>/train`, `model_appendix/<CODE>/test`, `TestImage_crop/<App_No>/`, `TestImage/<App_No>/`). Every script prints `--help`.

```bash
# 1. Train the nine classifiers; every improvement in training accuracy is saved as a snapshot.
python scripts/01_train.py --data-root model_appendix --epochs 100 --seed 2026
#    ESO run of eTable 3 (early stopping on the model-selection cohort):
python scripts/01_train.py --early-stopping static --patience 10

# 2. Score every snapshot on the model-selection cohort (one table per checkpoint and feature)
for F in AODC CC CAE AWC GAC FAC PFCC MSC BFC; do
  python scripts/02_predict_checkpoints.py --feature $F --models model_appendix/$F/models \
     --class-json model_appendix/$F/json/model_class.json --images TestImage_crop \
     --labels data/Testdata_184.xlsx --out results/selection/$F
done

# 3. FMO / NO selection and the Monte Carlo weight-sensitivity analysis (eTable 2)
python scripts/03_select_checkpoints.py --pred-root results/selection --out results/selection_summary.xlsx

# 4. Structured reports of the internal test cohort under each strategy
#    (deployed epochs from config/features.json; override with --epochs CODE=epoch)
python scripts/04_structured_reports.py --strategy FMO --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_FMO_3214.xlsx
python scripts/04_structured_reports.py --strategy ESO --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_ESO_3214.xlsx
python scripts/04_structured_reports.py --strategy NO  --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_NO_3214.xlsx

# 5. Table 2 (and eTable 9 with --deployable): discrepancy thresholds, substandard (>=4) and discarded (>=6) rates
python scripts/05_endpoints.py --reference data/Test_data_3214.xlsx \
   --reports FMO=results/reports_FMO_3214.xlsx,ESO=results/reports_ESO_3214.xlsx,NO=results/reports_NO_3214.xlsx --out results/table2.xlsx

# 6. Expert-reference cohorts (eTables 10 and 15): rates against the panel majority, pooled
#    feature-level metrics with bootstrap CIs, cluster-adjusted McNemar test, alert yield
python scripts/06_expert_cohorts.py --expert expert_majority_300.xlsx --original data/Test_data_3214.xlsx \
   --reports FMO=results/reports_FMO_3214.xlsx,ESO=results/reports_ESO_3214.xlsx,NO=results/reports_NO_3214.xlsx,Routine=data/Test_data_3214.xlsx \
   --out results/etable10_15.xlsx

# 7. Threshold recalibration (eTable 20); needs checkpoint tables on the selection and test cohorts
python scripts/07_recalibration.py --selection-root results/selection --test-root results/test3214 --strategy ESO \
   --reference data/Test_data_3214.xlsx --out results/etable20_ESO.xlsx

# 8. Selection-set bootstrap (eTables 18-19)
python scripts/08_selection_bootstrap.py --pred-root results/selection --B 1000 --out results/etable18.xlsx
```

Table 3 (mean per-feature metrics) is the per-epoch sheet of step 3 evaluated on the test cohort tables of step 2 (`fms.select.score_checkpoints`) with `fms.stats.paired_metric_tests` across the nine features.

### Mapping of paper elements to code

| Paper | Code |
|---|---|
| eMethods 1 image-to-patient aggregation | `fms.predict.predict_checkpoint` |
| Comprehensive index, FMO / NO selection, eTable 2 Monte Carlo column | `fms.select` |
| ESO rules (eTable 3) | `fms.train` callbacks; offline equivalents `fms.select.eso_epoch`, `eso_dynamic_epoch`, `eso_target_epoch` |
| Table 2, eTables 9, 13 | `fms.report.threshold_table`, `endpoint_summary` |
| Table 3 paired tests, Friedman test | `fms.stats.paired_metric_tests`, `friedman` |
| eTables 10, 15 | `scripts/06_expert_cohorts.py` (`fms.stats.durkalski`, `bootstrap_ci`, `cohen_kappa`, `fleiss_kappa`; `fms.report.alerts`) |
| eTable 17 report-quality score | `fms.report.quality_score` |
| eTables 18-19, eFigure 7 | `fms.bootstrap` |
| eTable 20 | `fms.recalibrate` |

## Conventions

* **Positive class** = the finding present (eg, "Thickening", "Presence of fecal stones"); see `config/features.json`.
* **Discrepancy** = automated label ≠ reference label for a feature; **substandard report** = ≥ 4 of 9 discrepancies; **discarded report** = ≥ 6. The 8-feature deployable configuration excludes AWC (appendiceal wall condition), which is withheld from automated output.
* **Deployed epochs** (eTable 2) and the seeds of the published analyses are stored in `config/features.json`. The original training run did not fix a random seed; `scripts/01_train.py` fixes one (`--seed`, default 2026) for new runs, but GPU kernels may still introduce small run-to-run differences.
* `P` values follow the paper: exact binomial McNemar when the discordant total is ≤ 1000, continuity-corrected χ² otherwise; Wilson CIs for proportions; Newcombe CIs for paired differences.

## Licence and citation

Licence: see `LICENSE` (to be added by the authors; MIT is suggested). Please cite the article and this repository (`CITATION.cff`).

## Contact

Corresponding authors: Linyuan Jin (jinlinyuan-dr@foxmail.com) and Qinghai Peng (pqh12079@aliyun.com).
