"""FMS-Appendicitis-US: feature-specific model optimization (FMO) for automated
quality review of appendiceal ultrasound reports.

Modules
-------
features     feature definitions, label conventions, Chinese/English translation
data         readers for the Excel files produced by the pipeline
predict      checkpoint inference; patient-level class = most confident image-level prediction (Methods, Patient-level aggregation)
select       FMO / ESO / NO checkpoint selection and the comprehensive index
report       nine-feature structured reports, discrepancy counts, end points
stats        statistical procedures used in the paper (Methods, Statistical analysis)
recalibrate  post hoc threshold recalibration (earlier versions of this work; not reported in the accompanying article)
bootstrap    selection-set bootstrap (Supplementary Table 7 and Supplementary Fig. 3)
train        model training with snapshot saving and early-stopping rules

Deep-learning modules (train, predict) require TensorFlow 2.x with the Keras 2
API; the analysis modules only need numpy, pandas, scipy and scikit-learn.
"""
__version__ = "2.1.0"
