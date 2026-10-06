"""FMS-Appendicitis-US: feature-specific model optimization (FMO) for automated
quality review of appendiceal ultrasound reports.

Modules
-------
features     feature definitions, label conventions, Chinese/English translation
data         readers for the Excel files produced by the pipeline
predict      checkpoint inference with image-to-patient max-pooling (eMethods 1)
select       FMO / ESO / NO checkpoint selection and the comprehensive index
report       nine-feature structured reports, discrepancy counts, end points
stats        statistical procedures used in the paper (eMethods 2)
recalibrate  post hoc threshold recalibration (eMethods 4, eTable 20)
bootstrap    selection-set bootstrap (eMethods 4, eTables 18-19)
train        model training with snapshot saving and early-stopping rules

Deep-learning modules (train, predict) require TensorFlow 2.x with the Keras 2
API; the analysis modules only need numpy, pandas, scipy and scikit-learn.
"""
__version__ = "2.0.0"
