"""Post hoc threshold recalibration of the deployed checkpoints (eMethods 4; eTable 20).

Example
    python scripts/07_recalibration.py --selection-root results/selection --test-root results/test3214 \
        --strategy ESO --reference data/Test_data_3214.xlsx --out results/etable20_ESO.xlsx
Expects <selection-root>/<CODE>/<CODE>_ex-NNN.xlsx and <test-root>/<CODE>/<CODE>_ex-NNN.xlsx
for the deployed epoch of each feature (config/features.json).
"""
import argparse
import os
import pandas as pd
from _common import ROOT  # noqa: F401
from fms.data import labels_to_bool, read_labels, read_predictions
from fms.features import CODES, DEPLOYED
from fms.recalibrate import apply_thresholds, pick_threshold, recalibrate, test_set_optimal
from fms.report import endpoint_summary

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--selection-root", required=True)
p.add_argument("--test-root", required=True)
p.add_argument("--strategy", choices=list(DEPLOYED), default="ESO")
p.add_argument("--reference", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
ep = DEPLOYED[a.strategy]
sel = {c: read_predictions(os.path.join(a.selection_root, c, f"{c}_ex-{ep[c]:03d}.xlsx"), c) for c in CODES}
tst = {c: read_predictions(os.path.join(a.test_root, c, f"{c}_ex-{ep[c]:03d}.xlsx"), c) for c in CODES}
tab = recalibrate(sel, tst)
thr = tab["threshold"].to_dict()
ref = labels_to_bool(read_labels(a.reference))
rep05 = apply_thresholds(tst, {c: 0.5 for c in CODES}); repT = apply_thresholds(tst, thr)
repO = apply_thresholds(tst, test_set_optimal(tst))
ids = ref.index.intersection(rep05.index)
es = endpoint_summary({f"{a.strategy}_0.5": rep05.loc[ids], f"{a.strategy}_recalibrated": repT.loc[ids],
                       f"{a.strategy}_test_optimal_ceiling": repO.loc[ids]}, ref.loc[ids], CODES, f"{a.strategy}_0.5")
with pd.ExcelWriter(a.out) as w:
    tab.to_excel(w, sheet_name="thresholds"); es.to_excel(w, sheet_name="end_points", index=False)
print(tab.round(3).to_string()); print(es.to_string()); print("written", a.out)
