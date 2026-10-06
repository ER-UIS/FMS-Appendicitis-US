"""Table 2 / eTable 9 / eTable 13: discrepancy thresholds and the study end points.

Example
    python scripts/05_endpoints.py --reference data/Test_data_3214.xlsx \
        --reports FMO=results/reports_FMO_3214.xlsx,ESO=results/reports_ESO_3214.xlsx,NO=results/reports_NO_3214.xlsx \
        --out results/table2.xlsx
Add --deployable to restrict to the 8 deployable features (eTable 9).
"""
import argparse
import pandas as pd
from _common import ROOT, load_reports
from fms.data import labels_to_bool, read_labels
from fms.features import CODES, DEPLOYABLE
from fms.report import endpoint_summary, threshold_table

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--reference", required=True, help="label table used as reference (original reports or expert majority)")
p.add_argument("--reports", required=True, help="NAME=file.xlsx,NAME=file.xlsx (first name is the primary strategy)")
p.add_argument("--deployable", action="store_true")
p.add_argument("--out", required=True)
a = p.parse_args()
codes = DEPLOYABLE if a.deployable else CODES
ref = labels_to_bool(read_labels(a.reference))
reports = load_reports(a.reports)
ids = ref.index.intersection(next(iter(reports.values())).index)
reports = {k: v.loc[ids] for k, v in reports.items()}
primary = next(iter(reports))
tt = threshold_table(reports, ref, codes)
es = endpoint_summary(reports, ref, codes, primary)
with pd.ExcelWriter(a.out) as w:
    tt.to_excel(w, sheet_name="thresholds"); es.to_excel(w, sheet_name="end_points", index=False)
print(tt[[c for c in tt.columns if not (c.endswith("_lo") or c.endswith("_hi"))]].to_string())
print(es.to_string())
print("written", a.out)
