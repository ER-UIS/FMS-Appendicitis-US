"""Expert-reference analyses (eTable 10 / eTable 15): substandard rates against the
panel majority, pooled feature-level accuracy / sensitivity / specificity with
patient-level bootstrap CIs and the cluster-adjusted McNemar test, and the
second-reader (alert) analysis.

Example
    python scripts/06_expert_cohorts.py --expert data/expert_majority_300.xlsx \
        --original data/Test_data_3214.xlsx \
        --reports FMO=results/reports_FMO_3214.xlsx,ESO=...,NO=...,Routine=data/Test_data_3214.xlsx \
        --out results/etable10_15.xlsx
"""
import argparse
import numpy as np
import pandas as pd
from _common import ROOT, load_reports
from fms.data import labels_to_bool, read_labels
from fms.features import CODES, DEPLOYABLE
from fms.report import alerts, endpoint_summary, threshold_table
from fms.stats import bootstrap_ci, durkalski, format_p

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--expert", required=True, help="expert-majority label table (panel A or panel B cohort)")
p.add_argument("--original", required=True, help="original clinical report labels (for the alert analysis)")
p.add_argument("--reports", required=True, help="NAME=file,... (first = primary, usually FMO)")
p.add_argument("--B", type=int, default=2000)
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--out", required=True)
a = p.parse_args()
expert = labels_to_bool(read_labels(a.expert)); ids = expert.index
original = labels_to_bool(read_labels(a.original)).loc[ids]
reports = {k: v.loc[ids] for k, v in load_reports(a.reports).items()}
primary = next(iter(reports))
sheets = {"thresholds_9f": threshold_table(reports, expert, CODES), "end_points_9f": endpoint_summary(reports, expert, CODES, primary),
          "thresholds_8f": threshold_table(reports, expert, DEPLOYABLE), "end_points_8f": endpoint_summary(reports, expert, DEPLOYABLE, primary)}
# pooled feature-level metrics (eTable 15, Part A/C)
G = expert[CODES].values; n = len(ids); rows = []
Cp = (reports[primary][CODES].values == G)
for name, rep in reports.items():
    R = rep[CODES].values; C = (R == G)
    tp = (G & R).sum(); fn = (G & ~R).sum(); tn = (~G & ~R).sum(); fp = (~G & R).sum()
    def stat(idx, R=R):
        return float((R[idx] == G[idx]).mean())
    lo, hi = bootstrap_ci(stat, n, a.B, a.seed)
    row = {"method": name, "correct": int(C.sum()), "total": C.size, "accuracy": C.mean(), "acc_lo": lo, "acc_hi": hi,
           "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)}
    if name != primary:
        d = durkalski(Cp, C); row.update({"cluster_McNemar_stat": d["statistic"], "P_vs_primary": format_p(d["p"])})
    rows.append(row)
sheets["feature_level"] = pd.DataFrame(rows)
sheets["alerts"] = alerts(reports[primary], original, expert, DEPLOYABLE)
with pd.ExcelWriter(a.out) as w:
    for k, v in sheets.items():
        v.to_excel(w, sheet_name=k)
for k in ("end_points_9f", "end_points_8f", "feature_level", "alerts"):
    print(f"== {k}\n{sheets[k].to_string()}")
print("written", a.out)
