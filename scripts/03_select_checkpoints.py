"""FMO and NO checkpoint selection from the selection-cohort prediction tables,
with the Monte Carlo weight-sensitivity analysis (eTable 2).

Example
    python scripts/03_select_checkpoints.py --pred-root results/selection --out results/selection_summary.xlsx
    (expects results/selection/<CODE>/*.xlsx written by 02_predict_checkpoints.py)
"""
import argparse
import os
import pandas as pd
from _common import ROOT  # noqa: F401
from fms.data import read_checkpoint_dir
from fms.features import CODES
from fms.select import fmo_epoch, monte_carlo_weights, no_epoch, score_checkpoints

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pred-root", required=True)
p.add_argument("--out", default="results/selection_summary.xlsx")
p.add_argument("--features", nargs="+", default=CODES)
p.add_argument("--mc-iter", type=int, default=10000)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
rows, per_epoch = [], {}
for code in a.features:
    sc = score_checkpoints(read_checkpoint_dir(os.path.join(a.pred_root, code), code))
    per_epoch[code] = sc
    f, n = fmo_epoch(sc), no_epoch(sc)
    mc = monte_carlo_weights(sc, a.mc_iter, a.seed)
    rows.append({"feature": code, "FMO_epoch": f, "NO_epoch": n, "n_checkpoints": len(sc),
                 **{f"FMO_{k}": round(sc.loc[f, k], 4) for k in ("accuracy", "precision", "recall", "f1", "auc", "comprehensive_index")},
                 "MC_top1_rate": round(mc["top1_rate"], 4), "MC_level": mc["level"]})
    print(f"{code}: FMO epoch {f} (index {sc.loc[f, 'comprehensive_index']:.3f}), NO epoch {n}, MC top-1 {mc['top1_rate']:.1%}")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with pd.ExcelWriter(a.out) as w:
    pd.DataFrame(rows).to_excel(w, sheet_name="selection", index=False)
    for code, sc in per_epoch.items():
        sc.round(4).to_excel(w, sheet_name=f"{code}_per_epoch")
print("written", a.out)
