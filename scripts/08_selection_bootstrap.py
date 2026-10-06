"""Selection-set bootstrap (eMethods 4; eTables 18-19; eFigure 7).

Example
    python scripts/08_selection_bootstrap.py --pred-root results/selection --B 1000 --out results/etable18.xlsx
"""
import argparse
import json
import os
import pandas as pd
from _common import ROOT  # noqa: F401
from fms.bootstrap import feature_bootstrap, joint_bootstrap
from fms.data import read_checkpoint_dir
from fms.features import CODES, SEEDS

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pred-root", required=True)
p.add_argument("--B", type=int, default=1000)
p.add_argument("--seed", type=int, default=SEEDS["selection_set_bootstrap"])
p.add_argument("--reach", type=float, default=0.01)
p.add_argument("--features", nargs="+", default=CODES)
p.add_argument("--out", required=True)
a = p.parse_args()
CODES = a.features
preds = {c: read_checkpoint_dir(os.path.join(a.pred_root, c), c) for c in CODES}
rows = []
for c in CODES:
    r = feature_bootstrap(preds[c], a.B, a.seed, a.reach)
    rows.append({"feature": c, "deployed_epoch": r["deployed_epoch"], "top1_rate": r["top1_rate"],
                 "reachable": ", ".join(map(str, r["reachable"])), "coverage": r["coverage"], "n_checkpoints": len(preds[c])})
    print(f"{c}: deployed {r['deployed_epoch']}, top-1 {r['top1_rate']:.1%}, reachable {r['reachable']}, coverage {r['coverage']:.1%}")
joint = joint_bootstrap(preds, a.B, a.seed)
with pd.ExcelWriter(a.out) as w:
    pd.DataFrame(rows).to_excel(w, sheet_name="per_feature", index=False)
    joint.to_excel(w, sheet_name="joint_selections", index=False)
print("written", a.out)
