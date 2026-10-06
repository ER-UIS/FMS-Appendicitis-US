"""Generate nine-feature structured reports for a cohort from deployed checkpoints.

Example
    python scripts/04_structured_reports.py --strategy FMO --data-root model_appendix \
        --images TestImage --labels data/Test_data_3214.xlsx --out results/reports_FMO_3214.xlsx
The deployed epoch of each feature is taken from config/features.json
(deployed_checkpoints) unless --epochs is given, eg --epochs AODC=14 CC=41 ...
"""
import argparse
import glob
import os
import pandas as pd
from _common import ROOT  # noqa: F401
from fms.data import read_labels
from fms.features import BY_CODE, CODES, DEPLOYED
from fms.predict import structured_report

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--strategy", choices=list(DEPLOYED), default="FMO")
p.add_argument("--epochs", nargs="*", default=[], help="override: CODE=epoch ...")
p.add_argument("--data-root", default="model_appendix")
p.add_argument("--images", required=True)
p.add_argument("--labels", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
epochs = {k: int(v) for k, v in DEPLOYED[a.strategy].items() if not k.startswith("_")}
for item in a.epochs:
    k, v = item.split("="); epochs[k] = int(v)
ckpt, cj = {}, {}
for code in CODES:
    pat = os.path.join(a.data_root, code, "models", f"*ex-{epochs[code]:03d}*.h5")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"no snapshot for {code} epoch {epochs[code]}: {pat}")
    ckpt[code] = hits[0]; cj[code] = os.path.join(a.data_root, code, "json", "model_class.json")
labels = read_labels(a.labels)
rep = structured_report(ckpt, cj, a.images, labels.index)
out = pd.DataFrame({c: rep[c].map(lambda v, c=c: BY_CODE[c].positive if v else BY_CODE[c].negative) for c in CODES}, index=rep.index)
out.index.name = "App_No"
out.reset_index().to_excel(a.out, index=False)
print("written", a.out, out.shape)
