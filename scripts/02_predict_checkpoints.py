"""Score every saved snapshot of a feature on a cohort (one table per checkpoint).

Example (model-selection cohort, 184 patients)
    python scripts/02_predict_checkpoints.py --feature AODC \
        --models model_appendix/AODC/models --class-json model_appendix/AODC/json/model_class.json \
        --images TestImage_crop --labels data/Testdata_184.xlsx --out results/selection/AODC
"""
import argparse
from _common import ROOT  # noqa: F401
from fms.predict import predict_all_checkpoints

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--feature", required=True)
p.add_argument("--models", required=True, help="directory with .h5 snapshots")
p.add_argument("--class-json", required=True)
p.add_argument("--images", required=True, help="root directory with one sub-folder of images per App_No")
p.add_argument("--labels", required=True, help="label table (App_No + feature columns)")
p.add_argument("--out", required=True)
p.add_argument("--image-size", type=int, default=224)
a = p.parse_args()
for f in predict_all_checkpoints(a.feature, a.models, a.class_json, a.images, a.labels, a.out, a.image_size):
    print("written", f)
