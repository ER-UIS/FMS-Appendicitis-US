"""Train the nine feature classifiers (or a subset) and save improving snapshots.

Example
    python scripts/01_train.py --features AODC CC --epochs 100 --data-root model_appendix
    python scripts/01_train.py --early-stopping static      # ESO baseline run
"""
import argparse
from _common import ROOT  # noqa: F401
from fms.features import CODES
from fms.train import train_feature

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--features", nargs="+", default=CODES, choices=CODES)
p.add_argument("--data-root", default="model_appendix")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--image-size", type=int, default=224)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--early-stopping", choices=["none", "static", "dynamic", "target"], default="none",
               help="none = save all improving snapshots (FMO/NO pool); static = ESO (val_accuracy, patience 10)")
p.add_argument("--patience", type=int, default=10)
p.add_argument("--no-augment", action="store_true")
p.add_argument("--pretrained", default="imagenet", help="'imagenet', path to .h5 weights, or 'none'")
a = p.parse_args()
for code in a.features:
    d = train_feature(code, a.data_root, a.epochs, a.batch_size, a.image_size, a.lr, seed=a.seed,
                      augment=not a.no_augment, early_stopping=a.early_stopping, patience=a.patience,
                      pretrained=None if a.pretrained == "none" else a.pretrained)
    print(f"[{code}] snapshots written to {d}")
