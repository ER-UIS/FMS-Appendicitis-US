"""Shared helpers for the command-line scripts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_reports(spec: str):
    """Parse 'NAME=path.xlsx,NAME2=path2.xlsx' into {name: boolean report DataFrame}."""
    from fms.data import labels_to_bool, read_labels
    out = {}
    for item in spec.split(","):
        name, path = item.split("=", 1)
        out[name.strip()] = labels_to_bool(read_labels(path.strip()))
    return out
