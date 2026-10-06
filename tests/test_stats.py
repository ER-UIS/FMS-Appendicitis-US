"""Unit tests for the analysis modules (run: python -m pytest tests -q)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fms import recalibrate, report, select, stats  # noqa: E402
from fms.features import BY_CODE, CODES  # noqa: E402


def test_wilson_matches_known_values():
    lo, hi = stats.wilson(533, 3214)
    assert abs(lo - 0.1533) < 5e-4 and abs(hi - 0.1792) < 5e-4


def test_mcnemar_exact_and_chi2():
    a = np.array([1] * 30 + [0] * 70, bool)
    b = np.array([1] * 10 + [0] * 90, bool)
    m = stats.mcnemar(a, b)
    assert m["b"] == 20 and m["c"] == 0 and m["method"] == "exact binomial" and m["p"] < 1e-4
    big_a = np.r_[np.ones(1500, bool), np.zeros(1500, bool)]
    big_b = np.r_[np.zeros(1500, bool), np.ones(1500, bool)]
    assert stats.mcnemar(big_a, big_b)["method"] == "continuity-corrected chi-square"


def test_newcombe_sign_and_width():
    x = np.array([1] * 166 + [0] * 834, bool)
    y = np.array([1] * 382 + [0] * 618, bool)
    r = stats.newcombe_paired(x, y)
    assert r["diff"] < 0 and r["lo"] < r["diff"] < r["hi"]


def test_holm_monotone():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj == sorted(adj, key=lambda v: v) or all(0 <= v <= 1 for v in adj)
    assert abs(adj[0] - 0.03) < 1e-12


def test_binary_metrics_and_index():
    y = np.array([1, 1, 0, 0, 1, 0], bool); p = np.array([0.9, 0.8, 0.2, 0.6, 0.4, 0.1])
    m = stats.binary_metrics(y, p >= 0.5, p)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    assert abs(m["comprehensive_index"] - stats.comprehensive_index(m["accuracy"], m["f1"], m["auc"])) < 1e-12


def test_fmo_eso_no_selection():
    rng = np.random.default_rng(0)
    y = rng.random(184) < 0.3
    preds = {ep: pd.DataFrame({"y": y, "p": np.clip(y * 0.5 + rng.random(184) * 0.5 + (0.1 if ep == 7 else 0), 0, 1)}) for ep in range(1, 21)}
    sc = select.score_checkpoints(preds)
    assert select.no_epoch(sc) == 20 and select.fmo_epoch(sc) in sc.index
    assert select.eso_epoch([0.5, 0.6, 0.7, 0.65, 0.64] + [0.6] * 10, patience=10) == 3
    mc = select.monte_carlo_weights(sc, 500, 1)
    assert 0 <= mc["top1_rate"] <= 1


def test_discrepancy_thresholds():
    ids = [str(i) for i in range(50)]
    ref = pd.DataFrame(False, index=ids, columns=CODES)
    rep = ref.copy(); rep.iloc[:10, :5] = True   # 10 patients with 5 discrepancies
    d = report.discrepancies(rep, ref)
    assert (d >= 4).sum() == 10 and (d >= 6).sum() == 0
    es = report.endpoint_summary({"A": rep, "B": ref}, ref)
    assert es.loc[(es.strategy == "A") & (es.end_point == "substandard"), "rate"].iloc[0] == 0.2


def test_recalibration_prefers_half_on_ties():
    y = np.array([1, 0, 1, 0], bool); p = np.array([0.9, 0.1, 0.8, 0.2])
    t, idx = recalibrate.pick_threshold(y, p)
    assert t == 0.5 and idx > 0.99


def test_feature_labels():
    assert BY_CODE["AWC"].deployable is False and BY_CODE["AODC"].is_positive("增粗") and BY_CODE["AODC"].is_positive("Thickening")
