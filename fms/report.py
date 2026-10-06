"""Nine-feature structured reports, discrepancy counts and the study end points.

A structured report is a boolean DataFrame (patients x feature codes; True =
finding present).  Against a reference (original clinical report, or the
blinded expert majority), the *discrepancy count* is the number of features on
which the automated report and the reference disagree.  A report is
*substandard* with >= 4 of 9 discrepancies and *discarded* with >= 6
(``config/features.json``).  The deployable 8-feature configuration excludes
appendiceal wall condition (AWC).
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .features import CODES, DEPLOYABLE, THRESHOLDS
from .stats import format_p, mcnemar, newcombe_paired, wilson


def discrepancies(report: pd.DataFrame, reference: pd.DataFrame, codes: Iterable[str] = CODES) -> pd.Series:
    """Number of discrepant features per patient (index aligned on the report)."""
    codes = list(codes)
    ref = reference.loc[report.index, codes].astype(bool)
    return (report[codes].astype(bool).values != ref.values).sum(1).astype(int)


def threshold_table(reports: Dict[str, pd.DataFrame], reference: pd.DataFrame, codes: Iterable[str] = CODES,
                    kmax: int = 8, comparisons: Optional[List[tuple]] = None) -> pd.DataFrame:
    """Table 2 / eTable 9 / eTable 13: proportion of reports with >= k discrepancies.

    `reports`: strategy name -> boolean report DataFrame on the same patients.
    `comparisons`: list of (a, b) strategy pairs for patient-matched McNemar P
    values (default: every pair in order).
    """
    codes = list(codes)
    names = list(reports)
    d = {k: discrepancies(reports[k], reference, codes) for k in names}
    n = len(next(iter(d.values())))
    if comparisons is None:
        comparisons = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rows = []
    for k in range(1, kmax + 1):
        row = {"threshold": f">={k}"}
        for s in names:
            c = int((d[s] >= k).sum()); lo, hi = wilson(c, n)
            row[s] = f"{c}/{n} ({100 * c / n:.1f}%)"; row[f"{s}_lo"] = lo; row[f"{s}_hi"] = hi
        for a, b in comparisons:
            m = mcnemar(d[a] >= k, d[b] >= k)
            row[f"P {a} vs {b}"] = format_p(m["p"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def endpoint_summary(reports: Dict[str, pd.DataFrame], reference: pd.DataFrame, codes: Iterable[str] = CODES,
                     primary: Optional[str] = None) -> pd.DataFrame:
    """Substandard (>=4) and discarded (>=6) rates with Wilson CIs, and paired
    differences (Newcombe CI, McNemar P) of every strategy versus `primary`
    (default: the first strategy in `reports`)."""
    codes = list(codes)
    primary = primary or next(iter(reports))
    d = {k: discrepancies(v, reference, codes) for k, v in reports.items()}
    n = len(d[primary])
    rows = []
    for name, thr in (("substandard", THRESHOLDS["substandard"]), ("discarded", THRESHOLDS["discarded"])):
        for s in reports:
            ev = d[s] >= thr; c = int(ev.sum()); lo, hi = wilson(c, n)
            row = {"end_point": name, "strategy": s, "events": c, "n": n, "rate": c / n, "ci_lo": lo, "ci_hi": hi}
            if s != primary:
                nc = newcombe_paired(d[primary] >= thr, ev); m = mcnemar(d[primary] >= thr, ev)
                row.update({"diff_vs_primary_pp": nc["diff"], "diff_lo": nc["lo"], "diff_hi": nc["hi"],
                            "mcnemar_b": m["b"], "mcnemar_c": m["c"], "P": format_p(m["p"]), "p_raw": m["p"]})
            rows.append(row)
    return pd.DataFrame(rows)


def alerts(report: pd.DataFrame, original: pd.DataFrame, expert: pd.DataFrame,
           codes: Iterable[str] = DEPLOYABLE) -> pd.DataFrame:
    """Second-reader analysis (eTable 10, Part C).

    An *alert* is a feature on which the automated report disagrees with the
    original clinical report; it is *true* if the expert majority agrees with
    the automated report (ie, the original report was wrong).  Returns per-
    feature alert counts, true alerts, yield and the share of report errors
    flagged.
    """
    codes = list(codes)
    rows = []
    for c in codes:
        r = report.loc[expert.index, c].astype(bool).values
        o = original.loc[expert.index, c].astype(bool).values
        e = expert[c].astype(bool).values
        alert = r != o
        report_error = o != e
        true_alert = alert & (r == e)
        rows.append({"feature": c, "alerts": int(alert.sum()), "true_alerts": int(true_alert.sum()),
                     "yield": true_alert.sum() / alert.sum() if alert.sum() else float("nan"),
                     "report_errors": int(report_error.sum()),
                     "errors_flagged": int((report_error & alert).sum())})
    df = pd.DataFrame(rows).set_index("feature")
    tot = df.sum(numeric_only=True)
    df.loc["Total"] = tot
    df.loc["Total", "yield"] = tot["true_alerts"] / tot["alerts"] if tot["alerts"] else float("nan")
    return df


def quality_score(report: pd.DataFrame, reference: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """SHAP-weighted 0-100 report-quality score (eMethods 2; a within-study construct)."""
    codes = list(weights)
    w = np.array([weights[c] for c in codes], float)
    correct = (report[codes].astype(bool).values == reference.loc[report.index, codes].astype(bool).values)
    return pd.Series(100 * (correct * w).sum(1) / w.sum(), index=report.index)
