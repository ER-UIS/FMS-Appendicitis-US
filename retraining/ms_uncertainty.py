# -*- coding: utf-8 -*-
"""
多种子重训 —— 第 4 步（云端）：多种子分歧作为低可信度标注、跨种子选择、人机融合模拟
==================================================================================
用法：python ms_uncertainty.py <解压目录> [--report-text 报告原文核对表.xlsx] [--out 结果.xlsx]

① 种子间分歧（低可信度标注）：每个病例每个特征，5 个种子的 FMO 预测中阳性票数 k；
   0<k<n 记为"低可信度"。以多数票为预测，比较被标注与未标注特征的错误率、标注覆盖的错误比例，
   并与"同样标注比例下按平均 softmax 概率最接近 0.5 选出"的标注比较；另报告原始（已部署）FMO
   的错误被分歧标注覆盖的比例。参照：专家组 A、B（分开）、内部报告标签、外部。
② 跨种子选择：每个特征取选择队列综合指数最高的种子的 FMO 检查点；以及 5 种子多数票集成。
③ 人机融合模拟（需 --report-text）：以报告原文为底，报告"未提及"且 AI 判阳性的特征补为阳性；
   变体：AI = 原 FMO / 多数票 / 一致票（5/5）。报告每例补充数、补充正确率、遗漏召回率、不合格率。
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_common import CODES, BY_CODE, read_labels

CODES8 = [c for c in CODES if c != 'AWC']
BASE = os.path.join(os.environ.get('FMS_RETRAIN_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')), 'labels') + os.sep


def mcn(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); x = int((a & ~b).sum()); y = int((~a & b).sum())
    return binomtest(min(x, y), x + y).pvalue if x + y else float('nan')


def load_fmo(root, coh):
    """→ pred[seed] (DataFrame bool), prob[seed], selection composite[seed][code]"""
    P, Q, C = {}, {}, {}
    for sd in sorted(glob.glob(os.path.join(root, 'results', 'seed*'))):
        s = os.path.basename(sd); ok = True; pp, qq, cc = {}, {}, {}
        for code in CODES:
            sj = os.path.join(root, 'runs', s, code, 'selected.json'); f = os.path.join(sd, f'{coh}_{code}.csv')
            if not (os.path.isfile(sj) and os.path.isfile(f)):
                ok = False; break
            sel = json.load(open(sj, encoding='utf-8'))['selections']['FMO']; e = sel['epoch']
            d = pd.read_csv(f, dtype={'申请号': str}).set_index('申请号')
            pp[code] = d[f'ex{e:03d}_pred'].astype(bool); qq[code] = d[f'ex{e:03d}_p']; cc[code] = sel['composite']
        if ok:
            P[s] = pd.DataFrame(pp)[CODES]; Q[s] = pd.DataFrame(qq)[CODES]; C[s] = cc
    orig = None
    if all(os.path.isfile(os.path.join(root, 'results', 'original', f'{coh}_{c}.csv')) for c in CODES):
        o = {}
        for c in CODES:
            d = pd.read_csv(os.path.join(root, 'results', 'original', f'{coh}_{c}.csv'), dtype={'申请号': str}).set_index('申请号')
            o[c] = d[[x for x in d.columns if x.endswith('_pred')][0]].astype(bool)
        orig = pd.DataFrame(o)[CODES]
    return P, Q, C, orig


def sub_rate(P, R, codes):
    s = ((P[codes] != R[codes]).sum(axis=1) >= 4).values
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--report-text', default=None)
    ap.add_argument('--out', default='多种子_低可信度与融合.xlsx'); a = ap.parse_args()
    REF = {'panelA': (read_labels(BASE + 'expert_majority_panelA_300.csv'), 'internal'),
           'panelB': (read_labels(BASE + 'expert_majority_panelB_321.csv'), 'internal'),
           'internal': (read_labels(BASE + 'internal_test_3214_report_labels.csv'), 'internal'),
           'external': (read_labels(BASE + 'external_544_expert_labels.csv'), 'external')}
    data = {coh: load_fmo(a.root, coh) for coh in ('internal', 'external')}
    rows_u, rows_s, rows_f = [], [], []
    for rname, (R, coh) in REF.items():
        P, Q, C, orig = data[coh]
        if len(P) < 2:
            continue
        seeds = sorted(P)
        ids = [i for i in R.index if all(i in P[s].index for s in seeds)]
        R = R.loc[ids, CODES]
        votes = sum(P[s].loc[ids].astype(int) for s in seeds); n = len(seeds)
        maj = votes > n / 2
        flag = (votes > 0) & (votes < n)
        meanp = sum(Q[s].loc[ids] for s in seeds) / n
        err = (maj != R)
        for codes, tag in ((CODES, 9), (CODES8, 8)):
            fl = flag[codes].values; er = err[codes].values
            k = fl.sum()
            # softmax comparator: same number of flags, closest mean probability to 0.5
            conf = (meanp[codes] - 0.5).abs().values.ravel()
            thr = np.sort(conf)[k - 1] if k > 0 else -1
            fl_sm = (conf <= thr).reshape(fl.shape)
            r = dict(reference=rname, features=tag, seeds=n, cells=fl.size, flagged=int(k), flag_rate=round(k / fl.size * 100, 1),
                     err_rate_flagged=round(er[fl].mean() * 100, 1) if k else np.nan, err_rate_unflagged=round(er[~fl].mean() * 100, 1),
                     errors_captured=round(er[fl].sum() / max(er.sum(), 1) * 100, 1),
                     softmax_errors_captured=round(er[fl_sm].sum() / max(er.sum(), 1) * 100, 1),
                     flags_per_exam=round(fl.sum() / len(ids), 2))
            if orig is not None:
                eo = (orig.loc[ids, codes] != R[codes]).values
                r['orig_FMO_errors_captured'] = round(eo[fl].sum() / max(eo.sum(), 1) * 100, 1)
            rows_u.append(r)
        # ② cross-seed selection and ensemble
        best = pd.DataFrame({c: P[max(seeds, key=lambda s: C[s][c])][c] for c in CODES}).loc[ids]
        cands = {**{s: P[s].loc[ids] for s in seeds}, 'cross-seed best': best, 'majority vote': maj}
        if orig is not None:
            cands['original deployed'] = orig.loc[ids]
        for k_, Pk in cands.items():
            for codes, tag in ((CODES, 9), (CODES8, 8)):
                s = sub_rate(Pk, R, codes)
                rows_s.append(dict(reference=rname, features=tag, model=k_, substandard=int(s.sum()), rate=round(s.mean() * 100, 2),
                                   feature_accuracy=round((Pk[codes] == R[codes]).values.mean() * 100, 2)))
        # ③ fusion with report text
        if a.report_text and rname in ('panelA', 'panelB'):
            raw = pd.read_excel(a.report_text, sheet_name='核对表', dtype={'申请号': str}).set_index('申请号')
            RT, MEN = {}, {}
            for c in CODES:
                col = [x for x in raw.columns if x.startswith(c + '\n人工') or x.startswith(f'{c} ') and '人工' in x][0]
                RT[c] = raw[col] == '阳性'; MEN[c] = raw[col] != '未提及'
            RT = pd.DataFrame(RT).loc[ids]; MEN = pd.DataFrame(MEN).loc[ids]
            fill = [c for c in CODES if c not in ('AODC', 'AWC')]
            ai = {'majority vote': maj, f'unanimous ({n}/{n})': votes == n}
            if orig is not None:
                ai['original FMO'] = orig.loc[ids]
            for name, A in [('report text', None)] + list(ai.items()):
                F_ = RT.copy(); added = correct = 0
                if A is not None:
                    for c in fill:
                        add = (~MEN[c]) & A[c]; F_.loc[add, c] = True
                        added += int(add.sum()); correct += int((add & R[c]).sum())
                miss = int(sum(((~MEN[c]) & R[c]).sum() for c in fill))
                s8 = sub_rate(F_, R, CODES8); s0 = sub_rate(RT, R, CODES8)
                gp = R[CODES8].values; pp = F_[CODES8].values
                rows_f.append(dict(reference=rname, AI=name, added=added, added_per_exam=round(added / len(ids), 2),
                                   added_correct_pct=round(correct / added * 100, 1) if added else np.nan,
                                   omissions_recovered_pct=round(correct / miss * 100, 1) if miss else np.nan,
                                   substandard8=int(s8.sum()), rate8=round(s8.mean() * 100, 1), P_vs_report=mcn(s8, s0) if A is not None else np.nan,
                                   sens8=round((pp & gp).sum() / gp.sum() * 100, 1), spec8=round((~pp & ~gp).sum() / (~gp).sum() * 100, 1)))
    U, S, Fu = pd.DataFrame(rows_u), pd.DataFrame(rows_s), pd.DataFrame(rows_f)
    with pd.ExcelWriter(a.out) as w:
        U.to_excel(w, sheet_name='①种子分歧标注', index=False); S.to_excel(w, sheet_name='②跨种子选择与集成', index=False)
        Fu.to_excel(w, sheet_name='③人机融合模拟', index=False)
    pd.set_option('display.width', 250)
    print(U.to_string(index=False)); print(S[S.features == 9].to_string(index=False)); print(Fu.to_string(index=False)); print('saved', a.out)


if __name__ == '__main__':
    main()
