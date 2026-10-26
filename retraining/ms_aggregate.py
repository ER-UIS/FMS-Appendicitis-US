# -*- coding: utf-8 -*-
"""
多种子重训 —— 第 3 步：汇总分析（ms_aggregate.py）
==================================================================================
输入：解压后的 多种子结果_上传.zip（runs/、results/）+ 标签与专家组参照表。
对每个种子 × 每种策略组成 9 项特征结构化报告，计算：
  * 内部测试队列（3214 例，报告标签）与外部队列（544 例）不合格报告率（≥4 项不符；9 项与 8 项）
  * 专家组 A（300 例）与 B（321 例）多数意见参照下的不合格报告率（两组分开，不合并）
  * 逐特征准确率、灵敏度、特异度、AUC
  * 每个种子内 FMO 与 ESO 的配对差（McNemar 精确检验）
跨种子给出均值 ± SD、中位数、范围，及"FMO 优于 ESO"的种子数；原始（已部署）训练作为参照行。
用法：python ms_aggregate.py <解压目录> [--labels-internal ...] [--labels-external ...] [--panel-a ...] [--panel-b ...] [--out 汇总.xlsx]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_common import CODES, BY_CODE, STRATEGIES, read_labels  # noqa: E402

CODES8 = [c for c in CODES if c != 'AWC']
PUBLISHED = {('internal', 'FMO', 9): 16.6, ('internal', 'ESO', 9): 38.2, ('external', 'FMO', 9): 18.8, ('external', 'ESO', 9): 30.9}


def mcnemar(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    return n10, n01, (binomtest(min(n10, n01), n10 + n01).pvalue if n10 + n01 else float('nan'))


def load_seed(root, seed_dir):
    """→ {strategy: {cohort: (pred DataFrame bool, p DataFrame)}}, selected epochs, selection metrics"""
    seed = os.path.basename(seed_dir)
    sel_rows, picks = [], {}
    for code in CODES:
        sj = os.path.join(root, 'runs', seed, code, 'selected.json')
        if not os.path.isfile(sj):
            continue
        with open(sj, encoding='utf-8') as f:
            s = json.load(f)
        for strat, v in s['selections'].items():
            picks.setdefault(strat, {})[code] = v['epoch']
            sel_rows.append(dict(seed=seed, feature=code, strategy=strat, **v))
    out = {}
    for strat, ep in picks.items():
        if len(ep) < len(CODES):
            continue
        for coh in ('internal', 'external'):
            P, Q = {}, {}
            for code, e in ep.items():
                f = os.path.join(root, 'results', seed, f'{coh}_{code}.csv')
                if not os.path.isfile(f):
                    break
                d = pd.read_csv(f, dtype={'申请号': str}).set_index('申请号')
                P[code] = d[f'ex{e:03d}_pred'].astype(bool); Q[code] = d[f'ex{e:03d}_p']
            else:
                out.setdefault(strat, {})[coh] = (pd.DataFrame(P)[CODES], pd.DataFrame(Q)[CODES])
    return out, picks, sel_rows


def load_original(root):
    """已部署 FMO（results/original，由 ms_predict.py --original 生成）。"""
    P, Q = {}, {}
    res = {}
    for coh in ('internal', 'external'):
        P, Q = {}, {}
        for code in CODES:
            f = os.path.join(root, 'results', 'original', f'{coh}_{code}.csv')
            if not os.path.isfile(f):
                break
            d = pd.read_csv(f, dtype={'申请号': str}).set_index('申请号')
            c = [x for x in d.columns if x.endswith('_pred')][0]
            P[code] = d[c].astype(bool); Q[code] = d[c.replace('_pred', '_p')]
        else:
            res[coh] = (pd.DataFrame(P)[CODES], pd.DataFrame(Q)[CODES])
    return res


def evaluate(pred, prob, ref, codes):
    ids = [i for i in ref.index if i in pred.index]
    Pp, R = pred.loc[ids, codes], ref.loc[ids, codes]
    err = (Pp != R).sum(axis=1)
    sub = (err >= 4).values
    feat = {}
    for c in codes:
        y = R[c].values; p = Pp[c].values
        from sklearn.metrics import roc_auc_score
        feat[c] = dict(acc=(p == y).mean(), sens=(p & y).sum() / max(y.sum(), 1), spec=(~p & ~y).sum() / max((~y).sum(), 1),
                       auc=roc_auc_score(y, prob.loc[ids, c].values) if 0 < y.sum() < len(y) else np.nan)
    return dict(n=len(ids), substandard=sub, rate=sub.mean() * 100, k=int(sub.sum()), ids=ids, feat=feat,
                feat_acc=(Pp.values == R.values).mean() * 100)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(os.environ.get('FMS_RETRAIN_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')), 'labels') + os.sep
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--labels-internal', default=base + 'internal_test_3214_report_labels.csv')
    ap.add_argument('--labels-external', default=base + 'external_544_expert_labels.csv')
    ap.add_argument('--panel-a', default=base + 'expert_majority_panelA_300.csv')
    ap.add_argument('--panel-b', default=base + 'expert_majority_panelB_321.csv')
    ap.add_argument('--out', default=os.path.join(here, '多种子重训_汇总.xlsx'))
    a = ap.parse_args()

    REF = {'internal': read_labels(a.labels_internal), 'external': read_labels(a.labels_external),
           'panelA': read_labels(a.panel_a), 'panelB': read_labels(a.panel_b)}
    SRC = {'internal': 'internal', 'external': 'external', 'panelA': 'internal', 'panelB': 'internal'}

    seeds = sorted(glob.glob(os.path.join(a.root, 'results', 'seed*')), key=lambda p: int(os.path.basename(p)[4:]))
    rows, frows, sel_all, pick_rows, paired = [], [], [], [], []
    runs = {}
    for sd in seeds:
        res, picks, sel_rows = load_seed(a.root, sd)
        s = os.path.basename(sd)
        runs[s] = res
        sel_all += sel_rows
        for strat, ep in picks.items():
            pick_rows.append(dict(seed=s, strategy=strat, **ep))
    orig = load_original(a.root)
    if orig:
        runs['original'] = {'FMO': orig}

    for s, res in runs.items():
        for strat, cohs in res.items():
            for ref_name, ref in REF.items():
                src = SRC[ref_name]
                if src not in cohs:
                    continue
                pred, prob = cohs[src]
                for codes, tag in ((CODES, 9), (CODES8, 8)):
                    ev = evaluate(pred, prob, ref, codes)
                    rows.append(dict(seed=s, strategy=strat, reference=ref_name, features=tag, n=ev['n'], substandard=ev['k'],
                                     rate=round(ev['rate'], 2), feature_accuracy=round(ev['feat_acc'], 2)))
                    if tag == 9:
                        for c, m in ev['feat'].items():
                            frows.append(dict(seed=s, strategy=strat, reference=ref_name, feature=c, **{k: round(v, 4) for k, v in m.items()}))
            # 种子内 FMO vs 其他策略配对比较
        if 'FMO' in res:
            for other in ('ESO', 'NO', 'ACC', 'ESOval'):
                if other not in res:
                    continue
                for ref_name, ref in REF.items():
                    src = SRC[ref_name]
                    if src not in res['FMO'] or src not in res[other]:
                        continue
                    for codes, tag in ((CODES, 9), (CODES8, 8)):
                        e1 = evaluate(*res['FMO'][src], ref, codes); e2 = evaluate(*res[other][src], ref, codes)
                        n10, n01, p = mcnemar(e1['substandard'], e2['substandard'])
                        paired.append(dict(seed=s, comparison=f'FMO vs {other}', reference=ref_name, features=tag,
                                           FMO=round(e1['rate'], 2), other=round(e2['rate'], 2), diff=round(e1['rate'] - e2['rate'], 2),
                                           FMO_only=n10, other_only=n01, P=p))
    R = pd.DataFrame(rows); FR = pd.DataFrame(frows); PR = pd.DataFrame(paired)
    SEL = pd.DataFrame(sel_all); PK = pd.DataFrame(pick_rows)

    ms = R[R.seed != 'original']
    summ = (ms.groupby(['reference', 'features', 'strategy'])['rate']
            .agg(n_seeds='count', mean='mean', sd='std', median='median', min='min', max='max').round(2).reset_index())
    if 'original' in set(R.seed):
        o = R[R.seed == 'original'][['reference', 'features', 'strategy', 'rate']].rename(columns={'rate': 'original_run'})
        summ = summ.merge(o, on=['reference', 'features', 'strategy'], how='left')
    summ['published'] = [PUBLISHED.get((r.reference, r.strategy, r.features)) for r in summ.itertuples()]
    if len(PR):
        pm = PR[PR.seed != 'original']
        win = (pm.groupby(['comparison', 'reference', 'features'])
               .agg(n_seeds=('diff', 'count'), mean_diff=('diff', 'mean'), sd_diff=('diff', 'std'), min_diff=('diff', 'min'),
                    max_diff=('diff', 'max'), FMO_better=('diff', lambda x: int((x < 0).sum())),
                    P_lt_05=('P', lambda x: int((x < 0.05).sum()))).round(2).reset_index())
    else:
        win = pd.DataFrame()
    fsum = (FR[FR.seed != 'original'].groupby(['reference', 'strategy', 'feature'])[['acc', 'sens', 'spec', 'auc']]
            .agg(['mean', 'std']).round(3)) if len(FR) else pd.DataFrame()

    with pd.ExcelWriter(a.out) as w:
        summ.to_excel(w, sheet_name='跨种子汇总', index=False)
        win.to_excel(w, sheet_name='FMO与其他策略配对', index=False)
        R.to_excel(w, sheet_name='逐种子不合格率', index=False)
        PR.to_excel(w, sheet_name='逐种子配对检验', index=False)
        if len(fsum):
            fsum.to_excel(w, sheet_name='逐特征(跨种子)')
        FR.to_excel(w, sheet_name='逐特征(逐种子)', index=False)
        PK.to_excel(w, sheet_name='选中epoch', index=False)
        SEL.to_excel(w, sheet_name='选择队列指标', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
    print(summ[summ.features == 9].to_string(index=False))
    if len(win):
        print(win[win.features == 9].to_string(index=False))
    print('saved', a.out)


if __name__ == '__main__':
    main()
