# -*- coding: utf-8 -*-
"""
阅读实验统计（analyze_reader_study.py）
用法：python analyze_reader_study.py <结果csv文件夹> [--key cases_reference.csv] [--out 阅读实验_统计.xlsx]
输入：阅读工具导出的 CSV（每位阅读者一个），以及 cases_reference.csv（含专家多数意见、草稿、最终诊断；阅读者不可见）。
输出：xlsx（各表）+ 屏幕摘要。统计口径：
  1. 有草稿条件下的处置：直接采纳 / 修改后采纳 / 重写 的例数与比例（Wilson 95% CI），按阅读者、按最终诊断分层；修改项数分布及被修改的特征。
  2. 用时：有草稿 vs 无草稿，中位数（IQR）；病例级配对差（同一病例两种条件由不同阅读者完成，设计上两位阅读者在两种条件各占一半），Wilcoxon 符号秩检验；
     各阅读者内部的两条件比较（Mann–Whitney）。
  3. 最终报告对专家多数意见的准确率（8 项可部署特征）：有草稿 vs 无草稿（病例级配对，聚类校正 McNemar），草稿本身的准确率，
     以及阅读者对草稿的每处修改是"改对"还是"改错"。
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from math import sqrt
from scipy.stats import wilcoxon, mannwhitneyu, chi2

C8 = ['AODC', 'CC', 'CAE', 'GAC', 'FAC', 'PFCC', 'MSC', 'BFC']
ZH = dict(AODC='阑尾外径', CC='粪石', CAE='腹腔积液', GAC='阑尾腔内积气', FAC='阑尾腔内积液', PFCC='阑尾周边积液', MSC='阑尾周围系膜', BFC='阑尾区血流')

def r1(x):
    """round half up to one decimal (avoids binary floating-point artefacts such as 23.95 -> 23.9)"""
    from decimal import Decimal, ROUND_HALF_UP
    return f"{Decimal(repr(round(float(x), 6))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}"
def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d
def pct(k, n):
    lo, hi = wilson(k, n); return f'{k}/{n} ({100 * k / n:.1f}%; 95% CI {lo:.1f}–{hi:.1f})' if n else '—'
def durk(ca, cb):
    """cluster-adjusted McNemar (Durkalski): ca, cb = correctness matrices [n clusters, k]"""
    b = (ca & ~cb).sum(1); c = (~ca & cb).sum(1); den = ((b - c) ** 2).sum()
    st = (b.sum() - c.sum()) ** 2 / den if den > 0 else float('nan'); return st, chi2.sf(st, 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--key', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    key = pd.read_csv(a.key or os.path.join(here, 'cases_reference.csv'), dtype={'App_No': str}).set_index('App_No')
    files = sorted(glob.glob(os.path.join(a.folder, '*.csv')))
    if not files: sys.exit('没有找到结果 CSV')
    R = pd.concat([pd.read_csv(f, dtype={'case_id': str}) for f in files], ignore_index=True)
    R = R.drop_duplicates(['reader', 'case_id'], keep='last')
    print(f'读取 {len(files)} 个文件，{len(R)} 条记录；阅读者 {sorted(R.reader.unique())}')
    R['dx'] = key.loc[R.case_id, 'final_diagnosis'].values
    R['dx_group'] = R.dx.map(lambda d: '正常阑尾' if d == 'normal appendix' else ('坏疽性阑尾炎' if d == 'gangrenous appendicitis' else '化脓性/单纯性阑尾炎'))
    out = {}
    # ---------- 1. decisions ----------
    W = R[R.condition == 'with']
    rows = []
    def dec_rows(sub, label):
        n = len(sub)
        for d, zh in [('adopt', '直接采纳'), ('edit', '修改后采纳'), ('rewrite', '重写')]:
            rows.append(dict(分层=label, 处置=zh, 例数=int((sub.decision == d).sum()), 比例=pct(int((sub.decision == d).sum()), n)))
        rows.append(dict(分层=label, 处置='采纳或修改（≤3 项改动）', 例数=int(((sub.decision == 'adopt') | ((sub.decision == 'edit') & (sub.n_changed <= 3))).sum()), 比例=pct(int(((sub.decision == 'adopt') | ((sub.decision == 'edit') & (sub.n_changed <= 3))).sum()), n)))
    dec_rows(W, '全部')
    for r, sub in W.groupby('reader'): dec_rows(sub, f'阅读者 {r}')
    for g, sub in W.groupby('dx_group'): dec_rows(sub, g)
    out['1_处置'] = pd.DataFrame(rows)
    chg = []
    for c in C8:
        m = W[W.decision != 'rewrite']
        chg.append(dict(特征=ZH[c], 被修改例数=int((m[f'final_{c}'] != m[f'draft_{c}']).sum()), 占有草稿例数=pct(int((m[f'final_{c}'] != m[f'draft_{c}']).sum()), len(m))))
    out['1b_修改的特征'] = pd.DataFrame(chg)
    out['1c_修改项数分布'] = W.n_changed.value_counts().sort_index().rename_axis('修改项数').reset_index(name='例数')
    # ---------- 2. time ----------
    trows = []
    for cond, zh in [('with', '有草稿'), ('without', '无草稿')]:
        s = R[R.condition == cond].seconds
        trows.append(dict(条件=zh, 例数=len(s), 中位数_s=r1(s.median()), IQR=f'{r1(s.quantile(.25))}–{r1(s.quantile(.75))}', 均值_s=r1(s.mean())))
    out['2_用时'] = pd.DataFrame(trows)
    piv = R.pivot_table(index='case_id', columns='condition', values='seconds', aggfunc='first')
    paired = piv.dropna()
    res2 = {}
    if len(paired) >= 10:
        st, p = wilcoxon(paired['with'], paired['without'])
        d = paired['with'] - paired['without']
        res2['病例级配对（有−无）'] = f'n = {len(paired)}；中位差 {r1(d.median())} s（IQR {r1(d.quantile(.25))} 至 {r1(d.quantile(.75))}）；Wilcoxon p = {p:.3g}；有草稿更快的病例 {int((d < 0).sum())}/{len(d)}'
    for r, sub in R.groupby('reader'):
        a1 = sub[sub.condition == 'with'].seconds; a0 = sub[sub.condition == 'without'].seconds
        if len(a1) and len(a0):
            p = mannwhitneyu(a1, a0).pvalue
            res2[f'阅读者 {r}'] = f'有草稿中位 {a1.median():.1f} s vs 无草稿 {a0.median():.1f} s；Mann–Whitney p = {p:.3g}'
    # by decision
    for d, zh in [('adopt', '直接采纳'), ('edit', '修改后采纳'), ('rewrite', '重写')]:
        s = W[W.decision == d].seconds
        if len(s): res2[f'有草稿·{zh}'] = f'n = {len(s)}，中位 {s.median():.1f} s'
    out['2b_用时比较'] = pd.DataFrame(list(res2.items()), columns=['比较', '结果'])
    # ---------- 3. accuracy vs expert majority ----------
    def corr_matrix(sub):
        return np.stack([(sub[f'final_{c}'].values == key.loc[sub.case_id, f'expert_{c}'].values) for c in C8], 1)
    acc_rows = []
    for cond, zh in [('with', '有草稿'), ('without', '无草稿')]:
        sub = R[R.condition == cond]; cm = corr_matrix(sub)
        acc_rows.append(dict(条件=zh, 病例数=len(sub), 评估数=cm.size, 准确率=pct(int(cm.sum()), cm.size)))
    dm = np.stack([(key.loc[W.case_id, f'draft_{c}'].values == key.loc[W.case_id, f'expert_{c}'].values) for c in C8], 1)
    acc_rows.append(dict(条件='草稿本身（有草稿病例）', 病例数=len(W), 评估数=dm.size, 准确率=pct(int(dm.sum()), dm.size)))
    # paired comparison with vs without (same case, different reader)
    common = sorted(set(R[R.condition == 'with'].case_id) & set(R[R.condition == 'without'].case_id))
    if common:
        w = R[(R.condition == 'with')].set_index('case_id').loc[common]; wo = R[(R.condition == 'without')].set_index('case_id').loc[common]
        cw = np.stack([(w[f'final_{c}'].values == key.loc[common, f'expert_{c}'].values) for c in C8], 1)
        cwo = np.stack([(wo[f'final_{c}'].values == key.loc[common, f'expert_{c}'].values) for c in C8], 1)
        st, p = durk(cw, cwo)
        acc_rows.append(dict(条件=f'配对比较（n = {len(common)} 例）', 病例数=len(common), 评估数=cw.size, 准确率=f'有草稿 {100 * cw.mean():.1f}% vs 无草稿 {100 * cwo.mean():.1f}%；聚类校正 McNemar p = {p:.3g}'))
        # reader's final vs draft: did edits help?
        edits = []
        for c in C8:
            ch = w[f'final_{c}'].values != w[f'draft_{c}'].values
            good = ch & (w[f'final_{c}'].values == key.loc[common, f'expert_{c}'].values)
            edits.append(dict(特征=ZH[c], 修改数=int(ch.sum()), 改向专家意见=int(good.sum()), 改离专家意见=int(ch.sum() - good.sum())))
        out['3b_修改的方向'] = pd.DataFrame(edits)
    out['3_准确率'] = pd.DataFrame(acc_rows)
    # per-feature accuracy by condition
    pf = []
    for c in C8:
        r = dict(特征=ZH[c])
        for cond, zh in [('with', '有草稿'), ('without', '无草稿')]:
            sub = R[R.condition == cond]; ok = (sub[f'final_{c}'].values == key.loc[sub.case_id, f'expert_{c}'].values)
            r[zh] = f'{100 * ok.mean():.1f}%'
        ok = (key.loc[W.case_id, f'draft_{c}'].values == key.loc[W.case_id, f'expert_{c}'].values); r['草稿'] = f'{100 * ok.mean():.1f}%'
        pf.append(r)
    out['3c_逐特征准确率'] = pd.DataFrame(pf)
    outp = a.out or os.path.join(a.folder, '阅读实验_统计.xlsx')
    with pd.ExcelWriter(outp) as xw:
        for k, v in out.items(): v.to_excel(xw, sheet_name=k[:31], index=False)
        R.to_excel(xw, sheet_name='原始记录', index=False)
    for k, v in out.items():
        print('\n==', k); print(v.to_string(index=False))
    print('\n已写入', outp)

if __name__ == '__main__':
    main()
