# -*- coding: utf-8 -*-
"""
多种子重训 —— 第 2 步：预测（ms_predict.py）
============================================
对每个种子、每个特征、每种选择策略选中的检查点（runs/seed<k>/<CODE>/selected.json），在
内部测试队列（3214 例，TestImage_all）和外部队列（544 例）上做病例级预测（规则与原脚本相同），
结果写入 results/seed<k>/<队列>_<CODE>.csv（每行一例：申请号、各 epoch 的阳性概率与预测）。
同一 epoch 被多种策略选中时只预测一次。已存在的结果文件自动跳过（可续跑）。

最后把所有小文件（训练日志、epochs.csv、selected.json、选择队列逐例概率、预测结果）打包为
    多种子结果_上传.zip
用于汇总分析（ms_aggregate.py）。不包含图像和权重，体积约几十 MB。

用法：
    python ms_predict.py                      # 全部已完成训练的种子，内部 + 外部队列
    python ms_predict.py --original           # 另外用已部署的 9 个 FMO 检查点（model_appendix\\models_k）
                                              # 跑一遍，用于核对本流程能复现已发表的 FMO 结果
    python ms_predict.py --seeds 1 --cohorts internal
    python ms_predict.py --cleanup            # 预测完成后删除除 FMO 以外的权重（节省硬盘）
    python ms_predict.py --pack-only          # 只打包
"""
import argparse
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from ms_common import (ALGO, CODES, COHORTS, HERE, RESULTS, RUNS, build_model, cohort_cache, gpu_memory_growth, log,
                       patient_level, positive_index, predict_images, read_labels)


def predict_epochs(code, weight_files, class_json, cache, batch, logfile):
    """weight_files: {epoch_tag: path} → DataFrame(申请号, <tag>_p, <tag>_pred)。"""
    import keras
    X, pids, slices, names = cache
    pos_idx = positive_index(class_json, code)
    out = pd.DataFrame({'申请号': pids})
    for tag, path in weight_files.items():
        keras.backend.clear_session()
        model = build_model(len(class_json), weights=None)
        model.load_weights(path)
        t0 = time.time()
        probs = predict_images(model, X, batch)
        pred, ppos, best = patient_level(probs, slices, pos_idx)
        out[f'{tag}_p'] = np.round(ppos, 6)
        out[f'{tag}_pred'] = pred.astype(int)
        out[f'{tag}_image'] = [names[i][b] for i, b in enumerate(best)]
        log(f'  {code} {tag}: {len(pids)} 例 / {len(X)} 张，{time.time() - t0:.0f} 秒；阳性 {pred.mean() * 100:.1f}%', logfile)
    return out


def run_seed(seed, codes, cohorts, caches, args, logfile):
    sdir = os.path.join(RUNS, f'seed{seed}')
    odir = os.path.join(RESULTS, f'seed{seed}')
    os.makedirs(odir, exist_ok=True)
    for code in codes:
        rdir = os.path.join(sdir, code)
        if not os.path.isfile(os.path.join(rdir, 'done.json')):
            log(f'种子 {seed} {code} 尚未训练完成，跳过', logfile); continue
        with open(os.path.join(rdir, 'selected.json'), encoding='utf-8') as f:
            sel = json.load(f)['selections']
        with open(os.path.join(rdir, 'model_class.json'), encoding='utf-8') as f:
            classes = json.load(f)
        epochs = sorted({v['epoch'] for v in sel.values()})
        wf = {f'ex{e:03d}': os.path.join(rdir, 'weights', f'ex-{e:03d}.h5') for e in epochs}
        for coh in cohorts:
            dst = os.path.join(odir, f'{coh}_{code}.csv')
            if os.path.isfile(dst):
                continue
            miss = [p for p in wf.values() if not os.path.isfile(p)]
            if miss:
                log(f'[错误] 种子 {seed} {code} 缺少权重 {miss}（可能已被 --cleanup 删除），跳过 {coh}', logfile); continue
            log(f'预测 种子 {seed} {code} {coh}：epoch {epochs}', logfile)
            df = predict_epochs(code, wf, classes, caches[coh], args.batch, logfile)
            df.to_csv(dst + '.tmp', index=False, encoding='utf-8-sig')
            os.replace(dst + '.tmp', dst)
        if args.cleanup and all(os.path.isfile(os.path.join(odir, f'{c}_{code}.csv')) for c in cohorts):
            keep = sel['FMO']['epoch']
            for e, p in zip(epochs, wf.values()):
                if e != keep and os.path.isfile(p):
                    os.remove(p)


def run_original(codes, cohorts, caches, args, logfile):
    """已部署 FMO 检查点（model_appendix\\models_k + json_k）。"""
    odir = os.path.join(RESULTS, 'original')
    os.makedirs(odir, exist_ok=True)
    for k, code in enumerate(CODES, start=1):
        if code not in codes:
            continue
        mdir = os.path.join(ALGO, 'model_appendix', f'models_{k}')
        jf = os.path.join(ALGO, 'model_appendix', f'json_{k}', 'model_class.json')
        h5 = sorted(glob.glob(os.path.join(mdir, '*.h5')))
        if len(h5) != 1 or not os.path.isfile(jf):
            log(f'[跳过] {mdir} 中应有 1 个 .h5（现有 {len(h5)} 个）', logfile); continue
        with open(jf, encoding='utf-8') as f:
            classes = json.load(f)
        ep = int(os.path.basename(h5[0]).split('_ex-')[1][:3])
        for coh in cohorts:
            dst = os.path.join(odir, f'{coh}_{code}.csv')
            if os.path.isfile(dst):
                continue
            log(f'预测 已部署 FMO {code} epoch {ep} {coh}', logfile)
            df = predict_epochs(code, {f'ex{ep:03d}': h5[0]}, classes, caches[coh], args.batch, logfile)
            df.to_csv(dst, index=False, encoding='utf-8-sig')


def pack(logfile):
    zpath = os.path.join(HERE, '多种子结果_上传.zip')
    n = 0
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, base in ((RUNS, 'runs'), (RESULTS, 'results')):
            for dp, _, fs in os.walk(root):
                for fn in fs:
                    if fn.endswith(('.csv', '.json', '.txt')):
                        p = os.path.join(dp, fn)
                        z.write(p, os.path.join(base, os.path.relpath(p, root)))
                        n += 1
        for fn in os.listdir(os.path.join(HERE, 'data', 'cache')) if os.path.isdir(os.path.join(HERE, 'data', 'cache')) else []:
            if fn.endswith('.json'):
                z.write(os.path.join(HERE, 'data', 'cache', fn), os.path.join('cache', fn))
    log(f'已打包 {n} 个文件 → {zpath}（{os.path.getsize(zpath) / 1e6:.1f} MB）', logfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='*', default=None, help='默认：runs 下全部种子')
    ap.add_argument('--features', nargs='+', default=CODES, choices=CODES)
    ap.add_argument('--cohorts', nargs='+', default=['internal', 'external'], choices=['internal', 'external', 'selection'])
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--original', action='store_true')
    ap.add_argument('--cleanup', action='store_true')
    ap.add_argument('--pack-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    logfile = os.path.join(RESULTS, 'predict_log.txt')
    if args.pack_only:
        pack(logfile); return
    gpu_memory_growth()
    caches = {}
    for coh in args.cohorts:
        c = COHORTS[coh]
        if not os.path.isdir(c['images']):
            log(f'[缺失] {coh} 图像文件夹 {c["images"]}', logfile); sys.exit(1)
        ids = list(read_labels(c['labels']).index) if os.path.isfile(c['labels']) else None
        caches[coh] = cohort_cache(coh, ids=ids, logfile=logfile)
        log(f'{coh}: {len(caches[coh][1])} 例，{len(caches[coh][0])} 张图像', logfile)
    if args.original:
        run_original(args.features, args.cohorts, caches, args, logfile)
    seeds = args.seeds
    if seeds is None:
        seeds = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(RUNS, 'seed*')) if os.path.isdir(d))
    for s in seeds:
        run_seed(s, args.features, args.cohorts, caches, args, logfile)
    pack(logfile)


if __name__ == '__main__':
    main()
