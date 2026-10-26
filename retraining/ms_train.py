# -*- coding: utf-8 -*-
"""
多种子重训 —— 第 1 步：训练（ms_train.py）
==========================================
对 9 个特征、每个随机种子各训练一次，设置与原始训练完全相同（见 ms_common.py 顶部说明），并且：
  * 每个 epoch 结束时在 184 例选择队列上做病例级评估（准确率、精确率、召回率、F1、AUC、综合指数），
    写入 runs/seed<k>/<CODE>/epochs.csv；每例的阳性概率与预测写入 sel_ppos.csv / sel_pred.csv
    （以后可离线做 bootstrap 再选择、权重敏感性分析，不需要重新训练）
  * 在线确定 FMO / ACC / ESO / NO / ESOval / ESOdyn / FMOall 各自选中的 epoch，只保留这些 epoch 的权重
    （runs/seed<k>/<CODE>/weights/ex-<epoch>.h5），其余随时删除，每个特征约 0.6–1.2 GB
  * 结束后写 selected.json（各策略的 epoch 及其选择队列指标）和 done.json；
    已完成的（存在 done.json）自动跳过，中断后重新运行即可续跑（未完成的特征从头重训）

用法（在 Anaconda Prompt 中，先 cd 到本文件夹）：
    python ms_train.py                         # 默认种子 1 2 3 4 5，全部 9 个特征
    python ms_train.py --seeds 1               # 先跑 1 个种子试试
    python ms_train.py --seeds 1 --features AODC --epochs 3      # 冒烟测试（约 10–20 分钟）
    python ms_train.py --seeds 6 7 8 9 10      # 追加种子
    python ms_train.py --check-only            # 只检查数据是否放对
可选：--deterministic（启用确定性 GPU 运算，更可复现但更慢）、--weights <ImageNet 无顶层权重 .h5>
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

from ms_common import (BY_CODE, CODES, COHORTS, DATA, IMAGE_SIZE, IMAGENET_NOTOP_MD5, RUNS, SelectionTracker,
                       build_model, cohort_cache, file_md5, find_imagenet_weights, gpu_memory_growth, log, lr_schedule,
                       make_optimizer, patient_level, positive_index, predict_images, read_labels, binary_metrics, set_seed)


def check_inputs(features, logfile):
    ok = True
    for code in features:
        d = os.path.join(DATA, 'train', code)
        _, _, zh, pz, nz, pe, ne = BY_CODE[code]
        subs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))) if os.path.isdir(d) else []
        if sorted(subs) not in (sorted([pz, nz]), sorted([pe, ne])):   # Chinese (original) or English (figshare) class folders
            log(f'[缺失/不符] {d} 下应有且只有两个子文件夹：{pz}、{nz}（或 {pe}、{ne}）；现在是 {subs}', logfile); ok = False
        else:
            n = {s: len(os.listdir(os.path.join(d, s))) for s in subs}
            log(f'训练集 {code}（{zh}）：{n}', logfile)
    sel = COHORTS['selection']
    if not os.path.isfile(sel['labels']):
        log(f'[缺失] 选择队列标签 {sel["labels"]}', logfile); ok = False
    if not os.path.isdir(sel['images']):
        log(f'[缺失] 选择队列图像文件夹 {sel["images"]}', logfile); ok = False
    return ok


class EpochSelection:
    """keras 回调：每个 epoch 评估选择队列、在线选择检查点、管理权重文件。"""

    def __new__(cls, *a, **k):
        import keras

        class _CB(keras.callbacks.Callback):
            def __init__(self, code, outdir, X, slices, y, pids, pos_idx, eval_batch, logfile):
                super().__init__()
                self.code, self.outdir, self.X, self.slices, self.y = code, outdir, X, slices, y
                self.pids, self.pos_idx, self.eval_batch, self.logfile = pids, pos_idx, eval_batch, logfile
                self.tr = SelectionTracker()
                self.wdir = os.path.join(outdir, 'weights'); os.makedirs(self.wdir, exist_ok=True)
                self.metrics = {}
                self.t0 = time.time()
                self.csv = os.path.join(outdir, 'epochs.csv')
                with open(self.csv, 'w', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow(['epoch', 'lr', 'loss', 'train_accuracy', 'snapshot', 'sel_accuracy', 'sel_precision',
                                            'sel_recall', 'sel_f1', 'sel_auc', 'sel_composite', 'sel_specificity',
                                            'tp', 'fp', 'fn', 'tn', 'seconds'])
                for name in ('sel_ppos.csv', 'sel_pred.csv'):
                    with open(os.path.join(outdir, name), 'w', newline='', encoding='utf-8-sig') as f:
                        csv.writer(f).writerow(['epoch'] + list(pids))

            def _wpath(self, ep):
                return os.path.join(self.wdir, f'ex-{ep:03d}.h5')

            def on_epoch_end(self, epoch, logs=None):
                import keras
                logs = logs or {}
                ep = epoch + 1
                acc = logs.get('accuracy', logs.get('acc'))
                probs = predict_images(self.model, self.X, self.eval_batch)
                pred, ppos, _ = patient_level(probs, self.slices, self.pos_idx)
                m = binary_metrics(self.y, pred, ppos)
                self.metrics[ep] = dict(m, train_accuracy=float(acc), loss=float(logs.get('loss', np.nan)))
                save = self.tr.update(ep, float(acc), m)
                if save:
                    self.model.save_weights(self._wpath(ep))
                keep = self.tr.referenced(include_last=True)
                for p in glob.glob(os.path.join(self.wdir, 'ex-*.h5')):
                    e = int(os.path.basename(p)[3:6])
                    if e not in keep:
                        os.remove(p)
                try:
                    lr = float(keras.backend.get_value(self.model.optimizer.lr))
                except Exception:
                    lr = float('nan')
                with open(self.csv, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([ep, lr, logs.get('loss'), acc, int(ep in self.tr.snapshots), m['accuracy'],
                                            m['precision'], m['recall'], m['f1'], m['auc'], m['composite'], m['specificity'],
                                            m['tp'], m['fp'], m['fn'], m['tn'], round(time.time() - self.t0, 1)])
                with open(os.path.join(self.outdir, 'sel_ppos.csv'), 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([ep] + [f'{v:.6f}' for v in ppos])
                with open(os.path.join(self.outdir, 'sel_pred.csv'), 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([ep] + [int(v) for v in pred])
                pk = {k: v for k, v in self.tr.pick.items() if v is not None}
                log(f'{self.code} epoch {ep}: 训练准确率 {acc:.4f}{"（快照）" if ep in self.tr.snapshots else ""}；'
                    f'选择队列 准确率 {m["accuracy"]:.3f} F1 {m["f1"]:.3f} AUC {m["auc"]:.3f} 综合 {m["composite"]:.4f}；当前选择 {pk}',
                    self.logfile)
                if acc is not None and acc >= 1.0:          # 原 EarlyStoppingAtAccuracy(target_accuracy=1.0)
                    log(f'{self.code}: 训练准确率达到 100%，停止训练（与原训练相同）', self.logfile)
                    self.model.stop_training = True

        return _CB(*a, **k)


def train_one(seed, code, args, sel, logfile):
    import keras
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except Exception:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

    outdir = os.path.join(RUNS, f'seed{seed}', code)
    if os.path.isfile(os.path.join(outdir, 'done.json')):
        log(f'种子 {seed} {code} 已完成，跳过', logfile); return
    if os.path.isdir(outdir):
        log(f'种子 {seed} {code} 有未完成的记录，删除后从头训练', logfile)
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    keras.backend.clear_session()
    set_seed(seed, args.deterministic)

    gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, height_shift_range=0.1, width_shift_range=0.1)
    train = gen.flow_from_directory(os.path.join(DATA, 'train', code), target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    batch_size=args.batch, class_mode='categorical', shuffle=True, seed=seed)
    classes = {str(v): k for k, v in train.class_indices.items()}
    with open(os.path.join(outdir, 'model_class.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=4, separators=(',', ' : '), ensure_ascii=True)
    pos_idx = positive_index(classes, code)

    X, pids, slices, _ = sel['cache']
    lab = sel['labels']
    keep = [i for i, p in enumerate(pids) if p in lab.index and code in lab.columns]
    if len(keep) != len(pids):
        log(f'  注意：选择队列中 {len(pids) - len(keep)} 例无 {code} 标签，评估时排除', logfile)
    slices_k = [slices[i] for i in keep]; pids_k = [pids[i] for i in keep]
    y = lab.loc[pids_k, code].values.astype(bool)

    model = build_model(len(classes), weights=args.weights_path)
    model.compile(loss='categorical_crossentropy', optimizer=make_optimizer(1e-3, 1e-4), metrics=['accuracy'])
    cb = EpochSelection(code, outdir, X, slices_k, y, pids_k, pos_idx, args.eval_batch, logfile)
    steps = int(train.samples / args.batch)
    log(f'开始训练 种子 {seed} {code}：训练图像 {train.samples}，类别 {classes}，steps_per_epoch {steps}', logfile)
    t0 = time.time()
    fit = getattr(model, 'fit')
    fit(train, steps_per_epoch=steps, epochs=args.epochs,
        callbacks=[cb, keras.callbacks.LearningRateScheduler(lr_schedule)], verbose=2)
    picks = cb.tr.finalize()
    # ESO 若在训练结束时才确定，其权重即最后一个快照，已保留
    keep = {v for v in picks.values() if v is not None}
    for p in glob.glob(os.path.join(cb.wdir, 'ex-*.h5')):
        if int(os.path.basename(p)[3:6]) not in keep:
            os.remove(p)
    missing = [e for e in keep if not os.path.isfile(cb._wpath(e))]
    if missing:
        raise RuntimeError(f'选中的 epoch {missing} 没有权重文件（不应发生）')
    sel_json = {s: dict(epoch=e, **{k: cb.metrics[e][k] for k in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'composite',
                                                                   'specificity', 'train_accuracy')})
                for s, e in picks.items() if e is not None}
    with open(os.path.join(outdir, 'selected.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(seed=seed, feature=code, selections=sel_json, snapshots=cb.tr.snapshots), f, indent=2, ensure_ascii=False)
    import tensorflow as tf
    with open(os.path.join(outdir, 'done.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(seed=seed, feature=code, epochs_run=max(cb.metrics), n_train=train.samples, classes=classes,
                       batch_size=args.batch, max_epochs=args.epochs, deterministic=args.deterministic,
                       minutes=round((time.time() - t0) / 60, 1), tensorflow=tf.__version__, keras=keras.__version__,
                       imagenet_weights=os.path.basename(str(args.weights_path)), selection_n=len(pids_k),
                       finished=time.strftime('%Y-%m-%d %H:%M:%S')), f, indent=2, ensure_ascii=False)
    log(f'完成 种子 {seed} {code}：{(time.time() - t0) / 60:.1f} 分钟；选择 {picks}', logfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--features', nargs='+', default=CODES, choices=CODES)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--weights', default=None, help='ImageNet 无顶层权重 .h5（默认自动查找）')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--check-only', action='store_true', help='只检查数据放置是否正确并建立选择队列图像缓存，不训练')
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    logfile = os.path.join(RUNS, 'train_log.txt')
    gpu_memory_growth()
    if not check_inputs(args.features, logfile):
        log('输入数据不完整，请按 README_多种子重训.md 放置数据后再运行', logfile); sys.exit(1)
    args.weights_path = find_imagenet_weights(args.weights)
    if args.weights_path != 'imagenet':
        md5 = file_md5(args.weights_path)
        log(f'ImageNet 权重：{args.weights_path}（md5 {md5}{"，与 Keras 官方一致" if md5 == IMAGENET_NOTOP_MD5 else "，注意：与 Keras 官方无顶层文件不同，请确认是原训练所用文件"}）', logfile)
    else:
        log('未找到本地 ImageNet 权重，将由 Keras 自动下载', logfile)
    import tensorflow as tf
    log(f'TensorFlow {tf.__version__}；GPU：{tf.config.list_physical_devices("GPU")}', logfile)
    lab = read_labels(COHORTS['selection']['labels'])
    cache = cohort_cache('selection', ids=list(lab.index), logfile=logfile)
    log(f'选择队列：标签 {len(lab)} 例，有图像 {len(cache[1])} 例，图像 {len(cache[0])} 张', logfile)
    sel = dict(labels=lab, cache=cache)
    if args.check_only:
        log('数据检查通过（--check-only），未开始训练', logfile); return
    for seed in args.seeds:
        for code in args.features:
            train_one(seed, code, args, sel, logfile)
    log('全部训练完成。下一步：python ms_predict.py', logfile)


if __name__ == '__main__':
    main()
