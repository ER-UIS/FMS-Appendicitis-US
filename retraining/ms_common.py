# -*- coding: utf-8 -*-
"""
多种子重训 —— 公共模块（ms_common.py）
======================================
供 ms_train.py / ms_predict.py 调用；不需要单独运行。

与原始训练（ERTrain.py + ImageTrain.py）保持一致的设置：
  * InceptionResNetV2（本目录上一级 KerasModels/inception_resnet_v2.py），224×224×3，
    ImageNet 预训练权重（include_top=False, pooling='avg'）+ Dense(2, softmax)
  * Adam(lr=1e-3, decay=1e-4)，categorical cross-entropy，batch 16，最多 200 个 epoch
  * 学习率分段：>80 epoch ×0.1，>120 ×0.01，>160 ×0.001，>180 ×0.0001
  * 在线增强：水平翻转、高度/宽度平移 0.1；像素 /255；steps_per_epoch = int(训练图像数 / 16)
  * 训练准确率提高时生成快照（ModelCheckpoint(monitor='accuracy', save_best_only=True) 的规则）
  * 训练准确率达到 100% 时停止（EarlyStoppingAtAccuracy）
  * 病例级规则：该例全部图像 × 两个类别中概率最高的一次预测的类别；
    记录该图像的阳性类别概率（用于 AUC）
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# FMS_ALGO_DIR: folder of the original training scripts (contains KerasModels/, TestImage_all/ and the internal label table);
# FMS_RETRAIN_DATA: data folder of this analysis (train/, selection/, external/, labels/, weights/). See retraining/README.md.
ALGO = os.environ.get('FMS_ALGO_DIR', os.path.dirname(HERE))
if ALGO not in sys.path:
    sys.path.insert(0, ALGO)                                 # 使 KerasModels 可导入（与原脚本相同的模型定义）

DATA = os.environ.get('FMS_RETRAIN_DATA', os.path.join(HERE, 'data'))
RUNS = os.path.join(HERE, 'runs')
RESULTS = os.path.join(HERE, 'results')
CACHE = os.path.join(DATA, 'cache')

IMAGE_SIZE = 224
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
IMAGENET_NOTOP = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGENET_NOTOP_MD5 = 'd19885ff4a710c122648d3b5c3b684e4'

# 编号、代码、中文列名、阳性类别、阴性类别、英文阳性、英文阴性
FEATURES = [
    (1, 'AODC', '阑尾外径', '增粗', '无增粗', 'Thickening', 'No thickening'),
    (2, 'CC', '粪石', '有粪石', '无粪石', 'Presence of fecal stones', 'No fecal stones'),
    (3, 'CAE', '腹腔积液', '有积液', '无积液', 'Presence of fluid accumulation', 'No fluid accumulation'),
    (4, 'AWC', '阑尾壁情况', '阑尾壁异常', '阑尾壁连续', 'Abnormal appendix wall', 'Continuous appendix wall'),
    (5, 'GAC', '阑尾腔内积气', '有积气', '无积气', 'Presence of gas accumulation', 'No gas accumulation'),
    (6, 'FAC', '阑尾腔内积液', '有积液', '无积液', 'Presence of fluid accumulation', 'No fluid accumulation'),
    (7, 'PFCC', '阑尾周边积液', '有积液', '无积液', 'Presence of fluid accumulation', 'No fluid accumulation'),
    (8, 'MSC', '阑尾周围系膜', '有肿胀', '无肿胀', 'Presence of swelling', 'No swelling'),
    (9, 'BFC', '阑尾区血流', '有血流', '无血流', 'Presence of blood flow', 'No blood flow'),
]
CODES = [f[1] for f in FEATURES]
BY_CODE = {f[1]: f for f in FEATURES}

# 队列：图像根目录（其下每例一个以申请号命名的文件夹）与标签文件
COHORTS = {
    'selection': dict(images=os.path.join(DATA, 'selection', 'images'),
                      labels=os.path.join(DATA, 'labels', 'model_selection_184_verified_labels.csv')),
    'internal': dict(images=os.path.join(ALGO, 'TestImage_all'),
                     labels=os.path.join(DATA, 'labels', 'internal_test_3214_report_labels.csv')),
    'external': dict(images=os.path.join(DATA, 'external', 'images'),
                     labels=os.path.join(DATA, 'labels', 'external_544_expert_labels.csv')),
}

STRATEGIES = ['FMO', 'ACC', 'ESO', 'NO', 'ESOval', 'ESOdyn', 'FMOall']
STRATEGY_NOTE = {
    'FMO': '快照池内选择队列综合指数 0.3×准确率+0.4×F1+0.3×AUC 最高（并列取最早）——与原研究相同',
    'ACC': '快照池内选择队列病例级准确率最高（并列取最早）',
    'ESO': '训练准确率连续 10 个 epoch 不提高前的最后一个快照（与原研究 ESO 相同）',
    'NO': '最后一个快照（未优化）',
    'ESOval': '选择队列病例级准确率早停（patience 10），取最佳 epoch（对应原研究 ESO2）',
    'ESOdyn': '同上，但改善下限 0.9（对应原研究 ESO3）',
    'FMOall': '全部 epoch（不限快照）中综合指数最高（敏感性分析）',
}


# ----------------------------------------------------------------------------- 通用
def log(msg, path=None):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + str(msg)
    print(line, flush=True)
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def set_seed(seed, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def gpu_memory_growth():
    try:
        import tensorflow as tf
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def file_md5(path, chunk=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def find_imagenet_weights(arg=None):
    """ImageNet 预训练（无顶层）权重：命令行 > data/weights > %USERPROFILE%/.keras/models > 自动下载。"""
    cands = [arg] if arg else []
    cands += [os.path.join(DATA, 'weights', IMAGENET_NOTOP),
              os.path.join(os.path.expanduser('~'), '.keras', 'models', IMAGENET_NOTOP)]
    wdir = os.path.join(DATA, 'weights')
    if os.path.isdir(wdir):
        cands += [os.path.join(wdir, n) for n in sorted(os.listdir(wdir)) if n.lower().endswith('.h5')]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return 'imagenet'


# ----------------------------------------------------------------------------- 标签
def _norm_id(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s


def read_labels(path):
    """读取标签表 → DataFrame(index=申请号, columns=CODES, 值=True 表示阳性)。
    Accepts the original Excel tables (Chinese or English column names and class names) and the figshare CSV files
    (01_labels: columns label_<F> or expert_<F>, values 0/1)."""
    df = pd.read_csv(path, dtype=str) if str(path).lower().endswith('.csv') else pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    idc = [c for c in df.columns if c in ('申请号', 'App_No', 'app_no', 'ID')]
    if not idc:
        raise KeyError(f'{path} 中没有 申请号/App_No 列')
    df.index = df[idc[0]].map(_norm_id)
    out = pd.DataFrame(index=df.index)
    for _, code, zh, pz, nz, pe, ne in FEATURES:
        col = next((c for c in (zh, code, f'label_{code}', f'expert_{code}') if c in df.columns), None)
        if col is None:
            continue
        v = df[col].astype(str).str.strip()
        bad = set(v) - {pz, nz, pe, ne, '0', '1'}
        if bad:
            raise ValueError(f'{path} 列 {col} 有无法识别的值：{sorted(bad)[:5]}')
        out[code] = v.isin([pz, pe, '1'])
    out = out[~out.index.duplicated()]
    return out


def positive_index(class_json, code):
    """model_class.json（{"0": "增粗", "1": "无增粗"}）中阳性类别的下标。"""
    _, _, _, pz, nz, pe, ne = BY_CODE[code]
    idx = [int(k) for k, v in class_json.items() if v in (pz, pe)]
    if len(idx) != 1 or len(class_json) != 2:
        raise ValueError(f'{code}: 无法从类别 {class_json} 中确定阳性类别（应为 {pz}/{nz}）')
    return idx[0]


# ----------------------------------------------------------------------------- 图像缓存
def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXT))


def load_image_uint8(path, size=IMAGE_SIZE):
    """与原脚本相同：keras load_img(target_size=(224,224))（默认最近邻插值、RGB）。"""
    try:
        from keras.preprocessing.image import load_img
    except Exception:
        from keras.utils import load_img
    img = load_img(path, target_size=(size, size))
    return np.asarray(img, dtype=np.uint8)


def cohort_cache(name, ids=None, image_root=None, size=IMAGE_SIZE, logfile=None):
    """把一个队列的全部图像缓存为 uint8 数组（data/cache/<name>_<size>.npy，内存映射读取）。

    返回 (X, pids, slices, names)：X[n_img, size, size, 3] uint8；slices[i] = (start, end) 对应 pids[i]。
    """
    os.makedirs(CACHE, exist_ok=True)
    npy = os.path.join(CACHE, f'{name}_{size}.npy')
    meta = os.path.join(CACHE, f'{name}_{size}.json')
    image_root = image_root or COHORTS[name]['images']
    if os.path.isfile(npy) and os.path.isfile(meta):
        with open(meta, encoding='utf-8') as f:
            m = json.load(f)
        if ids is None or set(map(str, ids)) <= set(m['pids']) | set(m.get('missing', [])):
            X = np.load(npy, mmap_mode='r')
            return X, m['pids'], [tuple(s) for s in m['slices']], m['names']
        log(f'缓存 {npy} 与所需病例不一致，重新生成', logfile)
    if ids is None:
        ids = sorted(d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d)))
    ids = [str(i) for i in ids]
    plan, missing = [], []
    for pid in ids:
        names = list_images(os.path.join(image_root, pid))
        if names:
            plan.append((pid, names))
        else:
            missing.append(pid)
    n_img = sum(len(n) for _, n in plan)
    log(f'建立图像缓存 {name}: {len(plan)} 例 / {n_img} 张图像（无图像 {len(missing)} 例）', logfile)
    X = np.lib.format.open_memmap(npy + '.tmp', mode='w+', dtype=np.uint8, shape=(n_img, size, size, 3))
    pids, slices, allnames, k = [], [], [], 0
    t0 = time.time()
    for i, (pid, names) in enumerate(plan):
        s = k
        for n in names:
            X[k] = load_image_uint8(os.path.join(image_root, pid, n), size)
            k += 1
        pids.append(pid); slices.append((s, k)); allnames.append(names)
        if (i + 1) % 500 == 0:
            log(f'  {i + 1}/{len(plan)} 例，{time.time() - t0:.0f} 秒', logfile)
    X.flush(); del X
    os.replace(npy + '.tmp', npy)
    with open(meta, 'w', encoding='utf-8') as f:
        json.dump({'pids': pids, 'slices': slices, 'names': allnames, 'missing': missing,
                   'image_root': image_root, 'size': size}, f, ensure_ascii=False)
    if missing:
        log(f'  注意：以下病例没有图像，已跳过（前 10 个）：{missing[:10]}', logfile)
    return np.load(npy, mmap_mode='r'), pids, slices, allnames


def predict_images(model, X, batch=64):
    """X uint8（可为内存映射）→ 每张图像的类别概率（像素 /255，与原脚本相同）。"""
    out = []
    for s in range(0, len(X), batch):
        xb = np.asarray(X[s:s + batch], dtype=np.float32) / 255.0
        out.append(model.predict_on_batch(xb))
    out = [o.numpy() if hasattr(o, 'numpy') else np.asarray(o) for o in out]
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2), np.float32)


def patient_level(probs, slices, pos_idx):
    """病例级规则（原脚本）：全部图像×类别中概率最高的一次预测 → 类别；返回 (pred_pos[bool], p_pos, 图像下标)。"""
    pred, ppos, best = [], [], []
    for s, e in slices:
        P = probs[s:e]
        i, j = np.unravel_index(int(np.argmax(P)), P.shape)
        pred.append(j == pos_idx); ppos.append(float(P[i, pos_idx])); best.append(int(i))
    return np.array(pred, bool), np.array(ppos, float), best


# ----------------------------------------------------------------------------- 指标
def binary_metrics(y, pred, p):
    """与 FMS v2.0.1 fms/stats.py binary_metrics 相同。"""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool); pred = np.asarray(pred, bool)
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())
    n = len(y)
    acc = (tp + tn) / n if n else float('nan')
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < n else float('nan')
    comp = 0.3 * acc + 0.4 * f1 + 0.3 * (auc if auc == auc else 0.0)
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc,
                composite=comp, specificity=tn / (tn + fp) if tn + fp else float('nan'))


# ----------------------------------------------------------------------------- 模型
def build_model(num_classes=2, weights=None, size=IMAGE_SIZE):
    """与原训练相同：KerasModels.inception_resnet_v2（include_top=False, pooling='avg'）+ Dense(softmax)。"""
    import keras
    try:                                                     # original model definition (vendored KerasModels)
        from KerasModels.inception_resnet_v2 import InceptionResNetV2
    except ImportError:                                      # identical architecture from Keras
        from keras.applications import InceptionResNetV2
    base = InceptionResNetV2(input_shape=(size, size, 3), weights=weights, include_top=False, pooling='avg')
    out = keras.layers.Dense(num_classes, activation='softmax', use_bias=True)(base.output)
    return keras.models.Model(inputs=base.input, outputs=out)


def make_optimizer(lr=1e-3, decay=1e-4):
    """Adam(lr=1e-3, decay=1e-4)，兼容 Keras 2.3–2.15。"""
    import keras
    for factory in (lambda: keras.optimizers.legacy.Adam(learning_rate=lr, decay=decay),
                    lambda: keras.optimizers.Adam(learning_rate=lr, decay=decay),
                    lambda: keras.optimizers.Adam(lr=lr, decay=decay)):
        try:
            return factory()
        except Exception:
            continue
    raise RuntimeError('无法创建 Adam(decay=1e-4) 优化器；请使用 TensorFlow/Keras 2.x（< 2.16）')


def lr_schedule(epoch, lr=None):
    """原 ERTrain.py 的学习率表（epoch 从 0 开始计）。"""
    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


# ----------------------------------------------------------------------------- 在线检查点选择
class SelectionTracker:
    """逐 epoch 更新各策略所选的 epoch；与 FMS v2.0.1 fms/select.py 的离线定义等价。

    每个 epoch 调用 update(ep, train_acc, m)（m = 选择队列病例级指标），返回本 epoch 是否需要保存权重。
    referenced() 给出仍被引用的 epoch，其余权重文件可删除。
    """

    def __init__(self, patience=10, dyn_floor=0.9):
        self.patience = patience
        self.dyn_floor = dyn_floor
        self.best_train = -np.inf
        self.snapshots = []
        self.pick = {s: None for s in STRATEGIES}
        self.score = {'FMO': -np.inf, 'ACC': -np.inf, 'FMOall': -np.inf, 'ESOval': -np.inf, 'ESOdyn': self.dyn_floor}
        self.eso_done = False
        self.val_wait = 0; self.val_done = False
        self.dyn_wait = 0; self.dyn_done = False; self.dyn_seen = False
        self.dyn_any = (-np.inf, None)

    def update(self, ep, train_acc, m):
        snap = train_acc > self.best_train
        if snap:
            self.best_train = train_acc
            self.snapshots.append(ep)
            self.pick['NO'] = ep
        comp, acc = m['composite'], m['accuracy']
        if snap and comp > self.score['FMO']:
            self.score['FMO'] = comp; self.pick['FMO'] = ep
        if snap and acc > self.score['ACC']:
            self.score['ACC'] = acc; self.pick['ACC'] = ep
        if comp > self.score['FMOall']:
            self.score['FMOall'] = comp; self.pick['FMOall'] = ep
        # ESO：训练准确率连续 patience 个 epoch 未提高 → 取此前最后一个快照
        if not self.eso_done and self.snapshots and ep - self.snapshots[-1] >= self.patience:
            self.pick['ESO'] = self.snapshots[-1]; self.eso_done = True
        # ESOval：选择队列准确率早停
        if not self.val_done:
            if acc > self.score['ESOval']:
                self.score['ESOval'] = acc; self.pick['ESOval'] = ep; self.val_wait = 0
            else:
                self.val_wait += 1
                if self.val_wait >= self.patience:
                    self.val_done = True
        # ESOdyn：改善下限 0.9
        if not self.dyn_done:
            if acc > self.dyn_any[0]:
                self.dyn_any = (acc, ep)
            if acc > self.score['ESOdyn']:
                self.score['ESOdyn'] = acc; self.pick['ESOdyn'] = ep; self.dyn_wait = 0; self.dyn_seen = True
            else:
                self.dyn_wait += 1
                if self.dyn_wait >= self.patience:
                    self.dyn_done = True
            if not self.dyn_seen:
                self.pick['ESOdyn'] = self.dyn_any[1]
        return ep in self.referenced(include_last=True)

    def referenced(self, include_last=True):
        r = {v for k, v in self.pick.items() if v is not None and (k != 'ESO' or self.eso_done)}
        if include_last and self.snapshots and not self.eso_done:
            r.add(self.snapshots[-1])                      # ESO 候选（尚未触发）
        return r

    def finalize(self):
        if not self.eso_done and self.snapshots:
            self.pick['ESO'] = self.snapshots[-1]; self.eso_done = True
        return dict(self.pick)
