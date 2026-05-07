"""
Ablation: LedoitWolf vs EmpiricalCovariance for Maha++
-------------------------------------------------------
Maha++ (EC)  — full ImageNet-1K (1.28M), EmpiricalCovariance  [baseline]
Maha++ (LW)  — ImageNet-LT-scale subsample (115K), LedoitWolf  [ablation]

Both use the cached norm-layer features from the standard ViT-B/16
(vit_base_patch16_224.augreg2_in21k_ft_in1k), no fine-tuning.

Run:
    conda run -n NINCO_maha python ablation_lw_maha.py
"""
import os, glob, gc
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
CACHE     = os.path.join(BASE, 'cache_imagenet_full')
MODEL     = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
TRAIN_INT = os.path.join(CACHE, 'cache_train_inter', MODEL)
TRAIN_LBL = os.path.join(CACHE, 'cache_train',       MODEL)
VAL_INT   = os.path.join(CACHE, 'cache_val_inter',   MODEL)
OOD_INT   = os.path.join(CACHE, 'cache_ood_inter',   MODEL)

N_CLASSES     = 1000
N_IMAGENETLT  = 115_846   # ImageNet-LT training set size (matched subsample)
OOD_DATASETS  = ['inaturalist', 'sun', 'places365', 'texture',
                 'imagenet_o', 'places365_old_36500']


# ── Helpers ────────────────────────────────────────────────────────────────

def load_shards(pattern, dtype=np.float32):
    shards = sorted(glob.glob(pattern),
                    key=lambda p: int(p.split('_')[-1].replace('.npy', '')))
    return np.concatenate([np.load(s).astype(dtype) for s in shards], axis=0)


def l2_norm(f):
    return f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-10)


def fast_maha(feats, means, prec):
    """Mahalanobis score (negative min dist). feats:[N,D] means:[C,D] prec:[D,D]"""
    Pmu    = means @ prec
    mu_Pmu = (means * Pmu).sum(axis=1)
    xP     = feats @ prec
    xPx    = (feats * xP).sum(axis=1)
    return -np.min(xPx[:, None] - 2 * (xP @ means.T) + mu_Pmu[None, :], axis=1)


def fit_and_score(train_feats, train_labels, val_feats, ood_feats_dict, tag):
    print(f'\n[{tag}] n_train={len(train_feats):,}  fitting...')
    train_feats = l2_norm(train_feats.astype(np.float64))

    # class-conditional means
    means    = np.empty((N_CLASSES, train_feats.shape[1]), dtype=np.float64)
    centered = np.empty_like(train_feats)
    row = 0
    for c in range(N_CLASSES):
        fs = train_feats[train_labels == c]
        m  = fs.mean(axis=0) if len(fs) > 0 else np.zeros(train_feats.shape[1])
        means[c] = m
        if len(fs) > 0:
            centered[row:row + len(fs)] = fs - m
            row += len(fs)

    if 'LW' in tag:
        est = LedoitWolf(assume_centered=True)
    else:
        est = EmpiricalCovariance(assume_centered=True)
    est.fit(centered[:row])
    prec = est.precision_
    print(f'[{tag}] fit done.')

    val_f  = l2_norm(val_feats.astype(np.float64))
    scores_id = fast_maha(val_f, means, prec)

    print(f'[{tag}] AUROC:')
    for ds_name, ood_f in ood_feats_dict.items():
        ood_f_n = l2_norm(ood_f.astype(np.float64))
        scores_ood = fast_maha(ood_f_n, means, prec)
        labels_bin = np.concatenate([np.ones(len(scores_id)),
                                     np.zeros(len(scores_ood))])
        scores_all = np.concatenate([scores_id, scores_ood])
        auroc = roc_auc_score(labels_bin, scores_all) * 100
        print(f'  {ds_name:<15} {auroc:.2f}%')


# ── Load data ──────────────────────────────────────────────────────────────

print('Loading full ImageNet norm features (train) ...')
train_feats  = load_shards(os.path.join(TRAIN_INT, 'layer_norm_features_*.npy'))
train_labels = load_shards(os.path.join(TRAIN_LBL, 'labels_true_*.npy'), dtype=np.int64)
print(f'  loaded {train_feats.shape}')

print('Loading val norm features ...')
val_feats = np.load(os.path.join(VAL_INT, 'layer_norm_features_0.npy'))
print(f'  loaded {val_feats.shape}')

print('Loading OOD norm features ...')
ood_feats = {}
for ds in OOD_DATASETS:
    ood_feats[ds] = np.load(os.path.join(OOD_INT, ds, 'layer_norm_features_0.npy'))
    print(f'  {ds}: {ood_feats[ds].shape}')

# ── Baseline: full ImageNet, EmpiricalCovariance ───────────────────────────
fit_and_score(train_feats, train_labels, val_feats, ood_feats,
              tag='Maha++ (EC, full 1.28M)')

# ── Ablation: ImageNet-LT-scale subsample, LedoitWolf ─────────────────────
rng = np.random.default_rng(42)
idx = rng.choice(len(train_feats), N_IMAGENETLT, replace=False)
fit_and_score(train_feats[idx], train_labels[idx], val_feats, ood_feats,
              tag=f'Maha++ (LW, {N_IMAGENETLT//1000}K subsample)')

# ── Extra: same 115K subsample with EC (isolates estimator effect) ─────────
fit_and_score(train_feats[idx], train_labels[idx], val_feats, ood_feats,
              tag=f'Maha++ (EC, {N_IMAGENETLT//1000}K subsample)')

# ── Ablation: full 1.28M, LedoitWolf ──────────────────────────────────────
fit_and_score(train_feats, train_labels, val_feats, ood_feats,
              tag='Maha++ (LW, full 1.28M)')
