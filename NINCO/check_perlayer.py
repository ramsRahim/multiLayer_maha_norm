import sys
sys.path.insert(0, '.')
import numpy as np
from sklearn.covariance import LedoitWolf
from utils import auroc_ood
import glob, json

path = 'cache/cache_methods/resnet50.tv_in1k'
ood_inter = 'cache/cache_ood_inter/resnet50.tv_in1k/NINCO_OOD_classes'
train_inter = 'cache/cache_train_inter/resnet50.tv_in1k'
val_inter = 'cache/cache_val_inter/resnet50.tv_in1k'
layer_dims = [('layer1',256),('layer2',512),('layer3',1024),('layer4',2048)]

train_labels = np.concatenate([np.load(f) for f in sorted(glob.glob('cache/cache_train/resnet50.tv_in1k/labels_true_*.npy'))])

# Load OOD class structure from a reference npz
ref = np.load('cache/scores/resnet50.tv_in1k/NINCO_OOD_classes/E2026-03-26-01-43-02.npz', allow_pickle=True)
ref_res = ref['methods_results'].item()['MM_plus_plus_cat2']
ood_classes = [k for k in ref_res.keys() if k not in
               {'scores_id','scores_ood','samples_mean_auroc','samples_mean_fpr_at_95',
                'ood_classes_mean_auroc','ood_classes_mean_fpr_at_95','bootstrap_ci'}]

# Load OOD labels
ood_labels_true = np.concatenate([np.load(f) for f in sorted(glob.glob('cache/cache_ood/resnet50.tv_in1k/NINCO_OOD_classes/labels_true_*.npy'))])

def per_class_auroc(id_s, ood_s, ood_labels):
    classes = np.unique(ood_labels)
    aurocs = []
    for c in classes:
        mask = ood_labels == c
        if mask.sum() > 0:
            aurocs.append(auroc_ood(id_s, ood_s[mask]))
    return float(np.mean(aurocs))

def l2n(x): return x / np.linalg.norm(x,axis=-1,keepdims=True).clip(min=1e-10)
def fast_maha(feats, means, prec):
    Pmu = means @ prec; mu_Pmu = np.sum(means*Pmu,axis=1)
    xP = feats @ prec; xPx = np.sum(feats*xP,axis=1)
    return -np.min(xPx[:,None] - 2*(xP@means.T) + mu_Pmu[None,:], axis=1)
def fit_score(tr, tl, va, od):
    means = np.zeros((1000,tr.shape[1]))
    for c in range(1000):
        mask = tl==c
        if mask.sum(): means[c]=tr[mask].mean(0)
    lw = LedoitWolf(assume_centered=True); lw.fit(tr-means[tl])
    prec = lw.precision_
    return fast_maha(va,means,prec), fast_maha(od,means,prec), means, prec

print("Loading features...", flush=True)
tr = {l: l2n(np.concatenate([np.load(x) for x in sorted(glob.glob(f'{train_inter}/layer_{l}_features_*.npy'))],0).astype(np.float64)) for l,_ in layer_dims}
va = {l: l2n(np.concatenate([np.load(x) for x in sorted(glob.glob(f'{val_inter}/layer_{l}_features_*.npy'))],0).astype(np.float64)) for l,_ in layer_dims}
od = {l: l2n(np.load(f'{ood_inter}/layer_{l}_features_0.npy').astype(np.float64)) for l,_ in layer_dims}

er_bs = np.array([float(np.load(f'{path}/mm_pp_{l}_er_b.npy').flat[0]) for l,_ in layer_dims])
order = np.argsort(er_bs); ranks = np.empty(4,dtype=float); ranks[order]=np.arange(1,5)
alpha = ranks/ranks.sum()
print(f"Rank weights: {dict(zip([l for l,_ in layer_dims], alpha.round(3)))}", flush=True)

print(f"\n{'Method':45s} | sample AUROC | per-class AUROC", flush=True)
print('-'*70, flush=True)

combos = [
    ('cat(layer3, layer4) equal', ['layer3','layer4'], [1.0, 1.0]),
    ('cat(layer3*0.43, layer4*0.57) rank-wt', ['layer3','layer4'], [alpha[2]/(alpha[2]+alpha[3]), alpha[3]/(alpha[2]+alpha[3])]),
    ('cat(layer4) only', ['layer4'], [1.0]),
    ('rank-wt cat all 4', [l for l,_ in layer_dims], alpha.tolist()),
]

for name, layers, weights in combos:
    lidx = [([l for l,_ in layer_dims].index(l)) for l in layers]
    w = np.array(weights)
    tr_c = np.concatenate([w[i]*tr[l] for i,l in enumerate(layers)], axis=1)
    va_c = np.concatenate([w[i]*va[l] for i,l in enumerate(layers)], axis=1)
    od_c = np.concatenate([w[i]*od[l] for i,l in enumerate(layers)], axis=1)
    id_s, ood_s, _, _ = fit_score(tr_c, train_labels, va_c, od_c)
    samp = auroc_ood(id_s, ood_s)*100
    pclass = per_class_auroc(id_s, ood_s, ood_labels_true)*100
    print(f"{name:45s} | {samp:.2f}%       | {pclass:.2f}%", flush=True)
