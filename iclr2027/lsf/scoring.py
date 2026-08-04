"""
Fused Mahalanobis scoring for a selected layer subset.

Everything needed to score a subset comes from (a) the joint covariance built
once by joint_stats.py and (b) the per-layer class means already cached by
evaluate_MM_plus_plus. Training features are never re-read.

FOUR SCORING RULES -- the P2 comparison
    Given selected layers l in A, per-class residual r_{c,l}(x), and precision P:

    'joint'        S = -min_c  r_c^T P r_c                  with FULL joint P
                   (MM++ as published: one quadratic form, cross-layer blocks live)

    'block_diag'   S = -min_c  sum_l r_{c,l}^T P_ll r_{c,l} with cross-blocks ZEROED
                   (the covariance is block-diagonal, but the class hypothesis c is
                   still SHARED across layers)

    'additive_min' S = sum_l ( -min_c r_{c,l}^T P_ll r_{c,l} )
                   (each layer picks its OWN nearest class, then scores are summed
                   -- the classical Lee et al. 2018 style fusion)

    'additive_min_z' as above but each layer's score standardised by its ID-val
                   mean/std before summing.

    The pair that matters is 'block_diag' vs 'additive_min'. They use the SAME
    per-layer precisions and differ only in whether the class hypothesis is shared
    (min of the sum) or per-layer (sum of the mins). If the paper's cross-layer
    story is to be replaced by a "shared class hypothesis" story, this is the
    experiment that has to carry it -- and it was missing from the NeurIPS
    submission.

    Standardisation matters: raw d_{c,l} scales with layer dimension and spectrum,
    so an unstandardised sum is dominated by the widest layer. 'additive_min_z'
    removes that confound, using ID-val statistics only (the same convention as
    the existing additive fusion path in detection_methods.py).
"""

import numpy as np

__all__ = [
    'l2_normalize',
    'fuse_features',
    'fused_class_means',
    'block_diagonal_precision',
    'maha_scores',
    'per_layer_maha_scores',
    'score_subset',
]


def l2_normalize(x, eps=1e-10):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)


def fuse_features(layer_feats, all_names, subset):
    """Concatenate L2-normalised features of `subset`, in the order given."""
    idx = [all_names.index(n) for n in subset]
    return np.concatenate([l2_normalize(layer_feats[i]) for i in idx], axis=1)


def fused_class_means(means_list, all_names, subset):
    """Concatenate cached per-layer class means for `subset`, same order."""
    idx = [all_names.index(n) for n in subset]
    return np.concatenate([means_list[i] for i in idx], axis=1)


def block_diagonal_precision(jc):
    """
    Block-diagonal precision of a (subset) JointCovariance: zero the cross-layer
    covariance blocks, then invert each diagonal block.

    Order of operations matches detection_methods._block_diagonal_precision:
    zeroing happens on the COVARIANCE, before inversion, so the result is exactly
    the precision of a block-diagonal covariance -- not the zeroed inverse.
    """
    D = jc.blocks.D
    prec = np.zeros((D, D), dtype=np.float64)
    for p in range(len(jc.blocks)):
        s, e = int(jc.blocks.offsets[p]), int(jc.blocks.offsets[p + 1])
        prec[s:e, s:e] = np.linalg.inv(jc.Sigma[s:e, s:e])
    return prec


def maha_scores(feats, means, prec):
    """S(x) = -min_c (x-mu_c)^T P (x-mu_c). Vectorised; feats [N,D], means [C,D]."""
    feats = np.asarray(feats, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    Pmu = means @ prec
    mu_Pmu = np.sum(means * Pmu, axis=1)
    xP = feats @ prec
    xPx = np.sum(feats * xP, axis=1)
    return -np.min(xPx[:, None] - 2 * (xP @ means.T) + mu_Pmu[None, :], axis=1)


def per_layer_maha_scores(feats, means, jc):
    """
    Per-layer scores using the block-diagonal precisions: returns [N, K] where
    column p is -min_c d_{c,l_p}(x) for the p-th layer of the subset.
    """
    out = []
    for p in range(len(jc.blocks)):
        s, e = int(jc.blocks.offsets[p]), int(jc.blocks.offsets[p + 1])
        prec_p = np.linalg.inv(jc.Sigma[s:e, s:e])
        out.append(maha_scores(feats[:, s:e], means[:, s:e], prec_p))
    return np.stack(out, axis=1)


def score_subset(jc_sub, subset, all_names, means_list,
                 val_layer_feats, ood_layer_feats, mode='joint'):
    """
    Score ID-val and OOD samples for one layer subset.

    Args:
        jc_sub:   JointCovariance restricted to `subset` (use JointCovariance.subset()),
                  so the Ledoit-Wolf intensity is the one that subset would have got.
        subset:   selected layer names, in concatenation order.
        all_names: full ordered layer name list (indexes into the feature lists).
        means_list: per-layer class means, aligned with all_names.
        val_layer_feats / ood_layer_feats: per-layer feature arrays.
        mode:     'joint' | 'block_diag' | 'additive_min' | 'additive_min_z'.

    Returns:
        (scores_id, scores_ood), higher = more ID-like.
    """
    if mode not in ('joint', 'block_diag', 'additive_min', 'additive_min_z'):
        raise ValueError(f'unknown mode={mode!r}')

    means = fused_class_means(means_list, all_names, subset)
    Xv = fuse_features(val_layer_feats, all_names, subset)
    Xo = fuse_features(ood_layer_feats, all_names, subset)

    if mode in ('joint', 'block_diag'):
        prec = jc_sub.precision() if mode == 'joint' else block_diagonal_precision(jc_sub)
        return maha_scores(Xv, means, prec), maha_scores(Xo, means, prec)

    Sv = per_layer_maha_scores(Xv, means, jc_sub)      # [N_val, K]
    So = per_layer_maha_scores(Xo, means, jc_sub)      # [N_ood, K]
    if mode == 'additive_min_z':
        # Standardise each layer by ID-val statistics only (no OOD leakage).
        m = Sv.mean(axis=0, keepdims=True)
        s = Sv.std(axis=0, keepdims=True).clip(min=1e-8)
        Sv = (Sv - m) / s
        So = (So - m) / s
    return Sv.sum(axis=1), So.sum(axis=1)
