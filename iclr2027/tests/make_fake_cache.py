#!/usr/bin/env python3
"""
Generate a tiny synthetic cache with the EXACT on-disk layout evaluate.py produces.

Purpose: end-to-end-test run_selection.py and run_oracle_sweep.py without needing
ImageNet, a GPU, or the multi-hundred-GB real caches. The synthetic backbone has a
planted redundancy structure, so the selector's behaviour is predictable:

    layer_00 .. layer_05, with layer_05 the anchor (final).
    layer_04 is ~a linear copy of layer_05   -> redundant, should be REJECTED
    layer_01 is independent of layer_05      -> complementary, should be PICKED

It also writes 12 training slices on purpose, so the lexicographic-vs-numeric
shard-order divergence (see lsf/cache_io.py) is present and `preflight` has
something real to detect.

Usage:
    python3 iclr2027/tests/make_fake_cache.py --out /tmp/fake_cache
    python3 iclr2027/experiments/run_selection.py \
        --model_name fake_model --path_to_cache /tmp/fake_cache
"""

import argparse
import json
import os

import numpy as np

MODEL = 'fake_model'
LAYERS = [f'layer_{i:02d}' for i in range(6)]
# Isotropic widths, like ViT-B/16. This matters: the entropy DENSITY rho = H/D is
# bounded by ln(D)/D, so with heterogeneous widths and near-flat spectra the
# entropy-density DROP mostly tracks where the layer width changes rather than
# where the representation compresses. Keeping D constant isolates the spectrum,
# which is the effect the criterion is supposed to measure.
DIMS = [32] * 6
N_CLASSES = 8
N_TRAIN_PER_SLICE = 250
N_SLICES = 12          # > 10 on purpose: triggers the shard-order divergence
N_VAL = 400
N_OOD = 400

# Within-class covariance spectra (the quantity the selector actually sees; class
# means are removed before the covariance is estimated).
FLAT = np.linspace(1.0, 0.75, DIMS[0])                       # high entropy
MODERATE = np.concatenate([np.linspace(1.0, 0.30, 8), np.full(DIMS[0] - 8, 0.12)])
CONCENTRATED = np.concatenate([[1.0], np.full(DIMS[0] - 1, 0.02)])  # low entropy


def _basis(rng, d):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def _noise(rng, n, basis, spectrum):
    """Within-class noise with a prescribed covariance spectrum in `basis`."""
    return (rng.standard_normal((n, len(spectrum))) * np.sqrt(spectrum)) @ basis.T


def _make_features(rng, n, class_ids, P, ood_shift=0.0):
    """
    Per-layer features with a planted structure (all widths equal):

      layer_05 (anchor)  concentrated within-class spectrum
      layer_04           ~0.99 * rotation(layer_05) + tiny noise -> REDUNDANT, and
                         (being a copy) also concentrated, so it shows the LARGEST
                         entropy-density drop -> a relevance-only rule wants it
      layer_03           flat spectrum (high entropy) -> creates that large drop
      layer_01           independent, moderate spectrum -> COMPLEMENTARY
      layer_00, _02      flat nuisance
    """
    feats = {}
    d = DIMS[0]
    anchor = _noise(rng, n, P['b5'], CONCENTRATED) + P['l5'][class_ids]
    if ood_shift:
        anchor = anchor + ood_shift * rng.standard_normal((1, d))
    feats[LAYERS[5]] = anchor

    # Near-copy of the anchor through an orthogonal map: redundant by construction.
    feats[LAYERS[4]] = 0.99 * (anchor @ P['W']) + 0.02 * rng.standard_normal((n, d))

    feats[LAYERS[3]] = _noise(rng, n, P['b3'], FLAT) + 0.2 * P['l3'][class_ids]
    feats[LAYERS[2]] = _noise(rng, n, P['b2'], FLAT) + 0.2 * P['l2'][class_ids]
    feats[LAYERS[0]] = _noise(rng, n, P['b0'], FLAT) + 0.2 * P['l0'][class_ids]

    comp = _noise(rng, n, P['b1'], MODERATE) + P['l1'][class_ids]
    if ood_shift:
        comp = comp + 0.6 * ood_shift * rng.standard_normal((1, d))
    feats[LAYERS[1]] = comp
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/fake_cache')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    # spurious numpy/Accelerate matmul warnings on macOS (see lsf README)
    import warnings
    warnings.filterwarnings('ignore', message='.*encountered in matmul',
                            category=RuntimeWarning)
    rng = np.random.default_rng(args.seed)

    root = args.out
    d_train = os.path.join(root, 'cache_train', MODEL)
    d_tinter = os.path.join(root, 'cache_train_inter', MODEL)
    d_vinter = os.path.join(root, 'cache_val_inter', MODEL)
    d_ointer = os.path.join(root, 'cache_ood_inter', MODEL)
    d_meth = os.path.join(root, 'cache_methods', MODEL)
    for d in (d_train, d_tinter, d_vinter, d_meth):
        os.makedirs(d, exist_ok=True)

    d = DIMS[0]
    class_offsets = {
        'l5': rng.standard_normal((N_CLASSES, d)) * 1.3,
        'l1': rng.standard_normal((N_CLASSES, d)) * 1.3,
        'l0': rng.standard_normal((N_CLASSES, d)),
        'l2': rng.standard_normal((N_CLASSES, d)),
        'l3': rng.standard_normal((N_CLASSES, d)),
        'W': _basis(rng, d),                       # orthogonal: a clean copy
        'b0': _basis(rng, d), 'b1': _basis(rng, d), 'b2': _basis(rng, d),
        'b3': _basis(rng, d), 'b5': _basis(rng, d),
    }

    for d in (d_tinter, d_vinter):
        with open(os.path.join(d, 'layer_names.json'), 'w') as f:
            json.dump(LAYERS, f)

    # ── training slices ──────────────────────────────────────────────────────
    all_labels = []
    train_norm_sums = {ln: np.zeros((N_CLASSES, dim)) for ln, dim in zip(LAYERS, DIMS)}
    class_counts = np.zeros(N_CLASSES)
    for s in range(N_SLICES):
        y = rng.integers(0, N_CLASSES, N_TRAIN_PER_SLICE)
        feats = _make_features(rng, N_TRAIN_PER_SLICE, y, class_offsets)
        np.save(os.path.join(d_train, f'labels_true_{s}.npy'), y)
        np.save(os.path.join(d_train, f'features_{s}.npy'),
                feats[LAYERS[5]].astype(np.float32))
        np.save(os.path.join(d_train, f'logits_{s}.npy'),
                rng.standard_normal((N_TRAIN_PER_SLICE, N_CLASSES)).astype(np.float32))
        for ln in LAYERS:
            np.save(os.path.join(d_tinter, f'layer_{ln}_features_{s}.npy'),
                    feats[ln].astype(np.float32))
            fn = feats[ln] / np.linalg.norm(feats[ln], axis=1, keepdims=True).clip(min=1e-10)
            for c in range(N_CLASSES):
                train_norm_sums[ln][c] += fn[y == c].sum(axis=0)
        for c in range(N_CLASSES):
            class_counts[c] += (y == c).sum()
        all_labels.append(y)

    # class means on L2-normalised features -- exactly what evaluate_MM_plus_plus caches
    for ln in LAYERS:
        means = train_norm_sums[ln] / class_counts[:, None].clip(min=1)
        np.save(os.path.join(d_meth, f'mm_pp_{ln}_mean.npy'), means)

    # ── ID val ───────────────────────────────────────────────────────────────
    yv = rng.integers(0, N_CLASSES, N_VAL)
    vfeats = _make_features(rng, N_VAL, yv, class_offsets)
    for ln in LAYERS:
        np.save(os.path.join(d_vinter, f'layer_{ln}_features_0.npy'),
                vfeats[ln].astype(np.float32))

    # ── two OOD sets with different shift strengths ──────────────────────────
    for name, shift in [('fake_ood_near', 0.10), ('fake_ood_far', 0.22)]:
        d = os.path.join(d_ointer, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'layer_names.json'), 'w') as f:
            json.dump(LAYERS, f)
        yo = rng.integers(0, N_CLASSES, N_OOD)
        ofeats = _make_features(rng, N_OOD, yo, class_offsets, ood_shift=shift)
        for ln in LAYERS:
            np.save(os.path.join(d, f'layer_{ln}_features_0.npy'),
                    ofeats[ln].astype(np.float32))

    n_train = N_SLICES * N_TRAIN_PER_SLICE
    print(f'Wrote fake cache to {root}')
    print(f'  model      : {MODEL}')
    print(f'  layers     : {LAYERS} dims={DIMS}')
    print(f'  train      : {n_train} samples in {N_SLICES} slices '
          f'(>10 -> shard-order divergence is present)')
    print(f'  val / ood  : {N_VAL} / {N_OOD} x 2 sets')
    print(f'  planted    : layer_04 redundant with anchor layer_05; '
          f'layer_01 complementary')


if __name__ == '__main__':
    main()
