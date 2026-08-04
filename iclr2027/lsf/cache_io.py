"""
Cache-layout helpers: everything that knows where the NeurIPS pipeline puts files.

Layout produced by evaluate.py / utils.py under --path_to_cache:

    {cache}/cache_train/{model}/         features_{i}.npy, labels_true_{i}.npy, logits_{i}.npy
    {cache}/cache_train_inter/{model}/   layer_{name}_features_{i}.npy, layer_names.json
    {cache}/cache_val_inter/{model}/     same, for the ID val split
    {cache}/cache_ood_inter/{model}/{ood}/ same, per OOD dataset
    {cache}/cache_methods/{model}/       mm_pp_{layer}_mean.npy / _prec.npy / _er.npy / _er_b.npy

Both caches slice the dataset identically (slice_length = 50000 in utils.py), so
slice i covers the same samples in both.

⚠️ SHARD ORDER -- READ THIS BEFORE TRUSTING ANY ImageNet-1K NUMBER
    The two caches are concatenated with DIFFERENT sort orders:

      * evaluate.OODScore.check_complete() concatenates labels_true_*.npy using
        plain `sorted()` -- LEXICOGRAPHIC: 0, 1, 10, 11, ..., 19, 2, 20, ...
      * utils.load_intermediate_features() and
        detection_methods.evaluate_MM_plus_plus_topk_gating() concatenate
        layer_*_features_*.npy with `key=int(...)` -- NUMERIC: 0, 1, 2, 3, ...

    These agree only while there are at most 10 slices. ImageNet-LT (~115k train
    -> 3 slices) is therefore unaffected, but ImageNet-1K (1.28M train -> 26
    slices) is NOT: the labels handed to MM++ would be a block permutation of the
    labels belonging to the concatenated intermediate features.

    `check_shard_order()` below detects this for a given cache, and
    `load_train_labels()` always concatenates NUMERICALLY so it is aligned with
    the feature shards. Run `preflight()` on the machine that holds the caches
    before regenerating any results.
"""

import json
import os
import re

import numpy as np

__all__ = [
    'CacheLayout',
    'load_train_labels',
    'load_layer_features',
    'load_class_means',
    'check_shard_order',
    'preflight',
    'auroc',
    'fpr_at_tpr',
]

_IDX = re.compile(r'_(\d+)\.npy$')


def _shards(directory, prefix):
    """Files `{prefix}{i}.npy` in NUMERIC slice order, with their indices."""
    out = []
    for fn in os.listdir(directory):
        if fn.startswith(prefix) and fn.endswith('.npy'):
            m = _IDX.search(fn)
            if m:
                out.append((int(m.group(1)), fn))
    out.sort(key=lambda t: t[0])
    return out


class CacheLayout:
    """Resolves the cache directories for one (cache root, model) pair."""

    def __init__(self, path_to_cache, model_name):
        self.root = path_to_cache
        self.model = model_name
        self.train = os.path.join(path_to_cache, 'cache_train', model_name)
        self.train_inter = os.path.join(path_to_cache, 'cache_train_inter', model_name)
        self.val_inter = os.path.join(path_to_cache, 'cache_val_inter', model_name)
        self.ood_inter_root = os.path.join(path_to_cache, 'cache_ood_inter', model_name)
        self.methods = os.path.join(path_to_cache, 'cache_methods', model_name)

    def ood_inter(self, ood_name):
        return os.path.join(self.ood_inter_root, ood_name)

    def layer_names(self):
        p = os.path.join(self.train_inter, 'layer_names.json')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} not found. Run evaluate.py with an MM_plus_plus method once to '
                f'extract intermediate features for {self.model!r}.')
        with open(p) as f:
            return json.load(f)

    def available_ood(self):
        if not os.path.isdir(self.ood_inter_root):
            return []
        return sorted(d for d in os.listdir(self.ood_inter_root)
                      if os.path.isdir(self.ood_inter(d)))

    def describe(self):
        lines = [f'cache root : {self.root}', f'model      : {self.model}']
        for label, p in [('train', self.train), ('train_inter', self.train_inter),
                         ('val_inter', self.val_inter), ('methods', self.methods)]:
            lines.append(f'{label:<11}: {p}  {"OK" if os.path.isdir(p) else "MISSING"}')
        lines.append(f'ood sets   : {", ".join(self.available_ood()) or "(none)"}')
        return '\n'.join(lines)


def load_train_labels(layout):
    """
    Training labels concatenated in NUMERIC slice order, i.e. aligned with the
    intermediate feature shards. See the module docstring for why this matters.
    """
    if not os.path.isdir(layout.train):
        raise FileNotFoundError(f'{layout.train} not found (train feature cache).')
    parts = _shards(layout.train, 'labels_true_')
    if not parts:
        raise FileNotFoundError(f'no labels_true_*.npy in {layout.train}')
    return np.concatenate([np.load(os.path.join(layout.train, fn)) for _, fn in parts])


def load_layer_features(directory, layer_names=None, n_expected=None):
    """
    Per-layer features from an intermediate cache dir, NUMERIC shard order.

    Returns:
        (list of [N, D_l] float64 arrays, layer_names)
    """
    names_path = os.path.join(directory, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(f'{names_path} not found')
    with open(names_path) as f:
        available = json.load(f)
    names = list(layer_names) if layer_names is not None else list(available)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f'layers {missing} not in {names_path}')

    feats = []
    for n in names:
        parts = _shards(directory, f'layer_{n}_features_')
        if not parts:
            raise FileNotFoundError(f'no shards for layer {n!r} in {directory}')
        arr = np.concatenate([np.load(os.path.join(directory, fn)) for _, fn in parts],
                             axis=0).astype(np.float64)
        if n_expected is not None and len(arr) != n_expected:
            raise ValueError(f'layer {n!r}: {len(arr)} samples, expected {n_expected}')
        feats.append(arr)
    return feats, names


def load_class_means(layout, layer_names):
    """Per-layer class means cached by evaluate_MM_plus_plus (already on L2-normalised feats)."""
    means = []
    for n in layer_names:
        p = os.path.join(layout.methods, f'mm_pp_{n}_mean.npy')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} missing. Run evaluate.py --method MM_plus_plus_topk_cat once to '
                f'build the per-layer caches.')
        means.append(np.load(p).astype(np.float64))
    return means


def check_shard_order(layout):
    """
    Detect the lexicographic-vs-numeric shard-order divergence described in the
    module docstring.

    Returns a dict with `aligned` (bool), the slice count, and both orderings.
    """
    result = {'checked': False, 'aligned': None, 'n_slices': 0,
              'numeric': [], 'lexicographic': []}
    if not os.path.isdir(layout.train):
        return result
    names = [fn for fn in os.listdir(layout.train)
             if fn.startswith('labels_true_') and fn.endswith('.npy') and _IDX.search(fn)]
    if not names:
        return result
    numeric = [fn for _, fn in sorted(((int(_IDX.search(f).group(1)), f) for f in names))]
    lexicographic = sorted(names)
    result.update(checked=True, aligned=(numeric == lexicographic), n_slices=len(names),
                  numeric=numeric, lexicographic=lexicographic)
    return result


def preflight(layout, verbose=True):
    """
    Validate a cache before running anything expensive.

    Checks directory presence, shard-order alignment, and that the training label
    count matches the training feature row count.

    Returns a dict of findings; `ok` is False if anything would invalidate results.
    """
    findings = {'ok': True, 'errors': [], 'warnings': []}

    for label, p in [('train', layout.train), ('train_inter', layout.train_inter),
                     ('methods', layout.methods)]:
        if not os.path.isdir(p):
            findings['errors'].append(f'missing {label} cache: {p}')
            findings['ok'] = False
    if not findings['ok']:
        if verbose:
            _print_findings(findings)
        return findings

    order = check_shard_order(layout)
    findings['shard_order'] = order
    if order['checked'] and not order['aligned']:
        findings['warnings'].append(
            f"shard-order divergence: {order['n_slices']} train slices, so "
            f"evaluate.check_complete()'s LEXICOGRAPHIC label order "
            f"({order['lexicographic'][:4]}...) does NOT match the NUMERIC feature "
            f"order ({order['numeric'][:4]}...). Labels passed to MM++ by the "
            f"existing pipeline are a block permutation of the correct ones for this "
            f"cache. lsf loads labels numerically, so results from this folder are "
            f"aligned -- but any previously reported number for this model/ID set "
            f"should be re-checked.")

    try:
        labels = load_train_labels(layout)
        findings['n_train_labels'] = int(len(labels))
        findings['n_classes'] = int(labels.max()) + 1
    except Exception as exc:  # noqa: BLE001
        findings['errors'].append(f'could not load train labels: {exc}')
        findings['ok'] = False
        if verbose:
            _print_findings(findings)
        return findings

    try:
        names = layout.layer_names()
        findings['n_layers'] = len(names)
        parts = _shards(layout.train_inter, f'layer_{names[0]}_features_')
        n_rows = sum(len(np.load(os.path.join(layout.train_inter, fn), mmap_mode='r'))
                     for _, fn in parts)
        findings['n_train_features'] = int(n_rows)
        if n_rows != len(labels):
            findings['errors'].append(
                f'{n_rows} training feature rows but {len(labels)} labels -- caches are '
                f'inconsistent; re-extract before proceeding.')
            findings['ok'] = False
    except Exception as exc:  # noqa: BLE001
        findings['errors'].append(f'could not inspect intermediate features: {exc}')
        findings['ok'] = False

    if verbose:
        _print_findings(findings)
    return findings


def _print_findings(f):
    print('[preflight] ' + ('OK' if f['ok'] else 'PROBLEMS FOUND'))
    for k in ('n_layers', 'n_train_labels', 'n_train_features', 'n_classes'):
        if k in f:
            print(f'[preflight]   {k}: {f[k]}')
    for w in f['warnings']:
        print(f'[preflight]   WARNING: {w}')
    for e in f['errors']:
        print(f'[preflight]   ERROR:   {e}')


# ── metrics (numpy-only equivalents of utils.auroc_ood / utils.fpr_at_tpr) ────

def auroc(scores_id, scores_ood):
    """
    AUROC with ID as the positive class. Mann-Whitney U form -- exactly equal to
    sklearn.metrics.roc_auc_score (including tie handling) but without the import,
    so this module stays numpy-only.
    """
    a = np.nan_to_num(np.asarray(scores_id, dtype=np.float64).ravel())
    b = np.nan_to_num(np.asarray(scores_ood, dtype=np.float64).ravel())
    n1, n0 = len(a), len(b)
    if n1 == 0 or n0 == 0:
        return float('nan')
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # average ranks within ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fpr_at_tpr(scores_id, scores_ood, tpr=0.95):
    """FPR at the given TPR. Matches utils.fpr_at_tpr (quantile + >= threshold)."""
    a = np.asarray(scores_id, dtype=np.float64).ravel()
    b = np.asarray(scores_ood, dtype=np.float64).ravel()
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    t = np.quantile(a, 1 - tpr)
    return float((b >= t).mean())
