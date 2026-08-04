"""
Joint (all-candidate-layers) class-conditional covariance, estimated in one pass.

WHY THIS EXISTS
    Every quantity the LSF selector needs -- conditional covariances Sigma_{l|S},
    canonical correlations, condition numbers, the K<=3 oracle sweep -- is a
    SUBMATRIX of a single joint covariance over all candidate layers. Estimating
    that matrix once and slicing it has three consequences:

      1. Correctness. Ledoit-Wolf shrinkage is applied ONCE to the joint matrix,
         so every Schur complement taken from it is PSD by construction. Fitting
         one Ledoit-Wolf per layer block and assembling them (the obvious
         implementation) does not have this property: the assembled matrix is not
         the covariance of anything and Sigma_{l|S} can come out indefinite.

      2. Cost. The expensive part of the pipeline is reading features off disk.
         One streaming pass gives every layer pair, so the selector, the
         diagnostics, and the oracle subset sweep share a single pass instead of
         re-reading features per candidate subset.

      3. Reproducibility. The result is a single cacheable artifact (.npz) that
         can be versioned with the paper -- see iclr2027/results/.

MEMORY
    The accumulator is D_joint x D_joint float64, where D_joint is the summed
    width of all candidate layers (ViT-B/16, 12 blocks x 768 = 9216 -> ~680 MB).
    Pass `layer_subset` to restrict the candidate set if that is too large, or
    `dtype=np.float32` to halve it at some precision cost.

EXACT LEDOIT-WOLF FROM STREAMING ACCUMULATORS
    sklearn's ledoit_wolf_shrinkage needs the full data matrix, which we never
    hold in memory. It turns out only three accumulators are required:

        G = sum_i x_i x_i^T        (D x D)
        q = sum_i ||x_i||^4        (scalar)
        N = number of samples

    because sklearn's `beta_` term is exactly sum_i ||x_i||^4 and its `delta_`
    term is exactly ||G||_F^2. `ledoit_wolf_from_accumulators` below reproduces
    sklearn's shrinkage intensity bit-for-bit (verified in tests against sklearn
    when it is installed), so the joint fit stays consistent with the per-layer
    fits the rest of the codebase already uses.

    x_i here are L2-normalised per layer block and centred by CLASS mean -- the
    same "tied within-class covariance" convention as detection_methods.py.

EXACT SHRINKAGE FOR *EVERY* LAYER SUBSET, FROM THE SAME PASS
    One extra L x L accumulator makes the oracle subset sweep nearly free:

        M[l, l'] = sum_i s_il * s_il' ,   s_il = ||x_i restricted to layer l||^2

    For any subset A of layers, sum_i ||x_i restricted to A||^4
    = sum_i (sum_{l in A} s_il)^2 = sum_{l,l' in A} M[l, l'], so the exact
    Ledoit-Wolf intensity of ANY subset follows from a submatrix of M and a
    submatrix of G. `JointCovariance.subset()` uses this: evaluating all 286
    subsets of size <= 3 needs ONE pass over the training features, not 286.
"""

import json
import os

import numpy as np

from .novelty import LayerBlocks

__all__ = [
    'ledoit_wolf_from_accumulators',
    'shrink',
    'JointCovariance',
    'StreamingJointCovariance',
    'joint_covariance_from_features',
    'build_joint_covariance',
]


def ledoit_wolf_from_accumulators(G, q, n_samples):
    """
    Ledoit-Wolf shrinkage intensity from streaming accumulators.

    Reproduces sklearn.covariance.ledoit_wolf_shrinkage(X, assume_centered=True)
    exactly, without needing X.

    Args:
        G:         [D, D] sum_i x_i x_i^T  (NOT divided by N)
        q:         scalar sum_i ||x_i||^4
        n_samples: N

    Returns:
        (shrinkage, mu) where mu = tr(emp_cov)/D is the shrinkage target scale.
    """
    G = np.asarray(G, dtype=np.float64)
    n = int(n_samples)
    p = G.shape[0]
    if n < 2:
        raise ValueError(f'need at least 2 samples, got {n}')

    emp_cov = G / n
    mu = float(np.trace(emp_cov) / p)

    # delta_ = ||emp_cov||_F^2  (sklearn: sum of squared coefficients of X.T@X / n^2)
    delta_ = float(np.sum(emp_cov ** 2))
    # beta_  = 1/(p*n) * (sum_i ||x_i||^4 / n - ||emp_cov||_F^2)
    beta_ = (1.0 / (p * n)) * (float(q) / n - delta_)
    # delta  = ||emp_cov - mu*I||_F^2 / p = (||emp_cov||_F^2 - p*mu^2)/p
    delta = (delta_ - p * mu ** 2) / p

    if delta <= 0:
        return 0.0, mu
    beta = min(beta_, delta)
    shrinkage = 0.0 if beta <= 0 else float(beta / delta)
    return float(np.clip(shrinkage, 0.0, 1.0)), mu


def shrink(emp_cov, shrinkage, mu):
    """Sigma = (1 - gamma) * emp_cov + gamma * mu * I   (Ledoit-Wolf, Eq. 3/9 of the paper)."""
    emp_cov = np.asarray(emp_cov, dtype=np.float64)
    out = (1.0 - shrinkage) * emp_cov
    out.flat[:: emp_cov.shape[0] + 1] += shrinkage * mu
    return out


class JointCovariance:
    """
    A jointly-estimated, jointly-shrunk covariance over all candidate layers,
    plus the accumulators needed to re-derive any SUBSET's exact shrinkage.

    Attributes:
        Sigma:      [D, D] shrunk joint covariance -- the matrix to slice.
        emp_cov:    [D, D] unshrunk empirical joint covariance (= G / N).
        G:          [D, D] sum_i x_i x_i^T.
        M:          [L, L] sum_i s_il s_il' (see module docstring), or None.
        blocks:     LayerBlocks layout.
        shrinkage:  Ledoit-Wolf intensity gamma actually applied.
        mu:         shrinkage target scale.
        n_samples:  samples used.
    """

    def __init__(self, G, M, n_samples, blocks, meta=None, shrinkage=None):
        self.G = np.asarray(G, dtype=np.float64)
        self.M = None if M is None else np.asarray(M, dtype=np.float64)
        self.n_samples = int(n_samples)
        self.blocks = blocks
        self.meta = dict(meta or {})

        self.emp_cov = self.G / self.n_samples
        if shrinkage is None:
            if self.M is None:
                raise ValueError(
                    'cannot derive Ledoit-Wolf shrinkage without the M accumulator; '
                    'pass shrinkage=... explicitly or rebuild with StreamingJointCovariance')
            gamma, mu = ledoit_wolf_from_accumulators(self.G, float(self.M.sum()), self.n_samples)
        else:
            gamma = float(shrinkage)
            mu = float(np.trace(self.emp_cov) / self.blocks.D)
        self.shrinkage = float(gamma)
        self.mu = float(mu)
        self.Sigma = shrink(self.emp_cov, self.shrinkage, self.mu)

    @classmethod
    def from_covariance(cls, Sigma, blocks, n_samples=100000, shrinkage=0.0):
        """Wrap an analytic/known covariance (tests, fixtures). No M accumulator."""
        Sigma = np.asarray(Sigma, dtype=np.float64)
        return cls(Sigma * n_samples, None, n_samples, blocks, shrinkage=shrinkage)

    def block(self, layer_a, layer_b=None):
        """Covariance block Sigma[a, b] of the SHRUNK matrix (b defaults to a)."""
        ca = self.blocks.cols(layer_a)
        cb = ca if layer_b is None else self.blocks.cols(layer_b)
        return self.Sigma[np.ix_(ca, cb)]

    def subset(self, layers):
        """
        JointCovariance restricted to `layers`, with the exact Ledoit-Wolf
        intensity that fitting THAT subset alone would have produced.

        This is what makes the oracle subset sweep affordable: the fused
        covariance and shrinkage of any layer subset come from submatrices of the
        single joint pass, so no training features are re-read.

        Falls back to the parent's shrinkage (with a warning) when the M
        accumulator is unavailable.
        """
        layers = list(layers)
        idx = [self.blocks.index(l) for l in layers]
        cols = self.blocks.cols_of(layers)
        sub_blocks = LayerBlocks([self.blocks.names[i] for i in idx],
                                 [self.blocks.dims[i] for i in idx])
        G_sub = self.G[np.ix_(cols, cols)]
        if self.M is None:
            print('[JointCov] warning: no M accumulator; reusing the joint shrinkage '
                  'intensity for the subset instead of re-deriving it.')
            return JointCovariance(G_sub, None, self.n_samples, sub_blocks,
                                   dict(self.meta, subset_of=self.blocks.names),
                                   shrinkage=self.shrinkage)
        M_sub = self.M[np.ix_(idx, idx)]
        return JointCovariance(G_sub, M_sub, self.n_samples, sub_blocks,
                               dict(self.meta, subset_of=self.blocks.names))

    def precision(self):
        """Inverse of the shrunk joint covariance (PD by construction)."""
        return np.linalg.inv(self.Sigma)

    def condition_number(self, layers=None):
        """Condition number of the shrunk covariance restricted to `layers`."""
        if layers is None:
            A = self.Sigma
        else:
            c = self.blocks.cols_of(layers)
            A = self.Sigma[np.ix_(c, c)]
        eig = np.linalg.eigvalsh(0.5 * (A + A.T))
        lo, hi = float(eig.min()), float(eig.max())
        return float(hi / lo) if lo > 0 else float('inf')

    def save(self, path):
        np.savez_compressed(
            path,
            G=self.G,
            M=np.array([]) if self.M is None else self.M,
            has_M=self.M is not None,
            names=np.array(self.blocks.names, dtype=object),
            dims=np.array(self.blocks.dims, dtype=np.int64),
            shrinkage=self.shrinkage,
            mu=self.mu,
            n_samples=self.n_samples,
            meta=json.dumps(self.meta),
        )
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        blocks = LayerBlocks([str(n) for n in z['names']], [int(d) for d in z['dims']])
        meta = json.loads(str(z['meta'])) if 'meta' in z else {}
        M = z['M'] if bool(z['has_M']) else None
        # shrinkage is re-derived from the accumulators when M is present, which
        # also serves as a consistency check against the stored value.
        jc = cls(z['G'], M, int(z['n_samples']), blocks, meta,
                 shrinkage=None if M is not None else float(z['shrinkage']))
        stored = float(z['shrinkage'])
        if abs(jc.shrinkage - stored) > 1e-9:
            print(f'[JointCov] warning: re-derived gamma={jc.shrinkage:.9f} differs from '
                  f'stored {stored:.9f}')
        return jc

    def __repr__(self):
        return (f'JointCovariance(D={self.blocks.D}, L={len(self.blocks)}, '
                f'N={self.n_samples}, gamma={self.shrinkage:.6f})')


class StreamingJointCovariance:
    """
    Accumulates G, M, N over chunks of class-centred features.

    Usage:
        acc = StreamingJointCovariance(blocks)
        for chunk in ...:                     # chunk: [n, D] already centred
            acc.update(chunk)
        jc = acc.finalize()
    """

    def __init__(self, blocks, dtype=np.float64):
        self.blocks = blocks
        self.G = np.zeros((blocks.D, blocks.D), dtype=dtype)
        self.M = np.zeros((len(blocks), len(blocks)), dtype=np.float64)
        self.n = 0

    def update(self, Xc):
        """Xc: [n, D] class-mean-centred rows."""
        Xc = np.asarray(Xc, dtype=np.float64)
        if Xc.ndim != 2 or Xc.shape[1] != self.blocks.D:
            raise ValueError(f'expected [n, {self.blocks.D}], got {Xc.shape}')
        self.G += Xc.T @ Xc
        # s[:, l] = ||x_i restricted to layer l||^2, so M enables exact per-subset LW.
        s = np.empty((Xc.shape[0], len(self.blocks)), dtype=np.float64)
        for p in range(len(self.blocks)):
            blk = Xc[:, self.blocks.offsets[p]:self.blocks.offsets[p + 1]]
            s[:, p] = np.einsum('ij,ij->i', blk, blk)
        self.M += s.T @ s
        self.n += Xc.shape[0]
        return self

    @property
    def q(self):
        """sum_i ||x_i||^4 over the FULL joint vector (= sum of all M entries)."""
        return float(self.M.sum())

    def finalize(self, meta=None):
        if self.n < 2:
            raise ValueError('no samples accumulated')
        return JointCovariance(self.G, self.M, self.n, self.blocks, meta)


def joint_covariance_from_features(X, labels, class_means, blocks, meta=None):
    """
    In-memory convenience path (tests, small calibration sets).

    Args:
        X:           [N, D] concatenated features, ALREADY L2-normalised per block.
        labels:      [N] integer class labels.
        class_means: [C, D] per-class means in the same concatenated layout.
        blocks:      LayerBlocks.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    Xc = X - np.asarray(class_means, dtype=np.float64)[labels]
    return StreamingJointCovariance(blocks).update(Xc).finalize(meta)


def _shard_files(train_inter_path, layer_name):
    parts = [fn for fn in os.listdir(train_inter_path)
             if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')]
    if not parts:
        raise FileNotFoundError(
            f'No training feature shards for layer {layer_name!r} in {train_inter_path}')
    return sorted(parts, key=lambda x: int(x.split('_')[-1].replace('.npy', '')))


def build_joint_covariance(train_inter_path, cache_path, train_labels,
                           layer_subset=None, max_samples=None, seed=42,
                           out_path=None, verbose=True):
    """
    Stream the per-layer feature shards written by utils.extract_intermediate_features
    and build the joint class-conditional covariance over all candidate layers.

    Mirrors the feature convention of detection_methods.evaluate_MM_plus_plus_topk_gating:
    each layer block is L2-normalised, then centred by its cached class mean
    (mm_pp_{layer}_mean.npy), i.e. a tied within-class covariance.

    Args:
        train_inter_path: dir with layer_{name}_features_{i}.npy + layer_names.json.
        cache_path:       dir with the per-layer mm_pp_{name}_mean.npy caches.
        train_labels:     [N_train] integer labels, aligned with shard order.
        layer_subset:     candidate layer names (default: all layers).
        max_samples:      approximate cap on samples used (subsampled per shard
                          with a fixed seed). None = use everything.
        out_path:         if given, save the JointCovariance .npz here.

    Returns:
        JointCovariance
    """
    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(f'layer_names.json not found in {train_inter_path}')
    with open(names_path) as f:
        all_names = json.load(f)

    names = list(layer_subset) if layer_subset is not None else list(all_names)
    missing = [n for n in names if n not in all_names]
    if missing:
        raise ValueError(f'layers not present in {names_path}: {missing}')

    means = []
    dims = []
    for n in names:
        mp = os.path.join(cache_path, f'mm_pp_{n}_mean.npy')
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f'class-mean cache missing for {n!r} ({mp}). '
                f'Run evaluate_MM_plus_plus first to build per-layer caches.')
        m = np.load(mp).astype(np.float64)
        means.append(m)
        dims.append(m.shape[1])

    blocks = LayerBlocks(names, dims)
    class_means = np.concatenate(means, axis=1)          # [C, D_joint]
    train_labels = np.asarray(train_labels)

    gb = blocks.D * blocks.D * 8 / 1e9
    if verbose:
        print(f'[JointCov] {len(names)} candidate layers, D_joint={blocks.D} '
              f'(accumulator ~{gb:.2f} GB)')

    shards = {n: _shard_files(train_inter_path, n) for n in names}
    n_shards = len(shards[names[0]])
    for n in names:
        if len(shards[n]) != n_shards:
            raise ValueError(
                f'layer {n!r} has {len(shards[n])} shards but {names[0]!r} has {n_shards}; '
                f'shard layouts must match for row alignment.')

    acc = StreamingJointCovariance(blocks)
    rng = np.random.default_rng(seed)
    row0 = 0
    for si in range(n_shards):
        chunk = []
        n_rows = None
        for n in names:
            arr = np.load(os.path.join(train_inter_path, shards[n][si])).astype(np.float64)
            if n_rows is None:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError(
                    f'shard {si} row mismatch: {n!r} has {arr.shape[0]} rows, '
                    f'expected {n_rows}. Shards must be row-aligned across layers.')
            arr /= np.linalg.norm(arr, axis=-1, keepdims=True).clip(min=1e-10)
            chunk.append(arr)
        X = np.concatenate(chunk, axis=1)
        del chunk
        y = train_labels[row0:row0 + n_rows]
        row0 += n_rows

        if max_samples is not None:
            keep = min(n_rows, max(2, int(round(max_samples * n_rows / len(train_labels)))))
            if keep < n_rows:
                idx = rng.choice(n_rows, keep, replace=False)
                X = X[idx]
                y = y[idx]

        X -= class_means[y]
        acc.update(X)
        del X
        if verbose:
            print(f'[JointCov] shard {si + 1}/{n_shards}: {acc.n} samples accumulated')

    meta = {
        'train_inter_path': train_inter_path,
        'cache_path': cache_path,
        'layers': names,
        'max_samples': max_samples,
        'seed': seed,
    }
    jc = acc.finalize(meta)
    if verbose:
        print(f'[JointCov] done: N={jc.n_samples}, Ledoit-Wolf gamma={jc.shrinkage:.6f}, '
              f'mu={jc.mu:.6e}')
    if out_path:
        jc.save(out_path)
        if verbose:
            print(f'[JointCov] saved -> {out_path}')
    return jc
