"""
Conditional-covariance novelty scores for layer selection (LSF / ICLR 2027).

This module answers one question: given a set of already-selected layers S, how
much NEW second-order information does candidate layer l carry?

The quantity is the Schur complement (conditional covariance of layer l given S)

    Sigma_{l|S} = Sigma_ll - Sigma_lS Sigma_SS^{-1} Sigma_Sl,

read off a SINGLE jointly-estimated, jointly-shrunk covariance matrix over all
candidate layers (see joint_stats.py). Two novelty scores are derived from it:

  nu_logdet (default, recommended)
      nu = exp( [logdet Sigma_{l|S} - logdet Sigma_ll] / D_l )
         = ( prod_i (1 - rho_i^2) )^{1/D_l}
      i.e. the geometric mean of (1 - rho_i^2) over the canonical correlations
      rho_i between layer l and the selected set S. Properties:
        * lies in (0, 1]: 1 = fully complementary, ->0 = fully redundant;
        * INVARIANT to any invertible linear reparameterisation of either block
          (canonical correlations are);
        * log nu is a differential-entropy density in nats/dimension, so it is
          commensurable with the paper's entropy-density drop Delta_l and the two
          can be added rather than multiplied (see selector.py);
        * equals (negative, per-dimension) Gaussian mutual information between the
          layer and the selected set, so "redundancy" has an information-theoretic
          meaning rather than being a heuristic.

  nu_trace (ablation only)
      nu = tr(Sigma_{l|S}) / tr(Sigma_ll)
      This is the form in the original ICLR roadmap draft. It is retained so the
      paper can report it as an ablation, but it is NOT the default: it is
      dominated by a few high-variance directions and is invariant only to
      isotropic rescaling of layer l, not to general linear reparameterisation.
      test_novelty.py demonstrates that difference concretely.

WHY JOINT SHRINKAGE MATTERS
    If Sigma_ll and Sigma_SS are shrunk SEPARATELY (e.g. one Ledoit-Wolf fit per
    layer), the assembled matrix is not a covariance of anything, and the Schur
    complement above can fail to be PSD -- yielding nu > 1 or negative logdets.
    Shrinking the joint matrix once and slicing blocks out of it keeps every
    Schur complement PSD by construction (a Schur complement of a PD matrix is
    PD). All functions here therefore take an already-jointly-shrunk Sigma and
    refuse to assemble one from per-block fits.

All linear algebra is Cholesky/SVD-based (no explicit inverses) and depends only
on numpy.
"""

import numpy as np

__all__ = [
    'LayerBlocks',
    'canonical_correlations',
    'conditional_covariance',
    'novelty_logdet',
    'novelty_trace',
    'novelty_scores',
]


class LayerBlocks:
    """
    Index bookkeeping for a joint covariance matrix laid out as layer blocks.

    Args:
        names: layer names in concatenation order.
        dims:  per-layer dimensionality, same order; sum(dims) == D_joint.
    """

    def __init__(self, names, dims):
        names = list(names)
        dims = [int(d) for d in dims]
        if len(names) != len(dims):
            raise ValueError(f'{len(names)} names but {len(dims)} dims')
        self.names = names
        self.dims = dims
        self.offsets = np.cumsum([0] + dims)
        self.D = int(self.offsets[-1])

    def __len__(self):
        return len(self.names)

    def index(self, layer):
        """Accept either a layer name or a positional index; return position."""
        if isinstance(layer, (int, np.integer)):
            return int(layer)
        return self.names.index(layer)

    def cols(self, layer):
        """Column indices of one layer block."""
        p = self.index(layer)
        return np.arange(self.offsets[p], self.offsets[p + 1])

    def cols_of(self, layers):
        """Concatenated column indices of several layer blocks, in given order."""
        if len(layers) == 0:
            return np.array([], dtype=int)
        return np.concatenate([self.cols(l) for l in layers])

    def dim(self, layer):
        return self.dims[self.index(layer)]

    def __repr__(self):
        parts = ', '.join(f'{n}:{d}' for n, d in zip(self.names, self.dims))
        return f'LayerBlocks(D={self.D}, [{parts}])'


def _chol(A, jitter=1e-10, name='matrix'):
    """Cholesky with a small adaptive jitter; raises if the matrix is far from PD."""
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + A.T)                      # enforce exact symmetry
    scale = float(np.trace(A)) / max(A.shape[0], 1)
    for k in range(6):
        try:
            return np.linalg.cholesky(A + (0.0 if k == 0 else jitter * (10.0 ** k) * scale) * np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        f'{name} is not positive definite even with jitter; the joint covariance '
        f'was probably assembled from per-block fits or is rank deficient. '
        f'Shrink the JOINT matrix once (see joint_stats.py).'
    )


def canonical_correlations(Sigma, blocks, layer, selected):
    """
    Canonical correlations rho_i between layer `layer` and the union of `selected`.

    Computed by whitening both blocks with Cholesky factors and taking singular
    values of the cross term -- numerically stable, no explicit inverses:

        A = chol(Sigma_ll), B = chol(Sigma_SS)
        T = A^{-1} Sigma_lS B^{-T},   rho = svd(T)

    Returns:
        rho: [min(D_l, D_S)] canonical correlations in [0, 1], descending.
             Empty array if `selected` is empty.
    """
    if len(selected) == 0:
        return np.zeros(0, dtype=np.float64)

    cl = blocks.cols(layer)
    cs = blocks.cols_of(selected)
    S_ll = Sigma[np.ix_(cl, cl)]
    S_ss = Sigma[np.ix_(cs, cs)]
    S_ls = Sigma[np.ix_(cl, cs)]

    A = _chol(S_ll, name=f'Sigma_ll[{layer}]')
    B = _chol(S_ss, name='Sigma_SS')
    # T = A^{-1} S_ls B^{-T}
    T = np.linalg.solve(A, S_ls)                       # A^{-1} S_ls
    T = np.linalg.solve(B, T.T).T                      # (B^{-1} (A^{-1}S_ls)^T)^T
    rho = np.linalg.svd(T, compute_uv=False)
    # Numerical guard: exact arithmetic gives rho in [0,1].
    return np.clip(rho, 0.0, 1.0)


def conditional_covariance(Sigma, blocks, layer, selected):
    """
    Schur complement Sigma_{l|S} = Sigma_ll - Sigma_lS Sigma_SS^{-1} Sigma_Sl.

    PSD by construction when `Sigma` is PD (i.e. jointly shrunk). Returns
    Sigma_ll unchanged when `selected` is empty.
    """
    cl = blocks.cols(layer)
    S_ll = np.asarray(Sigma[np.ix_(cl, cl)], dtype=np.float64)
    if len(selected) == 0:
        return 0.5 * (S_ll + S_ll.T)

    cs = blocks.cols_of(selected)
    S_ss = Sigma[np.ix_(cs, cs)]
    S_ls = Sigma[np.ix_(cl, cs)]

    B = _chol(S_ss, name='Sigma_SS')
    M = np.linalg.solve(B, S_ls.T)                     # B^{-1} Sigma_Sl, [D_S, D_l]
    cond = S_ll - M.T @ M                              # = S_ll - S_lS S_SS^{-1} S_Sl
    return 0.5 * (cond + cond.T)


def novelty_logdet(Sigma, blocks, layer, selected, return_parts=False):
    """
    nu_logdet = exp( [logdet Sigma_{l|S} - logdet Sigma_ll] / D_l )
              = geometric mean over i of (1 - rho_i^2).

    Range (0, 1]:  1.0 = layer is second-order independent of S (fully novel),
                   ->0 = layer is a linear function of S (fully redundant).

    Uses the canonical-correlation identity rather than two slogdet calls: it is
    better conditioned, and it makes the CCA diagnostic and the novelty score
    numerically the same object (they are proved equal in tests/test_novelty.py).

    Args:
        return_parts: if True also return (rho, log_nu_per_dim).
    """
    D_l = blocks.dim(layer)
    if len(selected) == 0:
        return (1.0, (np.zeros(0), 0.0)) if return_parts else 1.0

    rho = canonical_correlations(Sigma, blocks, layer, selected)
    # log prod (1 - rho_i^2), guarded against rho -> 1 (perfectly redundant direction)
    one_minus = np.clip(1.0 - rho ** 2, 1e-300, 1.0)
    log_nu = float(np.sum(np.log(one_minus)) / D_l)
    nu = float(np.exp(log_nu))
    if return_parts:
        return nu, (rho, log_nu)
    return nu


def novelty_trace(Sigma, blocks, layer, selected):
    """
    nu_trace = tr(Sigma_{l|S}) / tr(Sigma_ll)   -- ABLATION ONLY.

    The original roadmap formulation. Kept for the ablation table; see module
    docstring for why nu_logdet is the default.
    """
    cond = conditional_covariance(Sigma, blocks, layer, selected)
    cl = blocks.cols(layer)
    denom = float(np.trace(Sigma[np.ix_(cl, cl)]))
    if denom <= 0:
        return 0.0
    return float(np.trace(cond) / denom)


def novelty_scores(Sigma, blocks, selected, candidates=None, kind='logdet'):
    """
    Novelty of every candidate layer w.r.t. the current selected set.

    Args:
        Sigma:      [D, D] JOINTLY shrunk covariance over all candidate layers.
        blocks:     LayerBlocks describing Sigma's layout.
        selected:   list of already-selected layers (names or indices).
        candidates: layers to score (default: everything not in `selected`).
        kind:       'logdet' (default) or 'trace'.

    Returns:
        dict {layer_name: nu}
    """
    if kind not in ('logdet', 'trace'):
        raise ValueError(f"kind must be 'logdet' or 'trace', got {kind!r}")
    sel_pos = {blocks.index(l) for l in selected}
    if candidates is None:
        candidates = [p for p in range(len(blocks)) if p not in sel_pos]

    fn = novelty_logdet if kind == 'logdet' else novelty_trace
    out = {}
    for c in candidates:
        p = blocks.index(c)
        if p in sel_pos:
            continue
        out[blocks.names[p]] = fn(Sigma, blocks, p, list(sel_pos))
    return out
