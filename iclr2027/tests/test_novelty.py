"""
Correctness tests for the LSF selector math.

These run on synthetic data with a KNOWN redundancy structure, so they verify the
estimator before any feature extraction or GPU time is spent. Several of them are
also the evidence behind design choices in the paper:

  test_logdet_identity              nu_logdet really is the geometric mean of
                                    (1 - rho_i^2) over canonical correlations,
                                    so the CCA diagnostic and the selector score
                                    are the same object.
  test_logdet_invariant_trace_not   nu_logdet is invariant to invertible linear
                                    reparameterisation of a layer; nu_trace is
                                    not. This is why logdet is the default.
  test_per_block_shrinkage_breaks_psd
                                    shrinking each layer block separately can make
                                    the conditional covariance INDEFINITE, while
                                    shrinking the joint matrix once cannot. This is
                                    the concrete reason joint_stats.py exists.
  test_selector_avoids_redundant_layer
                                    reproduces the paper's K=3 failure in miniature:
                                    the entropy-drop-only rule picks the redundant
                                    layer, the LSF rule picks the complementary one.

Run:  python3 iclr2027/tests/test_novelty.py     (or with pytest)
Depends on numpy only; the sklearn cross-check skips itself if sklearn is absent.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsf import (  # noqa: E402
    JointCovariance,
    LayerBlocks,
    canonical_correlations,
    conditional_covariance,
    entropy_density_drops,
    joint_covariance_from_features,
    ledoit_wolf_from_accumulators,
    novelty_logdet,
    novelty_trace,
    select_layers,
    shrink,
)

TOL = 1e-8


# ─────────────────────────── synthetic constructions ──────────────────────────

def _random_pd(D, rng, cond=10.0):
    """Random PD matrix with a bounded condition number."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    eig = np.linspace(1.0, 1.0 / cond, D)
    return Q @ np.diag(eig) @ Q.T


def _four_layer_sigma(c_redundant=0.98):
    """
    Analytic 4-layer joint covariance reproducing the paper's K=3 pathology.

    Depth order [L0, L1, L2, L3]; L3 is the anchor (final layer).
      L3 : concentrated spectrum  -> low entropy density
      L2 : ~a copy of L3 (c_redundant) -> low entropy density, HIGHLY REDUNDANT
      L1 : independent, moderate spectrum -> higher entropy density, COMPLEMENTARY
      L0 : independent, near-flat spectrum -> highest entropy density

    Consequence: the largest entropy-density DROP sits at L2 (because rho_1 >> rho_2),
    so a relevance-only selector picks the redundant layer -- exactly the block_11
    behaviour at K=3 in the NeurIPS submission.
    """
    d = 6
    S3 = np.diag([1.0, 0.05, 0.05, 0.02, 0.02, 0.02])       # anchor, concentrated
    S1 = np.diag([1.0, 0.50, 0.30, 0.20, 0.15, 0.10])       # complementary
    S0 = np.diag([1.0, 0.90, 0.85, 0.80, 0.75, 0.70])       # near-flat
    noise2 = np.eye(d) * 2.5e-3

    c = c_redundant
    S22 = c ** 2 * S3 + noise2                              # L2 = c*L3 + noise
    S23 = c * S3

    D = 4 * d
    Sigma = np.zeros((D, D))
    sl = [slice(i * d, (i + 1) * d) for i in range(4)]
    Sigma[sl[0], sl[0]] = S0
    Sigma[sl[1], sl[1]] = S1
    Sigma[sl[2], sl[2]] = S22
    Sigma[sl[3], sl[3]] = S3
    Sigma[sl[2], sl[3]] = S23
    Sigma[sl[3], sl[2]] = S23.T

    blocks = LayerBlocks(['L0', 'L1', 'L2', 'L3'], [d] * 4)
    return Sigma, blocks


def _as_jc(Sigma, blocks, n_samples=100000):
    """Wrap an analytic covariance as a JointCovariance with no extra shrinkage."""
    return JointCovariance.from_covariance(Sigma, blocks, n_samples=n_samples, shrinkage=0.0)


# ────────────────────────────────── tests ────────────────────────────────────

def test_lw_matches_sklearn():
    """Streaming Ledoit-Wolf accumulators reproduce sklearn's shrinkage exactly."""
    try:
        from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage
    except ImportError:
        print('  ! sklearn not installed - skipping cross-check '
              '(run inside the mm_plus_plus conda env to enable)')
        return

    rng = np.random.default_rng(0)
    D, N = 25, 900
    X = rng.standard_normal((N, D)) @ _random_pd(D, rng, cond=50.0)

    G = X.T @ X
    q = float(np.sum(np.einsum('ij,ij->i', X, X) ** 2))
    gamma, mu = ledoit_wolf_from_accumulators(G, q, N)

    gamma_ref = float(ledoit_wolf_shrinkage(X, assume_centered=True))
    assert abs(gamma - gamma_ref) < 1e-12, f'{gamma} != {gamma_ref}'

    lw = LedoitWolf(assume_centered=True).fit(X)
    Sigma = shrink(G / N, gamma, mu)
    assert np.allclose(Sigma, lw.covariance_, atol=1e-12), 'shrunk covariance mismatch'
    print(f'  gamma={gamma:.10f} matches sklearn; covariance matches to 1e-12')


def test_schur_is_psd():
    """Conditional covariance from a jointly-shrunk matrix is PSD."""
    rng = np.random.default_rng(1)
    D, d = 24, 6
    Sigma = _random_pd(D, rng, cond=500.0)
    blocks = LayerBlocks(['a', 'b', 'c', 'd'], [d] * 4)

    for layer in range(4):
        others = [p for p in range(4) if p != layer]
        cond = conditional_covariance(Sigma, blocks, layer, others)
        eig = np.linalg.eigvalsh(cond)
        assert eig.min() > -1e-10, f'layer {layer}: min eig {eig.min()}'
    print('  all Schur complements PSD (min eig > -1e-10)')


def test_logdet_identity():
    """logdet Sigma_{l|S} - logdet Sigma_ll == sum_i log(1 - rho_i^2)."""
    rng = np.random.default_rng(2)
    D, d = 24, 6
    Sigma = _random_pd(D, rng, cond=100.0)
    blocks = LayerBlocks(['a', 'b', 'c', 'd'], [d] * 4)

    layer, sel = 0, [1, 2]
    cond = conditional_covariance(Sigma, blocks, layer, sel)
    cl = blocks.cols(layer)
    lhs = np.linalg.slogdet(cond)[1] - np.linalg.slogdet(Sigma[np.ix_(cl, cl)])[1]

    rho = canonical_correlations(Sigma, blocks, layer, sel)
    rhs = float(np.sum(np.log(1.0 - rho ** 2)))

    assert abs(lhs - rhs) < 1e-8, f'{lhs} != {rhs}'
    # and nu is exp of the per-dimension version
    nu = novelty_logdet(Sigma, blocks, layer, sel)
    assert abs(np.log(nu) - lhs / blocks.dim(layer)) < 1e-10
    print(f'  identity holds: {lhs:.12f} == {rhs:.12f};  nu={nu:.6f}')


def test_canonical_correlations_bounded():
    """Canonical correlations lie in [0, 1] and come back sorted descending."""
    rng = np.random.default_rng(3)
    Sigma = _random_pd(24, rng, cond=200.0)
    blocks = LayerBlocks(['a', 'b', 'c', 'd'], [6] * 4)
    rho = canonical_correlations(Sigma, blocks, 0, [1, 2, 3])
    assert rho.min() >= 0.0 and rho.max() <= 1.0, f'out of range: {rho}'
    assert np.all(np.diff(rho) <= 1e-12), f'not descending: {rho}'
    print(f'  rho in [{rho.min():.4f}, {rho.max():.4f}], descending, n={len(rho)}')


def test_logdet_invariant_trace_not():
    """
    nu_logdet is invariant to invertible linear reparameterisation of a layer;
    nu_trace is not. This is the concrete argument for the logdet default.
    """
    rng = np.random.default_rng(4)
    Sigma, blocks = _four_layer_sigma()
    d = blocks.dim('L2')

    nu_ld_before = novelty_logdet(Sigma, blocks, 'L2', ['L3'])
    nu_tr_before = novelty_trace(Sigma, blocks, 'L2', ['L3'])

    # Reparameterise layer L2 by an invertible M: Sigma -> T Sigma T^T
    M = rng.standard_normal((d, d))
    while abs(np.linalg.det(M)) < 1e-3:
        M = rng.standard_normal((d, d))
    T = np.eye(blocks.D)
    c2 = blocks.cols('L2')
    T[np.ix_(c2, c2)] = M
    Sigma_t = T @ Sigma @ T.T

    nu_ld_after = novelty_logdet(Sigma_t, blocks, 'L2', ['L3'])
    nu_tr_after = novelty_trace(Sigma_t, blocks, 'L2', ['L3'])

    assert abs(nu_ld_before - nu_ld_after) < 1e-8, (
        f'logdet novelty not invariant: {nu_ld_before} -> {nu_ld_after}')
    assert abs(nu_tr_before - nu_tr_after) > 1e-3, (
        f'trace novelty unexpectedly invariant: {nu_tr_before} -> {nu_tr_after}')
    print(f'  nu_logdet {nu_ld_before:.6f} -> {nu_ld_after:.6f} (invariant)')
    print(f'  nu_trace  {nu_tr_before:.6f} -> {nu_tr_after:.6f} (NOT invariant)')


def test_redundancy_detected():
    """A near-copy of the anchor scores low novelty; an independent layer scores ~1."""
    Sigma, blocks = _four_layer_sigma()
    nu_redundant = novelty_logdet(Sigma, blocks, 'L2', ['L3'])
    nu_complement = novelty_logdet(Sigma, blocks, 'L1', ['L3'])
    assert nu_redundant < 0.15, f'redundant layer scored {nu_redundant}'
    assert nu_complement > 0.99, f'independent layer scored {nu_complement}'
    rho = canonical_correlations(Sigma, blocks, 'L2', ['L3'])
    print(f'  nu(L2|L3)={nu_redundant:.4f} (max rho={rho.max():.4f}), '
          f'nu(L1|L3)={nu_complement:.4f}')


def test_per_block_shrinkage_breaks_psd():
    """
    Shrinking each layer block separately (leaving cross-blocks unshrunk) can make
    the conditional covariance indefinite; shrinking the joint matrix once cannot.
    """
    d = 2
    S_ll = np.diag([10.0, 0.1])
    S_ss = np.diag([10.0, 0.1])
    S_ls = np.diag([9.9, 0.09])
    Sigma = np.block([[S_ll, S_ls], [S_ls.T, S_ss]])
    blocks = LayerBlocks(['l', 's'], [d, d])
    assert np.linalg.eigvalsh(Sigma).min() > 0, 'test fixture must be PD'

    # (a) unshrunk: PSD
    cond0 = conditional_covariance(Sigma, blocks, 'l', ['s'])
    assert np.linalg.eigvalsh(cond0).min() > 0

    # (b) naive per-block shrinkage: diagonal blocks shrunk, cross-block left alone
    gamma = 0.5
    bad = Sigma.copy()
    for name in ('l', 's'):
        c = blocks.cols(name)
        blk = Sigma[np.ix_(c, c)]
        bad[np.ix_(c, c)] = shrink(blk, gamma, float(np.trace(blk) / d))
    cond_bad = conditional_covariance(bad, blocks, 'l', ['s'])
    min_eig_bad = float(np.linalg.eigvalsh(cond_bad).min())
    assert min_eig_bad < 0, (
        'expected per-block shrinkage to break PSD in this fixture; '
        f'min eig = {min_eig_bad}')

    # (c) joint shrinkage: still PSD
    good = shrink(Sigma, gamma, float(np.trace(Sigma) / (2 * d)))
    cond_good = conditional_covariance(good, blocks, 'l', ['s'])
    min_eig_good = float(np.linalg.eigvalsh(cond_good).min())
    assert min_eig_good > 0, f'joint shrinkage should stay PSD, got {min_eig_good}'

    print(f'  per-block shrinkage -> min eig {min_eig_bad:+.4f} (INDEFINITE)')
    print(f'  joint shrinkage     -> min eig {min_eig_good:+.4f} (PSD)')


def test_selector_avoids_redundant_layer():
    """
    The paper's K=3 pathology in miniature: the entropy-drop-only rule selects the
    redundant layer L2; the LSF relevance-minus-redundancy rule selects L1.
    """
    Sigma, blocks = _four_layer_sigma()
    jc = _as_jc(Sigma, blocks)

    rho, delta = entropy_density_drops(jc)
    # Precondition: the pathology exists -- largest drop is at the redundant layer.
    assert int(np.argmax(delta)) == blocks.index('L2'), (
        f'fixture broken: drops={delta}, expected argmax at L2')

    # relevance_only reproduces the published MM++ rule: no novelty floor, so it
    # is free to walk into the redundant layer (which is the point of the test).
    old = select_layers(jc, K_max=2, combine='relevance_only', verbose=False)
    assert old.params['novelty_floor_applied'] is False, (
        'baseline must not inherit the novelty floor')
    new = select_layers(jc, K_max=2, combine='additive', verbose=False)

    assert old.selected == ['L2', 'L3'], f'relevance-only picked {old.selected}'
    assert new.selected == ['L1', 'L3'], f'LSF picked {new.selected}'
    nu_old = novelty_logdet(Sigma, blocks, 'L2', ['L3'])
    nu_new = novelty_logdet(Sigma, blocks, 'L1', ['L3'])
    print(f'  entropy-drop only : {old.selected}   (redundant,     nu={nu_old:.4f})')
    print(f'  LSF additive      : {new.selected}   (complementary, nu={nu_new:.4f})')


def test_adaptive_k_falls_back():
    """
    When every candidate is redundant with the anchor, greedy stops immediately and
    the detector degrades to the single anchored layer (K=1) -- the ID-only
    fallback, with no separate rule.
    """
    d = 6
    S2 = np.diag([1.0, 0.05, 0.05, 0.02, 0.02, 0.02])       # anchor, concentrated
    S0 = np.diag([1.0, 0.90, 0.85, 0.80, 0.75, 0.70])       # flat, high entropy
    c = 0.995
    D = 3 * d
    Sigma = np.zeros((D, D))
    sl = [slice(i * d, (i + 1) * d) for i in range(3)]
    Sigma[sl[0], sl[0]] = S0
    # L1 is a near-copy of the anchor L2: eligible (rho_0 > rho_1 so delta_1 > 0)
    # but carries essentially no new information.
    Sigma[sl[1], sl[1]] = c ** 2 * S2 + np.eye(d) * 1e-4
    Sigma[sl[1], sl[2]] = c * S2
    Sigma[sl[2], sl[1]] = (c * S2).T
    Sigma[sl[2], sl[2]] = S2
    blocks = LayerBlocks(['L0', 'L1', 'L2'], [d] * 3)
    jc = _as_jc(Sigma, blocks)

    # Precondition: L1 IS eligible under the relevance gate, so the floor -- not
    # the gate -- must be what stops greedy.
    _, delta = entropy_density_drops(jc)
    assert delta[blocks.index('L1')] > 0, f'fixture broken: delta={delta}'

    res = select_layers(jc, K_max=3, tau_novelty=0.10, verbose=False)
    assert res.selected == ['L2'], f'expected fallback to anchor only, got {res.selected}'
    assert res.K == 1 and not res.used_fusion
    assert res.stopped == 'novelty_floor', res.stopped

    # Without the floor the same fixture fuses the redundant layer -- i.e. the
    # floor is doing the work, not the relevance gate.
    res_nofloor = select_layers(jc, K_max=2, tau_novelty=None, verbose=False)
    assert res_nofloor.selected == ['L1', 'L2'], res_nofloor.selected
    print(f'  eligible but redundant (delta={delta[1]:.4f}, '
          f'nu={res.steps[0]["candidates"][0]["nu"]:.4f}) -> selected {res.selected}, '
          f'stop={res.stopped}; without floor: {res_nofloor.selected}')


def test_end_to_end_from_samples():
    """
    Sampling path: draw class-conditional data with the known covariance, estimate
    the joint covariance through the streaming accumulator, and confirm the
    selection is unchanged by estimation + Ledoit-Wolf shrinkage.
    """
    rng = np.random.default_rng(7)
    Sigma, blocks = _four_layer_sigma()
    D = blocks.D
    C, per_class = 20, 400

    A = np.linalg.cholesky(Sigma + np.eye(D) * 1e-10)
    class_means = rng.standard_normal((C, D)) * 0.3
    labels = np.repeat(np.arange(C), per_class)
    Z = rng.standard_normal((len(labels), D)) @ A.T
    X = class_means[labels] + Z

    jc = joint_covariance_from_features(X, labels, class_means, blocks)
    err = np.abs(jc.emp_cov - Sigma).max()
    assert err < 0.05, f'covariance recovery error {err}'

    res = select_layers(jc, K_max=2, combine='additive', verbose=False)
    assert res.selected == ['L1', 'L3'], f'got {res.selected}'
    print(f'  N={jc.n_samples}, max|Sigma_hat - Sigma|={err:.4f}, '
          f'gamma={jc.shrinkage:.6f}, selected={res.selected}')


def test_selection_result_serializable():
    """SelectionResult.to_dict() is JSON-serialisable (results must be versionable)."""
    import json
    Sigma, blocks = _four_layer_sigma()
    res = select_layers(_as_jc(Sigma, blocks), K_max=3, verbose=False)
    s = json.dumps(res.to_dict())
    assert len(s) > 100 and '"selected"' in s
    print(f'  to_dict() -> {len(s)} bytes of JSON, K={res.K}, stop={res.stopped}')


# ────────────────────────────────── runner ───────────────────────────────────

def main():
    # numpy 2.x on macOS/Accelerate emits spurious "overflow/divide-by-zero in
    # matmul" RuntimeWarnings on perfectly finite inputs (verified: results are
    # exact and finite). Silence them in the runner only -- never in lsf/ -- so a
    # genuine numerical problem in library code still surfaces.
    import warnings
    warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)

    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = []
    for t in tests:
        print(f'\n{t.__name__}:')
        try:
            t()
            print('  PASS')
        except Exception as exc:  # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f'  FAIL: {type(exc).__name__}: {exc}')
    print(f'\n{"=" * 70}')
    print(f'{len(tests) - len(failed)}/{len(tests)} passed')
    for name, exc in failed:
        print(f'  FAILED {name}: {exc}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
