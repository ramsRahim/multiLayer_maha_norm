"""
LSF layer selector: relevance (entropy-density drop) + redundancy (conditional novelty).

THE CRITERION
    The NeurIPS version of MM++ ranked candidate layers by the entropy-density
    drop alone,

        Delta_l = rho_{l-1} - rho_l,     rho_l = H_l / D_l   [nats per dimension]

    which is a RELEVANCE signal: it marks where the network compresses semantically.
    Its failure mode is documented in the paper's own K-sensitivity curve: at K=3
    on ViT-B/16 it picks block_11, which carries almost the same information as the
    anchored penultimate layer (canonical correlation 0.976), so the joint
    covariance grows in dimension without gaining information and AUROC drops.

    LSF adds a REDUNDANCY signal, the conditional novelty of a candidate given the
    already-selected set (novelty.py):

        log nu(l|S) = (1/D_l) * sum_i log(1 - rho_i^2)      [nats per dimension]

    Both terms are differential-entropy densities in nats per dimension, so they
    are commensurable and the criterion is their SUM, not their product:

        q(l|S) = Delta_l + lambda * log nu(l|S),     subject to Delta_l > 0

    This is the classical relevance-minus-redundancy (mRMR / forward-stepwise)
    form under a Gaussian model, with both terms in matched physical units.
    lambda = 1.0 is the natural default (equal weighting of one nat of compression
    against one nat of redundancy); lambda is exposed for sensitivity analysis.

    `combine='multiplicative'` reproduces the earlier draft's q = max(Delta,0)*nu
    for the ablation table, and `combine='relevance_only'` reproduces the NeurIPS
    MM++ selector. Reporting all three in one table is the point: it shows the
    contribution comes from the redundancy term, not from re-tuning.

ADAPTIVE K (SUBSUMES THE SEPARATE FALLBACK RULE)
    Greedy selection stops when the best remaining candidate carries less novelty
    than a floor tau: no candidate is worth its dimensions. When that triggers on
    the very first step, the selected set is just the anchored final layer -- i.e.
    the detector degrades gracefully to single-layer Mahalanobis++ on backbones
    whose intermediate layers are redundant (the DeiT3 / SigLIP behaviour in the
    paper's checkpoint sweep). One mechanism gives both the layer count and the
    fusion/no-fusion decision, and tau is calibrated from ID statistics only.

    IMPORTANT (stated here so it survives into the paper): tau must be frozen on a
    development subset of checkpoints and validated leave-one-architecture-family-out.
    Choosing tau by looking at which checkpoints it rescues on OOD test data would
    make the rule circular, and reviewers will check this.
"""

import numpy as np

from .novelty import novelty_logdet, novelty_trace

__all__ = [
    'entropy_density',
    'entropy_density_drops',
    'SelectionResult',
    'select_layers',
]


def entropy_density(Sigma_ll, eps=1e-8):
    """
    rho_l = H_l / D_l, the within-class covariance entropy per dimension.

    H_l = -sum_i lambda_bar_i * ln(lambda_bar_i) over the normalised eigenspectrum
    of the (shrunk) within-class covariance -- Eq. (4) of the paper, computed here
    directly from a covariance block rather than from a cached precision matrix.
    """
    Sigma_ll = np.asarray(Sigma_ll, dtype=np.float64)
    D = Sigma_ll.shape[0]
    eig = np.linalg.eigvalsh(0.5 * (Sigma_ll + Sigma_ll.T))
    eig = np.clip(eig, eps, None)
    lam_bar = eig / eig.sum()
    H = -float(np.sum(lam_bar * np.log(lam_bar)))
    return H / D


def entropy_density_drops(jc, layers=None):
    """
    Entropy densities and consecutive drops for the candidate layers, in depth order.

    Returns:
        (rho, delta) where rho[i] is the entropy density of layer i and
        delta[i] = rho[i-1] - rho[i] with delta[0] = 0.0 (no predecessor).
    """
    names = list(layers) if layers is not None else list(jc.blocks.names)
    rho = np.array([entropy_density(jc.block(n)) for n in names], dtype=np.float64)
    delta = np.zeros_like(rho)
    delta[1:] = rho[:-1] - rho[1:]
    return rho, delta


class SelectionResult:
    """
    Outcome of a selection run, including the full decision trace.

    Attributes:
        selected:  layer names in depth order (always includes the anchor).
        steps:     per-step records with every candidate's relevance, novelty and
                   combined score -- this is what the paper's selection figure and
                   the auditability story are built from.
        stopped:   why greedy stopped ('K_max', 'novelty_floor', 'no_candidates').
        rho, delta: entropy densities and drops for all candidates.
    """

    def __init__(self, selected, steps, stopped, rho, delta, names, params):
        self.selected = selected
        self.steps = steps
        self.stopped = stopped
        self.rho = rho
        self.delta = delta
        self.names = names
        self.params = params

    @property
    def K(self):
        return len(self.selected)

    @property
    def used_fusion(self):
        return len(self.selected) > 1

    def to_dict(self):
        return {
            'selected': list(self.selected),
            'K': self.K,
            'used_fusion': self.used_fusion,
            'stopped': self.stopped,
            'params': dict(self.params),
            'entropy_density': {n: float(r) for n, r in zip(self.names, self.rho)},
            'entropy_density_drop': {n: float(d) for n, d in zip(self.names, self.delta)},
            'steps': self.steps,
        }

    def summary(self):
        lines = [
            f'[LSF] selected {self.selected}  (K={self.K}, stop={self.stopped})',
        ]
        for st in self.steps:
            if st['chosen'] is None:
                lines.append(f"[LSF]   step {st['step']}: no layer added ({st['reason']})")
                continue
            lines.append(
                f"[LSF]   step {st['step']}: + {st['chosen']}  "
                f"q={st['q']:.5f}  Delta={st['delta']:.5f}  nu={st['nu']:.4f}"
            )
        return '\n'.join(lines)

    def __repr__(self):
        return f'SelectionResult(selected={self.selected}, stop={self.stopped})'


def select_layers(jc, K_max=3, anchor='last', tau_novelty=0.10, lam=1.0,
                  combine='additive', novelty_kind='logdet',
                  require_positive_delta=True, verbose=True):
    """
    Greedy relevance-minus-redundancy layer selection with adaptive K.

    Args:
        jc:          JointCovariance over all candidate layers.
        K_max:       maximum layers to select, including the anchor.
        anchor:      'last' (penultimate/final feature layer, the paper's anchor),
                     a layer name, or None for no anchoring.
        tau_novelty: novelty floor in (0, 1], or None to disable. Stop when the
                     best candidate's nu falls below it. nu is a geometric mean of
                     (1 - rho_i^2), so tau=0.10 means "a candidate must retain at
                     least 10% of its volume after projecting out the selected set".
                     IGNORED when combine='relevance_only': that mode exists to
                     reproduce the published MM++ selector as a baseline, and a
                     baseline must not inherit the new method's stopping rule, or
                     the ablation table silently overstates the old rule.
        lam:         weight on the redundancy term (additive mode).
        combine:     'additive' (default, q = Delta + lam*log nu),
                     'multiplicative' (q = max(Delta,0) * nu, earlier draft),
                     'relevance_only' (q = Delta, the NeurIPS MM++ selector).
        novelty_kind: 'logdet' (default) or 'trace' (ablation).
        require_positive_delta: keep the paper's Delta_l > 0 eligibility gate.

    Returns:
        SelectionResult
    """
    if combine not in ('additive', 'multiplicative', 'relevance_only'):
        raise ValueError(f'unknown combine={combine!r}')
    if tau_novelty is not None and not (0.0 < tau_novelty <= 1.0):
        raise ValueError(f'tau_novelty must be in (0, 1] or None, got {tau_novelty}')
    # The baseline reproduction must not inherit the new stopping rule.
    apply_floor = tau_novelty is not None and combine != 'relevance_only'

    names = list(jc.blocks.names)
    L = len(names)
    rho, delta = entropy_density_drops(jc)

    if anchor is None:
        selected = []
    elif anchor == 'last':
        selected = [names[-1]]
    else:
        if anchor not in names:
            raise ValueError(f'anchor {anchor!r} not among candidates {names}')
        selected = [anchor]

    nov_fn = novelty_logdet if novelty_kind == 'logdet' else novelty_trace
    steps = []
    stopped = 'K_max'

    while len(selected) < K_max:
        sel_pos = [names.index(s) for s in selected]
        cands = [p for p in range(L) if p not in sel_pos]
        if require_positive_delta:
            cands = [p for p in cands if delta[p] > 0]
        if not cands:
            stopped = 'no_candidates'
            steps.append({'step': len(steps) + 1, 'chosen': None,
                          'reason': 'no candidate with positive entropy-density drop',
                          'candidates': []})
            break

        records = []
        for p in cands:
            nu = float(nov_fn(jc.Sigma, jc.blocks, p, sel_pos))
            d = float(delta[p])
            if combine == 'additive':
                q = d + lam * float(np.log(max(nu, 1e-300)))
            elif combine == 'multiplicative':
                q = max(d, 0.0) * nu
            else:
                q = d
            records.append({'layer': names[p], 'delta': d, 'nu': nu, 'q': q})

        records.sort(key=lambda r: r['q'], reverse=True)
        best = records[0]

        # Adaptive-K / fallback: the best available candidate is too redundant.
        if apply_floor and best['nu'] < tau_novelty:
            stopped = 'novelty_floor'
            steps.append({
                'step': len(steps) + 1, 'chosen': None,
                'reason': (f"best candidate {best['layer']} has nu={best['nu']:.4f} "
                           f"< tau={tau_novelty}"),
                'candidates': records,
            })
            break

        selected.append(best['layer'])
        steps.append({
            'step': len(steps) + 1,
            'chosen': best['layer'],
            'q': best['q'], 'delta': best['delta'], 'nu': best['nu'],
            'reason': 'highest relevance-minus-redundancy score',
            'candidates': records,
        })

    # Return in depth order so the concatenation layout is deterministic.
    selected = [n for n in names if n in set(selected)]
    params = {
        'K_max': K_max, 'anchor': anchor, 'tau_novelty': tau_novelty, 'lam': lam,
        'novelty_floor_applied': apply_floor,
        'combine': combine, 'novelty_kind': novelty_kind,
        'require_positive_delta': require_positive_delta,
        'shrinkage': jc.shrinkage, 'n_samples': jc.n_samples,
    }
    res = SelectionResult(selected, steps, stopped, rho, delta, names, params)
    if verbose:
        print(res.summary())
    return res
