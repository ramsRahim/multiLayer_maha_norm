#!/usr/bin/env python3
"""
Build the joint covariance for a backbone and run every selector variant on it.

This is step 1 of the ICLR revision: it produces the layer sets that the scoring
experiments consume, plus the selection trace behind the paper's selection figure.
It does NOT score OOD data -- use run_oracle_sweep.py for that.

What it writes to --out_dir (default iclr2027/results):
    joint_cov_<model>[_<tag>].npz     the joint covariance + accumulators (cacheable,
                                      reused by run_oracle_sweep.py)
    selection_<model>[_<tag>].json    every variant's selected layers, the full
                                      greedy trace, entropy densities, novelty
                                      matrix and pairwise canonical correlations

Typical use (on the machine that holds the caches):

    conda run -n mm_plus_plus python iclr2027/experiments/run_selection.py \
        --model_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --path_to_cache ./cache_imagenetlt \
        --tag imagenetlt

Prerequisite: the per-layer caches must exist, i.e. evaluate.py must have been run
once with an MM_plus_plus method for this model/cache. Run with --preflight_only
first to check.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsf import (  # noqa: E402
    CacheLayout,
    JointCovariance,
    build_joint_covariance,
    canonical_correlations,
    entropy_density_drops,
    load_train_labels,
    novelty_logdet,
    novelty_trace,
    preflight,
    select_layers,
)

# (name, kwargs) -- every row of the selector ablation table.
VARIANTS = [
    ('mmpp_neurips',      dict(combine='relevance_only')),
    ('lsf_additive',      dict(combine='additive')),
    ('lsf_multiplicative', dict(combine='multiplicative')),
    ('lsf_additive_trace', dict(combine='additive', novelty_kind='trace')),
    ('lsf_no_floor',      dict(combine='additive', tau_novelty=None)),
    ('lsf_no_anchor',     dict(combine='additive', anchor=None)),
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_name', required=True,
                   help='timm model key, as passed to evaluate.py')
    p.add_argument('--path_to_cache', default='./cache',
                   help='same --path_to_cache used with evaluate.py')
    p.add_argument('--tag', default='',
                   help="suffix for output filenames, e.g. 'imagenetlt'")
    p.add_argument('--out_dir', default=None,
                   help='default: iclr2027/results')
    p.add_argument('--layers', nargs='*', default=None,
                   help='restrict candidate layers (default: all)')
    p.add_argument('--max_samples', type=int, default=300000,
                   help='cap on training samples for the covariance fit '
                        '(0 = use all). 300k matches detection_methods.py.')
    p.add_argument('--K_max', type=int, default=3)
    p.add_argument('--tau_novelty', type=float, default=0.10)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rebuild', action='store_true',
                   help='recompute the joint covariance even if the .npz exists')
    p.add_argument('--preflight_only', action='store_true',
                   help='validate the cache layout and exit')
    return p


def resolve_joint_path(out_dir, model, tag):
    suffix = f'_{tag}' if tag else ''
    return os.path.join(out_dir, f'joint_cov_{model}{suffix}.npz')


def main():
    args = build_parser().parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(here, 'results')
    os.makedirs(out_dir, exist_ok=True)

    layout = CacheLayout(args.path_to_cache, args.model_name)
    print(layout.describe())
    print()
    checks = preflight(layout)
    if args.preflight_only:
        return 0 if checks['ok'] else 1
    if not checks['ok']:
        print('\n[run_selection] aborting: preflight failed. Fix the cache first.')
        return 1

    joint_path = resolve_joint_path(out_dir, args.model_name, args.tag)
    if os.path.exists(joint_path) and not args.rebuild:
        print(f'[run_selection] loading cached joint covariance {joint_path}')
        jc = JointCovariance.load(joint_path)
    else:
        labels = load_train_labels(layout)
        jc = build_joint_covariance(
            train_inter_path=layout.train_inter,
            cache_path=layout.methods,
            train_labels=labels,
            layer_subset=args.layers,
            max_samples=(args.max_samples or None),
            seed=args.seed,
            out_path=joint_path,
        )
    print(f'[run_selection] {jc}')
    names = jc.blocks.names

    rho, delta = entropy_density_drops(jc)
    print(f'[run_selection] entropy density : {np.round(rho, 5).tolist()}')
    print(f'[run_selection] density drops   : {np.round(delta, 5).tolist()}')

    # Diagnostics vs the anchored final layer -- this is the table that explains
    # WHY a layer is or is not selected (the CCA-0.976 story, generalised).
    anchor = names[-1]
    diagnostics = {}
    for n in names[:-1]:
        rho_cca = canonical_correlations(jc.Sigma, jc.blocks, n, [anchor])
        diagnostics[n] = {
            'entropy_density': float(rho[names.index(n)]),
            'entropy_density_drop': float(delta[names.index(n)]),
            'nu_logdet_given_anchor': float(novelty_logdet(jc.Sigma, jc.blocks, n, [anchor])),
            'nu_trace_given_anchor': float(novelty_trace(jc.Sigma, jc.blocks, n, [anchor])),
            'max_canonical_corr': float(rho_cca.max()),
            'mean_canonical_corr': float(rho_cca.mean()),
        }

    print(f'\n[run_selection] novelty w.r.t. anchor {anchor!r}:')
    print(f'  {"layer":<24} {"Delta":>9} {"nu_logdet":>10} {"nu_trace":>9} {"max rho":>8}')
    for n, d in diagnostics.items():
        print(f'  {n:<24} {d["entropy_density_drop"]:>9.5f} '
              f'{d["nu_logdet_given_anchor"]:>10.4f} {d["nu_trace_given_anchor"]:>9.4f} '
              f'{d["max_canonical_corr"]:>8.4f}')

    results = {}
    print()
    for vname, kw in VARIANTS:
        kwargs = dict(K_max=args.K_max, tau_novelty=args.tau_novelty, lam=args.lam)
        kwargs.update(kw)
        print(f'[run_selection] --- {vname} ---')
        res = select_layers(jc, verbose=True, **kwargs)
        results[vname] = res.to_dict()

    payload = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': args.model_name,
        'path_to_cache': args.path_to_cache,
        'tag': args.tag,
        'joint_cov_file': os.path.basename(joint_path),
        'n_samples': jc.n_samples,
        'joint_shrinkage': jc.shrinkage,
        'joint_dim': jc.blocks.D,
        'layers': names,
        'layer_dims': jc.blocks.dims,
        'anchor': anchor,
        'condition_number_all': jc.condition_number(),
        'diagnostics_vs_anchor': diagnostics,
        'selection': results,
        'preflight': {k: v for k, v in checks.items() if k != 'shard_order'},
        'shard_order_aligned': checks.get('shard_order', {}).get('aligned'),
        'args': vars(args),
    }
    suffix = f'_{args.tag}' if args.tag else ''
    out_json = os.path.join(out_dir, f'selection_{args.model_name}{suffix}.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[run_selection] wrote {out_json}')
    print('[run_selection] commit this file -- it is the audit trail for the paper.')

    print('\n[run_selection] summary:')
    for vname, r in results.items():
        print(f'  {vname:<22} K={r["K"]}  {r["selected"]}  (stop={r["stopped"]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
