#!/usr/bin/env python3
"""
Exhaustive layer-subset sweep + selector regret, and the P2 scoring comparison.

THE EXPERIMENT
    For every layer subset of size <= K_max (66 pairs + 220 triples on a 12-block
    ViT), score ID-val against each OOD set and record AUROC / FPR95. That gives:

      * the ORACLE ceiling -- how much headroom layer fusion has at all;
      * the REGRET of each selector (LSF, entropy-drop-only, random, penultimate-
        only) against that ceiling, which is the honest way to argue a selector is
        good without tuning it on OOD data;
      * the ranking correlation between the ID-only novelty score and actual OOD
        gain, i.e. direct evidence that the selection signal is predictive.

    This is affordable because the joint covariance built by run_selection.py
    already contains every subset's fused covariance, and the L x L accumulator
    lets each subset recover the exact Ledoit-Wolf intensity it would have been
    fitted with. No training features are re-read: only ID-val and OOD features,
    once each.

    --score_modes additionally runs the P2 comparison on the SAME subsets:
        joint          -min_c r^T P r          (full precision, MM++ as published)
        block_diag     -min_c sum_l d_{c,l}    (cross-blocks zeroed, class shared)
        additive_min   sum_l -min_c d_{c,l}    (per-layer argmin, Lee et al. style)
        additive_min_z as above, per-layer z-scored on ID val
    'block_diag' vs 'additive_min' isolates the shared-class-hypothesis effect,
    which is the comparison missing from the NeurIPS submission.

OUTPUT (--out_dir, default iclr2027/results)
    oracle_sweep_<model>[_<tag>].csv    one row per (subset, ood set, score mode)
    oracle_sweep_<model>[_<tag>].json   summary: oracle, per-selector regret, args

Typical use:

    conda run -n mm_plus_plus python iclr2027/experiments/run_oracle_sweep.py \
        --model_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --path_to_cache ./cache_imagenetlt --tag imagenetlt \
        --K_max 2 --score_modes joint block_diag additive_min additive_min_z

Run run_selection.py FIRST -- this script reuses its joint_cov_*.npz.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsf import (  # noqa: E402
    CacheLayout,
    JointCovariance,
    auroc,
    fpr_at_tpr,
    load_class_means,
    load_layer_features,
    novelty_logdet,
    score_subset,
    select_layers,
)
from experiments.run_selection import VARIANTS, resolve_joint_path  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_name', required=True)
    p.add_argument('--path_to_cache', default='./cache')
    p.add_argument('--tag', default='')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--joint_cov', default=None,
                   help='path to joint_cov_*.npz (default: derived from model/tag)')
    p.add_argument('--ood', nargs='*', default=None,
                   help='OOD dataset dir names under cache_ood_inter/<model>/ '
                        '(default: all available)')
    p.add_argument('--K_max', type=int, default=2,
                   help='max subset size INCLUDING the anchor. 2 = all pairs; '
                        '3 adds all triples (slower).')
    p.add_argument('--anchor_only', action='store_true',
                   help='only sweep subsets that contain the final layer (matches '
                        'the anchored design; much cheaper)')
    p.add_argument('--score_modes', nargs='*',
                   default=['joint'],
                   choices=['joint', 'block_diag', 'additive_min', 'additive_min_z'],
                   help='scoring rules to evaluate (P2 comparison)')
    p.add_argument('--tau_novelty', type=float, default=0.10)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--max_subsets', type=int, default=0,
                   help='cap the number of subsets (0 = no cap). Any truncation is '
                        'recorded in the JSON so it is never silent.')
    p.add_argument('--seed', type=int, default=0)
    return p


def enumerate_subsets(names, K_max, anchor_only):
    """All subsets of size 2..K_max, in depth order, optionally anchored."""
    anchor = names[-1]
    subsets = []
    for k in range(2, K_max + 1):
        for combo in itertools.combinations(names, k):
            if anchor_only and anchor not in combo:
                continue
            subsets.append(list(combo))
    return subsets


def main():
    args = build_parser().parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(here, 'results')
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    layout = CacheLayout(args.path_to_cache, args.model_name)
    joint_path = args.joint_cov or resolve_joint_path(out_dir, args.model_name, args.tag)
    if not os.path.exists(joint_path):
        print(f'[oracle] {joint_path} not found. Run run_selection.py first.')
        return 1
    jc = JointCovariance.load(joint_path)
    names = jc.blocks.names
    anchor = names[-1]
    print(f'[oracle] {jc}')
    if jc.M is None:
        print('[oracle] WARNING: joint covariance has no M accumulator; subsets will '
              'reuse the joint shrinkage instead of their own exact Ledoit-Wolf. '
              'Rebuild with --rebuild in run_selection.py for exact per-subset fits.')

    ood_sets = args.ood if args.ood else layout.available_ood()
    if not ood_sets:
        print(f'[oracle] no OOD caches under {layout.ood_inter_root}')
        return 1
    print(f'[oracle] OOD sets: {ood_sets}')

    print('[oracle] loading ID val features...')
    val_feats, _ = load_layer_features(layout.val_inter, layer_names=names)
    means_list = load_class_means(layout, names)
    n_val = len(val_feats[0])
    print(f'[oracle] {n_val} ID val samples, {len(names)} layers')

    subsets = enumerate_subsets(names, args.K_max, args.anchor_only)
    truncated = 0
    if args.max_subsets and len(subsets) > args.max_subsets:
        truncated = len(subsets) - args.max_subsets
        subsets = subsets[:args.max_subsets]
        print(f'[oracle] NOTE: capped at {args.max_subsets} subsets, '
              f'{truncated} DROPPED (recorded in the JSON).')
    # Single-layer anchor baseline is always included as subset of size 1.
    all_subsets = [[anchor]] + subsets
    print(f'[oracle] {len(all_subsets)} subsets x {len(ood_sets)} OOD sets '
          f'x {len(args.score_modes)} score modes')

    # Pre-slice each subset's covariance once (cheap, reused across OOD sets).
    jc_subs = {tuple(s): jc.subset(s) for s in all_subsets}
    nu_of = {}
    for s in all_subsets:
        if len(s) == 1:
            nu_of[tuple(s)] = float('nan')
        else:
            others = [x for x in s if x != anchor] or s[:-1]
            nu_of[tuple(s)] = float(novelty_logdet(jc.Sigma, jc.blocks, others[0],
                                                   [x for x in s if x != others[0]]))

    rows = []
    for ood_name in ood_sets:
        ood_dir = layout.ood_inter(ood_name)
        if not os.path.isdir(ood_dir):
            print(f'[oracle] skipping {ood_name}: {ood_dir} missing')
            continue
        print(f'[oracle] loading OOD features: {ood_name}')
        try:
            ood_feats, _ = load_layer_features(ood_dir, layer_names=names)
        except Exception as exc:  # noqa: BLE001
            print(f'[oracle] skipping {ood_name}: {exc}')
            continue

        for si, subset in enumerate(all_subsets):
            jsub = jc_subs[tuple(subset)]
            single = len(subset) == 1
            # For a single layer every scoring rule is the same quadratic form, so
            # compute once and record it under each mode -- that keeps the
            # single-layer baseline present in every mode's summary, which is the
            # reference point for "does fusion help at all".
            cached = None
            for mode in args.score_modes:
                if single:
                    if cached is None:
                        cached = score_subset(jsub, subset, names, means_list,
                                              val_feats, ood_feats, mode='joint')
                    s_id, s_ood = cached
                else:
                    s_id, s_ood = score_subset(jsub, subset, names, means_list,
                                               val_feats, ood_feats, mode=mode)
                rows.append({
                    'model': args.model_name,
                    'ood': ood_name,
                    'subset': '+'.join(subset),
                    'K': len(subset),
                    'score_mode': mode,
                    'auroc': 100.0 * auroc(s_id, s_ood),
                    'fpr95': 100.0 * fpr_at_tpr(s_id, s_ood, 0.95),
                    'nu_logdet': nu_of[tuple(subset)],
                    'shrinkage': jsub.shrinkage,
                    'dim': jsub.blocks.D,
                })
            if (si + 1) % 25 == 0:
                print(f'[oracle]   {ood_name}: {si + 1}/{len(all_subsets)} subsets')

    if not rows:
        print('[oracle] no results produced')
        return 1

    suffix = f'_{args.tag}' if args.tag else ''
    csv_path = os.path.join(out_dir, f'oracle_sweep_{args.model_name}{suffix}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[oracle] wrote {csv_path}  ({len(rows)} rows)')

    # ── mean AUROC per (subset, mode) across OOD sets -> oracle + regret ──────
    summary = {}
    for mode in args.score_modes:
        per_subset = {}
        for r in rows:
            if r['score_mode'] != mode:
                continue
            per_subset.setdefault(r['subset'], []).append(r['auroc'])
        if not per_subset:
            continue
        mean_auroc = {k: float(np.mean(v)) for k, v in per_subset.items()}
        best = max(mean_auroc, key=mean_auroc.get)

        selectors = {}
        for vname, kw in VARIANTS:
            kwargs = dict(K_max=args.K_max, tau_novelty=args.tau_novelty, lam=args.lam)
            kwargs.update(kw)
            try:
                sel = select_layers(jc, verbose=False, **kwargs)
            except Exception as exc:  # noqa: BLE001
                selectors[vname] = {'error': str(exc)}
                continue
            key = '+'.join(sel.selected)
            selectors[vname] = {
                'selected': sel.selected,
                'K': sel.K,
                'mean_auroc': mean_auroc.get(key),
                'regret_vs_oracle': (None if key not in mean_auroc
                                     else float(mean_auroc[best] - mean_auroc[key])),
                'in_sweep': key in mean_auroc,
            }

        rand_keys = list(mean_auroc.keys())
        rand_sample = [mean_auroc[rand_keys[i]]
                       for i in rng.choice(len(rand_keys), min(200, len(rand_keys)),
                                           replace=True)]
        summary[mode] = {
            'oracle_subset': best,
            'oracle_mean_auroc': mean_auroc[best],
            'anchor_only_mean_auroc': mean_auroc.get(anchor),
            'random_subset_mean_auroc': float(np.mean(rand_sample)),
            'n_subsets': len(mean_auroc),
            'selectors': selectors,
            'mean_auroc_by_subset': mean_auroc,
        }

    payload = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': args.model_name,
        'tag': args.tag,
        'joint_cov_file': os.path.basename(joint_path),
        'ood_sets': ood_sets,
        'anchor': anchor,
        'n_subsets_evaluated': len(all_subsets),
        'n_subsets_dropped_by_cap': truncated,
        'exact_per_subset_shrinkage': jc.M is not None,
        'summary': summary,
        'args': vars(args),
    }
    json_path = os.path.join(out_dir, f'oracle_sweep_{args.model_name}{suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[oracle] wrote {json_path}')

    for mode, s in summary.items():
        print(f'\n[oracle] === {mode} ===')
        print(f'  oracle       {s["oracle_subset"]}  {s["oracle_mean_auroc"]:.2f}')
        print(f'  anchor only  {anchor}  {s["anchor_only_mean_auroc"]:.2f}'
              if s['anchor_only_mean_auroc'] is not None else '  anchor only  n/a')
        print(f'  random mean  {s["random_subset_mean_auroc"]:.2f}')
        for vname, d in s['selectors'].items():
            if 'error' in d:
                print(f'  {vname:<22} ERROR {d["error"]}')
            elif d['mean_auroc'] is None:
                print(f'  {vname:<22} {"+".join(d["selected"]):<28} '
                      f'(not in sweep; K={d["K"]})')
            else:
                print(f'  {vname:<22} {"+".join(d["selected"]):<28} '
                      f'{d["mean_auroc"]:.2f}  regret {d["regret_vs_oracle"]:+.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
