#!/usr/bin/env python3
"""
End-to-end integration test: fake cache -> run_selection -> run_oracle_sweep.

Validates the PLUMBING (cache discovery, preflight, streaming covariance, subset
shrinkage, all four scoring rules, CSV/JSON outputs) without ImageNet or a GPU.
It does NOT validate that layer fusion helps -- the synthetic OOD shift is
arbitrary. Method-level correctness lives in test_novelty.py.

Run:  python3 iclr2027/tests/test_pipeline.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _run(cmd, label):
    print(f'  $ {label}')
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise AssertionError(f'{label} exited {p.returncode}')
    return p.stdout


def main():
    tmp = tempfile.mkdtemp(prefix='lsf_pipeline_')
    cache = os.path.join(tmp, 'cache')
    results = os.path.join(tmp, 'results')
    py = sys.executable
    failures = []
    try:
        print('test_pipeline:')
        _run([py, os.path.join(HERE, 'make_fake_cache.py'), '--out', cache],
             'make_fake_cache.py')

        out = _run([py, os.path.join(ROOT, 'experiments', 'run_selection.py'),
                    '--model_name', 'fake_model', '--path_to_cache', cache,
                    '--out_dir', results, '--tag', 'test',
                    '--max_samples', '0', '--K_max', '2'], 'run_selection.py')

        # preflight must flag the deliberate 12-slice shard-order divergence
        if 'shard-order divergence' not in out:
            failures.append('preflight did not report the shard-order divergence')

        sel_path = os.path.join(results, 'selection_fake_model_test.json')
        if not os.path.exists(sel_path):
            raise AssertionError(f'missing {sel_path}')
        sel = json.load(open(sel_path))

        if sel['shard_order_aligned'] is not False:
            failures.append('shard_order_aligned should be False for the fake cache')

        diag = sel['diagnostics_vs_anchor']
        # layer_04 is a planted near-copy of the anchor: lowest novelty, highest rho
        most_redundant = min(diag, key=lambda k: diag[k]['nu_logdet_given_anchor'])
        if most_redundant != 'layer_04':
            failures.append(f'expected layer_04 most redundant, got {most_redundant}')
        if diag['layer_04']['max_canonical_corr'] < 0.8:
            failures.append(f"planted copy should have high canonical correlation, "
                            f"got {diag['layer_04']['max_canonical_corr']:.3f}")

        # ...and it also has the largest entropy-density drop, so the baseline
        # selector takes it while LSF does not. That contrast is the whole point.
        drops = {k: v['entropy_density_drop'] for k, v in diag.items()}
        if max(drops, key=drops.get) != 'layer_04':
            failures.append(f'fixture broken: largest drop at {max(drops, key=drops.get)}')

        base = sel['selection']['mmpp_neurips']['selected']
        lsf = sel['selection']['lsf_additive']['selected']
        if 'layer_04' not in base:
            failures.append(f'relevance-only should select the redundant layer, got {base}')
        if 'layer_04' in lsf:
            failures.append(f'LSF should avoid the redundant layer, got {lsf}')
        if sel['selection']['mmpp_neurips']['params']['novelty_floor_applied'] is not False:
            failures.append('baseline must not apply the novelty floor')
        print(f'  baseline selected {base}; LSF selected {lsf}')

        joint = os.path.join(results, 'joint_cov_fake_model_test.npz')
        if not os.path.exists(joint):
            raise AssertionError(f'missing {joint}')

        _run([py, os.path.join(ROOT, 'experiments', 'run_oracle_sweep.py'),
              '--model_name', 'fake_model', '--path_to_cache', cache,
              '--out_dir', results, '--tag', 'test', '--K_max', '2',
              '--score_modes', 'joint', 'block_diag', 'additive_min',
              'additive_min_z'], 'run_oracle_sweep.py')

        csv_path = os.path.join(results, 'oracle_sweep_fake_model_test.csv')
        json_path = os.path.join(results, 'oracle_sweep_fake_model_test.json')
        for p in (csv_path, json_path):
            if not os.path.exists(p):
                raise AssertionError(f'missing {p}')
        sweep = json.load(open(json_path))

        if not sweep['exact_per_subset_shrinkage']:
            failures.append('per-subset Ledoit-Wolf should be exact (M accumulator present)')

        modes = set(sweep['summary'].keys())
        expected = {'joint', 'block_diag', 'additive_min', 'additive_min_z'}
        if modes != expected:
            failures.append(f'expected modes {expected}, got {modes}')

        for mode, s in sweep['summary'].items():
            if s['anchor_only_mean_auroc'] is None:
                failures.append(f'{mode}: single-layer anchor baseline missing')
            for vname, d in s['selectors'].items():
                if d.get('regret_vs_oracle') is None:
                    failures.append(f'{mode}/{vname}: no regret computed')
                elif d['regret_vs_oracle'] < -1e-9:
                    failures.append(f'{mode}/{vname}: negative regret '
                                    f'{d["regret_vs_oracle"]} (oracle must dominate)')

        with open(csv_path) as f:
            n_rows = sum(1 for _ in f) - 1
        print(f'  sweep: {n_rows} rows, {len(modes)} score modes, '
              f'{sweep["n_subsets_evaluated"]} subsets')

        if failures:
            print('\n  FAILURES:')
            for f_ in failures:
                print(f'    - {f_}')
            return 1
        print('\n  PASS: pipeline produced selection + sweep artifacts and all '
              'invariants held')
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
