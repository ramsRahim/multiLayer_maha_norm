# LSF — ICLR 2027 revision workspace

Working folder for the ICLR 2027 revision of MM++. Nothing here modifies the
NeurIPS codebase; `lsf/` is a standalone library that reads the same feature
caches and is wired into `detection_methods.py` only when a selector is validated.

Full evaluation of the roadmap this implements:
`~/.claude/plans/this-paper-is-pending-adaptive-sparrow.md`

## The one-paragraph version

The NeurIPS selector ranks layers by entropy-density drop `Δ_l` alone — a
**relevance** signal. Its documented failure is at K=3 on ViT-B/16, where it picks
`block_11`, which carries almost the same information as the anchored penultimate
layer (canonical correlation 0.976), so dimension grows without information and
AUROC drops 81.12 → 78.58. LSF adds a **redundancy** signal — the conditional
novelty of a candidate given the already-selected set — and combines the two in
matched units:

```
q(l|S) = Δ_l + λ · log ν(l|S)          [both terms in nats per dimension]

ν(l|S) = exp( [logdet Σ_{l|S} − logdet Σ_ll] / D_l )
       = ( Π_i (1 − ρ_i²) )^(1/D_l)     geometric mean over canonical correlations
```

This is relevance-minus-redundancy (mRMR / forward-stepwise) under a Gaussian
model. Greedy selection stops when the best candidate's `ν` falls below a floor
`τ`, which makes **K adaptive** and subsumes the separate "fallback" rule: when no
layer is worth fusing, the detector degrades to single-layer Mahalanobis++.

## Layout

```
lsf/
  joint_stats.py   ONE jointly-shrunk covariance over all candidate layers, built
                   in a single streaming pass. Exact Ledoit–Wolf from accumulators
                   (G, M, N) — including exact shrinkage for ANY layer subset.
  novelty.py       Schur complements, canonical correlations, ν_logdet / ν_trace.
  selector.py      Entropy densities, drops, greedy adaptive-K selection.
  scoring.py       Fused Mahalanobis scoring: joint / block_diag / additive_min
                   / additive_min_z (the P2 comparison).
  cache_io.py      Cache-layout resolution, preflight checks, AUROC/FPR95.
experiments/
  run_selection.py     build joint covariance + run every selector variant
  run_oracle_sweep.py  exhaustive subset sweep, selector regret, P2 comparison
tests/
  test_novelty.py      selector math (11 tests, numpy only)
  test_pipeline.py     end-to-end integration on a synthetic cache
  make_fake_cache.py   generates that synthetic cache
results/           Versioned result artifacts — see "Reproducibility" below.
```

## Running it on the machine with the caches

Everything reads the caches `evaluate.py` already writes under `--path_to_cache`;
nothing needs a GPU or the datasets. Prerequisite: `evaluate.py` must have been
run once with an `MM_plus_plus*` method for the model, so the per-layer caches
(`cache_methods/<model>/mm_pp_*_mean.npy`) exist.

```bash
# 0. sanity-check the cache layout (fast, no computation)
python iclr2027/experiments/run_selection.py \
    --model_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --path_to_cache ./cache_imagenetlt --preflight_only

# 1. joint covariance + all selector variants  (one pass over train features)
conda run -n mm_plus_plus python iclr2027/experiments/run_selection.py \
    --model_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --path_to_cache ./cache_imagenetlt --tag imagenetlt --K_max 3

# 2. oracle subset sweep + regret + the P2 scoring comparison
#    (reuses step 1's .npz; only val/OOD features are read)
conda run -n mm_plus_plus python iclr2027/experiments/run_oracle_sweep.py \
    --model_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --path_to_cache ./cache_imagenetlt --tag imagenetlt \
    --K_max 2 --score_modes joint block_diag additive_min additive_min_z
```

Start with `--K_max 2` (all pairs). `--K_max 3` adds all triples — use
`--anchor_only` to keep just the anchored subsets, which is the design the paper
actually proposes and is much cheaper. Any `--max_subsets` truncation is recorded
in the JSON so coverage is never silently overstated.

Verify locally first — this needs no data at all:

```bash
python3 iclr2027/tests/test_novelty.py      # selector math
python3 iclr2027/tests/test_pipeline.py     # full pipeline on a synthetic cache
```

## ⚠️ Check this before regenerating any ImageNet-1K number

`preflight` reports a **shard-order divergence** that exists in the current
pipeline. Training labels and intermediate features are concatenated with
different sort orders:

- `evaluate.OODScore.check_complete()` uses plain `sorted()` — **lexicographic**:
  `0, 1, 10, 11, …, 19, 2, 20, …`
- `utils.load_intermediate_features()` and
  `detection_methods.evaluate_MM_plus_plus_topk_gating()` sort by `int(...)` —
  **numeric**: `0, 1, 2, 3, …`

These agree only up to 10 slices (`slice_length = 50000` in `utils.py`).
**ImageNet-LT** (~115k train → 3 slices) is fine. **ImageNet-1K** (1.28M → 26
slices) is not: the labels MM++ receives would be a block permutation of the ones
belonging to the concatenated features, so class means and the tied covariance
would be estimated with shuffled labels.

`lsf` always loads labels numerically, so results produced from this folder are
aligned. This has **not** been confirmed against a real cache — I have no
ImageNet-1K cache here — so run `--preflight_only` on the real cache to confirm,
and if it reports the divergence, the Table 1 (ImageNet-1K) numbers need
re-checking before they go into the ICLR submission.

## Three design decisions worth defending in the paper

**1. Shrink the joint covariance once, then slice.**
Fitting one Ledoit–Wolf per layer block and assembling them is the obvious
implementation and it is wrong: the assembled matrix is not the covariance of
anything, and `Σ_{l|S}` can come out **indefinite**. `test_per_block_shrinkage_breaks_psd`
exhibits a fixture where per-block shrinkage gives min eigenvalue **−5.50** while
joint shrinkage gives **+2.57**. A Schur complement of a PD matrix is always PD,
so the joint-first order of operations makes every downstream quantity valid by
construction. Slicing one matrix is also cheaper: the selector, the CCA/CKA
diagnostics, and the oracle subset sweep share a single pass over features.

**2. log-det, not trace ratio.**
The earlier draft used `ν = tr(Σ_{l|S})/tr(Σ_ll)`. That is dominated by a few
high-variance directions and is invariant only to *isotropic* rescaling.
`test_logdet_invariant_trace_not` shows an invertible reparameterisation of one
layer leaving `ν_logdet` at 0.046190 → 0.046190 while `ν_trace` moves
0.013285 → 0.023551. The log-det form is basis-invariant, equals (negative,
per-dimension) Gaussian mutual information, and — being a differential-entropy
density in nats/dim — is **commensurable with `Δ_l`**, so the two terms can be
added rather than multiplied. `ν_trace` is retained for the ablation table.

**3. The baseline must not inherit the new stopping rule.**
`combine='relevance_only'` reproduces the published MM++ selector and therefore
ignores `tau_novelty`. This was caught by a failing test: with the floor applied,
the "old" selector refused the redundant layer and the ablation would have
understated the new method's contribution. `params['novelty_floor_applied']`
records which behaviour was used.

## Status

Verified now (no GPU, no features needed) — `11/11` passing:

| Test | Establishes |
|---|---|
| `test_logdet_identity` | `ν` really is the CCA geometric mean — diagnostic and score are one object |
| `test_logdet_invariant_trace_not` | why log-det is the default |
| `test_per_block_shrinkage_breaks_psd` | why `joint_stats.py` exists |
| `test_schur_is_psd` | conditional covariances valid across random PD fixtures |
| `test_redundancy_detected` | near-copy → ν=0.046; independent → ν=1.000 |
| `test_selector_avoids_redundant_layer` | **the K=3 pathology in miniature**: entropy-drop-only picks the redundant layer, LSF picks the complementary one |
| `test_adaptive_k_falls_back` | all-redundant → K=1, `stop='novelty_floor'`; without the floor the same fixture fuses |
| `test_end_to_end_from_samples` | survives estimation + shrinkage from finite samples |
| `test_lw_matches_sklearn` | **skipped here** — needs the `mm_plus_plus` env (see below) |

```bash
python3 iclr2027/tests/test_novelty.py     # numpy only
```

`test_pipeline.py` additionally runs the two drivers end-to-end against a
synthetic cache and asserts that the planted redundant layer is the one the
baseline selects and LSF rejects. It validates **plumbing**, not efficacy: the
synthetic OOD shift is arbitrary, so its AUROC numbers mean nothing.

Not yet done: wiring the selected layer set back into `detection_methods.py` as a
named method, CKA diagnostics (CCA is implemented), τ calibration across
checkpoints, KPCA backend.

## A caveat about the criterion, worth checking on real data

The entropy density is `ρ_l = H_l / D_l`, and `H_l ≤ ln D_l`, so for any layer
whose within-class spectrum is near-flat, `ρ_l ≈ ln(D_l)/D_l` — a function of the
layer's **width**, not its representation. In the synthetic fixture this was
stark: with widths 24/32/40 the entropy densities came out 0.132/0.108/0.091,
matching `ln(D)/D` to three decimals and completely masking the spectrum.

For isotropic backbones (ViT: every block 768) this cancels and `Δ_l` is driven by
the spectrum, as intended. For **heterogeneous** backbones (Swin, ConvNeXt) it
means the entropy-density drop partly tracks where the channel width changes.
That is consistent with the rebuttal's own finding that `H_l/D_l` selects
stage-boundary layers on Swin/ConvNeXt — possibly for the wrong reason.

`run_selection.py` prints `entropy density` per layer and stores the layer dims in
the JSON, so this is directly checkable: if `ρ_l · D_l / ln(D_l)` is near 1 for
most layers, width is doing the work. Worth resolving before the paper leans on
the criterion for hierarchical architectures.

## Environment

The core library is **numpy-only and runs on system python3** (3.9+), deliberately,
so the math can be tested without the full stack. Two caveats on this machine:

- `sklearn` is absent, so `test_lw_matches_sklearn` skips. Run it inside the
  `mm_plus_plus` conda env (`environment.yml`) before trusting the joint fit —
  it asserts the streaming shrinkage matches `ledoit_wolf_shrinkage` to 1e-12.
- numpy 2.x on macOS/Accelerate emits **spurious** `overflow/divide-by-zero in
  matmul` warnings on finite inputs. Verified harmless; suppressed in the test
  runner only, never in `lsf/`.

## Reproducibility — the discipline that was missing

The block-diagonal numbers (82.36 / 82.54), the CCA/CKA table, and the
33-checkpoint sweep from the NeurIPS rebuttal exist **only as markdown prose**;
no logs, score files, or generating scripts were committed, and the feature caches
are gone. Every number in the ICLR submission should be traceable to a committed
artifact.

Rules for this folder:

1. Every experiment writes a JSON/CSV under `results/` and commits it. These are
   small — it is the multi-hundred-GB feature caches that stay gitignored.
2. `JointCovariance.save()` produces a single `.npz` per (backbone, ID set)
   carrying `shrinkage`, `n_samples`, and the layer layout, so any selection can
   be recomputed without re-extracting features.
3. `SelectionResult.to_dict()` records the full decision trace — every candidate's
   `Δ`, `ν`, and `q` at every greedy step, plus why greedy stopped. This is both
   the audit trail and the source for the selection figure.

## Next steps

1. Regenerate feature caches for the 4 main backbones (the real bottleneck).
2. `experiments/run_selection.py` — build the joint covariance per backbone, run
   all three `combine` modes, write `results/selection_<backbone>.json`.
3. Oracle subset sweep (all K≤3 subsets) + selector regret — the highest-leverage
   missing experiment.
4. Wire the selected layer set into `evaluate_MM_plus_plus_topk_gating` as a new
   selector mode, so scoring is unchanged and only selection differs.
5. Calibrate `τ` on a development subset of checkpoints, frozen, then validate
   leave-one-architecture-family-out. **Do not** tune `τ` against OOD test AUROC —
   the failing checkpoints were identified from test data, so a rule fitted to
   rescue them is circular and reviewers will check.
