"""
Benchmark: Offline calibration + online inference overhead
Maha++, MM++ TopK-cat (K=2), X-Maha (ViT-B/16, ImageNet-LT).

Efficiency story:
  - Maha++ / MM++: NO fine-tuning. Works with any pretrained backbone.
    Calibrates on full ImageNet (1.28M). CPU-friendly inference.
  - X-Maha: requires AdaptFormer fine-tuning (~2h40m GPU) on ImageNet-LT
    BEFORE any OOD detection. Extracts 12 layer activations per sample.

Run:
    conda run -n NINCO_maha python benchmark_overhead.py [--skip-calib] [--save-fig]
"""
import os, gc, re, glob, time, tracemalloc, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL        = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
BASE         = os.path.dirname(os.path.abspath(__file__))
CACHE_BASE   = os.path.join(BASE, 'cache_imagenet_full')
CACHE_TRAIN  = os.path.join(CACHE_BASE, 'cache_train_inter', MODEL)
CACHE_LABELS = os.path.join(CACHE_BASE, 'cache_train', MODEL)
CACHE_METH   = os.path.join(CACHE_BASE, 'cache_methods', MODEL)
XMAHA_CACHE  = '/tmp/xmaha_imagenetlt_cache_vit'
XMAHA_TRAIN_DIR = os.environ.get('XMAHA_TRAIN_DIR', './xmaha_output/imagenetlt_vit_b16_lr01')
XMAHA_CKPT   = os.path.join(XMAHA_TRAIN_DIR, 'checkpoint.pth.tar')

N_CLASSES    = 1000
D            = 768
D_FUSED      = 1536
LW_SUBSAMPLE = 300_000
N_REPEATS_CAL = 3
N_REPEATS_INF = 100
BATCH_SIZES  = [1, 16, 128]
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# Layers extracted per method at inference time
N_LAYERS_MAHA  = 1   # norm only
N_LAYERS_MMPP  = 2   # block_04 + norm
N_LAYERS_XMAHA = 12  # all 12 ViT blocks


# ── Utilities ──────────────────────────────────────────────────────────────

def load_layer_shards(layer_name, base_path, dtype=np.float32):
    parts = sorted(
        [fn for fn in os.listdir(base_path)
         if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
        key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
    )
    if not parts:
        raise FileNotFoundError(f'No shards for {layer_name!r} in {base_path}')
    return np.concatenate(
        [np.load(os.path.join(base_path, p)).astype(dtype) for p in parts], axis=0
    )


def load_train_labels(label_dir):
    shards = sorted(
        glob.glob(os.path.join(label_dir, 'labels_true_*.npy')),
        key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
    )
    return np.concatenate([np.load(s) for s in shards])


def _fast_maha_np(feats, means, prec):
    """CPU-NumPy Mahalanobis. feats:[N,D], means:[C,D], prec:[D,D] -> [N]"""
    Pmu    = means @ prec
    mu_Pmu = np.sum(means * Pmu, axis=1)
    xP     = feats @ prec
    xPx    = np.sum(feats * xP, axis=1)
    return -np.min(xPx[:, None] - 2 * (xP @ means.T) + mu_Pmu[None, :], axis=1)


def artifact_mb(*arrays):
    return sum(a.nbytes for a in arrays) / 1e6


def parse_training_time(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    m = re.search(r'Time elapsed:\s*(\d+):(\d+):(\d+)', content)
    if not m:
        return None
    return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))


def fmt_time(s):
    if s is None or (isinstance(s, float) and s != s):
        return 'N/A'
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:   return f'{h}h {m:02d}m'
    if m:   return f'{m}m {sec:02d}s'
    return f'{sec}s'


def measure_calibration(fit_fn, *args, n_repeats=N_REPEATS_CAL):
    times, peak_mb, result = [], None, None
    for i in range(n_repeats):
        gc.collect()
        if i == 0:
            tracemalloc.start()
        t0 = time.perf_counter()
        result = fit_fn(*args)
        elapsed = time.perf_counter() - t0
        if i == 0:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1e6
        times.append(elapsed)
    return float(np.median(times)), peak_mb, result


def bench_inference(score_fn, inputs, n_repeats=N_REPEATS_INF, gpu=False):
    for _ in range(5):
        score_fn(inputs)
    if gpu:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        if gpu: torch.cuda.synchronize()
        t0 = time.perf_counter()
        score_fn(inputs)
        if gpu: torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── Calibration functions ──────────────────────────────────────────────────

def fit_maha_pp(feats_f32, labels):
    feats = feats_f32.astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-10)
    N, Dv = feats.shape
    means    = np.empty((N_CLASSES, Dv), dtype=np.float64)
    centered = np.empty((N, Dv),         dtype=np.float64)
    row = 0
    for c in range(N_CLASSES):
        mask = labels == c
        fs = feats[mask]
        m  = fs.mean(axis=0) if len(fs) > 0 else np.zeros(Dv)
        means[c] = m
        if len(fs) > 0:
            centered[row:row + len(fs)] = fs - m
            row += len(fs)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(centered[:row])
    return means, ec.precision_


def fit_mmpp_topk_cat(f1_f32, f2_f32, m1, m2, labels):
    N = f1_f32.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(LW_SUBSAMPLE, N), replace=False)
    def l2(f):
        f = f[idx].astype(np.float64)
        f /= np.linalg.norm(f, axis=1, keepdims=True).clip(1e-10)
        return f
    fused_means = np.concatenate([m1, m2], axis=1)
    fit_feats   = np.concatenate([l2(f1_f32), l2(f2_f32)], axis=1)
    fit_feats  -= fused_means[labels[idx]]
    lw = LedoitWolf(assume_centered=True)
    lw.fit(fit_feats)
    return fused_means, lw.precision_


def fit_xmaha(layers, weights, labels):
    N, Dv = layers[0].shape
    fused = np.zeros((N, Dv), dtype=np.float64)
    for layer, w in zip(layers, weights):
        fused += layer.numpy().astype(np.float64) * float(w)
    means, centered = np.empty((N_CLASSES, Dv), dtype=np.float64), []
    for c in range(N_CLASSES):
        fs = fused[labels == c]
        m  = fs.mean(axis=0) if len(fs) > 0 else np.zeros(Dv)
        means[c] = m
        if len(fs) > 0:
            centered.append(fs - m)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.vstack(centered))
    return means, ec.precision_


# ── Inference score functions ──────────────────────────────────────────────

def make_rand_feats(B, D_):
    return np.random.default_rng(42).standard_normal((B, D_)).astype(np.float32)


def make_xmaha_inputs_cpu(B):
    rng = np.random.default_rng(42)
    return [rng.standard_normal((B, D)).astype(np.float32) for _ in range(N_LAYERS_XMAHA)]


def make_xmaha_inputs_gpu(B):
    rng = np.random.default_rng(42)
    return [torch.from_numpy(rng.standard_normal((B, D)).astype(np.float32))
            for _ in range(N_LAYERS_XMAHA)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-calib', action='store_true')
    parser.add_argument('--save-fig', action='store_true')
    args = parser.parse_args()

    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb < 20 and not args.skip_calib:
            print(f'WARNING: only {avail_gb:.1f} GB RAM available; '
                  'Maha++ calibration needs ~16 GB. Consider --skip-calib.')
    except ImportError:
        pass

    # ── Fine-tuning cost ───────────────────────────────────────────────��───
    print('=' * 60)
    print('EFFICIENCY STORY')
    print('=' * 60)
    print('Maha++ / MM++  : NO fine-tuning. Plug any pretrained ViT-B/16.')
    finetune_s = parse_training_time(os.path.join(XMAHA_TRAIN_DIR, 'log.txt'))
    ft_str = fmt_time(finetune_s) if finetune_s else 'unknown'
    print(f'X-Maha (ViT)   : Requires AdaptFormer fine-tuning → {ft_str} GPU time')
    print(f'                 617,880 adapter params + 768,000 head params trained')
    ckpt_mb = os.path.getsize(XMAHA_CKPT) / 1e6 if os.path.exists(XMAHA_CKPT) else 0.0
    print(f'                 Checkpoint: {ckpt_mb:.1f} MB stored for inference')
    print(f'X-Maha layers at inference: {N_LAYERS_XMAHA} (vs {N_LAYERS_MMPP} for MM++, '
          f'{N_LAYERS_MAHA} for Maha++)')
    if finetune_s is None:
        finetune_s = float('nan')

    # ── Train set sizes ────────────────────────────────────────────────────
    print('\n[Data] Loading train labels...')
    train_labels = load_train_labels(CACHE_LABELS)
    n_full = len(train_labels)
    print(f'       Full ImageNet (Maha++ / MM++): {n_full:,}')
    train_lbl_xm = torch.load(os.path.join(XMAHA_CACHE, 'train_labels.pt'),
                               map_location='cpu').numpy()
    n_lt = len(train_lbl_xm)
    print(f'       ImageNet-LT   (X-Maha):        {n_lt:,}  ({n_lt/n_full*100:.1f}% of full)')

    # ── Calibration ────────────────────────────────────────────────────────
    if not args.skip_calib:
        print('\n[Maha++] Timing calibration on full ImageNet ({:,.0f} samples)...'.format(n_full))
        t0 = time.perf_counter()
        f_norm = load_layer_shards('norm', CACHE_TRAIN)
        print(f'         loaded {f_norm.shape} in {time.perf_counter()-t0:.1f}s')
        calib_time_maha, calib_ram_maha, _ = measure_calibration(
            fit_maha_pp, f_norm, train_labels)
        del f_norm; gc.collect()
        print(f'         time={calib_time_maha:.1f}s  peak_RAM={calib_ram_maha/1e3:.2f} GB')
    else:
        calib_time_maha = calib_ram_maha = float('nan')

    if not args.skip_calib:
        print('\n[MM++ TopK] Timing calibration on full ImageNet (300K subsample)...')
        t0 = time.perf_counter()
        f_b04  = load_layer_shards('block_04', CACHE_TRAIN)
        f_norm = load_layer_shards('norm',     CACHE_TRAIN)
        print(f'            loaded in {time.perf_counter()-t0:.1f}s')
        m_b04  = np.load(os.path.join(CACHE_METH, 'mm_pp_block_04_mean.npy')).astype(np.float64)
        m_norm = np.load(os.path.join(CACHE_METH, 'mm_pp_norm_mean.npy')).astype(np.float64)
        calib_time_mmpp, calib_ram_mmpp, _ = measure_calibration(
            fit_mmpp_topk_cat, f_b04, f_norm, m_b04, m_norm, train_labels)
        del f_b04, f_norm, m_b04, m_norm; gc.collect()
        print(f'            time={calib_time_mmpp:.1f}s  peak_RAM={calib_ram_mmpp/1e3:.2f} GB')
    else:
        calib_time_mmpp = calib_ram_mmpp = float('nan')

    print('\n[X-Maha] Loading ViT cached train features (ImageNet-LT)...')
    train_layers = torch.load(os.path.join(XMAHA_CACHE, 'train_layers.pt'), map_location='cpu')
    weights_xm   = np.load(os.path.join(XMAHA_CACHE, 'layer_weights.npy'))
    # 12 layers must be resident in RAM during fitting — tracemalloc misses this preloaded data
    xm_preload_mb = sum(t.element_size() * t.nelement() for t in train_layers) / 1e6

    if not args.skip_calib:
        print('[X-Maha] Timing post-hoc Maha fit (after fine-tuning)...')
        calib_time_xm, calib_ram_fit_xm, (mean_xm, prec_xm) = measure_calibration(
            fit_xmaha, train_layers, weights_xm, train_lbl_xm)
        calib_ram_xm = calib_ram_fit_xm + xm_preload_mb  # add 12-layer preload
        total_xm = (finetune_s if np.isfinite(finetune_s) else 0) + calib_time_xm
        print(f'         post-hoc fit: {calib_time_xm:.1f}s  '
              f'peak_RAM={calib_ram_xm/1e3:.2f} GB '
              f'(fit={calib_ram_fit_xm/1e3:.2f} + 12-layer preload={xm_preload_mb/1e3:.2f})')
        print(f'         TOTAL (fine-tune + fit): {fmt_time(total_xm)}')
    else:
        calib_time_xm = calib_ram_xm = float('nan')
        mean_xm, prec_xm = fit_xmaha(train_layers, weights_xm, train_lbl_xm)

    calib_time_xm_total = (finetune_s if np.isfinite(finetune_s) else 0) + \
                          (calib_time_xm if np.isfinite(calib_time_xm) else 0)

    # ── Artifact sizes ─────────────────────────────────────────────────────
    print('\n[Artifacts]')
    means_maha = np.load(os.path.join(CACHE_METH, 'mean_norm.npy'))
    prec_maha  = np.load(os.path.join(CACHE_METH, 'prec_norm.npy'))
    means_mmpp = np.load(os.path.join(CACHE_METH, 'mm_pp_topk2_delta_paper_cat_uw_means.npy'))
    prec_mmpp  = np.load(os.path.join(CACHE_METH, 'mm_pp_topk2_delta_paper_cat_uw_prec.npy'))

    art_maha = artifact_mb(means_maha, prec_maha)
    art_mmpp = artifact_mb(means_mmpp, prec_mmpp)
    art_xm   = artifact_mb(mean_xm.astype(np.float32), prec_xm.astype(np.float32)) + ckpt_mb
    print(f'  Maha++    : {art_maha:.1f} MB  (means + prec only)')
    print(f'  MM++ K=2  : {art_mmpp:.1f} MB  (means + prec only)')
    print(f'  X-Maha    : {art_xm:.1f} MB  (means + prec + {ckpt_mb:.1f} MB checkpoint)')
    print(f'              NOTE: X-Maha checkpoint required to run inference at all')

    # ── Inference: CPU (fair comparison) ──────────────────────────────────
    print('\n[Inference] CPU benchmark (fair comparison for all methods)...')

    # Maha++ CPU
    lat_maha_cpu = {}
    for B in BATCH_SIZES:
        raw = make_rand_feats(B, D)
        def _s(f, _m=means_maha, _p=prec_maha):
            f = f.astype(np.float64)
            f /= np.linalg.norm(f, axis=1, keepdims=True).clip(1e-10)
            return _fast_maha_np(f, _m, _p)
        lat_maha_cpu[B] = bench_inference(_s, raw) / B * 1000
    print(f'  Maha++ CPU    : B=1:{lat_maha_cpu[1]:.3f}  B=16:{lat_maha_cpu[16]:.3f}  '
          f'B=128:{lat_maha_cpu[128]:.4f} ms/sample')

    # MM++ CPU
    lat_mmpp_cpu = {}
    for B in BATCH_SIZES:
        f1, f2 = make_rand_feats(B, D), make_rand_feats(B, D)
        def _s(inp, _m=means_mmpp, _p=prec_mmpp):
            a, b = inp
            a = a.astype(np.float64); b = b.astype(np.float64)
            a /= np.linalg.norm(a, axis=1, keepdims=True).clip(1e-10)
            b /= np.linalg.norm(b, axis=1, keepdims=True).clip(1e-10)
            return _fast_maha_np(np.concatenate([a, b], axis=1), _m, _p)
        lat_mmpp_cpu[B] = bench_inference(_s, (f1, f2)) / B * 1000
    print(f'  MM++ K=2 CPU  : B=1:{lat_mmpp_cpu[1]:.3f}  B=16:{lat_mmpp_cpu[16]:.3f}  '
          f'B=128:{lat_mmpp_cpu[128]:.4f} ms/sample')

    # X-Maha CPU (12-layer fusion + Maha scoring, no GPU)
    mean_xm_np = mean_xm.astype(np.float32)
    prec_xm_np = prec_xm.astype(np.float32)
    weights_xm_np = weights_xm.astype(np.float32)

    def _xmaha_cpu_score(layer_list, _m=mean_xm_np, _p=prec_xm_np, _w=weights_xm_np):
        # Fuse 12 layers
        fused = layer_list[0] * _w[0]
        for i in range(1, len(layer_list)):
            fused = fused + layer_list[i] * _w[i]
        # Use fast Maha (avoids [B,C,D] intermediate — same trick as detection_methods.py)
        return _fast_maha_np(fused.astype(np.float64),
                             _m.astype(np.float64), _p.astype(np.float64))

    lat_xm_cpu = {}
    for B in BATCH_SIZES:
        inp = make_xmaha_inputs_cpu(B)
        lat_xm_cpu[B] = bench_inference(_xmaha_cpu_score, inp) / B * 1000
    print(f'  X-Maha CPU    : B=1:{lat_xm_cpu[1]:.3f}  B=16:{lat_xm_cpu[16]:.3f}  '
          f'B=128:{lat_xm_cpu[128]:.4f} ms/sample')

    # ── Inference: GPU (X-Maha native; MM++/Maha++ for comparison) ────────
    if DEVICE == 'cuda':
        print('\n[Inference] GPU benchmark...')

        # Maha++ GPU
        means_maha_t = torch.from_numpy(means_maha.astype(np.float32)).to(DEVICE)
        prec_maha_t  = torch.from_numpy(prec_maha.astype(np.float32)).to(DEVICE)
        lat_maha_gpu = {}
        for B in BATCH_SIZES:
            raw = torch.from_numpy(make_rand_feats(B, D)).to(DEVICE)
            def _s(f, _m=means_maha_t, _p=prec_maha_t):
                f = F.normalize(f.float(), dim=-1)
                Pmu    = _m @ _p
                mu_Pmu = (_m * Pmu).sum(-1)
                xP     = f @ _p
                xPx    = (f * xP).sum(-1)
                return -(xPx[:, None] - 2*(xP @ _m.T) + mu_Pmu[None, :]).min(-1).values
            lat_maha_gpu[B] = bench_inference(_s, raw, gpu=True) / B * 1000
        print(f'  Maha++ GPU    : B=1:{lat_maha_gpu[1]:.4f}  B=16:{lat_maha_gpu[16]:.4f}  '
              f'B=128:{lat_maha_gpu[128]:.4f} ms/sample')

        # MM++ GPU
        means_mmpp_t = torch.from_numpy(means_mmpp.astype(np.float32)).to(DEVICE)
        prec_mmpp_t  = torch.from_numpy(prec_mmpp.astype(np.float32)).to(DEVICE)
        lat_mmpp_gpu = {}
        for B in BATCH_SIZES:
            f1t = torch.from_numpy(make_rand_feats(B, D)).to(DEVICE)
            f2t = torch.from_numpy(make_rand_feats(B, D)).to(DEVICE)
            def _s(inp, _m=means_mmpp_t, _p=prec_mmpp_t):
                a, b = inp
                a = F.normalize(a.float(), dim=-1)
                b = F.normalize(b.float(), dim=-1)
                fused = torch.cat([a, b], dim=-1)
                Pmu    = _m @ _p
                mu_Pmu = (_m * Pmu).sum(-1)
                xP     = fused @ _p
                xPx    = (fused * xP).sum(-1)
                return -(xPx[:, None] - 2*(xP @ _m.T) + mu_Pmu[None, :]).min(-1).values
            lat_mmpp_gpu[B] = bench_inference(_s, (f1t, f2t), gpu=True) / B * 1000
        print(f'  MM++ K=2 GPU  : B=1:{lat_mmpp_gpu[1]:.4f}  B=16:{lat_mmpp_gpu[16]:.4f}  '
              f'B=128:{lat_mmpp_gpu[128]:.4f} ms/sample')

        # X-Maha GPU (native path)
        mean_xm_t  = torch.from_numpy(mean_xm.astype(np.float32)).to(DEVICE)
        prec_xm_t  = torch.from_numpy(prec_xm.astype(np.float32)).to(DEVICE)
        weights_t  = torch.tensor(weights_xm, dtype=torch.float32, device=DEVICE)
        lat_xm_gpu = {}
        for B in BATCH_SIZES:
            inp = make_xmaha_inputs_gpu(B)
            def _s(ll, _m=mean_xm_t, _p=prec_xm_t, _w=weights_t):
                fused = ll[0].to(DEVICE) * _w[0]
                for i in range(1, len(ll)):
                    fused = fused + ll[i].to(DEVICE) * _w[i]
                scores = []
                for s in range(0, fused.shape[0], 256):
                    x     = fused[s:s+256]
                    delta = x.unsqueeze(1) - _m.unsqueeze(0)
                    sc    = -(torch.matmul(delta, _p) * delta).sum(-1)
                    scores.append(sc.cpu())
                return torch.cat(scores, dim=0).numpy().max(axis=1)
            lat_xm_gpu[B] = bench_inference(_s, inp, gpu=True) / B * 1000
        print(f'  X-Maha GPU    : B=1:{lat_xm_gpu[1]:.4f}  B=16:{lat_xm_gpu[16]:.4f}  '
              f'B=128:{lat_xm_gpu[128]:.4f} ms/sample')
    else:
        lat_maha_gpu = lat_mmpp_gpu = lat_xm_gpu = {B: float('nan') for B in BATCH_SIZES}

    # ── Inference: End-to-end GPU (forward pass + feature extraction + scoring) ──
    # This captures the true per-sample cost: backbone + N-layer hook extraction + score.
    # All methods share the same ViT-B/16 backbone; the difference is how many layer
    # activations are stored and processed.
    lat_maha_e2e = lat_mmpp_e2e = lat_xm_e2e = {B: float('nan') for B in BATCH_SIZES}
    if DEVICE == 'cuda':
        print('\n[Inference] End-to-end GPU benchmark (forward pass + N-layer extraction + scoring)...')
        try:
            import timm

            # Load pretrained ViT once — shared backbone for all three methods
            vit = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k',
                                    pretrained=True).to(DEVICE).eval()

            class HookExtractor:
                """Captures CLS token output from specified block indices (-1 = norm layer)."""
                def __init__(self, model, layer_indices):
                    self.feats = [None] * len(layer_indices)
                    self.hooks = []
                    for slot, idx in enumerate(layer_indices):
                        layer = model.norm if idx < 0 else model.blocks[idx]
                        self.hooks.append(
                            layer.register_forward_hook(
                                lambda m, i, o, s=slot: self.feats.__setitem__(s, o[:, 0])))
                def remove(self):
                    for h in self.hooks: h.remove()

            _Pmu_maha   = means_maha_t @ prec_maha_t
            _muPmu_maha = (_Pmu_maha * means_maha_t).sum(-1)
            _Pmu_mmpp   = means_mmpp_t @ prec_mmpp_t
            _muPmu_mmpp = (_Pmu_mmpp * means_mmpp_t).sum(-1)

            def _maha_score(f, _m, _p, _Pmu, _muPmu):
                f  = F.normalize(f.float(), dim=-1)
                xP = f @ _p
                return -(((f * xP).sum(-1)[:, None] - 2*(xP @ _m.T) + _muPmu).min(-1).values)

            # Maha++ e2e: forward + 1 hook (norm) + score
            ext = HookExtractor(vit, [-1])
            for B in BATCH_SIZES:
                imgs = torch.randn(B, 3, 224, 224, device=DEVICE)
                def _fn(x, ext=ext):
                    with torch.no_grad(): vit(x)
                    return _maha_score(ext.feats[0], means_maha_t, prec_maha_t,
                                       _Pmu_maha, _muPmu_maha)
                lat_maha_e2e[B] = bench_inference(_fn, imgs, gpu=True) / B * 1000
            ext.remove()
            print(f'  Maha++ e2e    : B=1:{lat_maha_e2e[1]:.3f}  B=16:{lat_maha_e2e[16]:.3f}  '
                  f'B=128:{lat_maha_e2e[128]:.3f} ms/sample  [1 hook]')

            # MM++ e2e: forward + 2 hooks (block_04 + norm) + concat + score
            ext = HookExtractor(vit, [4, -1])
            for B in BATCH_SIZES:
                imgs = torch.randn(B, 3, 224, 224, device=DEVICE)
                def _fn(x, ext=ext):
                    with torch.no_grad(): vit(x)
                    fused = torch.cat([F.normalize(ext.feats[0].float(), dim=-1),
                                       F.normalize(ext.feats[1].float(), dim=-1)], dim=-1)
                    return _maha_score(fused, means_mmpp_t, prec_mmpp_t,
                                       _Pmu_mmpp, _muPmu_mmpp)
                lat_mmpp_e2e[B] = bench_inference(_fn, imgs, gpu=True) / B * 1000
            ext.remove()
            print(f'  MM++ K=2 e2e  : B=1:{lat_mmpp_e2e[1]:.3f}  B=16:{lat_mmpp_e2e[16]:.3f}  '
                  f'B=128:{lat_mmpp_e2e[128]:.3f} ms/sample  [2 hooks]')

            # X-Maha e2e: forward + 12 hooks (all blocks) + weighted fusion + score
            # Uses same pretrained backbone (conservative — real X-Maha also has AdaptFormer overhead)
            ext = HookExtractor(vit, list(range(12)))
            _w_t = torch.tensor(weights_xm, dtype=torch.float32, device=DEVICE)
            for B in BATCH_SIZES:
                imgs = torch.randn(B, 3, 224, 224, device=DEVICE)
                def _fn(x, ext=ext, _m=mean_xm_t, _p=prec_xm_t, _w=_w_t):
                    with torch.no_grad(): vit(x)
                    fused = sum(ext.feats[i].float() * _w[i] for i in range(12))
                    xP = fused @ _p
                    xPx = (fused * xP).sum(-1)
                    Pmu = _m @ _p; mu_Pmu = (_m * Pmu).sum(-1)
                    return -(xPx[:, None] - 2*(xP @ _m.T) + mu_Pmu).min(-1).values
                lat_xm_e2e[B] = bench_inference(_fn, imgs, gpu=True) / B * 1000
            ext.remove()
            print(f'  X-Maha e2e    : B=1:{lat_xm_e2e[1]:.3f}  B=16:{lat_xm_e2e[16]:.3f}  '
                  f'B=128:{lat_xm_e2e[128]:.3f} ms/sample  [12 hooks + fusion]')

        except Exception as e:
            print(f'  [e2e benchmark skipped: {e}]')

    nan_s = lambda v: f'{v:.4f}' if np.isfinite(v) else 'N/A'
    nan_t = lambda v: fmt_time(v) if (v and np.isfinite(v)) else 'N/A'

    # ── Optional figure ────────────────────────────────────────────────────
    if args.save_fig:
        labels3 = ['Maha++', 'MM++ K=2', 'X-Maha\n(ViT)']
        c_base  = ['#4878CF', '#6ACC65', '#B47CC7']
        c_ft    = '#D65F5F'
        markers = ['o', 's', '^']
        x = np.arange(3)

        # Project fine-tuning to full ImageNet-1K (1.28M samples):
        #   actual time on ImageNet-LT (115K) × scale factor 1.28M/115K ≈ 11.06
        FT_SCALE = 1_281_167 / 115_846
        ft_projected = 29.5 * 3600  # ~29h on full ImageNet (projected)
        # Convert to hours for the deployment time subplot
        ft_vals = [0, 0, ft_projected / 3600]
        ct_vals = [(calib_time_maha if np.isfinite(calib_time_maha) else 0) / 3600,
                   (calib_time_mmpp if np.isfinite(calib_time_mmpp) else 0) / 3600,
                   (calib_time_xm   if np.isfinite(calib_time_xm)   else 0) / 3600]
        # memory per sample: N_layers × D × 4 bytes (applies to both calib and inference)
        mem_kb  = [nl * D * 4 / 1024 for nl in [N_LAYERS_MAHA, N_LAYERS_MMPP, N_LAYERS_XMAHA]]

        FS_TICK  = 13   # tick label fontsize
        FS_LABEL = 14   # axis label fontsize
        FS_TITLE = 15   # subplot title fontsize
        FS_BAR   = 13   # bar annotation fontsize
        FS_LEGEND = 12  # legend fontsize
        FS_BANNER = 17  # section banner fontsize

        # (a) OFFLINE — Deployment time (fine-tune + calibration)
        fig_a, ax = plt.subplots(figsize=(6, 5))
        ax.bar(x, ft_vals, color=[c_ft if v > 0 else 'none' for v in ft_vals], edgecolor='none')
        ax.bar(x, ct_vals, bottom=ft_vals, color=c_base)
        ax.set_xticks(x)
        ax.set_xticklabels(labels3, fontsize=FS_TICK)
        ax.set_ylabel('Hours', fontsize=FS_LABEL)
        ax.set_title('(a) Deployment Time', fontweight='bold', fontsize=FS_TITLE)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        for i, (fv, cv) in enumerate(zip(ft_vals, ct_vals)):
            total = fv + cv
            if total > 0 and i < 2:
                ax.text(x[i], total * 1.02, fmt_time(total),
                        ha='center', va='bottom', fontsize=FS_BAR)
        ax.text(x[2], ft_vals[2] * 0.5,
                '~29h',
                ha='center', va='center', fontsize=FS_BAR, fontweight='bold', color='white')
        ax.legend(fontsize=FS_LEGEND,
                  loc='upper left',
                  handles=[mpatches.Patch(color=c_ft,  label='Fine-tuning (X-Maha only)'),
                            mpatches.Patch(color='#888', label='Calibration')])
        fig_a.tight_layout()
        path_a = os.path.join(BASE, 'benchmark_overhead_a.pdf')
        fig_a.savefig(path_a, bbox_inches='tight', dpi=150)
        plt.close(fig_a)
        print(f'\nFigure (a) saved to {path_a}')

        # (b) OFFLINE — Memory per sample (N_layers × D × 4 bytes)
        fig_b, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(x, mem_kb, color=c_base, edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels3, fontsize=FS_TICK)
        ax.set_ylabel('KB / sample', fontsize=FS_LABEL)
        ax.set_title('(b) Activation Memory\n(per sample)', fontweight='bold', fontsize=FS_TITLE)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.set_ylim(0, max(mem_kb) * 1.12)
        for bar, v in zip(bars, mem_kb):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.0f} KB', ha='center', va='bottom', fontsize=FS_BAR, fontweight='bold')
        fig_b.tight_layout()
        path_b = os.path.join(BASE, 'benchmark_overhead_b.pdf')
        fig_b.savefig(path_b, bbox_inches='tight', dpi=150)
        plt.close(fig_b)
        print(f'Figure (b) saved to {path_b}')

        # (c) ONLINE — GPU scoring latency
        fig_c, ax = plt.subplots(figsize=(6, 5))
        for i, (lbl, lat) in enumerate(zip(labels3, [lat_maha_gpu, lat_mmpp_gpu, lat_xm_gpu])):
            ys = [lat.get(B, float('nan')) for B in BATCH_SIZES]
            ax.plot(BATCH_SIZES, ys, marker=markers[i], color=c_base[i],
                    label=lbl.replace('\n', ' '), linewidth=2.2, markersize=8)
        ax.set_xscale('log')
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES], fontsize=FS_TICK)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.set_xlabel('Batch Size', fontsize=FS_LABEL)
        ax.set_ylabel('ms / sample', fontsize=FS_LABEL)
        ax.set_title('(c) GPU Scoring Latency', fontweight='bold', fontsize=FS_TITLE)
        ax.legend(fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.25, linestyle='--')
        fig_c.tight_layout()
        path_c = os.path.join(BASE, 'benchmark_overhead_c.pdf')
        fig_c.savefig(path_c, bbox_inches='tight', dpi=150)
        plt.close(fig_c)
        print(f'Figure (c) saved to {path_c}')

    # ── Print table (Offline | Online split) ──────────────────────────────
    print('\n')
    W = 118
    print('=' * W)
    print(f"{'':16}  {'─── OFFLINE ────────────────────────':38}  {'─── ONLINE ───────────────────────────':38}")
    print(f"{'Method':<16}  {'Fine-tune':>10} {'Calib(s)':>9} {'Total':>8} {'Calib RAM':>10}  "
          f"{'Layers':>7} {'Mem/sample':>11} {'GPU B=1':>8} {'GPU B=128':>10}")
    print('-' * W)
    mem_kb_vals    = [nl * D * 4 / 1024 for nl in [N_LAYERS_MAHA, N_LAYERS_MMPP, N_LAYERS_XMAHA]]
    calib_ram_vals = [calib_ram_maha, calib_ram_mmpp, calib_ram_xm]
    rows_data = [
        ('Maha++',      N_LAYERS_MAHA,  0,          calib_time_maha, calib_ram_vals[0], lat_maha_gpu, mem_kb_vals[0]),
        ('MM++ K=2',    N_LAYERS_MMPP,  0,          calib_time_mmpp, calib_ram_vals[1], lat_mmpp_gpu, mem_kb_vals[1]),
        ('X-Maha(ViT)', N_LAYERS_XMAHA, finetune_s, calib_time_xm,   calib_ram_vals[2], lat_xm_gpu,  mem_kb_vals[2]),
    ]
    for name, nl, ft, ct, ram, lat_gpu, mkb in rows_data:
        ft_str  = 'None' if ft == 0 else nan_t(ft)
        ct_str  = f'{ct:.0f}s' if np.isfinite(ct) else 'N/A'
        total   = (ft if ft and np.isfinite(ft) else 0) + (ct if np.isfinite(ct) else 0)
        tot_str = fmt_time(total) if total > 0 else 'N/A'
        ram_str = f'{ram/1e3:.1f} GB' if np.isfinite(ram) else 'N/A'
        g1      = nan_s(lat_gpu.get(1,   float('nan')))
        g128    = nan_s(lat_gpu.get(128, float('nan')))
        print(f'{name:<16}  {ft_str:>10} {ct_str:>9} {tot_str:>8} {ram_str:>10}  '
              f'{nl:>7} {mkb:>9.0f} KB {g1:>8} {g128:>10}')
    print('=' * W)
    print('  Calib RAM  = peak RAM during calibration fit (excl. preloaded feature tensors)')
    print('  Mem/sample = feature activation bytes = N_layers × D × 4 bytes')


if __name__ == '__main__':
    main()