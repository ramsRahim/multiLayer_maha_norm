"""Create a teaser figure for MM++ (MM_plus_plus_topk_cat).

The figure combines three pieces:
  1. Last-layer Maha++ score distributions.
  2. The MM++ offline layer-selection and concatenation pipeline.
  3. MM++ score distributions plus a compact FPR@95 summary.

By default the script uses a cached ImageNet-LT ConvNeXt-T run where MM++ gives
a large FPR@95 reduction on ImageNet-C. Use --model and --dataset to regenerate
the teaser for other cached runs.
"""
import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np


os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',
})


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.environ.get(
    'IMAGENETLT_CACHE',
    os.path.join(REPO_ROOT, 'cache_imgnetlt'),
)
OUT_DIR = os.path.join(REPO_ROOT, 'assets')

BASELINE_KEY = 'Mahalanobis_norm'
MMPP_KEY = 'MM_plus_plus_topk_cat'

DEFAULT_MODEL = 'convnext_tiny.fb_in1k'
DEFAULT_DATASET = 'imagenet_c'
DEFAULT_K = 2

ID_CLASSES_FOR_TSNE = (10, 300, 633)

MODEL_TITLES = {
    'resnet50.tv2_in1k': 'ResNet-50',
    'convnext_tiny.fb_in1k': 'ConvNeXt-T',
    'swin_tiny_patch4_window7_224.ms_in1k': 'Swin-T',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'ViT-B/16',
    'vit_base_patch16_224.augreg_in21k_ft_in1k': 'ViT-B/16',
    'vit_base_patch16_224.in21k': 'ViT-B/16',
    'vit_large_patch16_224.augreg_in21k_ft_in1k': 'ViT-L/16',
}

DATASET_TITLES = {
    'NINCO_OOD_classes': 'NINCO',
    'openimages_o': 'OpenImage-O',
    'imagenet_r': 'ImageNet-R',
    'imagenet_c': 'ImageNet-C',
    'imagenet_es': 'ImageNet-ES',
    'imagenet_v2': 'ImageNet-V2',
    'imagenet_o': 'ImageNet-O',
    'ssb_hard': 'SSB-Hard',
    'sun': 'SUN',
}

DATASET_SHORT_TITLES = {
    'NINCO_OOD_classes': 'NINCO',
    'openimages_o': 'OpenImage-O',
    'imagenet_r': 'IN-R',
    'imagenet_c': 'IN-C',
    'imagenet_es': 'IN-ES',
    'imagenet_v2': 'IN-V2',
    'imagenet_o': 'IN-O',
    'ssb_hard': 'SSB',
    'sun': 'SUN',
    'texture': 'Texture',
    'places365': 'Places365',
    'inaturalist': 'iNaturalist',
}

DATASET_ORDER = [
    'NINCO_OOD_classes',
    'imagenet_r',
    'imagenet_c',
    'imagenet_es',
    'imagenet_v2',
    'imagenet_o',
]

# FPR@95 values from the ImageNet-LT main results table (ConvNeXt-T):
# (Maha++, MM++ (Ours)) per OOD benchmark.
TABLE_FPR_VALUES = OrderedDict([
    ('NINCO_OOD_classes', (56.49, 55.61)),
    ('openimages_o',      (39.51, 37.69)),
    ('imagenet_c',        (64.26, 56.39)),
    ('imagenet_r',        (52.25, 49.12)),
    ('imagenet_es',       (78.41, 69.36)),
    ('imagenet_v2',       (90.32, 90.33)),
])

BLUE = '#1F77B4'
ORANGE = '#FF7F0E'
MAUVE = '#B279A2'
MAUVE_DARK = '#6F3F5F'
TEAL = '#4CB9B1'
TEAL_DARK = '#267C79'
GOLD = '#E8C173'
GRAY = '#6E7681'
LIGHT_GRAY = '#D7DCE2'
PANEL = '#F8F9FB'


def model_title(model):
    return MODEL_TITLES.get(model, model.replace('_', r'\_'))


def dataset_title(dataset):
    return DATASET_TITLES.get(dataset, dataset.replace('_', '-'))


def dataset_short_title(dataset):
    return DATASET_SHORT_TITLES.get(dataset, dataset_title(dataset))


def latest_score_file(cache_root, model, dataset):
    score_dir = os.path.join(cache_root, 'scores', model, dataset)
    paths = sorted(glob.glob(os.path.join(score_dir, '*.npz')))
    if not paths:
        raise FileNotFoundError(f'No score .npz files found in {score_dir}')
    return paths[-1]


def load_score_results(cache_root, model, dataset):
    score_dir = os.path.join(cache_root, 'scores', model, dataset)
    paths = sorted(glob.glob(os.path.join(score_dir, '*.npz')))
    if not paths:
        raise FileNotFoundError(f'No score .npz files found in {score_dir}')
    merged = {}
    for path in paths:
        try:
            z = np.load(path, allow_pickle=True)
            results = z['methods_results'].item()
        except Exception:
            continue
        for key, value in results.items():
            merged[key] = value
    missing = [key for key in (BASELINE_KEY, MMPP_KEY) if key not in merged]
    if missing:
        raise KeyError(f'Missing method scores {missing} in {score_dir}')
    return merged, score_dir


def method_scores(results, key):
    method = results[key]
    return (
        np.asarray(method['scores_id'], dtype=np.float64).ravel(),
        np.asarray(method['scores_ood'], dtype=np.float64).ravel(),
    )


def auroc(scores_id, scores_ood):
    scores = np.nan_to_num(np.concatenate([scores_id, scores_ood]))
    labels = np.concatenate([
        np.ones(len(scores_id), dtype=np.int8),
        np.zeros(len(scores_ood), dtype=np.int8),
    ])
    order = np.argsort(-scores, kind='mergesort')
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = np.r_[0.0, tp / max(tp[-1], 1), 1.0]
    fpr = np.r_[0.0, fp / max(fp[-1], 1), 1.0]
    return float(np.trapezoid(tpr, fpr))


def fpr_at_tpr(scores_id, scores_ood, tpr=0.95):
    threshold = float(np.quantile(np.nan_to_num(scores_id), 1.0 - tpr))
    fpr = float((np.nan_to_num(scores_ood) >= threshold).mean())
    return fpr, threshold


def standardize(scores, mu, sigma):
    return (np.nan_to_num(scores) - mu) / max(sigma, 1e-10)


def score_panel_data(scores_id, scores_ood):
    mu = float(np.nanmean(scores_id))
    sigma = max(float(np.nanstd(scores_id)), 1e-10)
    threshold_raw_fpr, threshold = fpr_at_tpr(scores_id, scores_ood)
    return {
        'id': standardize(scores_id, mu, sigma),
        'ood': standardize(scores_ood, mu, sigma),
        'threshold': (threshold - mu) / sigma,
        'fpr': threshold_raw_fpr,
        'auroc': auroc(scores_id, scores_ood),
    }


def style_axis(ax):
    ax.grid(True, axis='y', linestyle=':', color=LIGHT_GRAY, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)


def draw_score_axis(ax, panel, header, xlabel):
    combined = np.concatenate([panel['id'], panel['ood']])
    lo, hi = np.quantile(combined, [0.002, 0.998])
    pad = max(0.08 * (hi - lo), 0.35)
    bins = np.linspace(lo - pad, hi + pad, 90)

    ax.hist(panel['id'], bins=bins, density=True, histtype='step',
            color=BLUE, linewidth=1.5, label='ID')
    ax.hist(panel['ood'], bins=bins, density=True, histtype='step',
            color=ORANGE, linewidth=1.5, label='OOD')
    ax.axvline(panel['threshold'], color=GRAY, linestyle=':', linewidth=1.4)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, labelpad=2)
    style_axis(ax)
    ax.text(0.0, 1.02, header, transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', color='#1A202C',
            ha='left', va='bottom')

    # Inline FPR label on the TPR=95% threshold line.
    y_top = ax.get_ylim()[1]
    x_offset = 0.012 * (bins[-1] - bins[0])
    ax.text(panel['threshold'] + x_offset, 0.78 * y_top,
            f"FPR={100.0 * panel['fpr']:.1f}%",
            fontsize=8.5, color='#2D3748', ha='left', va='center',
            rotation=0,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='none', alpha=0.85))
    ax.legend(loc='upper left', frameon=False, handlelength=1.3,
              fontsize=8)


def load_layer_names(cache_root, model):
    for subdir in ('cache_val_inter', 'cache_train_inter'):
        path = os.path.join(cache_root, subdir, model, 'layer_names.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f'No layer_names.json found for {model}')


def _load_concat_features(base_dir, layer_names, selected_idx):
    feats = []
    for idx in selected_idx:
        name = layer_names[idx]
        path = os.path.join(base_dir, f'layer_{name}_features_0.npy')
        x = np.load(path).astype(np.float32)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)
        feats.append(x)
    return np.concatenate(feats, axis=1)


def compute_tsne_data(cache_root, model, dataset, selected_idx, layer_names,
                      id_classes=ID_CLASSES_FOR_TSNE, n_per_class=50,
                      n_ood=80, seed=0):
    sel_str = '_'.join(map(str, selected_idx))
    cls_str = '_'.join(map(str, id_classes))
    cache_dir = os.path.join(cache_root, '_tsne_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f'tsne_{model}_{dataset}_sel{sel_str}_cls{cls_str}_n{n_per_class}_o{n_ood}.npz',
    )
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        return {k: d[k] for k in d.files}

    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    id_labels = np.load(os.path.join(cache_root, 'cache_val', model,
                                     'labels_true_0.npy'))
    id_features = _load_concat_features(
        os.path.join(cache_root, 'cache_val_inter', model),
        layer_names, selected_idx,
    )
    ood_features = _load_concat_features(
        os.path.join(cache_root, 'cache_ood_inter', model, dataset),
        layer_names, selected_idx,
    )

    id_idx = []
    id_class_labels = []
    for c in id_classes:
        cls_idx = np.where(id_labels == c)[0]
        choice = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)),
                            replace=False)
        id_idx.extend(choice.tolist())
        id_class_labels.extend([int(c)] * len(choice))
    id_idx = np.asarray(id_idx, dtype=np.int64)
    id_class_labels = np.asarray(id_class_labels, dtype=np.int64)
    ood_idx = rng.choice(len(ood_features),
                         size=min(n_ood, len(ood_features)), replace=False)

    X = np.concatenate([id_features[id_idx], ood_features[ood_idx]],
                       axis=0).astype(np.float32)
    n_id = len(id_idx)
    perplexity = max(5, min(30, n_id // 5))
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca',
                random_state=seed, max_iter=1500)
    Y = tsne.fit_transform(X)
    result = {
        'embed_id': Y[:n_id].astype(np.float32),
        'embed_ood': Y[n_id:].astype(np.float32),
        'id_class_labels': id_class_labels,
        'id_classes': np.asarray(id_classes, dtype=np.int64),
    }
    np.savez_compressed(cache_path, **result)
    return result


def layer_rank_metrics(cache_root, model, layer_names):
    """Return (H_per_layer, H/D, effective rank exp(H), log-gaps Delta_l)."""
    method_dir = os.path.join(cache_root, 'cache_methods', model)
    H = []
    h_over_d = []
    eff_rank = []
    for layer_name in layer_names:
        prec_path = os.path.join(method_dir, f'mm_pp_{layer_name}_prec.npy')
        prec = np.load(prec_path).astype(np.float64)
        eig_prec = np.linalg.eigvalsh(prec).clip(min=1e-10)
        eig_cov = (1.0 / eig_prec).clip(min=1e-8)
        lam = (eig_cov / eig_cov.sum()).clip(min=1e-300)
        entropy = -float(np.sum(lam * np.log(lam)))
        D = max(prec.shape[0], 1)
        H.append(entropy)
        h_over_d.append(entropy / D)
        eff_rank.append(float(np.exp(entropy)))
    H = np.asarray(H, dtype=np.float64)
    h_over_d = np.asarray(h_over_d, dtype=np.float64)
    eff_rank = np.asarray(eff_rank, dtype=np.float64)
    log_rel = np.log(h_over_d.clip(min=1e-300))
    gaps = log_rel[:-1] - log_rel[1:]
    return H, h_over_d, eff_rank, gaps


def precision_thumbnails(cache_root, model, layer_names, selected_idx, size=20):
    """Return (Sigma corner, Sigma^-1 corner) heatmaps for the final picked layer."""
    target_idx = int(selected_idx[-1])
    layer_name = layer_names[target_idx]
    prec_path = os.path.join(cache_root, 'cache_methods', model,
                             f'mm_pp_{layer_name}_prec.npy')
    prec = np.load(prec_path).astype(np.float64)
    w, V = np.linalg.eigh(prec)
    w = w.clip(min=1e-10)
    cov = (V * (1.0 / w)) @ V.T
    s = min(size, prec.shape[0])
    return cov[:s, :s], prec[:s, :s], layer_name


def selected_layers_from_gaps(gaps, num_layers, k):
    final_idx = num_layers - 1
    if k <= 1:
        return np.array([final_idx], dtype=int)
    candidates = np.array(
        [gap_idx for gap_idx in range(len(gaps)) if gap_idx + 1 != final_idx],
        dtype=int,
    )
    k_extra = min(k - 1, len(candidates))
    extra = candidates[np.argsort(gaps[candidates])[-k_extra:]] + 1
    return np.sort(np.unique(np.concatenate([[final_idx], extra]))).astype(int)


def short_layer_name(name, idx):
    if name == 'norm':
        return 'norm'
    if name.startswith('block_'):
        return f'B{idx}'
    if name.startswith('stage_'):
        # ConvNeXt: 'stage_2_block_07' -> 'S2B7'
        parts = name.split('_')
        try:
            stage = int(parts[1])
            block = int(parts[3])
            return f'S{stage}B{block}'
        except (IndexError, ValueError):
            pass
    return f'L{idx + 1}'


def draw_final_layer_axis(ax, layer_names):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, 'Baseline (Maha++)', fontsize=10,
            fontweight='bold', ha='left', va='top', color='#1A202C')
    ax.text(0.0, 0.92, 'Mahalanobis on final-layer features',
            fontsize=8, ha='left', va='top', color='#2D3748')

    y_positions = np.linspace(0.78, 0.20, 6)
    shown = np.linspace(0, len(layer_names) - 1, 6, dtype=int)
    for rank, idx in enumerate(shown):
        y = y_positions[rank]
        is_final = idx == len(layer_names) - 1
        face = '#F0F2F5' if not is_final else '#DDECF8'
        edge = '#B9C0CA' if not is_final else BLUE
        rect = patches.FancyBboxPatch(
            (0.14, y - 0.035), 0.40, 0.058,
            boxstyle='round,pad=0.006,rounding_size=0.012',
            facecolor=face, edgecolor=edge, linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(0.34, y - 0.008, short_layer_name(layer_names[idx], idx),
                ha='center', va='center', fontsize=8.5, color='#2D3748')
        if rank < len(y_positions) - 1:
            ax.annotate('', xy=(0.34, y_positions[rank + 1] + 0.035),
                        xytext=(0.34, y - 0.045),
                        arrowprops=dict(arrowstyle='->', lw=0.8, color='#A0A7B2'))
    ax.text(0.60, 0.22, r'$\tilde{h}_L(x)$', ha='left', va='center',
            fontsize=12, color=BLUE, fontweight='bold')
    ax.text(0.60, 0.13, 'one score per $x$',
            ha='left', va='center', fontsize=7.5, color='#2D3748')


def draw_method_axis(ax, gaps, h_over_d, eff_rank, selected_idx, layer_names,
                     k, cov_thumb, prec_thumb, tsne_data):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ----- Step 1: rank layers - three subplots with proper x/y axes. -----
    ax.text(0.02, 0.99, '(1) Rank layers',
            fontsize=9.5, ha='left', va='top', color='#1A202C',
            fontweight='bold')
    ax.text(0.02, 0.945,
            r'(1a) gap $\Delta_l$ $\cdot$ '
            r'(1b) compression $H/D$ $\cdot$ '
            r'(1c) effective rank $\exp(H)$',
            fontsize=7.5, ha='left', va='top', color='#1A202C')
    gap_inset = ax.inset_axes([0.03, 0.62, 0.16, 0.26])
    _draw_gap_inset(gap_inset, gaps, selected_idx, len(layer_names))
    hd_inset = ax.inset_axes([0.22, 0.62, 0.16, 0.26])
    _draw_hd_inset(hd_inset, h_over_d, selected_idx, len(layer_names))
    rank_inset = ax.inset_axes([0.41, 0.62, 0.16, 0.26])
    _draw_rank_inset(rank_inset, eff_rank, selected_idx, len(layer_names))

    # ----- Bottom row: pipeline (steps 2 -> 3 -> 4) + score (step 5). -----
    band_y_arrow = 0.27   # y for connecting arrows
    # Single combined caption above the pipeline visuals.
    ax.text(0.02, 0.55,
            f'(2) Top-K + $\\ell_2$  ($K={k}$)  $\\cdot$  '
            r'(3) $\hat{\Sigma} \rightarrow \hat{\Sigma}^{-1}$ (LW)  $\cdot$  '
            r'(4) Concat $\phi(x)$',
            fontsize=8.5, ha='left', va='top', color='#1A202C',
            fontweight='bold')

    # Step 2 visual: top-K stack.
    sx = 0.02
    box_w, box_h = 0.075, 0.045
    ax.text(sx + box_w / 2, band_y_arrow + 0.13, '(2)',
            fontsize=7.5, ha='center', color=MAUVE_DARK, fontweight='bold')
    for pos, idx in enumerate(selected_idx):
        y = band_y_arrow + 0.05 - 0.057 * pos
        color = MAUVE if idx != len(layer_names) - 1 else BLUE
        ax.add_patch(patches.FancyBboxPatch(
            (sx, y), box_w, box_h,
            boxstyle='round,pad=0.003,rounding_size=0.008',
            facecolor=color, edgecolor='white', linewidth=0.9, alpha=0.95,
        ))
        ax.text(sx + box_w / 2, y + box_h / 2,
                short_layer_name(layer_names[idx], idx),
                color='white', ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(sx + box_w + 0.045, band_y_arrow),
                xytext=(sx + box_w + 0.005, band_y_arrow),
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#1A202C'))

    # Step 3 visual: covariance + precision matrices with invert arrow.
    cx_pos = sx + box_w + 0.05
    cov_inset = ax.inset_axes([cx_pos, band_y_arrow - 0.05, 0.075, 0.13])
    _draw_matrix_thumb(cov_inset, cov_thumb, r'$\hat{\Sigma}$', '#1A202C')
    prec_inset = ax.inset_axes([cx_pos + 0.10, band_y_arrow - 0.05, 0.075, 0.13])
    _draw_matrix_thumb(prec_inset, prec_thumb, r'$\hat{\Sigma}^{-1}$',
                       MAUVE_DARK)
    ax.annotate('', xy=(cx_pos + 0.098, band_y_arrow + 0.015),
                xytext=(cx_pos + 0.078, band_y_arrow + 0.015),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='#1A202C'))
    ax.text(cx_pos + 0.088, band_y_arrow + 0.078, 'invert',
            fontsize=6, ha='center', va='bottom', color='#1A202C')
    ax.text(cx_pos + 0.088, band_y_arrow + 0.135, '(3)',
            fontsize=7.5, ha='center', color=MAUVE_DARK, fontweight='bold')

    cov_block_end = cx_pos + 0.10 + 0.075
    ax.annotate('', xy=(cov_block_end + 0.045, band_y_arrow),
                xytext=(cov_block_end + 0.005, band_y_arrow),
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#1A202C'))

    # Step 4 visual: ID vs OOD concatenated representations.
    vx = cov_block_end + 0.05
    vy = band_y_arrow - 0.05
    seg_w = 0.018
    seg_gap = 0.003
    n_seg = len(selected_idx)
    stack_w = n_seg * (seg_w + seg_gap) - seg_gap
    ax.text(vx + stack_w / 2, band_y_arrow + 0.13, '(4)',
            fontsize=7.5, ha='center', color=MAUVE_DARK, fontweight='bold')
    stack_specs = [
        (BLUE, vy + 0.075, r'$\phi(x_{\mathrm{in}})$'),
        (ORANGE, vy + 0.000, r'$\phi(x_{\mathrm{ood}})$'),
    ]
    for stream_color, y_base, label in stack_specs:
        for j in range(n_seg):
            ax.add_patch(patches.Rectangle(
                (vx + (seg_w + seg_gap) * j, y_base), seg_w, 0.06,
                facecolor=stream_color, edgecolor='white', linewidth=0.6,
                alpha=0.88,
            ))
        ax.text(vx + stack_w + 0.004, y_base + 0.030, label,
                ha='left', va='center', fontsize=7.5, color=stream_color)

    # Step 5: t-SNE + score equation.
    inset_x0 = 0.60
    concat_end = vx + stack_w + 0.045
    ax.annotate('', xy=(inset_x0 - 0.010, band_y_arrow),
                xytext=(concat_end, band_y_arrow),
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#1A202C'))
    ax.text(inset_x0, 0.99,
            r'(5) Score',
            fontsize=9.5, ha='left', va='top', color='#1A202C',
            fontweight='bold')
    ax.text(inset_x0, 0.945,
            r'$\mathcal{S}_{\mathrm{MM++}}(x) = '
            r'-\min_{c}\, d_M^{2}\!\left(\phi(x),\,\hat{\mu}_{c}^{\mathcal{K}}\right)$',
            fontsize=8, ha='left', va='top', color='#1A202C')
    inset = ax.inset_axes([inset_x0, 0.02, 0.38, 0.68])
    _draw_tsne_inset(inset, tsne_data)


def _draw_matrix_thumb(inset, mat, label, label_color):
    inset.imshow(mat, cmap='RdBu_r', vmin=-np.max(np.abs(mat)),
                 vmax=np.max(np.abs(mat)), aspect='equal',
                 interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for s in inset.spines.values():
        s.set_edgecolor('#1A202C')
        s.set_linewidth(0.6)
    inset.set_title(label, fontsize=8, color=label_color, pad=2,
                    fontweight='bold')


def _style_small_inset(inset):
    inset.tick_params(axis='both', labelsize=6.5, pad=1.5,
                      colors='#1A202C', length=2.5, width=0.5)
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    for s in inset.spines.values():
        s.set_linewidth(0.6)
        s.set_edgecolor('#1A202C')


def _draw_gap_inset(inset, gaps, selected_idx, n_layers):
    gap_x = np.arange(1, n_layers)
    selected_set = {int(i) for i in selected_idx if int(i) > 0}
    final_idx = n_layers - 1
    bar_colors, bar_edges = [], []
    for gx in gap_x:
        if int(gx) in selected_set or int(gx) == final_idx:
            bar_colors.append(MAUVE)
            bar_edges.append(MAUVE_DARK)
        else:
            bar_colors.append('#C2D0E0')
            bar_edges.append('#7A8A9C')
    inset.bar(gap_x, gaps, color=bar_colors, edgecolor=bar_edges,
              linewidth=0.5, width=0.74)
    inset.set_xlabel(r'layer index $l$', fontsize=7.5, color='#1A202C',
                     labelpad=2)
    inset.set_ylabel(r'gap $\Delta_l$', fontsize=8, color='#1A202C',
                     labelpad=2)
    _style_small_inset(inset)
    inset.set_xlim(-0.5, n_layers - 0.5)
    inset.set_xticks(np.arange(0, n_layers, max(1, n_layers // 6)))


def _draw_hd_inset(inset, h_over_d, selected_idx, n_layers):
    layer_x = np.arange(n_layers)
    selected_set = {int(i) for i in selected_idx}
    selected_mask = np.array([int(i) in selected_set for i in layer_x])
    inset.plot(layer_x, h_over_d, color='#1A202C', linewidth=1.1, zorder=2)
    inset.scatter(layer_x[~selected_mask], h_over_d[~selected_mask],
                  s=12, color='#C2D0E0', edgecolor='#1A202C',
                  linewidth=0.4, zorder=3)
    if selected_mask.any():
        inset.scatter(layer_x[selected_mask], h_over_d[selected_mask],
                      s=22, color=MAUVE, edgecolor=MAUVE_DARK,
                      linewidth=0.6, zorder=4)
    inset.set_xlabel(r'layer index $l$', fontsize=7.5, color='#1A202C',
                     labelpad=2)
    inset.set_ylabel(r'$H/D$', fontsize=8, color='#1A202C', labelpad=2)
    _style_small_inset(inset)
    inset.set_xlim(-0.5, n_layers - 0.5)
    inset.set_xticks(np.arange(0, n_layers, max(1, n_layers // 5)))
    inset.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    inset.yaxis.get_offset_text().set_fontsize(6)
    inset.yaxis.get_offset_text().set_color('#1A202C')


def _draw_rank_inset(inset, eff_rank, selected_idx, n_layers):
    layer_x = np.arange(n_layers)
    selected_set = {int(i) for i in selected_idx}
    selected_mask = np.array([int(i) in selected_set for i in layer_x])
    inset.plot(layer_x, eff_rank, color='#1A202C', linewidth=1.1, zorder=2)
    inset.scatter(layer_x[~selected_mask], eff_rank[~selected_mask],
                  s=12, color='#C2D0E0', edgecolor='#1A202C',
                  linewidth=0.4, zorder=3)
    if selected_mask.any():
        inset.scatter(layer_x[selected_mask], eff_rank[selected_mask],
                      s=22, color=MAUVE, edgecolor=MAUVE_DARK,
                      linewidth=0.6, zorder=4)
    inset.set_xlabel(r'layer index $l$', fontsize=7.5, color='#1A202C',
                     labelpad=2)
    inset.set_ylabel(r'$\exp(H)$', fontsize=8, color='#1A202C', labelpad=2)
    _style_small_inset(inset)
    inset.set_xlim(-0.5, n_layers - 0.5)
    inset.set_xticks(np.arange(0, n_layers, max(1, n_layers // 5)))


def _draw_tsne_inset(inset, tsne_data):
    embed_id = np.asarray(tsne_data['embed_id'])
    embed_ood = np.asarray(tsne_data['embed_ood'])
    class_labels = np.asarray(tsne_data['id_class_labels'])
    classes = np.asarray(tsne_data['id_classes'])
    palette = ['#2CA02C', '#9467BD', '#17BECF']

    centroids = []
    for ci, c in enumerate(classes):
        mask = class_labels == int(c)
        pts = embed_id[mask]
        color = palette[ci % len(palette)]
        inset.scatter(pts[:, 0], pts[:, 1], s=10, color=color,
                      alpha=0.7, edgecolor='none', zorder=2)
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        inset.scatter([centroid[0]], [centroid[1]], s=58, marker='X',
                      facecolor=color, edgecolor='white', linewidth=0.9,
                      zorder=5)
        inset.text(centroid[0], centroid[1] + 0.6, f'class {int(c)}',
                   ha='center', va='bottom', fontsize=6.5, color=color,
                   fontweight='bold', zorder=6)
    centroids = np.array(centroids)

    inset.scatter(embed_ood[:, 0], embed_ood[:, 1], s=11, marker='^',
                  color=ORANGE, alpha=0.65, edgecolor='none', zorder=2,
                  label='OOD')

    densest = int(np.argmax([(class_labels == int(c)).sum() for c in classes]))
    chosen_pts = embed_id[class_labels == int(classes[densest])]
    cdists = np.linalg.norm(chosen_pts - centroids[densest], axis=1)
    id_pick = chosen_pts[np.argsort(cdists)[len(cdists) // 4]]
    id_nearest = centroids[densest]

    centroid_dists = np.linalg.norm(
        embed_ood[:, None, :] - centroids[None, :, :], axis=2)
    ood_pick_idx = int(np.argmax(centroid_dists.min(axis=1)))
    ood_pick = embed_ood[ood_pick_idx]
    ood_nearest = centroids[int(np.argmin(np.linalg.norm(
        centroids - ood_pick, axis=1)))]

    inset.scatter([id_pick[0]], [id_pick[1]], s=110, facecolor='none',
                  edgecolor=BLUE, linewidth=1.6, zorder=6)
    inset.scatter([ood_pick[0]], [ood_pick[1]], s=110, facecolor='none',
                  edgecolor='#A0451B', linewidth=1.6, marker='s', zorder=6)
    inset.plot([id_pick[0], id_nearest[0]], [id_pick[1], id_nearest[1]],
               color=BLUE, linewidth=1.0, linestyle='--', zorder=4)
    inset.plot([ood_pick[0], ood_nearest[0]], [ood_pick[1], ood_nearest[1]],
               color=ORANGE, linewidth=1.0, linestyle='--', zorder=4)

    id_mid = 0.5 * (id_pick + id_nearest)
    ood_mid = 0.5 * (ood_pick + ood_nearest)
    inset.text(id_mid[0], id_mid[1] - 0.4,
               r'small $d_M^2$' '\nID: high score',
               fontsize=6.8, color=BLUE, ha='center', va='top', zorder=6,
               bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                         edgecolor='none', alpha=0.85))
    inset.text(ood_mid[0], ood_mid[1] + 0.4,
               r'large $d_M^2$' '\nOOD: low score',
               fontsize=6.8, color='#A0451B', ha='center', va='bottom',
               zorder=6,
               bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                         edgecolor='none', alpha=0.85))

    pad = 0.06
    all_pts = np.concatenate([embed_id, embed_ood], axis=0)
    xlo, xhi = all_pts[:, 0].min(), all_pts[:, 0].max()
    ylo, yhi = all_pts[:, 1].min(), all_pts[:, 1].max()
    dx, dy = xhi - xlo, yhi - ylo
    span = max(dx, dy)
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    half = (1 + pad) * span / 2
    inset.set_xlim(cx - half, cx + half)
    inset.set_ylim(cy - half, cy + half)
    inset.set_aspect('equal', adjustable='box')
    inset.set_box_aspect(1)
    inset.set_xlabel('t-SNE dim 1', fontsize=7, color='#1A202C', labelpad=2)
    inset.set_ylabel('t-SNE dim 2', fontsize=7, color='#1A202C', labelpad=2)
    inset.tick_params(axis='both', labelsize=6, pad=1.5,
                      colors='#1A202C', length=2.5, width=0.5)
    # Drop the surrounding box: keep only bottom and left spines.
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    for sname in ('left', 'bottom'):
        s = inset.spines[sname]
        s.set_edgecolor('#1A202C')
        s.set_linewidth(0.6)


def collect_fpr_summary(cache_root, model):
    rows = OrderedDict()
    datasets = DATASET_ORDER + [
        ds for ds in sorted(os.listdir(os.path.join(cache_root, 'scores', model)))
        if ds not in DATASET_ORDER
    ]
    for dataset in datasets:
        try:
            results, _ = load_score_results(cache_root, model, dataset)
            base_id, base_ood = method_scores(results, BASELINE_KEY)
            mm_id, mm_ood = method_scores(results, MMPP_KEY)
        except (FileNotFoundError, KeyError, NotADirectoryError):
            continue
        base_fpr, _ = fpr_at_tpr(base_id, base_ood)
        mm_fpr, _ = fpr_at_tpr(mm_id, mm_ood)
        rows[dataset] = (100.0 * base_fpr, 100.0 * mm_fpr)
    return rows


def draw_summary_axis(ax, fpr_rows=None, dataset=None):
    rows = TABLE_FPR_VALUES if fpr_rows is None else fpr_rows
    shown = list(rows.keys())
    x = np.arange(len(shown))
    base = np.asarray([rows[ds][0] for ds in shown])
    mm = np.asarray([rows[ds][1] for ds in shown])
    width = 0.36
    ax.bar(x - width / 2, base, width=width, color='#B9C0CA',
           edgecolor='#1A202C', linewidth=0.7, label='Maha++')
    ax.bar(x + width / 2, mm, width=width, color=MAUVE,
           edgecolor=MAUVE_DARK, linewidth=0.7, label='MM++ (Ours)')
    ax.set_ylabel('FPR@95 (%)', labelpad=2, color='#1A202C')
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_short_title(ds) for ds in shown],
                       rotation=0, ha='center', fontsize=7, color='#1A202C')
    ax.tick_params(axis='y', colors='#1A202C')
    y_max = max(80.0, float(max(base.max(), mm.max()) + 12.0))
    ax.set_ylim(0, y_max)
    style_axis(ax)
    ax.legend(loc='upper left', frameon=False, ncol=1, handlelength=1.0,
              columnspacing=0.7, fontsize=7.5, labelcolor='#1A202C',
              borderpad=0.2)
    ax.text(0.0, 1.02, 'FPR@95 on ImageNet-LT OOD benchmarks (lower is better)',
            transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            color='#1A202C', ha='left', va='bottom')

    biggest = int(np.argmax(base - mm))
    diff = base[biggest] - mm[biggest]
    text_x = max(0.5, biggest - 0.6)
    text_y = min(y_max - 6, max(mm[biggest] + 22, 32))
    ax.annotate(
        f"{diff:.1f} pt lower",
        xy=(biggest + width / 2, mm[biggest] + 1),
        xytext=(text_x, text_y),
        arrowprops=dict(arrowstyle='->', lw=0.8, color=MAUVE_DARK),
        fontsize=8.5, color=MAUVE_DARK, ha='center',
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--out-prefix',
                        default=os.path.join(OUT_DIR, 'mmpp_teaser_convnext'))
    return parser.parse_args()


def main():
    args = parse_args()

    results, _ = load_score_results(args.cache_root, args.model, args.dataset)
    base_id, base_ood = method_scores(results, BASELINE_KEY)
    mm_id, mm_ood = method_scores(results, MMPP_KEY)
    base_panel = score_panel_data(base_id, base_ood)
    mm_panel = score_panel_data(mm_id, mm_ood)

    layer_names = load_layer_names(args.cache_root, args.model)
    H, h_over_d, eff_rank, gaps = layer_rank_metrics(
        args.cache_root, args.model, layer_names)
    selected_idx = selected_layers_from_gaps(gaps, len(layer_names), args.k)
    cov_thumb, prec_thumb, prec_layer_name = precision_thumbnails(
        args.cache_root, args.model, layer_names, selected_idx, size=22)
    tsne_data = compute_tsne_data(
        args.cache_root, args.model, args.dataset, selected_idx, layer_names,
    )

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.95, 2.25, 1.05],
        height_ratios=[1.0, 1.0],
        left=0.025, right=0.992, bottom=0.10, top=0.94,
        wspace=0.22, hspace=0.55,
    )

    ax_base_concept = fig.add_subplot(gs[0, 0])
    ax_method = fig.add_subplot(gs[:, 1])
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_mm = fig.add_subplot(gs[1, 2])

    draw_final_layer_axis(ax_base_concept, layer_names)
    draw_method_axis(ax_method, gaps, h_over_d, eff_rank, selected_idx,
                     layer_names, args.k, cov_thumb, prec_thumb, tsne_data)
    draw_summary_axis(ax_summary)
    draw_score_axis(
        ax_base, base_panel,
        'Last-layer Maha++ scores',
        r'standardized $S_{\mathrm{Maha++}}$',
    )
    draw_score_axis(
        ax_mm, mm_panel,
        'MM++ fused scores',
        r'standardized $S_{\mathrm{MM++}}$',
    )

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    png_path = f'{args.out_prefix}.png'
    pdf_path = f'{args.out_prefix}.pdf'
    svg_path = f'{args.out_prefix}.svg'
    xml_path = f'{args.out_prefix}.xml'
    fig.savefig(png_path, dpi=450, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=450, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(xml_path, bbox_inches='tight')
    plt.close(fig)

    print(f'saved: {png_path}')
    print(f'saved: {pdf_path}')
    print(f'saved: {svg_path}')
    print(f'baseline FPR@95: {100.0 * base_panel["fpr"]:.2f}')
    print(f'MM++ FPR@95: {100.0 * mm_panel["fpr"]:.2f}')


if __name__ == '__main__':
    main()
