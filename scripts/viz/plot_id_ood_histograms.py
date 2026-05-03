"""Before/after ID and OOD score distribution comparison across models.

Before: Maha++ baseline (Mahalanobis_norm), using the L2-normalized final layer.
After:  MM++ (MM_plus_plus_topk_cat), the fused multi-layer representation.

Each method is centered on its own ID mean, but both methods share the same
yardstick width — Maha++ ID's std:
    Maha++ ID/OOD -> (s - mean(Maha++ ID)) / std(Maha++ ID)
    MM++   ID/OOD -> (s - mean(MM++ ID))   / std(Maha++ ID)
Both ID curves are pinned at 0 (aligned); the OOD curves are then comparable
across methods in a common Maha++-ID-sigma unit. The OOD position relative to
0 is each method's "OOD shift" -- directly comparable since the unit is shared.
"""
import argparse
import glob
import os

import numpy as np


os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.dpi': 300,
})


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO_ROOT, 'cache_imgnetlt')
RESNET_CACHE = os.path.join(REPO_ROOT, 'rahim_s_caches', 'cache_imagenetlt')
XMAHA_CACHE = os.path.join(REPO_ROOT, 'cache', 'xmaha_scores_imagenetlt_vit')
OUT_DIR = os.path.join(REPO_ROOT, 'assets')

OOD_NAME = 'NINCO_OOD_classes'
VIT_B16_MODEL = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
CONVNEXT_T_MODEL = 'convnext_tiny.fb_in1k'
SWIN_T_MODEL = 'swin_tiny_patch4_window7_224.ms_in1k'

DEFAULT_DATASETS = [
    'NINCO_OOD_classes',
    'imagenet_r',
    'imagenet_c',
    'imagenet_es',
    'imagenet_o',
    'imagenet_v2',
]

MODEL_TITLES = {
    'resnet50.tv2_in1k': 'ResNet-50',
    'convnext_tiny.fb_in1k': 'ConvNeXt-T',
    'swin_tiny_patch4_window7_224.ms_in1k': 'Swin-T',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'ViT-B/16',
}

MODEL_SLUGS = {
    'resnet50.tv2_in1k': 'resnet50',
    CONVNEXT_T_MODEL: 'convnext_t',
    SWIN_T_MODEL: 'swin_t',
    VIT_B16_MODEL: 'vitb16',
}

MODEL_ORDER = [
    VIT_B16_MODEL,
    CONVNEXT_T_MODEL,
    SWIN_T_MODEL,
    # 'resnet50.tv2_in1k',
]

MODEL_CACHE_ROOTS = {
    'resnet50.tv2_in1k': RESNET_CACHE,
}

METHODS_TO_PLOT = [
    ('Mahalanobis_norm', 'Maha++', '#4C78A8'),
    ('MM_plus_plus_topk_cat', 'MM++', '#B279A2'),
]

BEFORE_KEY = 'Mahalanobis_norm'
AFTER_KEY = 'MM_plus_plus_topk_cat'

DATASET_TITLES = {
    'NINCO_OOD_classes': 'NINCO',
    'openimages_o': 'OpenImage-O',
    'ssb_hard': 'SSB-Hard',
    'imagenet_r': 'ImageNet-R',
    'imagenet_c': 'ImageNet-C',
    'imagenet_es': 'ImageNet-ES',
    'imagenet_v2': 'ImageNet-V2',
    'imagenet_o': 'ImageNet-O',
    'sun': 'SUN',
}

XMAHA_OOD_FILES = {
    'NINCO_OOD_classes': 'ood_scores_NINCO.npy',
    'imagenet_c': 'ood_scores_ImageNet-C.npy',
    'imagenet_es': 'ood_scores_ImageNet-ES.npy',
    'imagenet_o': 'ood_scores_ImageNet-O.npy',
    'imagenet_r': 'ood_scores_ImageNet-R.npy',
}

X_LOWER_BOUND = -10.0
CONVNEXT_X_LOWER_BOUND = -5.0
SWIN_X_LOWER_BOUND = -5.0
SWIN_X_LIMIT_DATASETS = {'NINCO_OOD_classes', 'imagenet_r', 'imagenet_o', 'imagenet_v2'}
VIT_STACKED_X_LOWER_BOUND = -5.0
VIT_STACKED_X_LIMIT_DATASETS = {'NINCO_OOD_classes', 'imagenet_r', 'imagenet_o'}

GRID = '#D7DCE2'


def cache_root(model):
    return MODEL_CACHE_ROOTS.get(model, CACHE)


def model_title(model):
    return MODEL_TITLES.get(model, model.replace('_', r'\_'))


def model_slug(model):
    return MODEL_SLUGS.get(model, model.replace('/', '_').replace('.', '_'))


def dataset_title(dataset):
    return DATASET_TITLES.get(dataset, dataset.replace('_', '-'))


def dataset_slug(dataset):
    return dataset.replace('/', '_').replace('.', '_')


def score_files(model, dataset):
    score_dir = os.path.join(cache_root(model), 'scores', model, dataset)
    files = sorted(glob.glob(os.path.join(score_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f'No score .npz files found in {score_dir}')
    return files


def load_results(model, dataset):
    results = {}
    paths = score_files(model, dataset)
    for path in paths:
        z = np.load(path, allow_pickle=True)
        results.update(z['methods_results'].item())
    print(f'[histograms] loaded {len(paths)} score file(s) for {model}')
    return results, paths


def xmaha_available(dataset):
    if dataset not in XMAHA_OOD_FILES:
        return False
    return (
        os.path.exists(os.path.join(XMAHA_CACHE, 'id_scores.npy')) and
        os.path.exists(os.path.join(XMAHA_CACHE, XMAHA_OOD_FILES[dataset]))
    )


def load_xmaha_scores(dataset):
    if dataset not in XMAHA_OOD_FILES:
        raise FileNotFoundError(f'No X-Maha cache mapping for {dataset}')
    id_path = os.path.join(XMAHA_CACHE, 'id_scores.npy')
    ood_path = os.path.join(XMAHA_CACHE, XMAHA_OOD_FILES[dataset])
    if not os.path.exists(id_path) or not os.path.exists(ood_path):
        raise FileNotFoundError(f'Missing X-Maha scores: {id_path}, {ood_path}')
    return (
        np.asarray(np.load(id_path), dtype=np.float64).ravel(),
        np.asarray(np.load(ood_path), dtype=np.float64).ravel(),
    )


def auroc(scores_id, scores_ood):
    scores = np.nan_to_num(np.concatenate([scores_id, scores_ood]).ravel())
    labels = np.concatenate([
        np.ones(len(scores_id), dtype=np.int8),
        np.zeros(len(scores_ood), dtype=np.int8),
    ])
    order = np.argsort(-scores, kind='mergesort')
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = np.r_[0.0, tps / max(tps[-1], 1), 1.0]
    fpr = np.r_[0.0, fps / max(fps[-1], 1), 1.0]
    return float(np.trapezoid(tpr, fpr))


def fpr_at_tpr(scores_id, scores_ood, tpr_target=0.95):
    threshold = np.quantile(np.nan_to_num(scores_id), 1 - tpr_target)
    return float((np.nan_to_num(scores_ood) >= threshold).mean())


def standardize_scores(scores, mu, sigma):
    return (scores - mu) / sigma


def kde_curve(samples, x_grid, bandwidth_factor=1.1):
    from scipy.stats import gaussian_kde

    samples = np.nan_to_num(samples).ravel()
    kde = gaussian_kde(samples, bw_method='silverman')
    kde.set_bandwidth(kde.factor * bandwidth_factor)
    return kde(x_grid)


def score_grid(*score_arrays):
    scores = np.concatenate([np.nan_to_num(arr).ravel() for arr in score_arrays])
    lo, hi = float(scores.min()), float(scores.max())
    pad = max(0.10 * (hi - lo), 0.25)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
        pad = 0.0
    return np.linspace(lo - pad, hi + pad, 900), lo - pad, hi + pad


def style_axis(ax):
    ax.grid(True, axis='y', linestyle=':', color=GRID, linewidth=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', pad=1.5)
    ax.tick_params(axis='y', pad=1.5)


def required_scores(results):
    missing = [key for key in (BEFORE_KEY, AFTER_KEY) if key not in results]
    if missing:
        raise KeyError(f'Missing method scores: {missing}. Available: {sorted(results)}')

    before_id = np.asarray(results[BEFORE_KEY]['scores_id'], dtype=np.float64)
    before_ood = np.asarray(results[BEFORE_KEY]['scores_ood'], dtype=np.float64)
    after_id = np.asarray(results[AFTER_KEY]['scores_id'], dtype=np.float64)
    after_ood = np.asarray(results[AFTER_KEY]['scores_ood'], dtype=np.float64)
    return before_id, before_ood, after_id, after_ood


CURVE_STYLES = {
    'maha_id':  {'fill': '#E8C173'},  # cream  (Maha++ ID — warm light)
    'mm_id':    {'fill': '#B58DA8'},  # mauve  (MM++ ID    — warm)
    'maha_ood': {'fill': '#9E9E9E'},  # gray   (Maha++ OOD — cool light)
    'mm_ood':   {'fill': '#8B95C9'},  # lavender (MM++ OOD — cool)
    'xmaha_id': {'fill': '#E8C173'},  # same palette as Maha++ ID
    'xmaha_ood': {'fill': '#9E9E9E'}, # same palette as Maha++ OOD
}


def standardized_curves(results):
    before_id, before_ood, after_id, after_ood = required_scores(results)
    before_mu = float(np.nanmean(before_id))
    after_mu = float(np.nanmean(after_id))
    sigma = max(float(np.nanstd(before_id)), 1e-10)  # shared Maha++ ID std
    return {
        'maha_id':  standardize_scores(before_id,  before_mu, sigma),
        'maha_ood': standardize_scores(before_ood, before_mu, sigma),
        'mm_id':    standardize_scores(after_id,   after_mu,  sigma),
        'mm_ood':   standardize_scores(after_ood,  after_mu,  sigma),
    }


def standardized_mm_xmaha_curves(results, xmaha_id, xmaha_ood):
    _, _, mm_id, mm_ood = required_scores(results)
    mm_mu = float(np.nanmean(mm_id))
    xmaha_mu = float(np.nanmean(xmaha_id))
    sigma = max(float(np.nanstd(mm_id)), 1e-10)  # shared MM++ ID std
    return {
        'mm_id':      standardize_scores(mm_id,      mm_mu,    sigma),
        'mm_ood':     standardize_scores(mm_ood,     mm_mu,    sigma),
        'xmaha_id':   standardize_scores(xmaha_id,   xmaha_mu, sigma),
        'xmaha_ood':  standardize_scores(xmaha_ood,  xmaha_mu, sigma),
    }


def draw_curve(ax, x, y, key, label):
    color = CURVE_STYLES[key]['fill']
    ax.fill_between(x, y, color=color, alpha=0.72, linewidth=0.8,
                    edgecolor=color, label=label)


def x_lower_bound(model, dataset, stacked=False):
    if stacked and model == VIT_B16_MODEL and dataset in VIT_STACKED_X_LIMIT_DATASETS:
        return VIT_STACKED_X_LOWER_BOUND
    if model == CONVNEXT_T_MODEL:
        return CONVNEXT_X_LOWER_BOUND
    if model == SWIN_T_MODEL and dataset in SWIN_X_LIMIT_DATASETS:
        return SWIN_X_LOWER_BOUND
    return X_LOWER_BOUND


def apply_dataset_xlim(ax, model, dataset, lo, hi, stacked=False):
    ax.set_xlim(x_lower_bound(model, dataset, stacked=stacked), hi)


def plot_curves_axis(ax, model, dataset, curves_to_plot, title=None, stacked=False):
    z = {key: samples for key, samples, _ in curves_to_plot}
    x, lo, hi = score_grid(*z.values())

    curves = {key: kde_curve(samples, x) for key, samples in z.items()}

    for key, _, label in curves_to_plot:
        draw_curve(ax, x, curves[key], key, label)

    density_hi = max(float(c.max()) for c in curves.values())
    apply_dataset_xlim(ax, model, dataset, lo, hi, stacked=stacked)
    ax.set_ylim(0.0, density_hi * 1.12)
    ax.set_ylabel('Density')
    if title is not None:
        ax.set_title(title, pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)


def plot_dataset_axis(ax, model, dataset, results, stacked=False):
    z = standardized_curves(results)
    curves_to_plot = [
        ('maha_id',  z['maha_id'],  'Maha++ ID'),
        ('maha_ood', z['maha_ood'], 'Maha++ OOD'),
        ('mm_id',    z['mm_id'],    'MM++ ID'),
        ('mm_ood',   z['mm_ood'],   'MM++ OOD'),
    ]
    plot_curves_axis(ax, model, dataset, curves_to_plot,
                     title=dataset_title(dataset), stacked=stacked)


def plot_xmaha_axis(ax, model, dataset, results, stacked=False):
    xmaha_id, xmaha_ood = load_xmaha_scores(dataset)
    z = standardized_mm_xmaha_curves(results, xmaha_id, xmaha_ood)
    curves_to_plot = [
        ('xmaha_id',   z['xmaha_id'],   'X-Maha ID'),
        ('xmaha_ood',  z['xmaha_ood'],  'X-Maha OOD'),
        ('mm_id',      z['mm_id'],      'MM++ ID'),
        ('mm_ood',     z['mm_ood'],     'MM++ OOD'),
    ]
    plot_curves_axis(ax, model, dataset, curves_to_plot, stacked=stacked)


def add_axis_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc='upper left', frameon=True, framealpha=0.95,
        edgecolor='0.6', fancybox=False, borderpad=0.5, labelspacing=0.45,
        handlelength=1.8, handleheight=1.2, fontsize=10
    )


def plot_model_across_datasets(model, datasets_with_results, out_dir,
                               include_xmaha_row=False, out_suffix=None):
    n = len(datasets_with_results)
    n_rows = 2 if include_xmaha_row else 1
    fig, axes = plt.subplots(n_rows, n, figsize=(3.6 * n, 3.55 * n_rows),
                             sharey=False, squeeze=False)

    for (dataset, results), ax in zip(datasets_with_results, axes[0]):
        plot_dataset_axis(ax, model, dataset, results, stacked=include_xmaha_row)

    if include_xmaha_row:
        for (dataset, results), ax in zip(datasets_with_results, axes[1]):
            plot_xmaha_axis(ax, model, dataset, results, stacked=True)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.08,
                        wspace=0.28, hspace=0.32)

    add_axis_legend(axes[0, -1])
    if include_xmaha_row:
        add_axis_legend(axes[1, -1])

    os.makedirs(out_dir, exist_ok=True)
    slug_parts = ['id_ood_score_histograms_imagenetlt', model_slug(model)]
    if n == 1:
        slug_parts.append(dataset_slug(datasets_with_results[0][0]))
    if out_suffix:
        slug_parts.append(out_suffix)
    out_prefix = os.path.join(out_dir, '_'.join(slug_parts))
    png_path = f'{out_prefix}.png'
    pdf_path = f'{out_prefix}.pdf'
    svg_path = f'{out_prefix}.svg'
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {png_path}')
    print(f'saved: {pdf_path}')
    print(f'saved: {svg_path}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=MODEL_ORDER)
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--no-xmaha-row', action='store_true',
                        help='Do not emit the extra ViT-B/16 MM++ vs X-Maha stacked figure.')
    return parser.parse_args()


def main():
    args = parse_args()
    for model in args.models:
        datasets_with_results = []
        for dataset in args.datasets:
            try:
                results, _ = load_results(model, dataset)
                required_scores(results)  # validate before plotting
                datasets_with_results.append((dataset, results))
            except (FileNotFoundError, KeyError) as exc:
                print(f'[skip] {model} / {dataset}: {exc}')
        if not datasets_with_results:
            print(f'[skip] {model}: no usable datasets')
            continue
        plot_model_across_datasets(model, datasets_with_results, args.out_dir)
        if model == VIT_B16_MODEL and not args.no_xmaha_row:
            xmaha_datasets = [
                (dataset, results)
                for dataset, results in datasets_with_results
                if xmaha_available(dataset)
            ]
            if xmaha_datasets:
                plot_model_across_datasets(
                    model, xmaha_datasets, args.out_dir,
                    include_xmaha_row=True, out_suffix='mmpp_xmaha_stacked'
                )
            else:
                print(f'[skip] {model}: no selected datasets have X-Maha caches')


if __name__ == '__main__':
    main()
