"""Plot MM++ AUROC sensitivity to the number of fused layers K.

Default cache roots:
  * ResNet-50: cache/rahim_s_caches/cache_imagenetlt
  * Other architectures: cache_imgnetlt

The plotted method matches the cached "MM_plus_plus_topk_cat" variant:
select the final penultimate representation plus K-1 layers with the largest
within-class log-rank gaps, concatenate L2-normalized selected features, fit a
joint class-conditional Ledoit-Wolf precision, and score with Mahalanobis.
"""
import argparse
import csv
import json
import os
import time
from glob import glob

import numpy as np


os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7.5,
    'xtick.labelsize': 5.8,
    'ytick.labelsize': 6.4,
    'legend.fontsize': 6.0,
    'axes.linewidth': 0.55,
    'xtick.major.width': 0.55,
    'ytick.major.width': 0.55,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get('IMAGENETLT_CACHE', os.path.join(REPO_ROOT, 'cache_imgnetlt'))
RESNET_CACHE_CANDIDATES = [
    os.path.join(REPO_ROOT, 'cache', 'rahim_s_caches', 'cache_imagenetlt'),
    os.path.join(REPO_ROOT, 'rahim_s_caches', 'cache_imagenetlt'),
]
if os.environ.get('RESNET_IMAGENETLT_CACHE'):
    RESNET_CACHE_CANDIDATES.insert(0, os.environ['RESNET_IMAGENETLT_CACHE'])
RESNET_CACHE = next(
    (path for path in RESNET_CACHE_CANDIDATES if os.path.isdir(path)),
    RESNET_CACHE_CANDIDATES[0],
)
OUT_DIR = os.path.join(REPO_ROOT, 'assets')
DEFAULT_OUT_PREFIX = os.path.join(OUT_DIR, 'k_sensitivity_imagenetlt')
LEGACY_METRIC_CACHE = os.path.join(OUT_DIR, 'k_sensitivity_architectures_ood.metrics.json')

MODEL_TITLES = {
    'resnet50.tv2_in1k': 'ResNet-50',
    'convnext_tiny.fb_in1k': 'ConvNeXt-T',
    'swin_tiny_patch4_window7_224.ms_in1k': 'Swin-T',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'ViT-B/16',
}

MODEL_FIGURE_SLUGS = {
    'resnet50.tv2_in1k': 'resnet50',
    'convnext_tiny.fb_in1k': 'convnext_t',
    'swin_tiny_patch4_window7_224.ms_in1k': 'swint',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'vitb_16',
}

MODEL_ORDER = [
    'resnet50.tv2_in1k',
    'convnext_tiny.fb_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
]

COMBINED_MODEL_ORDER = [
    'resnet50.tv2_in1k',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    'convnext_tiny.fb_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
]

MODEL_CACHE_ROOTS = {
    'resnet50.tv2_in1k': RESNET_CACHE,
}

OOD_DATASETS = [
    'NINCO_OOD_classes',
    'openimages_o',
    'imagenet_r',
    'imagenet_c',
    'imagenet_es',
    'imagenet_v2',
]

DATASET_TITLES = {
    'NINCO_OOD_classes': 'NINCO',
    'openimages_o': 'OpenImage-O',
    'imagenet_r': 'ImageNet-R',
    'imagenet_c': 'ImageNet-C',
    'imagenet_es': 'ImageNet-ES',
    'imagenet_v2': 'ImageNet-V2',
    'imagenet_o': 'ImageNet-O',
}

CURVE_STYLES = {
    'NINCO_OOD_classes': {'color': '#4477AA', 'marker': 'o'},  # blue
    'openimages_o': {'color': '#EE6677', 'marker': 's'},       # coral
    'imagenet_r': {'color': '#228833', 'marker': 'D'},         # green
    'imagenet_c': {'color': '#CCBB44', 'marker': 'v'},         # yellow
    'imagenet_es': {'color': '#66CCEE', 'marker': 'P'},        # cyan
    'imagenet_v2': {'color': '#AA3377', 'marker': 'X'},        # purple
    'imagenet_o': {'color': '#BBBBBB', 'marker': '*'},         # gray
}

LEGEND_MODELS = {
    'resnet50.tv2_in1k',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
}

GRID = '#D7DCE2'
METRIC_CACHE_VERSION = 1
BEST_MARKER = 'H'  # Matplotlib's rotated hexagon marker.
BEST_MARKERSIZE = 4.8


def cache_root(model):
    return MODEL_CACHE_ROOTS.get(model, CACHE)


def model_title(model):
    return MODEL_TITLES.get(model, model.replace('_', r'\_'))


def dataset_title(dataset):
    return DATASET_TITLES.get(dataset, dataset.replace('_', '-'))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_layer_names(model):
    root = cache_root(model)
    for subdir in ('cache_val_inter', 'cache_train_inter'):
        path = os.path.join(root, subdir, model, 'layer_names.json')
        if os.path.exists(path):
            return load_json(path)
    raise FileNotFoundError(f'No layer_names.json found for {model}')


def sorted_shards(directory, pattern):
    paths = glob(os.path.join(directory, pattern))
    return sorted(paths, key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))


def load_sharded_array(directory, pattern, dtype=None):
    paths = sorted_shards(directory, pattern)
    if not paths:
        raise FileNotFoundError(f'No shards matching {pattern} in {directory}')
    arrays = [np.load(path) for path in paths]
    out = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    return out.astype(dtype, copy=False) if dtype is not None else out


def load_train_labels(model):
    train_dir = os.path.join(cache_root(model), 'cache_train', model)
    return load_sharded_array(train_dir, 'labels_true_*.npy').astype(int, copy=False)


def load_layer_features(root, model, split, layer_name, dataset=None, dtype=np.float64):
    if split == 'ood':
        directory = os.path.join(root, 'cache_ood_inter', model, dataset)
    else:
        directory = os.path.join(root, f'cache_{split}_inter', model)
    return load_sharded_array(directory, f'layer_{layer_name}_features_*.npy', dtype=dtype)


def l2_normalize(features):
    norms = np.linalg.norm(features, axis=-1, keepdims=True).clip(min=1e-10)
    return features / norms


def log_rank_gaps(model, layer_names):
    method_dir = os.path.join(cache_root(model), 'cache_methods', model)
    rel_rank = []
    for layer_name in layer_names:
        prec = np.load(os.path.join(method_dir, f'mm_pp_{layer_name}_prec.npy')).astype(np.float64)
        eig = np.linalg.eigvalsh(prec).clip(min=1e-10)
        cov_eig = (1.0 / eig).clip(min=1e-8)
        lam = cov_eig / cov_eig.sum()
        entropy = -float(np.sum(lam * np.log(lam.clip(min=1e-300))))
        rel_rank.append(entropy / max(prec.shape[0], 1))
    rel_rank = np.asarray(rel_rank, dtype=np.float64)
    return rel_rank[:-1] - rel_rank[1:]


def selected_layer_indices(model, layer_names, k):
    """Final penultimate layer plus earlier layers ranked by log-rank gaps."""
    layer_count = len(layer_names)
    k_eff = min(max(1, k), layer_count)
    final_idx = layer_count - 1
    if k_eff == 1:
        return np.array([final_idx], dtype=int), k_eff

    delta = log_rank_gaps(model, layer_names)
    candidates = np.array([i for i in range(len(delta)) if (i + 1) != final_idx])
    k_extra = min(k_eff - 1, len(candidates))
    if k_extra == 0:
        extra = np.array([], dtype=int)
    else:
        top = np.argsort(delta[candidates])[-k_extra:]
        extra = candidates[top] + 1

    selected = np.unique(np.concatenate([[final_idx], extra])).astype(int)
    if len(selected) < k_eff:
        remaining = [i for i in range(layer_count) if i not in set(selected.tolist())]
        selected = np.concatenate([selected, remaining[:k_eff - len(selected)]])
    return np.sort(selected), k_eff


def fused_tag(k_eff):
    return f'mm_pp_topk{k_eff}_delta_paper_cat_uw'


def load_layer_stats(model, layer_names):
    method_dir = os.path.join(cache_root(model), 'cache_methods', model)
    means = []
    dims = []
    for layer_name in layer_names:
        mean = np.load(os.path.join(method_dir, f'mm_pp_{layer_name}_mean.npy')).astype(np.float64)
        means.append(mean)
        dims.append(mean.shape[1])
    return means, dims


def build_fused_features(root, model, split, layer_names, sel_idx, dims, dataset=None):
    sample_layer = layer_names[sel_idx[0]]
    sample = load_layer_features(root, model, split, sample_layer, dataset=dataset, dtype=np.float64)
    n_samples = sample.shape[0]
    fused = np.zeros((n_samples, int(sum(dims[i] for i in sel_idx))), dtype=np.float64)

    col = 0
    for i in sel_idx:
        layer_name = layer_names[i]
        feats = sample if layer_name == sample_layer else load_layer_features(
            root, model, split, layer_name, dataset=dataset, dtype=np.float64)
        dim = dims[i]
        fused[:, col:col + dim] = l2_normalize(feats)
        col += dim
    return fused


def fit_or_load_fused_stats(model, layer_names, sel_idx, k_eff, train_labels, force=False):
    from sklearn.covariance import LedoitWolf

    root = cache_root(model)
    method_dir = os.path.join(root, 'cache_methods', model)
    tag = fused_tag(k_eff)
    means_path = os.path.join(method_dir, f'{tag}_means.npy')
    prec_path = os.path.join(method_dir, f'{tag}_prec.npy')

    if not force and os.path.exists(means_path) and os.path.exists(prec_path):
        return np.load(means_path), np.load(prec_path)

    layer_means, dims = load_layer_stats(model, layer_names)
    fused_means = np.concatenate([layer_means[i] for i in sel_idx], axis=1)
    fused_train = build_fused_features(root, model, 'train', layer_names, sel_idx, dims)
    centered = fused_train - fused_means[train_labels]

    print(f'[{model}] fitting {tag}: K={k_eff}, dim={centered.shape[1]}')
    lw = LedoitWolf(assume_centered=True)
    lw.fit(centered)
    precision = lw.precision_

    np.save(means_path, fused_means)
    np.save(prec_path, precision)
    return fused_means, precision


def fast_maha_scores(features, means, precision):
    p_mu = means @ precision
    mu_p_mu = np.sum(means * p_mu, axis=1)
    x_p = features @ precision
    x_p_x = np.sum(features * x_p, axis=1)
    distances = x_p_x[:, None] - 2.0 * (x_p @ means.T) + mu_p_mu[None, :]
    return -np.min(distances, axis=1)


def id_scores(model, layer_names, sel_idx, k_eff, means, precision, force=False):
    root = cache_root(model)
    method_dir = os.path.join(root, 'cache_methods', model)
    score_path = os.path.join(method_dir, f'{fused_tag(k_eff)}_id_scores.npy')
    if not force and os.path.exists(score_path):
        return np.load(score_path)

    _, dims = load_layer_stats(model, layer_names)
    val_features = build_fused_features(root, model, 'val', layer_names, sel_idx, dims)
    scores = fast_maha_scores(val_features, means, precision)
    np.save(score_path, scores)
    return scores


def ood_scores(model, dataset, layer_names, sel_idx, means, precision):
    root = cache_root(model)
    _, dims = load_layer_stats(model, layer_names)
    features = build_fused_features(root, model, 'ood', layer_names, sel_idx, dims, dataset=dataset)
    return fast_maha_scores(features, means, precision)


def load_ood_labels(model, dataset):
    directory = os.path.join(cache_root(model), 'cache_ood', model, dataset)
    try:
        return load_sharded_array(directory, 'labels_true_*.npy')
    except FileNotFoundError:
        return None


def sample_auroc(values_id, values_ood):
    from sklearn.metrics import roc_auc_score

    y_true = [1] * len(values_id) + [0] * len(values_ood)
    y_score = np.nan_to_num(np.concatenate([values_id, values_ood]).ravel())
    return float(roc_auc_score(y_true, y_score))


def classwise_auroc(values_id, values_ood, labels_ood):
    if labels_ood is None:
        return sample_auroc(values_id, values_ood)
    aurocs = []
    for label in np.unique(labels_ood):
        mask = labels_ood == label
        if np.any(mask):
            aurocs.append(sample_auroc(values_id, values_ood[mask]))
    return float(np.mean(aurocs)) if aurocs else np.nan


def dataset_available(model, dataset):
    root = cache_root(model)
    return os.path.isdir(os.path.join(root, 'cache_ood_inter', model, dataset))


def metric_kind(classwise):
    return 'classwise_auroc' if classwise else 'sample_auroc'


def metric_cache_key(model, dataset, k, classwise):
    return '\t'.join([metric_kind(classwise), model, dataset, str(k)])


def empty_metric_cache():
    return {
        'version': METRIC_CACHE_VERSION,
        'rows': {},
    }


def load_metric_cache(path):
    if not path or not os.path.exists(path):
        return empty_metric_cache()
    try:
        with open(path) as f:
            cache = json.load(f)
    except json.JSONDecodeError:
        print(f'[warning] could not decode metric cache, starting fresh: {path}')
        return empty_metric_cache()
    if not isinstance(cache, dict):
        return empty_metric_cache()
    cache.setdefault('version', METRIC_CACHE_VERSION)
    cache.setdefault('rows', {})
    if not isinstance(cache['rows'], dict):
        cache['rows'] = {}
    return cache


def write_metric_cache(cache, path):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f'{path}.{os.getpid()}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp_path, path)


def default_metric_cache_path(out_prefix):
    default_path = f'{out_prefix}.metrics.json'
    if out_prefix == DEFAULT_OUT_PREFIX and os.path.exists(LEGACY_METRIC_CACHE):
        return LEGACY_METRIC_CACHE
    return default_path


def serialize_auroc(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def row_from_metric_entry(entry):
    return {
        'model': entry['model'],
        'dataset': entry['dataset'],
        'K': int(entry['K']),
        'effective_K': int(entry['effective_K']),
        'auroc': np.nan if entry.get('auroc') is None else float(entry['auroc']),
    }


def cached_metric_row(cache, model, dataset, k, k_eff, sel_idx, classwise):
    entry = cache.get('rows', {}).get(metric_cache_key(model, dataset, k, classwise))
    if not entry:
        return None
    if int(entry.get('effective_K', -1)) != int(k_eff):
        return None
    if entry.get('selected_layer_indices') != [int(i) for i in sel_idx]:
        return None
    return row_from_metric_entry(entry)


def store_metric_row(cache, path, row, sel_idx, classwise):
    key = metric_cache_key(row['model'], row['dataset'], row['K'], classwise)
    cache['rows'][key] = {
        'metric_kind': metric_kind(classwise),
        'model': row['model'],
        'dataset': row['dataset'],
        'K': int(row['K']),
        'effective_K': int(row['effective_K']),
        'selected_layer_indices': [int(i) for i in sel_idx],
        'auroc': serialize_auroc(row['auroc']),
        'updated_at_unix': time.time(),
    }
    write_metric_cache(cache, path)


def print_metric_status(row, source):
    auroc = row['auroc']
    auroc_text = 'nan' if not np.isfinite(auroc) else f'{100 * auroc:.2f}'
    print(
        f"{row['model']} | {row['dataset']} | K={row['K']} "
        f"(effective {row['effective_K']}) | AUROC={auroc_text} [{source}]"
    )


def style_axis(ax):
    ax.grid(True, axis='y', linestyle=':', color=GRID, linewidth=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', pad=1.5)
    ax.tick_params(axis='y', pad=1.5)


def model_slug(model):
    return MODEL_FIGURE_SLUGS.get(
        model,
        ''.join(c if c.isalnum() else '_' for c in model).strip('_'),
    )


def figure_prefix(out_prefix, model):
    return f'{out_prefix}_{model_slug(model)}_auroc'


def combined_figure_prefix(out_prefix):
    return f'{out_prefix}_combined_auroc'


def series_overlaps_upper_right(series_by_dataset):
    finite_series = [
        y for series in series_by_dataset.values()
        for y in series
        if np.isfinite(y)
    ]
    if not finite_series:
        return False

    y_min = min(finite_series)
    y_max = max(finite_series)
    if y_min == y_max:
        return False

    top_threshold = y_min + 0.68 * (y_max - y_min)
    right_start = max(1, int(np.ceil(0.65 * max(len(next(iter(series_by_dataset.values()))), 1))))
    for series in series_by_dataset.values():
        for x_idx, y in enumerate(series, start=1):
            if x_idx >= right_start and np.isfinite(y) and y >= top_threshold:
                return True
    return False


def collect_results(models, datasets, max_k, classwise, force, metric_cache_path=None):
    metric_cache = load_metric_cache(metric_cache_path) if metric_cache_path else None
    rows = []
    for model in models:
        layer_names = load_layer_names(model)
        train_labels = load_train_labels(model)
        model_max_k = len(layer_names) if max_k is None else min(max_k, len(layer_names))
        for k in range(1, model_max_k + 1):
            sel_idx, k_eff = selected_layer_indices(model, layer_names, k)
            cached_rows = {}
            missing_datasets = []
            if metric_cache is not None and not force:
                for dataset in datasets:
                    row = cached_metric_row(
                        metric_cache, model, dataset, k, k_eff, sel_idx, classwise)
                    if row is None:
                        missing_datasets.append(dataset)
                    else:
                        cached_rows[dataset] = row
            else:
                missing_datasets = list(datasets)

            available_missing = {
                dataset: dataset_available(model, dataset)
                for dataset in missing_datasets
            }
            needs_scores = any(available_missing.values())
            if needs_scores:
                means, precision = fit_or_load_fused_stats(
                    model, layer_names, sel_idx, k_eff, train_labels, force=force)
                scores_id = id_scores(
                    model, layer_names, sel_idx, k_eff, means, precision, force=force)

            for dataset in datasets:
                if dataset in cached_rows:
                    row = cached_rows[dataset]
                    rows.append(row)
                    print_metric_status(row, 'cached')
                    continue

                if not available_missing.get(dataset, False):
                    row = {
                        'model': model,
                        'dataset': dataset,
                        'K': k,
                        'effective_K': k_eff,
                        'auroc': np.nan,
                    }
                    rows.append(row)
                    print_metric_status(row, 'missing')
                    continue

                scores_ood = ood_scores(model, dataset, layer_names, sel_idx, means, precision)
                labels_ood = load_ood_labels(model, dataset) if classwise else None
                auroc = classwise_auroc(scores_id, scores_ood, labels_ood) if classwise else sample_auroc(
                    scores_id, scores_ood)
                row = {
                    'model': model,
                    'dataset': dataset,
                    'K': k,
                    'effective_K': k_eff,
                    'auroc': auroc,
                }
                rows.append(row)
                if metric_cache is not None:
                    store_metric_row(metric_cache, metric_cache_path, row, sel_idx, classwise)
                print_metric_status(row, 'computed')
    return rows


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'dataset', 'K', 'effective_K', 'auroc'])
        writer.writeheader()
        writer.writerows(rows)


def model_rows(rows, model):
    return [row for row in rows if row['model'] == model]


def model_max_k_from_rows(rows, model):
    rows_for_model = model_rows(rows, model)
    if not rows_for_model:
        return None
    return max(int(row['K']) for row in rows_for_model)


def draw_model_axis(
    ax, rows, model, datasets, show_legend, show_title=True,
    show_ylabel=True, show_xlabel=True,
):
    model_max_k = model_max_k_from_rows(rows, model)
    if model_max_k is None:
        ax.set_visible(False)
        return False

    series_by_dataset = {}
    for dataset in datasets:
        ys = []
        for k in range(1, model_max_k + 1):
            match = [
                row for row in rows
                if row['model'] == model and row['dataset'] == dataset and row['K'] == k
            ]
            ys.append(100.0 * match[0]['auroc'] if match and np.isfinite(match[0]['auroc']) else np.nan)
        series_by_dataset[dataset] = ys
        style = CURVE_STYLES.get(dataset, {})
        ax.plot(
            range(1, model_max_k + 1), ys,
            color=style.get('color'),
            marker=style.get('marker', 'o'),
            markersize=2.3,
            linewidth=0.9,
            markeredgewidth=0.35,
            label=dataset_title(dataset),
        )
        finite = np.isfinite(ys)
        if np.any(finite):
            finite_ys = np.asarray(ys, dtype=np.float64)
            best_idx = int(np.nanargmax(finite_ys))
            best_k = best_idx + 1
            ax.plot(
                [best_k], [finite_ys[best_idx]],
                color=style.get('color'),
                marker=BEST_MARKER,
                markersize=BEST_MARKERSIZE,
                linestyle='None',
                markeredgecolor='black',
                markeredgewidth=0.45,
                zorder=4,
            )
    if show_title:
        ax.set_title(model_title(model), pad=2)
    ax.set_xlim(0.7, model_max_k + 0.3)
    xtick_step = 1 if model_max_k <= 8 else (2 if model_max_k <= 19 else 4)
    ax.set_xticks(list(range(1, model_max_k + 1, xtick_step)))
    style_axis(ax)
    if show_ylabel:
        ax.set_ylabel(r'AUROC (%)')
    if show_xlabel:
        ax.set_xlabel(r'Number of fused layers, $K$')

    if show_legend:
        if series_overlaps_upper_right(series_by_dataset):
            legend_kwargs = {
                'loc': 'upper left',
                'bbox_to_anchor': (1.02, 1.0),
                'borderaxespad': 0.0,
            }
        else:
            legend_kwargs = {'loc': 'upper right'}

        ax.legend(
            handles=ax.get_legend_handles_labels()[0] + [
                Line2D(
                    [0], [0],
                    marker=BEST_MARKER,
                    markersize=BEST_MARKERSIZE,
                    linestyle='None',
                    markerfacecolor='white',
                    markeredgecolor='black',
                    markeredgewidth=0.45,
                    label='Best across $K$',
                ),
            ],
            frameon=False, handlelength=1.5, handletextpad=0.35,
            labelspacing=0.25, borderpad=0.15, **legend_kwargs,
        )
    return True


def save_figure(fig, path_prefix):
    png_path = f'{path_prefix}.png'
    pdf_path = f'{path_prefix}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {png_path}')
    print(f'saved: {pdf_path}')


def plot_individual_results(rows, models, datasets, out_prefix):
    for model in models:
        fig, ax = plt.subplots(figsize=(2.75, 2.0), constrained_layout=True)
        if not draw_model_axis(
            ax, rows, model, datasets,
            show_legend=(model in LEGEND_MODELS),
            show_title=False,
        ):
            plt.close(fig)
            continue
        fig_prefix = figure_prefix(out_prefix, model)
        save_figure(fig, fig_prefix)


def plot_combined_results(rows, models, datasets, out_prefix):
    ordered_models = [model for model in COMBINED_MODEL_ORDER if model in models]
    if not ordered_models:
        return

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, model in zip(axes, ordered_models):
        draw_model_axis(
            ax, rows, model, datasets,
            show_legend=(model in LEGEND_MODELS),
            show_ylabel=(ax in (axes[0], axes[2])),
            show_xlabel=(ax in (axes[2], axes[3])),
        )
    for ax in axes[len(ordered_models):]:
        ax.set_visible(False)
    save_figure(fig, combined_figure_prefix(out_prefix))


def plot_results(rows, models, datasets, max_k, out_prefix):
    del max_k
    plot_individual_results(rows, models, datasets, out_prefix)
    plot_combined_results(rows, models, datasets, out_prefix)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--max-k',
        type=int,
        default=None,
        help='Maximum K to plot per model. Defaults to each model\'s cached layer count.',
    )
    parser.add_argument('--models', nargs='+', default=MODEL_ORDER)
    parser.add_argument('--datasets', nargs='+', default=OOD_DATASETS)
    parser.add_argument(
        '--out-prefix',
        default=DEFAULT_OUT_PREFIX,
        help='Output path prefix without extension. Figures are saved as <out-prefix>_<model>_auroc.',
    )
    parser.add_argument(
        '--sample-auroc',
        action='store_true',
        help='Use sample-level AUROC instead of mean classwise AUROC.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Refit fused statistics, ID scores, and metrics even if cached files exist.',
    )
    parser.add_argument(
        '--metrics-cache',
        default=None,
        help='JSON cache for completed AUROC metrics. Defaults to the existing legacy cache when available.',
    )
    parser.add_argument(
        '--no-metrics-cache',
        action='store_true',
        help='Disable resumable AUROC metric caching.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    max_k = args.max_k
    if max_k is None:
        model_k_text = ', '.join(
            f'{model_title(model)}={len(load_layer_names(model))}'
            for model in args.models
        )
        print(f'auto max_k per model: {model_k_text}')
    metric_cache_path = None if args.no_metrics_cache else (
        args.metrics_cache or default_metric_cache_path(args.out_prefix))
    if metric_cache_path:
        print(f'using metric cache: {metric_cache_path}')
    rows = collect_results(
        models=args.models,
        datasets=args.datasets,
        max_k=max_k,
        classwise=not args.sample_auroc,
        force=args.force,
        metric_cache_path=metric_cache_path,
    )
    csv_path = f'{args.out_prefix}.csv'
    write_csv(rows, csv_path)
    print(f'saved: {csv_path}')
    plot_results(rows, args.models, args.datasets, max_k, args.out_prefix)


if __name__ == '__main__':
    main()
