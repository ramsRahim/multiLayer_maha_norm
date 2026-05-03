"""Log-rank-gap diagnostics across ImageNet-LT model caches."""
import json
import os

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
    'legend.fontsize': 6.2,
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
CACHE = os.path.join(REPO_ROOT, 'cache_imgnetlt')
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

MODEL_TITLES = {
    'resnet50.tv2_in1k': 'ResNet-50',
    'convnext_tiny.fb_in1k': 'ConvNeXt-T',
    'swin_tiny_patch4_window7_224.ms_in1k': 'Swin-T',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'ViT-B/16',
}

MODEL_ORDER = [
    'resnet50.tv2_in1k',
    'convnext_tiny.fb_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
]

MODEL_CACHE_ROOTS = {
    'resnet50.tv2_in1k': RESNET_CACHE,
}

DISPLAY_ZERO_PAD = {
    'resnet50.tv2_in1k': 10,
}

BLUE_OUTLINE = '#6C8EBF'
MAUVE_OUTLINE = '#6F3F5F'
BLUE_FILL = '#DAE8FC'
MAUVE_FILL = '#B58DA8'
GRID = '#D7DCE2'
BAR_WIDTH = 0.035


def style_axis(ax):
    ax.grid(True, axis='y', linestyle=':', color=GRID, linewidth=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', pad=1.5)
    ax.tick_params(axis='y', pad=1.5)


def discover_models():
    usable = []
    skipped = []
    for model in MODEL_ORDER:
        cache_root = MODEL_CACHE_ROOTS.get(model, CACHE)
        score_dir = os.path.join(cache_root, 'scores', model)
        method_dir = os.path.join(cache_root, 'cache_methods', model)
        names_paths = [
            os.path.join(cache_root, 'cache_val_inter', model, 'layer_names.json'),
            os.path.join(cache_root, 'cache_train_inter', model, 'layer_names.json'),
        ]
        has_layer_names = any(os.path.exists(path) for path in names_paths)
        if os.path.isdir(score_dir) and os.path.isdir(method_dir) and has_layer_names:
            usable.append(model)
        else:
            skipped.append(model)
    if not usable:
        raise FileNotFoundError('No usable model caches found under cache_imgnetlt/scores.')
    return usable, skipped


def load_layer_names(model):
    cache_root = MODEL_CACHE_ROOTS.get(model, CACHE)
    for path in [
        os.path.join(cache_root, 'cache_val_inter', model, 'layer_names.json'),
        os.path.join(cache_root, 'cache_train_inter', model, 'layer_names.json'),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f'No layer_names.json found for {model}')


def compute_log_rank_gap(model, layer_names):
    """Return Delta_l = log(rho_{l-1}) - log(rho_l), indexed by l=2..L."""
    cache_root = MODEL_CACHE_ROOTS.get(model, CACHE)
    method_dir = os.path.join(cache_root, 'cache_methods', model)
    rel_rank = []
    for layer_name in layer_names:
        prec_path = os.path.join(method_dir, f'mm_pp_{layer_name}_prec.npy')
        prec = np.load(prec_path).astype(np.float64)
        eig = np.linalg.eigvalsh(prec).clip(min=1e-10)
        cov_eig = (1.0 / eig).clip(min=1e-8)
        lam = (cov_eig / cov_eig.sum()).clip(min=1e-300)
        entropy = -float(np.sum(lam * np.log(lam)))
        rel_rank.append(entropy / max(prec.shape[0], 1))

    rel_rank = np.asarray(rel_rank, dtype=np.float64)
    log_rel_rank = np.log(rel_rank.clip(min=1e-300))
    return log_rel_rank[:-1] - log_rel_rank[1:]


def selected_gap_layers(gap):
    final_idx = len(gap) - 1
    candidates = np.array(
        [idx for idx in range(len(gap)) if idx != final_idx],
        dtype=int,
    )
    selected = candidates[np.argmax(gap[candidates])]
    return selected, final_idx


def model_title(model):
    return MODEL_TITLES.get(model, model.replace('_', r'\_'))


def layer_tick_indices(num_layers, max_ticks=5):
    num_gaps = num_layers - 1
    n_ticks = min(max_ticks, num_gaps)
    return sorted(set(
        np.linspace(0, num_gaps - 1, n_ticks, dtype=int).tolist()
    ))


def layer_ticks_for_model(model, real_num_layers, display_num_layers):
    if model == 'resnet50.tv2_in1k':
        tick_indices = list(range(real_num_layers - 1))
    else:
        tick_indices = layer_tick_indices(display_num_layers)
    labels = [str(idx + 2) for idx in tick_indices]
    return tick_indices, labels


def display_gap(model, gap):
    pad = DISPLAY_ZERO_PAD.get(model, 0)
    if pad == 0:
        return gap
    return np.concatenate([gap, np.zeros(pad, dtype=gap.dtype)])


def main():
    models, skipped = discover_models()
    plot_data = []
    for model in models:
        layer_names = load_layer_names(model)
        gap = compute_log_rank_gap(model, layer_names)
        selected_idx, final_idx = selected_gap_layers(gap)
        gap_display = display_gap(model, gap)
        display_num_layers = len(gap_display) + 1
        plot_data.append((
            model, layer_names, gap_display, selected_idx, final_idx, display_num_layers
        ))

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.5, 1.75), sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    selected_layers = {}
    for ax, (model, layer_names, gap, selected_idx, final_idx, display_num_layers) in zip(axes, plot_data):
        selected_layer_num = selected_idx + 2
        selected_layers[model] = selected_layer_num

        x = np.linspace(0.0, 1.0, len(gap))
        fills = [BLUE_FILL] * len(gap)
        edges = [BLUE_OUTLINE] * len(gap)
        for idx in {selected_idx, final_idx}:
            fills[idx] = MAUVE_FILL
            edges[idx] = MAUVE_OUTLINE

        ax.bar(x, gap, width=BAR_WIDTH, color=fills, edgecolor=edges, linewidth=0.55)
        ax.scatter(
            [x[final_idx]], [gap[final_idx]], s=15, marker='X',
            color=MAUVE_FILL, edgecolor=MAUVE_OUTLINE, linewidth=0.65, zorder=4,
        )
        ax.scatter(
            [x[selected_idx]], [gap[selected_idx]], s=22, marker='D',
            color=MAUVE_FILL, edgecolor=MAUVE_OUTLINE, linewidth=0.75, zorder=5,
        )
        ax.axhline(0.0, color='black', linewidth=0.55)
        ax.set_title(model_title(model), pad=2)
        tick_indices, labels = layer_ticks_for_model(model, len(layer_names), display_num_layers)
        ax.set_xticks(x[tick_indices])
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.055, 1.055)
        style_axis(ax)

    axes[0].set_ylabel(r'Log-rank gap, $\Delta_l$')
    fig.supxlabel(r'Layer $l$', y=-0.04, fontsize=7)

    handles = [
        Line2D([0], [0], marker='D', color=MAUVE_OUTLINE, linestyle='None',
               markerfacecolor=MAUVE_FILL, markeredgewidth=0.75, markersize=4.2),
        Line2D([0], [0], marker='X', color=MAUVE_OUTLINE, linestyle='None',
               markerfacecolor=MAUVE_FILL, markeredgewidth=0.65, markersize=4.2),
    ]
    axes[-1].legend(
        handles, [r'Selected by $\Delta_l$', 'Last layer'],
        loc='upper right', frameon=False, handlelength=1.1,
        borderaxespad=0.25, labelspacing=0.35,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, 'imgnetlt_logrank_gap_architectures.png')
    out_pdf = os.path.join(OUT_DIR, 'imgnetlt_logrank_gap_architectures.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')

    print('plotted models:')
    for model in models:
        print(f'  {model}: selected layer {selected_layers[model]}')
    if skipped:
        print('skipped models without complete caches:')
        for model in skipped:
            print(f'  {model}')
    print(f'saved: {out_png}')
    print(f'saved: {out_pdf}')


if __name__ == '__main__':
    main()
