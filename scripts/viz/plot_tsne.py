"""
3 columns × 2 rows (6 panels).

    (a) Maha++ on selected intermediate layer  (block_04)
    (b) Maha++ on selected final feature layer (norm = features before classifier)
    (c) MM_plus_plus_topk_cat                  (joint space: block_04 + norm)

Row 1 — feature visualization map: t-SNE on a subset of ImageNet-1k classes
         plus NINCO OOD, computed in the Mahalanobis-whitened space of each
         method (distances match the detector's geometry).
Row 2 — score distribution (ID val vs NINCO OOD), AUROC / FPR@95 in title.

Reads all statistics / scores from cache; recomputes only what is missing
(single-layer OOD scores for block_04, and the whitened projections).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.labelsize': 7.5,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})

MODEL     = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
OOD_NAME  = 'NINCO_OOD_classes'
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE     = os.path.join(ROOT, 'cache')
METHODS   = os.path.join(CACHE, 'cache_methods', MODEL)
VAL       = os.path.join(CACHE, 'cache_val', MODEL)
OOD       = os.path.join(CACHE, 'cache_ood', MODEL, OOD_NAME)
VAL_INTER = os.path.join(CACHE, 'cache_val_inter', MODEL)
OOD_INTER = os.path.join(CACHE, 'cache_ood_inter', MODEL, OOD_NAME)
OUT_DIR   = os.path.join(ROOT, 'assets')

N_CLASSES_PLOT    = 30
N_SAMPLES_PER_CLS = 1000
N_OOD_PLOT        = 600
TSNE_PERPLEXITY   = 40
SEED              = 0

ID_SCORE_COLOR = '#0072B2'
OOD_SCORE_COLOR = '#D55E00'
OOD_POINT_COLOR = '#111111'
CLASS_COLORS = [
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442',
    '#56B4E9', '#E69F00', '#999999', '#332288', '#88CCEE',
]


def style_axis(ax, *, grid_axis='both'):
    ax.grid(True, axis=grid_axis, linestyle=':', color='0.82', linewidth=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── small utilities ──────────────────────────────────────────────────────────
def l2norm(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-10)
    return X / n


def load_layer(inter_dir, layer_name):
    parts = sorted(
        [f for f in os.listdir(inter_dir)
         if f.startswith(f'layer_{layer_name}_features_') and f.endswith('.npy')],
        key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
    )
    return np.concatenate([np.load(os.path.join(inter_dir, p)) for p in parts], axis=0)


def pca2(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def whitening_matrix(prec):
    """Return W such that X @ W puts features in Σ^{-1/2} coordinates."""
    eig, Q = np.linalg.eigh(prec.astype(np.float64))
    eig = eig.clip(min=0)
    return Q * np.sqrt(eig)[None, :]  # Q @ diag(sqrt(λ))


def tsne_whitened(feats_id, feats_ood, prec, perplexity=40, seed=0):
    """Whiten with Σ^{-1/2} so Euclidean distance = Mahalanobis, then t-SNE."""
    from sklearn.manifold import TSNE
    W = whitening_matrix(prec)
    X = np.concatenate([feats_id @ W, feats_ood @ W], axis=0)
    emb = TSNE(n_components=2, perplexity=perplexity,
               init='pca', learning_rate='auto',
               random_state=seed).fit_transform(X)
    return emb[:len(feats_id)], emb[len(feats_id):]


def fast_maha(feats, means, prec):
    Pmu    = means @ prec
    mu_Pmu = np.sum(means * Pmu, axis=1)
    xP     = feats @ prec
    xPx    = np.sum(feats * xP, axis=1)
    return -np.min(xPx[:, None] - 2 * (xP @ means.T) + mu_Pmu[None, :], axis=1)


def auroc(s_id, s_ood):
    s = np.concatenate([s_id, s_ood])
    y = np.concatenate([np.ones_like(s_id), np.zeros_like(s_ood)])
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / tp[-1]; fpr = fp / fp[-1]
    return float(np.trapz(tpr, fpr))


def fpr_at_tpr(s_id, s_ood, tpr_target=0.95):
    thr = np.quantile(s_id, 1 - tpr_target)
    return float((s_ood >= thr).mean())


def kde_curve(samples, x_grid, bandwidth_factor=1.0):
    """Gaussian KDE evaluated on x_grid; Silverman's rule times bandwidth_factor."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples, bw_method='silverman')
    kde.set_bandwidth(kde.factor * bandwidth_factor)
    return kde(x_grid)


# ── data assembly ───────────────────────────────────────────────────────────
def build_features_and_scores():
    """Returns a list of three (title, feat_id, feat_ood, s_id, s_ood) tuples."""
    labels_id = np.load(os.path.join(VAL, 'labels_true_0.npy'))

    # ── (a) Maha++ on selected intermediate layer (block_04) ─────────────────
    f_id_04  = l2norm(load_layer(VAL_INTER, 'block_04').astype(np.float64))
    f_ood_04 = l2norm(load_layer(OOD_INTER, 'block_04').astype(np.float64))
    mean_04  = np.load(os.path.join(METHODS, 'mm_pp_block_04_mean.npy'))
    prec_04  = np.load(os.path.join(METHODS, 'mm_pp_block_04_prec.npy'))
    s_id_a   = np.load(os.path.join(METHODS, 'mm_pp_block_04_id_scores.npy'))
    s_ood_a  = fast_maha(f_ood_04, mean_04, prec_04)

    # ── (b) Maha++ on penultimate (norm) ──────────────────────────────────────
    f_id_n   = l2norm(load_layer(VAL_INTER, 'norm').astype(np.float64))
    f_ood_n  = l2norm(load_layer(OOD_INTER, 'norm').astype(np.float64))
    mean_n   = np.load(os.path.join(METHODS, 'mm_pp_norm_mean.npy'))
    prec_n   = np.load(os.path.join(METHODS, 'mm_pp_norm_prec.npy'))
    # use scores already in the cached npz for consistency with Table 1
    z = np.load(sorted([
        os.path.join(CACHE, 'scores', MODEL, OOD_NAME, f)
        for f in os.listdir(os.path.join(CACHE, 'scores', MODEL, OOD_NAME))
        if f.endswith('.npz')
    ])[-1], allow_pickle=True)
    mr = z['methods_results'].item()
    s_id_b  = np.asarray(mr['Mahalanobis_norm']['scores_id'])
    s_ood_b = np.asarray(mr['Mahalanobis_norm']['scores_ood'])

    # ── (c) MM_plus_plus_topk_cat  (fused: block_04 + norm) ──────────────────
    fused_id   = np.concatenate([f_id_04,  f_id_n],  axis=1)
    fused_ood  = np.concatenate([f_ood_04, f_ood_n], axis=1)
    mean_fused = np.load(os.path.join(METHODS, 'mm_pp_topk2_delta_cat_uw_means.npy'))
    prec_fused = np.load(os.path.join(METHODS, 'mm_pp_topk2_delta_cat_uw_prec.npy'))
    s_id_c  = np.asarray(mr['MM_plus_plus_topk_cat']['scores_id'])
    s_ood_c = np.asarray(mr['MM_plus_plus_topk_cat']['scores_ood'])

    panels = [
        ('(a) Maha++ @ block_04',
            f_id_04, f_ood_04, mean_04,   prec_04,   s_id_a, s_ood_a),
        ('(b) Maha++ @ norm',
            f_id_n,  f_ood_n,  mean_n,    prec_n,    s_id_b, s_ood_b),
        ('(c) MM++ (block_04 + norm)',
            fused_id, fused_ood, mean_fused, prec_fused, s_id_c, s_ood_c),
    ]
    return panels, labels_id


# ── plotting ────────────────────────────────────────────────────────────────
def main():
    panels, labels_id = build_features_and_scores()

    rng     = np.random.default_rng(SEED)
    classes = rng.choice(1000, size=N_CLASSES_PLOT, replace=False)

    # equal-size per-class sampling so the scatter is visually balanced
    id_idx_parts = []
    for c in classes:
        pool = np.where(labels_id == c)[0]
        k    = min(N_SAMPLES_PER_CLS, len(pool))
        id_idx_parts.append(rng.choice(pool, size=k, replace=False))
    id_idx       = np.concatenate(id_idx_parts)
    id_labels_sub = labels_id[id_idx]

    # panels[i] = (title, X_id, X_ood, means, prec, s_id, s_ood)
    ood_N   = panels[0][2].shape[0]
    n_ood   = min(N_OOD_PLOT, ood_N)
    ood_idx = rng.choice(ood_N, size=n_ood, replace=False)

    fig, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.35),
        gridspec_kw={'height_ratios': [1.1, 0.9], 'hspace': 0.35, 'wspace': 0.28},
    )

    # ── Row 1: t-SNE on Maha-whitened features ───────────────────────────────
    for col, (title, Xid, Xood, _means, prec, _, _) in enumerate(panels):
        ax = axes[0, col]
        X_id  = Xid[id_idx]
        X_ood = Xood[ood_idx]
        print(f'[triptych] t-SNE {col+1}/{len(panels)} ({title}) ...')
        proj_id, proj_ood = tsne_whitened(
            X_id, X_ood, prec,
            perplexity=TSNE_PERPLEXITY, seed=SEED,
        )

        for k, c in enumerate(classes):
            m = id_labels_sub == c
            ax.scatter(proj_id[m, 0], proj_id[m, 1],
                       s=6, alpha=0.72, color=CLASS_COLORS[k % len(CLASS_COLORS)],
                       linewidths=0)
        ax.scatter(proj_ood[:, 0], proj_ood[:, 1],
                   s=9, alpha=0.45, color=OOD_POINT_COLOR, marker='x',
                   linewidths=0.55, label='NINCO OOD')

        ax.set_title(title, pad=3)
        ax.set_xlabel('t-SNE 1')
        if col == 0:
            ax.set_ylabel('t-SNE 2')
        else:
            ax.set_ylabel('')
        ax.set_xticks([]); ax.set_yticks([])
        style_axis(ax)
        if col == len(panels) - 1:
            ax.legend(
                loc='upper right',
                frameon=True,
                framealpha=0.95,
                fancybox=False,
                edgecolor='0.75',
                borderpad=0.25,
                handlelength=1.4,
            )

    # ── Row 2: score histograms ───────────────────────────────────────────────
    for col, (_, _, _, _, _, s_id, s_ood) in enumerate(panels):
        ax = axes[1, col]
        a   = auroc(s_id, s_ood)
        fpr = fpr_at_tpr(s_id, s_ood, 0.95)

        scores = np.concatenate([s_id, s_ood])
        lo, hi = float(scores.min()), float(scores.max())
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        x = np.linspace(lo, hi, 500)
        ax.plot(x, kde_curve(s_id, x, bandwidth_factor=1.1),
                color=ID_SCORE_COLOR, linewidth=1.35, label='ID')
        ax.plot(x, kde_curve(s_ood, x, bandwidth_factor=1.1),
                color=OOD_SCORE_COLOR, linewidth=1.35, label='OOD')

        ax.set_title(f'AUROC {a*100:.2f}  |  FPR@95 {fpr*100:.2f}', pad=3)
        ax.set_xlabel('OOD score')
        style_axis(ax)
        if col == 0:
            ax.set_ylabel('Density')
            ax.legend(
                loc='upper left',
                frameon=True,
                framealpha=0.95,
                fancybox=False,
                edgecolor='0.75',
                borderpad=0.25,
                handlelength=1.8,
            )
        else:
            ax.set_ylabel('')


    out_png = os.path.join(OUT_DIR, 'improvement_triptych_vitb16.png')
    out_pdf = os.path.join(OUT_DIR, 'improvement_triptych_vitb16.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'saved: {out_png}')
    print(f'saved: {out_pdf}')


if __name__ == '__main__':
    main()
