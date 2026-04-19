import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
import faiss


def evaluate_MM_plus_plus(train_inter_path, layer_feats_val, layer_feats_ood,
                          train_labels, path, n_classes=1000,
                          top_k=None, zscore=False):
    """
    MM++: Multilayer Mahalanobis++ Distance for OOD detection.

    Algorithm (Section 3.2–3.3 of the paper):
      For each intermediate layer l:
        1. L2-normalize features onto unit hypersphere  (Eq. 4)
        2. Compute class-conditional means              (Eq. 5)
        3. Estimate tied covariance with Ledoit-Wolf    (Eq. 13)
        4. Layer score S_l(x) = -min_c M++_{c,l}(x)   (Eq. 7)
      Aggregate:
        5. Initialize w_l = eps / std(S_l on ID val)   (Eq. 12)
        6. Learn w_l via logistic regression on a
           small OOD val split + all ID val             (Eq. 10-11)
        7. S_total(x) = Σ_l w_l S_l(x)                (Eq. 8)

    Training features are loaded one layer at a time from disk to avoid
    holding all L × N_train × D in memory simultaneously.

    Args:
        train_inter_path: directory of per-layer training features
                          (output of extract_intermediate_features on train set)
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels
        path:             cache directory for computed statistics / ID scores
        n_classes:        number of ID classes (1000 for ImageNet-1k)
        top_k:            if set, only aggregate the top-K layers by ER_B
        zscore:           if True, z-normalize each layer's scores using ID val
                          statistics before aggregation (fixes scale mismatch)

    Returns:
        score_id:  [N_val] MM++ scores for ID validation data
        score_ood: [N_ood] MM++ scores for OOD data
    """
    from sklearn.covariance import LedoitWolf
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f'layer_names.json not found in {train_inter_path}. '
            f'Run extract_intermediate_features() first.'
        )
    with open(names_path) as f:
        layer_names = json.load(f)

    L = len(layer_names)
    assert len(layer_feats_val) == L, \
        f'Expected {L} val-layer arrays, got {len(layer_feats_val)}'
    assert len(layer_feats_ood) == L, \
        f'Expected {L} OOD-layer arrays, got {len(layer_feats_ood)}'

    layer_scores_val   = []
    layer_scores_ood   = []
    effective_ranks    = []   # within-class ER  (kept for diagnostics)
    effective_ranks_b  = []   # between-class ER (used for aggregation)

    for l, layer_name in enumerate(layer_names):
        tag       = f'mm_pp_{layer_name}'
        mean_path = os.path.join(path, f'{tag}_mean.npy')
        prec_path = os.path.join(path, f'{tag}_prec.npy')
        er_path   = os.path.join(path, f'{tag}_er.npy')

        # ── Fit class means + Ledoit-Wolf precision + effective rank (cached) ─
        if os.path.exists(mean_path) and os.path.exists(prec_path) and os.path.exists(er_path):
            mean_l = np.load(mean_path)
            prec_l = np.load(prec_path)
            er_l   = float(np.load(er_path))
        else:
            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): loading training features...')
            parts = sorted(
                [fn for fn in os.listdir(train_inter_path)
                 if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
            )
            if not parts:
                raise FileNotFoundError(
                    f'No training features found for layer {layer_name} in {train_inter_path}'
                )
            f_train = np.concatenate(
                [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
            ).astype(np.float64)

            # L2-normalize (Eq. 4)
            norms = np.linalg.norm(f_train, axis=-1, keepdims=True).clip(min=1e-10)
            f_train /= norms

            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): computing class means...')
            train_means = []
            centered    = []
            for c in tqdm(range(n_classes), desc=f'  Layer {layer_name}', leave=False):
                fs = f_train[train_labels == c]
                m  = fs.mean(axis=0) if len(fs) > 0 else np.zeros(f_train.shape[1])
                train_means.append(m)
                if len(fs) > 0:
                    centered.extend(fs - m)

            centered_arr = np.array(centered, dtype=np.float64)  # [N, D]

            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): Ledoit-Wolf shrinkage...')
            lw = LedoitWolf(assume_centered=True)
            lw.fit(centered_arr)

            # ── Effective rank (Eqs. 7–8) via eigenspectrum of tied covariance ─
            # Dual-SVD: Gram matrix [N×N] if N<D, else direct D×D covariance.
            # Tied covariance = class-mean-centered, L2-normalized features.
            N_total, D = centered_arr.shape
            if N_total < D:
                G = centered_arr @ centered_arr.T / N_total       # [N, N]
                eigvals = np.linalg.eigvalsh(G)
            else:
                cov = centered_arr.T @ centered_arr / N_total     # [D, D]
                eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 0]
            eigvals /= eigvals.sum()
            er_l = float(np.exp(-np.dot(eigvals, np.log(eigvals))))

            mean_l = np.array(train_means)
            prec_l = lw.precision_
            np.save(mean_path, mean_l)
            np.save(prec_path, prec_l)
            np.save(er_path, er_l)
            del f_train, centered, centered_arr  # free memory

        effective_ranks.append(er_l)

        # ── Between-class ER: ER of class-means covariance (cached) ──────────
        # Measures how well-spread class means are on the hypersphere.
        # Increases with depth for both ViT and CNN → fixes the inverted-ER problem.
        er_b_path = os.path.join(path, f'{tag}_er_b.npy')
        if os.path.exists(er_b_path):
            er_b_l = float(np.load(er_b_path))
        else:
            mu_global = mean_l.mean(axis=0)                        # [D]
            M = (mean_l - mu_global).astype(np.float64)            # [C, D]
            C_cls = M.shape[0]
            sigma_b = M.T @ M / C_cls                              # [D, D]
            ev = np.linalg.eigvalsh(sigma_b)
            ev = ev[ev > 0]
            ev /= ev.sum()
            er_b_l = float(np.exp(-np.dot(ev, np.log(ev))))
            np.save(er_b_path, er_b_l)
        effective_ranks_b.append(er_b_l)

        # ── Move to GPU ───────────────────────────────────────────────────────
        mean_t = torch.from_numpy(mean_l).cuda().double()
        prec_t = torch.from_numpy(prec_l).cuda().double()

        def _maha_scores(feats_np):
            """Returns [N] array of -min_c M++_{c,l}(x) for each sample."""
            feats_np = feats_np.astype(np.float64)
            norms = np.linalg.norm(feats_np, axis=-1, keepdims=True).clip(min=1e-10)
            feats_np = feats_np / norms
            scores = []
            for f in torch.from_numpy(feats_np).cuda().double():
                diff = f - mean_t                         # [C, D]
                d    = ((diff @ prec_t) * diff).sum(-1)   # [C]
                scores.append(-d.min().cpu().item())
            return np.array(scores)

        # ── ID val scores (cached per layer) ─────────────────────────────────
        id_score_path = os.path.join(path, f'{tag}_id_scores.npy')
        if os.path.exists(id_score_path):
            s_val_l = np.load(id_score_path)
        else:
            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): computing ID scores...')
            s_val_l = _maha_scores(layer_feats_val[l])
            np.save(id_score_path, s_val_l)

        # ── OOD scores ────────────────────────────────────────────────────────
        print(f'[MM++] Layer {l+1}/{L} ({layer_name}): computing OOD scores...')
        s_ood_l = _maha_scores(layer_feats_ood[l])

        layer_scores_val.append(s_val_l)
        layer_scores_ood.append(s_ood_l)

    # ── Boltzmann rank-weighted sparsification (Eqs. 7–11, ER_B variant) ─────
    # Use between-class ER (ER_B) instead of total/within ER.
    # ER_B = ER of class-means covariance; increases with depth for both ViT and CNN
    # because class means become more spread as the network builds semantic structure.
    # This fixes the inverted-ER problem in isotropic ViTs, where total ER increases
    # with depth (early blocks uninformative, not collapsed), while remaining valid
    # for CNNs (ER_B also increases with depth as classes separate).
    er   = np.array(effective_ranks)    # [L]  within-class ER  (diagnostic only)
    er_b = np.array(effective_ranks_b)  # [L]  between-class ER (used for weighting)

    S_val = np.stack(layer_scores_val)   # [L, N_val]
    S_ood = np.stack(layer_scores_ood)   # [L, N_ood]

    # ── Optional top-K layer selection ───────────────────────────────────────
    # Discard early layers whose ER_B is too low to carry class-discriminative signal.
    if top_k is not None and top_k < len(er_b):
        keep  = np.argsort(er_b)[-top_k:]          # indices of top-K by ER_B
        er_b  = er_b[keep]
        er    = er[keep]
        S_val = S_val[keep]
        S_ood = S_ood[keep]
        print(f'[MM++] top_k={top_k}: keeping layers {keep.tolist()}')

    # ── Adaptive temperature: largest adjacent log-gain in ER_B ──────────────
    tau = float(np.max(np.log(er_b[1:] / er_b[:-1].clip(min=1e-8)))) if len(er_b) > 1 else 1.0

    # Boltzmann with positive sign: higher ER_B → more weight
    log_w = +tau * er_b
    log_w -= log_w.max()   # numerical stability
    w = np.exp(log_w)
    w /= w.sum()

    print(f'[MM++] Within-class ER:        {er.round(2)}')
    print(f'[MM++] Between-class ER (ER_B):{er_b.round(2)}')
    print(f'[MM++] Adaptive temperature:   τ = {tau:.4f}')
    print(f'[MM++] Aggregation weights:    {w.round(4)}')

    # ── Optional z-score normalization per layer (fixes scale mismatch) ──────
    # Standardize using ID val statistics; apply same shift/scale to OOD.
    if zscore:
        mean_l = S_val.mean(axis=1, keepdims=True)          # [L, 1]
        std_l  = S_val.std(axis=1, keepdims=True).clip(min=1e-8)
        S_val  = (S_val - mean_l) / std_l
        S_ood  = (S_ood - mean_l) / std_l

    # S_total(x) = Σ_l w_l · S_l(x)
    score_id  = w @ S_val   # [N_val]
    score_ood = w @ S_ood   # [N_ood]

    return score_id, score_ood


def evaluate_MM_plus_plus_topk_gating(train_inter_path, layer_feats_val, layer_feats_ood,
                                       train_labels, path, n_classes=1000, K=2, cache_suffix='',
                                       use_erb=False, relative=False, concat=False):
    """
    MM++ Top-K Information Gating (NeurIPS 2026 paper, Algorithm 1 & 2).

    Layer selection via within-class entropy and log-rank gaps (default, use_erb=False):

        H_l = -sum_i  lambda_bar_i * ln(lambda_bar_i)
              where lambda_bar_i = lambda_i / sum_j lambda_j
              and lambda_i are eigenvalues of the tied within-class covariance at layer l.
              (Derived from cached precision matrices: cov eigenvalues = 1 / prec eigenvalues.)

        Delta_l = ln(H_{l-1}/D_{l-1}) - ln(H_l/D_l)   for l in {2..L}
              A large Delta_l indicates strong semantic compression (neural collapse) at layer l.

    Selection (Algorithm 1):
        - Always include the final layer L.
        - Select K-1 additional layers with the largest Delta_l values.

    Aggregation weights:
        w_l = exp(Delta_l) / sum_{j in K} exp(Delta_j)   (softmax over log-rank gaps)

    ── Improved variant (use_erb=True) ─────────────────────────────────────────
    Uses between-class ER_B for both selection and weighting instead of
    within-class H/D gaps. ER_B = ER of the class-means covariance on the
    unit hypersphere; higher ER_B = more spread class means = better OOD discriminability.

        Selection:  top-K layers by absolute ER_B value
        Weighting:  softmax(ER_B) over selected layers

    This directly maximises between-class discriminability rather than finding
    where within-class distributions collapse (which lags by one layer on ViT).
    ────────────────────────────────────────────────────────────────────────────

    ── Relative variant (relative=True) ────────────────────────────────────────
    Subtracts a global (non-class-conditional) Mahalanobis score from the fused
    feature representation, mirroring Relative_Mahalanobis_norm. Removes global
    geometry bias in the fused feature space.
    ────────────────────────────────────────────────────────────────────────────

    Fusion:
        Homogeneous (ViT): additive — phi(x) = sum_l w_l * L2_norm(f_l(x))
                           then fit joint class means + LW precision, score with fast Maha.
        Heterogeneous (CNN): concatenation — phi(x) = [w_l * L2_norm(f_l(x)) for l in sel]
                             then fit joint class means + LW precision, score with fast Maha.

    Pre-requisite: evaluate_MM_plus_plus() must have been run first to cache
    per-layer precision matrices (mm_pp_{name}_prec.npy) and class means
    (mm_pp_{name}_mean.npy).

    Args:
        train_inter_path: directory of per-layer training feature shards
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels
        path:             cache directory used by evaluate_MM_plus_plus
        n_classes:        number of ID classes
        K:                total number of layers to select (paper: K=2 for ViT, K=3 for CNN)
        use_erb:          if True, select/weight by between-class ER_B instead of H/D gaps
        relative:         if True, subtract global Mahalanobis from fused-space scores

    Returns:
        score_id:  [N_val]  higher = more ID-like
        score_ood: [N_ood]  higher = more ID-like
    """
    from sklearn.covariance import LedoitWolf
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(f'layer_names.json not found in {train_inter_path}.')
    with open(names_path) as f:
        layer_names = json.load(f)
    L = len(layer_names)

    # ── Ensure per-layer caches exist (build via MM_plus_plus if missing) ────
    tag0 = f'mm_pp_{layer_names[0]}'
    if not os.path.exists(os.path.join(path, f'{tag0}_prec.npy')):
        print('[TopKGating] Per-layer caches missing — fitting per-layer statistics...')
        evaluate_MM_plus_plus(train_inter_path, layer_feats_val, layer_feats_ood,
                              train_labels, path, n_classes=n_classes)

    # ── Load cached precision matrices and class means per layer ─────────────
    prec_list = []
    mean_list = []
    D_vals    = []
    for layer_name in layer_names:
        prec_path = os.path.join(path, f'mm_pp_{layer_name}_prec.npy')
        mean_path = os.path.join(path, f'mm_pp_{layer_name}_mean.npy')
        prec_l = np.load(prec_path).astype(np.float64)   # [D, D]
        mean_l = np.load(mean_path).astype(np.float64)   # [C, D]
        prec_list.append(prec_l)
        mean_list.append(mean_l)
        D_vals.append(prec_l.shape[0])

    D_arr = np.array(D_vals, dtype=np.float64)  # [L]

    if use_erb:
        # ── ER_B-based selection: top-K by between-class discriminability ─────
        # ER_B = ER of class-means covariance on unit hypersphere.
        # Higher ER_B → class means more spread → more OOD-discriminative layer.
        # Guaranteed to include the highest-ER_B layer (avoids the off-by-one error
        # of the log-gap criterion, which picks the layer BEFORE the peak).
        er_b_vals = []
        for layer_name in layer_names:
            er_b_path = os.path.join(path, f'mm_pp_{layer_name}_er_b.npy')
            if not os.path.exists(er_b_path):
                raise FileNotFoundError(
                    f'ER_B cache missing for {layer_name}. Run evaluate_MM_plus_plus first.')
            er_b_vals.append(float(np.load(er_b_path)))
        er_b = np.array(er_b_vals, dtype=np.float64)  # [L]

        # Select K layers with highest absolute ER_B
        sel_idx = np.sort(np.argsort(er_b)[-K:])  # ascending order

        # Weights: softmax over ER_B values of selected layers
        sel_er_b = er_b[sel_idx]
        sel_er_b_shifted = sel_er_b - sel_er_b.max()
        exp_erb  = np.exp(sel_er_b_shifted)
        weights  = exp_erb / exp_erb.sum()

        print(f'[TopKGating] ER_B (all layers):        {er_b.round(2)}')
        print(f'[TopKGating] Selected layer indices:   {sel_idx.tolist()}')
        print(f'[TopKGating] Selected layers:          {[layer_names[i] for i in sel_idx]}')
        print(f'[TopKGating] Selected ER_B values:     {sel_er_b.round(4)}')
        print(f'[TopKGating] Softmax weights:          {weights.round(4)}')
    else:
        # ── Compute within-class entropy H_l from precision eigenvalues ───────
        # Covariance eigenvalues = 1 / precision eigenvalues
        # H_l = -sum_i  lambda_bar_i * ln(lambda_bar_i)
        H_vals = []
        for prec_l in prec_list:
            eig_prec = np.linalg.eigvalsh(prec_l)          # sorted ascending, all > 0
            eig_cov  = 1.0 / eig_prec.clip(min=1e-10)      # covariance eigenvalues
            eig_cov  = eig_cov.clip(min=0)
            total    = eig_cov.sum()
            if total <= 0:
                H_vals.append(0.0)
                continue
            lam_bar  = eig_cov / total
            lam_bar  = lam_bar.clip(min=1e-300)             # avoid log(0)
            H_l      = -float(np.sum(lam_bar * np.log(lam_bar)))
            H_vals.append(H_l)

        H = np.array(H_vals, dtype=np.float64)              # [L]

        # ── Log-rank gaps Delta_l (Eq. 9 of paper) ───────────────────────────
        # rel_rank_l = H_l / D_l
        rel_rank  = H / D_arr.clip(min=1)
        log_rel   = np.log(rel_rank.clip(min=1e-300))       # [L]
        # Delta[i] corresponds to Δ_{i+2} (0-indexed): gap from layer i → layer i+1
        delta     = log_rel[:-1] - log_rel[1:]              # [L-1], >= 0 near collapse

        print(f'[TopKGating] Within-class entropy H:   {H.round(4)}')
        print(f'[TopKGating] H/D (rel rank):           {rel_rank.round(6)}')
        print(f'[TopKGating] Log-rank gaps Delta:      {np.round(delta, 4)}')

        # ── Layer selection ───────────────────────────────────────────────────
        # Always include final layer (index L-1).
        # Select K-1 additional by largest Delta value.
        # delta[i] is the gap arriving at layer i+1 → selecting delta[i] means selecting layer i+1.
        final_idx   = L - 1
        K_extra     = max(0, K - 1)
        # Candidate positions in delta: all except those pointing to final_idx already
        # (delta[final_idx-1] is the gap arriving at final_idx; we exclude it from extra picks
        #  since we already include the final layer)
        candidates  = np.array([i for i in range(len(delta)) if (i + 1) != final_idx])
        if len(candidates) == 0 or K_extra == 0:
            extra_layer_idx = np.array([], dtype=int)
        else:
            k_pick = min(K_extra, len(candidates))
            top_c  = np.argsort(delta[candidates])[-k_pick:]
            extra_layer_idx = candidates[top_c] + 1   # actual layer indices

        sel_idx = np.sort(np.unique(np.concatenate([[final_idx], extra_layer_idx])))  # ascending

        # Delta used for weighting: delta value that "arrives" at each selected layer
        # For layer l > 0, the arriving gap is delta[l-1].  For layer 0, use 0 (never selected).
        sel_deltas = np.array([delta[i - 1] if i > 0 else 0.0 for i in sel_idx], dtype=np.float64)
        # Softmax weights
        sel_deltas_shifted = sel_deltas - sel_deltas.max()  # numerical stability
        exp_d  = np.exp(sel_deltas_shifted)
        weights = exp_d / exp_d.sum()

        print(f'[TopKGating] Selected layer indices:   {sel_idx.tolist()}')
        print(f'[TopKGating] Selected layers:          {[layer_names[i] for i in sel_idx]}')
        print(f'[TopKGating] Selected Delta values:    {sel_deltas.round(4)}')
        print(f'[TopKGating] Softmax weights:          {weights.round(4)}')

    # ── Determine fusion mode ─────────────────────────────────────────────────
    sel_dims     = [D_vals[i] for i in sel_idx]
    is_homogeneous = (len(set(sel_dims)) == 1) and not concat

    K_sel = len(sel_idx)

    # For concatenation, override weights to 1.0 — the joint LW precision
    # handles scale and cross-layer covariance implicitly.
    if concat:
        weights = np.ones(K_sel, dtype=np.float64)

    # ── Helper: fast CPU Mahalanobis scoring ──────────────────────────────────
    def _fast_maha(feats, means, prec):
        """feats:[N,D], means:[C,D], prec:[D,D] -> [N] scores"""
        Pmu    = means @ prec
        mu_Pmu = np.sum(means * Pmu, axis=1)
        xP     = feats @ prec
        xPx    = np.sum(feats * xP, axis=1)
        return -np.min(xPx[:, None] - 2 * (xP @ means.T) + mu_Pmu[None, :], axis=1)

    # Cache tag encodes variant so old caches are never overwritten.
    # For heterogeneous archs (natural concat), also encode whether weights
    # are uniform (concat=True) or Delta-based (concat=False) to avoid
    # sharing a cache across different weight schemes.
    sel_str   = 'erb' if use_erb else 'delta'
    rel_str   = '_rel' if relative else ''
    wt_str    = '_uw' if concat else ''   # 'uw' = uniform weights
    if is_homogeneous:
        fused_tag = f'mm_pp_topk{K}_{sel_str}{rel_str}{cache_suffix}'
        D_fused   = sel_dims[0]
        print(f'[TopKGating] Homogeneous (dim={D_fused}). Additive fusion.')
    else:
        fused_tag = f'mm_pp_topk{K}_{sel_str}_cat{wt_str}{rel_str}{cache_suffix}'
        D_fused   = sum(sel_dims)
        print(f'[TopKGating] {"Forced " if concat else ""}Concatenation fusion, joint dim={D_fused}.')

    fused_mean_path = os.path.join(path, f'{fused_tag}_means.npy')
    fused_prec_path = os.path.join(path, f'{fused_tag}_prec.npy')
    id_score_path   = os.path.join(path, f'{fused_tag}_id_scores.npy')

    # Paths for optional global (non-class-conditional) statistics
    global_mean_path = os.path.join(path, f'{fused_tag}_global_mean.npy')
    global_prec_path = os.path.join(path, f'{fused_tag}_global_prec.npy')

    # ── Fit fused class means + Ledoit-Wolf precision ─────────────────────────
    need_global = relative and not (os.path.exists(global_mean_path) and os.path.exists(global_prec_path))
    if os.path.exists(fused_mean_path) and os.path.exists(fused_prec_path) and not need_global:
        print('[TopKGating] Loading cached fused class means + precision...')
        fused_means = np.load(fused_mean_path)
        prec_fused  = np.load(fused_prec_path)
    else:
        N_train     = len(train_labels)
        fused_train = np.zeros((N_train, D_fused), dtype=np.float64)
        fused_means = np.zeros((n_classes, D_fused), dtype=np.float64)

        if is_homogeneous:
            for a_pos, l in enumerate(sel_idx):
                layer_name = layer_names[l]
                w_l        = weights[a_pos]
                print(f'[TopKGating] Layer {layer_name}: loading train features (w={w_l:.4f})...')
                parts = sorted(
                    [fn for fn in os.listdir(train_inter_path)
                     if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                    key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
                )
                if not parts:
                    raise FileNotFoundError(f'No training features for {layer_name} in {train_inter_path}')
                f_l    = np.concatenate(
                    [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
                ).astype(np.float64)
                norms  = np.linalg.norm(f_l, axis=-1, keepdims=True).clip(min=1e-10)
                f_l   /= norms
                fused_train += w_l * f_l
                fused_means += w_l * mean_list[l]
                del f_l
        else:
            col = 0
            for a_pos, l in enumerate(sel_idx):
                layer_name = layer_names[l]
                d_l        = sel_dims[a_pos]
                w_l        = weights[a_pos]
                print(f'[TopKGating] Layer {layer_name}: loading train features (dim={d_l}, w={w_l:.4f})...')
                parts = sorted(
                    [fn for fn in os.listdir(train_inter_path)
                     if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                    key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
                )
                if not parts:
                    raise FileNotFoundError(f'No training features for {layer_name} in {train_inter_path}')
                f_l    = np.concatenate(
                    [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
                ).astype(np.float64)
                norms  = np.linalg.norm(f_l, axis=-1, keepdims=True).clip(min=1e-10)
                f_l   /= norms
                fused_train[:, col:col+d_l]   = w_l * f_l
                fused_means[:, col:col+d_l]   = w_l * mean_list[l]
                col += d_l
                del f_l

        print('[TopKGating] Centering + fitting Ledoit-Wolf (class-conditional)...')
        lw = LedoitWolf(assume_centered=True)
        lw.fit(fused_train - fused_means[train_labels])
        prec_fused = lw.precision_
        np.save(fused_mean_path, fused_means)
        np.save(fused_prec_path, prec_fused)

        # ── Optional: fit global (non-class-conditional) precision ───────────
        if relative:
            print('[TopKGating] Fitting global Ledoit-Wolf for relative Mahalanobis...')
            global_mean_fused = fused_train.mean(axis=0)
            lw_global = LedoitWolf(assume_centered=True)
            lw_global.fit(fused_train - global_mean_fused)
            np.save(global_mean_path, global_mean_fused)
            np.save(global_prec_path, lw_global.precision_)

        del fused_train

    # ── Load global stats for relative scoring (if needed) ───────────────────
    if relative:
        if os.path.exists(global_mean_path) and os.path.exists(global_prec_path):
            global_mean_fused = np.load(global_mean_path)
            prec_global_fused = np.load(global_prec_path)
        else:
            raise RuntimeError(
                '[TopKGating] Global stats missing for relative mode. '
                'Delete fused caches and re-run to rebuild.'
            )

    # ── Build fused feature vector ────────────────────────────────────────────
    def _fuse(layer_feats_list):
        N = layer_feats_list[sel_idx[0]].shape[0]
        if is_homogeneous:
            fused = np.zeros((N, D_fused), dtype=np.float64)
            for a_pos, l in enumerate(sel_idx):
                feats = layer_feats_list[l].astype(np.float64)
                norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
                fused += weights[a_pos] * (feats / norms)
        else:
            fused = np.zeros((N, D_fused), dtype=np.float64)
            col   = 0
            for a_pos, l in enumerate(sel_idx):
                d_l   = sel_dims[a_pos]
                feats = layer_feats_list[l].astype(np.float64)
                norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
                fused[:, col:col+d_l] = weights[a_pos] * (feats / norms)
                col  += d_l
        return fused

    def _score(fused):
        s_class = _fast_maha(fused, fused_means, prec_fused)
        if relative:
            # global Mahalanobis: single-"class" distance to global mean
            diff    = fused - global_mean_fused
            s_glob  = -(np.sum((diff @ prec_global_fused) * diff, axis=1))
            return s_class - s_glob
        return s_class

    # ── ID val scores (cached) ────────────────────────────────────────────────
    if os.path.exists(id_score_path):
        score_id = np.load(id_score_path)
    else:
        print('[TopKGating] Computing ID val scores...')
        score_id = _score(_fuse(layer_feats_val))
        np.save(id_score_path, score_id)

    # ── OOD scores ────────────────────────────────────────────────────────────
    print('[TopKGating] Computing OOD scores...')
    score_ood = _score(_fuse(layer_feats_ood))

    return score_id, score_ood




# ── Baseline methods ──────────────────────────────────────────────────────────

def evaluate_MSP(softmax_id_val, softmax_ood):
    score_id  = softmax_id_val.max(axis=-1)
    score_ood = softmax_ood.max(axis=-1)
    return score_id, score_ood


def evaluate_Energy(logits_in_distribution, logits_out_of_distribution):
    score_id  = logsumexp(logits_in_distribution,   axis=1)
    score_ood = logsumexp(logits_out_of_distribution, axis=1)
    return score_id, score_ood


def evaluate_Energy_React(feature_id_train, feature_id_val, feature_ood, w, b, path, clip_quantile=0.99):
    clip_react_path = os.path.join(path, 'clip_react.npy')
    if os.path.exists(clip_react_path):
        clip = np.load(clip_react_path)
    else:
        clip = np.quantile(feature_id_train, clip_quantile)
        np.save(clip_react_path, clip)
    print(f'clip quantile {clip_quantile}, clip {clip:.4f}')
    score_id_path = os.path.join(path, 'score_id_energy_react.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
        score_id = logsumexp(logit_id_val_clip, axis=-1)
        np.save(score_id_path, score_id)
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    return score_id, score_ood


def evaluate_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    if os.path.exists(mean_path) and os.path.exists(prec_path):
        mean = np.load(mean_path)
        prec = np.load(prec_path)
    else:
        print('Computing classwise means...')
        train_means, centered = [], []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            m  = fs.mean(axis=0)
            train_means.append(m)
            centered.extend(fs - m)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(centered).astype(np.float64))
        mean = np.array(train_means)
        prec = ec.precision_
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    mean_t = torch.from_numpy(mean).cuda().double()
    prec_t = torch.from_numpy(prec).cuda().double()
    score_id_path = os.path.join(path, 'maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                               for f in tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)
    score_ood = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                            for f in tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood


def evaluate_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    feature_id_val = feature_id_val / np.linalg.norm(feature_id_val, axis=-1, keepdims=True)
    feature_ood    = feature_ood    / np.linalg.norm(feature_ood,    axis=-1, keepdims=True)
    mean_path = os.path.join(path, 'mean_norm.npy')
    prec_path = os.path.join(path, 'prec_norm.npy')
    if os.path.exists(mean_path) and os.path.exists(prec_path):
        mean = np.load(mean_path)
        prec = np.load(prec_path)
    else:
        print('Computing classwise means (norm)...')
        feature_id_train = feature_id_train / np.linalg.norm(feature_id_train, axis=-1, keepdims=True)
        train_means, centered = [], []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            m  = fs.mean(axis=0)
            train_means.append(m)
            centered.extend(fs - m)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(centered).astype(np.float64))
        mean = np.array(train_means)
        prec = ec.precision_
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    mean_t = torch.from_numpy(mean).cuda().double()
    prec_t = torch.from_numpy(prec).cuda().double()
    score_id_path = os.path.join(path, 'maha_id_scores_norm.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                               for f in tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)
    score_ood = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                            for f in tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood


def _rel_maha_scores(feature_val, mean_t, prec_t, mean_global, prec_global,
                     score_id_class_path, score_id_rel_path):
    if os.path.exists(score_id_rel_path):
        return np.load(score_id_rel_path)
    if os.path.exists(score_id_class_path):
        score_id_c = np.load(score_id_class_path)
    else:
        score_id_c = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                                 for f in tqdm(torch.from_numpy(feature_val).cuda().double())])
        np.save(score_id_class_path, score_id_c)
    score_id_g = -np.array([(((f - mean_global) @ prec_global) * (f - mean_global)).sum()
                             for f in tqdm(feature_val)])
    score_id = score_id_c - score_id_g
    np.save(score_id_rel_path, score_id)
    return score_id


def evaluate_Relative_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    if os.path.exists(mean_path) and os.path.exists(prec_path):
        mean = np.load(mean_path)
        prec = np.load(prec_path)
    else:
        print('Computing classwise means...')
        train_means, centered = [], []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            m  = fs.mean(axis=0)
            train_means.append(m)
            centered.extend(fs - m)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(centered).astype(np.float64))
        mean = np.array(train_means)
        prec = ec.precision_
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    mean_g_path = os.path.join(path, 'mean-global.npy')
    prec_g_path = os.path.join(path, 'prec-global.npy')
    if os.path.exists(mean_g_path) and os.path.exists(prec_g_path):
        mean_global = np.load(mean_g_path)
        prec_global = np.load(prec_g_path)
    else:
        mg = feature_id_train.mean(axis=0)
        ec_g = EmpiricalCovariance(assume_centered=True)
        ec_g.fit((feature_id_train - mg).astype(np.float64))
        mean_global = mg
        prec_global = ec_g.precision_
        np.save(mean_g_path, mean_global)
        np.save(prec_g_path, prec_global)
    mean_t = torch.from_numpy(mean).cuda().double()
    prec_t = torch.from_numpy(prec).cuda().double()
    score_id = _rel_maha_scores(feature_id_val, mean_t, prec_t, mean_global, prec_global,
                                os.path.join(path, 'maha_id_scores.npy'),
                                os.path.join(path, 'rel_maha_id_scores.npy'))
    score_ood_c = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                              for f in tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_g = -np.array([(((f - mean_global) @ prec_global) * (f - mean_global)).sum()
                              for f in tqdm(feature_ood)])
    return score_id, score_ood_c - score_ood_g


def evaluate_Relative_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    feature_id_val = feature_id_val / np.linalg.norm(feature_id_val, axis=-1, keepdims=True)
    feature_ood    = feature_ood    / np.linalg.norm(feature_ood,    axis=-1, keepdims=True)
    mean_path = os.path.join(path, 'mean_norm.npy')
    prec_path = os.path.join(path, 'prec_norm.npy')
    if os.path.exists(mean_path) and os.path.exists(prec_path):
        mean = np.load(mean_path)
        prec = np.load(prec_path)
    else:
        print('Computing classwise means (norm)...')
        feature_id_train = feature_id_train / np.linalg.norm(feature_id_train, axis=-1, keepdims=True)
        train_means, centered = [], []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            m  = fs.mean(axis=0)
            train_means.append(m)
            centered.extend(fs - m)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(centered).astype(np.float64))
        mean = np.array(train_means)
        prec = ec.precision_
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    mean_g_path = os.path.join(path, 'mean-global_norm.npy')
    prec_g_path = os.path.join(path, 'prec-global_norm.npy')
    if os.path.exists(mean_g_path) and os.path.exists(prec_g_path):
        mean_global = np.load(mean_g_path)
        prec_global = np.load(prec_g_path)
    else:
        mg = feature_id_train.mean(axis=0)
        ec_g = EmpiricalCovariance(assume_centered=True)
        ec_g.fit((feature_id_train - mg).astype(np.float64))
        mean_global = mg
        prec_global = ec_g.precision_
        np.save(mean_g_path, mean_global)
        np.save(prec_g_path, prec_global)
    mean_t = torch.from_numpy(mean).cuda().double()
    prec_t = torch.from_numpy(prec).cuda().double()
    score_id = _rel_maha_scores(feature_id_val, mean_t, prec_t, mean_global, prec_global,
                                os.path.join(path, 'maha_id_scores_norm.npy'),
                                os.path.join(path, 'rel_maha_id_scores_norm.npy'))
    score_ood_c = -np.array([(((f - mean_t) @ prec_t) * (f - mean_t)).sum(-1).min().cpu().item()
                              for f in tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_g = -np.array([(((f - mean_global) @ prec_global) * (f - mean_global)).sum()
                              for f in tqdm(feature_ood)])
    return score_id, score_ood_c - score_ood_g


def evaluate_KNN(feature_id_train, feature_id_val, feature_ood, path):
    normalizer  = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))
    scores_path = os.path.join(path, 'scores_id_knn.npy')
    index_path  = os.path.join(path, 'trained.index')
    K = 1000
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print('Building KNN index...')
        ftrain = prepos_feat(feature_id_train)
        index  = faiss.IndexFlatL2(ftrain.shape[1])
        index.add(ftrain)
        faiss.write_index(index, index_path)
    if os.path.exists(scores_path):
        score_id = np.load(scores_path)
    else:
        D, _ = index.search(prepos_feat(feature_id_val).astype(np.float32), K)
        score_id = -D[:, -1]
        np.save(scores_path, score_id)
    D, _ = index.search(prepos_feat(feature_ood).astype(np.float32), K)
    score_ood = -D[:, -1]
    return score_id, score_ood


def evaluate_ODIN(model, dataset_val, dataset_ood, path,
                  T=1000, epsilon=0.0014, batch_size=64, n_workers=8):
    """
    ODIN: Out-of-Distribution Detector via temperature scaling + input perturbation.
    Liang et al., ICLR 2018. https://arxiv.org/abs/1706.02690

    Score = max_c softmax(f(x̃) / T)  where x̃ = x + ε·sign(∇_x log p_T(y*|x))
    and y* = argmax f(x).

    Args:
        model:       timm model (cuda, eval mode)
        dataset_val: ID validation Dataset
        dataset_ood: OOD Dataset
        path:        cache directory
        T:           temperature (default 1000)
        epsilon:     perturbation magnitude (default 0.0014)
        batch_size:  inference batch size
        n_workers:   DataLoader workers

    Returns:
        score_id:  [N_val] higher = more ID-like
        score_ood: [N_ood] higher = more ID-like
    """
    # ImageNet per-channel normalisation stds (used to rescale gradient)
    _STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    tag = f'odin_T{T}_eps{epsilon}'
    id_path  = os.path.join(path, f'{tag}_id_scores.npy')
    ood_path = os.path.join(path, f'{tag}_ood_{dataset_ood.__name__}_scores.npy')

    def _score_loader(loader, cache_path):
        if os.path.exists(cache_path):
            return np.load(cache_path)
        scores = []
        for images, _ in tqdm(loader, desc=f'ODIN {cache_path.split("/")[-1]}'):
            with torch.enable_grad():
                images = images.cuda().requires_grad_(True)
                # Forward pass
                logits = model(images)
                # Predicted class (for loss)
                pred   = logits.detach().argmax(dim=1)
                # Temperature-scaled cross-entropy loss on predicted class
                loss   = torch.nn.functional.cross_entropy(logits / T, pred)
                model.zero_grad()
                loss.backward()
            # Sign of gradient, normalised to image space
            grad   = images.grad.data
            grad   = torch.sign(grad) / _STD          # rescale by channel std
            # Perturbed image (add perturbation in gradient direction)
            x_hat  = (images.detach() + epsilon * grad).detach()
            # Score on perturbed image
            with torch.no_grad():
                logits_hat = model(x_hat) / T
                s = torch.softmax(logits_hat, dim=1).max(dim=1).values
            scores.append(s.cpu().numpy())
        out = np.concatenate(scores)
        np.save(cache_path, out)
        return out

    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True)
    ood_loader = torch.utils.data.DataLoader(
        dataset_ood, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True)

    score_id  = _score_loader(val_loader,  id_path)
    score_ood = _score_loader(ood_loader,  ood_path)
    return score_id, score_ood
