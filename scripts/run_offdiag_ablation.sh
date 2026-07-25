#!/bin/bash
cd "$(dirname "$0")/.."
# Off-diagonal covariance ablation for MM++ (ViT-B/16, ImageNet-LT).
# Compares, on IDENTICAL selected layers / fused features / class means:
#   MM_plus_plus_topk_cat                 full joint LW precision  (MM++)
#   MM_plus_plus_topk_cat_blockdiag       zero cross-layer cov, same shrinkage
#   MM_plus_plus_topk_cat_blockdiag_indep per-block LW (independent shrinkage)
#   MM_plus_plus_topk_cat_permuted        within-class shuffle (destroys corresp.)
# Reuses caches produced by a prior MM_plus_plus_topk_cat run.

MODEL="vit_base_patch16_224.augreg2_in21k_ft_in1k"
CACHE="./cache_imagenetlt"
ENV="${CONDA_ENV:-NINCO_maha}"
TRAIN_DIR="${TRAIN_DIR:-/path/to/imagenetlt/train}"
VAL_DIR="${VAL_DIR:-/path/to/imagenetlt/test}"
PREFIX_OPENOOD="${PREFIX_OPENOOD:-/path/to/openood_data}"
PREFIX_NINCO="${PREFIX_NINCO:-/path/to/NINCO/NINCO}"
PERMUTE_SEED="${PERMUTE_SEED:-0}"

METHODS=(
    MM_plus_plus_topk_cat
    MM_plus_plus_topk_cat_blockdiag
    MM_plus_plus_topk_cat_blockdiag_indep
    MM_plus_plus_topk_cat_permuted
)

# dataset_csv : prefix
declare -A DATASETS
DATASETS["./data/NINCO_OOD_classes.csv"]="$PREFIX_NINCO"
DATASETS["./data/openimages_o.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/ssb_hard.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/imagenet_c.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/imagenet_es.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/imagenet_r.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/imagenet_v2.csv"]="$PREFIX_OPENOOD"

DATASET_ORDER=(
    "./data/NINCO_OOD_classes.csv"
    "./data/openimages_o.csv"
    "./data/ssb_hard.csv"
    "./data/imagenet_c.csv"
    "./data/imagenet_es.csv"
    "./data/imagenet_r.csv"
    "./data/imagenet_v2.csv"
)

LOG="/tmp/run_offdiag_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

for DS in "${DATASET_ORDER[@]}"; do
    PREFIX="${DATASETS[$DS]}"
    DS_LABEL=$(basename "$DS" .csv)
    echo ""
    echo "========================================"
    echo "=== Dataset: $DS_LABEL  $(date '+%H:%M:%S') ==="
    echo "========================================"

    for METHOD in "${METHODS[@]}"; do
        echo "[$(date '+%H:%M:%S')] $DS_LABEL :: $METHOD"
        conda run -n "$ENV" python -u evaluate.py \
            --model_name "$MODEL" \
            --dataset "$DS" \
            --dataset_paths_prefix "$PREFIX" \
            --method "$METHOD" \
            --path_to_cache "$CACHE" \
            --train_dir "$TRAIN_DIR" \
            --val_dir "$VAL_DIR" \
            --permute_seed "$PERMUTE_SEED" \
            2>&1 | grep -E "Selected layers|Auroc|fpr at|Error|Traceback|method not"
        echo "[$(date '+%H:%M:%S')] Done: $METHOD"
    done
done | tee "$LOG"

echo ""
echo "=== ALL DONE === $(date)"
