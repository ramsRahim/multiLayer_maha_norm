#!/bin/bash
cd "$(dirname "$0")/.."
# Evaluate 10 methods on 5 OOD datasets for ViT-B/16
# ID data: Standard ImageNet (train for covariance, val for ID scoring)

MODEL="vit_base_patch16_224.augreg2_in21k_ft_in1k"
CACHE="./cache_imagenet_full"
TRAIN_DIR="${TRAIN_DIR:-/path/to/imagenet/train}"
VAL_DIR="${VAL_DIR:-/path/to/imagenet/val}"
PREFIX_OPENOOD="${PREFIX_OPENOOD:-/path/to/openood_data}"

METHODS=(
    MSP
    Energy
    "Energy+React"
    ODIN
    Mahalanobis
    Mahalanobis_norm
    Relative_Mahalanobis
    Relative_Mahalanobis_norm
    knn
    MM_plus_plus_topk_cat
)

declare -A DATASETS
DATASETS["./data/imagenet_o.csv"]="${IMAGENET_O_DIR:-/path/to/imagenet-o}"
DATASETS["./data/texture.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/places.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/inaturalist.csv"]="$PREFIX_OPENOOD"
DATASETS["./data/sun.csv"]="$PREFIX_OPENOOD"

DATASET_ORDER=(
    "./data/imagenet_o.csv"
    "./data/texture.csv"
    "./data/places.csv"
    "./data/inaturalist.csv"
    "./data/sun.csv"
)

LOG="/tmp/run_eval_vit_imagenet_$(date +%Y%m%d_%H%M%S).log"
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
        conda run -n NINCO_maha python -u evaluate.py \
            --model "$MODEL" \
            --dataset "$DS" \
            --dataset_paths_prefix "$PREFIX" \
            --method "$METHOD" \
            --path_to_cache "$CACHE" \
            --train_dir "$TRAIN_DIR" \
            --val_dir "$VAL_DIR" \
            --batch_size 256 \
            2>&1 | grep -E "Auroc|fpr at|Error|Traceback|Done|method not"
        echo "[$(date '+%H:%M:%S')] Done: $METHOD"
    done
done | tee "$LOG"

echo ""
echo "=== ALL DONE === $(date)"
