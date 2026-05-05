#!/bin/bash
cd "$(dirname "$0")/.."
# Evaluate 10 methods on 7 OOD datasets for ResNet-50
# ID data: Standard ImageNet 100K (train for covariance, val for ID scoring)

MODEL="resnet50.tv2_in1k"
CACHE="./cache_imagenetlt"
TRAIN_DIR="/home/rahim/exp/Data/imagenetlt/train"
VAL_DIR="/home/rahim/exp/Data/imagenetlt/test"
PREFIX_OPENOOD="/home/rahim/exp/Data/openood_data"
PREFIX_NINCO="/home/rahim/exp/Data/NINCO/NINCO"

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

LOG="/tmp/run_eval_resnet50_$(date +%Y%m%d_%H%M%S).log"
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
            2>&1 | grep -E "Auroc|fpr at|Error|Traceback|Done|method not"
        echo "[$(date '+%H:%M:%S')] Done: $METHOD"
    done
done | tee "$LOG"

echo ""
echo "=== ALL DONE === $(date)"
