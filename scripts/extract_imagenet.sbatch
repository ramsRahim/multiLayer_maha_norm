#!/bin/bash
#SBATCH --job-name=extract_imgnet
#SBATCH --partition=Standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm/logs/extract_imagenet_%j.log
#SBATCH --error=/data/home/mislambhuian/exp/ood_project/multiLayer_maha_normx/logs/extract_imagenet_%j.err

set -euo pipefail

source /data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh
conda activate NINCO_maha

SCRIPT_DIR="/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm/scripts"
IMAGENET_ROOT="/data/home/mislambhuian/data/imagenet-1k"
PARQUET_DIR="${IMAGENET_ROOT}/data"
TRAIN_DIR="${IMAGENET_ROOT}/train"
VAL_DIR="${IMAGENET_ROOT}/val"

if [[ ! -d "${PARQUET_DIR}" ]]; then
    echo "Missing downloaded ImageNet-1K parquet directory: ${PARQUET_DIR}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

if [[ -d "${TRAIN_DIR}" ]]; then
    echo "Skipping train extraction; ${TRAIN_DIR} already exists."
else
    python3 - <<'PY'
from extract_imagenet import extract_split

extract_split("train", "train-*.parquet", "train")
PY
fi

if [[ -d "${VAL_DIR}" ]]; then
    echo "Skipping val extraction; ${VAL_DIR} already exists."
else
    python3 - <<'PY'
from extract_imagenet import extract_split

extract_split("val", "validation-*.parquet", "val")
PY
fi

echo "ImageNet-1K extraction job complete."
