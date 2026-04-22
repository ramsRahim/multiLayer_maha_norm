#!/bin/bash
#SBATCH --job-name=openimageo-all-methods
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpucompute-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

set -euo pipefail

DATASET="${1:-./data/openimages_o.csv}"
DATASET_PREFIX="${2:-${REPO_ROOT:-}/OpenOOD/data}"
BATCH_SIZE="${3:-128}"
OVERWRITE="${4:-no}"
BOOTSTRAP_SEEDS="${5:-3}"
EXTRA_ARGS=("${@:6}")

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
# Re-resolve DATASET_PREFIX now that REPO_ROOT is known
DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

source "/data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh"
conda activate NINCO_maha

cd "${REPO_ROOT}/NINCO"

readarray -t MODEL_NAMES < <(
    python3 -c '
from utils import timm_models

excluded = {
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
}

priority = [
    "resnet50.tv_in1k",
    "convnext_tiny.fb_in1k",
]

selected = [m for m in priority if m in timm_models and m not in excluded]
selected.extend(
    m for m in sorted(timm_models)
    if m not in excluded and m not in selected
)

print("\n".join(selected))
'
)

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "No models selected for evaluation."
    exit 1
fi

echo "Running all-models all-methods evaluation on openimage_o"
echo "  num_models: ${#MODEL_NAMES[@]}"
echo "  dataset: ${DATASET}"
echo "  dataset_prefix: ${DATASET_PREFIX}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  overwrite: ${OVERWRITE}"
echo "  bootstrap_seeds: ${BOOTSTRAP_SEEDS}"
echo "  workdir: $(pwd)"
echo "  first_models: ${MODEL_NAMES[*]:0:3}"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo
    echo "Running model: ${MODEL_NAME}"

    python3 evaluate.py \
        --model_name "${MODEL_NAME}" \
        --method all \
        --dataset "${DATASET}" \
        --dataset_paths_prefix "${DATASET_PREFIX}" \
        --batch_size "${BATCH_SIZE}" \
        --overwrite_model_outputs "${OVERWRITE}" \
        --n_bootstrap_seeds "${BOOTSTRAP_SEEDS}" \
        "${EXTRA_ARGS[@]}"
done
