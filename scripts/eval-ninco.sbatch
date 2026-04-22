#!/bin/bash
#SBATCH --job-name=ninco-all-methods
#SBATCH --output=logs/exps-v2/%x-%j.out
#SBATCH --error=logs/exps-v2/%x-%j.err
#SBATCH --partition=gpucompute-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

set -euo pipefail

DATASET="${1:-NINCO}"
BATCH_SIZE="${2:-128}"
OVERWRITE="${3:-no}"
BOOTSTRAP_SEEDS="${4:-3}"
EXTRA_ARGS=("${@:5}")

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
LOG_DIR="${REPO_ROOT}/logs/exps-v2"

mkdir -p "${LOG_DIR}"

source "/data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh"
conda activate NINCO_maha

cd "${REPO_ROOT}"

MODEL_NAMES=(
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"
)

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "No models selected for evaluation."
    exit 1
fi

echo "Running all-models all-methods evaluation"
echo "  num_models: ${#MODEL_NAMES[@]}"
echo "  dataset: ${DATASET}"
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
        --batch_size "${BATCH_SIZE}" \
        --overwrite_model_outputs "${OVERWRITE}" \
        --n_bootstrap_seeds "${BOOTSTRAP_SEEDS}" \
        "${EXTRA_ARGS[@]}"
done
