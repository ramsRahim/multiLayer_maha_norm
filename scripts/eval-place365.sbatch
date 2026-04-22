#!/bin/bash
#SBATCH --job-name=place365-all-methods
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpucompute-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

set -euo pipefail

DATASET="${1:-./data/places365.csv}"
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

MODEL_NAME="resnet50.tv_in1k"

echo "Running all-methods evaluation on places365"
echo "  model: ${MODEL_NAME}"
echo "  dataset: ${DATASET}"
echo "  dataset_prefix: ${DATASET_PREFIX}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  overwrite: ${OVERWRITE}"
echo "  bootstrap_seeds: ${BOOTSTRAP_SEEDS}"
echo "  workdir: $(pwd)"

python3 evaluate.py \
    --model_name "${MODEL_NAME}" \
    --method all \
    --dataset "${DATASET}" \
    --dataset_paths_prefix "${DATASET_PREFIX}" \
    --batch_size "${BATCH_SIZE}" \
    --overwrite_model_outputs "${OVERWRITE}" \
    --n_bootstrap_seeds "${BOOTSTRAP_SEEDS}" \
    "${EXTRA_ARGS[@]}"
