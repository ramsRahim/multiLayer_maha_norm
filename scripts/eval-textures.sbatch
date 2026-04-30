#!/bin/bash
#SBATCH --job-name=textures-all-methods
#SBATCH --output=logs/exps-textures/%x-%j.out
#SBATCH --error=logs/exps-textures/%x-%j.err
#SBATCH --partition=gpucompute-a40
#SBATCH --nodelist=ghpc006
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"

DATASET="${1:-./data/texture.csv}"
DATASET_PREFIX="${2:-${REPO_ROOT}/OpenOOD/data}"
BATCH_SIZE="${3:-128}"
OVERWRITE="${4:-no}"
BOOTSTRAP_SEEDS="${5:-3}"
EXTRA_ARGS=("${@:6}")

LOG_DIR="${REPO_ROOT}/logs/exps-textures"

mkdir -p "${LOG_DIR}"

source "/data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh"
conda activate NINCO_maha

cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/all_model_names.sh"

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "No models selected for evaluation."
    exit 1
fi

echo "Running all-models all-methods evaluation on textures"
echo "  num_models: ${#MODEL_NAMES[@]}"
echo "  dataset: ${DATASET}"
echo "  dataset_prefix: ${DATASET_PREFIX}"
echo "  data_dir: ${DATASET_PREFIX}/images_classic/texture"
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
