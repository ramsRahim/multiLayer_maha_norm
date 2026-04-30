#!/bin/bash
#SBATCH --job-name=all-datasets-one-model
#SBATCH --output=logs/exps-all-datasets/%x-%j.out
#SBATCH --error=logs/exps-all-datasets/%x-%j.err
#SBATCH --partition=gpucompute-a40
#SBATCH --nodelist=ghpc005
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

BATCH_SIZE="128"
OVERWRITE="no"
BOOTSTRAP_SEEDS="3"

LOG_DIR="${REPO_ROOT}/logs/exps-all-datasets"

mkdir -p "${LOG_DIR}"

source "/data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh"
conda activate NINCO_maha

cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/all_model_names.sh"

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "No models selected for evaluation."
    exit 1
fi

echo "Running all-datasets all-methods evaluation"
echo "  num_models: ${#MODEL_NAMES[@]}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  overwrite: ${OVERWRITE}"
echo "  bootstrap_seeds: ${BOOTSTRAP_SEEDS}"
echo "  workdir: $(pwd)"
echo "  first_models: ${MODEL_NAMES[*]:0:3}"

run_dataset() {
    local model_name="$1"
    local label="$2"
    local dataset="$3"
    local dataset_prefix="${4:-}"

    echo
    echo "Running model/dataset: ${model_name} / ${label}"
    echo "  dataset: ${dataset}"
    if [ -n "${dataset_prefix}" ]; then
        echo "  dataset_prefix: ${dataset_prefix}"
    fi

    local cmd=(
        python3 evaluate.py
        --model_name "${model_name}"
        --method all
        --dataset "${dataset}"
        --batch_size "${BATCH_SIZE}"
        --overwrite_model_outputs "${OVERWRITE}"
        --n_bootstrap_seeds "${BOOTSTRAP_SEEDS}"
    )

    if [ -n "${dataset_prefix}" ]; then
        cmd+=(--dataset_paths_prefix "${dataset_prefix}")
    fi

    "${cmd[@]}"
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    run_dataset "${MODEL_NAME}" "ImageNet-O" "./data/imagenet_o.csv" "${REPO_ROOT}/OpenOOD/data/images_largescale"
    run_dataset "${MODEL_NAME}" "NINCO" "NINCO"
    run_dataset "${MODEL_NAME}" "OpenImage-O" "./data/openimages_o.csv" "${REPO_ROOT}/OpenOOD/data"
    run_dataset "${MODEL_NAME}" "Places365" "./data/places365.csv" "${REPO_ROOT}/OpenOOD/data"
    run_dataset "${MODEL_NAME}" "Textures" "./data/texture.csv" "${REPO_ROOT}/OpenOOD/data"
done
