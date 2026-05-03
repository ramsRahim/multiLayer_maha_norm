#!/bin/bash
#SBATCH --job-name=ood-imagenet-lt
#SBATCH --output=logs/exps-ood-datasets-lt/%x-%j.out
#SBATCH --error=logs/exps-ood-datasets-lt/%x-%j.err
#SBATCH --partition=gpucompute-p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00

# Evaluate selected models (configured in scripts/all_model_names_lt.sh) on the
# OOD datasets: NINCO, Openimage-O, SSB-Hard, ImageNet-C, ImageNet-ES,
# ImageNet-R, ImageNet-V2.
# The ID dataset is ImageNet-LT (train/test split) at
# /data/home/mislambhuian/data/imagenetlt.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"

BATCH_SIZE="${BATCH_SIZE:-512}"
OVERWRITE="${OVERWRITE:-no}"
BOOTSTRAP_SEEDS="${BOOTSTRAP_SEEDS:-3}"
CACHE_DIR="${CACHE_DIR:-./cache_imgnetlt}"
PATH_TO_IMAGENET="${PATH_TO_IMAGENET:-/data/home/mislambhuian/data/imagenetlt}"

# Set OOD_DATASETS in the environment to a space-separated subset to limit
# the run, e.g. OOD_DATASETS="NINCO ImageNet-R" sbatch ...
DEFAULT_OOD_DATASETS=(
    NINCO
    Openimage-O
    ImageNet-C
    ImageNet-ES
    ImageNet-R
    ImageNet-V2
    SUN
    ImageNet-O
)
if [ -n "${OOD_DATASETS:-}" ]; then
    # shellcheck disable=SC2206
    OOD_LIST=(${OOD_DATASETS})
else
    OOD_LIST=("${DEFAULT_OOD_DATASETS[@]}")
fi

LOG_DIR="${REPO_ROOT}/logs/exps-ood-datasets-lt"
mkdir -p "${LOG_DIR}"

source "/data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh"
conda activate NINCO_maha

cd "${REPO_ROOT}"

# Model selection: edit scripts/all_model_names_lt.sh to choose which models run.
source "${SCRIPT_DIR}/all_model_names_lt.sh"

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "No models selected for evaluation. Edit scripts/all_model_names_lt.sh to enable models."
    exit 1
fi

echo "Running selected-models all-methods OOD evaluation (ID = ImageNet-LT)"
echo "  num_models: ${#MODEL_NAMES[@]}"
echo "  ood_datasets: ${OOD_LIST[*]}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  overwrite: ${OVERWRITE}"
echo "  bootstrap_seeds: ${BOOTSTRAP_SEEDS}"
echo "  cache_dir: ${CACHE_DIR}"
echo "  path_to_imagenet: ${PATH_TO_IMAGENET}"
echo "  workdir: $(pwd)"
echo "  first_models: ${MODEL_NAMES[*]:0:3}"

# Resolve a dataset label to (dataset_arg, dataset_paths_prefix).
# Empty prefix means do not pass --dataset_paths_prefix.
resolve_dataset() {
    local label="$1"
    case "${label}" in
        NINCO)
            DATASET_ARG="NINCO"
            DATASET_PREFIX=""
            ;;
        Openimage-O|openimage-o|openimage_o|Openimage-o)
            DATASET_ARG="./data/openimages_o.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        SSB-Hard|ssb-hard|ssb_hard|SSB-hard)
            DATASET_ARG="./data/ssb_hard.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        ImageNet-C|imagenet-c|imagenet_c)
            DATASET_ARG="./data/imagenet_c.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        ImageNet-ES|imagenet-es|imagenet_es)
            DATASET_ARG="./data/imagenet_es.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        ImageNet-R|imagenet-r|imagenet_r)
            DATASET_ARG="./data/imagenet_r.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        ImageNet-V2|imagenet-v2|imagenet_v2)
            DATASET_ARG="./data/imagenet_v2.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        SUN|sun)
            DATASET_ARG="./data/sun.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        ImageNet-O|imagenet-o|imagenet_o)
            DATASET_ARG="./data/imagenet_o.csv"
            DATASET_PREFIX="${REPO_ROOT}/OpenOOD/data"
            ;;
        *)
            echo "Unknown OOD dataset label: ${label}" >&2
            return 1
            ;;
    esac
}

run_dataset() {
    local model_name="$1"
    local label="$2"

    resolve_dataset "${label}"

    echo
    echo "Running model/dataset: ${model_name} / ${label}"
    echo "  dataset: ${DATASET_ARG}"
    if [ -n "${DATASET_PREFIX}" ]; then
        echo "  dataset_prefix: ${DATASET_PREFIX}"
    fi

    local cmd=(
        python3 evaluate_lt.py
        --model_name "${model_name}"
        --method all
        --dataset "${DATASET_ARG}"
        --batch_size "${BATCH_SIZE}"
        --overwrite_model_outputs "${OVERWRITE}"
        --n_bootstrap_seeds "${BOOTSTRAP_SEEDS}"
        --path_to_cache "${CACHE_DIR}"
        --path_to_imagenet "${PATH_TO_IMAGENET}"
    )

    if [ -n "${DATASET_PREFIX}" ]; then
        cmd+=(--dataset_paths_prefix "${DATASET_PREFIX}")
    fi

    "${cmd[@]}"
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for OOD_LABEL in "${OOD_LIST[@]}"; do
        run_dataset "${MODEL_NAME}" "${OOD_LABEL}"
    done
done
