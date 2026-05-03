#!/bin/bash
#SBATCH --job-name=k-sensitivity-viz
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

MAX_K="${MAX_K:-auto}"
OUT_PREFIX="${OUT_PREFIX:-${REPO_ROOT}/assets/k_sensitivity_architectures_ood}"
METRICS_CACHE="${METRICS_CACHE:-${OUT_PREFIX}.metrics.json}"
SAMPLE_AUROC="${SAMPLE_AUROC:-no}"
FORCE="${FORCE:-no}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

source "/data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh"
conda activate NINCO_maha

cd "${REPO_ROOT}"
mkdir -p logs assets

cmd=(
    python3 scripts/viz/plot_k_sensitivity_architectures.py
    --out-prefix "${OUT_PREFIX}"
    --metrics-cache "${METRICS_CACHE}"
)

if [ "${MAX_K}" != "auto" ]; then
    cmd+=(--max-k "${MAX_K}")
fi

if [ -n "${MODELS:-}" ]; then
    # shellcheck disable=SC2206
    MODEL_LIST=(${MODELS})
    cmd+=(--models "${MODEL_LIST[@]}")
fi

if [ -n "${DATASETS:-}" ]; then
    # shellcheck disable=SC2206
    DATASET_LIST=(${DATASETS})
    cmd+=(--datasets "${DATASET_LIST[@]}")
fi

case "${SAMPLE_AUROC}" in
    yes|true|1) cmd+=(--sample-auroc) ;;
esac

case "${FORCE}" in
    yes|true|1) cmd+=(--force) ;;
esac

cmd+=("$@")

echo "Running K-sensitivity architecture plot"
echo "  max_k: ${MAX_K}"
echo "  out_prefix: ${OUT_PREFIX}"
echo "  metrics_cache: ${METRICS_CACHE}"
echo "  sample_auroc: ${SAMPLE_AUROC}"
echo "  force: ${FORCE}"
echo "  models: ${MODELS:-default}"
echo "  datasets: ${DATASETS:-default}"
echo "  threads: ${OMP_NUM_THREADS}"
echo "  workdir: $(pwd)"
echo "  command: ${cmd[*]}"

"${cmd[@]}"
