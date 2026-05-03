#!/bin/bash
#SBATCH --job-name=download_imgnet
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# 1. Create logs directory if it doesn't exist
mkdir -p logs

source /data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in the environment}"

# 4. Run the script
python scripts/imagenet_1k_download.py \
    --local-dir /data/home/mislambhuian/data/imagenet-1k \
    --token $HF_TOKEN