#!/bin/bash
#SBATCH --job-name=extract_imgnet
#SBATCH --partition=Standard.2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_imagenet_%j.log
#SBATCH --error=logs/extract_imagenet_%j.err

source /data/home/mislambhuian/miniforge3/etc/profile.d/conda.sh
conda activate NINCO_maha

cd /data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm

python3 scripts/extract_imagenet.py
