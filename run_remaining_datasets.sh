#!/bin/bash
# Run remaining OOD dataset evaluations sequentially (to avoid GPU OOM)
# Features are already cached for all datasets

set -e

MODEL="vit_base_patch16_224.augreg2_in21k_ft_in1k"
CACHE="./cache"

echo "=== Starting sequential OOD evaluations ==="
echo "Model: $MODEL"
echo ""

# ImageNet-O
echo "[$(date '+%H:%M:%S')] Starting ImageNet-O..."
conda run -n NINCO_maha python -u evaluate.py \
    --model "$MODEL" \
    --dataset "ImageNet-O" \
    --method all \
    2>&1 | tee /tmp/vit_imagenet_o_seq.txt
echo "[$(date '+%H:%M:%S')] ImageNet-O DONE"

# OpenImages-O
echo "[$(date '+%H:%M:%S')] Starting OpenImages-O..."
conda run -n NINCO_maha python -u evaluate.py \
    --model "$MODEL" \
    --dataset ./data/openimages_o.csv \
    --dataset_paths_prefix /home/rahim/exp/Data/openood_data \
    --method all \
    2>&1 | tee /tmp/vit_openimages_seq.txt
echo "[$(date '+%H:%M:%S')] OpenImages-O DONE"

# Places365
echo "[$(date '+%H:%M:%S')] Starting Places365..."
conda run -n NINCO_maha python -u evaluate.py \
    --model "$MODEL" \
    --dataset ./data/places365.csv \
    --dataset_paths_prefix /home/rahim/exp/Data/openood_data \
    --method all \
    2>&1 | tee /tmp/vit_places365_seq.txt
echo "[$(date '+%H:%M:%S')] Places365 DONE"

echo ""
echo "=== All evaluations complete ==="
find "$CACHE/scores/$MODEL" -name "*.npz" | sort