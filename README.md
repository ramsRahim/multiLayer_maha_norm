# Mahalanobis++: Improving OOD Detection via Feature Normalization

Maximilian Müller, Matthias Hein — **University of Tübingen / Tübingen AI Center**

Paper: [https://arxiv.org/abs/2505.18032](https://arxiv.org/abs/2505.18032)

---

## Overview

Mahalanobis++ (MM++) is a **training-free, post-hoc** OOD detection method.
It combines $\ell_2$-normalized features from multiple intermediate layers of any pretrained backbone into a single Mahalanobis-distance score.
No fine-tuning or hyperparameter search on OOD data is required.

---

## Installation

```bash
conda env create -f environment.yml
conda activate NINCO_maha
pip install libmr==0.1.9
```

---

## Data Setup

### NINCO
```bash
wget https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1 -O NINCO_all.tar.gz
tar -xvzf NINCO_all.tar.gz
```

### OpenOOD datasets (Textures, iNaturalist, OpenImage-O, ImageNet-C/R/ES/V2, …)
```bash
python scripts/download_datasets.py \
    --contents datasets \
    --datasets ood_v1.5 \
    --save_dir ./data \
    --dataset_mode benchmark
```

### ImageNet
Set paths in the evaluation scripts (see below) to point to your local ImageNet-1K `train/` and `val/` directories.

---

## Repository Structure

```
evaluate.py              – main evaluation entry point
detection_methods.py     – all OOD scoring methods (MM++, Maha, KNN, MSP, …)
utils.py                 – backbone configs, feature extraction, metrics
datasets.py              – dataset wrappers
resnet50.py              – custom ResNet used by KNN-OOD baseline
benchmark_overhead.py    – offline/online efficiency comparison (Table 2)
environment.yml          – conda environment
data/                    – CSV manifests for each OOD benchmark
assets/                  – paper figures
scripts/
  run_eval.sh            – ViT-B/16 + ImageNet-LT ID, 7 OOD datasets
  run_eval_vit_imagenet.sh  – ViT-B/16 + full ImageNet-1K ID
  run_eval_resnet50.sh   – ResNet-50 + ImageNet-LT ID
  download_datasets.py   – download OOD benchmark datasets
```

---

## Running Evaluation

All scripts log to `/tmp/` and print AUROC + FPR@95 for each method × dataset.
Features are cached on first run — subsequent runs reuse them.

### ViT-B/16 · ImageNet-LT ID · 7 OOD datasets (main result)

```bash
bash scripts/run_eval.sh
```

Evaluates 10 methods: MSP, Energy, Energy+React, ODIN, Mahalanobis, **Mahalanobis_norm**, Relative_Mahalanobis, Relative_Mahalanobis_norm, KNN, **MM_plus_plus_topk_cat** (MM++).  
OOD datasets: NINCO, OpenImage-O, SSB-Hard, ImageNet-C, ImageNet-ES, ImageNet-R, ImageNet-V2.

### ViT-B/16 · full ImageNet-1K ID

```bash
bash scripts/run_eval_vit_imagenet.sh
```

### ResNet-50 · ImageNet-LT ID

```bash
bash scripts/run_eval_resnet50.sh
```

### Single method / dataset (quick test)

```bash
conda run -n NINCO_maha python evaluate.py \
    --model vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --method MM_plus_plus_topk_cat \
    --dataset ./data/NINCO_OOD_classes.csv \
    --dataset_paths_prefix /path/to/NINCO/NINCO \
    --path_to_cache ./cache_imagenetlt \
    --train_dir /path/to/imagenetlt/train \
    --val_dir   /path/to/imagenetlt/test
```

**Key flags:**

| Flag | Description |
|---|---|
| `--model` | timm model name (see `utils.py` for full list) |
| `--method` | OOD method name, or `all` for every method |
| `--dataset` | OOD dataset name or path to a `.csv` manifest |
| `--dataset_paths_prefix` | Root directory prepended to CSV image paths |
| `--path_to_cache` | Where to store/load cached features |
| `--train_dir` / `--val_dir` | Override default ImageNet train / val paths |

---

## Available Methods

| Method name | Description |
|---|---|
| `MSP` | Maximum softmax probability |
| `Energy` | Energy score |
| `Energy+React` | Energy with ReAct clipping |
| `ODIN` | ODIN (temperature scaling + input perturbation) |
| `Mahalanobis` | Standard Mahalanobis on pre-logit features |
| `Mahalanobis_norm` | **Maha++** — Mahalanobis with $\ell_2$-normalized features |
| `Relative_Mahalanobis` | Relative Mahalanobis |
| `Relative_Mahalanobis_norm` | Relative Mahalanobis with $\ell_2$-normalization |
| `knn` | KNN-OOD |
| `MM_plus_plus_topk_cat` | **MM++** — multi-layer Maha++ with automatic layer selection ($K=2$) |

---

## Efficiency Benchmark

Reproduces the offline/online cost comparison (Table 2):

```bash
# Inference only (fast, no large RAM needed)
conda run -n NINCO_maha python benchmark_overhead.py --skip-calib --save-fig

# Full run including calibration timing
conda run -n NINCO_maha python benchmark_overhead.py --save-fig
```

Outputs `benchmark_overhead.pdf` and prints a summary table.
To save each subplot as a separate PDF:

```bash
conda run -n NINCO_maha python save_subfigs.py --skip-calib
```

---

## Citation

```bibtex
@inproceedings{mueller2025mahalanobispp,
  title     = {Mahalanobis++: Improving OOD Detection via Feature Normalization},
  author    = {Maximilian Mueller and Matthias Hein},
  booktitle = {ICML},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.18032}
}
```

This repository builds on [NINCO](https://github.com/j-cb/NINCO) and [OpenOOD](https://github.com/Jingkang50/OpenOOD); please cite their work too.
