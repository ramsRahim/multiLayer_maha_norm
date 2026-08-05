We thank the reviewer for the comprehensive feedback we addressed below.

### Weaknesses

> **W1: ‘Fully unsupervised’ wording is questionable. The method uses ID labels to compute class means and class-conditional covariance, so it is not unsupervised in the strict sense; it is better described as no-OOD-data / post-hoc.”**

**Response:** We agree “fully unsupervised” is an overclaim as we need ID-calibration. We will replace it with “post-hoc without OOD supervision” throughout the paper. 

---

> **“W2: Mixed comparison to X-Mahalanobis. On ImageNet-1K, MM++ is clearly below X-Maha on average, on ImageNet-LT it slightly improves AUROC but not FPR95. Also, X-Maha uses fine-tuning in reported comparisons, making the fairness of the comparison somewhat complicated.”**

**Response:** 

Following this suggestion, we evaluated **X-Maha without fine-tuning**. In most cases, MM++ outperforms it on both AUROC / FPR95. We will update the tables in the revised paper. 

**ViT-B/16, ImageNet-1K as ID** 
| Method | ImageNet-O | Texture | Places365 | iNaturalist | SUN | Average |
|---|---:|---:|---:|---:|---:|---:|
| X-Maha w/o FT | 88.28 / **53.90** | 89.61 / 43.79 | 82.92 / 65.37 | 96.76 / 17.08 | 86.13 / 59.37 | 88.74 / 47.90 |
| MM++ | **88.76** /54.45 | **97.00** / **14.52** | **87.05** / **55.58** | **98.43** / **5.71** | **89.64** / **49.66** | **92.18** / **35.98** |

**ViT-B/16, ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| X-Maha w/o FT | 88.29 / 54.17 | 95.58 / 24.71 | 75.02 / 73.73 | 71.78 / 74.98 | 86.47 / 53.20 | 56.31 / 92.63 | 78.91 / 62.24 |
| MM++ | **91.34** / **43.20** | **96.43** / **20.98** | **84.35** / **53.80** | **81.70** / **52.43** | **88.84** / **41.83** | **60.78** / **88.94** | **83.91** / **50.20** |

**EVA02-S14, ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| X-Maha w/o FT | 78.47 / 87.96 | 85.73 / 85.36 | **89.28** / 60.49 | 77.25 / 85.82 | 84.28 / 65.93 | 62.38 / 93.65 | 79.56 / 79.87 |
| MM++| **82.06** / **72.87** | **87.49** / **66.96** | 89.25 / **42.51** | **88.24** / **47.31** | **90.31** / **42.91** | **65.68** / **85.62** | **83.84** / **59.70** |


---

> **“W3. Layer-selection criterion needs more justification. Entropy density may be sensitive to feature dimension, especially across heterogeneous architectures. It is not fully established that entropy density drops reliably identify semantic compression boundaries.”**

**Response:** Please refer to our response to Q1 below.

 
### Questions

> **“Q1. Justify entropy density. Why is entropy divided by feature dimension rather than normalized by log \(D_l\) or expressed as effective rank \exp(H_l)\? Could \H_l/D_l\ bias layer selection in heterogeneous architectures like Swin or ConvNeXt?”**

To address this thoughtful question, we evaluated all normalized/unnormalized criteria across isotropic (ViT-B/16) and hierarchical (Swin-T, ConvNeXt-T) backbones on the Open-OOD benchmark (9 OOD datasets): 

**ViT-B/16, ImageNet-1K as ID, Open-OOD benchmark** (AUROC $\uparrow$ / $\text{FPR}_{95} \downarrow$) 

| Method | Selected layers | NINCO | SSB-Hard | ImageNet-O | OpenImages-O | SUN | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| \(H_l/D_l\) | block_04 | 91.38 / 42.49 | 78.75 / 65.79 | 88.76 / 54.45 | 96.27 / 21.10 | 89.64 / 49.66 | 88.47 / 41.96 | 84.58 / 45.99 | 89.15 / 42.17 | 58.85 / 90.33 | **85.09 / 50.44** |
| \(H_l/\log D_l\) | block_04 | 91.38 / 42.49 | 78.75 / 65.79 | 88.76 / 54.45 | 96.27 / 21.10 | 89.64 / 49.66 | 88.47 / 41.96 | 84.58 / 45.99 | 89.15 / 42.17 | 58.85 / 90.33 | **85.09 / 50.44** |
| \(\exp(H_l)\) | block_11 | 85.34 / 78.14 | 74.85 / 89.65 | 81.65 / 82.55 | 92.12 / 52.77 | 84.51 / 71.76 | 72.63 / 88.39 | 72.71 / 87.37 | 86.13 / 55.45 | 59.16 / 90.27 | 78.79 / 77.37 |
| \(H_l\) | block_04 | 91.38 / 42.49 | 78.75 / 65.79 | 88.76 / 54.45 | 96.27 / 21.10 | 89.64 / 49.66 | 88.47 / 41.96 | 84.58 / 45.99 | 89.15 / 42.17 | 58.85 / 90.33 | **85.09 / 50.44** |
| \(\exp(H_l)/D_l\) | block_11 | 85.34 / 78.14 | 74.85 / 89.65 | 81.65 / 82.55 | 92.12 / 52.77 | 84.51 / 71.76 | 72.63 / 88.39 | 72.71 / 87.37 | 86.13 / 55.45 | 59.16 / 90.27 | 78.79 / 77.37 |


**Swin-T, ImageNet-1K as ID, Open-OOD benchmark** 
| Method | Selected layers | NINCO | SSB-Hard | ImageNet-O | OpenImages-O | SUN | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| \(H_l/D_l\) | stage_2_block_00 | 85.73 / 59.99 | 71.45 / 75.47 | 79.46 / 78.45 | 92.66 / 42.74 | 81.57 / 79.32 | 85.89 / 44.48 | 84.81 / 44.60 | 88.64 / 46.10 | 58.26 / 91.31 | **80.94 / 62.50** |
| \(H_l/\log D_l\) | stage_0_block_01 | 86.71 / 56.44 | 72.01 / 73.18 | 78.75 / 80.45 | 93.57 / 36.29 | 83.04 / 73.10 | 82.39 / 52.78 | 78.01 / 62.33 | 88.38 / 46.22 | 58.31 / 91.12 | 80.13 / 63.55 |
| \(\exp(H_l)\) | stage_0_block_01 | 86.71 / 56.44 | 72.01 / 73.18 | 78.75 / 80.45 | 93.57 / 36.29 | 83.04 / 73.10 | 82.39 / 52.78 | 78.01 / 62.33 | 88.38 / 46.22 | 58.31 / 91.12 | 80.13 / 63.55 |
| \(H_l\) | stage_0_block_01 | 86.71 / 56.44 | 72.01 / 73.18 | 78.75 / 80.45 | 93.57 / 36.29 | 83.04 / 73.10 | 82.39 / 52.78 | 78.01 / 62.33 | 88.38 / 46.22 | 58.31 / 91.12 | 80.13 / 63.55 |
| \(\exp(H_l)/D_l\) | stage_2_block_00 | 85.73 / 59.99 | 71.45 / 75.47 | 79.46 / 78.45 | 92.66 / 42.74 | 81.57 / 79.32 | 85.89 / 44.48 | 84.81 / 44.60 | 88.64 / 46.10 | 58.26 / 91.31 | **80.94 / 62.50** |

**ConvNeXt-T, ImageNet-1K as ID, Open-OOD benchmark** 
| Method | Selected layers | NINCO | SSB-Hard | ImageNet-O | OpenImages-O | SUN | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| \(H_l/D_l\) | stage_1_block_00 | 87.76 / 53.44 | 73.88 / 72.28 | 79.32 / 78.40 | 93.64 / 37.48 | 82.16 / 77.72 | 84.17 / 51.83 | 79.99 / 63.71 | 87.98 / 50.61 | 58.87 / 91.32 | **80.86 / 64.09** |
| \(H_l/\log D_l\) | stage_1_block_02 | 87.49 / 53.89 | 73.78 / 72.27 | 78.98 / 78.55 | 93.50 / 37.87 | 81.81 / 78.41 | 82.75 / 56.09 | 77.92 / 71.00 | 87.25 / 54.01 | 59.10 / 90.95 | 80.29 / 65.89 |
| \(\exp(H_l)\) | stage_3_block_01 | 84.58 / 71.05 | 71.10 / 85.37 | 77.11 / 84.95 | 93.05 / 40.99 | 83.54 / 72.52 | 76.93 / 72.13 | 73.43 / 81.51 | 86.93 / 53.08 | 58.16 / 91.51 | 78.31 / 72.57 |
| \(H_l\) | stage_1_block_02 | 87.49 / 53.89 | 73.78 / 72.27 | 78.98 / 78.55 | 93.50 / 37.87 | 81.81 / 78.41 | 82.75 / 56.09 | 77.92 / 71.00 | 87.25 / 54.01 | 59.10 / 90.95 | 80.29 / 65.89 |
| \(\exp(H_l)/D_l\) | stage_1_block_02 | 87.49 / 53.89 | 73.78 / 72.27 | 78.98 / 78.55 | 93.50 / 37.87 | 81.81 / 78.41 | 82.75 / 56.09 | 77.92 / 71.00 | 87.25 / 54.01 | 59.10 / 90.95 | 80.29 / 65.89 |

- $H_l/D_l$ is the **only criterion that consistently achieves the best performance across all tested backbones**, confirming its robustness across both isotropic and heterogeneous architectures.

- **Isotropic Equivalency (ViT-B/16):** Because $D_l = 768$ is constant across layers, $H_l/D_l$, $H_l/\log D_l$, and $H_l$ are strictly monotonic scalings of each other. They preserve rank order and select the exact same layer (block_04).

- **Hierarchical Advantage (Swin/ConvNeXt):** When channel capacity varies across stages, dividing by $D_l$ measures entropy density (information per channel dimension), eliminating stage-width bias.

**Effective Rank ($\exp(H_l)$) Failure:** Over-indexes on deep, high-dimensional layers (block_11 / stage_3), degrading AUROC by up to $6.30\%$.

We will add this theoretical context and full 9-dataset ablation table to Section 3 and Section 4 (or Appendix B), respectively.

---

> **“Q2. Clarify ‘fully unsupervised’. MM++ uses ID labels to estimate class means and class-conditional covariance. In what sense is the method ‘fully unsupervised’? Would ‘OOD-unsupervised’ or ‘no-OOD-data post-hoc’ be more accurate?”**

**Response:** We agree “fully unsupervised” is an overclaim as we need ID-calibration. We will replace it with “post-hoc without OOD supervision” throughout the paper.

---

> **“Q3. Fairness of comparison to X-Mahalanobis. The paper compares MM++ to X-Mahalanobis results that involve fine-tuning. Can the authors provide a strictly post-hoc / no-fine-tuning X-Mahalanobis comparison under the same setup?”**

**Response:** Please refer to our response to W2.
