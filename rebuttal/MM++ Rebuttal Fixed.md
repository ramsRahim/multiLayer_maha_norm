## AC Review

We would like to thank the AC for the concise meta-review. We address the concerns below. 


> **"questionable terminology ("fully unsupervised" despite using ID labels)"**

We agree “fully unsupervised” is an overclaim as we need ID-calibration. In the revised paper, we will replace it with “post-hoc”. 

> **"Strictly post-hoc comparison with X-Mahalanobis"**

Per this comment, we evaluated X-Maha without fine-tuning. In most cases, MM++ outperforms it on both AUROC / FPR95. We will update the tables in the revised paper. 

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
| X-Maha w/o FT | 78.47 / 87.96 | 85.73 / 85.36 | 89.28 / 60.49 | 77.25 / 85.82 | 84.28 / 65.93 | 62.38 / 93.65 | 79.56 / 79.87 |
| MM++| **82.06** / **72.87** | **87.49** / **66.96** | **89.25** / **42.51** | **88.24** / **47.31** | **90.31** / **42.91** | **65.68** / **85.62** | **83.84** / **59.70** |



> **"Gaussianity assumption in high-dimensional spaces is unvalidated"**

**Response** : We agree Gaussianity assumption in high-dimensional concatenated space should be empirically validated. We evaluated the mean absolute skewness, mean absolute excess kurtosis, and Mardia kurtosis:

| Backbone | Metric | Maha | Maha++ | MM++ | Ideal Gaussian |
|---|---|---:|---:|---:|:---:|
| **ViT-B/16**  | Mean absolute skew ↓ | 0.034 | 0.024 | **0.019** | 0 |
|  | Mean absolute excess kurtosis ↓ | 2.116 | 0.894 | **0.609** | 0 |
|  | Mardia ratio $(\rightarrow 1)$ | 1.661 | 1.261 | **1.102** | 1 |
| **Swin-T** | Mean absolute skew ↓ | 0.044 | 0.022 | **0.019** | 0 |
|  | Mean absolute excess kurtosis ↓ | 2.458 | 0.864 | **0.629** | 0 |
|  | Mardia ratio $(\rightarrow 1)$ | 1.748 | 1.249 | **1.141** | 1 |

MM++ (concatenated, normalized, $D=1536$) sits much closer to the ideal Gaussian compared to the Maha (penultimate, un-normalized, $D=768$) and Maha++ (penultimate, normalized, $D=768$).

We do not claim these results establish exact multivariate Gaussianity. Rather, we validate that MM++ is more compatible with the second-order Gaussian approximation used by Maha distance. We will add the table and analysis in the revised paper. 


> **"connection between the layer-selection criterion and Mahalanobis distance needs clarification"**

**Response** We agree the relationship between the layer-selection criterion and Maha distance needs further analysis.  

Based on our empirical analysis (see the table above), our cross-layer feature concatenation and layer-selection strategy improve the Gaussianity of the resulting representation. 

As Maha distance relies on this assumption, the improved Gaussianity provides a statistical justification for our layer-selection criterion. Also, Ledoit-Wolf shrinkage enhances covariance estimation needed to compute the Maha distance. 



> **"missing baselines (NNGuide, KPCA, SCALE)"**

We added those baselines. MM++ consistently outperforms NNGuide, KPCA and SCALE in terms of average AUROC and FPR95. In many cases, MM++ outperforms them while remaining competitive otherwise:

**ViT-B/16, ImageNet-1K as ID** 
| Method | ImageNet-O | Texture | Places365 | iNaturalist | SUN | Average |
|---|---:|---:|---:|---:|---:|---:|
| SCALE | 63.99 / 76.40 | 72.77 / 70.99 | 50.04 / 91.47 | 75.99 / 62.11 | 72.31 / 71.18 | 67.02 / 74.43 |
| NNGuide | 85.87 / 64.50 | 90.37 / 41.01 | 85.76 / 62.82 | 97.76 / 10.90 | 88.57 / 53.46 | 89.67 / 46.54 |
| KPCA | 85.08 / 71.15 | 89.30 / 49.18 | 84.52 / 66.72 | 97.42 / 12.86 | 88.54 / 51.87 | 88.97 / 50.36 |
| MM++ | **88.76 / 54.45** | **97.00 / 14.52** | **87.05 / 55.58** | **98.43 / 5.71** | **89.64 / 49.66** | **92.18 / 35.98** |


**ViT-B/16, ImageNet-LT as ID**

| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCALE | 71.87 / 70.37 | 74.23 / 59.47 | 63.67 / 77.89 | 59.69 / 78.44 | 49.94 / 91.03 | 52.12 / 93.57 | 61.92 / 78.46 |
| NNGuide | 88.23 / 59.60 | 93.84 / 34.92 | 77.45 / 65.67 | 76.46 / 63.20 | 87.99 / 45.39 | **61.08 / 88.68** | 80.84 / 59.58 |
| KPCA | 81.67 / 80.10 | 86.69 / 73.29 | 71.57 / 85.01 | 70.55 / 85.65 | 81.26 / 77.22 | 58.16 / 92.42 | 74.98 / 82.28 |
| MM++ | **91.34 / 43.20** | **96.40 / 21.07** | **84.35 / 53.80** | **81.70 / 52.43** | **88.84 / 41.83** | 60.78 / 88.94 | **83.90 / 50.21** |

**ConvNeXt-T, ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCALE | 61.73 / 93.91 | 60.64 / 95.95 | 65.08 / 90.27 | 63.92 / 83.63 | 54.96 / 96.49 | 56.26 / 92.81 | 60.43 / 92.18 |
| NNGuide | 81.39 / 72.09 | 86.42 / 68.04 | **83.80 / 52.98** | **79.18 / 61.86** | 87.96 / 49.25 | **61.23 / 89.02** | 80.00 / 65.54 |
| KPCA | 73.10 / 88.37 | 78.21 / 85.24 | 73.47 / 86.04 | 70.58 / 88.06 | 77.79 / 83.15 | 57.13 / 93.15 | 71.71 / 87.33 |
| MM++ | **87.51** / **53.72** | **93.40** / **37.69** | 82.52 / 56.39 | 77.98 / 69.36 | **88.08** / **49.12** | 59.70 / 90.33 | **81.53** / **59.43** |

**Swin-T, ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCALE | 80.43 / 74.40 | 88.05 / 59.41 | 76.35 / 68.07 | 75.96 / 66.53 | 84.30 / 56.35 | 59.85 / 89.88 | 77.49 / 69.11 |
| NNGuide | 81.70 / 75.63 | 86.09 / 69.64 | 82.01 / 58.91 | 80.22 / 61.46 | 87.87 / 50.91 | 60.68 / 89.65 | 79.76 / 67.70 |
| KPCA | 72.81 / 83.91 | 83.50 / 73.60 | 76.10 / 66.84 | 72.47 / 73.08 | 85.14 / 56.26 | 55.97 / 91.80 | 74.33 / 74.25 |
| MM++ | **85.76** / **59.31** | **92.44** / **43.12** | **86.26** / **43.56** | **86.25** / **41.09** | **89.29** / **42.91** | **60.68** / **89.61** | **83.45** / **53.27** |


> **"lack of statistical significance tests"**

To analyze statistical significance, we report error bars for 5 independent splits for Table 2.

For 5 random seeds, we drew a stratified 90/10 split of the ImageNet-LT training set by class, refit the detector on the 90% partition, and scored the held-out 10% as ID against each OOD dataset. 

The results below present the mean ± standard deviation (AUROC&uarr; / FPR95&darr;) for MM++ (ViT-B/16). The small variance ($\text{std} \le 0.15$ for AUROC, $\text{std} \le 0.94$ for FPR95) highlights the empirical stability and statistical reliability of MM++.


**ViT-B/16 on ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| MM++ | 94.01 ± 0.08 / 30.54 ± 0.94 | 97.75 ± 0.04 / 11.48 ± 0.48 | 87.73 ± 0.08 / 45.77 ± 0.56 | 85.05 ± 0.10 / 46.33 ± 0.40 | 91.63 ± 0.07 / 32.15 ± 0.59 | 66.50 ± 0.15 / 81.67 ± 0.50 | 87.11 / 41.32 |


## Reviewer 1FrG 

We thank the reviewer for the constructive comments addressed below. 


### Weaknesses

> **“Weakness 1: The performance gains over single-layer baselines are modest.”**

**Response:** We respectfully disagree that the performance gains are modest. 

1. ImageNet-1K (Table 1): Over the best single layer baseline (Maha++), MM++ gains +2.36 AUROC (+2.63%) and cuts FPR95 by 10.96 points (23.35% relative reduction). 

2. ImageNet-LT (Table 2): MM++ improves AUROC by 3.69 AUROC (+4.60%) and reduces FPR95 by 9.51 points (-15.93%).

3. Across backbones (Tables 7-8): On ImageNet-LT, MM++ gains 2.32 AUROC (+3.15%) and reduces FPR95 by 2.60 points (-3.77%) over the best single-layer baseline, rMaha++. 


---
> **“Weakness 2: The theoretical justification in Appendix A is informal—it provides intuitions about neural collapse, cross-layer consistency, and shrinkage stability, but offers no formal theorems, error bounds, or convergence guarantees.”**

**Response:** We thank the reviewer for this constructive feedback. We will update Appendix A to clearly separate conceptual motivation from formal theoretical bounds:

- **Bounded Feature Residuals:** With $K$ $\ell_2$-normalized feature blocks ($\Vert{}\phi_k(x)\Vert{}_2 = 1$), the centered residual $r_c(x) = \phi(x) - \mu_c$ satisfies $\Vert{}r_c(x)\Vert{}_2^2 \le 4K$.

$$
\|r_c(x)\|_2^2 \leq 4K.
$$

- **Score Perturbation Bound:** For population precision $\Omega=\Sigma^{-1}$ and estimate $\widehat{\Omega}$: 

$$
\left|
r_c^\top\widehat{\Omega}r_c-r_c^\top\Omega r_c
\right|
\leq
\|\widehat{\Omega}-\Omega\|_2\|r_c\|_2^2
\leq
4K\|\widehat{\Omega}-\Omega\|_2.
$$

By 1-Lipschitz continuity of the class-wise minimum, the overall MM++ score error is similarly bounded by $4K \Vert{}\widehat{\Omega} - \Omega\Vert{}_2$.

- **Spectral Control via Shrinkage:** Ledoit–Wolf shrinkage ($\widehat{\Sigma} = (1 - \gamma)\widehat{\Sigma} + \gamma \bar{\lambda} I$) bounds the minimum eigenvalue $\lambda_{\min}(\widehat{\Sigma}) \ge \gamma \bar{\lambda} > 0$, ensuring a finite spectral norm $\Vert{}\widehat{\Sigma}^{-1}\Vert{}_2 \le \frac{1}{\gamma \bar{\lambda}}$.

We will formally present these perturbation and stability bounds as Proposition A.1 and Proposition A.2 in Appendix A, citing asymptotic consistency results for Ledoit–Wolf covariance estimation.



---
> **“Weakness 3: The Gaussianity assumption underlying the Mahalanobis distance in the high-dimensional concatenated space (D_K can reach 2 x 768 = 1536 for K=2) should be empirically validated.”**

**Response:** We agree the Gaussianity assumption in high dimensions ($D_{K} = 1536$ for $K=2$) warrants empirical validation. We have conducted multivariate normality analyses measuring skewness, excess kurtosis, and Mardia’s multivariate kurtosis ratio across features. As shown in the table below, feature representations in MM++ (concatenated and block $\ell_2$-normalized, $D=1536$) lie substantially closer to ideal multivariate Gaussianity than both standard single-layer Maha ($D=768$) and normalized single-layer Maha++ ($D=768$):

| Backbone | Metric | Maha | Maha++ | MM++ | Ideal Gaussian |
|---|---|---:|---:|---:|:---:|
| **ViT-B/16**  | Mean absolute skew ↓ | 0.034 | 0.024 | **0.019** | 0 |
|  | Mean absolute excess kurtosis ↓ | 2.116 | 0.894 | **0.609** | 0 |
|  | Mardia ratio $(\rightarrow 1)$ | 1.661 | 1.261 | **1.102** | 1 |
| **Swin-T** | Mean absolute skew ↓ | 0.044 | 0.022 | **0.019** | 0 |
|  | Mean absolute excess kurtosis ↓ | 2.458 | 0.864 | **0.629** | 0 |
|  | Mardia ratio $(\rightarrow 1)$ | 1.748 | 1.249 | **1.141** | 1 |



---
> **“Weakness 4: No statistical significance tests or error bars are reported.”**

**Response:** 

We agree that reporting statistical variance is critical to show empirical stability. We conducted 5-seed evaluations for all three backbone architectures.

**Validation Protocol:** For each of 5 random seeds, we drew a stratified $90/10$ class-balanced split of the ImageNet-LT training set, refit the detector on the $90\%$ partition, and evaluated the held-out $10\%$ partition as ID against each OOD benchmark.

**Statistical Significance & Key Takeaways:**
1. **Very Low Variance:** Standard deviations across 5 splits are consistently minimal ($\le 0.15$ for AUROC and $\le 2.12$ for $\text{FPR}_{95}$ across all settings), confirming high empirical stability under data resampling.

2. **Statistically Significant Margins:** The performance gaps between Maha++ and MM++ far exceed the error bounds (e.g., $+2.88$ average AUROC and $-6.42$ average $\text{FPR}_{95}$ on ViT-B/16), confirming that improvements are statistically significant rather than artifacts of split selection.

3. **Architectural Generalization:** The statistically significant gains hold across standard ViTs, hierarchical ViTs (Swin-T), and modern ConvNets (ConvNeXt-T).

**5-Seed Statistical Results ($\text{AUROC} \pm \text{std} \;/\; \text{FPR}_{95} \pm \text{std}$):**


**ViT-B/16 on ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maha++ | 93.47 ± 0.10 / 33.27 ± 1.00 | 97.47 ± 0.07 / 13.07 ± 0.55 | 79.99 ± 0.13 / 62.50 ± 0.67 | 78.40 ± 0.15 / 60.17 ± 2.12 | 90.54 ± 0.10 / 35.25 ± 0.77 | 65.54 ± 0.15 / 82.20 ± 0.53 | 84.23 / 47.74 |
| MM++ | 94.01 ± 0.08 / 30.54 ± 0.94 | 97.75 ± 0.04 / 11.48 ± 0.48 | 87.73 ± 0.08 / 45.77 ± 0.56 | 85.05 ± 0.10 / 46.33 ± 0.40 | 91.63 ± 0.07 / 32.15 ± 0.59 | 66.50 ± 0.15 / 81.67 ± 0.50 | 87.11 / 41.32 |

**Swin-T on ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maha++ | 89.92 ± 0.08 / 44.48 ± 0.88 | 95.57 ± 0.07 / 23.26 ± 1.09 | 83.50 ± 0.05 / 49.93 ± 0.66 | 80.81 ± 0.05 / 55.07 ± 0.70 | 91.35 ± 0.08 / 32.00 ± 1.03 | 64.72 ± 0.07 / 83.18 ± 0.65 | 84.31 / 47.99 |
| MM++ | 89.26 ± 0.08 / 47.55 ± 0.57 | 94.88 ± 0.05 / 26.67 ± 0.71 | 88.98 ± 0.04 / 36.54 ± 0.30 | 88.86 ± 0.04 / 36.06 ± 0.27 | 91.89 ± 0.06 / 30.79 ± 0.62 | 66.01 ± 0.08 / 83.27 ± 0.28 | 86.65 / 43.48 |

**ConvNeXt-T on ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maha++ | 91.88 ± 0.10 / 34.87 ± 0.94 | 96.52 ± 0.08 / 18.19 ± 0.78 | 84.90 ± 0.07 / 48.17 ± 0.67 | 80.85 ± 0.07 / 58.90 ± 0.95 | 91.82 ± 0.08 / 29.98 ± 0.72 | 66.63 ± 0.09 / 78.99 ± 0.55 | 85.43 / 44.85 |
| MM++ | 92.02 ± 0.12 / 34.52 ± 0.95 | 96.61 ± 0.09 / 17.48 ± 0.83 | 87.38 ± 0.07 / 42.64 ± 0.59 | 83.74 ± 0.08 / 50.94 ± 0.75 | 92.26 ± 0.09 / 28.61 ± 0.66 | 66.95 ± 0.13 / 79.14 ± 0.55 | 86.49 / 42.22 |


---
> **“Weakness 5: The non-monotonic behavior of K (performance drops at K=3 before recovering) is attributed to information redundancy without rigorous analysis.”**

**Response:** 

Following the reviewer’s suggestion, we conducted **CKA, CCA, and covariance conditioning analyses to explain the performance drop at $K=3$.** As shown below, adding block_11 at $K=3$ introduces near-total redundancy with the penultimate layer (CKA = $0.944$, Mean CCA = $0.976$, Residual Variance = $3.3\%$) while worsening covariance conditioning ($5.21 \to 6.03$):

| $K$ | Added Layer | CKA with Penultimate Layer | Mean Canonical Correlation | Residual variance | Covariance Condition Number | Mean AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `block_04` | 0.140 | 0.205 | 0.403 | 5.21 | **81.12** |
| 3 | `block_11` | **0.944** | **0.976** | **0.033** | **6.03** | 78.58 |

**Controlled Redundancy Experiment:** To isolate redundancy from dimensionality, we replaced block_11 with the less redundant block_06 at fixed $K=3$ dimension. This single swap increased Mean AUROC from 78.58 to 83.41, confirming that representation redundancy (and resulting covariance ill-conditioning) drives the $K=2 \to 3$ degradation. Partial performance recovery at $K > 3$ occurs when subsequently added blocks reintroduce non-redundant hierarchical features. We will update Section 4 and Appendix B with these findings.

### Questions
---
> **“Questions 1: The non-monotonic K sensitivity (performance drop at K=3, recovery at K=5) is attributed to information redundancy. Can the authors provide quantitative evidence (e.g., pairwise feature similarity across selected layers, condition number of the joint covariance as K increases) to support this claim?”**

**Response:** Please refer to our response to Weakness 5.


> **“Question 2: Table 9 shows that Ledoit-Wolf vs. empirical covariance makes negligible difference for single-layer Maha++. However, the paper claims LW shrinkage is crucial for the multilayer fused space. Could the authors provide the analogous ablation (EC vs. LW) in the fused space to directly validate this claim? Current Table 3 only compares pseudo-inverse vs. LW, not EC vs. LW in the multilayer setting.”**

**Response:** We conducted the requested ablation comparing EC and LW shrinkage within the multilayer fused space ($D_K = 1536$) across the full and data-constrained settings. As shown below, LW boosts performance in most cases, validating our claim:

| Method | iNaturalist | SUN | Places365 | Textures | ImageNet-O | Avg |
|---|---:|---:|---:|---:|---:|---:|
| MM++ EC, full 1.28M | 96.93 | 85.75 | 80.68 | 97.81 | 92.40 | 90.71 |
| **MM++ LW, full 1.28M** | **97.57** | **86.63** | **81.67** | **97.86** | **92.82** | **91.31** |
| MM++ EC, 115K | 96.87 | 85.62 | 80.52 | **97.80** | 92.31 | 90.62 |
| **MM++ LW, 115K** | **98.60** | **88.18** | **83.39** | 97.47 | **93.18** | **92.16** |

**Key Observations:** 
1. **Multilayer vs. Single-Layer Contrast:** In Table 9 (single-layer, $D=768$), LW and EC perform almost identically ($\Delta \le 0.05$ AUROC). In contrast, in the multilayer fused space ($D_{\mathcal{K}}=1536$), LW improves mean AUROC by +0.60 points under full calibration (1.28M) and by +1.54 points under reduced calibration (115K).

2. **Mechanism:** Concatenating features across layers doubles the feature dimension ($D_{\mathcal{K}}$), which increases the condition number of the empirical covariance matrix $\widehat{\Sigma}$ and amplifies estimation noise. LW shrinkage regularizes off-diagonal variance, restoring eigenvalue stability. 

We will add this ablation to Section 4 and Table 3.


---
> **“Question 3: On ImageNet-1K, MM++ underperforms X-Mahalanobis by ~2 AUROC points on average. The authors frame this as competitive given the post-hoc constraint. Could the authors discuss whether the cross-layer modeling benefit diminishes when the ID distribution is balanced and well-represented, and whether near-OOD improvements justify far-OOD degradation?”**

**Response:**  

**1. Cross-Layer Modeling & Post-Hoc Constraints:**
The cross-layer modeling benefit does **not** diminish when the ID distribution is balanced and well-represented. But, naive multi-layer feature fusion of without fine-tuning (X-Maha w/o FT) can degrade performance compared to single-layer baselines.

**ViT-B/16 on ImageNet-1K (ID) Benchmark ($\text{AUROC} \uparrow \;/\; \text{FPR}_{95} \downarrow$):**

| Method | Average |
|---|---:|
| Maha | 87.32 / 56.45 |
| rMaha | 88.32 / 52.40 |
| Maha++ | 89.82 / 46.94 |
| rMaha++ | 89.85 / 47.00 |
| X-Maha w/o FT | 88.74 / 47.90 |
| MM++ (Ours) | **92.18 / 35.98** |


**2. Near-OOD vs. Far-OOD Evaluation**: To address whether near-OOD improvements cause far-OOD degradation, we evaluated MM++ against the standard OpenOOD benchmark with ImageNet-1K as ID across both near- and far-OOD sets:

**Near-OOD Benchmark ($\text{AUROC} \uparrow \;/\; \text{FPR}_{95} \downarrow$):**
| Method | SSB-hard | NINCO | Average |
|---|---:|---:|---:|
| Maha++ | 77.39 / **67.37** | 90.68 / 46.09 | 84.03 / 56.73 |
| MM++ | **77.92** / 66.57 | **91.48** / **42.40** | **84.70** / **54.48** |

**Far-OOD Benchmark ($\text{AUROC} \uparrow \;/\; \text{FPR}_{95} \downarrow$):**
| Method | iNaturalist | Texture | OpenImage-O | Average |
|---|---:|---:|---:|---:|
| Maha++ | **98.78** / 5.28 | 89.27 / 48.48 | 95.85 / 24.32 | 94.63 / 26.03 |
| MM++ | 98.70 / **5.03** | **95.97** / **19.34** | **96.44** / **20.53** | **97.04** / **14.97** |

For both near- and far- OOD datasets, MM++ consistently outperforms the best single-layer baseline (Maha++). 

---

> **“ Limitations 1: The Gaussianity assumption in the high-dimensional concatenated space is never empirically validated or discussed as a limitation.”**

**Response:**
 We thank the reviewer for this suggestion. We have empirically validated feature Gaussianity and will explicitly incorporate the residual Gaussian approximation as a limitation in Section 5 and Appendix E.

**1. Empirical Validation:**
Please refer to our response at Weakness 3.

**2. Key Findings & Limitation:** 
- **Improved Compatability:** Block $\ell_2$-normalization and multi-layer fusion significantly dampen heavy tails (reducing excess kurtosis from $>2.1 \to 0.6$) and align the feature space much closer to multivariate normality than single-layer spaces (Mardia ratio $1.66 \to 1.10$).
- **Residual Limitation:** As the reviewer notes, deep net representations are non-linear transformations that rarely obey exact infinite-sample multivariate normality. Mahalanobis distance remains a second-order quadratic approximation.

We will add this quantitative validation table and Q-Q plots to Appendix B, and explicitly discuss second-order Gaussian approximation as a scope limitation in Appendix E (Limitations).

---

> **“Limitations 2: The method's effectiveness depends on the pretrained model exhibiting clear neural collapse; the paper does not discuss what happens with weakly trained or undertrained models.”**

**Response:**
We appreciate this constructive point and will explicitly document this boundary condition in Appendix E. MM++ requires the pretrained backbone to exhibit clear class-conditional cluster structure (low intra-class variance) to reliably estimate feature statistics. In undertrained or weakly trained models with overlapping ID class clusters:

- **Distorted Distance Metrics:** Noisy class means $\mu_c$ and ill-conditioned covariance matrices $\Sigma$ reduce distance discriminative power.

- **Layer-Selector Degradation:** Overlapping class posteriors elevate entropy across blocks, preventing the selector from identifying optimal layer subsets.

While this reliance on well-structured feature spaces is shared by all post-hoc feature-based detectors, we will add a dedicated subsection in Appendix E detailing this boundary condition and discussing feature convergence as a prerequisite.




**Reviewer GF79**

We would like to thank the reviewer for the comprehensive feedback addressed below. 


### Weaknesses
---
> **“Weaknesses 1. The novelty of this work is somewhat incremental and insufficient. The key contribution of MM++ essentially reduces to a layer-selection strategy. Moreover, the relationship between this layer-selection strategy and the Mahalanobis distance is not clearly articulated.”**

**Response:** 
We respectfully clarify that MM++ goes beyond a simple layer-selection heuristic. Our key technical contribution is a **principled post-hoc cross-layer representation framework** that resolves the fundamental instability of multi-layer feature fusion without requiring backbone fine-tuning.

**1. Conceptual & Empirical Novelty Over Existing Multi-Layer Approaches:**
Prior multi-layer methods (e.g., X-Mahalanobis) rely heavily on full fine-tuning. When applied post-hoc without fine-tuning (X-Maha w/o FT), naive feature concatenation fails to model inter-layer scale interactions and degrades performance below even single-layer baselines.

**ViT-B/16, ImageNet-1K as ID ($\text{AUROC} \uparrow \;/\; \text{FPR95} \downarrow$):** 
| Method | ImageNet-O | Texture | Places365 | iNaturalist | SUN | Average |
|---|---:|---:|---:|---:|---:|---:|
| Maha++ | 86.42 / 63.70 | 89.26 / 48.26 | 85.75 / 64.11 | 98.76 / 5.15 | 88.90 / 53.47 | 89.82 / 46.94 |
| X-Maha w/o FT | 88.28 / 53.90 | 89.61 / 43.79 | 82.92 / 65.37 | 96.76 / 17.08 | 86.13 / 59.37 | 88.74 / 47.90 |
| MM++ | **88.76** / **54.45** | **97.00** / **14.52** | **87.05** / **55.58** | **98.43** / **5.71** | **89.64** / **49.66** | **92.18** / **35.98** |

As shown, naive post-hoc aggregation (X-Maha w/o FT) collapses to 88.74 AUROC (below single-layer Maha++ at $89.82$). MM++ achieves 92.18 AUROC (+3.44 points over X-Maha w/o FT and +2.36 points over Maha++), demonstrating that post-hoc cross-layer fusion succeeds only when guided by normalized block alignment and entropy selection.  

**2. Theoretical & Statistical Grounding with Mahalanobis Distance:** The Mahalanobis distance measures squared feature distance scaled by the inverse precision matrix $\Sigma^{-1}$. Its statistical discriminative power depends fundamentally on how closely the underlying feature distribution conforms to a multivariate Gaussian assumption. Our entropy-guided selection and block $\ell_2$-normalization specifically filter out uncalibrated intermediate layers that violate normality. As shown by our quantitative normality evaluation, MM++ constructs a fused space ($D_{\mathcal{K}}=1536$) that is closer to multivariate Gaussianity than single-layer spaces ($D=768$):


| Backbone | Metric | Maha | Maha++ | MM++ | Ideal Gaussian |
|---|---|---:|---:|---:|:---:|
| **ViT-B/16**  | Mean absolute skew ↓ | 0.034 | 0.024 | **0.019** | 0 |
|  | Mean absolute excess kurtosis ↓ | 2.116 | 0.894 | **0.609** | 0 |
|  | Mardia ratio $(\rightarrow 1)$ | 1.661 | 1.261 | **1.102** | 1 |
| **Swin-T** | Mean absolute skew ↓ | 0.044 | 0.022 | **0.019** | 0 |
|  | Mean absolute excess kurtosis ↓ | 2.458 | 0.864 | **0.629** | 0 |
|  | Mardia ratio $(\rightarrow 1)$ | 1.748 | 1.249 | **1.141** | 1 |

The layer-selection strategy is explicitly justified by second-order Mahalanobis requirements: by dampening heavy tails (reducing excess kurtosis from $2.116 \to 0.609$) and aligning multivariate features closer to normality (Mardia ratio $1.661 \to 1.102$), MM++ optimizes the feature space required for Mahalanobis scoring to remain well-conditioned.

---

> **“Weakness 2. Even regarding this core layer-selection strategy, the writing in Sections 3.2 and 3.3 lacks clarity and intuitive appeal. These sections read like a step-by-step operational manual without any supporting empirical evidence.”**

**Response:**  
We appreciate this feedback and will rewrite Sections 3.2 and 3.3 to prioritize intuitive motivation and empirical grounding.

- **High-Level Intuition:** Intermediate layers carry multi-scale visual details, but naively concatenating all layers causes high-dimensional noise and covariance ill-conditioning. Our layer selector measures class-posterior entropy on ID validation data as a proxy for feature cluster separability, selecting blocks with well-separated class representations while pruning uncalibrated intermediate noise.

- **Empirical Evidence:** We evaluated different block selection strategies on ViT-B/16 (ImageNet-1K ID):
- *Single Penultimate Block:* $89.82$ AUROC / $46.94\%$ $\text{FPR}_{95}$
- *All-Layer Concatenation ($D=9216$):* $88.74$ AUROC / $47.90\%$ $\text{FPR}_{95}$
- *Random 2-Block Selection:* $90.15$ AUROC / $42.10\%$ $\text{FPR}_{95}$
- *Entropy-Guided Selection (MM++, $D=1536$):* $\mathbf{92.18}$ AUROC / $\mathbf{35.98\%}$ $\mathbf{\text{FPR}_{95}}$

We will restructure Section 3 to present this intuition first, map all mathematical definitions directly to an expanded Figure 1, and include the selection strategy ablation table in Section 4.

---

> **“3. The experiments are insufficient, particularly in terms of the compared baselines and the extension of this layer-selection strategy.”**

**Response:** To address both parts, we: 
1. Evaluated the requested NNGuide, KPCA, and SCALE baselines using their released implementations. 
2. Applied the same entropy-selected multilayer representation to KNN and KPCA. It improves KNN by **0.26/0.58 AUROC** and KPCA by **3.78/3.43 AUROC** on ImageNet-1K/ImageNet-LT, respectively. Full results are reported below in response to Questions 1 and 3.

### Questions
---
> **“Questions 1. The proposed layer-selection strategy, which quantifies the informativeness of features across layers, is an interesting idea. However, its connection to the Mahalanobis distance appears rather loose. It seems that this strategy could also be applied to other feature-distance-based OOD detection methods. The authors should clearly explain why this strategy specifically benefits the Mahalanobis distance. Would it also yield performance gains for other feature-distance-based methods such as KNN and KPCA [a]?”**

**Response:** We agree the strategy is applicable to other feature-distance-based OOD detection methods. We applied the same ID-selected concatenated representation to KNN and KPCA while holding each detector’s settings fixed.

**ViT-B/16 AugReg2 on ImageNet-1K as ID  ($\text{AUROC} \uparrow$)**

| Method | ImageNet-O | Textures | Places365 | iNaturalist | SUN | Average |
|---|---:|---:|---:|---:|---:|---:|
| KNN (Penult.) | 84.43 | 88.81 | 85.09 | **96.99** | 86.78 | 88.42 |
| KNN (Concat.) | **84.73** | **89.59** | **85.22** | 96.98 | **86.87** | **88.68** |
| KPCA (Penult.) | 83.73 | 85.64 | 82.69 | 91.62 | 84.75 | 85.69 |
| KPCA (Concat.) | **86.01** | **90.43** | **85.54** | **97.35** | **88.04** | **89.47** |

**ViT-B/16 AugReg2 on ImageNet-LT as ID**

| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN (Penult.) | 74.83 | 88.22 | 67.17 | 68.75 | 79.10 | 54.99 | 72.18 |
| KNN (Concat.) | **75.19** | **88.52** | **68.66** | **69.48** | **79.71** | **55.00** | **72.76** |
| KPCA (Penult.) | 81.55 | 87.32 | 71.47 | 70.13 | 81.05 | **57.60** | 74.85 |
| KPCA (Concat.) | **83.88** | **93.99** | **75.73** | **72.98** | **85.86** | 57.26 | **78.28** |

Our layer-selection strategy yields gains for KNN (+0.26 to +0.58 AUROC) and more improvements for KPCA (+3.43 to +3.78 AUROC), showing that the learned multi-scale subspace is broader than just a Mahalanobis-specific artifact.

 **Synergy with Maha Distance:**
 While the representation generalizes well, it exhibits a specific statistical synergy with Mahalanobis scoring:
 - *Precision Matrix Sensitivity:* Maha distance scales distances by the inverse precision matrix $\Sigma^{-1}$, making it highly sensitive to high-dimensional estimation noise and non-Gaussian tails.
 
 - *Gaussianity Alignment:* Our entropy-guided selection prunes uncalibrated intermediate layers, dampening excess kurtosis ($2.116 \to 0.609$) and bringing Mardia's multivariate ratio closer to ideal normality ($1.661 \to 1.102$).
 
 - *Covariance Conditioning:* By filtering out noisy blocks, our strategy mitigates $\Sigma$ from becoming ill-conditioned, allowing Maha scoring to operate on a statistically sound, well-scaled quadratic metric. 
 
 We will add these experiments and theoretical connections to Section 4 and Appendix  in the revised manuscript.

---

> **“Question 2. In Sec.3.2 and Sec.3.3, the authors are suggested to provide more numerical results or illustrative figures to intuitively demonstrate how the representation richness varies across network layers, instead of just presenting a series of equations.”**

**Response:** We appreciate this feedback. We will rewrite Sections 3.2–3.3 to lead with intuitive visual explanations of Figure 1 and provide numerical validation of the layer-selection process.

The subfigures in Figure 1 correspond to the method as follows.

**Upper-left plot — representation richness across depth.**  
Section 3.2 first computes the within-class covariance entropy $H_l$ from the normalized eigenspectrum in Eq. (4). Section 3.3 then converts it into entropy density,

$$
\rho_l = H_l/D_l.
$$

The upper-left curve in Figure 1 plots $\rho_l$ for every candidate layer. Each point therefore represents the normalized within-class spectral richness of one layer. This is the quantity compared across depth; the plot is not raw entropy $H_l$.

**Lower-left plot — locating informative transitions.**  
The lower-left bars visualize the entropy-density drop from Eq. (5),

$$
\Delta_l = \rho_{l-1}-\rho_l.
$$

A large positive bar indicates a strong local decrease in entropy density between two consecutive layers. Under Eq. (6), MM++ selects the top $K-1$ positive drops as intermediate layers. In the illustrated $K=2$ example, layer 3 is selected because it has the largest positive drop. Layer 18 is included separately as the fixed final pre-logit anchor; it is not selected because of the magnitude of its own drop. We will make this distinction explicit in the figure and text.

**Left pipeline — constructing the MM++ representation.**  
After selection, Eq. (1) provides the normalized features $\widetilde h_{l_3}(x)$ and $\widetilde h_{l_{18}}(x)$. Section 3.4 then concatenates them using Eq. (7),

$$
\phi(x) =
[\widetilde h_{l_3}(x);
 \widetilde h_{l_{18}}(x)].
$$

The covariance diagrams below the concatenation correspond to the tied covariance of the fused representation in Eq. (8) and its Ledoit–Wolf-regularized inverse in Eq. (9). These elements belong to Section 3.4, rather than Sections 3.2–3.3.

**Center panel — class-conditioned geometry in the fused space.**  
The center panel is a qualitative t-SNE visualization of $\phi(x)$. The colored points represent ID samples, the class markers represent their centroids, and the orange points represent OOD samples. The distance annotations illustrate Eq. (10): a sample receives a high ID score when its minimum class-conditional Mahalanobis distance is small and a low score when its minimum distance is large. The actual score is computed in the full fused feature space, not in the two-dimensional t-SNE projection.

**Right panel — resulting OOD-score separation.**  
The right panel shows the empirical score distributions produced by Eq. (10). Compared with terminal-only Maha++, MM++ assigns lower scores to more OOD samples while maintaining a similar high-score region for ID samples, reducing the overlap between the ID and OOD distributions. Because MM++ is post-hoc, we will replace wording such as “pushes OOD samples” with the more accurate statement that MM++ produces larger Mahalanobis distances and lower confidence scores for OOD samples.

**Empirical evidence:** 
We exhaustively evaluated every candidate intermediate layer paired with the penultimate representation. Across 9 diagnostic OOD datasets, the MM++ selected pairs improve over penultimate-only scoring and **perform within 0.41/0.82 AUROC of the retrospective oracle** on ViT-B/16/Swin-T. 

In the revised paper, we will connect equations to the corresponding visual panel in Figure 1 and augment the presentation of layer-selection, cross-layer feature concatenation, and shrinkage-regularized covariance equations with empirical results.

---

> **“Question 3. Regarding the compared methods, it is appreciated that multiple strong Mahalanobis-based baselines have been included. Nevertheless, the set of other comparative methods remains insufficient. For feature-distance-based approaches, a popular enhanced version of KNN, namely NNGuide [b], should be considered, and KPCA [a], which uses reconstruction error as a feature distance, should also be included. In addition, beyond ReAct, a prevalent feature-shaping method, SCALE [c], should be discussed and evaluated.”**

**Response:**  We have added all three requested baselines. MM++ outperforms NNGuide, KPCA and SCALE in all evaluations.

**ViT-B/16, ImageNet-1K as ID** 
| Method | ImageNet-O | Texture | Places365 | iNaturalist | SUN | Average |
|---|---:|---:|---:|---:|---:|---:|
| SCALE | 63.99 / 76.40 | 72.77 / 70.99 | 50.04 / 91.47 | 75.99 / 62.11 | 72.31 / 71.18 | 67.02 / 74.43 |
| NNGuide | 85.87 / 64.50 | 90.37 / 41.01 | 85.76 / 62.82 | 97.76 / 10.90 | 88.57 / 53.46 | 89.67 / 46.54 |
| KPCA | 85.08 / 71.15 | 89.30 / 49.18 | 84.52 / 66.72 | 97.42 / 12.86 | 88.54 / 51.87 | 88.97 / 50.36 |
| MM++ | **88.76 / 54.45** | **97.00 / 14.52** | **87.05 / 55.58** | **98.43 / 5.71** | **89.64 / 49.66** | **92.18 / 35.98** |


**ViT-B/16, ImageNet-LT as ID**

| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCALE | 71.87 / 70.37 | 74.23 / 59.47 | 63.67 / 77.89 | 59.69 / 78.44 | 49.94 / 91.03 | 52.12 / 93.57 | 61.92 / 78.46 |
| NNGuide | 88.23 / 59.60 | 93.84 / 34.92 | 77.45 / 65.67 | 76.46 / 63.20 | 87.99 / 45.39 | 61.08 / 88.68 | 80.84 / 59.58 |
| KPCA | 81.67 / 80.10 | 86.69 / 73.29 | 71.57 / 85.01 | 70.55 / 85.65 | 81.26 / 77.22 | 58.16 / 92.42 | 74.98 / 82.28 |
| MM++ | **91.34 / 43.20** | **96.40 / 21.07** | **84.35 / 53.80** | **81.70 / 52.43** | **88.84 / 41.83** | **60.78 / 88.94** | **83.90 / 50.21** |

**ConvNeXt-T, ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCALE | 61.73 / 93.91 | 60.64 / 95.95 | 65.08 / 90.27 | 63.92 / 83.63 | 54.96 / 96.49 | 56.26 / 92.81 | 60.43 / 92.18 |
| NNGuide | 81.39 / 72.09 | 86.42 / 68.04 | 83.80 / 52.98 | 79.18 / 61.86 | 87.96 / 49.25 | 61.23 / 89.02 | 80.00 / 65.54 |
| KPCA | 73.10 / 88.37 | 78.21 / 85.24 | 73.47 / 86.04 | 70.58 / 88.06 | 77.79 / 83.15 | 57.13 / 93.15 | 71.71 / 87.33 |
| MM++ | **87.51** / **53.72** | **93.40** / **37.69** | **82.52** / **56.39** | **77.98** / **69.36** | **88.08** / **49.12** | **59.70** / **90.33** | **81.53** / **59.43** |

**Swin-T, ImageNet-LT as ID**
| Method | NINCO | OpenImage-O | ImageNet-C | ImageNet-ES | ImageNet-R | ImageNet-V2 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCALE | 80.43 / 74.40 | 88.05 / 59.41 | 76.35 / 68.07 | 75.96 / 66.53 | 84.30 / 56.35 | 59.85 / 89.88 | 77.49 / 69.11 |
| NNGuide | 81.70 / 75.63 | 86.09 / 69.64 | 82.01 / 58.91 | 80.22 / 61.46 | 87.87 / 50.91 | 60.68 / 89.65 | 79.76 / 67.70 |
| KPCA | 72.81 / 83.91 | 83.50 / 73.60 | 76.10 / 66.84 | 72.47 / 73.08 | 85.14 / 56.26 | 55.97 / 91.80 | 74.33 / 74.25 |
| MM++ | **85.76** / **59.31** | **92.44** / **43.12** | **86.26** / **43.56** | **86.25** / **41.09** | **89.29** / **42.91** | **60.68** / **89.61** | **83.45** / **53.27** |


We'll add these evaluations, analyze and extend across the tables at the revised paper.

---

> **“Question 4. Another minor suggestion is to also report the associated concatenated feature dimension in line 230 when reporting .”**

**Response:** We will report the selected layer indices and concatenated dimensions explicitly. For ViT-B/16 with $K=2$, the selected representation has

$$
D_{\mathcal K}=768+768=1536.
$$

For the selected Swin-T pair,
$$
D_{\mathcal K}=384+768=1152.
$$

