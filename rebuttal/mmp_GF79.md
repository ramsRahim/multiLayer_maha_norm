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

