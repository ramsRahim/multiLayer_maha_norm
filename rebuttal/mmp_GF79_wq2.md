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