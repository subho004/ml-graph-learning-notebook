# ml 47

# **Self-Organizing Maps (SOM)**

K-means/GMM clusters have no relation to each other (cluster 3 and 7 are arbitrary labels). SOM arranges prototypes on a fixed 2D grid so nearby grid positions = similar data — clustering + topology-preserving dimensionality reduction in one.

---

# **Setup**

M neurons, each with weight vector w_j ∈ ℝ^d (prototype) and FIXED grid position g_j ∈ ℝ² (never updated). Only w_j is learned.

---

# **Training Algorithm**

For each input x:

1. **Find BMU** (Best Matching Unit): c = argmin_j ||x - w_j||  — same as k-means' nearest-centroid step
2. **Update BMU AND its grid-neighbors**:

w_j ← w_j + η(t)·h(j,c,t)·(x - w_j)   for ALL j

h(j,c,t) = exp(-||g_j - g_c||² / 2σ(t)²)  — large near c ON THE GRID, small far away (regardless of data-space distance)

3. Decay η(t), σ(t) over training — large neighborhoods early (coarse ordering), only BMU+immediate neighbors late (fine-tuning)

## **Why This Gives Topology Preservation**

K-means: only the single winner updates (winner-take-all). SOM: winner + grid-neighbors update TOGETHER. Grid-adjacent units repeatedly co-updated toward similar inputs → converge to similar weights → grid distance approximates data-space similarity. Emerges purely from neighbors being updated together — no explicit topology term in any loss function.

---

# **Complexity**

O(NMd) per epoch (N examples, M units, d dims) — same form as k-means' O(NKd), grid size M plays role of K.

---

# **Key Pitfalls**

* Grid size/shape chosen a priori, no automatic selection (like k-means' K)
* Sensitive to learning-rate/neighborhood decay schedule — too fast freezes a poor map, too slow wastes time
* No probabilistic interpretation — heuristic, not generative (unlike GMM)
* Edge/corner units have fewer neighbors, behave differently
* Largely superseded by t-SNE/UMAP for visualization and autoencoders for learned embeddings — SOM still useful when an explicit interpretable 2D grid of cluster prototypes is specifically wanted

---

# **Connections**

* K-means — BMU step is identical to k-means assignment; difference is scope of the update (winner-only vs winner's whole neighborhood)
* PCA/Autoencoders — SOM does nonlinear dim-reduction + clustering simultaneously
* Competitive learning — canonical example of the broader "units compete, winner(+neighbors) updates" neural network family

---

# **Interview Summary**

A Self-Organizing Map clusters data onto a fixed low-dimensional (typically 2D) grid of prototype units. Like k-means, each input finds its nearest prototype (the Best Matching Unit), but unlike k-means, the update rule pulls not just the BMU but its **grid-neighbors** toward the input too, weighted by a neighborhood function that decays with grid distance and shrinks over training. Because grid-adjacent units are repeatedly updated together toward similar inputs, they converge to similar weight vectors — so grid distance ends up approximating data-space similarity, giving a topology-preserving map purely as an emergent property of the update rule, with no explicit topology term in any loss function.
