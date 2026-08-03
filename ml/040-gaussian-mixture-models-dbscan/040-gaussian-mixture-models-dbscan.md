# ml 40

# **Gaussian Mixture Models and DBSCAN**

K-means: hard assignment, assumes spherical/equal-size clusters, forces every point into a cluster. Two gaps: soft clustering with flexible cluster shape (GMM) and density-based clustering that can find arbitrary shapes and flag outliers (DBSCAN).

---

# **1. Gaussian Mixture Model (GMM)**

## **Generative Story**

z ~ Categorical(π)  
x | z=k ~ N(μ_k, Σ_k)

p(x) = Σ_k π_k · N(x; μ_k, Σ_k)

Each component has its own covariance → elliptical, differently shaped/oriented clusters (vs k-means' spherical assumption).

## **Why No Closed-Form MLE**

log p(x) has a log-of-sum (over components) → couples all parameters together, no closed form. Solved via **EM**.

## **EM Algorithm**

**E-step** — compute responsibilities (posterior prob. of component k given x_i):

γ_ik = π_k N(x_i; μ_k, Σ_k) / Σ_j π_j N(x_i; μ_j, Σ_j)

**M-step** — closed-form updates using responsibility-weighted stats:

N_k = Σ_i γ_ik  
μ_k = (Σ_i γ_ik x_i) / N_k  
Σ_k = (Σ_i γ_ik (x_i-μ_k)(x_i-μ_k)ᵀ) / N_k  
π_k = N_k / n

* E-step makes the likelihood bound tight; M-step maximizes it → log-likelihood never decreases across iterations
* Converges to a local optimum only (non-convex) → use multiple random restarts

## **Relationship to K-Means**

K-means = EM in the limit of shared, isotropic covariance Σ_k = εI, ε→0. Responsibilities become hard 0/1. "Hard EM."

## **Choosing K**

Use BIC / AIC (penalized log-likelihood), not the k-means elbow-on-inertia heuristic:

BIC = -2·log-likelihood + p·log(n)

---

# **2. DBSCAN (Density-Based Clustering)**

## **Core Definitions**

Given radius ε and minPts:

* **Core point** — ≥ minPts points within ε (incl. itself)
* **Border point** — < minPts neighbors, but within ε of a core point
* **Noise point** — neither core nor border

**Density-reachable**: chain of core points connecting two points.  
**Cluster** = maximal set of mutually density-connected points.

This chain-based definition (not distance-to-centroid) is what lets DBSCAN find arbitrarily shaped clusters.

## **Algorithm**

1. For each unvisited point: compute ε-neighborhood
2. If < minPts neighbors → mark noise (may later become a border point)
3. If ≥ minPts → start new cluster, expand outward through all reachable core points
4. Continue until all points visited

## **Choosing ε and minPts**

* minPts ≈ d+1 to 2d (data dimension d)
* ε via **k-distance plot**: sort each point's distance to its k-th nearest neighbor, plot, look for the "elbow"

---

# **Comparison**

| Aspect | K-Means | GMM | DBSCAN |
| ----- | ----- | ----- | ----- |
| Shape | spherical | elliptical | arbitrary |
| Assignment | hard | soft | hard + noise label |
| Needs K upfront | yes | yes (or BIC sweep) | no |
| Outlier handling | none | none (low prob only) | explicit noise class |

---

# **Key Pitfalls**

**GMM:**
* Singular covariance when a component collapses onto few points → regularize with +εI
* Local optima → multiple restarts
* Label switching — component indices are arbitrary across runs
* Compute in log-space to avoid underflow

**DBSCAN:**
* Single global (ε, minPts) fails when true clusters have very different densities → consider HDBSCAN/OPTICS
* High-dimensional distance concentration makes ε hard to choose, breaks spatial-index speedup
* Must standardize features first (same as k-means/GMM) — unscaled features dominate the distance metric

---

# **Interview Summary**

GMM and DBSCAN both fix limitations of k-means. **GMM** models clusters as a mixture of Gaussians with a full generative story, fit via the **EM algorithm**: an E-step computing soft "responsibilities" (posterior cluster probabilities) and an M-step updating means/covariances/weights in closed form using those responsibilities as weights — this alternation provably never decreases the log-likelihood, though it can converge to a local optimum. K-means is exactly the hard-assignment limit of GMM's EM under shared, isotropic covariance. **DBSCAN** instead clusters purely by density connectivity — a chain of "core points" with enough neighbors within radius ε — which lets it discover arbitrarily shaped clusters and explicitly label sparse points as noise, something neither k-means nor GMM can do, at the cost of needing good ε/minPts choices and struggling with clusters of very different densities.
