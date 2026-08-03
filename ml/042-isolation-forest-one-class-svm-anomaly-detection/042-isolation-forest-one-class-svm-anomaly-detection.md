# ml 42

# **Anomaly Detection: Isolation Forest and One-Class SVM**

GMM/DBSCAN flag outliers only as a side effect of modeling normal data. These two methods are purpose-built for anomaly detection directly.

---

# **1. Isolation Forest**

## **Core Idea**

Anomalies are few and different → easy to isolate via random partitioning. Randomly split feature space repeatedly (random feature, random threshold) → anomalies get separated in very few splits; normal points buried in dense clusters need many splits.

No distance, density, or distribution fitting required — purely structural/geometric.

## **Building an iTree**

1. Pick random feature, random split value between min/max
2. Partition, recurse on each side
3. Stop at max depth or 1 point left

Isolation Forest = ensemble of t such trees, each from an independent random subsample.

## **Path Length → Anomaly Score**

h(x) = path length (edges) to isolate x in a tree. Anomalies → short paths. Normal points → long paths.

An iTree is structurally a random BST. Average path length in a random BST of n nodes:

c(n) = 2H(n-1) - 2(n-1)/n,  H(i) = Σ 1/k ≈ ln(i) + 0.5772

Normalized anomaly score:

s(x,n) = 2^(-E[h(x)]/c(n))

* E[h(x)] → 0: s → 1 (strong anomaly)
* E[h(x)] → c(n): s → 0.5 (normal/typical)
* E[h(x)] >> c(n): s → 0 (deep inside dense cluster, clearly normal)

## **Why Subsample Instead of Full Dataset**

Default ψ=256 regardless of n. Full dataset → dense regions need very deep trees, dilutes signal. Subsampling keeps trees shallow, contrast sharp, AND makes training/scoring cost nearly independent of n — big scalability win.

---

# **2. One-Class SVM**

## **Setup**

Adapts standard SVM margin-maximization to unsupervised setting: no negative class, treat the **origin** as the implicit "outside" class. Find max-margin hyperplane separating data region from origin in kernel feature space.

## **ν-SVM Objective**

min ½||w||² - ρ + 1/(νn) Σξ_i   s.t.  w·φ(x_i) ≥ ρ - ξ_i, ξ_i ≥ 0

f(x) = sign(w·φ(x) - ρ)

Same soft-margin/slack structure as existing SVM chapters, just no second class.

## **Dual + Kernel Trick**

min ½ Σ α_iα_j K(x_i,x_j)   s.t. 0 ≤ α_i ≤ 1/(νn), Σα_i = 1

f(x) = sign(Σ α_i K(x_i,x) - ρ)

RBF kernel standard default → lets boundary wrap around arbitrary normal-region shapes (same benefit as Kernel SVM chapter).

## **Interpreting ν**

* ν ≥ fraction of training points that are outliers
* ν ≤ fraction of training points that become support vectors

Single interpretable knob for tightness of the boundary (replaces opaque C parameter).

## **SVDD (equivalent alternative)**

Directly fits smallest enclosing hypersphere around the data in feature space. Provably equivalent to One-Class SVM for RBF kernel (where K(x,x)=1 always, so all points lie on unit hypersphere).

---

# **Comparison**

| Aspect | Isolation Forest | One-Class SVM | GMM | DBSCAN |
| ----- | ----- | ----- | ----- | ----- |
| Models normal distribution? | no | yes (margin boundary) | yes (density) | no (connectivity) |
| Scales to large n | yes, near-linear | poor, O(n²)-O(n³) | moderate | moderate |
| Feature scaling sensitivity | low | high (kernel distance) | high | high |
| Output | continuous score | binary + signed distance | likelihood | binary noise label |

---

# **Key Pitfalls**

**Isolation Forest:**
* Irrelevant features dilute the path-length signal (random feature selection wastes splits)
* Axis-aligned splits limit boundary shapes — rotated/diagonal anomaly patterns harder to isolate quickly (Extended Isolation Forest fixes with random hyperplane splits)
* Categorical features need encoding like any threshold-based method

**One-Class SVM:**
* Very sensitive to ν and γ/kernel choice, no fully automatic tuning in unsupervised setting
* Poor scalability — O(n²)-O(n³) dual QP (use SGDOneClassSVM for large data)
* Requires feature scaling (RBF kernel is distance-based)
* Origin-as-outside-class assumption fits RBF kernel naturally (K(x,x)=1) but less so other kernels on unnormalized data

---

# **Connections**

* GMM/DBSCAN — density/connectivity-based alternative anomaly detection
* Random Forest — Isolation Forest reuses "ensemble of randomized trees" but repurposes path length as the signal instead of prediction averaging
* Kernel SVM — One-Class SVM's dual/KKT/kernel-trick derivation transfers almost unchanged
* Autoencoders (generative-models) — third anomaly paradigm: reconstruction error on normal-only training data

---

# **Interview Summary**

**Isolation Forest** detects anomalies by exploiting a purely structural fact: because anomalies are rare and different, random axis-aligned partitioning isolates them in far fewer splits than normal points buried in dense clusters. It scores each point via the average path length to isolation across many trees, normalized against the expected path length of a random BST search — c(n) = 2H(n-1) - 2(n-1)/n — giving a bounded score where short paths mean anomalies. Crucially it never models the normal-data distribution and trains on small fixed-size subsamples, making it fast and nearly independent of dataset size. **One-Class SVM** takes the opposite approach: it directly adapts the max-margin machinery from standard SVM to fit a boundary around the normal-data region in kernel feature space, treating the origin as the implicit outlier class, with the parameter ν giving simultaneous, interpretable bounds on both the outlier fraction and the support-vector fraction. It produces a sharper geometric boundary than Isolation Forest but scales far worse (O(n²)-O(n³) dual QP) and is much more sensitive to kernel/hyperparameter choice and feature scaling.
