# ml 39

# **Ridge, Lasso, and Elastic Net Regularization**

OLS minimizes only training error. With many/correlated features or limited data, coefficients get high variance. Regularization adds a penalty on coefficient size to trade bias for variance.

---

# **Why Regularize?**

OLS:

β̂ = (XᵀX)⁻¹Xᵀy

If XᵀX is singular or near-singular (collinear features, p > n), this blows up. Small data changes → wild coefficient swings.

---

# **1. Ridge Regression (L2)**

Objective:

min ||y - Xβ||² + λ||β||²

Closed form:

β̂_ridge = (XᵀX + λI)⁻¹Xᵀy

* Adding λI shifts eigenvalues up → always invertible
* Bayesian view: MAP under Gaussian prior on β
* Shrinks all coefficients smoothly toward 0, none reach exactly 0
* Constraint region = L2 ball (sphere) → smooth boundary → ellipse contours touch at generic point (all coords nonzero)

---

# **2. Lasso Regression (L1)**

Objective:

min ||y - Xβ||² + λ||β||₁

No closed form (non-differentiable at 0) → solved via **coordinate descent**.

## **Soft-Thresholding Update**

For each coordinate j, holding others fixed:

β̂_j = sign(ρ_j) * max(|ρ_j| - λ/2, 0)

where ρ_j = correlation of feature j with partial residual.

* If |ρ_j| ≤ λ/2 → β_j set to exactly 0
* Performs implicit feature selection

## **Why L1 Gives Sparsity (Geometric View)**

Constraint region = L1 ball (diamond/polytope) with corners on the axes. Elliptical loss contours tend to first touch a **corner** → most coefficients exactly 0.

Ridge's ball is smooth (no corners) → no sparsity.

---

# **3. Elastic Net**

Objective:

min ||y - Xβ||² + λ[α||β||₁ + (1-α)/2 ||β||²]

* α = 1 → Lasso
* α = 0 → Ridge
* Fixes two Lasso weaknesses:
  * Correlated features: Lasso arbitrarily picks one; Elastic Net selects/shrinks them together
  * p > n cap: Lasso can select at most n nonzero coefficients; Elastic Net removes this cap

---

# **Regularization Path**

As λ increases:

* Ridge: coefficients shrink smoothly, never hit exactly 0
* Lasso: coefficients hit exactly 0 one by one → sparse path
* Very large λ → all coefficients 0 (intercept-only)

---

# **Practical Notes**

* Always standardize features before penalizing
* Don't penalize the intercept
* Select λ via cross-validation, never training error (training error is monotone in λ, always prefers λ=0)
* scikit-learn's `alpha` param = λ (with 1/2n loss scaling) — not directly comparable across libraries

---

# **Interview Summary**

Ridge and Lasso both add a penalty on coefficient magnitude to OLS to control variance. **Ridge (L2)** has a closed-form solution, corresponds to a Gaussian prior on coefficients, and shrinks all coefficients smoothly without ever setting any to exactly zero — because its constraint region (a sphere) is smooth. **Lasso (L1)** has no closed form and is solved via coordinate descent using a soft-thresholding update; its constraint region (a diamond/polytope) has corners on the coordinate axes, so the solution tends to land exactly on an axis — some coefficients become exactly zero, giving automatic feature selection. **Elastic Net** blends both penalties to handle groups of correlated features and lift Lasso's selection cap when there are more features than samples.
