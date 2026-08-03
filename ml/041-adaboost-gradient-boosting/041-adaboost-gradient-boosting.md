# ml 41

# **AdaBoost and Gradient Boosting**

Bagging (Random Forest) trains independent models on resampled data, averages → reduces variance. Boosting trains models **sequentially**, each targeting the previous ensemble's mistakes → reduces bias. AdaBoost is the original; Gradient Boosting is the general framework (which XGBoost then optimizes further).

---

# **1. AdaBoost**

Binary classification, y ∈ {-1,+1}. Weak learner = decision stump (depth-1 tree). Maintain weights w_i over examples, reweight after each round.

## **Algorithm**

1. Init weights uniform: w_i = 1/n
2. For m = 1..M:
   * Fit weak learner h_m minimizing weighted error ε_m = Σ w_i · 1[h_m(x_i)≠y_i]
   * α_m = ½ log((1-ε_m)/ε_m)
   * Update: w_i ← w_i · exp(-α_m y_i h_m(x_i)), renormalize
3. H(x) = sign(Σ α_m h_m(x))

* Correct predictions → weight shrinks by e^(-α); wrong → grows by e^(+α)
* α_m → ∞ as ε_m→0 (trust good learner); α_m → 0 as ε_m→0.5 (ignore near-random learner); α_m < 0 if ε_m > 0.5 (flip predictions)

## **Why It Works — Exponential Loss**

AdaBoost = forward stagewise additive modeling minimizing exponential loss L(y,f) = exp(-yf).

Deriving the per-round objective shows the weight update formula falls out naturally as the accumulated loss from previous rounds — not an arbitrary heuristic.

## **Training Error Bound**

training error ≤ Π_m 2√(ε_m(1-ε_m))

Each factor < 1 whenever ε_m ≠ 0.5 → training error decreases **exponentially** in rounds M, as long as every weak learner beats random guessing.

---

# **2. Gradient Boosting**

Generalizes forward stagewise additive modeling to any differentiable loss (regression or classification), for arbitrary loss functions — not just exponential loss.

## **Functional Gradient Descent**

Think of predictions as a vector f = (f(x_1)...f(x_n)). Compute the negative gradient of the loss w.r.t. current predictions (pseudo-residual):

r_i = -∂L(y_i, f(x_i))/∂f(x_i) at f = f_{m-1}

Fit a regression tree h_m to these pseudo-residuals (not to y directly) → gives a generalizable direction in function space, not just values at training points.

f_m(x) = f_{m-1}(x) + η · h_m(x)     (η = learning rate/shrinkage)

## **Special Case: Squared Error → Residual Fitting**

For L = ½(y-f)²: -g_i = y_i - f_{m-1}(x_i) = the actual residual. The "fit the residual" intuition is just one instance (squared error case) of the general negative-gradient principle.

## **Newton Boosting**

Second-order Taylor expansion using gradient AND Hessian gives optimal step Δ* = -g_i/h_i per leaf. This is exactly what XGBoost formalizes and optimizes.

## **AdaBoost = Gradient Boosting with Exponential Loss**

Setting L(y,f) = exp(-yf) in the general framework reproduces AdaBoost's α_m formula exactly — AdaBoost (1997) came first via a reweighting heuristic; the gradient-boosting/statistical unification (2000-2001) came after.

---

# **Comparison**

| Aspect | AdaBoost | Gradient Boosting | XGBoost |
| ----- | ----- | ----- | ----- |
| Loss | fixed exponential | any differentiable | any + explicit regularization |
| Step | closed-form α_m | negative-gradient tree fit + line search | 2nd-order (Newton) + regularized closed-form leaf weights |
| Outlier sensitivity | high (exp loss blows up) | loss-dependent | mitigated by regularization |

---

# **Key Pitfalls**

* AdaBoost highly sensitive to label noise/outliers — exponential loss unbounded for confidently-wrong points
* ε_m exactly 0 or 1 → α_m = ±∞, must clip
* Overfitting with too many rounds — unlike bagging, boosting CAN overfit as M→∞ without small learning rate/early stopping
* Small learning rate + more rounds generally generalizes better than large rate + few rounds
* Feature scaling not required for tree-based weak learners (threshold splits are scale-invariant)
* Sequential dependency → rounds can't be parallelized across M (unlike Random Forest's independent trees)

---

# **Connections**

* XGBoost — heavily engineered descendant of Newton boosting (regularized leaf weights, split-gain, systems optimizations)
* Random Forest/Ensemble Learning — bagging (variance) vs boosting (bias) as complementary levers
* Newton's method — §Newton boosting is classical Newton's method performed in function space
* Margin theory — explains boosting's frequent resistance to overfitting even after training error hits 0

---

# **Interview Summary**

AdaBoost trains weak learners (typically decision stumps) sequentially, reweighting training examples after each round to emphasize previous mistakes, and combines them via a closed-form weight α_m = ½log((1-ε_m)/ε_m) derived from directly minimizing **exponential loss** — the training error provably decreases exponentially as long as every weak learner beats random guessing. **Gradient Boosting** generalizes this idea to any differentiable loss by performing gradient descent in function space: each new weak learner (a shallow regression tree) is fit to approximate the negative gradient (pseudo-residual) of the loss with respect to the current ensemble's predictions, and AdaBoost turns out to be exactly the special case of this framework under exponential loss. XGBoost then extends gradient boosting with a second-order (Newton) step and explicit regularization. Unlike bagging, boosting reduces bias by design but can overfit if run for too many rounds without shrinkage or early stopping, and its sequential nature limits parallelism across rounds.
