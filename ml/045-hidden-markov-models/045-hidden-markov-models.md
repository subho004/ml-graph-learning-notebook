# ml 45

# **Hidden Markov Models (HMMs)**

ARIMA = continuous linear state. LSTM = continuous nonlinear state (learned). HMM = discrete, small, interpretable state evolving via a Markov chain, each state emitting an observation. Used for speech, POS tagging, gene sequences, regime-switching.

---

# **Setup**

* N hidden states, hidden sequence z_1..z_T
* Observed sequence x_1..x_T
* π_i = p(z_1=i)  — initial distribution
* A_ij = p(z_t=j | z_{t-1}=i)  — transition matrix (Markov property)
* b_i(x) = p(x_t=x | z_t=i)  — emission distribution

Joint: p(z,x) = π_z1 b_z1(x1) · Π A_{z(t-1),zt} b_zt(xt)

**HMM vs GMM**: GMM = one latent per data point, independent. HMM = one latent per time step, chained via A. HMM = "mixture model + temporal dependence in the latent."

## **Three Fundamental Problems (Rabiner)**

1. Evaluation: p(x|λ) → **Forward algorithm**
2. Decoding: most likely z_1:T → **Viterbi algorithm**
3. Learning: estimate λ=(π,A,B) → **Baum-Welch** (EM)

---

# **1. Forward Algorithm**

Naive sum over all N^T paths is infeasible. Dynamic programming instead:

α_t(i) = p(x_1:t, z_t=i)

α_1(i) = π_i b_i(x_1)  
α_t(j) = [Σ_i α_{t-1}(i) A_ij] b_j(x_t)

p(x_1:T) = Σ_i α_T(i)

O(N²T) instead of O(N^T) — reuse partial sums instead of recomputing full paths.

## **Backward Algorithm** (needed for learning)

β_T(i) = 1  
β_t(i) = Σ_j A_ij b_j(x_{t+1}) β_{t+1}(j)

α_t(i)β_t(i) = p(x_1:T, z_t=i) — combines info from both directions.

---

# **2. Viterbi Algorithm — Decoding**

Same recursion as forward but **max instead of sum**:

δ_t(j) = [max_i δ_{t-1}(i) A_ij] b_j(x_t)  
ψ_t(j) = argmax_i δ_{t-1}(i) A_ij   (backpointer)

Backtrack from argmax_i δ_T(i) using stored ψ to recover the full optimal path.

Sum→max substitution is a general DP pattern (also turns sum-product BP into max-product BP).

---

# **3. Baum-Welch — Learning (EM for HMMs)**

Same obstacle as GMM: hidden states unobserved → can't do direct MLE → EM.

## **E-step**

γ_t(i) = p(z_t=i | x_1:T) = α_t(i)β_t(i) / Σ_j α_t(j)β_t(j)     [state occupancy, like GMM's responsibility]

ξ_t(i,j) = p(z_t=i, z_{t+1}=j | x_1:T)   [transition posterior]

Both computed exactly from one forward + one backward pass.

## **M-step** (closed form, mirrors GMM exactly)

π̂_i = γ_1(i)  
Â_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)  
Discrete emission: b̂_i(k) = Σ_{t:xt=k} γ_t(i) / Σ_t γ_t(i)  
Gaussian emission: μ̂_i, σ̂²_i — identical formula to GMM's M-step, just weighted by γ_t(i) instead of γ_ik

* Monotonic log-likelihood improvement (same proof structure as GMM), converges to local optimum only

---

# **Complexity**

| Algorithm | Time | Space |
| ----- | ----- | ----- |
| Forward/Backward | O(N²T) | O(NT) |
| Viterbi | O(N²T) | O(NT) |
| Baum-Welch (per iter) | O(N²T) | O(NT) |

---

# **Key Pitfalls**

* Underflow: α_t(i) shrinks geometrically with T → work in log-space (log-sum-exp) or rescale at each step
* Local optima in Baum-Welch → multiple random restarts, same as GMM
* Choosing N (num hidden states) → domain knowledge or AIC/BIC sweep
* Label switching — state indices arbitrary across different fits
* Zero probabilities from M-step (unseen state-symbol combos) → permanently impossible forever after → Laplace smoothing
* Discrete (CategoricalHMM) vs continuous (GaussianHMM) emissions — must match observation type
* Don't confuse forward's sum (total prob over ALL paths) with Viterbi's max (single best path probability)

---

# **Connections**

* GMM — HMM = GMM + Markov chain on the latent; Baum-Welch M-step ≈ GMM M-step reweighted by time
* ARIMA / Kalman filters — same "hidden state evolves, observations depend on it" template, but continuous linear-Gaussian state instead of discrete
* RNN/LSTM — discrete exact-inference state (HMM) vs continuous learned nonlinear state (RNN); exactness/interpretability vs expressive power trade-off
* Bellman equations/DP (Q-learning) — forward/Viterbi are the same optimal-substructure DP pattern
* CRFs — discriminative counterpart, models p(z|x) directly, no generative/emission model, often better for structured prediction

---

# **Interview Summary**

A Hidden Markov Model assumes a small number of discrete hidden states evolving via a Markov chain, each emitting an observation from its own distribution. The **forward algorithm** computes the total sequence probability in O(N²T) via dynamic programming, reusing partial path sums instead of enumerating all N^T paths. **Viterbi** finds the single most likely hidden state sequence using the identical recursion with max substituted for sum, plus backpointers for backtracking. **Baum-Welch** learns the parameters (π, A, B) via EM: an E-step computing exact state-occupancy and transition posteriors from one forward and one backward pass, and a closed-form M-step that — for Gaussian emissions — is essentially identical to GMM's M-step, just reweighted across time instead of across independent data points. HMMs trade the representational flexibility of RNNs/LSTMs for exact, polynomial-time inference and strong interpretability.
