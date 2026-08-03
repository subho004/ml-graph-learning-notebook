# distributed-ml 1

# **Federated Learning**

Every other chapter assumes centralized training data. Federated Learning: data stays ON the device (phone, hospital server, bank), only MODEL UPDATES are communicated, never raw data. A training paradigm/systems architecture — any existing model (CNN, Transformer, logistic regression) can be trained this way.

---

# **Motivation**

* Privacy/regulatory — healthcare/financial/personal data can't leave device/institution
* Bandwidth — millions of phones' raw data >> compact model updates
* Data ownership — cross-silo institutions benefit from joint model without exposing raw records to competitors

## **Formal Setting**

K clients, each with local data D_k (different size n_k, possibly very different distribution):

min_w F(w) = Σ (n_k/n)·F_k(w)

No single machine ever sees the full dataset.

---

# **FedAvg — Baseline Algorithm**

## **Why Not Send Gradients Every Step?**

Naive "FedSGD" = 1 round trip per gradient step → too slow given device communication >> local compute. FedAvg: let each client do MULTIPLE local epochs before communicating — trades staleness/divergence risk for far fewer communication rounds.

## **Algorithm**

1. Server broadcasts global weights w_t to a SAMPLED subset of clients (not all K — can't sync millions of phones every round)
2. Each client runs E local epochs of SGD starting from w_t → local weights w_t^k
3. Client sends ONLY the updated weights (not data) back
4. Server aggregates: w_{t+1} = Σ (n_k/Σn_k')·w_t^k     (size-weighted average)
5. Repeat

Size-weighting makes the aggregate an unbiased estimate of what one centralized gradient step over the pooled data would give (in the E=1, IID case) — generalizes ordinary minibatch SGD averaging across clients instead of within one batch.

---

# **Non-IID Challenge (harder than standard distributed SGD)**

Standard distributed SGD: data is merely SHARDED, same distribution everywhere. Federated clients: genuinely different distributions (one phone's vocabulary, one hospital's patient population).

## **Client Drift**

Running many local epochs (E>1) pulls w_t^k toward that CLIENT'S OWN systematically-biased local optimum. Averaging several drifted local models ≠ what one global gradient step would give — divergence grows with E and degree of non-IID-ness.

## **FedProx Fix**

Add proximal penalty to local objective:

min_w F_k(w) + (μ/2)||w - w_t||²

Same "penalize distance from trusted reference" pattern as ridge regression (Ch.39) / RLHF's KL-to-reference term (RL Ch.2). Larger μ → less drift, slower local progress. μ=0 → plain FedAvg.

---

# **Communication Efficiency**

Bandwidth (not compute) is the bottleneck here (opposite of typical data-center distributed training):
* Model compression — quantize updates to low precision, or send only sparse significant changes
* Structured/sketched updates — low-rank decomposition, more compact than full weight delta

---

# **Privacy — "Federated" Alone Isn't Enough**

* **Gradient leakage**: weight updates CAN leak info about local data (gradient inversion attacks can partially reconstruct training examples, esp. with small batches) — federated ≠ cryptographic privacy guarantee
* **Differential Privacy**: clip update norm + add calibrated Gaussian noise → formal, quantifiable guarantee. Orthogonal, composable on top of federated communication.
* **Secure Aggregation**: cryptographic protocol — server computes the SUM of updates without seeing any individual client's update in the clear. Protects against curious server, not against gradient-leakage in isolation.

---

# **Complexity**

| Aspect | Cost |
| ----- | ----- |
| Communication/round | O(\|S_t\|·\|w\|) — usually the bottleneck |
| Local compute | ordinary local SGD cost |
| Server aggregation | O(\|S_t\|·\|w\|) — cheap weighted average |

---

# **Key Pitfalls**

* Increasing E to save communication rounds can INCREASE divergence under non-IID data — genuine trade-off, not free
* Stragglers/dropout — real devices go offline mid-round, vary wildly in speed — production systems must handle partial/missing updates
* Assuming "federated" = privacy guarantee (it's not, without DP/secure aggregation layered on)
* No natural centralized test set in pure cross-device setting — evaluation is genuinely harder
* Fairness — aggregate can perform well on average while badly for small/unusual clients (their n_k is small → small influence on w_{t+1}) → q-FedAvg and similar exist to address this
* μ, E, LR schedule hyperparameters interact with non-IID degree unpredictably — needs empirical tuning per deployment

---

# **Connections**

* Distributed/parallel SGD — federated is the non-IID, communication-constrained, privacy-motivated cousin; aggregation math generalizes minibatch averaging
* Ridge regression (Ch.39) / RLHF KL penalty (RL Ch.2) — FedProx's proximal term is the same "stay close to reference" pattern
* Differential Privacy — orthogonal formal mechanism layered on top
* Multi-task learning — personalized federated learning reframes as shared representation + per-client personalization

---

# **Interview Summary**

Federated Learning trains a shared model across many clients (devices or institutions) whose local data never leaves them — only model weight updates are communicated. The standard algorithm, **FedAvg**, has each sampled client run several local SGD epochs before sending weights back, which the server aggregates via a dataset-size-weighted average — dramatically cutting the communication rounds needed compared to sending a gradient every single step. The central technical challenge, absent from ordinary distributed SGD, is that client data is typically **non-IID**: local training pulls each client's weights toward its own systematically different local optimum ("client drift"), so more local epochs trade communication efficiency for convergence quality. **FedProx** mitigates this with a proximal penalty keeping local updates close to the global model, the same "penalize distance from a reference" pattern seen in ridge regression and RLHF's KL-constrained objective. Critically, keeping raw data on-device is a systems-level privacy improvement, not a formal guarantee — weight updates themselves can leak information, which is why real deployments layer differential privacy (calibrated noise) and/or secure aggregation (cryptographically hiding individual updates from the server) on top of the basic federated communication pattern.
