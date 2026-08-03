# ml 49

# **Meta-Learning and Few-Shot Learning**

All prior chapters: one task, abundant data. Meta-learning: train across MANY tasks so a model can learn a genuinely NEW task from a handful (few-shot) or zero (zero-shot) examples. Shift: from "learn input→output function" to "learn a learning algorithm/initialization/representation that adapts quickly." — "learning to learn."

---

# **1. Episodic Setup**

**N-way K-shot**: N classes, K labeled examples each (support set) → classify query set from same N classes. Test-time classes typically NEVER seen during meta-training — model generalizes its LEARNING STRATEGY, not class boundaries.

Meta-training samples episodes (tasks) from a large pool of training classes:

T_i = (support S_i, query Q_i) ~ p(T)

Meta-objective: perform well in expectation ACROSS the task distribution, not on one fixed task.

---

# **2. Metric-Based: Prototypical Networks**

Learn embedding f_θ so same-class examples cluster; classification = nearest-prototype lookup, NO gradient adaptation at test time.

## **Prototypes**

p_c = mean of f_θ(x_i) for support examples of class c

p(y=c|x) = softmax(-d(f_θ(x), p_c))    [d = squared Euclidean, typically]

Trained end-to-end via cross-entropy across many sampled episodes. Test-time rule looks like one k-means E-step — but f_θ is specifically learned so this simple rule generalizes to never-seen class groupings.

## **Why Squared Euclidean Specifically**

Makes classifier mathematically equivalent to a GMM with one component per class, shared fixed covariance (callback to Ch.040) — exp(-||x-μ_c||²) matches the shared-covariance Gaussian density exactly. Cosine similarity has no such exponential-family justification, degrades performance empirically.

---

# **3. Optimization-Based: MAML**

Learn an INITIALIZATION θ such that a few ordinary gradient steps on a new task's support set gives a good model — explicit meta-objective, not accidental pretraining byproduct.

## **Bi-Level Optimization**

θ_i' = θ - α∇_θ L_support(θ)     [inner loop: adapt to task i]

min_θ Σ_i L_query(θ_i')     [outer loop: θ_i' should do well on task i's query set]

**Second-order**: outer gradient requires differentiating THROUGH the inner gradient step → Hessian-vector product:

∇_θ L_query(θ_i') = (I - α∇²_θ L_support(θ)) · ∇_θ_i' L_query(θ_i')

Intuition: ordinary training asks "how do I improve on this data?" MAML asks "how do I set θ so that TAKING A GRADIENT STEP from it improves performance?" — optimizing sensitivity to a future update, not the parameters directly.

## **First-Order MAML / Reptile**

Ignore the Hessian-vector term (treat θ_i' as independent of θ for the outer gradient) — cheaper, biased approximation, often nearly as good empirically.

## **Algorithm**
1. Sample batch of tasks
2. Per task: inner-loop gradient step(s) on support set → θ_i'
3. Evaluate query loss at θ_i' for each task
4. Backprop total query loss THROUGH the inner step back to θ (outer/meta update)

---

# **Complexity**

| Method | Per-episode cost | Test-time adaptation |
| ----- | ----- | ----- |
| ProtoNet | O(NKd) prototypes + O(NQd) classify | NONE — pure forward pass |
| MAML (2nd order) | inner step + Hessian-vector product, ~2-3x backprop cost | requires inner-loop step(s) on new task |
| FOMAML/Reptile | inner step, no Hessian-vector product | same as MAML |

---

# **Key Pitfalls**

* Overfits to meta-training TASK distribution — doesn't generalize to substantially different task domains
* MAML's 2nd-order compute/memory cost and instability → motivates FOMAML/Reptile/metric-based alternatives
* Distance metric choice in ProtoNet matters (§ shared-covariance GMM justification) — swapping metrics loses this
* Episode N,K sampling during training should roughly match test-time N,K — mismatch degrades performance
* Few-shot learning ≠ transfer learning/fine-tuning — properly-tuned fine-tuning baselines are surprisingly strong, much reported meta-learning gains narrowed once compared fairly (Chen et al. 2019)

---

# **Connections**

* GMM (Ch.40) — ProtoNet's rule = shared-covariance GMM posterior, with a learned embedding making it work on novel classes
* K-means — prototype computation = one M-step, applied per-episode
* Transfer learning/fine-tuning — the baseline meta-learning must beat, and the closest ordinary analog
* Newton boosting (Ch.41) — different context, same idea that some optimization needs curvature (2nd derivatives), not just gradients
* In-context learning (modern LLMs) — giving examples directly in the prompt, no parameter updates at all — gradient-free few-shot adaptation, now dominant for foundation models

---

# **Interview Summary**

Meta-learning trains across a distribution of small "episodes" (few-shot classification tasks) rather than one fixed task, so the model generalizes its learning STRATEGY to genuinely novel classes at test time. **Prototypical Networks** take a metric-based approach: learn an embedding space where a class prototype (mean support embedding) plus nearest-prototype classification generalizes to unseen classes — mathematically equivalent to a shared-covariance Gaussian mixture model, requiring zero gradient steps at test time. **MAML** takes an optimization-based approach: learn an initialization such that one or a few ordinary gradient steps on a new task's support set produces a good model, which requires a genuinely second-order meta-gradient (differentiating through the inner gradient step itself, i.e., a Hessian-vector product) — approximated more cheaply by First-Order MAML/Reptile. Both must be benchmarked carefully against well-tuned transfer-learning/fine-tuning baselines, which are often surprisingly competitive.
