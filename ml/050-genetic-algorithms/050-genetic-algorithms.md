# ml 50

# **Genetic Algorithms and Evolutionary Strategies**

Every other optimization method here needs a differentiable/well-behaved objective. GA/ES treat the objective as a BLACK BOX — no gradient, just fitness evaluation. Applicable to discrete/non-differentiable/noisy/combinatorial spaces where gradient methods can't be applied directly.

---

# **1. Core Loop**

Population of chromosomes → evaluate fitness → **selection** (favor high fitness) → **crossover** (combine parents) → **mutation** (random perturbation) → repeat over generations.

---

# **2. Selection**

## **Fitness-Proportionate (Roulette)**

p(i) = f_i / Σf_j

Pitfall: one super-individual dominates selection → **premature convergence** (population collapses before exploring).

## **Tournament Selection**

Sample k individuals randomly, pick the best as parent. k directly controls selection pressure: k=1 = random search, large k = near-always-pick-best (fast convergence, high premature-convergence risk). Explicit tunable knob, unlike roulette.

---

# **3. Crossover / Mutation**

**Single-point crossover**: cut both parents at same random position, swap tails.
**Uniform crossover**: each gene independently from either parent (p=0.5) — less positional bias than single-point.

**Mutation**: flip bits (binary) / add Gaussian noise (real-valued) with small probability.

Why mutation is ESSENTIAL not optional: crossover only recombines material already IN the population — if a value never appears anywhere, no crossover can create it. Mutation is the sole source of genuinely new information.

---

# **4. Schema Theorem (Classical Theory)**

Schema H = template with wildcards, e.g. 1**0*. order o(H) = fixed positions, defining length δ(H) = span between first/last fixed position.

E[m(H,t+1)] ≥ m(H,t) · [f(H)/f̄] · [1 - p_c·δ(H)/(n-1)] · (1-p_m)^o(H)

* f(H)/f̄ : exponential growth for above-average schemata under selection
* [1-p_c·δ(H)/(n-1)] : crossover disruption risk — larger span = more likely split
* (1-p_m)^o(H) : mutation disruption risk — more fixed bits = more disruption chances

**Building block hypothesis**: short, low-order, above-average schemata propagate exponentially — "implicit parallelism" across many schemata simultaneously with a modest population. (Classical argument, since debated/refined.)

---

# **5. Evolutionary Strategies (continuous params)**

(μ,λ)-ES: generate λ offspring via Gaussian mutation of shared mean, keep best μ as new parents (discard old parents entirely):

x_i = m + σε_i,  ε_i~N(0,I)
m ← mean of top-μ offspring

## **Connection to Policy-Gradient RL**

E_ε[F(m+σε)] is a smoothed objective; its gradient estimator:

∇_m E[F(m+σε)] ≈ (1/λσ) Σ F(m+σε_i)·ε_i

SAME log-derivative-trick structure as REINFORCE (RL Ch.2) — ES = policy-gradient RL where the "policy" is a Gaussian over parameters. Embarrassingly parallel (each perturbation independent, no backprop), higher variance per sample, needs more evaluations.

---

# **Complexity**

Fitness evaluation O(P·C_eval) is almost always the true bottleneck (e.g. training a neural net per fitness eval in neuroevolution) — GA overhead (selection/crossover/mutation) is O(Pn), usually negligible by comparison.

---

# **Key Pitfalls**

* Premature convergence — strong selection pressure collapses diversity too early
* No formal convergence guarantee (unlike EM/gradient descent) — pure heuristic
* **Encoding matters enormously** — determines which operators are valid; poor encoding can make an easy problem unsearchable. Single most important design decision.
* Poor choice when a gradient IS actually available — gradient descent will usually be far more sample-efficient. GA/ES earn their keep specifically in gradient-free/black-box/discrete settings.
* Heavy hyperparameter sensitivity (pop size, rates, pressure), no universal defaults

---

# **Connections**

* RL (Ch. RL-1, RL-2) — ES = gradient-free alternative to policy-gradient, same REINFORCE estimator structure
* Random search/Bayesian optimization/simulated annealing — sibling derivative-free global optimization methods
* Neuroevolution/NAS — GA/ES used to search discrete architectures or evolve weights directly, precisely because those choices are non-differentiable
* EM/gradient descent — contrast point: local + guaranteed (if only locally) vs global + guarantee-free heuristic

---

# **Interview Summary**

Genetic Algorithms optimize a black-box fitness function by maintaining a population of candidate solutions and iteratively applying selection (favoring fitter individuals — tournament selection gives an explicit, tunable selection-pressure knob that fitness-proportionate selection lacks), crossover (recombining parent chromosomes), and mutation (the sole source of genuinely new genetic material, since crossover can only recombine what's already present). The classical theoretical justification, Holland's Schema Theorem, shows that short, low-order, above-average-fitness building blocks propagate with exponentially increasing representation across generations while being only mildly disrupted by crossover and mutation. Evolutionary Strategies adapt the same idea to continuous parameters via Gaussian perturbation and selection of the best offspring, and modern large-scale ES reveals a direct connection to policy-gradient reinforcement learning — both use the same log-derivative-trick gradient estimator, making ES an embarrassingly parallel, gradient-free alternative to REINFORCE. GAs/ES are the right tool specifically when gradients are unavailable, unreliable, or the search space is discrete/combinatorial — where a gradient is actually available, ordinary gradient descent is almost always far more sample-efficient.
