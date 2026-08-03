# ml 48

# **Restricted Boltzmann Machines (RBM) and Deep Belief Networks (DBN)**

Pre-VAE/GAN (2006-2012) workhorse of unsupervised deep learning — notably enabled the first practical method for training deep nets at all, via layer-wise pretraining. RBM = energy-based model: define energy over (visible, hidden) pairs, derive probability via Boltzmann distribution.

---

# **1. Setup**

E(v,h) = -aᵀv - bᵀh - vᵀWh    (a,b = biases, W = visible-hidden weights)

p(v,h) = (1/Z)·exp(-E(v,h))     Z = partition function (sum over ALL v,h — intractable)

Low energy → high probability. Training makes energy low at real data, high elsewhere.

## **"Restricted" = No Intra-Layer Connections**

Only visible-hidden edges (no v-v or h-h) → conditionals **factorize completely**:

p(h_j=1|v) = σ(b_j + Σ v_i W_ij)
p(v_i=1|h) = σ(a_i + Σ W_ij h_j)

Each direction = one matrix-multiply + sigmoid, exactly one feedforward layer. This factorization is the entire computational payoff of "restricted."

---

# **2. Training: Contrastive Divergence**

## **ML Gradient**

∂log p(v)/∂W_ij = E_p(h|v)[v_i h_j]  −  E_p(v,h)[v_i h_j]
                    (positive phase,           (negative phase,
                     data-driven, tractable)     model-driven, intractable)

"Increase probability of what data shows, decrease probability of what model currently believes" — generic energy-based-model gradient form.

## **Why Negative Phase Is Hard**

Requires sampling the full model distribution p(v,h) → needs Gibbs sampling (alternate h~p(h|v), v~p(v|h)) run to convergence — expensive.

## **CD-k (Contrastive Divergence) — Practical Fix**

Hinton (2002): start Gibbs chain AT the real data v_data, run only k steps (k=1 common: "CD-1"):

v⁽⁰⁾ → sample h⁽⁰⁾ → sample v⁽¹⁾ → sample h⁽¹⁾  (stop, CD-1)

ΔW ∝ E[v⁽⁰⁾h⁽⁰⁾] − E[v⁽¹⁾h⁽¹⁾]

Biased estimator of true ML gradient, but works remarkably well — real data is already close to the model's locally plausible region, so even 1 step gives a useful contrast.

---

# **3. Deep Belief Networks — Stacking RBMs**

## **Motivation**

Pre-2012 (before batchnorm/resnets/good init), training deep nets via backprop from random init failed badly (vanishing gradients, like vanilla RNN Ch.038). Fix: greedy layer-wise pretraining.

## **Construction**

1. Train RBM_1 on raw data v → hidden features h⁽¹⁾
2. Freeze RBM_1; treat p(h⁽¹⁾|v) as "data" for RBM_2 → h⁽²⁾
3. Repeat, stack as many layers as desired (each trained greedily, one at a time)
4. Stack becomes initialization for a deep feedforward net → fine-tune end-to-end via backprop (supervised)

Why it helps: each RBM solves an easy shallow problem, sidesteps vanishing-gradient issue of training all layers jointly from scratch — gives backprop a much better starting point than random init.

---

# **Complexity**

| Op | Time |
| ----- | ----- |
| One Gibbs step | O(n_v·n_h) |
| CD-k training step | O(k·n_v·n_h) |

CD-1 ≈ cost of one feedforward layer's forward+backward — why it was practical on 2006-era hardware.

---

# **Key Pitfalls**

* CD-k is a BIASED gradient estimator, not true ML — Persistent CD (maintain chain across minibatches, don't restart from data) reduces this
* Binary-unit assumption — continuous data needs Gaussian-Bernoulli RBM variant
* Largely superseded — modern init/batchnorm/resnets/abundant labeled data eliminated the practical need for RBM pretraining; now mostly historical/pedagogical (energy-based modeling itself has seen research resurgence elsewhere)
* Can't directly compute log p(v) (Z intractable) → track reconstruction error instead, or use expensive Annealed Importance Sampling for approximate likelihood

---

# **Connections**

* VAE/GAN (Ch. 1-2) — modern successors; VAE replaces intractable Z with tractable ELBO+reparameterization, GAN sidesteps likelihood entirely via adversarial training
* Autoencoders — RBM's two conditionals LOOK like encoder/decoder but RBM is a genuine symmetric probabilistic model trained via contrastive divergence, not deterministic reconstruction-loss minimization
* MCMC — Gibbs sampling is a simple instance of the broader MCMC toolkit
* Diffusion models (Ch. 3) — same "energy/score, learn it, sample via Markov chain" lineage

---

# **Interview Summary**

A Restricted Boltzmann Machine is an energy-based model over visible and hidden binary units, with probability given by the Boltzmann distribution over an energy function; restricting connections to only visible-hidden edges (no within-layer connections) makes both conditional distributions p(h|v) and p(v|h) factorize into independent, tractable logistic units — exactly one feedforward layer each direction. Training maximizes likelihood via a gradient that contrasts a data-driven "positive phase" against a model-driven "negative phase," the latter requiring intractable sampling from the full model — approximated in practice via **Contrastive Divergence**, which starts a Gibbs chain at real data and runs just one or a few steps rather than to convergence, giving a biased but highly effective gradient estimate. **Deep Belief Networks** stack RBMs and train them one layer at a time (greedy layer-wise pretraining), historically providing the first practical way to initialize deep networks well enough for backpropagation to succeed, before modern initialization schemes and architectures (batch norm, ResNets) made this unnecessary — RBMs/DBNs are now mostly of historical and conceptual importance as the bridge that made deep learning trainable.
