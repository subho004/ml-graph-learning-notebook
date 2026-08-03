# generative-models 2

# **Generative Adversarial Networks (GANs)**

VAE trains a generator via a likelihood lower bound → structurally blurry. GAN abandons likelihood: two networks play a minimax game — generator tries to fool, discriminator tries to catch. No reconstruction target, purely adversarial signal → sharper samples, less stable training.

---

# **1. Setup**

z ~ p_z (e.g. N(0,I))  
x̂ = G_θ(z)  
D_φ(x) ∈ (0,1) = probability x is real

Minimax objective:

min_θ max_φ  E_data[log D(x)] + E_z[log(1 - D(G(z)))]

* Discriminator wants D(real)→1, D(fake)→0 (binary classification)
* Generator wants D(G(z))→1 (fool discriminator) — never sees real data directly, only gradient through D

## **Alternating Training**

1. Update discriminator φ (generator frozen)
2. Update generator θ (discriminator frozen)
3. Repeat

---

# **2. Optimal Discriminator**

For fixed G:

D*(x) = p_data(x) / (p_data(x) + p_g(x))

Bayes-optimal classifier output for real vs (mixture of real+fake).

---

# **3. Optimal Generator — Jensen-Shannon Divergence**

Substituting D* back into V gives:

V(D*, G) = 2·JS(p_data || p_g) − 2log2

* JS ≥ 0, global minimum at p_g = p_data
* At equilibrium: D*(x) = 0.5 everywhere (can't distinguish)
* Training a GAN to convergence ≈ minimizing JS divergence between real and generated distributions

---

# **4. Non-Saturating Loss**

Problem: early training, D(G(z)) ≈ 0 → original loss log(1-D(G(z))) sits in flat/saturated region → weak gradients exactly when generator needs them most.

Fix: generator instead maximizes log D(G(z)) (minimizes -log D(G(z)))

* Same fixed point/direction as minimax loss
* Much larger gradient when D(G(z)) is small (early training)
* This is the loss actually used in practice, not the literal minimax form

---

# **Training Steps**

1. Real batch + noise batch
2. Discriminator step: BCE(D(real)→1, D(fake)→0); update φ only
3. Generator step: fresh noise, BCE(D(G(z))→1) non-saturating loss; update θ only, gradient flows through frozen D
4. Generation: sample z, forward through G only — no D needed

---

# **Key Pitfalls**

* Mode collapse — generator finds a few outputs that fool D, sacrifices diversity
* Non-convergence / oscillation — simultaneous two-player game, no convergence guarantee even in simple cases
* Vanishing generator gradients if D too strong/well-trained
* BatchNorm interacts badly with adversarial training (real vs fake batch stats differ)
* Discrete data (text) — not directly differentiable through argmax; needs Gumbel-softmax or RL-style policy gradients
* No tractable likelihood → evaluation needs proxy metrics (Inception Score, FID), not a clean loss curve

---

# **Beyond Vanilla GAN**

* **WGAN** — Wasserstein distance instead of JS divergence; critic must be 1-Lipschitz (weight clipping / gradient penalty); fixes vanishing-gradient-from-strong-discriminator problem, correlates better with sample quality
* **Conditional GAN (cGAN)** — condition G(z,y) and D(x,y) on label y for controllable generation
* **DCGAN** — standard convolutional architecture template (strided convs, batchnorm, LeakyReLU in D, ReLU/Tanh in G)

---

# **Connections**

* VAE — same z→data mapping, likelihood-based vs adversarial objective
* f-divergences/IPMs — different D losses ≈ minimizing different divergences (f-GAN unifies these)
* Actor-critic RL — discriminator acts like a learned critic/reward for the generator (actor); shares instability of simultaneous two-network optimization

---

# **Interview Summary**

GANs train a generator and discriminator in a minimax game instead of optimizing an explicit likelihood. The optimal discriminator for a fixed generator is the Bayes-optimal real-vs-fake classifier, D*(x) = p_data/(p_data+p_g), and substituting this back shows that training a GAN to convergence is equivalent to minimizing the **Jensen-Shannon divergence** between the real and generated distributions, with the unique optimum at p_g = p_data. In practice the literal minimax generator loss saturates early in training (weak gradients when the generator is still bad), so implementations use the **non-saturating loss** instead, which has the same fixed point but much stronger early gradients. GANs produce sharper samples than VAEs since there's no explicit per-pixel reconstruction averaging, but training is a simultaneous two-player game with no convergence guarantee, prone to mode collapse and instability — motivating variants like WGAN that swap in a better-behaved distance.
