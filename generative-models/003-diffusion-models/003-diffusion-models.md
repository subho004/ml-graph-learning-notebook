# generative-models 3

# **Diffusion Models**

VAE: fast sampling, blurry. GAN: sharp, unstable adversarial training. Diffusion: fixed non-learned noising process + learned reverse denoising, trained via stable ordinary regression — no adversarial game — at the cost of slow multi-step sampling.

---

# **1. Forward (Noising) Process**

q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I)

Fixed schedule β_1..β_T, NO learned parameters (unlike VAE's learned encoder).

## **Closed-Form Jump to Any Timestep**

α_t = 1-β_t, ᾱ_t = Π α_s. Via reparameterization + repeated Gaussian-sum collapsing:

q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε,  ε~N(0,I)

**Key trick**: sample x_t directly from x_0 for ANY t in one step — no need to simulate the chain. Essential for efficient training. As t→T, ᾱ_T→0 → x_T ≈ N(0,I) pure noise.

---

# **2. Reverse (Denoising) Process**

True q(x_{t-1}|x_t) is intractable (needs integrating over q(x_0), same issue as VAE's true posterior). Approximate with learned Gaussian:

p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))

Justified: for small β_t (many steps), true reverse process is provably ≈ Gaussian.

## **Tractable posterior GIVEN x_0**

q(x_{t-1}|x_t,x_0) IS tractable (closed-form, linear-Gaussian chain):

μ̃_t(x_t,x_0) = [√α_t(1-ᾱ_{t-1})/(1-ᾱ_t)] x_t + [√ᾱ_{t-1}β_t/(1-ᾱ_t)] x_0
β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t

This is the exact target used in training (compare learned p_θ against this, not the intractable true q).

---

# **3. Training Objective — Hierarchical VAE**

Diffusion = hierarchical VAE: T latents x_1..x_T, fixed non-learned "encoder" (forward process), learned "decoder" (reverse process). Same ELBO derivation as VAE chapter, generalized to T steps:

ELBO = reconstruction term − Σ_t KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)) − prior-match term

Every KL term is closed-form Gaussian-vs-Gaussian (same as VAE's closed-form KL).

## **Noise-Prediction Reparameterization**

Since x_0 can be written in terms of x_t and the noise ε that generated it, substitute into μ̃_t → target mean is fully determined by x_t and ε. So instead of predicting the mean directly, train network to predict the **noise** ε_θ(x_t,t):

μ_θ(x_t,t) = (1/√α_t)(x_t − β_t/√(1-ᾱ_t) · ε_θ(x_t,t))

## **Simplified Objective (L_simple, DDPM)**

Dropping the ELBO's exact per-step weighting → plain unweighted MSE:

L_simple = E[t,x0,ε] [ ||ε − ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||² ]

Just ordinary regression: network sees noised input + timestep, predicts exactly which noise was added. No adversarial instability, no explicit reconstruction/KL balancing (absorbed into the fixed schedule β_t).

---

# **4. Training / Sampling**

## **Training**
1. Sample x_0, random t, random ε
2. Form x_t directly via closed-form jump (no sequential simulation)
3. MSE(ε, ε_θ(x_t,t)) → gradient step

Cost per step is **independent of T** (thanks to closed-form jump).

## **Sampling**
1. x_T ~ N(0,I)
2. For t=T..1: predict ε_θ(x_t,t) → compute μ_θ → sample x_{t-1} (inject noise except at t=1)
3. Return x_0

Requires T **sequential** network evaluations — cannot parallelize across t. Main structural cost vs VAE/GAN's O(1) sampling.

---

# **Complexity**

| Phase | Cost |
| ----- | ----- |
| Training/step | 1 forward+backward, independent of T |
| Sampling/sample | O(T) sequential forward passes |

---

# **Key Pitfalls**

* Noise schedule matters a lot — cosine schedule > linear schedule for quality
* Slow sampling is the main limitation — DDIM reformulates as deterministic, non-Markovian, skips steps, same trained model, no retraining needed
* Σ_θ (reverse variance): originally fixed, not learned; learning it improves quality/enables fewer steps
* Don't confuse T=1000 with 1000x training cost — training cost is T-independent, only sampling pays O(T)
* Better mode coverage than GANs (likelihood-based ELBO) at cost of much more sampling compute
* Conditional generation needs classifier guidance or classifier-free guidance — unconditional-only training gives no controllability
* Real image models need a UNet (conv + skip connections + attention) as ε_θ, not a plain MLP

---

# **Connections**

* VAE — diffusion IS a hierarchical VAE: fixed multi-step encoder, learned multi-step decoder; §3 is a direct generalization of VAE's single-step ELBO
* GAN — contrasted throughout: adversarial instability/mode collapse vs stable training/slow sampling
* Score-based models / Langevin dynamics — equivalent formulation predicting ∇log p(x) directly; DDPM's noise-prediction objective ≡ weighted score-matching objective
* Markov chains (HMM) — forward/reverse processes are literally Markov chains, same structure as HMM but continuous Gaussian transitions instead of discrete
* Autoregressive models — third likelihood-based generative family, shares the "sequential generation cost" trade-off for different reasons

---

# **Interview Summary**

A diffusion model defines a fixed, non-learned forward process that gradually destroys data into pure Gaussian noise over T steps, with a closed-form shortcut (x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε) letting training sample any noise level directly from x_0 without simulating the chain. It then trains a neural network to reverse this process by predicting the noise added at each step — derivable as a hierarchical VAE's ELBO, which collapses (via a reparameterization of the target mean in terms of noise) into a simple, unweighted noise-prediction MSE loss (L_simple). This gives stable, ordinary-regression training with no adversarial game, but generation requires T sequential denoising steps rather than one forward pass, making sampling far slower than a VAE or GAN — the central trade-off (training stability and sample quality vs. sampling speed) that motivates faster samplers like DDIM built on the same trained model.
