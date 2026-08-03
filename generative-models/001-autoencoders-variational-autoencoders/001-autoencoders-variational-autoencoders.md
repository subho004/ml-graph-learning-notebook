# generative-models 1

# **Autoencoders and Variational Autoencoders (VAEs)**

Autoencoders learn to compress and reconstruct data with no labels needed, using reconstruction error as the training signal. A VAE reinterprets this as a probabilistic latent-variable model — the piece that actually makes it a generative model, not just a compressor.

---

# **1. Vanilla Autoencoder**

Encoder: z = f(x) — compress to bottleneck  
Decoder: x̂ = g(z) — reconstruct

Objective: minimize ||x - g(f(x))||²

* Bottleneck (d_z << d_x) forces discarding info → prevents trivial identity function
* Linear autoencoder + squared error = same subspace as PCA
* Nonlinear version = nonlinear generalization of PCA (curved manifolds)

## **Why it's NOT generative**

Nothing constrains the shape of the latent space. Random z ~ N(0,I) decoded → garbage, because decoder never saw that region during training. No mechanism makes latent space "samplable."

## **Regularized Variants**

* Denoising autoencoder — reconstruct clean x from corrupted x̃
* Sparse autoencoder — penalize activation of z
* Contractive autoencoder — penalize sensitivity of z to small input changes

---

# **2. Variational Autoencoder (VAE)**

## **Generative Story**

z ~ N(0, I)  
x | z ~ p_θ(x|z)

True likelihood p_θ(x) = ∫ p_θ(x|z) p(z) dz → intractable (no closed form).  
True posterior p_θ(z|x) → also intractable (needs p_θ(x) in denominator).

Fix: learn approximate posterior q_φ(z|x) (the "encoder").

## **ELBO Derivation**

Starting from KL(q_φ(z|x) || p_θ(z|x)) ≥ 0, rearranging gives:

log p_θ(x) = ELBO(x) + KL(q_φ(z|x) || p_θ(z|x))

Since KL ≥ 0:

log p_θ(x) ≥ ELBO(x)

Maximizing ELBO = maximizing a valid lower bound on likelihood AND pushing q_φ toward the true posterior.

## **ELBO Decomposition**

ELBO(x) = E_q[log p_θ(x|z)] − KL(q_φ(z|x) || p(z))
           = reconstruction term        − regularizer term

* Reconstruction term: same objective as vanilla AE
* KL regularizer: forces q_φ(z|x) toward N(0,I) for every x → THIS is what makes the latent space samplable at generation time

## **Closed-Form Gaussian KL**

With q_φ(z|x) = N(μ, diag(σ²)) and prior N(0,I):

KL = ½ Σ (μ_j² + σ_j² − log σ_j² − 1)

## **Reparameterization Trick**

Can't backprop through a sampling operation z ~ q_φ(z|x) directly. Rewrite:

z = μ_φ(x) + σ_φ(x) ⊙ ε,  ε ~ N(0, I)

Now z is a deterministic, differentiable function of (μ, σ, ε) — ordinary backprop works.

---

# **Training Steps**

1. Encoder → μ_φ(x), log σ²_φ(x)
2. Sample ε ~ N(0,I); reparameterize to get z
3. Decoder(z) → reconstruction
4. Loss = reconstruction loss + β·KL (β-VAE knob)
5. Backprop through decoder → reparameterized z → encoder
6. Generation: discard encoder, sample z ~ N(0,I) directly, decode

---

# **Key Pitfalls**

* Predict log σ² (not σ² directly) → guarantees positivity, no constrained activation needed
* Posterior collapse — powerful decoder ignores z, KL → 0, model degenerates to unconditional generator
* β-VAE trade-off — β > 1 → more disentangled latents, blurrier reconstructions; β < 1 → sharper but less regularized/samplable
* Blurry samples — Gaussian decoder + squared error implicitly averages over multimodal reconstructions
* Latent dim too small → bottlenecks quality; too large → weaker regularization, risk of collapse in some dims

---

# **Connections**

* PCA — linear special case
* EM algorithm — same variational/ELBO principle, but VAE parameterizes both sides with neural nets and optimizes jointly via gradient descent
* GANs — same z→data generative mapping, adversarial objective instead of ELBO, sharper samples but no tractable likelihood, unstable training
* Diffusion models — can be derived as a hierarchical VAE with a fixed multi-step forward noising encoder

---

# **Interview Summary**

A vanilla autoencoder compresses and reconstructs data through a bottleneck, but has no mechanism forcing the latent space to be "samplable" — decoding random noise usually produces garbage. A **VAE** fixes this by treating the problem probabilistically: since the true data likelihood and posterior are intractable, it optimizes a tractable **ELBO** (evidence lower bound) instead, which decomposes into a reconstruction term (same as a vanilla AE) plus a **KL regularizer** that forces the encoder's output distribution toward a fixed prior N(0,I) for every input. This regularizer is what makes the latent space samplable at generation time. Backpropagating through the random sampling step is made possible by the **reparameterization trick**, rewriting the sample as a deterministic function of the encoder's outputs and external noise. VAEs give fast, single-pass sampling and a tractable (lower-bounded) likelihood, at the cost of characteristically blurrier samples than GANs or diffusion models.
