# reinforcement-learning 2

# **RLHF: Reinforcement Learning from Human Feedback**

Pretraining optimizes next-token likelihood — a proxy, not "be helpful/aligned." RLHF: learn a reward model from human preferences, then use it as RL signal to fine-tune the LM. Uses **policy-gradient** RL (Q-learning chapter covered value-based only).

---

# **Three-Stage Pipeline**

1. **SFT** — fine-tune pretrained model on human demonstrations (ordinary supervised cross-entropy) → π_SFT
2. **Reward Model** — train r_φ(x,y) on human pairwise preference comparisons
3. **RL Fine-Tuning** — maximize r_φ with KL penalty back to π_SFT, via PPO

---

# **1. Reward Model — Bradley-Terry**

Pairwise comparisons ("which is better?") are far more reliable to collect than absolute scores.

## **Bradley-Terry Model**

p(y_w ≻ y_l | x) = σ(r*(x,y_w) − r*(x,y_l))

Logistic function of reward difference — same structure as Elo ratings.

## **Training Loss (MLE)**

L_RM = −E[log σ(r_φ(x,y_w) − r_φ(x,y_l))]

Structurally = logistic regression where "features" are the difference of two learned scalar scores. Pushes preferred response's score up, rejected's down.

**Key limitation**: only relative ordering matters (reward shiftable by any per-prompt constant) — reward model only reliably ranks things similar to what it saw; no guarantee it generalizes to outputs the policy discovers later (→ reward hacking).

---

# **2. Policy Gradient (background, since Q-learning only covers value-based)**

Action space = entire token sequences → max_a Q(s,a) intractable. Need direct policy parameterization.

## **Policy Gradient Theorem**

∇J(θ) = E_τ~π_θ [∇log π_θ(τ) · R(τ)]      (REINFORCE, log-derivative trick)

Only needs sampling + log-probs, no differentiable reward needed.

**Problem**: high variance, single bad update can destroy a good policy → motivates PPO.

---

# **3. KL-Constrained RL Objective**

Unconstrained max r_φ(x,y) → policy drifts arbitrarily, exploits reward model's blind spots (**reward hacking**).

Fix:

max_θ  E[ r_φ(x,y) − β·KL(π_θ(·|x) ‖ π_SFT(·|x)) ]

Same "expected value − KL to reference" shape as VAE's ELBO, with π_SFT playing the role of the fixed prior.

Implemented as per-token penalty:

r_total = r_φ(x,y) − β Σ_i log[π_θ(y_i|·)/π_SFT(y_i|·)]

(sum of token log-ratios = sequence-level KL exactly, by chain rule)

---

# **4. PPO — Stabilizing the Update**

ρ_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)     (probability ratio)
Â_t = advantage estimate (reward model + learned value/critic)

L_PPO = E[ min( ρ_t·Â_t,  clip(ρ_t, 1-ε, 1+ε)·Â_t ) ]

Clip caps how far a single update can push the policy in one step — prevents destabilizing updates. min() ensures the clip only ever makes the objective MORE conservative.

## **Loop**
1. Sample prompts, generate responses from current π_θ
2. Score with frozen r_φ, compute KL-penalized reward vs frozen π_SFT
3. Estimate advantages (value/critic network)
4. PPO clipped update
5. Repeat

---

# **5. DPO — Skipping RL Entirely**

PPO needs 4 models (policy, reference, reward, critic), fiddly to tune. DPO reaches the SAME optimal policy via a supervised loss, no RL loop.

## **Derivation**

KL-constrained objective has known closed-form optimal policy:

π*(y|x) = (1/Z(x))·π_SFT(y|x)·exp(r_φ(x,y)/β)

Solve for reward → substitute into Bradley-Terry difference → the intractable Z(x) **cancels exactly** (same for both y_w, y_l under same prompt):

L_DPO = −E[ log σ( β log(π_θ(y_w|x)/π_SFT(y_w|x)) − β log(π_θ(y_l|x)/π_SFT(y_l|x)) ) ]

Same functional form as reward-model loss — but the "reward" IS the policy's own log-prob-ratio vs reference. Policy = implicit reward model. Ordinary supervised gradient descent on preference pairs, no sampling during training.

---

# **Complexity**

| Stage | Models held | Sampling during training? |
| ----- | ----- | ----- |
| SFT | 1 | no |
| Reward Model | 1 | no (fixed dataset) |
| PPO | 4 (policy, ref, reward, critic) | YES — every step, dominant cost |
| DPO | 2 (policy, frozen ref) | no — same cost profile as SFT |

---

# **Key Pitfalls**

* Reward hacking — policy exploits RM blind spots (classic: length bias, longer ≠ better but scores higher)
* Reward model OOD generalization gap is the root cause of hacking, not "insufficient KL penalty" alone
* β tuning — too small → hacking/degenerate outputs; too large → policy can't improve past π_SFT
* PPO instability in practice despite clipping — sensitive to LR, batch size, value-init, reward normalization
* Human labels are noisy/inconsistent — Bradley-Terry assumes one consistent underlying preference function
* DPO doesn't produce a standalone reusable reward model (unlike PPO's explicit r_φ) — matters if you need reward for e.g. best-of-n reranking
* Distribution shift between preference-collection prompts and deployment prompts breaks both PPO and DPO

---

# **Connections**

* Q-learning (RL Ch.1) — value-based vs policy-gradient RL; PPO needed because token-sequence action space makes max_a Q(s,a) intractable
* Actor-critic — PPO = clipped-update actor-critic (policy=actor, value fn=critic)
* Logistic regression — Bradley-Terry RM loss is logistic regression on a difference of two learned scores
* VAE ELBO — KL-constrained RL objective has identical "value − KL-to-reference" shape
* GANs — RM and GAN discriminator are both learned scalar scorers driving a separate generator/policy, though RM is trained once and frozen, not adversarially co-trained
* Transformer (Ch. 037) — RM and policy are typically the same architecture, different output head/objective

---

# **Interview Summary**

RLHF aligns a pretrained language model with human preferences in three stages: supervised fine-tuning on demonstrations, training a **reward model** via a Bradley-Terry maximum-likelihood loss on human pairwise preference comparisons, then RL fine-tuning the policy to maximize that learned reward subject to a **KL penalty** back to the SFT policy — the KL term is essential because an unconstrained policy would exploit the reward model's blind spots (reward hacking) rather than genuinely improving. The RL stage traditionally uses **PPO**, a clipped policy-gradient method that bounds how far a single update can shift the policy, requiring four models held simultaneously and on-policy sampling every training step. **DPO** shows the same KL-constrained optimum is reachable by a much simpler supervised loss: substituting the closed-form optimal policy back into the Bradley-Terry objective makes the reward model's normalizing constant cancel exactly, leaving a loss where the policy's own log-probability ratio (versus the frozen reference) directly plays the role of the reward — collapsing the entire RL loop into an ordinary, stable, sampling-free training objective.
