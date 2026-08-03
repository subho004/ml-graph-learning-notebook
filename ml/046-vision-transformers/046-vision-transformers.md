# ml 46

# **Vision Transformers (ViT)**

CNN (Ch. 029): locality + translation equivariance baked into architecture → data-efficient, limited long-range modeling. Transformer (Ch. 037): self-attention models any long-range dependency, no spatial inductive bias, must learn structure from data. ViT: apply near-vanilla Transformer directly to images. Result: worse than CNN at small data scale, matches/beats at large scale — a lesson about inductive bias vs scale.

---

# **1. Patch Embedding**

Self-attention needs a sequence; image is a grid. Pixel-level attention = O(N²) with N=HW → infeasible (224×224 image = 50k pixels).

Fix: chop image into P×P patches first.

x ∈ H×W×C → N = HW/P² patches, each flattened to P²C vector

Project each patch with ONE shared linear map (equivalently: a strided conv, kernel=stride=P):

z_0 = E·x_patch,  E ∈ ℝ^(D×P²C)

This linear projection is the ENTIRE image-specific part. Everything after is a standard Transformer encoder (Ch. 037), unmodified.

## **[CLS] token + Position Embeddings**

* Prepend learnable [CLS] token (BERT-style) — final layer's [CLS] output used for classification
* Self-attention has NO inherent position notion (permutation-equivariant) → add learned position embeddings (2D-grid analog of Transformer's 1D positional encoding)

## **Encoder Stack**

Exactly the multi-head self-attention + MLP + LayerNorm + residual structure from Ch. 037. No new attention mechanism — ViT's novelty is 100% in tokenization.

---

# **2. Why Strip Convolutional Inductive Bias?**

CNN bakes in: **locality** (small receptive field per layer) + **translation equivariance** (same filter everywhere) → sample-efficient, doesn't need to relearn "edges matter" at every position.

ViT gives up both — any patch attends to any patch from layer 1, no shift-invariance guarantee — must LEARN these patterns statistically instead of getting them for free.

## **Empirical Finding (Dosovitskiy et al. 2020)**

* Trained from scratch on ImageNet-1k alone: ViT < CNN (ResNet)
* Pretrained on JFT-300M (300M images), fine-tuned: ViT ≥ CNN, less fine-tuning compute

General lesson: hand-coded priors help most with scarce data, become a ceiling (not floor) once data is abundant enough for a flexible architecture to learn better structure on its own.

---

# **Self-Attention Range vs CNN Receptive Field**

* CNN: receptive field grows linearly with depth (stacked k×k convs)
* ViT: every patch ↔ every patch, single hop, from layer 1

Same "O(1)-hop vs O(depth)-hop" distinction as attention-vs-recurrence (Ch. 038) — now attention vs convolution.

---

# **Complexity**

| Component | Time |
| ----- | ----- |
| Patch embedding | O(N·P²C·D) |
| Self-attention/layer | O(N²D) — dominant, quadratic in N |
| MLP/layer | O(ND²) |
| Comparable CNN layer | O(HWC²K²) — linear in spatial size |

Halving patch size P → quadruples N → quadruples attention cost. Why P=16 (or 32) is standard, not finer.

---

# **Key Pitfalls**

* **Data hunger** — underperforms CNN from scratch on small/medium datasets unless heavily regularized/augmented → **DeiT** fixes this with data-efficient training recipe + distillation token, matches CNN even at ImageNet-1k scale
* **Position embedding resolution mismatch** — tied to pretraining's patch count N; different input resolution needs interpolation, not naive truncation/padding
* **Quadratic attention limits patch resolution** → **Swin Transformer** reintroduces windowed (local) attention + shifted windows for cross-window flow, trading "global from layer 1" for linear scaling
* Loss of sub-patch spatial detail — real info bottleneck, matters more for dense prediction (segmentation) than classification
* No built-in translation equivariance — mitigated via data augmentation (random crops), not architecture
* [CLS] token vs global average pooling — both work, minor variant not fundamental fork

---

# **Connections**

* CNN (Ch. 029) — direct contrast: hard-coded bias (efficient, less flexible) vs learned-from-data (data-hungry, higher ceiling)
* Transformer (Ch. 037) — encoder reused verbatim; only the tokenization (patches) is new
* Graph-level readout (Graph ML) — [CLS] token is conceptually parallel: reduce variable-size set of representations to one fixed-size output
* Word embeddings (Ch. 043) — patch projection E plays the same role as a word-embedding lookup table
* DeiT/knowledge distillation — teacher-student training for data-efficient ViT

---

# **Interview Summary**

Vision Transformer applies a near-unmodified Transformer encoder to images by first splitting the image into fixed-size patches, linearly projecting each into a token embedding (equivalent to a strided convolution), prepending a learnable [CLS] token, and adding learned position embeddings since self-attention itself has no notion of spatial position. Every patch can attend to every other patch from the very first layer, giving ViT a direct global receptive field that a CNN only reaches after stacking many local-receptive-field layers — but ViT also gives up the locality and translation-equivariance inductive biases that make CNNs sample-efficient, so it must learn spatial structure statistically from data rather than getting it for free from the architecture. This is why ViT underperforms comparable CNNs when trained from scratch on modest datasets, but matches or surpasses them once pretrained on very large datasets (or trained with data-efficient recipes like DeiT) — a clean illustration of the general inductive-bias-versus-scale trade-off in deep learning.
