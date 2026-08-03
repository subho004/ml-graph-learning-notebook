# ml 43

# **Word Embeddings: Word2Vec and GloVe**

One-hot/bag-of-words: zero similarity between any two distinct words, no generalization. Dense embeddings fix this — words with similar distributional behavior get similar vectors.

## **Distributional Hypothesis**

"You shall know a word by the company it keeps" (Firth, 1957) — turns semantics into a statistical, corpus-derivable problem.

---

# **1. Word2Vec: Skip-Gram**

Center word predicts context words within a window. Each word has 2 vectors: v_w (center role), u_w (context role).

## **Full Softmax (too slow)**

p(o|c) = exp(u_o·v_c) / Σ_w exp(u_w·v_c)

Denominator sums over entire vocabulary V (~10^5-10^6) for EVERY training pair → computationally infeasible at scale.

## **Negative Sampling**

Reframe as binary classification: real (center,context) pair vs k randomly sampled fake pairs.

p(D=1|c,o) = σ(u_o·v_c)

Objective per pair (maximize):

log σ(u_o·v_c) + Σ_i log σ(-u_wi·v_c)     [k negative samples w_i]

* First term: pull true context word closer
* Second term: push k random negatives away
* Replaces O(V) softmax with O(k) — THE trick that made web-scale training feasible

## **Noise Distribution**

Sample negatives ∝ unigram_freq^(3/4), not raw frequency — flattens distribution, avoids wasting negatives on "the".

## **CBOW (mirror variant)**

Context → center (averaged context vectors predict center word). Smooths noise, faster, better for frequent words. Skip-gram: more training signal per rare word, generally better for infrequent words.

---

# **2. GloVe: Global Vectors**

Uses global co-occurrence matrix X (X_ij = count of j in i's context, corpus-wide), not just local windows.

## **Key Insight: Ratios Encode Meaning**

Ratio P_ik/P_jk (not raw probabilities) discriminates: large when k relates to i not j, small vice versa, ~1 when irrelevant to both or relevant to both. This ratio behavior is what embeddings should reproduce.

## **Deriving the Objective**

Requiring F to depend only on vector differences (relative position) and combine via dot product, then requiring F to be a homomorphism ℝ,+ → ℝ>0,× forces F = exp. Working through the algebra + absorbing normalization terms into per-word biases gives:

v_i·u_k + b_i + b̃_k = log X_ik

## **Weighted Least-Squares Loss**

J = Σ f(X_ik) · (v_i·u_k + b_i + b̃_k − log X_ik)²

f(x) = (x/x_max)^α for x < x_max, else 1   (α=0.75, x_max≈100)

* Downweights rare (noisy) and very frequent (dominating) co-occurrences
* f(0)=0 → skip zero-count pairs entirely (avoids log(0), most pairs never co-occur)

## **GloVe = Implicit Matrix Factorization**

Objective is literally weighted low-rank factorization of log X. Levy & Goldberg (2014): Word2Vec skip-gram+negative-sampling implicitly factorizes a shifted PMI matrix too — both methods, despite different training procedures, are secretly the same kind of thing.

---

# **Word2Vec vs GloVe**

| Aspect | Word2Vec | GloVe |
| ----- | ----- | ----- |
| Signal | local context windows | global co-occurrence counts |
| Training | online SGD streaming | batch: build X once, then optimize |
| Structure | implicit PMI factorization | explicit log(X) factorization |
| Scalability | scales to streaming corpora | needs co-occurrence matrix built upfront |

---

# **Key Pitfalls**

* **Static embeddings can't handle polysemy** — one vector per word regardless of sense ("bank" river vs financial) → motivates contextual embeddings (BERT/Transformers)
* **OOV words** — no embedding for unseen words → FastText fixes via character n-gram subword embeddings
* Rare words: dropped below min-count threshold, or noisy if just above it
* Window size: small → syntactic similarity; large → topical/semantic similarity (real qualitative difference, not just a speed dial)
* Frequent-word subsampling needed separately from negative-sampling correction
* Embedding dim: too small underfits, too large overfits/wastes compute (typical range 100-300)

---

# **Connections**

* PCA — both GloVe and Word2Vec are low-rank factorizations of a co-occurrence/PMI-derived matrix
* Noise-Contrastive Estimation — negative sampling's "real vs fake" trick is an instance of NCE, same principle behind modern contrastive self-supervised learning (SimCLR etc.)
* Transformers — embedding layer input is often initialized/trained analogously; Transformers make embeddings *contextual* via attention instead of static
* king − man + woman ≈ queen — direct consequence of GloVe's "F depends only on vector differences" design choice, not a bolted-on trick

---

# **Interview Summary**

Word2Vec learns dense word vectors by predicting context words from a center word (Skip-gram) or vice versa (CBOW) within a sliding window; since a full softmax over the vocabulary is too expensive per training pair, **negative sampling** reframes it as k+1 binary classifications (real pair vs. sampled fake pairs), replacing O(V) cost with O(k). **GloVe** instead starts from the corpus-wide co-occurrence matrix and derives its objective from the requirement that co-occurrence probability *ratios* (not raw probabilities) encode meaning, producing a weighted least-squares regression on log co-occurrence counts. Despite very different training procedures, both methods are provably, at their core, low-rank matrix factorizations of co-occurrence-derived statistics (Word2Vec implicitly factorizes a shifted PMI matrix). Both produce **static** embeddings — one fixed vector per word regardless of context — which is exactly the limitation that motivated contextual, attention-based embeddings in Transformer models.
