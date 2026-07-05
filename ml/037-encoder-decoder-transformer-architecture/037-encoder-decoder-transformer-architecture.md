# encoder decoder

# **Encoder-Decoder Transformer Architecture**

The **Encoder-Decoder Transformer** is the original Transformer architecture introduced in the paper:

**"Attention Is All You Need" (2017)**

It is designed for **sequence-to-sequence (Seq2Seq)** tasks, where an input sequence is transformed into an output sequence.

Examples:

* Machine Translation  
* Text Summarization  
* Question Answering  
* Speech Recognition

Unlike RNNs and LSTMs, Transformers process the entire sequence in parallel using **Self-Attention**, making them much faster and better at capturing long-range dependencies.

---

# **Why Was the Transformer Introduced?**

Earlier Seq2Seq models used:

RNN

↓

LSTM

↓

Encoder-Decoder

Problems:

* Sequential processing (slow)  
* Vanishing gradients  
* Difficulty remembering long sequences

Transformers replace recurrence with:

Self-Attention

allowing all words to interact simultaneously.

---

# **High-Level Architecture**

Input Sentence  
      ↓  
Input Embedding  
      ↓  
Positional Encoding  
      ↓  
Encoder Stack  
      ↓  
Context Representation  
      ↓  
Decoder Stack  
      ↓  
Linear Layer  
      ↓  
Softmax  
      ↓  
Output Sentence

The architecture consists of:

* Encoder  
* Decoder

---

# **Overall Transformer Architecture**

               INPUT

                   │

         Token Embeddings

                   │

       Positional Encoding

                   │

          ┌───────────────┐  
          │    Encoder    │  
          └───────────────┘  
                   │  
         Context Embeddings  
                   │  
          ┌───────────────┐  
          │    Decoder    │  
          └───────────────┘  
                   │  
             Linear Layer  
                   │  
               Softmax  
                   │  
               Prediction

---

# **Input Embeddings**

Words are first converted into vectors.

Example:

Sentence:

I love AI

↓

Embeddings

I

↓

\[0.2,0.4,...\]

Each word becomes a dense numerical vector.

---

# **Positional Encoding**

Unlike RNNs,

Transformers have **no notion of word order**.

Therefore,

positional information is added.

Example:

Word

\+

Position

↓

Final Embedding

Example:

I

Position 1

Love

Position 2

AI

Position 3

This allows the model to distinguish:

Dog bites man

≠

Man bites dog

---

# **Encoder**

The encoder reads the entire input sentence and produces contextual representations.

It consists of multiple identical layers.

Each encoder layer contains:

Multi-Head Self Attention

↓

Add & Normalize

↓

Feed Forward Network

↓

Add & Normalize

Usually:

* 6 encoder layers (original paper)  
* 12 or more in modern models

---

# **Step 1: Multi-Head Self-Attention**

Each word attends to every other word.

Example:

Sentence:

The animal didn't cross the road because it was tired.

The word:

it

learns that it refers to:

animal

instead of:

road

This captures context.

---

# **Multi-Head Attention**

Instead of one attention mechanism,

multiple attention heads operate in parallel.

Example:

Head 1

↓

Grammar

Head 2

↓

Meaning

Head 3

↓

Long-distance Relation

Each head learns different relationships.

---

# **Feed Forward Network (FFN)**

Each token independently passes through a small neural network.

Purpose:

* Increase representation power  
* Learn nonlinear transformations

Pipeline:

Attention Output

↓

Linear

↓

ReLU/GELU

↓

Linear

---

# **Add & Normalize**

Residual (skip) connections are added:

Input

\+

Layer Output

Benefits:

* Stable gradients  
* Faster training  
* Enables deep Transformers

Layer Normalization stabilizes activations.

---

# **Encoder Output**

After all encoder layers,

every input word becomes a **context-aware embedding**.

Example:

Apple

In:

Apple released a new phone.

means the company.

In:

Apple is healthy.

means the fruit.

The encoder captures this difference.

---

# **Decoder**

The decoder generates the output sequence **one token at a time**.

Each decoder layer has:

Masked Multi-Head Self Attention

↓

Add & Normalize

↓

Encoder-Decoder Attention

↓

Add & Normalize

↓

Feed Forward Network

↓

Add & Normalize

---

# **1\. Masked Self-Attention**

The decoder cannot "see the future."

Example:

Generating:

I love AI

When predicting:

love

It cannot access:

AI

Masking ensures autoregressive generation.

---

# **2\. Encoder-Decoder Attention**

The decoder attends to the encoder outputs.

Example:

English:

I love AI

Encoder creates context.

Decoder attends to it while generating:

Je

↓

aime

↓

IA

(French)

This connects input and output sequences.

---

# **3\. Feed Forward Network**

Same as encoder.

Each token is processed independently.

---

# **Linear Layer**

Converts decoder output into vocabulary scores.

Example:

Vocabulary:

Cat

Dog

Car

Tree

Each word receives a score.

---

# **Softmax**

Softmax converts scores into probabilities.

Example:

Cat

0.05

Dog

0.85

Tree

0.10

Highest probability becomes the predicted token.

---

# **Training**

Training uses:

Teacher Forcing

The decoder receives the correct previous token during training.

Loss:

Cross Entropy Loss

Optimizer:

* Adam  
* AdamW

---

# **Inference**

During inference:

Step 1

Generate first word.

↓

Step 2

Feed generated word back into decoder.

↓

Generate next word.

↓

Repeat until:

\<END\>

token.

---

# **Encoder-Only vs Decoder-Only vs Encoder-Decoder**

| Architecture | Components | Example Models | Main Use |
| ----- | ----- | ----- | ----- |
| Encoder Only | Encoder | BERT, RoBERTa | Understanding, classification |
| Decoder Only | Decoder | GPT, LLaMA | Text generation |
| Encoder-Decoder | Both | T5, BART, Original Transformer | Translation, summarization, Seq2Seq |

---

# **Applications**

### **Machine Translation**

English → French

---

### **Text Summarization**

Long article → Short summary

---

### **Question Answering**

Question → Answer

---

### **Speech Recognition**

Speech → Text

---

### **Image Captioning**

Image → Sentence

---

### **Code Generation**

Description → Source Code

---

# **Advantages**

* Parallel processing  
* Captures long-range dependencies  
* Faster than RNNs/LSTMs  
* Highly scalable  
* State-of-the-art performance on Seq2Seq tasks

---

# **Limitations**

* High memory usage (attention is (O(n^2)) with sequence length)  
* Expensive for very long sequences  
* Requires large amounts of training data

---

# **Complete Transformer Pipeline**

Input Tokens  
      ↓  
Embeddings  
      ↓  
Positional Encoding  
      ↓  
Encoder Stack  
      ↓  
Context Representation  
      ↓  
Masked Self-Attention  
      ↓  
Encoder-Decoder Attention  
      ↓  
Feed Forward Network  
      ↓  
Linear Layer  
      ↓  
Softmax  
      ↓  
Output Tokens

---

# **Interview Summary**

The **Encoder-Decoder Transformer** is the original Transformer architecture introduced in **"Attention Is All You Need" (2017)** for **sequence-to-sequence (Seq2Seq)** tasks such as **machine translation, summarization, and question answering**. The **encoder** processes the entire input sequence using **multi-head self-attention**, **feed-forward networks**, **residual connections**, and **layer normalization** to produce context-aware embeddings. The **decoder** generates the output sequence one token at a time using **masked self-attention** (to prevent looking at future tokens), **encoder-decoder attention** (to attend to the encoder's output), and a **feed-forward network**, followed by a **linear layer** and **softmax** to predict the next token. **Positional encoding** is added to input embeddings because Transformers process tokens in parallel and do not inherently capture word order. Compared to RNNs and LSTMs, Transformers provide **parallel computation**, **better long-range dependency modeling**, and **higher scalability**. Modern Transformer models fall into three categories: **encoder-only models** (e.g., BERT) for understanding tasks, **decoder-only models** (e.g., GPT, LLaMA) for text generation, and **encoder-decoder models** (e.g., T5, BART) for sequence-to-sequence applications.

