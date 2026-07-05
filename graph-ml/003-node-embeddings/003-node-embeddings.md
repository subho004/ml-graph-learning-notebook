# graph ml 3

# **Node Embeddings**

A **Node Embedding** is a low-dimensional dense vector representation of a node that captures:

* Node features  
* Graph structure  
* Neighborhood information  
* Connectivity patterns

Instead of representing a node as an ID or one-hot vector:

Node A \= \[0,0,1,0,0,0,...\]

it is represented as:

Node A \= \[0.42, \-0.31, 1.27, 0.88, ...\]

Nodes that are similar in the graph have similar embeddings.

---

# **Why Do We Need Node Embeddings?**

Machine learning algorithms cannot directly work with graph structures.

Graphs contain:

* Nodes  
* Edges  
* Neighborhoods

We convert every node into a fixed-length vector.

Then traditional ML or Deep Learning models can use these vectors.

---

# **Intuition**

Suppose we have a social network:

Alice \---- Bob \---- Charlie  
   |  
 David

Alice and David are friends.

Alice and Bob share similar neighborhoods.

Their embeddings should be close together.

Example:

Alice

↓

\[0.51, \-0.22, 1.4\]

Bob

↓

\[0.48, \-0.18, 1.3\]

Charlie may have a different embedding.

---

# **What Should a Good Node Embedding Capture?**

A good embedding captures:

### **Graph Structure**

Nearby nodes have similar vectors.

---

### **Node Attributes**

Age

Location

Interests

etc.

---

### **Connectivity**

Who is connected to whom.

---

### **Community Information**

Nodes in the same community have similar embeddings.

---

# **Types of Node Embedding Methods**

There are two major categories.

Node Embeddings  
      │  
      ├── Unsupervised  
      │  
      └── Supervised

---

# **1\. Unsupervised Node Embeddings**

No labels are required.

Goal:

Learn graph structure automatically.

---

Popular methods:

* DeepWalk  
* Node2Vec  
* LINE  
* Graph Autoencoders

---

# **Intuition**

Suppose we have Facebook.

Nobody tells the algorithm:

Person A likes sports.

Instead it learns:

People with similar friends  
↓

Probably similar.

---

# **Advantages**

* No labels needed.  
* Works on massive graphs.  
* Learns useful representations.

---

# **2\. Supervised Node Embeddings**

Uses node labels.

Example:

Fraud

Normal

The embedding is optimized for prediction.

---

Example:

Bank accounts.

Input:

Account Features

\+

Graph Structure

Output:

Fraud Probability

During training, embeddings become better for fraud detection.

---

# **Comparison**

| Feature | Unsupervised | Supervised |
| ----- | ----- | ----- |
| Labels Needed | No | Yes |
| Learns | Structure | Structure \+ Task |
| Examples | Node2Vec, DeepWalk | GCN, GraphSAGE, GAT |

---

# **Node Similarity**

The goal of node embeddings is:

Similar Nodes

↓

Similar Vectors

---

Suppose:

Doctor A

Doctor B

Both:

* Work at same hospital  
* Have same specialization  
* Collaborate often

Their embeddings should be close.

---

# **Measuring Node Similarity**

After obtaining embeddings:

A \= \[0.4, 0.8\]

B \= \[0.5, 0.7\]

Similarity can be measured using:

---

## **Cosine Similarity**

Most popular.

Measures angle between vectors.

Range:

\-1 to 1

Interpretation:

1

↓

Very Similar

---

## **Euclidean Distance**

Measures physical distance.

Smaller distance:

↓

More Similar

---

## **Dot Product**

Often used during training.

Higher value:

↓

More Similar

---

# **Random Walk**

Random Walk is one of the most important ideas behind node embeddings.

---

# **Intuition**

Imagine standing on a node.

At every step:

Choose one neighboring node randomly.

Example:

A \---- B \---- C  
      |  
      D

Random walk:

A

↓

B

↓

D

↓

B

↓

C

Produces a sequence:

A B D B C

This is similar to a sentence in NLP.

---

# **Why Random Walk?**

Words appearing together in a sentence are related.

Similarly,

nodes appearing together in random walks are related.

This idea allows graph problems to be solved using NLP techniques.

---

# **DeepWalk**

DeepWalk combines:

Random Walk

\+

Word2Vec

Steps:

1. Perform many random walks.  
2. Treat each walk as a sentence.  
3. Train Word2Vec.  
4. Obtain node embeddings.

---

# **Node2Vec**

Node2Vec improves DeepWalk by making random walks more flexible.

Instead of purely random movement,

it controls how the walk explores the graph.

---

# **Intuition**

Suppose:

A

|

B

/ \\

C  D

Node2Vec decides whether to:

Stay Nearby

or

Explore Far Away

---

It introduces two parameters:

### **p (Return Parameter)**

Controls probability of returning.

Large p:

Don't go back.

Small p:

Return frequently.

---

### **q (In-Out Parameter)**

Controls exploration.

Small q:

Explore far.

Large q:

Stay local.

---

# **BFS vs DFS Behavior**

Node2Vec can imitate both.

---

## **BFS-like Walk**

Focuses nearby nodes.

Learns:

Community Similarity

Example:

Friends in same group.

---

## **DFS-like Walk**

Explores distant nodes.

Learns:

Structural Similarity

Example:

Managers from different departments.

---

# **Encoder-Decoder Framework**

Most node embedding methods follow an Encoder-Decoder framework.

Graph

↓

Encoder

↓

Embedding

↓

Decoder

↓

Prediction

---

# **Encoder**

Converts node into embedding.

Example:

Node

↓

128-D Vector

Examples:

* GCN  
* GraphSAGE  
* Node2Vec

---

# **Decoder**

Uses embeddings to solve tasks.

Examples:

Node Classification

Link Prediction

Similarity

Graph Classification

---

# **Node Similarity Measure**

The decoder often predicts similarity using:

Dot Product

or

Cosine Similarity

If:

Embedding(A)

≈

Embedding(B)

Then:

Nodes are similar.

---

# **Extension to Graph Embedding**

Instead of embedding one node,

we can embed the entire graph.

---

Suppose a molecule:

Atoms

\+

Chemical Bonds

↓

One vector.

Graph

↓

\[0.31, \-1.4, 2.8, ...\]

---

# **Why Graph Embeddings?**

Sometimes:

Entire graph

\=

One data sample.

Examples:

* Molecule  
* Protein  
* Social Network  
* Knowledge Graph

---

# **How Are Graph Embeddings Created?**

Step 1

Compute node embeddings.

Node A

↓

Vector

---

Step 2

Aggregate all node embeddings.

Methods:

### **Mean Pooling**

Average.

---

### **Sum Pooling**

Add vectors.

---

### **Max Pooling**

Take maximum values.

---

### **Attention Pooling**

Important nodes receive larger weights.

---

Result:

Entire Graph

↓

Single Embedding

---

# **Applications of Graph Embeddings**

### **Molecule Classification**

Predict:

Cancer Drug?

---

### **Protein Classification**

Predict:

Protein Function

---

### **Social Networks**

Predict:

Community Type

---

### **Cybersecurity**

Predict:

Malicious Network?

---

# **Node Embedding vs Graph Embedding**

| Feature | Node Embedding | Graph Embedding |
| ----- | ----- | ----- |
| Represents | One Node | Entire Graph |
| Output | Node Vector | Graph Vector |
| Applications | Node Classification, Link Prediction | Molecule Classification, Graph Classification |

---

# **Workflow of Node2Vec**

Graph  
   ↓  
Random Walks  
   ↓  
Generate Node Sequences  
   ↓  
Word2Vec (Skip-Gram)  
   ↓  
Node Embeddings  
   ↓  
Similarity / Classification / Link Prediction

---

# **Applications of Node Embeddings**

### **Social Networks**

Friend recommendation.

---

### **Banking**

Fraud detection.

---

### **Recommendation Systems**

User-product similarity.

---

### **Knowledge Graphs**

Entity representation.

---

### **Biological Networks**

Gene prediction.

---

# **Interview Summary**

A **Node Embedding** is a dense, low-dimensional vector representation of a graph node that captures its **features, neighborhood information, and graph topology**. Node embeddings can be learned using **unsupervised methods** such as **DeepWalk** and **Node2Vec**, which generate random walks and apply **Word2Vec** to learn node representations, or **supervised methods** such as **GCN, GraphSAGE, and GAT**, where embeddings are optimized for tasks like node classification. **Node2Vec** extends DeepWalk by introducing biased random walks through the **return parameter (p)** and **in-out parameter (q)**, allowing exploration of both local communities (BFS-like behavior) and structurally similar nodes (DFS-like behavior). Most graph representation learning follows an **encoder-decoder framework**, where the encoder generates embeddings and the decoder performs tasks such as node classification, link prediction, or similarity estimation using measures like **cosine similarity, dot product, or Euclidean distance**. The same idea extends to **graph embeddings**, where node embeddings are aggregated using **mean, sum, max, or attention pooling** to obtain a single vector representing the entire graph for tasks such as **molecule classification, protein function prediction, and graph classification**.

