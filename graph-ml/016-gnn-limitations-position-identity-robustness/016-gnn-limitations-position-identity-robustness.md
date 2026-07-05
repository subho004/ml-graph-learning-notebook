# graph ml 16

# **Limitations of Graph Neural Networks (GNNs)**

Although Graph Neural Networks (GNNs) have achieved remarkable success, they also face several theoretical and practical limitations.

Understanding these limitations has led to newer architectures such as **Position-Aware GNNs**, **Identity-Aware GNNs**, and **Robust GNNs**.

---

# **Major Limitations of GNNs**

---

## **1\. Over-Smoothing**

The most common limitation.

As more GNN layers are stacked,

node embeddings become increasingly similar.

Eventually,

All Nodes

↓

Nearly Identical Embeddings

The model can no longer distinguish between different nodes.

---

### **Intuition**

Initially:

Node A \= Engineer

Node B \= Doctor

Node C \= Student

After many message-passing layers,

neighbor information keeps getting averaged.

Eventually:

A ≈ B ≈ C

Unique information disappears.

---

## **2\. Over-Squashing**

Information from many distant nodes must pass through a few intermediate nodes.

This compresses a large amount of information into small embeddings.

---

### **Example**

Many Nodes

↓

One Bottleneck Node

↓

Target Node

Important information is lost during aggregation.

---

## **3\. Limited Expressiveness**

Most Message Passing GNNs are at most as expressive as the **1-Weisfeiler-Lehman (1-WL) Test**.

They cannot distinguish certain graph structures.

Example:

Two different graphs may receive identical embeddings.

---

## **4\. Scalability**

Large graphs may contain:

Millions of Nodes

Aggregating all neighbors becomes computationally expensive.

Solutions:

* Neighbor Sampling  
* GraphSAGE  
* Mini-batching

---

## **5\. Dynamic Graphs**

Many real networks change continuously.

Example:

* Social networks  
* Financial transactions  
* Communication networks

Traditional GNNs assume static graphs.

---

## **6\. Heterogeneous Graphs**

Standard GNNs assume one node type and one edge type.

Real-world graphs often contain multiple node and relation types.

Extensions:

* R-GCN  
* R-GAT  
* Heterogeneous GNNs

---

## **7\. Sensitivity to Noise**

Incorrect edges or node features can significantly affect predictions because information propagates through the graph.

---

# **Position-Aware GNNs**

One limitation of standard GNNs is that they know **who is connected**, but not **where a node is located within the graph**.

---

# **Why Position Matters**

Consider two graphs:

A \----- B \----- C

and

X \----- Y \----- Z

Suppose B and Y have identical features.

A standard GNN may generate the same embedding,

even if their structural positions differ in a larger graph.

---

# **Intuition**

Imagine two employees.

Both have:

* Same age  
* Same salary  
* Same department

However,

one is the **CEO**,

the other is an **intern**.

Their position within the organization matters.

---

# **How Position-Aware GNNs Work**

They augment node features with **positional information**.

Examples include:

* Shortest-path distances  
* Graph Laplacian eigenvectors (Laplacian Positional Encoding)  
* Random walk positional encodings  
* Centrality measures

Updated node features become:

Node Features

\+

Position Encoding

---

# **Benefits**

* Distinguishes structurally similar nodes  
* Improves graph-level learning  
* Essential for Graph Transformers

---

# **Applications**

* Molecule generation  
* Graph Transformers  
* Traffic networks  
* Knowledge graphs

---

# **Identity-Aware GNNs**

Standard GNNs often fail to uniquely identify nodes with similar neighborhoods.

Identity-aware GNNs explicitly encode **node identity**.

---

# **Problem**

Suppose:

Node A

Neighbors:

B

C

and

Node X

Neighbors:

Y

Z

If all node features are identical,

standard GNNs may produce identical embeddings.

---

# **Solution**

Assign each node an identity representation.

Example:

Node A

↓

ID Embedding

or use structural identifiers such as:

* One-hot node IDs (small graphs)  
* Learnable ID embeddings  
* Hash encodings  
* Structural role encodings

The model learns:

Node Features

\+

Identity Features

---

# **Advantages**

* Better node discrimination  
* Improved expressiveness  
* Distinguishes symmetric structures  
* Useful for link prediction

---

# **Applications**

* Social networks  
* Fraud detection  
* Knowledge graphs  
* Recommendation systems

---

# **Robustness of GNNs**

Robustness refers to:

**The ability of a GNN to maintain good performance even when the graph contains noise, missing data, or adversarial changes.**

---

# **Why Robustness Matters**

Real-world graphs are rarely perfect.

They may contain:

* Missing edges  
* Fake nodes  
* Incorrect labels  
* Noisy features  
* Adversarial attacks

---

# **Example**

Fraud Detection

Attacker creates:

Fake Transactions

↓

Mislead GNN

Without robustness,

the GNN may classify fraudulent accounts as legitimate.

---

# **Types of Robustness Challenges**

---

## **1\. Noisy Node Features**

Example:

Age

Incorrect

Salary

Missing

---

## **2\. Missing Edges**

Some relationships are absent.

---

## **3\. Spurious Edges**

Fake or irrelevant connections are added.

---

## **4\. Adversarial Attacks**

Small changes to nodes or edges are intentionally designed to fool the model.

Example:

Add One Edge

↓

Prediction Changes

---

# **Improving Robustness**

Several techniques improve GNN robustness.

---

## **1\. Graph Augmentation**

Randomly:

* Drop edges  
* Mask features  
* Sample subgraphs

Helps the model generalize better.

---

## **2\. Attention Mechanisms**

Graph Attention Networks (GAT) learn to assign lower weights to noisy neighbors.

---

## **3\. Adversarial Training**

During training,

intentionally perturb the graph.

The model learns to resist attacks.

---

## **4\. Regularization**

Use:

* Dropout  
* Weight decay  
* Feature smoothing

to reduce overfitting.

---

## **5\. Robust Aggregation**

Instead of simple mean aggregation,

use methods less sensitive to outliers, such as attention or robust statistical aggregators.

---

# **Comparison**

| Aspect | Standard GNN | Improved GNN |
| ----- | ----- | ----- |
| Position Awareness | No | Yes (Position-aware GNNs) |
| Identity Awareness | Limited | Explicit node identities |
| Noise Handling | Moderate | Robust training techniques |
| Over-Smoothing | Common | Mitigated with residual/skip connections |
| Large Graphs | Challenging | Neighbor sampling, GraphSAGE |
| Heterogeneous Graphs | Limited | R-GCN, R-GAT |

---

# **Applications**

### **Autonomous Driving**

Robust road-network understanding.

---

### **Banking**

Fraud detection despite fake transactions.

---

### **Social Networks**

Handle fake accounts and noisy friendships.

---

### **Healthcare**

Work with incomplete biological networks.

---

### **Cybersecurity**

Detect attacks despite adversarial modifications.

---

# **Complete Workflow**

Graph  
      ↓  
Position Encoding  
      ↓  
Identity Encoding  
      ↓  
Robust Graph Augmentation  
      ↓  
Message Passing  
      ↓  
Neighbor Aggregation  
      ↓  
Prediction  
      ↓  
Robust Evaluation

---

# **Interview Summary**

Graph Neural Networks (GNNs) have several important limitations, including **over-smoothing** (node embeddings become indistinguishable with increasing depth), **over-squashing** (large amounts of information are compressed through bottleneck nodes), **limited expressiveness** (bounded by the **1-Weisfeiler-Lehman test**), scalability challenges on large graphs, difficulty handling dynamic and heterogeneous graphs, and sensitivity to noisy or adversarial graph structures. **Position-Aware GNNs** address the inability of standard GNNs to recognize a node's location within a graph by incorporating positional encodings such as **Laplacian eigenvectors, shortest-path distances, or random-walk encodings**. **Identity-Aware GNNs** improve expressiveness by assigning explicit identity or structural role information to nodes, allowing the model to distinguish nodes that otherwise have identical neighborhoods. **Robust GNNs** aim to maintain reliable performance despite missing edges, noisy features, fake nodes, or adversarial attacks by using techniques such as **graph augmentation, attention mechanisms, adversarial training, robust aggregation, regularization, and neighbor sampling**. Together, these advances make modern GNNs more expressive, scalable, and reliable for real-world applications.

