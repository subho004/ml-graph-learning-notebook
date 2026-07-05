# graph ml 6

# **Introduction to Graph Neural Networks (GNN)**

A **Graph Neural Network (GNN)** is a Deep Learning model designed to work with **graph-structured data**, where data consists of **nodes (entities)** connected by **edges (relationships)**.

Unlike CNNs (images) and RNNs (sequences), GNNs learn from both:

* Features of individual nodes  
* Relationships between connected nodes

The key idea is:

**Connected nodes often have similar properties, so each node should learn from its neighbors.**

---

# **Why Deep Learning for Graphs?**

Traditional Deep Learning works well for structured data.

| Data Type | Model |
| ----- | ----- |
| Images | CNN |
| Text | RNN / Transformers |
| Audio | RNN / CNN |
| Graphs | GNN |

Graphs are challenging because:

* Variable number of neighbors  
* No fixed ordering of nodes  
* Irregular structure  
* Complex relationships

A graph can represent:

People → Friendships  
Roads → Cities  
Proteins → Chemical Bonds  
Bank Accounts → Transactions

Traditional Neural Networks cannot naturally process these relationships.

---

# **Why Not Use Traditional Machine Learning?**

Suppose we want to detect fraud.

Traditional ML uses:

Balance  
Age  
Income

But ignores:

Who transferred money to whom?

Fraud often depends on **connections**, not just individual features.

GNNs solve this problem.

---

# **Intuition Behind GNN**

Suppose we have:

     Bob  
       |  
Alice--John--David  
       |  
     Emily

Suppose John's profession is unknown.

His neighbors are:

Alice → Engineer

Bob → Engineer

Emily → Engineer

Even without John's features,

we can infer:

John is probably an Engineer.

This is the intuition of GNNs.

---

# **Graph Representation**

A graph is represented as:

\[  
G=(V,E)  
\]

Where:

* **V** \= Nodes  
* **E** \= Edges

Each node contains:

Features

Example:

Person

↓

Age

Salary

Location

Each edge represents:

Relationship

Example:

Friendship

Transaction

Road

---

# **Basic GNN Architecture**

Graph  
      ↓  
Node Features  
      ↓  
Message Passing  
      ↓  
Neighbor Aggregation  
      ↓  
Updated Node Embeddings  
      ↓  
Prediction

---

# **Deep Learning for Graphs**

Just like CNNs stack multiple convolution layers,

GNNs stack multiple **Graph Convolution Layers**.

Layer 1:

Learn from Immediate Neighbors

Layer 2:

Neighbors of Neighbors

Layer 3:

Larger Neighborhood

This allows the network to capture both local and global graph structure.

---

# **Forward Propagation in GNN**

Forward propagation is the prediction phase.

Each layer performs three operations:

### **Step 1: Message Passing**

Neighbors send information.

A

↓

C

←

B

Messages come from neighboring node embeddings.

---

### **Step 2: Aggregation**

Combine neighbor information.

Common methods:

* Mean  
* Sum  
* Max  
* Attention

Example:

Neighbor Features

↓

Average

---

### **Step 3: Update**

Combine:

Own Features

\+

Aggregated Neighbor Features

to compute a new node embedding.

Mathematically:

# **\[**

# **h\_v^{(k+1)}**

Update  
\\left(  
h\_v^{(k)},  
Aggregate(N(v))  
\\right)  
\]

where:

* (h\_v) \= Node embedding  
* (N(v)) \= Neighbor set

---

# **Example of Forward Propagation**

Suppose:

A \---- B \---- C

Initial embeddings:

A \= \[1,0\]

B \= \[0,1\]

C \= \[1,1\]

Update B:

Average(A,C)

↓

Combine with B

↓

New Embedding

After several layers,

each node contains information from a larger neighborhood.

---

# **Loss Function**

Suppose the task is:

Fraud Detection

Predictions:

Node A

↓

Fraud \= 0.82

Actual:

Fraud \= 1

Loss is computed using:

* Cross Entropy (Classification)  
* MSE (Regression)

---

# **Backward Propagation**

After computing the loss,

errors are propagated backward.

Purpose:

Determine how much each parameter contributed to the prediction error.

Backpropagation computes gradients for:

* Weight matrices  
* Aggregation functions  
* Node transformation layers

---

# **Steps of Backpropagation**

### **Step 1**

Forward propagation.

---

### **Step 2**

Compute loss.

---

### **Step 3**

Calculate gradients.

---

### **Step 4**

Update weights.

---

### **Step 5**

Repeat.

---

# **Stochastic Gradient Descent (SGD)**

Weights are updated using Gradient Descent.

Update rule:

\[  
w=w-\\eta\\frac{\\partial L}{\\partial w}  
\]

where:

* (w) \= Weight  
* (\\eta) \= Learning Rate

---

## **Batch Gradient Descent**

Uses:

Entire Dataset

Very stable but slow.

---

## **Stochastic Gradient Descent**

Uses:

One Training Sample

Advantages:

* Faster  
* Less memory  
* Can escape local minima

---

## **Mini-Batch SGD**

Uses small batches:

32

64

128

Most commonly used.

---

# **GraphSAGE (Graph Sample and Aggregate)**

GraphSAGE stands for:

Graph Sample and Aggregate

It is one of the most popular GNN architectures, proposed to efficiently learn node embeddings on **large graphs**.

Unlike standard GCNs, GraphSAGE **does not require the entire graph during training**. Instead, it samples a fixed number of neighbors, making it scalable.

---

# **Why GraphSAGE?**

Suppose a Facebook user has:

10 Million Friends

Aggregating all neighbors is impossible.

GraphSAGE solves this by:

Randomly Sample

↓

Fixed Number of Neighbors

↓

Aggregate

Example:

10 Million Friends

↓

Sample 25

↓

Use Only Those

This dramatically reduces computation.

---

# **GraphSAGE Architecture**

Graph  
      ↓  
Sample Neighbors  
      ↓  
Aggregate Neighbor Features  
      ↓  
Combine with Node Features  
      ↓  
Apply Neural Network  
      ↓  
New Node Embedding

---

# **GraphSAGE Algorithm**

### **Step 1: Neighbor Sampling**

Instead of using every neighbor,

sample a fixed number.

Example:

Node A

↓

Neighbors

↓

Randomly Pick 10

---

### **Step 2: Aggregation**

Aggregate sampled neighbor embeddings.

Common aggregators:

* Mean Aggregator  
* Max Pooling Aggregator  
* LSTM Aggregator

---

### **Step 3: Concatenation**

Combine:

Node Embedding

\+

Aggregated Neighbor Embedding

---

### **Step 4: Linear Transformation**

Pass through a neural network layer.

Result:

Updated Node Embedding

---

### **Step 5: Repeat**

Repeat for multiple layers.

Each layer captures information from farther nodes.

---

# **GraphSAGE Aggregators**

## **1\. Mean Aggregator**

Average neighbor embeddings.

Most common.

---

## **2\. Max Pooling Aggregator**

Apply a neural network to each neighbor, then take the maximum value across neighbors.

Captures the strongest features.

---

## **3\. LSTM Aggregator**

Uses an LSTM to combine neighbor information.

Can model complex interactions but depends on neighbor ordering, making it less commonly used.

---

# **GraphSAGE Workflow**

Graph  
      ↓  
Sample Neighbors  
      ↓  
Aggregate  
      ↓  
Concatenate  
      ↓  
Linear Layer  
      ↓  
ReLU  
      ↓  
Updated Embedding  
      ↓  
Repeat  
      ↓  
Node Prediction

---

# **GCN vs GraphSAGE**

| Feature | GCN | GraphSAGE |
| ----- | ----- | ----- |
| Uses All Neighbors | Yes | No |
| Neighbor Sampling | No | Yes |
| Scalability | Moderate | Excellent |
| Large Graphs | Difficult | Very Good |
| Inductive Learning | Limited | Yes |

---

# **Applications of GraphSAGE**

### **Social Networks**

Friend recommendation.

---

### **Banking**

Fraud detection.

---

### **Recommendation Systems**

User-product recommendations.

---

### **Knowledge Graphs**

Entity prediction.

---

### **Biology**

Protein interaction prediction.

---

# **Complete GNN Training Pipeline**

Graph Data  
      ↓  
Initialize Node Features  
      ↓  
Neighbor Sampling (GraphSAGE)  
      ↓  
Message Passing  
      ↓  
Aggregation  
      ↓  
Updated Node Embeddings  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Stochastic Gradient Descent  
      ↓  
Update Weights  
      ↓  
Repeat Until Convergence

---

# **Interview Summary**

A **Graph Neural Network (GNN)** is a deep learning model designed for **graph-structured data**, where nodes learn by exchanging information with their neighbors through **message passing**. During **forward propagation**, each node aggregates information from neighboring nodes, combines it with its own features, and generates updated node embeddings, which are used for tasks such as node, edge, or graph prediction. The prediction error is measured using a loss function, and **backpropagation** computes gradients to update the network parameters using **Stochastic Gradient Descent (SGD)** or Mini-Batch SGD. **GraphSAGE (Graph Sample and Aggregate)** is a scalable GNN architecture that addresses large graphs by **sampling a fixed number of neighbors** instead of aggregating information from all neighbors. It then aggregates the sampled neighbor embeddings (using mean, max pooling, or LSTM aggregators), combines them with the node's own features, and applies a neural network to produce updated embeddings. GraphSAGE supports **inductive learning**, making it highly effective for large-scale social networks, recommendation systems, fraud detection, and biological network analysis.

