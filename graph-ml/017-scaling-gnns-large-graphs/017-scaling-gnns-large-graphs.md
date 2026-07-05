# graph ml 17

# **Scaling Up GNNs to Large Graphs**

One of the biggest challenges in **Graph Neural Networks (GNNs)** is scalability.

Real-world graphs can contain:

* Millions of nodes  
* Billions of edges  
* Highly connected hub nodes

Examples:

* Facebook Social Graph  
* Google Knowledge Graph  
* Twitter/X Network  
* Financial Transaction Networks

A standard GNN cannot process the entire graph efficiently because each node aggregates information from all of its neighbors.

---

# **Why Do Standard GNNs Not Scale?**

Suppose one node has:

1 Million Neighbors

One GNN layer requires aggregating information from all of them.

Now consider 3 GNN layers:

Layer 1 → 1-hop neighbors

Layer 2 → Neighbors of neighbors

Layer 3 → 3-hop neighbors

The number of accessed nodes grows exponentially.

This is called the **Neighborhood Explosion Problem**.

---

# **Neighborhood Explosion**

Suppose:

Each node has 100 neighbors

Then

1 Layer

↓

100 nodes

2 Layers

↓

100 × 100

\=

10,000 nodes

3 Layers

↓

100³

\=

1,000,000 nodes

Clearly impossible for large graphs.

---

# **Approaches for Scaling GNNs**

Several techniques have been proposed:

1. Neighbor Sampling (GraphSAGE)  
2. Cluster-based Training (Cluster-GCN)  
3. Graph Sampling  
4. Mini-batch Training  
5. Simplified GNNs (SGC)  
6. Layer Sampling (FastGCN, LADIES)

---

# **GraphSAGE: Neighbor Sampling**

GraphSAGE (**Graph Sample and Aggregate**) was one of the first scalable GNNs.

Instead of aggregating **all neighbors**, GraphSAGE samples only a fixed number.

---

# **Intuition**

Suppose a node has:

1000 neighbors

Instead of using all:

1000

↓

Sample 25

↓

Aggregate

The sampled neighbors approximate the full neighborhood.

---

# **GraphSAGE Neighbor Sampling**

Suppose:

Node A

↓

100 Neighbors

Sampling:

Randomly Choose

↓

20 Neighbors

Then aggregate only these 20\.

---

# **Multi-Layer Sampling**

Suppose:

2 GNN Layers

Layer 1:

Sample

25 Neighbors

Layer 2:

Each sampled neighbor again samples

10 Neighbors

Instead of visiting:

Thousands

↓

Only

25 × 10

\=

250 Nodes

Huge computational savings.

---

# **GraphSAGE Pipeline**

Target Node  
      ↓  
Sample Neighbors  
      ↓  
Aggregate Features  
      ↓  
Combine with Own Features  
      ↓  
Updated Embedding

---

# **Advantages of GraphSAGE**

* Works on massive graphs  
* Supports mini-batches  
* Memory efficient  
* Inductive learning  
* Faster training

---

# **Limitations**

Neighbor sampling introduces approximation.

Some important neighbors may not be sampled.

---

# **Cluster-GCN**

GraphSAGE samples **neighbors**.

Cluster-GCN samples **subgraphs (clusters)**.

---

# **Intuition**

Large graphs naturally contain communities.

Example:

Community A

Community B

Community C

Instead of training on the entire graph,

Cluster-GCN trains on one cluster at a time.

---

# **Step 1**

Partition graph.

Algorithms such as:

* METIS

produce:

Large Graph

↓

Cluster 1

Cluster 2

Cluster 3

---

# **Step 2**

Train on one cluster.

Example:

Only Cluster 2

↓

Forward Pass

↓

Backward Pass

---

# **Step 3**

Repeat for remaining clusters.

Eventually,

all clusters are seen.

---

# **Cluster-GCN Pipeline**

Large Graph  
      ↓  
Graph Partitioning  
      ↓  
Clusters  
      ↓  
Mini-batch Training  
      ↓  
GNN  
      ↓  
Prediction

---

# **Why Cluster-GCN Works**

Nodes inside a community are highly connected.

Most useful information remains inside each cluster.

Therefore,

training on clusters preserves graph structure while reducing computation.

---

# **Advantages**

* Very memory efficient  
* Supports huge graphs  
* Fast mini-batch training  
* Better locality than random sampling

---

# **Limitations**

Quality depends on graph partitioning.

Poor clusters may reduce accuracy.

---

# **Scaling Up by Simplifying GNNs**

Instead of changing training,

another idea is:

**Simplify the GNN architecture itself.**

Observation:

Many GNNs repeatedly perform:

Aggregation

↓

Linear Layer

↓

ReLU

↓

Aggregation

↓

Linear Layer

↓

ReLU

Research showed that some nonlinearities contribute less than expected for certain tasks.

---

# **Simplified Graph Convolution (SGC)**

**Simplified Graph Convolution (SGC)** removes unnecessary operations.

Idea:

Instead of repeatedly performing:

Aggregate

↓

Transform

↓

ReLU

SGC performs aggregation first,

then applies one classifier.

---

# **Standard GCN**

Aggregation

↓

Linear Layer

↓

ReLU

↓

Aggregation

↓

Linear Layer

↓

Prediction

---

# **Simplified GCN**

Repeated Aggregation

↓

One Linear Classifier

↓

Prediction

---

# **Intuition**

Suppose we need information from

3-Hop Neighbors

Instead of:

Aggregate

↓

ReLU

↓

Aggregate

↓

ReLU

↓

Aggregate

Simply perform:

Aggregate

3 Times

↓

Classifier

---

# **Advantages of SGC**

* Extremely fast  
* Much lower memory usage  
* Easy implementation  
* Similar accuracy on many benchmark datasets

---

# **Limitations**

Removing nonlinearities reduces model capacity.

Less suitable for highly complex graphs requiring expressive representations.

---

# **Comparison of Scaling Techniques**

| Technique | Main Idea | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| GraphSAGE | Sample neighbors | Simple, scalable, inductive | Approximate neighbors |
| Cluster-GCN | Train on graph clusters | Efficient mini-batching, preserves locality | Depends on partition quality |
| SGC (Simplified GCN) | Remove intermediate transformations and nonlinearities | Very fast, lightweight | Lower expressiveness |
| FastGCN | Sample graph nodes layer-wise | Faster training | Sampling approximation |
| LADIES | Layer-dependent importance sampling | Better sampling quality | More complex implementation |

---

# **GraphSAGE vs Cluster-GCN**

| Feature | GraphSAGE | Cluster-GCN |
| ----- | ----- | ----- |
| Sampling Unit | Neighbor | Graph Cluster |
| Mini-batching | Yes | Yes |
| Inductive Learning | Yes | Primarily designed for efficient training |
| Memory Usage | Low | Very Low |
| Large Graphs | Excellent | Excellent |

---

# **Standard GCN vs SGC**

| Feature | Standard GCN | Simplified GCN |
| ----- | ----- | ----- |
| Nonlinear Layers | Multiple | Removed |
| Training Speed | Moderate | Very Fast |
| Computational Cost | Higher | Lower |
| Expressiveness | Higher | Lower |
| Suitable For | Complex tasks | Large-scale node classification |

---

# **Complete Scalable GNN Pipeline**

Large Graph  
      ↓  
Choose Scaling Strategy  
      │  
      ├── Neighbor Sampling (GraphSAGE)  
      │  
      ├── Graph Clustering (Cluster-GCN)  
      │  
      ├── Layer Sampling (FastGCN/LADIES)  
      │  
      └── Simplified GCN (SGC)  
              ↓  
Mini-Batch Training  
              ↓  
Message Passing  
              ↓  
Forward Propagation  
              ↓  
Backpropagation  
              ↓  
Prediction

---

# **Interview Summary**

Scaling Graph Neural Networks (GNNs) to large graphs is challenging because of the **neighborhood explosion problem**, where the number of nodes involved in message passing grows exponentially with the number of layers. **GraphSAGE** addresses this by using **neighbor sampling**, where each node aggregates information from only a fixed number of randomly sampled neighbors, enabling scalable mini-batch and inductive learning. **Cluster-GCN** takes a different approach by partitioning the graph into **clusters (communities)** and training on one cluster at a time, preserving local graph structure while greatly reducing memory usage. Another strategy is **simplifying the GNN architecture** itself. **Simplified Graph Convolution (SGC)** removes intermediate nonlinearities and weight matrices between propagation steps, performing repeated feature propagation followed by a single linear classifier. This significantly speeds up training while maintaining competitive accuracy on many node classification tasks. Other scalable methods include **FastGCN** and **LADIES**, which perform layer-wise node sampling. Together, these approaches make GNNs practical for graphs with millions of nodes and billions of edges.

