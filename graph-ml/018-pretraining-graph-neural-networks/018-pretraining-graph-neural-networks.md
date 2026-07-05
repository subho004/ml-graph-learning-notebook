# graph ml 18

# **Pre-Training Graph Neural Networks (Pre-trained GNNs)**

Training Graph Neural Networks (GNNs) from scratch requires **large amounts of labeled graph data**, which is often expensive to obtain.

Inspired by NLP (BERT) and Computer Vision (ImageNet pretraining), researchers introduced **Pre-trained Graph Neural Networks (Pre-trained GNNs)**.

The idea is:

**First learn general graph representations from large unlabeled graphs, then fine-tune on a specific downstream task.**

---

# **Why Pre-train GNNs?**

Many graph datasets have:

* Very few labeled nodes  
* Limited labeled graphs  
* Expensive annotations

Examples:

* Drug molecules  
* Protein networks  
* Knowledge graphs

Instead of learning from scratch,

the model first learns:

General Graph Knowledge

then transfers it.

---

# **Analogy**

Just as:

BERT

↓

Learns Language

↓

Fine-tuned for QA

Similarly,

Pre-trained GNN

↓

Learns Graph Structure

↓

Fine-tuned for Node Classification

---

# **Pre-training Pipeline**

Large Unlabeled Graph Dataset  
            ↓  
Self-Supervised Learning  
            ↓  
Pre-trained GNN  
            ↓  
Fine-Tuning  
            ↓  
Task-Specific Model

---

# **Types of Pre-training Tasks**

---

## **1\. Node-Level Pre-training**

Learn node embeddings.

Example tasks:

* Predict masked node attributes  
* Contrastive learning  
* Context prediction

---

## **2\. Edge-Level Pre-training**

Predict:

Missing Edges

Useful for:

* Link prediction  
* Recommendation

---

## **3\. Graph-Level Pre-training**

Learn embeddings for entire graphs.

Used in:

* Molecule classification  
* Protein prediction

---

# **Self-Supervised Learning**

No labels are required.

Instead,

the graph itself provides supervision.

Popular methods:

* Masked node prediction  
* Contrastive learning  
* Graph reconstruction  
* Edge prediction

---

# **Fine-Tuning**

After pre-training,

replace the final prediction layer.

Train only on the target dataset.

Benefits:

* Faster convergence  
* Better accuracy  
* Requires fewer labeled examples

---

# **Advantages**

* Better generalization  
* Reduced labeling cost  
* Faster training  
* Improved transfer learning

---

# **Applications**

* Drug discovery  
* Protein prediction  
* Fraud detection  
* Recommendation systems  
* Knowledge graphs

---

# **Hyperbolic Graph Embeddings**

Traditional graph embeddings represent nodes in **Euclidean space**.

However,

many real-world graphs have **hierarchical structures**.

Examples:

Company Hierarchy

↓

CEO

↓

Managers

↓

Employees

Euclidean embeddings struggle to represent such hierarchies efficiently.

---

# **Hyperbolic Space**

Hyperbolic space is a **non-Euclidean geometry** with **negative curvature**.

Unlike a flat plane,

it expands exponentially as you move away from the center.

---

# **Intuition**

Imagine:

Flat paper:

Euclidean Space

versus

A continuously expanding surface:

Hyperbolic Space

The hyperbolic space has much more "room" for representing branching structures.

---

# **Why Hyperbolic Space?**

Tree-like graphs grow exponentially.

Example:

CEO

↓

2 Managers

↓

4 Employees

↓

8 Interns

↓

16 ...

Hyperbolic geometry naturally models this exponential growth.

---

# **Euclidean vs Hyperbolic Embeddings**

Euclidean:

Root

↓

Children

↓

Crowded

Hyperbolic:

Root

↓

Children

↓

Plenty of Space

---

# **Advantages**

* Better represents trees  
* Preserves hierarchical distances  
* Requires lower embedding dimensions  
* Improves hierarchical classification

---

# **Applications**

* Knowledge Graphs  
* Taxonomies  
* Organizational Charts  
* Protein Hierarchies  
* Recommendation Systems

---

# **Limitations**

* More difficult optimization  
* Complex distance computations  
* Higher implementation complexity

---

# **Design Space of GNNs**

The **Design Space of GNNs** refers to all the architectural choices involved in building a Graph Neural Network.

A GNN is composed of several independent design components.

Choosing different options leads to different GNN architectures.

---

# **General GNN Architecture**

Input Graph  
      ↓  
Message Construction  
      ↓  
Message Aggregation  
      ↓  
Node Update  
      ↓  
Readout  
      ↓  
Prediction

Each stage has multiple design choices.

---

# **1\. Message Construction**

Defines **what information is sent** from neighboring nodes.

Possible inputs:

* Node features  
* Edge features  
* Positional encodings  
* Structural features

---

# **2\. Aggregation Function**

Determines **how messages from neighbors are combined**.

Common methods:

Sum

Mean

Max

Attention

Examples:

* GCN → Mean/normalized sum  
* GraphSAGE → Mean, Max, LSTM  
* GAT → Attention  
* GIN → Sum

---

# **3\. Update Function**

Updates the node embedding after aggregation.

Common choices:

* MLP  
* Linear layer  
* GRU/LSTM  
* Residual updates

---

# **4\. Number of Layers**

Each GNN layer aggregates information from one additional hop.

* 2 layers → 2-hop neighborhood  
* 3 layers → 3-hop neighborhood

Too many layers may cause:

* Over-smoothing  
* Over-squashing

---

# **5\. Skip Connections**

Residual or skip connections help preserve original node information.

Example:

Input  
   ↓  
Layer 1  
   ↓  
Layer 2  
   ↓  
Output

↘────────────↗

Benefits:

* Easier optimization  
* Reduces over-smoothing  
* Enables deeper GNNs

---

# **6\. Readout Function**

Combines node embeddings into a graph embedding.

Methods:

* Sum Pooling  
* Mean Pooling  
* Max Pooling  
* Attention Pooling

Used for graph-level prediction.

---

# **7\. Loss Function**

Depends on the task.

Examples:

* Cross Entropy (Classification)  
* MSE (Regression)  
* Contrastive Loss (Self-supervised)  
* Ranking Loss (Link Prediction)

---

# **8\. Training Strategy**

Possible choices:

* Full-batch training  
* Mini-batch training  
* Neighbor sampling  
* Cluster-GCN  
* GraphSAINT

---

# **9\. Positional and Structural Encodings**

Modern GNNs often include:

* Laplacian eigenvectors  
* Random walk encodings  
* Degree encodings  
* Centrality measures

These improve expressiveness.

---

# **10\. Graph Type**

Design also depends on graph characteristics:

* Homogeneous graphs  
* Heterogeneous graphs  
* Dynamic graphs  
* Knowledge graphs

Different architectures are appropriate for different graph types.

---

# **Popular GNN Designs**

| Model | Aggregation | Special Feature |
| ----- | ----- | ----- |
| GCN | Mean/Normalized Sum | Simple and efficient |
| GraphSAGE | Mean/Max/LSTM | Neighbor sampling |
| GAT | Attention | Learns neighbor importance |
| GIN | Sum | Maximum expressiveness among standard message-passing GNNs |
| R-GCN | Relation-specific aggregation | Heterogeneous graphs |
| Graph Transformer | Global attention | Long-range dependencies |

---

# **Complete Design Workflow**

Input Graph  
      ↓  
Choose Graph Type  
      ↓  
Message Function  
      ↓  
Aggregation Method  
      ↓  
Update Function  
      ↓  
Number of Layers  
      ↓  
Readout  
      ↓  
Loss Function  
      ↓  
Training Strategy  
      ↓  
Prediction

---

# **Interview Summary**

**Pre-trained Graph Neural Networks (Pre-trained GNNs)** follow the same philosophy as BERT in NLP and ImageNet pretraining in computer vision: a GNN is first trained on **large unlabeled graph datasets using self-supervised objectives** such as masked node prediction, contrastive learning, edge prediction, or graph reconstruction, and is then **fine-tuned** for downstream tasks like node classification, graph classification, or link prediction. This improves accuracy and reduces the need for labeled data. **Hyperbolic Graph Embeddings** represent nodes in **hyperbolic (negatively curved) space** instead of Euclidean space, making them particularly effective for **hierarchical and tree-like graphs** because hyperbolic space naturally accommodates exponential branching while using lower-dimensional embeddings. The **Design Space of GNNs** refers to the collection of architectural choices involved in building a GNN, including **message construction, aggregation function (sum, mean, max, attention), update function, number of layers, skip connections, readout function, loss function, training strategy, positional encodings, and graph type**. Different combinations of these components lead to architectures such as **GCN, GraphSAGE, GAT, GIN, R-GCN, and Graph Transformers**, each optimized for different graph learning scenarios.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAloAAADtCAYAAABnC9sOAAAdM0lEQVR4Xu3da4xdZb3HcaK+8IWJMSb6QhNTeXEinV5G22nLgECddiwWIpRKLPYUOVXAE8AQMFH0NFwCyeHaRE7wEhIJEY8oNWkEIqQXSIlgRWoioWiCUAikVS4dJTElXSfPOnk2a37/teey9vPfe621vzv5ZO/5rbWmMy37v388a82eE0444YQMAAAALkwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEzgpuym+3g7fvx4dvHFF5scAADAgQncDKJYKYoWAADoIxO4malohQIUvfDCC9Py73//+/l9/PiRRx7p7KvHFo/Tzx/vY9Hatm1b6bE7duzILr300mnHAwAAVGACN6HMTExMdCxatCjPn3nmmWzDhg2d/Q4ePJitXLmyc8z73//+aZ9DP2d8vG/fvuwDH/iAyYsfF4vWhRde2Nm+cOHC7Dvf+U7+eNeuXdnVV1897XgAAIAKTOCmuHoU3H///Z28uN9JJ52Ul6aybTN9HFaoVqxYYfLix8WipR588EGTAQAA9MAEbrT8dMvf8573ZC+++GLptpk+nm/RCo/vvvvu7Pzzz8/Gxsayhx56aNoxAAAAPTKBGy0/0Z///OdsfHy88/ETTzyRn1osO2amj7sVrU984hOlReuFwrVg119/PUULAACkZgI3WpJ0W/Tmm292PWamj4tF69ChQ53Pd/nll5cWreKf+aEPfahz6jB8/M4770z7cwAAACowAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwQd+deOKJ2UknnQQA83bKKadkn/70pwFgmjAftG8MiAn6RgcmAMwXRQvAbLR/9JkJ+kKHJQBUQdECMBfaQ/rIBO50UAJAVRQtAHOlfaRPTOBKhyQA9IKiBWA+tJf0gQncvPe97zVDEgB6QdECMB/ve9/7TD9xZgI3OiABoFcULQDzpf3EmQlcfPSjHzUDEgB6RdECMF8f//jHTU9xZAIXOhwBDK87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>

[image2]: <data:image/png;base64,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>