# graph ml 8

# **GNN Augmentation and Training**

Training a **Graph Neural Network (GNN)** is similar to training other deep learning models but includes additional steps to handle graph structure, neighborhood sampling, and message passing.

The overall pipeline is:

Graph Data  
      ↓  
Graph Augmentation  
      ↓  
Neighbor Sampling  
      ↓  
Forward Propagation  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Gradient Descent  
      ↓  
Model Evaluation

---

# **Why is GNN Training Different?**

Unlike CNNs or MLPs, every node depends on its neighbors.

Example:

Node A

↓

Needs Features of

↓

Neighbors

↓

Neighbors of Neighbors

Therefore, training must efficiently handle graph connectivity.

---

# **GNN Augmentation**

Graph Augmentation refers to modifying the graph during training to improve **generalization**, reduce **overfitting**, and make the model more robust.

It is similar to **image augmentation** in CNNs (rotation, flipping, cropping), but for graph data.

---

# **Why Augment Graphs?**

Real-world graphs may contain:

* Missing edges  
* Noisy connections  
* Incomplete node features  
* Small labeled datasets

Augmentation helps the model learn more robust representations.

---

# **Types of Graph Augmentation**

---

## **1\. Edge Dropping**

Randomly remove some edges.

Example:

Before:

A \----- B \----- C

After:

A     B \----- C

Benefits:

* Prevents over-reliance on specific edges.  
* Reduces overfitting.

---

## **2\. Node Dropping**

Randomly remove nodes during training.

Useful for:

Large Graphs

Makes the model robust to missing nodes.

---

## **3\. Feature Masking**

Hide some node features.

Example:

Before:

Age \= 25

Salary \= 50K

After:

Age \= ?

Salary \= 50K

The model learns to rely on neighborhood information.

---

## **4\. Subgraph Sampling**

Instead of using the entire graph,

train on smaller subgraphs.

Benefits:

* Faster training  
* Lower memory usage  
* Better scalability

---

## **5\. Edge Addition**

Randomly add edges.

Improves robustness against missing links.

---

# **Neighborhood Sampling**

One of the biggest challenges in GNNs is that some nodes have **thousands or millions of neighbors**.

Example:

Facebook User

↓

10 Million Friends

Aggregating all neighbors is computationally impossible.

Solution:

Neighborhood Sampling

---

# **Intuition**

Instead of using:

1,000,000 Neighbors

Sample:

20

50

100

neighbors.

The sampled neighbors approximate the full neighborhood.

---

# **Types of Neighborhood Sampling**

---

## **1\. Uniform Random Sampling**

Every neighbor has equal probability.

Example:

100 Neighbors

↓

Randomly Select 25

Simple and efficient.

---

## **2\. Fixed-Size Sampling**

Always sample a fixed number.

Example:

25 Neighbors

Used by:

GraphSAGE

---

## **3\. Importance Sampling**

More important neighbors have higher probability.

Importance may depend on:

* Edge weight  
* Degree  
* Attention score

---

## **4\. Layer-wise Sampling**

Instead of sampling neighbors for every node,

sample once for an entire layer.

Used by:

* FastGCN  
* LADIES

---

# **GraphSAGE Neighbor Sampling**

GraphSAGE uses:

Sample

↓

Aggregate

↓

Update

Example:

Node

↓

1000 Neighbors

↓

Sample 20

↓

Aggregate

↓

Update Embedding

Makes GNNs scalable.

---

# **Training a GNN**

Training follows the same basic idea as other neural networks.

---

## **Step 1**

Initialize:

* Node Features  
* Edge Features  
* Weights

---

## **Step 2**

Sample neighbors (if needed).

---

## **Step 3**

Perform Message Passing.

Each node gathers information.

---

## **Step 4**

Aggregate neighbor embeddings.

Example:

Mean

Sum

Max

Attention

---

## **Step 5**

Update node embeddings.

---

## **Step 6**

Forward Propagation.

Generate predictions.

---

## **Step 7**

Compute Loss.

Examples:

Classification:

Cross Entropy Loss

Regression:

Mean Squared Error

---

## **Step 8**

Backpropagation.

Compute gradients.

---

## **Step 9**

Update weights using:

* SGD  
* Adam  
* RMSProp

---

## **Step 10**

Repeat until convergence.

---

# **GNN Training Pipeline**

Input Graph  
      ↓  
Node Features  
      ↓  
Neighbor Sampling  
      ↓  
Message Passing  
      ↓  
Aggregation  
      ↓  
Update Node Embeddings  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Function  
      ↓  
Backpropagation  
      ↓  
Optimizer  
      ↓  
Updated Parameters

---

# **Setting Up GNN Prediction Tasks**

A GNN can solve different types of prediction problems.

---

# **1\. Node-Level Prediction**

Predict properties of individual nodes.

Examples:

* Fraud Detection  
* User Classification  
* Disease Gene Prediction

Input:

Node \+ Neighborhood

Output:

Node Label

---

# **2\. Edge-Level Prediction**

Predict properties of relationships.

Examples:

* Friend Recommendation  
* Link Prediction  
* Drug Interaction Prediction

Input:

Two Nodes

\+

Edge Features

Output:

Edge Label

---

# **3\. Graph-Level Prediction**

Predict a label for the entire graph.

Examples:

* Molecule Classification  
* Protein Function Prediction  
* Malware Detection

Input:

Entire Graph

Output:

Graph Label

---

# **Train, Validation and Test Split**

Like traditional ML,

graph datasets are divided into:

Training Set

Validation Set

Test Set

---

For node classification:

Nodes

↓

Train

Validation

Test

---

For graph classification:

Entire graphs are split.

---

# **Loss Functions**

---

## **Node Classification**

Cross Entropy Loss

---

## **Graph Classification**

Cross Entropy Loss

---

## **Regression**

Mean Squared Error

---

## **Link Prediction**

Binary Cross Entropy

---

# **Optimizers**

Common optimizers:

* SGD  
* Adam (Most Popular)  
* RMSProp

Adam usually converges faster than plain SGD.

---

# **Evaluation Metrics**

Evaluation depends on the task.

---

# **Node Classification**

### **Accuracy**

Most common.

\[  
Accuracy=  
\\frac{Correct}{Total}  
\]

---

### **Precision**

Useful for imbalanced datasets.

\[  
Precision=  
\\frac{TP}{TP+FP}  
\]

---

### **Recall**

Measures ability to detect positives.

\[  
Recall=  
\\frac{TP}{TP+FN}  
\]

---

### **F1 Score**

Balances Precision and Recall.

\[  
F1=  
\\frac{2PR}{P+R}  
\]

---

### **ROC-AUC**

Measures ranking quality.

Useful for binary classification.

---

# **Link Prediction**

Common metrics:

* ROC-AUC  
* Average Precision (AP)  
* Hits@K  
* Mean Reciprocal Rank (MRR)

---

# **Graph Classification**

Common metrics:

* Accuracy  
* Precision  
* Recall  
* F1 Score  
* ROC-AUC

---

# **Regression Tasks**

Metrics:

* Mean Absolute Error (MAE)  
* Mean Squared Error (MSE)  
* Root Mean Squared Error (RMSE)  
* R² Score

---

# **Early Stopping**

To prevent overfitting:

Monitor:

Validation Loss

If it stops improving for several epochs,

stop training.

---

# **Hyperparameters in GNN Training**

Common hyperparameters:

* Number of GNN layers  
* Hidden dimension  
* Learning rate  
* Batch size  
* Number of sampled neighbors  
* Dropout rate  
* Weight decay  
* Number of epochs

---

# **Complete GNN Workflow**

Graph Dataset  
      ↓  
Graph Augmentation  
      ↓  
Train / Validation / Test Split  
      ↓  
Neighbor Sampling  
      ↓  
Message Passing  
      ↓  
Aggregation  
      ↓  
Updated Embeddings  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Optimizer (Adam/SGD)  
      ↓  
Parameter Update  
      ↓  
Repeat Until Convergence  
      ↓  
Evaluation Metrics

---

# **Interview Summary**

**GNN training** extends standard deep learning by incorporating graph structure through **message passing** and **neighbor aggregation**. To improve robustness, **graph augmentation** techniques such as **edge dropping, node dropping, feature masking, subgraph sampling, and edge addition** are used during training. Since large graphs make full-neighborhood aggregation computationally expensive, **neighbor sampling** methods such as **uniform sampling, fixed-size sampling (GraphSAGE), importance sampling, and layer-wise sampling** are employed. A typical GNN training pipeline consists of **graph preprocessing, augmentation, train/validation/test splitting, neighbor sampling, message passing, forward propagation, loss computation, backpropagation, and optimization using SGD or Adam**. GNNs can be configured for **node-level, edge-level, or graph-level prediction tasks**, with evaluation metrics chosen accordingly: **Accuracy, Precision, Recall, F1-score, and ROC-AUC** for classification tasks; **Hits@K** and **MRR** for link prediction; and **MSE, RMSE, MAE, and R²** for regression tasks. Proper augmentation, sampling, optimization, and evaluation are essential for building scalable and accurate graph neural networks.

