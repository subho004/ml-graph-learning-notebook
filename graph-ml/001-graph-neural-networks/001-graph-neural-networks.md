# Graph ml 1

# **Graph Neural Networks (GNN)**

A **Graph Neural Network (GNN)** is a Deep Learning model specifically designed to work with **graph-structured data**.

Unlike CNNs (for images) and RNNs (for sequences), GNNs learn from:

* Nodes (Vertices)  
* Edges (Connections)  
* Graph Structure

They capture both **feature information** and **relationships** between entities.

---

# **Why Do We Need GNNs?**

Many real-world datasets are naturally represented as graphs.

Examples:

Facebook

Users connected by friendships.

Google Maps

Cities connected by roads.

Molecules

Atoms connected by chemical bonds.

Bank Transactions

Accounts connected by money transfers.

Traditional ML treats every sample independently.

GNNs learn using:

Node Features  
\+  
Neighborhood Information  
\+  
Graph Structure

---

# **What is a Graph?**

A graph is represented as:

\[  
G=(V,E)  
\]

Where:

* **V** \= Set of Nodes (Vertices)  
* **E** \= Set of Edges (Connections)

Example:

A \----- B  
|       |  
|       |  
C \----- D

Nodes:

A B C D

Edges:

AB AC BD CD

---

# **Components of a Graph**

## **Node (Vertex)**

Represents an entity.

Examples:

* Person  
* Bank Account  
* Molecule Atom  
* Product

---

## **Edge**

Represents a relationship.

Examples:

* Friendship  
* Money Transfer  
* Chemical Bond  
* Road

---

## **Node Features**

Each node may contain attributes.

Example:

Person:

Age  
Salary  
City

---

## **Edge Features**

Each edge may contain information.

Example:

Bank Transfer:

Amount  
Date  
Currency

---

# **Types of Graphs**

---

# **1\. Directed Graph**

Edges have direction.

Example:

A → B

Meaning:

A follows B

Examples:

* Twitter  
* Citation Networks  
* Bank Transactions

---

# **2\. Undirected Graph**

Edges have no direction.

Example:

A \----- B

Examples:

* Facebook Friendship  
* Collaboration Networks

---

# **3\. Weighted Graph**

Edges have weights.

Example:

A \----5---- B

Weight may represent:

* Distance  
* Cost  
* Similarity  
* Transaction Amount

---

# **4\. Unweighted Graph**

Every edge has equal importance.

Example:

A \----- B

---

# **5\. Bipartite Graph**

Nodes belong to two different sets.

Connections occur only across sets.

Example:

Students \---- Courses

No:

Student → Student

Course → Course

Applications:

* Recommendation Systems  
* Job Matching

---

# **6\. Cyclic Graph**

Contains cycles.

Example:

A → B

↑   ↓

D ← C

---

# **7\. Acyclic Graph**

Contains no cycles.

Example:

Decision Trees.

---

# **Connectivity**

Connectivity determines whether nodes are reachable from one another.

---

## **Connected Graph**

Every node can reach every other node.

Example:

A \----- B  
|       |  
C \----- D

---

## **Disconnected Graph**

Contains isolated groups.

Example:

A \----- B

C \----- D

---

# **Connected Components**

A **Connected Component** is a group of nodes where every node is reachable from every other node.

Example:

A \---- B

C \---- D \---- E

F

Connected Components:

{A,B}

{C,D,E}

{F}

---

# **Adjacency Matrix**

A graph can be represented using an **Adjacency Matrix**.

Suppose:

A \---- B

A \---- C

B \---- C

Matrix:

|  | A | B | C |
| ----- | ----- | ----- | ----- |
| A | 0 | 1 | 1 |
| B | 1 | 0 | 1 |
| C | 1 | 1 | 0 |

Meaning:

1 → Edge Exists

0 → No Edge

---

# **Adjacency List**

Alternative representation.

Example:

A → B,C

B → A,C

C → A,B

Memory efficient for sparse graphs.

---

# **Intuition Behind GNN**

Suppose we want to predict whether a user is interested in a product.

Instead of only using:

User Age

Salary

GNN also considers:

Friends

Purchases

Interactions

The prediction depends on both the node and its neighbors.

---

# **How GNN Works**

The key idea is:

A node learns from its neighbors.

This process is called **Message Passing** or **Neighborhood Aggregation**.

---

# **Message Passing**

Suppose:

     B  
      |  
A \---- C \---- D

To update node **C**:

Step 1:

Receive information from:

A

B

D

Step 2:

Aggregate neighbor information.

Step 3:

Combine with C's own features.

Step 4:

Generate updated representation.

---

# **Graph Representation (Embeddings)**

After several message-passing layers, each node is represented by a dense vector called a **Node Embedding**.

Similar nodes have similar embeddings.

---

# **Types of GNN Prediction Tasks**

---

# **1\. Node-Level Prediction**

Predict properties of individual nodes.

Example:

Facebook:

Predict User Interests

Fraud Detection:

Fraudulent Bank Account?

Examples:

* Node Classification  
* Node Regression

---

# **2\. Edge-Level Prediction**

Predict whether an edge exists or estimate an edge property.

Examples:

Friend Recommendation:

Will A connect to B?

Knowledge Graph Completion.

Drug Interaction Prediction.

---

# **3\. Graph-Level Prediction**

Predict a label for the entire graph.

Example:

Molecule:

Will this molecule cure cancer?

Protein Classification.

Document Graph Classification.

---

# **Types of Graph Representations (Embeddings)**

---

## **Node Embedding**

Each node gets its own vector.

Used for:

* Node Classification  
* Clustering

---

## **Edge Embedding**

Each relationship gets a vector.

Used for:

* Link Prediction  
* Recommendation

---

## **Graph Embedding**

Entire graph represented as one vector.

Used for:

* Graph Classification  
* Molecule Prediction

---

# **Common GNN Architectures**

---

## **Graph Convolutional Network (GCN)**

Most popular GNN.

Idea:

Similar to CNN convolution.

Instead of neighboring pixels:

Neighboring Nodes

are aggregated.

---

## **GraphSAGE**

Learns by sampling neighbors.

Useful for:

Large graphs.

---

## **Graph Attention Network (GAT)**

Uses attention mechanism.

Important neighbors receive higher weights.

---

## **Graph Isomorphism Network (GIN)**

Powerful for graph classification.

---

# **Applications of GNN**

---

## **Social Networks**

* Friend Recommendation  
* Community Detection  
* User Classification

---

## **Recommendation Systems**

Amazon:

User → Product

Netflix:

Movie Recommendation.

---

## **Fraud Detection**

Banks detect suspicious transaction networks.

---

## **Drug Discovery**

Predict molecular properties.

---

## **Knowledge Graphs**

Google Search.

Question Answering.

---

## **Traffic Networks**

Road prediction.

Navigation.

---

## **Cybersecurity**

Attack graph analysis.

---

## **Supply Chain Networks**

Route optimization.

---

# **Advantages of GNN**

* Naturally models relationships  
* Captures graph structure  
* Works on irregular data  
* State-of-the-art for graph problems

---

# **Disadvantages**

* Computationally expensive for very large graphs  
* Training can be slow  
* Oversmoothing in deep GNNs (node representations become too similar)  
* Difficult to scale to billion-node graphs

---

# **CNN vs RNN vs GNN**

| Feature | CNN | RNN | GNN |
| ----- | ----- | ----- | ----- |
| Data Type | Images | Sequences | Graphs |
| Input Structure | Grid | Ordered Sequence | Nodes & Edges |
| Learns From | Nearby Pixels | Previous Time Steps | Neighboring Nodes |
| Examples | Image Classification | Language Translation | Social Networks |

---

# **GNN Workflow**

Graph Data  
(Nodes \+ Edges)  
        ↓  
Initialize Node Features  
        ↓  
Message Passing  
        ↓  
Neighbor Aggregation  
        ↓  
Update Node Embeddings  
        ↓  
Repeat for Multiple Layers  
        ↓  
Node / Edge / Graph Embeddings  
        ↓  
Prediction

---

# **Interview Summary**

A **Graph Neural Network (GNN)** is a deep learning model designed for **graph-structured data**, where information is represented as **nodes (entities)** and **edges (relationships)**. Unlike traditional neural networks, GNNs learn by **aggregating information from neighboring nodes** through a process called **message passing**, producing rich node, edge, or graph embeddings. Graphs can be **directed, undirected, weighted, unweighted, bipartite, cyclic, or acyclic**, and can be represented using **adjacency matrices** or **adjacency lists**. GNNs support **node-level tasks** (e.g., node classification), **edge-level tasks** (e.g., link prediction), and **graph-level tasks** (e.g., molecule classification). Popular GNN architectures include **GCN**, **GraphSAGE**, **Graph Attention Networks (GAT)**, and **Graph Isomorphism Networks (GIN)**. They are widely used in **social networks, fraud detection, recommendation systems, knowledge graphs, drug discovery, cybersecurity, and traffic network analysis**.

