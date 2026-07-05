# graph ml 2

# **Node-Level, Edge-Level, and Graph-Level Features & Predictions in GNN**

One of the key strengths of **Graph Neural Networks (GNNs)** is that they can learn representations (embeddings) at **three different levels**:

1. **Node-Level**  
2. **Edge (Link)-Level**  
3. **Graph-Level**

Each level has its own **features**, **embeddings**, and **prediction tasks**.

---

# **Overview**

| Level | What is Learned? | Prediction Target |
| ----- | ----- | ----- |
| Node Level | Representation of each node | Predict node properties |
| Edge (Link) Level | Representation of relationships | Predict edge existence or attributes |
| Graph Level | Representation of the entire graph | Predict graph properties |

---

# **1\. Node-Level Features**

Node features describe the characteristics of an individual node.

Every node has its own feature vector.

Example:

### **Social Network**

Node \= User

Features:

Age

Gender

Location

Followers

Interests

---

### **Banking Network**

Node \= Bank Account

Features:

Account Age

Balance

Number of Transactions

Account Type

---

### **Molecular Graph**

Node \= Atom

Features:

Atomic Number

Charge

Hybridization

Atomic Mass

---

# **Node Embedding**

During message passing, each node learns information from its neighbors.

After training:

Node  
↓

Dense Vector (Embedding)

↓

Prediction

Example:

User A

↓

\[0.34, 1.25, \-0.81, ...\]

This vector captures:

* Node features  
* Neighbor information  
* Graph structure

---

# **Node-Level Prediction**

Goal:

Predict a property of an individual node.

---

## **Examples**

### **Social Networks**

Predict:

User Interests

---

### **Fraud Detection**

Predict:

Fraudulent Account?

---

### **Citation Networks**

Predict:

Research Paper Topic

---

### **Recommendation Systems**

Predict:

Customer Preference

---

# **Examples of Node-Level Tasks**

* Node Classification  
* Node Regression  
* Node Clustering

---

# **2\. Edge-Level (Link-Level) Features**

Edges also contain useful information.

Example:

### **Facebook**

Edge:

Friendship

Feature:

Friend Since \= 2018

---

### **Banking**

Edge:

Money Transfer

Features:

Amount

Date

Currency

Transaction Type

---

### **Transportation**

Edge:

Road

Features:

Distance

Travel Time

Traffic

---

# **Edge Embedding**

An edge can also have its own embedding.

Example:

(Account A)

↓

(Transaction)

↓

(Account B)

The embedding represents:

* Relationship strength  
* Interaction frequency  
* Similarity  
* Transaction history

---

# **Edge-Level Prediction**

Goal:

Predict information about relationships.

---

## **Examples**

### **Friend Recommendation**

Predict:

Will User A become friends with User B?

---

### **LinkedIn**

Predict:

Connection Recommendation

---

### **Drug Discovery**

Predict:

Will Drug A interact with Protein B?

---

### **Banking**

Predict:

Will Account A send money to Account B?

---

# **Examples of Edge-Level Tasks**

* Link Prediction  
* Edge Classification  
* Edge Regression

---

# **3\. Graph-Level Features**

Sometimes the entire graph represents one object.

Instead of learning individual nodes,

learn one embedding for the entire graph.

---

### **Example 1**

One molecule

Entire graph:

Atoms  
\+  
Chemical Bonds

↓

Predict:

Toxic or Non-Toxic

---

### **Example 2**

Social Community

Entire graph:

Users  
Connections

↓

Predict:

Community Type

---

### **Example 3**

Protein Structure

Entire graph:

Protein Network

↓

Predict:

Disease Related?

---

# **Graph Embedding**

All node embeddings are combined using:

Pooling

Readout

Aggregation

to create one vector.

Graph

↓

Graph Embedding

↓

Prediction

---

# **Graph-Level Prediction**

Goal:

Predict a property of the whole graph.

---

## **Examples**

### **Chemistry**

Predict:

Molecule Toxicity

---

### **Drug Discovery**

Predict:

Drug Effectiveness

---

### **Social Networks**

Predict:

Community Category

---

### **Cybersecurity**

Predict:

Network Under Attack?

---

# **Graph Readout (Pooling)**

To obtain graph-level embeddings,

node embeddings are aggregated.

Common methods:

---

## **Mean Pooling**

Take average.

Graph Embedding

\=

Average(Node Embeddings)

---

## **Sum Pooling**

Add all embeddings.

---

## **Max Pooling**

Take maximum value from each dimension.

---

## **Attention Pooling**

More important nodes receive higher weights.

---

# **Comparison of Prediction Types**

| Prediction Type | Input | Output | Example |
| ----- | ----- | ----- | ----- |
| Node Prediction | One Node | Node Label | Fraudulent Account? |
| Edge Prediction | Two Nodes \+ Edge | Edge Label | Friend Recommendation |
| Graph Prediction | Entire Graph | Graph Label | Molecule Toxicity |

---

# **Real-World Examples**

---

## **Node-Level**

### **Facebook**

Input:

User Node

Prediction:

Political Interest

---

### **Banking**

Input:

Account Node

Prediction:

Fraud Score

---

## **Edge-Level**

### **LinkedIn**

Input:

User A

User B

Prediction:

Should Connect?

---

### **Amazon**

Input:

Customer

Product

Prediction:

Will Purchase?

---

## **Graph-Level**

### **Chemistry**

Input:

Entire Molecule

Prediction:

Toxic?

---

### **Cybersecurity**

Input:

Computer Network

Prediction:

Malicious Network?

---

# **Feature Summary**

| Level | Example Entity | Features |
| ----- | ----- | ----- |
| Node | User | Age, Salary, Interests |
| Edge | Transaction | Amount, Date, Distance |
| Graph | Molecule | Entire Structure |

---

# **Embedding Summary**

| Embedding | Represents | Used For |
| ----- | ----- | ----- |
| Node Embedding | Individual Node | Node Classification |
| Edge Embedding | Relationship | Link Prediction |
| Graph Embedding | Entire Graph | Graph Classification |

---

# **Interview Summary**

Graph Neural Networks learn representations at **three levels**:

1. **Node-Level**, where each node has its own features (e.g., age, balance, atomic number) and the model predicts properties of individual nodes such as **fraud detection, user classification, or paper topic classification**.  
2. **Edge-Level (Link-Level)**, where edges represent relationships with features like **transaction amount, friendship duration, or road distance**. GNNs perform **link prediction, edge classification, and edge regression**, such as recommending friends or predicting future transactions.  
3. **Graph-Level**, where the entire graph represents a single object, such as a **molecule, protein, or social community**. Node embeddings are aggregated using **graph pooling/readout** techniques (mean, sum, max, or attention pooling) to produce a **graph embedding**, which is used for tasks like **molecule toxicity prediction, drug discovery, protein classification, and network anomaly detection**. These three levels of representation make GNNs highly versatile for learning from complex relational data.

