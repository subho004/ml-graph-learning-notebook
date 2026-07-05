# graph ml 5

# **Message Passing in Graph Neural Networks (GNN)**

**Message Passing** is the core mechanism of Graph Neural Networks.

The main idea is:

Every node learns by collecting information (messages) from its neighboring nodes.

Instead of making predictions using only its own features, a node also uses information from connected nodes.

---

# **Intuition**

Suppose we have a social network:

     Bob  
       |  
Alice--John--David  
       |  
     Emily

Suppose we want to classify **John**.

Using only John's information:

Age  
Profession  
Location

may not be enough.

Instead, GNN also considers:

Alice  
Bob  
David  
Emily

The neighbors send information to John.

John combines:

Own Features

\+

Neighbor Features

to produce a better representation.

---

# **Message Passing Process**

Each GNN layer performs three steps:

Neighbor Nodes  
      ↓  
Message Generation  
      ↓  
Aggregation  
      ↓  
Node Update

---

## **Step 1: Message Generation**

Each neighboring node creates a message.

Example:

Alice → John's Message

Bob → John's Message

Emily → John's Message

Usually, the message is simply the node's embedding.

---

## **Step 2: Aggregation**

Combine all incoming messages.

Common aggregation methods:

### **Sum**

m \= Alice \+ Bob \+ Emily

---

### **Mean**

m \= Average(Messages)

---

### **Max**

Take the maximum value from each feature.

---

### **Attention**

More important neighbors receive larger weights.

Used in:

Graph Attention Networks (GAT)

---

## **Step 3: Node Update**

Combine:

Current Node Features

\+

Aggregated Messages

to compute a new node embedding.

Example:

Old John Embedding

↓

Combine Neighbor Information

↓

New John Embedding

After several layers:

John indirectly learns from:

Friends

Friends of Friends

Entire Neighborhood

---

# **Message Passing Formula**

One GNN layer updates node (v) as:

\[  
h\_v^{(k+1)}=  
Update  
\\left(  
h\_v^{(k)},  
Aggregate({h\_u^{(k)}:u\\in N(v)})  
\\right)  
\]

Where:

* (h\_v) \= Node embedding  
* (N(v)) \= Neighbor set  
* Aggregate \= Mean, Sum, Max, Attention

---

# **Why Multiple Message Passing Layers?**

Layer 1:

Learn from Immediate Neighbors

Layer 2:

Learn from Neighbors of Neighbors

Layer 3:

Learn from Larger Neighborhood

---

# **Node Classification**

Node Classification predicts labels for individual nodes.

Input:

Graph

\+

Node Features

Output:

Label for each Node

---

# **Examples**

### **Social Network**

Predict:

User Interests

---

### **Banking**

Predict:

Fraudulent Account

---

### **Citation Network**

Predict:

Research Topic

---

### **Healthcare**

Predict:

Disease Gene

---

# **Workflow of Node Classification**

Graph  
      ↓  
Message Passing  
      ↓  
Node Embeddings  
      ↓  
Classifier  
      ↓  
Node Labels

---

# **Relational Classification**

Relational Classification predicts node labels by considering both:

* Node features  
* Neighbor labels

Instead of assuming samples are independent,

it exploits relationships.

---

# **Intuition**

Suppose we want to classify a student.

Known:

Student A

↓

Friends with

↓

Five Computer Science Students

Even if A's features are incomplete,

neighbors strongly suggest:

A probably belongs to Computer Science.

---

# **Example**

Email Network

Nodes:

Employees

Edges:

Email Communication

Task:

Predict department.

Employees communicating frequently usually belong to the same department.

---

# **Collective Classification**

Collective Classification classifies **multiple connected nodes simultaneously**.

Instead of classifying each node independently,

the labels influence one another.

---

# **Intuition**

Suppose:

A

↓

Friend

↓

B

↓

Friend

↓

C

Initially:

Unknown Labels

As the algorithm predicts A,

it improves prediction for B,

which improves prediction for C.

Predictions evolve together.

---

# **Why Collective Classification?**

Traditional ML assumes:

Samples Independent

Graphs violate this assumption.

Connected nodes are often correlated.

Collective Classification leverages these dependencies.

---

# **Iterative Classification Algorithm (ICA)**

ICA is a simple algorithm for Collective Classification.

Instead of predicting once,

it repeatedly updates predictions.

---

# **ICA Steps**

### **Step 1**

Predict labels using only node features.

Example:

A → Student

B → Unknown

C → Unknown

---

### **Step 2**

Use predicted neighbor labels.

Example:

If most neighbors are:

Computer Science

increase probability for Computer Science.

---

### **Step 3**

Update predictions.

---

### **Step 4**

Repeat until labels stop changing.

---

# **ICA Workflow**

Initial Prediction  
        ↓  
Neighbor Labels  
        ↓  
Update Prediction  
        ↓  
Repeat  
        ↓  
Convergence

---

# **Belief Propagation**

Belief Propagation is another inference algorithm for graphical models.

Instead of exchanging embeddings,

nodes exchange **probabilities (beliefs)**.

---

# **Intuition**

Suppose:

A

↓

B

↓

C

A believes:

90% Fraud

This belief influences B.

Then B updates its belief and influences C.

Eventually,

beliefs stabilize.

---

# **Belief Propagation Process**

Each node maintains:

Probability Distribution

Example:

Node A

Fraud \= 0.8

Normal \= 0.2

Nodes repeatedly send beliefs to neighbors.

Neighbors update their own beliefs.

Repeat until convergence.

---

# **Message Passing vs Belief Propagation**

| Message Passing | Belief Propagation |
| ----- | ----- |
| Exchanges embeddings | Exchanges probabilities (beliefs) |
| Used in GNNs | Used in Probabilistic Graphical Models |
| Learns node representations | Performs probabilistic inference |
| Neural network training | Bayesian inference |

---

# **Relational vs Collective Classification**

| Relational Classification | Collective Classification |
| ----- | ----- |
| Uses neighbor information | Uses neighbor predictions jointly |
| Predicts one node at a time | Predicts all nodes together |
| Simpler | More accurate but computationally heavier |

---

# **Node Classification Pipeline in GNN**

Graph  
      ↓  
Initialize Node Features  
      ↓  
Message Passing  
      ↓  
Aggregate Neighbor Information  
      ↓  
Updated Node Embeddings  
      ↓  
Node Classifier  
      ↓  
Predicted Node Labels

---

# **Real-World Applications**

### **Fraud Detection**

Node:

Bank Account

Prediction:

Fraud / Normal

Neighbors:

Transaction Partners

---

### **Social Networks**

Node:

User

Prediction:

Community

Interest

Fake Account

---

### **Citation Networks**

Node:

Research Paper

Prediction:

Research Area

---

### **Healthcare**

Node:

Gene

Prediction:

Disease Association

---

# **Comparison of Methods**

| Method | Main Idea | Uses |
| ----- | ----- | ----- |
| Message Passing | Aggregate neighbor embeddings | GNN Training |
| Node Classification | Predict node labels | Fraud Detection, Recommendation |
| Relational Classification | Use node features \+ neighbor information | Social Networks |
| Collective Classification | Predict labels jointly | Citation Networks |
| Iterative Classification (ICA) | Repeatedly update predictions | Collective Classification |
| Belief Propagation | Exchange probabilities between nodes | Bayesian Networks, Error Correction |

---

# **Interview Summary**

**Message Passing** is the fundamental operation in Graph Neural Networks, where each node gathers information from its neighbors, aggregates it (using mean, sum, max, or attention), and updates its own embedding. This process enables nodes to learn from both their own features and the graph structure. **Node Classification** uses these learned embeddings to predict labels for individual nodes, such as fraud detection or user classification. **Relational Classification** improves predictions by incorporating relationships between connected nodes, while **Collective Classification** jointly predicts labels for multiple interconnected nodes, allowing predictions to reinforce each other. A common approach is the **Iterative Classification Algorithm (ICA)**, which repeatedly updates node labels based on neighbors until convergence. **Belief Propagation**, in contrast, is a probabilistic inference algorithm where nodes exchange probability distributions (beliefs) instead of embeddings, iteratively refining their estimates until they stabilize. Together, these methods exploit the relational structure of graph data to achieve more accurate predictions than treating nodes independently.

