# graph ml 10

# **Heterogeneous Graphs**

A **Heterogeneous Graph (Heterograph)** is a graph that contains **multiple types of nodes and/or multiple types of edges (relations)**.

Unlike homogeneous graphs, where all nodes and edges are of the same type, heterogeneous graphs model much richer real-world systems.

---

# **Homogeneous vs Heterogeneous Graph**

### **Homogeneous Graph**

Only one node type and one edge type.

Example:

Person \---- Friend \---- Person

Everything is the same type.

---

### **Heterogeneous Graph**

Multiple node types and edge types.

Example:

User \------ Buys \------ Product  
  |                          |  
Follows                  Belongs To  
  |                          |  
 User \------ Sold By \------ Seller

Node types:

User

Product

Seller

Edge types:

Buys

Follows

Sold By

---

# **Formal Definition**

A heterogeneous graph is represented as

\[  
G=(V,E,T\_V,T\_E)  
\]

Where:

* (V) \= Nodes  
* (E) \= Edges  
* (T\_V) \= Node Types  
* (T\_E) \= Edge Types

---

# **Examples of Heterogeneous Graphs**

## **E-Commerce**

Nodes:

* User  
* Product  
* Seller

Edges:

* Buys  
* Reviews  
* Sells

---

## **Academic Network**

Nodes:

* Author  
* Paper  
* Conference

Edges:

* Writes  
* Published In  
* Cites

---

## **Healthcare**

Nodes:

* Patient  
* Disease  
* Drug  
* Doctor

Edges:

* Diagnosed With  
* Prescribed  
* Treats

---

# **Why Ordinary GNNs Fail**

Standard GCN assumes:

Every edge has the same meaning.

But in a heterogeneous graph:

User \----Buys---- Product

User \----Follows---- User

These relations have different semantics.

Treating them identically loses information.

---

# **Knowledge Graph (KG)**

A **Knowledge Graph** is a special type of heterogeneous graph where information is stored as **entities** connected by **typed relationships**.

Knowledge is represented as **triples**.

---

# **Triple Representation**

Each fact is represented as:

(Head, Relation, Tail)

or

\[  
(h,r,t)  
\]

Example:

(Paris, CapitalOf, France)

(Albert Einstein, BornIn, Germany)

(ChatGPT, DevelopedBy, OpenAI)

---

# **Visual Example**

Einstein  
     |  
   BornIn  
     |  
Germany  
     |  
Capital  
     |  
Berlin

---

# **Applications of Knowledge Graphs**

* Google Search  
* Recommendation Systems  
* Question Answering  
* Drug Discovery  
* Chatbots  
* Knowledge Retrieval

---

# **Knowledge Graph Embeddings (KGE)**

Knowledge Graph Embeddings convert:

Entities

Relations

into dense vectors.

Instead of storing:

Paris

France

store:

Paris → \[0.3,0.7,-1.2\]

France → \[1.4,-0.6,0.8\]

Relations also receive embeddings.

---

# **Why KG Embeddings?**

They enable:

* Link Prediction  
* Missing Fact Prediction  
* Recommendation  
* Reasoning

using vector operations.

---

# **Knowledge Graph Completion**

Knowledge Graphs are often incomplete.

Example:

Known:

(Paris, CapitalOf, France)

Missing:

(France, LocatedIn, Europe)

Goal:

Predict missing triples.

This task is called:

Knowledge Graph Completion (KGC)

---

# **KG Completion Using Embeddings**

Pipeline:

Knowledge Graph  
        ↓  
Learn Entity Embeddings  
        ↓  
Learn Relation Embeddings  
        ↓  
Score Candidate Triples  
        ↓  
Predict Missing Links

---

# **Relational GCN (R-GCN)**

**Relational Graph Convolutional Network (R-GCN)** extends GCN to heterogeneous graphs.

Instead of one weight matrix,

R-GCN learns:

One Weight Matrix

Per Relation Type

---

# **Intuition**

Suppose:

User \--Buys--\> Product

User \--Follows--\> User

Different relations require different transformations.

R-GCN performs:

Buy Relation

↓

Transformation W₁

Follow Relation

↓

Transformation W₂

---

# **R-GCN Workflow**

Node  
     ↓  
Group Neighbors by Relation  
     ↓  
Apply Relation-specific Weights  
     ↓  
Aggregate  
     ↓  
Updated Embedding

---

# **Advantages of R-GCN**

* Handles multiple relation types  
* Preserves relation semantics  
* Excellent for knowledge graphs

---

# **Limitations of R-GCN**

Many relation types require:

Many Weight Matrices

Large KGs become memory intensive.

---

# **Relational GraphSAGE (R-GraphSAGE)**

GraphSAGE samples neighbors.

R-GraphSAGE extends it to heterogeneous graphs.

Pipeline:

Sample Neighbors  
        ↓  
Group by Relation  
        ↓  
Relation-wise Aggregation  
        ↓  
Combine  
        ↓  
Embedding

Benefits:

* Scalable  
* Suitable for very large heterogeneous graphs

---

# **Relational Graph Attention Network (R-GAT)**

GAT uses attention.

R-GAT extends attention to heterogeneous graphs.

Instead of equal importance,

attention is computed separately for each relation.

Example:

User → Bought → Product

Attention \= 0.9

User → Viewed → Product

Attention \= 0.3

The model learns:

Bought

\>

Viewed

for purchase prediction.

---

# **Comparison**

| Model | Supports Relations | Neighbor Sampling | Attention |
| ----- | ----- | ----- | ----- |
| GCN | No | No | No |
| GraphSAGE | No | Yes | No |
| GAT | No | No | Yes |
| R-GCN | Yes | No | No |
| R-GraphSAGE | Yes | Yes | No |
| R-GAT | Yes | Optional | Yes |

---

# **Knowledge Graph Embedding Models**

Several models learn embeddings differently.

---

# **1\. TransE**

Most famous KG embedding model.

Core idea:

Head

\+

Relation

≈

Tail

Formula:

\[  
h+r\\approx t  
\]

---

## **Intuition**

Example:

(Paris, CapitalOf, France)

Embedding:

Paris

\+

CapitalOf

≈

France

Relation acts like a translation vector.

Hence the name:

Translation Embedding

---

## **Advantages**

* Simple  
* Fast  
* Easy to train

---

## **Limitations**

Poor for:

* One-to-many  
* Many-to-one  
* Many-to-many relations

Example:

Person

BornIn

City

Many people share one city.

TransE struggles.

---

# **2\. TransR**

Improves TransE.

Idea:

Entities and relations belong to different spaces.

Instead of:

One Embedding Space

TransR learns:

Entity Space

Relation Space

Entities are projected into relation-specific spaces before translation.

Advantages:

* Better modeling of complex relations  
* Handles many-to-many relationships better

---

# **3\. DistMult**

Instead of translation,

DistMult uses multiplicative interactions.

Score function:

\[  
score(h,r,t)=h^Tdiag(r)t  
\]

Intuition:

Each relation scales the dimensions of entity embeddings.

Advantages:

* Fast  
* Simple

Limitation:

The score is symmetric:

ParentOf

≈

ChildOf

which is incorrect for asymmetric relations.

---

# **4\. ComplEx**

ComplEx extends DistMult using **complex-valued embeddings**.

Entities and relations are represented in the complex number space.

Advantages:

* Handles symmetric relations  
* Handles asymmetric relations  
* Better than DistMult for many KGs

Example:

ParentOf

≠

ChildOf

ComplEx can distinguish these.

---

# **Comparison of KG Embedding Models**

| Model | Main Idea | Strength | Limitation |
| ----- | ----- | ----- | ----- |
| TransE | (h+r\\approx t) | Simple, Fast | Poor for one-to-many & many-to-many |
| TransR | Separate entity and relation spaces | Better complex relations | More parameters |
| DistMult | Bilinear multiplication | Efficient | Cannot model asymmetric relations |
| ComplEx | Complex-valued embeddings | Handles asymmetric relations | More computationally expensive |

---

# **Applications of Knowledge Graph Embeddings**

### **Search Engines**

Google Knowledge Graph.

---

### **Recommendation Systems**

Netflix

Amazon

Spotify

---

### **Drug Discovery**

Predict unknown drug-disease interactions.

---

### **Question Answering**

Chatbots

Virtual Assistants

---

### **Fraud Detection**

Detect suspicious entity relationships.

---

### **Social Networks**

Friend recommendation.

---

# **Complete Pipeline**

Knowledge Graph  
      ↓  
Entities \+ Relations  
      ↓  
Knowledge Graph Embeddings  
      ↓  
R-GCN / R-GraphSAGE / R-GAT  
      ↓  
Node Embeddings  
      ↓  
Knowledge Graph Completion  
      ↓  
Predict Missing Triples

---

# **Interview Summary**

A **heterogeneous graph** contains **multiple types of nodes and relations**, making it suitable for modeling real-world systems such as e-commerce platforms, healthcare networks, and knowledge graphs. A **Knowledge Graph (KG)** represents facts as **triples (head, relation, tail)**, and **Knowledge Graph Embeddings (KGE)** map entities and relations into dense vectors for tasks such as **link prediction** and **knowledge graph completion**, where missing facts are inferred. Standard GNNs are extended to heterogeneous graphs through **Relational GCN (R-GCN)**, which learns **relation-specific weight matrices**, **R-GraphSAGE**, which combines neighbor sampling with relation-aware aggregation for scalability, and **R-GAT**, which applies **relation-aware attention** to weigh different edge types. Popular knowledge graph embedding models include **TransE**, which models relations as translations ((h+r\\approx t)); **TransR**, which projects entities into relation-specific spaces; **DistMult**, which uses bilinear multiplicative interactions but assumes symmetric relations; and **ComplEx**, which uses complex-valued embeddings to model both symmetric and asymmetric relations effectively. These techniques are widely used in search engines, recommendation systems, biomedical knowledge discovery, question answering, and fraud detection.

