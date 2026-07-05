# graph ml 11

# **Reasoning in Knowledge Graphs using Embeddings**

A **Knowledge Graph (KG)** stores facts as triples:

(Head, Relation, Tail)

(h, r, t)

Example:

(Paris, CapitalOf, France)

(France, LocatedIn, Europe)

However, real-world Knowledge Graphs are **incomplete**.

Many facts are missing.

Instead of manually adding them, we use **Knowledge Graph Embeddings (KGE)** to automatically infer new facts.

This process is called:

Knowledge Graph Reasoning

---

# **What is Reasoning?**

Reasoning means:

**Inferring new facts from existing knowledge.**

Example:

Known facts:

(Paris, CapitalOf, France)

(France, LocatedIn, Europe)

Reasoning allows us to infer:

(Paris, LocatedIn, Europe)

Even though this triple was never explicitly stored.

---

# **Why Do We Need Reasoning?**

Knowledge Graphs are usually:

* Incomplete  
* Sparse  
* Missing relationships

Reasoning helps:

* Discover hidden knowledge  
* Answer complex queries  
* Recommend entities  
* Improve search engines

---

# **Embedding-Based Reasoning**

Instead of symbolic logic,

modern systems use embeddings.

Every entity becomes a vector.

Example:

Paris

↓

\[0.32,1.5,-0.7\]

Relation:

CapitalOf

↓

\[-0.2,0.8,1.1\]

Reasoning becomes a geometric problem in vector space.

---

# **Intuition**

Suppose:

Paris

\+

CapitalOf

≈

France

Now ask:

France

\+

LocatedIn

\=

?

The nearest embedding may be:

Europe

Thus,

the model predicts:

(France, LocatedIn, Europe)

---

# **Types of Reasoning**

---

# **1\. Link Prediction**

Predict missing entities.

Example:

(Paris, CapitalOf, ?)

Answer:

France

---

# **2\. Relation Prediction**

Predict missing relation.

Example:

(Paris, ?, France)

Answer:

CapitalOf

---

# **3\. Entity Prediction**

Predict missing head or tail.

Example:

(?, CapitalOf, France)

Answer:

Paris

---

# **Predictive Queries**

A **Predictive Query** asks the model to predict missing information instead of retrieving existing facts.

Example:

Database Query:

Who is the president of France?

Requires stored data.

---

Predictive Query:

Who is likely to become friends with Alice?

Requires inference.

---

# **Examples**

Social Network:

(User A

Friend

?)

Recommendation:

Suggest New Friend

---

Healthcare:

(Drug A

Treats

?)

Predict unknown disease.

---

E-commerce:

(User

Will Buy

?)

Predict future purchases.

---

# **Answering Predictive Queries**

Pipeline:

Knowledge Graph  
       ↓  
Learn Embeddings  
       ↓  
Encode Query  
       ↓  
Score Candidate Answers  
       ↓  
Choose Highest Score

---

# **Example**

Query:

(Paris, LocatedIn, ?)

Candidates:

Germany

Europe

Asia

Embedding model computes scores.

Highest score:

Europe

Predicted answer.

---

# **Scoring Functions**

Most embedding models compute:

Score(h,r,t)

Higher score:

More Likely Triple

Lower score:

Unlikely Triple

---

# **Multi-Hop Reasoning**

Some queries require multiple reasoning steps.

Example:

Known:

(Alice, LivesIn, Paris)

(Paris, CapitalOf, France)

Question:

Where does Alice live?

↓

France?

Need:

Alice

↓

Paris

↓

France

Multiple relations are combined.

---

# **Limitation of Traditional Embeddings**

Models like:

* TransE  
* DistMult  
* ComplEx

are excellent for:

Single-Hop Reasoning

But struggle with:

Complex Logical Queries

Example:

LivesIn

AND

WorksIn

---

# **Reasoning Using Box Embeddings**

A newer approach represents entities and queries as **boxes** instead of points.

Instead of:

Point Embedding

Use:

Hyper-Rectangle (Box)

---

# **Intuition**

Traditional embedding:

Paris

↓

•

Box embedding:

\+---------+  
|  Paris  |  
\+---------+

The box represents a **region** in embedding space.

---

# **Why Boxes?**

Logical concepts often represent **sets**, not individual points.

Example:

Concept:

European Cities

Contains:

* Paris  
* Berlin  
* Madrid

Instead of one point,

represent the entire set as a box.

---

# **Example**

Box:

European Cities

Contains:

Paris

Berlin

Rome

Another box:

Capital Cities

Contains:

Paris

London

Tokyo

Intersection:

European Capitals

Only:

Paris

Berlin

---

# **Why Box Embeddings are Powerful**

Boxes naturally support logical operations.

---

## **Intersection (AND)**

Query:

European

AND

Capital

Intersection of boxes.

---

## **Union (OR)**

Query:

Doctor

OR

Professor

Union of boxes.

---

## **Negation (NOT)**

Approximate complement.

Although harder than AND/OR,

modern methods support it.

---

# **Reasoning with Boxes**

Suppose:

Query:

Cities

↓

LocatedIn Europe

AND

Population \> 1 Million

Each condition is a box.

Intersection:

Valid Cities

Entities inside the final box satisfy the query.

---

# **Multi-Hop Queries with Boxes**

Example:

Known:

Alice

LivesIn

Paris

LocatedIn

France

MemberOf

EU

Question:

Which EU country does Alice live in?

Box embeddings combine all constraints geometrically.

---

# **Point Embeddings vs Box Embeddings**

| Point Embeddings | Box Embeddings |
| ----- | ----- |
| Entity → Point | Entity/Query → Region (Box) |
| Good for link prediction | Good for logical reasoning |
| Hard to represent sets | Naturally represent sets |
| Limited AND/OR reasoning | Excellent AND/OR reasoning |
| TransE, DistMult, ComplEx | Query2Box, BoxE, BetaE (extensions) |

---

# **Applications**

### **Search Engines**

Answer complex semantic queries.

---

### **Recommendation Systems**

Recommend items satisfying multiple conditions.

---

### **Drug Discovery**

Reason over:

Drug

↓

Protein

↓

Disease

relationships.

---

### **Knowledge Graph Completion**

Infer missing facts.

---

### **Question Answering**

Answer complex multi-hop questions.

---

### **Scientific Knowledge Graphs**

Infer hidden relationships.

---

# **Complete Pipeline**

Knowledge Graph  
      ↓  
Entity & Relation Embeddings  
      ↓  
Embedding Model  
(TransE / ComplEx / Box Embeddings)  
      ↓  
Logical Reasoning  
      ↓  
Predictive Queries  
      ↓  
Knowledge Graph Completion  
      ↓  
Missing Facts

---

# **Interview Summary**

**Reasoning in Knowledge Graphs** is the process of **inferring new facts from existing triples** rather than simply retrieving stored information. Modern approaches perform reasoning using **Knowledge Graph Embeddings (KGE)**, where entities and relations are mapped into a continuous vector space and missing facts are predicted by scoring candidate triples. This enables **predictive queries**, such as predicting missing entities, relations, or future links, which are essential for **knowledge graph completion**. Traditional embedding models such as **TransE**, **DistMult**, and **ComplEx** are effective for **single-hop reasoning** and link prediction but are limited when answering complex logical or multi-hop queries. To overcome this limitation, **Box Embeddings** represent entities or query results as **hyper-rectangles (boxes)** rather than points. Boxes naturally represent **sets of entities** and support logical operations such as **intersection (AND)**, **union (OR)**, and approximate **negation (NOT)**, making them highly effective for **multi-hop reasoning, predictive query answering, and complex knowledge graph completion**. These techniques are widely used in search engines, recommendation systems, question answering, biomedical knowledge graphs, and scientific discovery.

