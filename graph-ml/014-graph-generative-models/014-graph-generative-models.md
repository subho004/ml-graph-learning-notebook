# graph ml 14

# **Generative Models for Graphs**

A **Graph Generative Model** is a model that **learns the structure of existing graphs and generates new graphs with similar properties**.

Instead of predicting labels or links, these models answer:

**How can we generate realistic networks that resemble real-world graphs?**

Examples:

* Generate synthetic social networks  
* Simulate internet topology  
* Create molecular structures  
* Generate protein interaction networks

---

# **Why Do We Need Graph Generative Models?**

Real-world graphs are often:

* Huge  
* Sensitive (privacy issues)  
* Expensive to collect

Instead of sharing real data,

we generate synthetic graphs that preserve important network properties.

---

# **Applications**

* Social Network Simulation  
* Drug Discovery  
* Recommendation Systems  
* Cybersecurity  
* Benchmarking Graph Algorithms  
* Privacy-Preserving Data Generation

---

# **Types of Graph Generative Models**

Classical Models:

* Erdős-Rényi Random Graph  
* Small World Model  
* Barabási-Albert Model  
* Kronecker Graph Model

Deep Learning Models:

* GraphVAE  
* GraphGAN  
* GraphRNN  
* Graph Diffusion Models

---

# **Key Network Properties**

A good graph generator should preserve important characteristics of real-world networks.

---

## **1\. Degree Distribution**

Degree \= Number of neighbors of a node.

Example:

A

↓

Degree \= 4

Real social networks often follow a **power-law distribution**:

Few nodes have many connections (hubs), while most have only a few.

---

## **2\. Clustering Coefficient**

Measures how well neighbors of a node are connected.

Example:

A

/ \\

B---C

Neighbors of A are also connected.

High clustering is common in social networks.

---

## **3\. Average Path Length**

The average number of hops required to travel between two nodes.

Small values imply efficient communication.

---

## **4\. Connected Components**

The number of disconnected groups in the graph.

Many real networks contain one **giant connected component** plus a few isolated groups.

---

## **5\. Diameter**

The maximum shortest-path distance between any two nodes.

---

## **6\. Community Structure**

Real graphs naturally contain densely connected communities.

---

## **7\. Power-Law Degree Distribution**

Many real networks follow:

\[  
P(k)\\propto k^{-\\gamma}  
\]

Meaning:

Few Hub Nodes

Many Low-Degree Nodes

---

# **Erdős-Rényi Random Graph (G(n,p))**

One of the earliest graph generation models.

Developed by:

* Paul Erdős  
* Alfréd Rényi

---

# **Intuition**

Suppose we have:

n Nodes

Every possible pair of nodes has:

Probability p

of being connected.

Every edge is generated independently.

---

# **Example**

Suppose:

n \= 5

p \= 0.4

Each possible edge has a 40% chance of existing.

Result:

A random graph.

---

# **G(n,p) Model**

Notation:

\[  
G(n,p)  
\]

Where:

* (n) \= Number of nodes  
* (p) \= Probability of edge creation

---

# **Properties**

Expected number of edges:

\[  
E=\\frac{n(n-1)}{2}p  
\]

Expected degree:

\[  
(n-1)p  
\]

---

# **Advantages**

* Very simple  
* Easy to analyze mathematically  
* Fast to generate

---

# **Limitations**

Real networks do **not** behave randomly.

Problems:

* Low clustering  
* No hubs  
* Degree distribution follows a binomial/Poisson distribution instead of a power law  
* Weak community structure

---

# **Real Networks vs G(n,p)**

| Property | Real Networks | Erdős-Rényi |
| ----- | ----- | ----- |
| Degree Distribution | Power Law | Binomial / Poisson |
| Hubs | Yes | Rare |
| Clustering | High | Low |
| Communities | Strong | Weak |
| Average Path Length | Small | Small |
| Realism | High | Low |

---

# **The Small World Model**

Proposed by:

**Watts and Strogatz (1998)**

Goal:

Create graphs having:

* High clustering  
* Small average path length

---

# **Motivation**

Real-world networks exhibit:

Friends of Friends

↓

Usually Friends

and yet

Any Person

↓

Few Hops Away

This is known as the **Small World Phenomenon** or **"Six Degrees of Separation."**

---

# **Construction**

### **Step 1**

Create a regular ring lattice.

Example:

Each node connected to its 4 nearest neighbors.

---

### **Step 2**

Randomly rewire each edge with probability:

β

---

### **Step 3**

Some long-range shortcuts appear.

Result:

High Clustering

\+

Short Paths

---

# **Properties**

* High clustering coefficient  
* Small average path length  
* Better reflects social networks than Erdős-Rényi

---

# **Applications**

* Social Networks  
* Brain Networks  
* Transportation Networks  
* Communication Systems

---

# **Kronecker Graph Generative Model**

The **Kronecker Graph Model** is designed to generate **large realistic networks** while preserving important structural properties.

Instead of randomly adding edges,

it repeatedly expands a small graph using the **Kronecker Product**.

---

# **Intuition**

Start with a tiny graph.

Example:

2 Nodes

Repeatedly duplicate and connect it.

After several iterations:

Tiny Graph

↓

Medium Graph

↓

Large Graph

↓

Realistic Network

---

# **Kronecker Product**

Suppose the initiator matrix is

\[  
K\_1=  
\\begin{bmatrix}  
0.9 & 0.5\\  
0.5 & 0.2  
\\end{bmatrix}  
\]

Each iteration computes the **Kronecker product** of the matrix with itself.

This produces progressively larger probability matrices from which edges are sampled.

---

# **Advantages**

* Generates massive graphs efficiently  
* Preserves self-similar (fractal-like) structure  
* Produces realistic degree distributions  
* Captures densification and shrinking diameter observed in many evolving networks

---

# **Applications**

* Internet topology modeling  
* Citation networks  
* Social media simulation  
* Biological networks

---

# **Comparison of Graph Generative Models**

| Model | Main Idea | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| Erdős-Rényi | Random edge creation | Simple, mathematically tractable | Unrealistic for many real networks |
| Small World (Watts-Strogatz) | Rewire a regular lattice | High clustering, short paths | Does not naturally produce hubs |
| Kronecker Graph | Recursive graph expansion | Scalable, preserves many real network properties | More complex to fit parameters |

---

# **Real Networks vs Classical Models**

| Property | Real Networks | Erdős-Rényi | Small World | Kronecker |
| ----- | ----- | ----- | ----- | ----- |
| Power-law Degree Distribution | ✔ | ✖ | ✖ | ✔ |
| High Clustering | ✔ | ✖ | ✔ | ✔ (approx.) |
| Short Average Path | ✔ | ✔ | ✔ | ✔ |
| Community Structure | ✔ | Weak | Moderate | Better |
| Hubs | ✔ | ✖ | Limited | ✔ |
| Scalability | High | High | High | Very High |

---

# **Modern Deep Generative Models for Graphs**

Recent approaches use deep learning to generate graphs.

Examples:

* **GraphVAE** – Variational Autoencoder for graphs  
* **GraphRNN** – Sequential graph generation  
* **GraphGAN** – Adversarial graph generation  
* **Graph Diffusion Models** – State-of-the-art graph generation using diffusion processes

These models learn directly from graph data and can generate realistic molecules, social networks, and knowledge graphs.

---

# **Complete Workflow**

Real Network  
      ↓  
Learn Network Properties  
      ↓  
Choose Generative Model  
(Erdős-Rényi / Small World / Kronecker / Deep Models)  
      ↓  
Generate New Graph  
      ↓  
Evaluate  
(Degree Distribution,  
Clustering,  
Path Length,  
Communities)

---

# **Interview Summary**

A **Graph Generative Model** learns the structural characteristics of existing graphs and generates **synthetic graphs** with similar properties. Important network characteristics include the **degree distribution**, **clustering coefficient**, **average path length**, **diameter**, **connected components**, and **community structure**. The **Erdős-Rényi Random Graph (G(n,p))** creates edges independently with probability (p), making it mathematically simple but unrealistic because it lacks hubs, strong clustering, and power-law degree distributions. The **Watts-Strogatz Small World Model** improves realism by rewiring a regular lattice to produce networks with **high clustering** and **short path lengths**, capturing the small-world phenomenon observed in social and biological networks. The **Kronecker Graph Generative Model** recursively expands a small initiator graph using the **Kronecker product**, producing scalable synthetic graphs that preserve many properties of real evolving networks, including heavy-tailed degree distributions and self-similar structure. Modern graph generation increasingly relies on deep learning models such as **GraphVAE**, **GraphRNN**, **GraphGAN**, and **Graph Diffusion Models**, which can learn complex graph distributions directly from data.

