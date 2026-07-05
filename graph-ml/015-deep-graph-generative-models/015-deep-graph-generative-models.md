# graph ml 15

# **Deep Generative Models for Graphs**

Classical graph generation models (Erdős-Rényi, Small World, Kronecker) rely on **hand-designed rules**.

Deep Generative Models learn these rules **automatically from data**.

Instead of specifying how graphs should be generated,

the model learns:

**The probability distribution of real graphs and generates new graphs that resemble them.**

---

# **Why Deep Generative Models?**

Real-world graphs are highly complex.

They contain:

* Communities  
* Hubs  
* Rich node attributes  
* Different edge types  
* Complex dependencies

Classical models cannot capture all these characteristics.

Deep learning models can.

---

# **Applications**

* Molecule Generation  
* Drug Discovery  
* Protein Design  
* Social Network Simulation  
* Circuit Design  
* Knowledge Graph Generation

---

# **Types of Deep Graph Generative Models**

The main categories are:

Auto-Regressive Models

Variational Autoencoders (GraphVAE)

Generative Adversarial Networks (GraphGAN)

Diffusion Models

Among these,

Auto-Regressive models are the easiest to understand.

---

# **Auto-Regressive Models**

An **Auto-Regressive (AR) Model** generates a graph **one step at a time**, conditioning each step on everything generated so far.

General idea:

**Generate the next node or edge based on the current partial graph.**

---

# **Intuition**

Imagine drawing a graph.

You don't draw everything simultaneously.

Instead:

Start

↓

Add Node

↓

Connect Edge

↓

Add Another Node

↓

Connect More Edges

↓

Repeat

The graph grows sequentially.

---

# **Auto-Regressive Generation**

Instead of predicting:

Entire Graph

Predict:

Next Node

or

Next Edge

The probability is factorized as:

\[  
P(G)=\\prod\_i P(G\_i|G\_{\<i})  
\]

Meaning:

Each generation step depends on previous steps.

---

# **Advantages**

* Produces valid graphs  
* Captures sequential dependencies  
* Flexible generation

---

# **Limitations**

* Slow generation  
* Difficult for very large graphs  
* Sensitive to node ordering

---

# **GraphRNN**

**GraphRNN** is one of the most influential autoregressive graph generation models.

It generates graphs:

Node-by-Node

Edge-by-Edge

using Recurrent Neural Networks (RNNs).

---

# **Intuition**

Instead of generating the whole graph,

GraphRNN asks:

### **Step 1**

Should we create a new node?

↓

Yes

---

### **Step 2**

Which previous nodes should this new node connect to?

↓

Predict edges

---

### **Step 3**

Repeat.

---

# **GraphRNN Architecture**

GraphRNN uses **two RNNs**.

---

## **1\. Node-Level RNN**

Responsible for:

Generate New Node

---

## **2\. Edge-Level RNN**

Responsible for:

Predict Edges

for that Node

---

# **GraphRNN Workflow**

Empty Graph  
      ↓  
Node RNN  
      ↓  
Generate Node  
      ↓  
Edge RNN  
      ↓  
Predict Connections  
      ↓  
Updated Graph  
      ↓  
Repeat

---

# **Example**

Initially

No Nodes

↓

Generate:

Node A

↓

Generate:

Node B

↓

Predict:

A \----- B

↓

Generate:

Node C

↓

Predict:

A \----- C

B \----- C

Continue until the graph is complete.

---

# **Why GraphRNN Works Well**

Instead of learning an enormous graph distribution,

it learns many small decisions:

* Add node?  
* Connect edge?  
* Stop generation?

This makes training easier.

---

# **Generating Realistic Graphs**

A generated graph should resemble the real dataset.

It should preserve:

* Degree distribution  
* Clustering coefficient  
* Community structure  
* Shortest path distribution  
* Number of connected components  
* Motif frequencies

---

# **Training Deep Graph Generative Models**

Pipeline:

Graph Dataset  
      ↓  
Graph Encoder  
      ↓  
Learn Graph Distribution  
      ↓  
Train Neural Generator  
      ↓  
Generate New Graphs

---

# **Challenges**

Graph generation is difficult because:

* Graphs have variable sizes  
* Variable numbers of edges  
* Different node orderings  
* Complex dependencies  
* Large search space

---

# **Scaling Up Graph Generation**

Large graphs may contain:

Millions of Nodes

Generating every node sequentially becomes slow.

Modern methods improve scalability using:

---

## **1\. Hierarchical Generation**

Generate communities first.

Then generate nodes inside communities.

---

## **2\. Parallel Generation**

Generate multiple nodes simultaneously.

Common in diffusion models.

---

## **3\. Graph Partitioning**

Split large graphs into smaller subgraphs.

Generate each independently.

---

## **4\. Sparse Generation**

Generate only likely edges.

Ignore improbable ones.

---

# **Evaluating Graph Generation**

After generating graphs,

we compare them with real graphs.

---

# **Structural Metrics**

## **Degree Distribution**

Generated graph should have similar degree statistics.

---

## **Clustering Coefficient**

Measures local connectivity.

---

## **Average Path Length**

Generated graphs should preserve small-world behavior.

---

## **Graph Diameter**

Maximum shortest-path distance.

---

## **Connected Components**

Should match the real graph.

---

## **Community Structure**

Generated communities should resemble real communities.

---

## **Motif Distribution**

Small recurring patterns (e.g., triangles, stars) should appear with similar frequencies.

---

# **Statistical Evaluation**

Common measures include:

* Maximum Mean Discrepancy (MMD)  
* Precision and Recall for Graphs  
* Validity  
* Novelty  
* Uniqueness

---

# **Applications to Molecule Generation**

One of the biggest successes of graph generation is in **drug discovery**.

---

# **Why Molecules are Graphs**

A molecule naturally forms a graph.

Nodes:

Atoms

Edges:

Chemical Bonds

Example:

Carbon

↓

Connected to

↓

Hydrogen

---

# **Goal**

Instead of predicting properties,

generate entirely new molecules.

---

# **Pipeline**

Known Molecules  
      ↓  
Train Graph Generator  
      ↓  
Learn Chemical Rules  
      ↓  
Generate New Molecules  
      ↓  
Test Drug Properties

---

# **Constraints**

Generated molecules must satisfy:

* Valid chemical bonds  
* Correct atom valency  
* Chemical stability  
* Synthesizability  
* Desired biological activity

---

# **Example**

Training molecules:

Aspirin

Ibuprofen

Paracetamol

Model learns common chemical patterns.

Then generates:

Completely New Molecule

with similar medicinal properties.

---

# **Applications**

### **Drug Discovery**

Design new medicines.

---

### **Material Science**

Generate new polymers and materials.

---

### **Protein Engineering**

Design proteins with desired functions.

---

### **Chemistry**

Generate catalysts and reactions.

---

### **Battery Research**

Design better electrolytes.

---

# **Deep Graph Generative Models Comparison**

| Model | Main Idea | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| GraphRNN | Sequential node and edge generation | Flexible, realistic | Slow for very large graphs |
| GraphVAE | Encode and decode entire graphs | Fast inference | Limited graph size |
| GraphGAN | Adversarial graph generation | Produces realistic graphs | Training instability |
| Graph Diffusion Models | Gradually denoise random graphs | State-of-the-art quality | Computationally expensive |

---

# **Complete Workflow**

Training Graphs  
      ↓  
Graph Encoder  
      ↓  
Learn Graph Distribution  
      ↓  
Deep Graph Generator  
(GraphRNN / GraphVAE / GraphGAN / Diffusion)  
      ↓  
Generate New Graphs  
      ↓  
Evaluate  
(Degree Distribution,  
Clustering,  
Communities,  
Motifs,  
Validity)  
      ↓  
Applications  
(Molecule Generation,  
Drug Discovery,  
Network Simulation)

---

# **Interview Summary**

**Deep Generative Models for Graphs** learn the probability distribution of real-world graphs and generate new graphs with similar structural properties. Unlike classical graph generators, they automatically capture complex characteristics such as **communities, hubs, motifs, and node attributes**. **Auto-regressive models** generate graphs sequentially by predicting the next node or edge based on the partially generated graph. **GraphRNN** is a landmark autoregressive model that uses a **Node RNN** to generate new nodes and an **Edge RNN** to predict connections for each node, producing realistic graphs one step at a time. The quality of generated graphs is evaluated using structural metrics such as **degree distribution, clustering coefficient, average path length, diameter, connected components, community structure, and motif frequencies**, along with statistical measures like **Maximum Mean Discrepancy (MMD)**, **validity**, **novelty**, and **uniqueness**. Scaling graph generation to large networks is achieved through **hierarchical generation, graph partitioning, sparse edge generation, and parallel generation**. One of the most important applications is **molecule generation**, where atoms are represented as nodes and chemical bonds as edges. Deep graph generative models learn chemical structures from existing molecules and generate novel, chemically valid molecules for **drug discovery, material science, protein engineering, and catalyst design**, significantly accelerating scientific research.

