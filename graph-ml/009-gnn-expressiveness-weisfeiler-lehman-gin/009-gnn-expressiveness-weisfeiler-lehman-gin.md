# graph ml 9

# **Expressiveness of Graph Neural Networks (GNNs)**

One of the most important theoretical questions in Graph Neural Networks is:

**How powerful is a GNN at distinguishing different graph structures?**

This ability is called the **expressiveness** of a GNN.

A highly expressive GNN can distinguish graphs that have different structures, while a less expressive GNN may incorrectly produce the same embedding for different graphs.

---

# **What Does Expressiveness Mean?**

Expressiveness refers to:

**The ability of a GNN to generate different embeddings for structurally different graphs or nodes.**

A more expressive GNN can capture:

* Graph topology  
* Node neighborhoods  
* Complex structural patterns  
* Different graph structures

---

# **Intuition**

Suppose we have two graphs.

Graph 1:

A  
|  
B  
|  
C

Graph 2:

  A  
 / | \\  
B  C  D

A powerful GNN should assign:

Different Embeddings

because the structures are clearly different.

A weak GNN might incorrectly produce:

Same Embedding

making the graphs indistinguishable.

---

# **Why Does Expressiveness Matter?**

Many applications depend on graph structure.

Examples:

* Molecule Classification  
* Fraud Detection  
* Protein Analysis  
* Social Networks

If the GNN cannot distinguish different graph structures,

predictions become inaccurate.

---

# **Why Are Some GNNs Less Expressive?**

Most GNNs update node embeddings using:

Mean

Sum

Max

aggregation.

Sometimes different neighborhoods produce identical aggregated values.

Example:

Neighborhood 1:

2

4

Mean:

3

Neighborhood 2:

3

3

Mean:

3

Different neighborhoods,

same mean.

The GNN cannot distinguish them.

---

# **Designing More Expressive GNNs**

To improve expressiveness, researchers focus on:

### **Better Aggregation**

Instead of simple averaging,

learn richer aggregation functions.

---

### **Better Update Functions**

Use deeper neural networks.

---

### **Preserve Structural Information**

Include:

* Degree  
* Position  
* Edge Features  
* Attention

---

### **Avoid Over-Smoothing**

Maintain unique node information.

---

# **Weisfeiler-Lehman (WL) Test**

The **Weisfeiler-Lehman (WL) Test** is the standard benchmark used to measure the expressive power of GNNs.

It is also called:

1-WL Test

Color Refinement Algorithm

---

# **Purpose of WL Test**

Determine whether:

Two Graphs

↓

Are Structurally Different

If WL distinguishes them,

a highly expressive GNN should also distinguish them.

---

# **Intuition Behind WL Test**

Initially,

every node receives the same color.

Example:

Gray

---

Iteration 1:

Each node collects colors from neighbors.

Example:

Node

↓

Neighbor Colors

↓

Generate New Color

---

Iteration 2:

Repeat.

Eventually,

different graph structures receive different color patterns.

---

# **Example**

Graph A:

A

|

B

|

C

Graph B:

A

/ \\

B  C

After several WL iterations,

nodes receive different colors,

indicating different graph structures.

---

# **Relationship Between WL and GNNs**

Most Message Passing GNNs perform operations similar to the WL Test.

Both:

Collect Neighbor Information

↓

Aggregate

↓

Update Representation

Therefore,

the expressive power of many GNNs is bounded by the **1-WL Test**.

---

# **Limitation of Standard GNNs**

Standard GCNs and GraphSAGE cannot distinguish some non-isomorphic graphs.

They are:

No More Expressive

Than

1-WL

---

# **Graph Isomorphism**

Two graphs are **isomorphic** if they have identical structures but different node labels.

Example:

Graph 1:

A \----- B

|

C

Graph 2:

X \----- Y

|

Z

Structurally,

they are identical.

A good GNN should assign similar embeddings.

---

# **Graph Isomorphism Network (GIN)**

The **Graph Isomorphism Network (GIN)** was proposed to achieve the maximum expressive power possible for Message Passing GNNs.

It is theoretically proven to be:

As Powerful

As

1-WL Test

---

# **Intuition Behind GIN**

Instead of averaging neighbors,

GIN uses:

Sum Aggregation

Why?

Because:

Sum

↓

Preserves More Information

Than

Mean

---

# **GIN Update Rule**

Node embedding update:

# **\[**

# **h\_v^{(k)}**

MLP  
\\left(  
(1+\\epsilon)h\_v^{(k-1)}  
\+  
\\sum\_{u\\in N(v)}h\_u^{(k-1)}  
\\right)  
\]

Where:

* (h\_v) \= Node embedding  
* (N(v)) \= Neighbor set  
* MLP \= Multi-Layer Perceptron  
* (\\epsilon) \= Learnable or fixed parameter

---

# **Why Sum Aggregation?**

Suppose:

Neighborhood A:

1

2

Sum:

3

Neighborhood B:

3

Sum:

3

Although sums can occasionally match, **combined with an MLP and repeated message passing, sum aggregation is injective over multisets under suitable conditions**, making it more expressive than mean or max in theory.

---

# **Why Is GIN More Expressive?**

GIN preserves:

* Number of neighbors  
* Node multiplicities  
* Structural differences

Mean aggregation may lose this information.

---

# **GCN vs GraphSAGE vs GIN**

| Feature | GCN | GraphSAGE | GIN |
| ----- | ----- | ----- | ----- |
| Aggregation | Mean/Normalized Sum | Mean, Max, LSTM | Sum |
| Expressiveness | Moderate | Moderate | Highest among standard message-passing GNNs |
| Neighbor Sampling | No | Yes | No |
| Large Graphs | Moderate | Excellent | Moderate |
| Theoretical Power | ≤ 1-WL | ≤ 1-WL | Matches 1-WL |

---

# **Can GIN Beat WL?**

No.

A standard GIN **matches** the expressive power of the **1-Weisfeiler-Lehman (1-WL)** test but does **not exceed it**.

If the WL test cannot distinguish two graphs,

GIN also cannot.

To go beyond WL,

researchers use more advanced methods such as:

* Higher-Order GNNs  
* k-WL-based GNNs  
* Graph Transformers  
* Subgraph GNNs

---

# **Designing More Expressive GNNs**

Researchers improve expressiveness by:

### **Better Aggregation**

Use injective aggregation (e.g., sum).

---

### **Richer Node Features**

Include:

* Degree  
* Position  
* Edge attributes  
* Structural encodings

---

### **Attention Mechanisms**

Learn which neighbors are more important.

Example:

Graph Attention Networks (GAT)

---

### **Deeper Architectures with Skip Connections**

Capture larger neighborhoods while reducing over-smoothing.

---

### **Higher-Order Message Passing**

Capture interactions among groups of nodes instead of only pairwise neighbors.

---

# **Applications Requiring High Expressiveness**

### **Drug Discovery**

Different molecular structures require distinct embeddings.

---

### **Protein Analysis**

Small structural changes affect protein function.

---

### **Fraud Detection**

Transaction patterns matter more than individual accounts.

---

### **Social Networks**

Community structures should be distinguished accurately.

---

# **Interview Summary**

The **expressiveness of a Graph Neural Network (GNN)** refers to its ability to distinguish different graph structures and produce unique node or graph embeddings. Most message-passing GNNs update node representations by aggregating information from neighbors, but simple aggregation methods such as **mean** or **max** can lose structural information, limiting expressive power. The **Weisfeiler-Lehman (1-WL) Test**, also known as the **Color Refinement Algorithm**, is the standard benchmark for evaluating GNN expressiveness. Most traditional GNNs, including **GCN** and **GraphSAGE**, are at most as expressive as the **1-WL test**. The **Graph Isomorphism Network (GIN)** improves expressiveness by using **sum aggregation** together with a **Multi-Layer Perceptron (MLP)**, and it has been proven to match the expressive power of the **1-WL test**, making it one of the most expressive standard message-passing GNN architectures. However, GIN cannot distinguish graphs that 1-WL itself cannot distinguish; surpassing 1-WL requires more advanced architectures such as **higher-order GNNs**, **k-WL-based models**, or **Graph Transformers**.

