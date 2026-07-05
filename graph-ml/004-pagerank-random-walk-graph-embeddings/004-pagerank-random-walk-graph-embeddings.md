# graph ml 4

# **PageRank**

**PageRank** is a graph algorithm originally developed by **Google founders Larry Page and Sergey Brin** to rank web pages.

The idea is:

A webpage is important if many important webpages link to it.

PageRank assigns every node (webpage) a numerical importance score based on the graph structure.

---

# **Intuition Behind PageRank**

Imagine each webpage is a person voting for another webpage.

Example:

A → B

C → B

D → B

Page **B** receives three votes.

So B appears more important.

However,

not all votes are equal.

A vote from an important webpage is worth more than a vote from an unimportant webpage.

---

# **Example**

Suppose:

A → B

B → C

D → C

Page C receives links from:

* B  
* D

If B itself is highly important,

then C becomes even more important.

---

# **Random Surfer Model**

PageRank is based on the **Random Surfer Model**.

Imagine a person browsing the internet.

They:

1. Open a webpage.  
2. Randomly click one of its links.  
3. Continue forever.

Pages visited more frequently receive higher PageRank.

---

# **Basic PageRank Formula**

The PageRank of page A is:

\[  
PR(A)=\\sum\_{B\\in In(A)}\\frac{PR(B)}{L(B)}  
\]

Where:

* (PR(B)) \= PageRank of page B  
* (L(B)) \= Number of outgoing links from B  
* (In(A)) \= Pages pointing to A

Meaning:

A page receives importance from all pages linking to it.

---

# **Damping Factor**

Real users don't click forever.

Sometimes they type a new URL.

This is modeled using the **Damping Factor**.

Usually:

d \= 0.85

Meaning:

* 85% probability → Follow a link  
* 15% probability → Jump to a random page

---

# **Complete PageRank Formula**

\[  
PR(A)=\\frac{1-d}{N}+d\\sum\_{B\\in In(A)}\\frac{PR(B)}{L(B)}  
\]

Where:

* (N) \= Number of pages  
* (d) \= Damping factor (typically 0.85)

---

# **Solving PageRank (Power Iteration)**

Initially:

Every page receives equal rank.

Example:

4 Pages

PR \= 0.25 each

Then repeatedly:

1. Compute new PageRanks.  
2. Update all nodes.  
3. Repeat until values stop changing (convergence).

This iterative method is called **Power Iteration**.

---

# **Example**

Graph:

A → B

B → C

C → A

Initial:

A \= 1/3

B \= 1/3

C \= 1/3

After several iterations,

the PageRank values stabilize.

---

# **Applications of PageRank**

* Google Search Ranking  
* Citation Networks  
* Social Network Influence  
* Recommendation Systems  
* Fraud Detection

---

# **Random Walk and Embeddings**

Random Walk is the foundation of many graph embedding methods.

---

# **Intuition**

Suppose:

A — B — C  
     |  
     D

Start at A.

Walk randomly:

A

↓

B

↓

D

↓

B

↓

C

Sequence generated:

A B D B C

This sequence behaves like a sentence in NLP.

---

# **Why Random Walk?**

Words occurring together are related.

Similarly,

nodes visited together during random walks are likely related.

Embedding methods exploit this idea.

---

# **Random Walk → Node Embeddings**

Pipeline:

Graph  
   ↓  
Random Walks  
   ↓  
Node Sequences  
   ↓  
Word2Vec (Skip-Gram)  
   ↓  
Node Embeddings

Examples:

* DeepWalk  
* Node2Vec

---

# **Random Walk with Restarts (RWR)**

Standard random walk may wander far away from the starting node.

Sometimes we want to remain close to the starting node.

Solution:

**Random Walk with Restarts.**

---

# **Intuition**

Start from node A.

Each step:

* With probability **c** → Return to A.  
* With probability **1−c** → Continue walking.

Example:

A

↓

B

↓

C

↓

Restart

↓

A

↓

D

---

# **Why Restart?**

Keeps exploration centered around the source node.

Useful for:

* Personalized recommendations  
* Similarity search  
* Local influence analysis

---

# **Personalized PageRank (PPR)**

Personalized PageRank is an extension of PageRank.

Instead of jumping to **any random page**, the surfer jumps back to a **specific preferred node or set of nodes**.

---

# **Intuition**

Suppose you're interested in:

Machine Learning

Search results should prioritize pages related to ML.

Instead of random jumps:

Restart

↓

Machine Learning Page

This creates rankings personalized to your interests.

---

# **Difference Between PageRank and Personalized PageRank**

| PageRank | Personalized PageRank |
| ----- | ----- |
| Random restart anywhere | Restart to chosen node(s) |
| Global importance | Personalized importance |
| Same ranking for everyone | Different rankings for different users |

---

# **Applications of Personalized PageRank**

* Personalized Search  
* Friend Recommendation  
* Product Recommendation  
* Social Networks  
* Knowledge Graph Search

---

# **Matrix Factorization and Node Embeddings**

Another way to learn node embeddings is through **Matrix Factorization**.

Instead of random walks,

we factorize matrices that capture node relationships.

---

# **Intuition**

Suppose we have an adjacency matrix:

|  | A | B | C |
| ----- | ----- | ----- | ----- |
| A | 0 | 1 | 1 |
| B | 1 | 0 | 0 |
| C | 1 | 0 | 0 |

Rather than using this large matrix directly,

we approximate it as:

Adjacency Matrix

≈

Embedding Matrix × Embedding Matrixᵀ

Each row of the embedding matrix becomes a node embedding.

---

# **Why Matrix Factorization?**

Nearby nodes should have similar embeddings.

The factorization learns latent features that preserve graph structure.

---

# **Matrix Factorization Workflow**

Adjacency Matrix  
        ↓  
Factorization  
        ↓  
Low-Dimensional Matrix  
        ↓  
Node Embeddings

---

# **Matrix Factorization vs Random Walk**

| Matrix Factorization | Random Walk |
| ----- | ----- |
| Uses adjacency/similarity matrix | Uses graph traversal |
| Global graph structure | Local neighborhoods |
| Mathematical decomposition | Sampling-based |
| Examples: GraRep, NetMF | DeepWalk, Node2Vec |

---

# **Extension to Graph Embeddings**

Instead of embedding each node,

we embed the **entire graph**.

---

# **Intuition**

Example:

One molecule:

Atoms

\+

Chemical Bonds

↓

One vector.

Entire Molecule

↓

Graph Embedding

---

# **How Graph Embeddings are Created**

### **Step 1**

Compute node embeddings.

Node A

↓

Embedding

---

### **Step 2**

Aggregate all node embeddings.

Methods include:

* Mean Pooling  
* Sum Pooling  
* Max Pooling  
* Attention Pooling

---

### **Step 3**

Produce one graph embedding.

Graph

↓

Single Vector

---

# **Applications of Graph Embeddings**

### **Molecule Classification**

Predict:

Drug or Non-Drug

---

### **Protein Classification**

Predict:

Protein Function

---

### **Social Networks**

Predict:

Community Type

---

### **Cybersecurity**

Predict:

Malicious Network

---

# **Relationship Between These Concepts**

Graph  
   │  
   ├── PageRank  
   │      ↓  
   │   Node Importance  
   │  
   ├── Random Walk  
   │      ↓  
   │   Node Sequences  
   │      ↓  
   │   DeepWalk / Node2Vec  
   │      ↓  
   │   Node Embeddings  
   │  
   ├── Matrix Factorization  
   │      ↓  
   │   Node Embeddings  
   │  
   └── Pooling  
          ↓  
      Graph Embedding

---

# **Interview Summary**

**PageRank** is a graph algorithm that ranks nodes based on the importance of incoming links using the **Random Surfer Model**. It iteratively computes scores using the **PageRank equation** with a **damping factor** to model random jumps. **Random Walks** generate sequences of connected nodes, which can be treated like sentences and used by algorithms such as **DeepWalk** and **Node2Vec** to learn node embeddings. **Random Walk with Restarts (RWR)** periodically returns to a starting node, making it suitable for local similarity and recommendation tasks. **Personalized PageRank (PPR)** extends PageRank by restarting from user-specific nodes, producing personalized rankings. Another approach to learning embeddings is **Matrix Factorization**, which decomposes graph matrices (such as adjacency or similarity matrices) into low-dimensional latent vectors representing nodes. Finally, **Graph Embeddings** are obtained by aggregating node embeddings using pooling techniques, enabling graph-level tasks such as molecule classification, protein function prediction, and network classification.

