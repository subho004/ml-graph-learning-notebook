# graph ml 13

# **Community Detection in Networks**

**Community Detection** is the process of identifying groups of nodes that are **more densely connected to each other than to the rest of the network**.

A community is also called a:

* Cluster  
* Module  
* Group

The objective is:

**Find naturally occurring groups in a network without predefined labels.**

---

# **Intuition**

Consider a social network.

Alice \---- Bob  
 |         |  
Charlie \--- David

Emma \---- Frank  
 |         |  
Grace \---- Henry

Clearly,

Alice, Bob, Charlie, David

form one community,

while

Emma, Frank, Grace, Henry

form another.

Nodes inside a community interact more frequently than with nodes outside the community.

---

# **Why Community Detection?**

Real-world networks naturally form communities.

Examples:

### **Social Networks**

* Friend groups  
* College classmates  
* Families

---

### **Banking**

* Fraud rings  
* Money laundering groups

---

### **Biology**

* Protein interaction modules  
* Gene regulatory networks

---

### **E-commerce**

* Customer segments  
* Product groups

---

# **Network Communities**

A **Network Community** is a subset of nodes satisfying:

Strong Internal Connections

Weak External Connections

Mathematically,

nodes inside a community have:

* High edge density  
* Frequent interactions

compared to nodes outside the community.

---

# **Characteristics of Good Communities**

A good community has:

* Many internal edges  
* Few external edges  
* Strong structural similarity  
* High connectivity

---

# **Example**

A \----- B  
|       |  
C \----- D

      E \----- F  
      |       |  
      G \----- H

Communities:

Community 1

A B C D

Community 2

E F G H

---

# **Granovetter's Theory (Strength of Weak Ties)**

One of the most influential theories in network science was proposed by **Mark Granovetter**.

It is called:

Strength of Weak Ties

---

# **Main Idea**

**Weak connections between communities are often more valuable than strong connections within communities.**

---

# **Intuition**

Suppose:

Family

↓

Strong Relationships

Everyone already knows similar information.

Now imagine:

Friend from Another Company

This is a **weak tie**.

That friend may provide:

* New job opportunities  
* New ideas  
* New information

Weak ties connect different communities.

---

# **Example**

A \----- B \----- C  
|       |  
D       E \----- F

Inside:

A B C D

Strong connections.

Between:

B

↓

E

Weak connection.

This single edge connects two communities.

---

# **Importance of Weak Ties**

Weak ties help in:

* Information diffusion  
* Disease spread modeling  
* Viral marketing  
* Friend recommendations  
* Knowledge transfer

---

# **Strong vs Weak Ties**

| Strong Tie | Weak Tie |
| ----- | ----- |
| Family | Acquaintance |
| Close Friends | Colleagues |
| Frequent Interaction | Rare Interaction |
| Within Community | Between Communities |

---

# **Modularity**

Modularity is the most common metric for evaluating community quality.

It measures:

**How well a network is divided into communities.**

---

# **Intuition**

Suppose we randomly divide a graph.

Communities may look poor.

If nodes inside communities have significantly more connections than expected by chance,

modularity becomes high.

---

# **Modularity Formula**

## **\[**

## **Q=**

## **\\frac{1}{2m}**

## **\\sum\_{ij}**

## **\\left(**

## **A\_{ij}**

\\frac{k\_i k\_j}{2m}  
\\right)  
\\delta(c\_i,c\_j)  
\]

Where:

* (A\_{ij}) \= Edge exists between i and j  
* (k\_i) \= Degree of node i  
* (m) \= Total number of edges  
* (\\delta) \= 1 if both nodes belong to the same community

---

# **Interpretation**

Modularity ranges approximately from:

\-1 to 1

Higher values indicate stronger community structure.

Typical interpretation:

Q ≈ 0

↓

Weak Communities

Q \> 0.3

↓

Good Communities

Q \> 0.7

↓

Very Strong Communities

---

# **Louvain Algorithm**

The **Louvain Algorithm** is one of the most popular algorithms for community detection.

Goal:

Maximize Modularity

It is:

* Fast  
* Scalable  
* Widely used

---

# **Intuition**

Initially,

every node forms its own community.

Example:

A

B

C

D

Each node:

Own Community

---

Then,

nodes are moved to neighboring communities

only if modularity increases.

---

# **Louvain Algorithm Steps**

---

## **Step 1**

Initialize.

Every node is its own community.

---

## **Step 2**

For each node,

evaluate moving it to each neighboring community.

Choose the move that gives the largest modularity gain.

---

## **Step 3**

Repeat until no move increases modularity.

---

## **Step 4**

Collapse each detected community into a single "super-node."

Example:

Community A

↓

Single Node

---

## **Step 5**

Repeat the process on the new graph.

Continue until modularity no longer improves.

---

# **Louvain Workflow**

Graph  
      ↓  
Each Node \= Community  
      ↓  
Move Nodes  
      ↓  
Increase Modularity  
      ↓  
Merge Communities  
      ↓  
Build New Graph  
      ↓  
Repeat

---

# **Advantages of Louvain**

* Very fast  
* Handles millions of nodes  
* Produces hierarchical communities  
* Maximizes modularity automatically

---

# **Limitation**

Louvain assumes:

One Node

↓

One Community

Real-world networks often violate this assumption.

---

# **Overlapping Communities**

In reality,

people belong to multiple communities.

Example:

Student

↓

College Friends

↓

Family

↓

Football Team

↓

Office

One person belongs to multiple groups simultaneously.

These are called:

Overlapping Communities

---

# **Why Overlapping Communities?**

Examples:

Social Networks

Person

↓

Family

\+

Office

\+

Friends

Protein Networks

One protein participates in multiple biological processes.

---

# **BigCLAM**

**BigCLAM** stands for:

Cluster Affiliation Model for Big Networks

It is one of the most popular algorithms for detecting overlapping communities.

---

# **Intuition**

Instead of assigning:

One Node

↓

One Community

BigCLAM learns:

Node

↓

Membership Scores

↓

Multiple Communities

---

# **Example**

Person A

Membership:

Family

0.9

Office

0.8

Football Club

0.6

Person belongs to all three communities.

---

# **BigCLAM Model**

Every node learns a **community affiliation vector**.

Example:

Node A

↓

\[0.8,0.1,0.7\]

Each dimension corresponds to one community.

Higher values indicate stronger membership.

---

# **Edge Probability**

Nodes sharing stronger community memberships are more likely to be connected.

Thus,

BigCLAM predicts edges based on shared affiliations.

---

# **BigCLAM Workflow**

Network  
      ↓  
Initialize Community Scores  
      ↓  
Learn Membership Strengths  
      ↓  
Optimize Likelihood  
      ↓  
Detect Multiple Communities

---

# **Advantages of BigCLAM**

* Detects overlapping communities  
* Scalable to large graphs  
* Suitable for social networks  
* Handles multiple memberships naturally

---

# **Louvain vs BigCLAM**

| Feature | Louvain | BigCLAM |
| ----- | ----- | ----- |
| Community Type | Non-overlapping | Overlapping |
| Objective | Maximize Modularity | Learn Community Memberships |
| Output | One community per node | Multiple communities per node |
| Scalability | Very High | High |
| Common Applications | General clustering | Social networks, biological networks |

---

# **Applications**

### **Social Networks**

* Friend group detection  
* Interest communities  
* Influencer analysis

---

### **Banking**

* Fraud rings  
* Money laundering detection

---

### **Biology**

* Protein complexes  
* Gene modules

---

### **Recommendation Systems**

* User communities  
* Product communities

---

### **Cybersecurity**

* Botnet detection  
* Attack groups

---

# **Complete Community Detection Pipeline**

Network  
     ↓  
Node & Edge Analysis  
     ↓  
Community Detection  
(Louvain / BigCLAM)  
     ↓  
Community Assignment  
     ↓  
Evaluate using Modularity  
     ↓  
Applications

---

# **Interview Summary**

**Community Detection** is the task of identifying groups of densely connected nodes within a network. A **network community** contains many internal connections and relatively few external ones. **Granovetter's Strength of Weak Ties Theory** states that **weak ties**, which connect different communities, play a crucial role in spreading information and connecting otherwise separate groups. The quality of a community partition is commonly measured using **modularity**, which compares the density of edges within communities against what would be expected in a random network. The **Louvain Algorithm** is a fast, hierarchical method that detects **non-overlapping communities** by iteratively maximizing modularity and merging communities into super-nodes. However, many real-world networks have **overlapping communities**, where a node belongs to multiple groups simultaneously. **BigCLAM (Cluster Affiliation Model for Big Networks)** addresses this by learning a **community affiliation vector** for each node, allowing multiple community memberships. Louvain is ideal when each node belongs to a single community, while BigCLAM is preferred for networks such as social or biological systems where overlapping memberships are common.

