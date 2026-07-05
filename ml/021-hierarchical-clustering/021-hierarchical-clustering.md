# 27

# **Hierarchical Clustering**

Hierarchical Clustering is an **Unsupervised Machine Learning Algorithm** used to group similar data points into clusters.

Unlike K-Means:

* No need to specify K initially  
* Produces a hierarchy (tree structure) of clusters  
* Useful when the number of clusters is unknown

---

# **Intuition Behind Hierarchical Clustering**

Imagine organizing books in a library.

You first group:

* Science books  
* Literature books

Then Science books are further grouped into:

* Physics  
* Chemistry  
* Biology

Then Physics books may be grouped further.

This creates a hierarchy.

Hierarchical Clustering works in a similar way.

---

# **Types of Hierarchical Clustering**

There are two approaches:

## **1\. Agglomerative Hierarchical Clustering (Bottom-Up)**

Starts with:

Each data point \= One cluster

Then repeatedly merges the closest clusters.

Most commonly used.

---

## **2\. Divisive Hierarchical Clustering (Top-Down)**

Starts with:

All points \= One cluster

Then repeatedly splits clusters.

Less commonly used.

---

# **Agglomerative Hierarchical Clustering (AHC)**

This is the method most interviewers refer to when discussing Hierarchical Clustering.

---

# **Intuition**

Suppose we have 5 points:

A B C D E

Initially:

{A}  
{B}  
{C}  
{D}  
{E}

Each point is its own cluster.

---

## **Step 1**

Find closest clusters.

Example:

A and B

Merge them.

{AB}  
{C}  
{D}  
{E}

---

## **Step 2**

Find next closest clusters.

D and E

Merge them.

{AB}  
{C}  
{DE}

---

## **Step 3**

Continue merging until one cluster remains.

---

# **Dendrogram**

A dendrogram is a tree-like diagram showing cluster merging.

Example:

       \---------  
       |         |  
     \-----      \---  
    |     |    |   |  
   A      B   C   D E

---

# **Why Dendrogram is Important?**

It helps determine:

Optimal Number of Clusters

without using the Elbow Method.

---

# **Choosing Number of Clusters from Dendrogram**

Rule:

Draw a horizontal line through the largest vertical gap.

Example:

      |  
       |  
\-------|---------  
       |  
   \----|----  
   |       |

Number of branches cut \= Number of clusters.

---

# **Distance Between Clusters (Linkage Criteria)**

The most important concept in Hierarchical Clustering is:

How do we measure distance between clusters?

Different answers create different clustering results.

---

# **1\. Single Linkage (Nearest Neighbor)**

Distance between two clusters is the distance between their closest points.

Formula:

D(A,B)=\\min(d(x,y))

---

## **Example**

Cluster A:

A1 A2

Cluster B:

B1 B2

Use:

Closest(A,B)

---

## **Characteristics**

Advantages:

* Can find irregular shapes

Disadvantages:

* Chaining effect  
* Sensitive to noise

---

# **2\. Complete Linkage (Farthest Neighbor)**

Distance between two clusters is the distance between their farthest points.

Formula:

D(A,B)=\\max(d(x,y))

---

## **Example**

Use:

Farthest(A,B)

distance.

---

## **Characteristics**

Advantages:

* Produces compact clusters

Disadvantages:

* Sensitive to outliers

---

# **3\. Average Linkage**

Distance between clusters is the average distance between all pairs of points.

Formula:

D(A,B)=\\frac{1}{mn}\\sum d(x,y)

where:

* m \= points in cluster A  
* n \= points in cluster B

---

## **Characteristics**

Advantages:

* Balanced clustering  
* Less sensitive to noise

Most commonly used after Ward linkage.

---

# **4\. Centroid Linkage**

Distance is measured between cluster centroids.

Formula:

D(A,B)=d(C\_A,C\_B)

Where:

* (C\_A) \= centroid of cluster A  
* (C\_B) \= centroid of cluster B

---

## **Characteristics**

Advantages:

* Computationally efficient

Disadvantages:

* Can produce inversions in dendrogram

---

# **5\. Ward Linkage (Most Important)**

Most commonly used in Machine Learning.

Instead of distance, it minimizes the increase in variance after merging clusters.

Goal:

Minimize Within Cluster Variance

---

# **Intuition of Ward Linkage**

Suppose two clusters:

Cluster A  
Cluster B

Ward asks:

How much does total variance increase  
if we merge them?

Choose the merge causing the smallest increase.

---

## **Characteristics**

Advantages:

* Produces compact clusters  
* Usually best results  
* Most common in practice

Disadvantages:

* Prefers spherical clusters

---

# **Distance Metrics Used Inside Clustering**

Apart from linkage criteria, we also need point-to-point distances.

---

# **Euclidean Distance (Most Common)**

Formula:

d=\\sqrt{\\sum (x\_i-y\_i)^2}

Best for:

* Continuous numerical features

---

# **Manhattan Distance**

Formula:

d=\\sum |x\_i-y\_i|

Best for:

* Grid-based movement  
* High-dimensional data

---

# **Minkowski Distance**

Generalized distance measure.

Formula:

d=\\left(\\sum |x\_i-y\_i|^p\\right)^{1/p}

Special cases:

* p=1 → Manhattan  
* p=2 → Euclidean

---

# **Agglomerative HC Algorithm**

### **Step 1**

Start with:

n clusters

(one point per cluster)

---

### **Step 2**

Compute distance matrix.

---

### **Step 3**

Merge nearest clusters.

---

### **Step 4**

Update distance matrix.

---

### **Step 5**

Repeat until one cluster remains.

---

# **Hierarchical Clustering vs K-Means**

| Feature | Hierarchical Clustering | K-Means |
| ----- | ----- | ----- |
| Need K Initially | No | Yes |
| Output | Dendrogram | Clusters |
| Cluster Shape | Flexible | Mostly Spherical |
| Speed | Slower | Faster |
| Large Datasets | Poor | Better |
| Interpretability | High | Moderate |

---

# **Advantages of Hierarchical Clustering**

* No need to choose K initially  
* Produces dendrogram  
* Easy visualization  
* Works for complex cluster structures

---

# **Disadvantages**

* Computationally expensive  
* Not suitable for very large datasets  
* Sensitive to noise and outliers  
* Cannot undo merges once made

---

# **Python Example**

from sklearn.cluster import AgglomerativeClustering

hc \= AgglomerativeClustering(  
    n\_clusters=5,  
    linkage='ward'  
)

y\_pred \= hc.fit\_predict(X)

---

# **Interview Summary**

**Hierarchical Clustering** is an unsupervised learning algorithm that creates a hierarchy of clusters. The most common approach is **Agglomerative Hierarchical Clustering (AHC)**, which starts with each point as its own cluster and repeatedly merges the closest clusters. The clustering process is visualized using a **dendrogram**, which helps determine the optimal number of clusters. Different linkage methods define how cluster distances are measured: **Single Linkage (nearest neighbor), Complete Linkage (farthest neighbor), Average Linkage, Centroid Linkage, and Ward Linkage**. Among these, **Ward Linkage** is the most commonly used because it minimizes within-cluster variance and generally produces compact, well-separated clusters.

