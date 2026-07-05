# 26

# **K-Means Clustering**

K-Means Clustering is an **Unsupervised Machine Learning Algorithm** used to group similar data points into clusters.

Unlike classification algorithms, K-Means does not use labeled data.

Its goal is:

Group similar observations together and separate dissimilar observations.

---

# **Real-Life Example**

Suppose a shopping mall has customer data:

| Customer | Annual Income | Spending Score |
| ----- | ----- | ----- |
| A | High | High |
| B | High | Low |
| C | Low | High |

The mall wants to identify groups of customers for targeted marketing.

K-Means can automatically create customer segments such as:

Cluster 1 → Premium Customers  
Cluster 2 → Budget Customers  
Cluster 3 → Occasional Buyers

---

# **Intuition Behind K-Means**

Imagine dropping several magnets on a table full of iron particles.

Each particle gets attracted to the nearest magnet.

Eventually:

Particles around Magnet 1 → Cluster 1  
Particles around Magnet 2 → Cluster 2  
Particles around Magnet 3 → Cluster 3

K-Means works similarly.

---

# **What Does K Mean?**

K represents:

Number of Clusters

Example:

K \= 3

means create 3 clusters.

---

# **Key Terminology**

## **Cluster**

A group of similar observations.

---

## **Centroid**

The center point of a cluster.

Example:

Cluster A  
● ● ● ● ●

Center \= Centroid

The centroid represents the average location of all points in that cluster.

---

# **Working of K-Means**

---

## **Step 1: Choose K**

Suppose:

K \= 3

We want 3 clusters.

---

## **Step 2: Initialize K Centroids**

Randomly select K points as initial centroids.

Example:

Centroid 1  
Centroid 2  
Centroid 3

---

## **Step 3: Assign Points**

For every data point:

Calculate distance to each centroid.

Usually using Euclidean Distance.

Assign point to nearest centroid.

---

### **Euclidean Distance**

For points:

\[  
(x\_1,y\_1)  
\]

and

\[  
(x\_2,y\_2)  
\]

Distance:

genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"d=\\sqrt{(x\_2-x\_1)^2+(y\_2-y\_1)^2}"}}

---

## **Step 4: Update Centroids**

Compute new centroid as the average of all points in that cluster.

Formula:

Centroid=\\frac{\\sum x\_i}{n}

---

## **Step 5: Repeat**

Repeat:

Assign Points  
Update Centroids

until centroids stop changing.

This is called:

Convergence

---

# **Objective Function of K-Means**

K-Means tries to minimize:

### **Within Cluster Sum of Squares (WCSS)**

Also called Inertia.

Formula:

WCSS=\\sum\_{i=1}^{K}\\sum\_{x\\in C\_i}||x-\\mu\_i||^2

Where:

* (C\_i) \= Cluster  
* (\\mu\_i) \= Centroid

Goal:

Minimize WCSS

---

# **Elbow Method**

One major challenge:

How do we choose K?

The most common solution is the Elbow Method.

---

# **Intuition of Elbow Method**

As K increases:

WCSS decreases

because clusters become smaller.

Example:

| K | WCSS |
| ----- | ----- |
| 1 | 1000 |
| 2 | 600 |
| 3 | 300 |
| 4 | 200 |
| 5 | 180 |
| 6 | 170 |

Notice:

Big drop until K=3  
Small drop afterward

---

# **Elbow Curve**

Plot:

X-axis → Number of Clusters (K)  
Y-axis → WCSS

Graph:

WCSS  
|  
|\\  
| \\  
|  \\  
|   \\\_\_  
|      \\\_  
|  
\+----------------  
     K

The bend point resembles an elbow.

Choose K at that point.

Example:

Elbow at K=3

Therefore:

Optimal K \= 3

---

# **Limitation of K-Means**

K-Means is sensitive to initial centroid selection.

Different initial centroids may produce:

Different Clusters

This problem leads to local optima.

---

# **K-Means++**

K-Means++ is an improved initialization technique for K-Means.

It selects better starting centroids.

Result:

✔ Faster convergence

✔ Better clustering

✔ Lower WCSS

✔ More stable results

---

# **Why K-Means++?**

Suppose all initial centroids are chosen close together:

●●●      ○○○○○○○○

Poor clustering occurs.

K-Means++ spreads centroids apart before clustering starts.

---

# **Working of K-Means++**

---

## **Step 1**

Choose first centroid randomly.

---

## **Step 2**

Calculate distance of every point from chosen centroid.

---

## **Step 3**

Choose next centroid with probability proportional to squared distance.

Farther points have higher chance of selection.

---

## **Step 4**

Repeat until K centroids are selected.

---

## **Step 5**

Run standard K-Means.

---

# **K-Means vs K-Means++**

| Feature | K-Means | K-Means++ |
| ----- | ----- | ----- |
| Centroid Initialization | Random | Smart Initialization |
| Convergence Speed | Slower | Faster |
| Clustering Quality | Lower | Higher |
| Stability | Lower | Higher |
| Risk of Poor Clusters | High | Low |

---

# **Advantages of K-Means**

* Simple and easy to implement  
* Fast for large datasets  
* Scalable  
* Easy interpretation

---

# **Disadvantages of K-Means**

* Need to choose K manually  
* Sensitive to outliers  
* Sensitive to initialization  
* Assumes spherical clusters  
* Struggles with clusters of varying sizes

---

# **Applications of K-Means**

### **Customer Segmentation**

Group customers by behavior.

---

### **Recommendation Systems**

Identify similar users.

---

### **Image Compression**

Reduce image colors.

---

### **Document Clustering**

Group similar documents.

---

### **Market Basket Analysis**

Identify purchasing patterns.

---

# **Python Example**

from sklearn.cluster import KMeans

kmeans \= KMeans(  
    n\_clusters=5,  
    init='k-means++',  
    random\_state=0  
)

y\_kmeans \= kmeans.fit\_predict(X)

---

# **Interview Summary**

**K-Means Clustering** is an unsupervised learning algorithm that partitions data into K clusters by assigning each point to its nearest centroid and iteratively updating centroids until convergence. The algorithm minimizes the **Within Cluster Sum of Squares (WCSS)**. The **Elbow Method** helps determine the optimal number of clusters by identifying the point where adding more clusters yields diminishing reductions in WCSS. **K-Means++** improves standard K-Means by selecting smarter initial centroids that are well separated, leading to faster convergence and better clustering performance.

