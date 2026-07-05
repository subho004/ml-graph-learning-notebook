# 39

# **Principal Component Analysis (PCA)**

Principal Component Analysis (PCA) is an **Unsupervised Dimensionality Reduction Technique** used to reduce the number of features while preserving as much information (variance) as possible.

It is widely used for:

* Feature Reduction  
* Data Visualization  
* Noise Reduction  
* Faster Model Training  
* Handling Multicollinearity

---

# **Intuition Behind PCA**

Suppose a dataset contains:

Age  
Salary  
Experience  
Education  
City  
Credit Score

Some features may be highly correlated.

Example:

Experience ↑  
Salary ↑

These features contain overlapping information.

PCA combines such features into new variables called:

Principal Components

which retain most of the important information.

---

# **Why PCA?**

Problems with high-dimensional data:

* More computation  
* More storage  
* Overfitting  
* Curse of Dimensionality

PCA reduces dimensions while retaining maximum variance.

---

# **Intuition Using a 2D Dataset**

Suppose data points form an elongated cloud:

     \*  
    \*   \*  
  \*      \*  
 \*         \*

Most variation occurs along one direction.

Instead of using:

X-axis  
Y-axis

PCA finds a new axis that captures maximum variance.

This new axis becomes:

Principal Component 1 (PC1)

---

# **Principal Components**

Principal Components are:

* New variables  
* Linear combinations of original features  
* Uncorrelated with each other

---

# **PC1 (First Principal Component)**

Captures:

Maximum Variance

among all possible directions.

---

# **PC2 (Second Principal Component)**

Captures:

Remaining Maximum Variance

and is perpendicular to PC1.

---

# **Example**

Suppose:

Original Features \= 100

PCA may reduce them to:

10 Principal Components

while preserving:

95% Information

---

# **PCA Algorithm**

---

## **Step 1: Standardize the Dataset**

PCA is affected by feature scales.

Example:

| Feature | Range |
| ----- | ----- |
| Age | 20–60 |
| Salary | 10,000–1,000,000 |

Salary dominates.

Therefore standardization is required.

Formula:

\[  
z=\\frac{x-\\mu}{\\sigma}  
\]

Result:

Mean \= 0  
Variance \= 1

---

## **Step 2: Compute Covariance Matrix**

Covariance measures relationships between features.

For two variables:

\[  
Cov(X,Y)  
\]

Interpretation:

### **Positive Covariance**

X increases  
Y increases

---

### **Negative Covariance**

X increases  
Y decreases

---

### **Zero Covariance**

No linear relationship

Covariance Matrix:

\[  
C=  
\\begin{bmatrix}  
Cov(X\_1,X\_1) & Cov(X\_1,X\_2) \\  
Cov(X\_2,X\_1) & Cov(X\_2,X\_2)  
\\end{bmatrix}  
\]

---

## **Step 3: Compute Eigenvalues and Eigenvectors**

This is the heart of PCA.

From the covariance matrix we calculate:

### **Eigenvectors**

Represent directions.

These become:

Principal Components

---

### **Eigenvalues**

Represent importance.

Higher eigenvalue:

More Variance Explained

---

# **Example**

| Component | Eigenvalue |
| ----- | ----- |
| PC1 | 8 |
| PC2 | 3 |
| PC3 | 1 |

PC1 captures most information.

---

## **Step 4: Sort Eigenvalues**

Arrange in descending order.

Example:

PC1 \= 8  
PC2 \= 3  
PC3 \= 1

Order:

PC1  
PC2  
PC3

---

## **Step 5: Select Top K Components**

Choose components explaining most variance.

Example:

| Component | Variance Explained |
| ----- | ----- |
| PC1 | 70% |
| PC2 | 20% |
| PC3 | 5% |
| PC4 | 5% |

Choose:

PC1 \+ PC2

Total:

90% Variance

---

## **Step 6: Transform Data**

Project original data onto selected principal components.

Result:

Original Features  
↓  
Principal Components

Reduced-dimensional dataset obtained.

---

# **Explained Variance Ratio**

Measures information retained.

Formula:

\[  
\\frac{\\lambda\_i}{\\sum \\lambda}  
\]

where:

* (\\lambda\_i) \= Eigenvalue

---

# **Example**

Eigenvalues:

8, 3, 1

Total:

12

PC1:

8/12 \= 66.7%

PC2:

3/12 \= 25%

PC3:

1/12 \= 8.3%

---

# **Choosing Number of Components**

### **Explained Variance Rule**

Keep enough components to explain:

90%  
95%  
99%

variance.

---

### **Scree Plot**

Plot:

X-axis → Components  
Y-axis → Eigenvalues

Look for the elbow point.

---

# **Geometric Interpretation**

PCA performs:

Rotation of Coordinate System

to align axes with maximum variance directions.

The data itself does not change.

Only the coordinate system changes.

---

# **Advantages of PCA**

### **Dimensionality Reduction**

Fewer features.

---

### **Faster Training**

Smaller datasets.

---

### **Noise Reduction**

Removes low-variance components.

---

### **Visualization**

High-dimensional data visualized in 2D or 3D.

---

### **Removes Multicollinearity**

Principal components are orthogonal.

---

# **Disadvantages of PCA**

### **Reduced Interpretability**

Principal components lose original meaning.

---

### **Linear Technique**

Cannot capture complex non-linear relationships.

---

### **Information Loss**

Some variance is discarded.

---

# **PCA Example**

Suppose:

100 Features

After PCA:

10 Components

Explained Variance:

95%

Result:

90% reduction in dimensions  
Only 5% information lost

---

# **PCA vs Feature Selection**

| Feature | PCA | Feature Selection |
| ----- | ----- | ----- |
| Creates New Features | Yes | No |
| Removes Features | No | Yes |
| Interpretability | Lower | Higher |
| Handles Correlation | Excellent | Moderate |

---

# **Applications of PCA**

### **Face Recognition**

Eigenfaces.

---

### **Image Compression**

Reduce image dimensions.

---

### **Data Visualization**

100D → 2D.

---

### **Genomics**

Gene expression analysis.

---

### **Finance**

Stock market analysis.

---

### **Preprocessing for ML**

Reduce feature count before training.

---

# **Python Example**

from sklearn.decomposition import PCA

pca \= PCA(n\_components=2)

X\_pca \= pca.fit\_transform(X)

---

# **Interview Summary**

**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated variables called **Principal Components**. The first principal component captures the maximum variance in the data, while each subsequent component captures the remaining variance and is orthogonal to previous components. The PCA algorithm consists of **standardizing the data, computing the covariance matrix, finding eigenvalues and eigenvectors, sorting them by explained variance, selecting the top components, and projecting the data onto those components**. PCA is widely used for dimensionality reduction, visualization, noise reduction, and handling multicollinearity while retaining most of the original information.

