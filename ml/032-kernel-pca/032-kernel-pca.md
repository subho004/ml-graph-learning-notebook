# 41

# **Kernel PCA (Kernel Principal Component Analysis)**

**Kernel PCA** is an extension of **Principal Component Analysis (PCA)** that can capture **non-linear relationships** in data.

While PCA works well when data is linearly separable, Kernel PCA can uncover complex structures by mapping data into a higher-dimensional feature space.

---

# **Why Do We Need Kernel PCA?**

Regular PCA assumes:

Important patterns lie along  
straight-line directions.

This works for many datasets but fails when data has non-linear structures.

---

# **Example Where PCA Fails**

Suppose data forms concentric circles:

    ○ ○ ○ ○ ○

   ○           ○

       ● ● ●

   ○           ○

     ○ ○ ○ ○ ○

Classes are clearly different.

However:

No straight line  
can separate them.

Regular PCA cannot discover this structure.

---

# **Intuition Behind Kernel PCA**

Kernel PCA says:

Instead of performing PCA in the original space, first map data into a higher-dimensional space where separation becomes easier.

Example:

Original Space:

2D

Mapped Space:

100D  
1000D  
Infinite Dimensions

In this new space:

Non-linear patterns  
become linear.

Then standard PCA is applied.

---

# **The Core Idea**

Regular PCA:

Original Data  
     ↓  
Compute Covariance  
     ↓  
Eigenvectors  
     ↓  
Principal Components

Kernel PCA:

Original Data  
     ↓  
Kernel Transformation  
     ↓  
Higher-Dimensional Space  
     ↓  
PCA  
     ↓  
Kernel Principal Components

---

# **The Problem with Explicit Mapping**

Suppose:

x \= (x₁,x₂)

Mapping:

\[  
\\phi(x)=(x\_1^2,x\_2^2,\\sqrt2x\_1x\_2)  
\]

For small dimensions this is manageable.

But imagine:

100 Features

Mapping could create:

Thousands  
or Millions  
of Features

Computationally expensive.

---

# **The Kernel Trick**

The genius idea behind Kernel PCA:

Compute inner products in the higher-dimensional space without explicitly performing the transformation.

Instead of computing:

\[  
\\phi(x)  
\]

we compute:

\[  
K(x\_i,x\_j)  
\]

called the Kernel Function.

This is known as:

Kernel Trick

---

# **Kernel Function**

A kernel measures similarity between two points.

General form:

\[  
K(x\_i,x\_j)=\\phi(x\_i)^T\\phi(x\_j)  
\]

This allows computations in the high-dimensional space indirectly.

---

# **Steps of Kernel PCA**

---

## **Step 1**

Choose a kernel.

Examples:

* RBF (Gaussian)  
* Polynomial  
* Sigmoid

---

## **Step 2**

Compute Kernel Matrix.

\[  
K=  
\\begin{bmatrix}  
K(x\_1,x\_1) & K(x\_1,x\_2) \\  
K(x\_2,x\_1) & K(x\_2,x\_2)  
\\end{bmatrix}  
\]

---

## **Step 3**

Center the Kernel Matrix.

Equivalent to centering data in feature space.

---

## **Step 4**

Compute Eigenvalues and Eigenvectors.

Just like PCA.

---

## **Step 5**

Select Top Components.

Choose components with largest eigenvalues.

---

## **Step 6**

Project Data.

Transform original data into Kernel PCA space.

---

# **Types of Kernel Functions**

---

# **1\. Linear Kernel**

Formula:

\[  
K(x\_i,x\_j)=x\_i^Tx\_j  
\]

---

### **Characteristics**

* Equivalent to ordinary PCA  
* No non-linear transformation

Used when:

Data already linear

---

# **2\. Polynomial Kernel**

Formula:

\[  
K(x\_i,x\_j)=(x\_i^Tx\_j+c)^d  
\]

Where:

* (d) \= Degree

---

### **Example**

Degree \= 2

Creates quadratic relationships

---

### **Intuition**

Can separate curved boundaries.

---

# **3\. Gaussian RBF Kernel (Most Important)**

Formula:

\[  
K(x\_i,x\_j)=e^{-\\gamma||x\_i-x\_j||^2}  
\]

---

### **Intuition**

Measures similarity based on distance.

If points are close:

Similarity ≈ 1

If far apart:

Similarity ≈ 0

---

### **Why Popular?**

Can model extremely complex non-linear patterns.

Most commonly used Kernel PCA kernel.

---

# **4\. Sigmoid Kernel**

Formula:

\[  
K(x\_i,x\_j)=tanh(ax\_i^Tx\_j+c)  
\]

Inspired by neural networks.

Less commonly used.

---

# **Visual Intuition**

Original Space:

Concentric Circles

○ ○ ○ ○

  ● ●

○ ○ ○ ○

Not linearly separable.

---

After Kernel PCA:

● ● ● ●

\-----------

○ ○ ○ ○

Now separation becomes easy.

---

# **PCA vs Kernel PCA**

| Feature | PCA | Kernel PCA |
| ----- | ----- | ----- |
| Linear Data | Excellent | Excellent |
| Non-linear Data | Poor | Excellent |
| Complexity | Lower | Higher |
| Kernel Needed | No | Yes |
| Computation | Fast | Slower |
| Interpretability | Higher | Lower |

---

# **Advantages of Kernel PCA**

### **Captures Non-linear Structure**

Major advantage.

---

### **Better Feature Extraction**

For complex datasets.

---

### **Works with Arbitrary Kernels**

Highly flexible.

---

### **Powerful Visualization Tool**

Useful for reducing complex data to 2D or 3D.

---

# **Disadvantages of Kernel PCA**

### **Computationally Expensive**

Requires kernel matrix computation.

---

### **Kernel Selection Required**

Wrong kernel can hurt performance.

---

### **Less Interpretable**

Features lose physical meaning.

---

### **Poor Scalability**

Large datasets become expensive.

---

# **Applications of Kernel PCA**

### **Image Recognition**

Extract non-linear image features.

---

### **Face Recognition**

Improve class separability.

---

### **Bioinformatics**

Gene expression analysis.

---

### **Anomaly Detection**

Detect unusual patterns.

---

### **Financial Modeling**

Capture non-linear market relationships.

---

# **PCA vs LDA vs Kernel PCA**

| Feature | PCA | LDA | Kernel PCA |
| ----- | ----- | ----- | ----- |
| Supervised | No | Yes | No |
| Linear | Yes | Yes | No |
| Uses Labels | No | Yes | No |
| Goal | Max Variance | Max Class Separation | Non-linear Variance |
| Non-linear Data | Poor | Poor | Excellent |

---

# **Python Example**

from sklearn.decomposition import KernelPCA

kpca \= KernelPCA(  
    n\_components=2,  
    kernel='rbf',  
    gamma=15  
)

X\_kpca \= kpca.fit\_transform(X)

---

# **Interview Summary**

**Kernel PCA (Kernel Principal Component Analysis)** is a non-linear dimensionality reduction technique that extends PCA using the **Kernel Trick**. Instead of performing PCA directly in the original feature space, it implicitly maps data into a higher-dimensional space where non-linear structures become linearly separable. It then performs PCA in that transformed space using a **Kernel Matrix**. Common kernels include the **Linear Kernel**, **Polynomial Kernel**, **Gaussian RBF Kernel**, and **Sigmoid Kernel**, with the **RBF Kernel** being the most widely used. Kernel PCA is especially useful for datasets with complex non-linear patterns where traditional PCA fails, making it valuable in image processing, pattern recognition, bioinformatics, and anomaly detection.

