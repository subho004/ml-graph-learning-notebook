# 40

# **Linear Discriminant Analysis (LDA)**

**Linear Discriminant Analysis (LDA)** is a **Supervised Dimensionality Reduction Technique** used to reduce features while preserving maximum class separability.

It is commonly used for:

* Classification Problems  
* Feature Extraction  
* Pattern Recognition  
* Face Recognition  
* Preprocessing before Classification

---

# **Intuition Behind LDA**

Suppose we have two classes:

Class A → Red Points  
Class B → Blue Points

with two features:

Height  
Weight

PCA would find directions with maximum variance.

But variance does not necessarily separate classes.

LDA asks:

Which projection best separates different classes?

---

# **PCA vs LDA Intuition**

Imagine students from two departments:

Computer Science  
Mechanical Engineering

---

### **PCA**

Looks for:

Maximum Spread

ignoring class labels.

---

### **LDA**

Looks for:

Maximum Class Separation

using class labels.

This is why LDA is supervised while PCA is unsupervised.

---

# **Goal of LDA**

LDA tries to:

### **Maximize**

Distance Between Classes

and

### **Minimize**

Spread Within Classes

---

# **Visual Intuition**

Suppose two classes overlap:

○ ○ ○ ○  
○ ○ ○ ○

△ △ △ △  
△ △ △ △

LDA finds a line where:

Projected classes  
are as far apart as possible

---

# **Key Idea**

LDA finds new axes called:

Linear Discriminants

that maximize class separability.

---

# **Terminology**

---

## **Within-Class Scatter**

Measures how spread out points are inside each class.

Small within-class scatter is desirable.

Example:

Class A points tightly grouped

Good.

---

## **Between-Class Scatter**

Measures distance between class centers.

Large between-class scatter is desirable.

Example:

Class A center far from Class B center

Good.

---

# **Objective Function of LDA**

LDA maximizes:

\[  
\\frac{Between\\ Class\\ Scatter}{Within\\ Class\\ Scatter}  
\]

In matrix form:

\[  
J(w)=\\frac{w^TS\_Bw}{w^TS\_Ww}  
\]

Where:

* (S\_B) \= Between-class scatter matrix  
* (S\_W) \= Within-class scatter matrix

---

# **Intuition of the Formula**

Large numerator:

Classes far apart

Small denominator:

Classes compact

Result:

Excellent separation

---

# **LDA Algorithm**

---

## **Step 1: Standardize Data**

Normalize features so large-scale variables do not dominate.

---

## **Step 2: Compute Mean of Each Class**

Example:

Class A Mean  
Class B Mean  
Class C Mean

---

## **Step 3: Compute Within-Class Scatter Matrix**

Measures variability inside each class.

\[  
S\_W=\\sum (x-\\mu\_c)(x-\\mu\_c)^T  
\]

---

## **Step 4: Compute Between-Class Scatter Matrix**

Measures distance between class means.

\[  
S\_B=\\sum n\_c(\\mu\_c-\\mu)(\\mu\_c-\\mu)^T  
\]

where:

* (\\mu\_c) \= Class mean  
* (\\mu) \= Overall mean

---

## **Step 5: Solve Eigenvalue Problem**

Find:

\[  
S\_W^{-1}S\_B  
\]

Then compute:

* Eigenvalues  
* Eigenvectors

---

## **Step 6: Select Top Discriminants**

Choose eigenvectors corresponding to largest eigenvalues.

These become:

LD1  
LD2  
LD3  
...

---

## **Step 7: Project Data**

Transform original features onto discriminant axes.

Result:

High-Dimensional Data  
          ↓  
     LDA Space  
          ↓  
 Better Separation

---

# **Number of Components in LDA**

This is a common interview question.

For:

C Classes

Maximum number of LDA components:

\[  
C-1  
\]

---

## **Example**

2 Classes:

Max Components \= 1

---

3 Classes:

Max Components \= 2

---

5 Classes:

Max Components \= 4

---

# **Why C \- 1?**

Only (C-1) independent directions are needed to separate (C) class means.

---

# **Example**

Suppose:

Features \= 100  
Classes \= 3

LDA can produce at most:

2 Discriminant Components

even though there are 100 features.

---

# **PCA vs LDA**

| Feature | PCA | LDA |
| ----- | ----- | ----- |
| Type | Unsupervised | Supervised |
| Uses Labels | No | Yes |
| Goal | Maximize Variance | Maximize Class Separation |
| Components | Up to Features | Up to C−1 |
| Classification Performance | Moderate | Better |
| Dimensionality Reduction | Yes | Yes |

---

# **Example**

Dataset:

| Feature 1 | Feature 2 | Class |
| ----- | ----- | ----- |
| 2 | 3 | A |
| 3 | 4 | A |
| 7 | 8 | B |
| 8 | 9 | B |

LDA finds a projection where:

Class A  
     |  
     |  
     |  
\-----|----------------  
     |  
     |  
Class B

Classes become maximally separated.

---

# **Advantages of LDA**

### **Better Class Separation**

Designed specifically for classification.

---

### **Reduces Overfitting**

Fewer dimensions.

---

### **Faster Training**

Smaller feature space.

---

### **Works Well with Small Datasets**

Unlike deep learning.

---

# **Disadvantages of LDA**

### **Assumes Linear Separation**

Classes should be linearly separable.

---

### **Assumes Normal Distribution**

Works best when features are Gaussian.

---

### **Assumes Equal Covariance**

All classes should have similar covariance matrices.

---

### **Sensitive to Outliers**

Outliers can distort class means.

---

# **Applications of LDA**

### **Face Recognition**

Eigenfaces and Fisherfaces.

---

### **Medical Diagnosis**

Disease classification.

---

### **Customer Segmentation**

Classifying customer groups.

---

### **Speech Recognition**

Speaker classification.

---

### **Bioinformatics**

Gene classification.

---

# **LDA as a Classifier**

Besides dimensionality reduction, LDA itself can be used directly as a classification algorithm.

It assumes:

Each class follows  
a Gaussian distribution  
with common covariance matrix.

Then predicts the most likely class.

---

# **PCA vs LDA Interview Answer**

**PCA** finds directions of maximum variance without considering class labels, whereas **LDA** finds directions that maximize separation between classes using class labels. PCA is unsupervised and useful for general dimensionality reduction, while LDA is supervised and particularly useful as a preprocessing step for classification tasks.

---

# **Interview Summary**

**Linear Discriminant Analysis (LDA)** is a supervised dimensionality reduction technique that projects data onto lower-dimensional spaces while maximizing class separability. It achieves this by maximizing **between-class scatter** and minimizing **within-class scatter**. The algorithm computes class means, within-class scatter matrices, and between-class scatter matrices, then solves an eigenvalue problem to find optimal discriminant directions. Unlike PCA, which focuses on preserving variance, LDA focuses on improving classification performance. For a dataset with **C classes**, LDA can produce at most **C−1 discriminant components** and is widely used in classification, face recognition, medical diagnosis, and pattern recognition.

