# 9

# **Support Vector Machine (SVM) Regression**

Support Vector Machine Regression, also called **Support Vector Regression (SVR)**, is a supervised Machine Learning algorithm used for predicting continuous values.

Unlike Linear Regression, SVR tries to fit the best line within a certain error margin instead of minimizing only prediction error.

SVR works well for:

* Linear data  
* Non-linear data  
* High-dimensional datasets

---

# **Intuition Behind SVR**

In Linear Regression, the goal is to minimize the distance between actual and predicted points.

In SVR, the goal is different:

* Fit a line such that most points lie within a margin (boundary)  
* Ignore small errors inside the margin  
* Penalize only large errors outside the margin

---

# **SVR Concept**

SVR tries to create a tube or margin around the regression line.

This margin is controlled by:

called the **epsilon-insensitive region**.

Points inside the margin are ignored.

Only points outside the margin affect the model.

---

# **SVR Regression Function**

The regression equation is:

Where:

* (w) \= weight vector  
* (x) \= input feature vector  
* (b) \= bias

---

# **Key Idea of SVR**

The objective is:

* Maximize margin  
* Keep prediction error within epsilon  
* Use only important data points called **support vectors**

Support vectors are the points closest to the boundary.

---

# **Advantages of SVR**

* Handles high-dimensional data  
* Effective for non-linear problems  
* Good generalization performance  
* Robust against overfitting

---

# **Limitations**

* Computationally expensive for large datasets  
* Sensitive to kernel and parameter selection  
* Slower training time

---

# **Non-Linear SVR**

Sometimes data is not linearly separable.

Example:

* Curved stock trends  
* Population growth  
* Temperature prediction

A straight line cannot fit such data properly.

In these cases, **Non-Linear SVR** is used.

---

# **Intuition of Non-Linear SVR**

Instead of fitting a straight line in lower dimensions, SVR transforms data into a higher-dimensional space where it becomes linearly separable.

---

# **Example**

Suppose data forms a circular pattern in 2D.

It cannot be separated using a straight line.

After mapping into 3D space, the data may become separable using a plane.

---

# **Mapping to a Higher Dimension**

Data transformation can be represented as:

Where:

* (n) \= lower dimension  
* (m) \= higher dimension

This transformation helps model complex relationships.

---

# **Kernel SVM Intuition**

Computing higher-dimensional transformations directly is expensive.

To solve this, SVM uses the **Kernel Trick**.

A kernel function computes similarity between points without explicitly transforming them into higher dimensions.

This makes computations efficient.

---

# **The Kernel Trick**

Instead of calculating:

SVM directly computes:

Where:

* (K) \= Kernel Function

This avoids expensive computations.

---

# **Types of Kernel Functions**

## **1\. Linear Kernel**

Used when data is linearly separable.

Formula:

### **Applications**

* Text classification  
* High-dimensional data

---

# **2\. Polynomial Kernel**

Used for curved relationships.

Formula:

Where:

* (d) \= degree of polynomial

### **Applications**

* Image processing  
* Pattern recognition

---

# **3\. Radial Basis Function (RBF) Kernel**

Most commonly used kernel.

Handles highly non-linear data.

Formula:

Where:

* (\\gamma) controls influence of points

### **Applications**

* Complex classification problems  
* Medical diagnosis  
* Recommendation systems

---

# **4\. Sigmoid Kernel**

Behaves similarly to neural networks.

Formula:

### **Applications**

* Neural-network-like learning

---

# **Difference Between Linear SVR and Non-Linear SVR**

| Basis | Linear SVR | Non-Linear SVR |
| ----- | ----- | ----- |
| Data Type | Linear | Non-linear |
| Kernel | Linear | RBF/Polynomial/etc. |
| Complexity | Lower | Higher |
| Accuracy on Complex Data | Lower | Higher |

---

# **Python Example of SVR**

from sklearn.svm import SVR  
from sklearn.preprocessing import StandardScaler

\# Feature scaling  
sc\_X \= StandardScaler()  
sc\_y \= StandardScaler()

X \= sc\_X.fit\_transform(X)  
y \= sc\_y.fit\_transform(y.reshape(-1,1))

\# SVR model  
regressor \= SVR(kernel='rbf')  
regressor.fit(X, y.ravel())

---

# **Applications of SVR**

* Stock price prediction  
* Weather forecasting  
* Energy demand prediction  
* Risk analysis  
* Time-series forecasting

---

# **Conclusion**

Support Vector Regression (SVR) is a powerful regression algorithm that works for both linear and non-linear datasets.

Key concepts include:

* Margin-based prediction  
* Support vectors  
* Higher-dimensional mapping  
* Kernel trick  
* Kernel functions

SVR is widely used when complex relationships exist in data and high prediction accuracy is required.

