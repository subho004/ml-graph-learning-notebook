# 18

# **Support Vector Machine (SVM)**

Support Vector Machine (SVM) is a powerful supervised Machine Learning algorithm used primarily for **classification** and sometimes for **regression**.

The main objective of SVM is to find the **best boundary (hyperplane)** that separates different classes with the **maximum margin**.

---

# **Intuition Behind SVM**

Suppose we have two classes:

* Red points  
* Blue points

Many lines can separate them.

Example:

Red ● ● ●

\-------------  
Line 1

Blue ▲ ▲ ▲

But SVM does not choose just any separating line.

It chooses the line that leaves the **maximum distance from both classes**.

This line is called the **Optimal Hyperplane**.

---

# **Why Maximum Margin?**

Imagine two roads:

* One narrow road between two mountains  
* One wide road between two mountains

Driving through the wider road is safer.

Similarly, a larger margin makes the model:

* More robust  
* Less sensitive to noise  
* Better at generalization

---

# **Important Terminologies**

## **1\. Hyperplane**

A decision boundary that separates classes.

For 2D data:

represents a line.

For higher dimensions it becomes a hyperplane.

---

## **2\. Margin**

The distance between the hyperplane and the nearest data points from each class.

SVM tries to maximize this margin.

---

## **3\. Support Vectors**

The data points closest to the hyperplane.

These points determine the position of the decision boundary.

Without support vectors, the hyperplane would change.

---

# **Working of SVM**

## **Step 1**

Plot the training data.

Example:

● ● ● ●

     \---- Hyperplane \----

▲ ▲ ▲ ▲

---

## **Step 2**

Identify support vectors.

These are the nearest points from both classes.

---

## **Step 3**

Find the hyperplane with maximum margin.

---

## **Step 4**

Use the hyperplane to classify new data points.

---

# **Mathematical Representation**

Hyperplane:

Decision rule:

If:

Class \= 1

If:

Class \= \-1

---

# **Hard Margin SVM**

Used when data is perfectly separable.

Example:

● ● ● ●

\------------

▲ ▲ ▲ ▲

No misclassification allowed.

---

# **Soft Margin SVM**

Real-world data usually contains noise.

Example:

● ● ● ▲ ●

One blue point appears inside the red region.

Soft Margin SVM allows some classification errors while still maximizing margin.

This is controlled using parameter:

---

# **Understanding Parameter C**

## **Large C**

* Fewer classification errors  
* Smaller margin  
* Risk of overfitting

---

## **Small C**

* Larger margin  
* Allows some errors  
* Better generalization

---

# **Linear SVM**

Used when data can be separated using a straight line.

Example:

Red Points

\----------

Blue Points

A linear hyperplane is sufficient.

---

# **Non-Linear SVM**

Sometimes data cannot be separated using a straight line.

Example:

    ● ● ●  
   ●       ●

      ▲

   ●       ●  
     ● ● ●

No straight line can separate classes.

In such cases, SVM uses kernels.

---

# **Mapping to Higher Dimensions**

The idea is:

Transform data into a higher-dimensional space where separation becomes possible.

Example:

2D Data:

Circle inside another circle

Not linearly separable.

Transform to 3D:

Data becomes separable by a plane

---

# **The Kernel Trick**

Computing higher-dimensional transformations directly is expensive.

SVM avoids this using the **Kernel Trick**.

Instead of explicitly transforming data, a kernel function computes similarity directly.

Kernel Function:

This provides the benefits of higher-dimensional mapping without the computational cost.

---

# **Types of Kernel Functions**

## **1\. Linear Kernel**

Used when data is linearly separable.

Formula:

Applications:

* Text classification  
* Sentiment analysis

---

## **2\. Polynomial Kernel**

Creates curved decision boundaries.

Formula:

Where:

* (d) \= polynomial degree

Applications:

* Pattern recognition  
* Image processing

---

## **3\. Radial Basis Function (RBF) Kernel**

Most commonly used kernel.

Formula:

Where:

* (\\gamma) controls influence of nearby points

Applications:

* Medical diagnosis  
* Fraud detection  
* Image classification

---

## **4\. Sigmoid Kernel**

Similar to neural network activation functions.

Formula:

Applications:

* Neural-network-like classification tasks

---

# **Advantages of SVM**

* Effective in high-dimensional spaces  
* Works well with small datasets  
* Robust against overfitting  
* Can handle non-linear data using kernels  
* Uses only support vectors for decision making

---

# **Disadvantages of SVM**

* Computationally expensive for large datasets  
* Sensitive to kernel selection  
* Difficult to interpret  
* Requires feature scaling

---

# **Applications of SVM**

* Face Recognition  
* Image Classification  
* Spam Detection  
* Text Categorization  
* Bioinformatics  
* Handwriting Recognition  
* Fraud Detection

---

# **Python Example**

from sklearn.svm import SVC

classifier \= SVC(  
    kernel='rbf',  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

# **Difference Between Logistic Regression and SVM**

| Basis | Logistic Regression | SVM |
| ----- | ----- | ----- |
| Goal | Predict Probability | Maximize Margin |
| Boundary | Linear | Linear / Non-linear |
| Output | Probability | Class |
| Small Datasets | Good | Excellent |
| Non-linear Problems | Limited | Very Effective |

---

# **Conclusion**

Support Vector Machine (SVM) is one of the most powerful classification algorithms. It works by finding the optimal hyperplane that maximizes the margin between classes. The nearest points, called **support vectors**, determine this boundary. For complex datasets, SVM uses **kernel functions** and the **kernel trick** to create non-linear decision boundaries efficiently. This makes SVM highly effective for tasks such as image recognition, spam filtering, and fraud detection.

