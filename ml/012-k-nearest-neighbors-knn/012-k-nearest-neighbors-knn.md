# 17

# **K-Nearest Neighbors (KNN) Algorithm**

K-Nearest Neighbors (KNN) is a **Supervised Machine Learning** algorithm used for both:

* Classification  
* Regression

It is one of the simplest and most intuitive Machine Learning algorithms.

KNN makes predictions based on the similarity between data points.

---

# **Intuition Behind KNN**

The basic idea is:

"Similar data points tend to belong to the same category."

Suppose you move into a new neighborhood.

To predict your interests, one could look at your closest neighbors:

* If most neighbors like cricket, you probably do too.  
* If most neighbors like football, you may belong to that group.

KNN follows the same principle.

---

# **Example of KNN Classification**

Suppose we want to classify a fruit.

Training Data:

| Weight | Fruit |
| ----- | ----- |
| 100g | Apple |
| 110g | Apple |
| 120g | Apple |
| 180g | Orange |
| 190g | Orange |

New Fruit:

130g

KNN checks the nearest fruits.

Nearest neighbors:

120g → Apple  
110g → Apple  
180g → Orange

Majority class:

Apple

Prediction:

Apple

---

# **How KNN Works**

## **Step 1: Choose K**

Select the number of neighbors.

Example:

K \= 3

means consider the 3 nearest points.

---

## **Step 2: Calculate Distance**

Distance is calculated between the new point and all training points.

Most commonly using Euclidean Distance.

---

# **Euclidean Distance Formula**

For two points:

\[  
(x\_1,y\_1)  
\]

and

\[  
(x\_2,y\_2)  
\]

Distance:

---

# **Step 3: Find K Nearest Neighbors**

Sort distances.

Select the K smallest distances.

---

# **Step 4: Make Prediction**

### **Classification**

Use majority voting.

Example:

| Neighbor | Class |
| ----- | ----- |
| N1 | Red |
| N2 | Red |
| N3 | Blue |

Prediction:

Red

---

### **Regression**

Take average value.

Example:

| Neighbor | Price |
| ----- | ----- |
| N1 | 100 |
| N2 | 120 |
| N3 | 140 |

Prediction:

---

# **Choosing the Value of K**

## **Small K**

Example:

K \= 1

Advantages:

* Captures local patterns

Disadvantages:

* Sensitive to noise  
* Overfitting

---

## **Large K**

Example:

K \= 20

Advantages:

* More stable

Disadvantages:

* Underfitting  
* May ignore local structure

---

# **Rule for Choosing K**

A common choice:

Where:

* (N) \= Number of training samples

Usually choose an odd value for binary classification to avoid ties.

---

# **Why Feature Scaling is Important in KNN**

KNN is distance-based.

Consider:

| Feature | Value |
| ----- | ----- |
| Age | 25 |
| Salary | 500000 |

Salary dominates distance calculations.

Therefore normalization or standardization is usually required before applying KNN.

---

# **Advantages of KNN**

* Simple and easy to understand  
* No training phase (lazy learner)  
* Works for classification and regression  
* Handles non-linear decision boundaries

---

# **Disadvantages of KNN**

* Slow for large datasets  
* Requires feature scaling  
* Sensitive to irrelevant features  
* High memory usage

---

# **KNN Classification vs KNN Regression**

| Feature | KNN Classification | KNN Regression |
| ----- | ----- | ----- |
| Output | Class Label | Continuous Value |
| Prediction | Majority Vote | Average |
| Example | Spam Detection | House Price Prediction |

---

# **Applications of KNN**

* Recommendation Systems  
* Image Classification  
* Pattern Recognition  
* Medical Diagnosis  
* Fraud Detection  
* Customer Segmentation

---

# **Python Example**

from sklearn.neighbors import KNeighborsClassifier

classifier \= KNeighborsClassifier(  
    n\_neighbors=5,  
    metric='minkowski',  
    p=2  
)

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

Where:

* `n_neighbors=5` → K \= 5  
* `p=2` → Euclidean distance

---

# **Decision Boundary in KNN**

Unlike Logistic Regression, which creates a straight decision boundary, KNN can create highly flexible and non-linear boundaries because predictions depend on nearby points.

This makes KNN powerful for complex datasets.

---

# **Conclusion**

K-Nearest Neighbors (KNN) is a simple yet powerful supervised learning algorithm that classifies or predicts data based on the nearest training examples. It works by measuring distances, selecting the K closest neighbors, and making predictions through majority voting (classification) or averaging (regression). KNN is easy to implement and effective for many real-world classification and regression problems.

