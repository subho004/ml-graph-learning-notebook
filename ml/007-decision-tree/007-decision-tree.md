# 10

# **Decision Tree in Machine Learning**

A Decision Tree is a supervised Machine Learning algorithm used for:

* Classification problems  
* Regression problems

It works like a tree structure where:

* Each internal node represents a decision/condition  
* Each branch represents an outcome  
* Each leaf node represents the final prediction

Decision Trees are easy to understand and visualize.

---

# **Intuition Behind Decision Tree**

A Decision Tree works similarly to human decision-making.

---

# **Example**

Suppose we want to decide whether a person will buy a car.

Questions may be:

1. Is salary greater than 50K?  
2. Is age less than 30?  
3. Does the person own a house?

Based on answers, the model reaches a final decision:

* Buy  
* Not Buy

The algorithm continuously splits the dataset into smaller groups based on the most important feature.

---

# **Structure of a Decision Tree**

## **Root Node**

The topmost node representing the entire dataset.

## **Decision Node**

A node where data is split based on a condition.

## **Leaf Node**

Final output or prediction.

---

# **Working of Decision Tree**

1. Select the best feature  
2. Split dataset based on that feature  
3. Repeat splitting recursively  
4. Stop when:  
   * Pure nodes are formed  
   * Maximum depth is reached  
   * No further improvement occurs

---

# **How Best Feature is Selected**

Decision Trees use measures like:

* Information Gain  
* Gini Index  
* Variance Reduction

to select the best split.

---

# **Decision Tree for Regression (Regression Trees)**

Regression Trees are used when the target variable is continuous.

Example:

* Predicting salary  
* House price prediction  
* Temperature forecasting

---

# **Intuition of Regression Trees**

Instead of predicting categories, Regression Trees divide data into regions and predict the average value of that region.

---

# **Example**

Suppose we predict salary based on experience.

| Experience | Salary |
| ----- | ----- |
| 1 | 30K |
| 2 | 35K |
| 3 | 50K |
| 4 | 80K |

The tree creates intervals like:

* Experience \< 2.5 → Salary \= 32K  
* Experience ≥ 2.5 → Salary \= 65K

Predictions are averages of regions.

---

# **Regression Tree Splitting**

Regression Trees minimize variance within each split.

The goal is to reduce:

where:

* (y\_i) \= actual values  
* (\\bar{y}) \= mean of region

---

# **Advantages of Regression Trees**

* Handles non-linear relationships  
* No feature scaling required  
* Easy interpretation

---

# **Limitations**

* Can overfit easily  
* Sensitive to small data changes  
* Lower accuracy compared to ensemble methods

---

# **Decision Tree for Classification (Classification Trees)**

Classification Trees are used when the target variable is categorical.

Example:

* Spam detection  
* Disease prediction  
* Fraud detection

---

# **Intuition of Classification Trees**

The tree splits data into classes based on feature conditions.

Example:

* Age \< 30 → Buy  
* Age ≥ 30 → Not Buy

Final output is a category/class label.

---

# **Measures Used in Classification Trees**

## **1\. Entropy**

Measures randomness or impurity.

Formula:

Lower entropy means purer nodes.

---

# **2\. Information Gain**

Measures reduction in entropy after splitting.

Formula:

The feature with highest Information Gain is selected.

---

# **3\. Gini Index**

Measures impurity of nodes.

Formula:

Lower Gini value indicates better split.

---

# **Difference Between Regression Tree and Classification Tree**

| Basis | Regression Tree | Classification Tree |
| ----- | ----- | ----- |
| Output | Continuous value | Category/Class |
| Example | Salary prediction | Spam detection |
| Split Criterion | Variance reduction | Entropy/Gini |
| Leaf Node | Numeric value | Class label |

---

# **Advantages of Decision Trees**

* Easy to understand  
* No feature scaling needed  
* Handles numerical and categorical data  
* Works for both regression and classification

---

# **Limitations of Decision Trees**

* Overfitting problem  
* Unstable with small data changes  
* Can become very complex

---

# **Python Example**

## **Classification Tree**

from sklearn.tree import DecisionTreeClassifier

classifier \= DecisionTreeClassifier()  
classifier.fit(X\_train, y\_train)

---

## **Regression Tree**

from sklearn.tree import DecisionTreeRegressor

regressor \= DecisionTreeRegressor()  
regressor.fit(X\_train, y\_train)

---

# **Applications of Decision Trees**

* Medical diagnosis  
* Fraud detection  
* Customer segmentation  
* Loan approval systems  
* Sales prediction

---

# **Conclusion**

Decision Trees are powerful and interpretable Machine Learning algorithms used for both classification and regression tasks.

* **Classification Trees** predict categories/classes.  
* **Regression Trees** predict continuous numerical values.

They work by recursively splitting data into smaller groups using the most informative features.

