# 21

# **Decision Tree**

Decision Tree is a **Supervised Machine Learning Algorithm** used for both:

* Classification Problems  
* Regression Problems

It works by recursively splitting data into smaller subsets based on feature values, forming a tree-like structure.

The final prediction is obtained by following a path from the root node to a leaf node.

---

# **Intuition Behind Decision Trees**

A Decision Tree mimics human decision-making.

### **Example: Should a Bank Approve a Loan?**

Instead of making a decision directly, the bank may ask:

Is Income \> ₹50,000?  
        |  
       Yes  
        |  
Is Credit Score \> 700?  
        |  
       Yes  
        |  
   Approve Loan

A Decision Tree follows the same process by asking a sequence of questions.

---

# **Real-Life Example**

Suppose we want to predict whether a customer will buy a product.

Features:

* Age  
* Salary  
* Previous Purchases

Decision Tree:

                Age \< 30?  
                /         \\  
              Yes         No  
              /             \\  
     Salary \> 50K?       Buy  
          /   \\  
       Yes    No  
       Buy   Don't Buy

Each decision splits the dataset into more homogeneous groups.

---

# **Structure of a Decision Tree**

### **1\. Root Node**

Topmost node.

Represents the entire dataset.

Example:

Age \< 30?

---

### **2\. Decision Node**

Intermediate nodes where splitting occurs.

Example:

Salary \> 50K?

---

### **3\. Branch**

Represents outcomes of a decision.

Example:

Yes / No

---

### **4\. Leaf Node**

Final prediction.

Example:

Buy  
Don't Buy

---

# **How Does a Decision Tree Learn?**

The tree repeatedly chooses the best feature for splitting.

Goal:

Create groups that are as pure as possible.

A pure node contains observations belonging mostly to one class.

---

# **Example Dataset**

| Age | Purchased |
| ----- | ----- |
| 22 | Yes |
| 24 | Yes |
| 45 | No |
| 50 | No |

A good split:

Age \< 30

Produces:

Yes Yes

No No

Perfect separation.

---

# **Classification Trees**

Used when target variable is categorical.

Examples:

* Spam / Not Spam  
* Disease / No Disease  
* Buy / Not Buy

---

# **How Classification Trees Choose Splits?**

Common criteria:

1. Entropy  
2. Information Gain  
3. Gini Index

---

# **Entropy**

Entropy measures impurity or randomness.

Formula:

Entropy=-\\sum p\_i\\log\_2(p\_i)

Where:

* (p\_i) \= probability of class i

---

## **Understanding Entropy**

### **Pure Node**

100% Yes

Entropy:

0

No uncertainty.

---

### **Mixed Node**

50% Yes  
50% No

Entropy:

1

Maximum uncertainty.

---

# **Information Gain**

Measures reduction in entropy after splitting.

Formula:

Information\\ Gain=Entropy(parent)-Entropy(children)

The feature with highest Information Gain is selected.

---

## **Example**

Before split:

Entropy \= 1

After split:

Entropy \= 0.2

Information Gain:

1 \- 0.2 \= 0.8

Large gain → Good split.

---

# **Gini Index**

Another impurity measure used by CART (Classification and Regression Trees).

Formula:

Gini=1-\\sum p\_i^2

---

## **Interpretation**

### **Pure Node**

Gini \= 0

### **Mixed Node**

Higher Gini value.

The algorithm chooses the split with the lowest weighted Gini.

---

# **Regression Trees**

Used when the target variable is continuous.

Examples:

* House Price Prediction  
* Salary Prediction  
* Temperature Forecasting

---

# **Intuition of Regression Trees**

Instead of predicting classes, Regression Trees predict numerical values.

The tree divides data into regions.

Each leaf predicts the average value of observations in that region.

---

# **Example**

| Experience | Salary |
| ----- | ----- |
| 1 | 30K |
| 2 | 35K |
| 3 | 50K |
| 4 | 80K |

Tree:

Experience \< 2.5  
      /      \\  
    Yes      No

Salary      Salary  
\=32.5K      \=65K

---

# **Splitting Criterion for Regression Trees**

Regression Trees use **Variance Reduction**.

Goal:

Minimize prediction variance within each node.

Formula:

Variance=\\frac{1}{n}\\sum (y\_i-\\bar{y})^2

The split that reduces variance the most is chosen.

---

# **Overfitting in Decision Trees**

Decision Trees can grow very deep.

Example:

Every training point gets its own leaf.

This memorizes training data and causes overfitting.

---

# **Pruning**

Pruning removes unnecessary branches.

Two approaches:

### **1\. Pre-Pruning**

Stop tree growth early.

Parameters:

* max\_depth  
* min\_samples\_split  
* min\_samples\_leaf

---

### **2\. Post-Pruning**

Build full tree first, then remove weak branches.

Produces simpler trees.

---

# **Advantages of Decision Trees**

### **Easy to Understand**

Can be visualized like a flowchart.

### **No Feature Scaling Required**

Works directly on raw features.

### **Handles Numerical and Categorical Data**

Both types can be used.

### **Non-Linear Relationships**

Can learn complex decision boundaries.

### **Feature Importance**

Automatically identifies important features.

---

# **Disadvantages of Decision Trees**

### **Overfitting**

Most common issue.

### **Unstable**

Small changes in data can create different trees.

### **Greedy Nature**

Chooses locally optimal splits.

May not find globally optimal tree.

---

# **Decision Tree vs Random Forest**

| Feature | Decision Tree | Random Forest |
| ----- | ----- | ----- |
| Trees | One | Many |
| Accuracy | Lower | Higher |
| Overfitting | High | Lower |
| Interpretability | Easy | Harder |
| Speed | Faster | Slower |

---

# **Python Example**

## **Classification Tree**

from sklearn.tree import DecisionTreeClassifier

classifier \= DecisionTreeClassifier(  
    criterion='entropy',  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

---

## **Regression Tree**

from sklearn.tree import DecisionTreeRegressor

regressor \= DecisionTreeRegressor(  
    random\_state=0  
)

regressor.fit(X\_train, y\_train)

---

# **Applications of Decision Trees**

* Loan Approval Systems  
* Medical Diagnosis  
* Fraud Detection  
* Customer Churn Prediction  
* Risk Assessment  
* Product Recommendation  
* Credit Scoring

---

# **Interview Summary**

**Decision Tree** is a supervised learning algorithm that recursively splits data based on feature values to create a tree-like model. For **classification**, it uses measures such as **Entropy, Information Gain, and Gini Index** to create pure class groups. For **regression**, it uses **variance reduction** and predicts the average value in each leaf node. Decision Trees are easy to interpret and require minimal preprocessing, but they are prone to overfitting, which can be controlled through pruning techniques. They form the foundation of powerful ensemble algorithms such as **Random Forest** and **Gradient Boosting**.

