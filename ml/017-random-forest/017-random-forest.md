# 22

# **Random Forest**

Random Forest is a powerful **Ensemble Learning** algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

It can be used for:

* Classification  
* Regression

Random Forest is one of the most widely used Machine Learning algorithms because of its high accuracy and robustness.

---

# **Intuition Behind Random Forest**

Suppose you want to decide whether to invest in a company.

Instead of asking one expert, you ask:

Expert 1 → Invest  
Expert 2 → Don't Invest  
Expert 3 → Invest  
Expert 4 → Invest  
Expert 5 → Don't Invest

Majority opinion:

Invest

This collective decision is usually more reliable than relying on a single expert.

Random Forest works in the same way.

Instead of one Decision Tree, it uses many Decision Trees and combines their predictions.

---

# **Why Not Use a Single Decision Tree?**

A Decision Tree can easily overfit.

Example:

Training Accuracy \= 100%  
Testing Accuracy \= 70%

The tree memorizes the training data.

Random Forest reduces this problem by averaging multiple trees.

---

# **What is a Forest?**

A Forest is simply:

Collection of Decision Trees

Example:

Tree 1  
Tree 2  
Tree 3  
Tree 4  
Tree 5  
...  
Tree N

Together they form a Random Forest.

---

# **Why "Random"?**

Two sources of randomness are introduced:

### **1\. Random Sampling of Data**

Each tree is trained on a random subset of observations.

---

### **2\. Random Sampling of Features**

Each split considers only a random subset of features.

This makes trees different from one another.

---

# **Working of Random Forest**

---

## **Step 1: Bootstrap Sampling**

Random Forest creates multiple datasets from the original dataset using sampling with replacement.

This is called:

Bootstrap Sampling

Example:

Original Dataset

A B C D E

Possible Bootstrap Sample

A C C D E

Notice:

C repeated  
B missing

This is allowed.

---

## **Step 2: Build Multiple Decision Trees**

Each bootstrap sample trains one Decision Tree.

Example:

Dataset 1 → Tree 1  
Dataset 2 → Tree 2  
Dataset 3 → Tree 3

Each tree sees slightly different data.

---

## **Step 3: Random Feature Selection**

Suppose we have 10 features.

At each split:

Decision Tree → Uses all 10 features

Random Forest:

Choose only random subset

Example:

Feature 2  
Feature 5  
Feature 9

This creates diversity among trees.

---

## **Step 4: Make Predictions**

---

# **Random Forest Classification**

Each tree votes.

Example:

| Tree | Prediction |
| ----- | ----- |
| Tree 1 | Yes |
| Tree 2 | No |
| Tree 3 | Yes |
| Tree 4 | Yes |
| Tree 5 | No |

Majority Vote:

Yes

Final Prediction:

Yes

---

# **Random Forest Regression**

Each tree predicts a value.

Example:

| Tree | Prediction |
| ----- | ----- |
| Tree 1 | 100 |
| Tree 2 | 120 |
| Tree 3 | 110 |

Final prediction is the average:

\\frac{100+120+110}{3}=110

---

# **Bagging (Bootstrap Aggregating)**

Random Forest is based on a technique called:

Bagging

Bagging Steps:

1. Create bootstrap samples  
2. Train multiple models independently  
3. Combine predictions

---

# **Why Does Random Forest Work So Well?**

Decision Trees have:

Low Bias  
High Variance

They fit training data very well but overfit.

Random Forest reduces variance by averaging many trees.

Result:

Lower Variance  
Better Generalization  
Higher Accuracy

---

# **Out-of-Bag (OOB) Error**

Since bootstrap sampling leaves some observations unused:

Example:

Dataset Size \= 1000

A tree may use:

630 samples

and leave:

370 samples

unused.

These unused samples are called:

Out-of-Bag Samples

They can be used to estimate model performance without needing a separate validation set.

---

# **Feature Importance**

Random Forest can rank features according to importance.

Example:

| Feature | Importance |
| ----- | ----- |
| Salary | 45% |
| Age | 30% |
| Experience | 15% |
| City | 10% |

This helps in feature selection.

---

# **Important Hyperparameters**

---

## **1\. n\_estimators**

Number of trees.

Example:

n\_estimators=100

More trees:

✔ Better performance

✖ More computation

---

## **2\. max\_depth**

Maximum depth of each tree.

Controls overfitting.

---

## **3\. min\_samples\_split**

Minimum samples required to split a node.

---

## **4\. min\_samples\_leaf**

Minimum samples in a leaf node.

---

## **5\. max\_features**

Number of random features considered at each split.

---

# **Advantages of Random Forest**

### **High Accuracy**

Often outperforms a single Decision Tree.

---

### **Reduces Overfitting**

Averaging reduces variance.

---

### **Handles Missing Values**

Can work with incomplete data.

---

### **Feature Importance**

Provides importance rankings.

---

### **Works with Large Datasets**

Scales well.

---

### **Handles Non-linear Data**

Learns complex relationships.

---

# **Disadvantages of Random Forest**

### **Slower Training**

Many trees must be trained.

---

### **Higher Memory Usage**

Stores many trees.

---

### **Less Interpretable**

Harder to explain than a single Decision Tree.

---

# **Decision Tree vs Random Forest**

| Feature | Decision Tree | Random Forest |
| ----- | ----- | ----- |
| Number of Trees | 1 | Many |
| Overfitting | High | Low |
| Accuracy | Lower | Higher |
| Variance | High | Low |
| Interpretability | Easy | Hard |
| Training Speed | Fast | Slower |

---

# **Random Forest vs Bagging**

| Feature | Bagging | Random Forest |
| ----- | ----- | ----- |
| Bootstrap Sampling | Yes | Yes |
| Random Feature Selection | No | Yes |
| Diversity of Trees | Lower | Higher |
| Accuracy | Good | Better |

---

# **Python Example**

## **Random Forest Classification**

from sklearn.ensemble import RandomForestClassifier

classifier \= RandomForestClassifier(  
    n\_estimators=100,  
    criterion='entropy',  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

## **Random Forest Regression**

from sklearn.ensemble import RandomForestRegressor

regressor \= RandomForestRegressor(  
    n\_estimators=100,  
    random\_state=0  
)

regressor.fit(X\_train, y\_train)

y\_pred \= regressor.predict(X\_test)

---

# **Applications of Random Forest**

* Fraud Detection  
* Disease Diagnosis  
* Credit Risk Assessment  
* Customer Churn Prediction  
* Recommendation Systems  
* Stock Market Prediction  
* Image Classification  
* Spam Detection

---

# **Interview Summary**

**Random Forest** is an Ensemble Learning algorithm based on **Bagging** that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting. Each tree is trained on a bootstrap sample of the dataset and uses a random subset of features during splitting. For classification, predictions are made using **majority voting**, while for regression, predictions are made using **averaging**. Random Forest is highly accurate, robust to noise, resistant to overfitting, and capable of providing feature importance, making it one of the most popular algorithms in Machine Learning.

