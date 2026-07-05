# 15

# **Classification in Machine Learning**

Classification is a type of **Supervised Machine Learning** in which the model learns from labeled data and predicts a **categorical output (class label)**.

Unlike Regression, which predicts continuous values (e.g., salary, temperature), Classification predicts categories (e.g., Yes/No, Spam/Not Spam).

---

# **What is Classification?**

Classification involves assigning an input data point to one of several predefined classes.

### **Examples**

| Input | Output |
| ----- | ----- |
| Email | Spam / Not Spam |
| Patient Data | Disease / No Disease |
| Transaction | Fraud / Genuine |
| Image | Cat / Dog |

Here, the output belongs to a fixed category.

---

# **Intuition Behind Classification**

Suppose a bank wants to determine whether a loan applicant is likely to repay a loan.

Input Features:

* Age  
* Income  
* Credit Score

Output:

* Approved  
* Rejected

The model learns patterns from historical data and classifies new applicants accordingly.

---

# **Types of Classification**

## **1\. Binary Classification**

The output contains only two classes.

### **Examples**

* Yes / No  
* Spam / Not Spam  
* Fraud / Genuine  
* Pass / Fail

Example:

| Email Features | Prediction |
| ----- | ----- |
| Contains suspicious links | Spam |
| Normal content | Not Spam |

---

## **2\. Multi-Class Classification**

The output contains more than two classes.

### **Examples**

* Digit Recognition (0–9)  
* Animal Classification  
* Fruit Classification

Example:

| Image | Prediction |
| ----- | ----- |
| Apple Image | Apple |
| Banana Image | Banana |
| Orange Image | Orange |

---

## **3\. Multi-Label Classification**

A single data point can belong to multiple classes simultaneously.

### **Example**

Movie Genres:

| Movie | Output |
| ----- | ----- |
| Avengers | Action, Sci-Fi |
| Frozen | Animation, Family |

One movie can have multiple labels.

---

# **How Classification Works**

### **Step 1: Collect Data**

Example:

| Age | Income | Loan Status |
| ----- | ----- | ----- |
| 25 | 40K | Approved |
| 45 | 20K | Rejected |

---

### **Step 2: Train Model**

The algorithm learns relationships between inputs and outputs.

---

### **Step 3: Make Predictions**

New Applicant:

| Age | Income |
| ----- | ----- |
| 30 | 50K |

Prediction:

Approved

---

# **Common Classification Algorithms**

## **1\. Logistic Regression**

Used for binary classification.

Examples:

* Disease prediction  
* Spam detection

---

## **2\. K-Nearest Neighbors (KNN)**

Classifies based on nearest neighbors.

Example:

* Similar customers  
* Recommendation systems

---

## **3\. Support Vector Machine (SVM)**

Finds the optimal boundary separating classes.

Example:

* Image recognition  
* Text classification

---

## **4\. Decision Tree**

Uses a tree structure for decision-making.

Example:

* Loan approval  
* Customer segmentation

---

## **5\. Random Forest**

Uses multiple Decision Trees and voting.

Example:

* Fraud detection  
* Medical diagnosis

---

## **6\. Naive Bayes**

Based on probability and Bayes' theorem.

Example:

* Spam filtering  
* Sentiment analysis

---

# **Classification Metrics**

Classification models are evaluated using a **Confusion Matrix**.

---

## **Confusion Matrix**

|  | Predicted Positive | Predicted Negative |
| ----- | ----- | ----- |
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

Where:

* TP \= True Positive  
* TN \= True Negative  
* FP \= False Positive  
* FN \= False Negative

---

## **Accuracy**

Measures overall correctness.

---

## **Precision**

Measures correctness of positive predictions.

---

## **Recall**

Measures how many actual positives are identified.

---

## **F1 Score**

Harmonic mean of Precision and Recall.

---

# **Difference Between Classification and Regression**

| Basis | Classification | Regression |
| ----- | ----- | ----- |
| Output | Category/Class | Continuous Value |
| Example | Spam/Not Spam | House Price |
| Goal | Assign Class Label | Predict Numeric Value |
| Algorithms | Logistic Regression, SVM, KNN | Linear Regression, SVR |

---

# **Applications of Classification**

* Email Spam Detection  
* Fraud Detection  
* Disease Diagnosis  
* Face Recognition  
* Sentiment Analysis  
* Loan Approval  
* Customer Churn Prediction

---

# **Python Example**

from sklearn.model\_selection import train\_test\_split  
from sklearn.ensemble import RandomForestClassifier

\# Split data  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
    X, y, test\_size=0.2, random\_state=42  
)

\# Train classifier  
classifier \= RandomForestClassifier()  
classifier.fit(X\_train, y\_train)

\# Prediction  
y\_pred \= classifier.predict(X\_test)

---

# **Conclusion**

Classification is a supervised learning technique used to predict categorical outcomes. It is one of the most widely used Machine Learning tasks and forms the foundation of applications such as spam detection, fraud detection, medical diagnosis, recommendation systems, and image recognition. The goal is to learn patterns from labeled data and correctly assign new observations to predefined classes.

