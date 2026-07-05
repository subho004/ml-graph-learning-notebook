# 24

# **False Positive (FP) and False Negative (FN)**

False Positives and False Negatives are two of the most important concepts in classification problems and are derived from the Confusion Matrix.

---

# **Recall the Confusion Matrix**

|  | Predicted Positive | Predicted Negative |
| ----- | ----- | ----- |
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

---

# **False Positive (FP)**

A False Positive occurs when:

Actual Class \= Negative  
Predicted Class \= Positive

The model incorrectly predicts a positive outcome.

---

## **Example: Disease Detection**

Suppose a patient is healthy.

Actual \= Healthy

But the model predicts:

Predicted \= Disease

This is a False Positive.

---

## **Why Is It Called False Positive?**

* Positive prediction was made  
* But it was false

Hence:

False Positive

---

## **Real-World Examples**

### **Spam Detection**

Actual:

Normal Email

Predicted:

Spam

Important email goes to spam folder.

---

### **Fraud Detection**

Actual:

Valid Transaction

Predicted:

Fraud

Customer's card gets blocked unnecessarily.

---

# **False Negative (FN)**

A False Negative occurs when:

Actual Class \= Positive  
Predicted Class \= Negative

The model misses a positive case.

---

## **Example: Disease Detection**

Actual:

Disease Present

Predicted:

Healthy

This is a False Negative.

---

## **Why Is It Called False Negative?**

* Negative prediction was made  
* But it was wrong

Hence:

False Negative

---

# **Real-World Examples**

### **Cancer Detection**

Actual:

Cancer Present

Predicted:

No Cancer

Very dangerous.

---

### **Fraud Detection**

Actual:

Fraudulent Transaction

Predicted:

Valid Transaction

Fraud goes undetected.

---

# **False Positive vs False Negative**

| Metric | False Positive | False Negative |
| ----- | ----- | ----- |
| Actual Class | Negative | Positive |
| Predicted Class | Positive | Negative |
| Example | Healthy → Diseased | Diseased → Healthy |
| Error Type | False Alarm | Missed Detection |

---

# **Which Is More Dangerous?**

Depends on the application.

### **Disease Detection**

False Negative is more dangerous.

Sick patient sent home

---

### **Spam Detection**

False Positive is more annoying.

Important email marked as spam

---

# **CAP Curve (Cumulative Accuracy Profile Curve)**

CAP Curve is a graphical method used to evaluate classification models, especially in:

* Credit Risk Analysis  
* Banking  
* Marketing  
* Customer Churn Prediction

It measures how well a model identifies positive cases compared to random selection.

---

# **Intuition Behind CAP Curve**

Suppose a bank wants to identify loan defaulters.

Total Customers:

1000

Actual Defaulters:

100

A good model should identify most defaulters by examining only a small percentage of customers.

CAP Curve measures this ability.

---

# **CAP Curve Axes**

### **X-Axis**

Percentage of Total Population

Example:

0% → 100%

---

### **Y-Axis**

Percentage of Positive Cases Captured

Example:

Defaulters Identified

---

# **Three Curves in CAP Analysis**

---

## **1\. Random Model**

Selecting customers randomly.

Graph:

|  
|      /  
|    /  
|  /  
|/  
\+-------------

Straight diagonal line.

---

## **2\. Perfect Model**

Captures all positives immediately.

Graph:

|  
|---------  
|  
|  
|  
\+-------------

Ideal but unrealistic.

---

## **3\. Actual Model**

Real model performance.

Graph lies between:

Random Model  
and  
Perfect Model

---

# **Interpretation**

### **Closer to Perfect Curve**

Excellent Model

---

### **Closer to Random Curve**

Poor Model

---

# **Accuracy Ratio (AR)**

CAP analysis often uses:

Accuracy Ratio

Formula:

AR=\\frac{Area\_{Model}-Area\_{Random}}{Area\_{Perfect}-Area\_{Random}}

---

# **Interpretation of Accuracy Ratio**

| AR Value | Meaning |
| ----- | ----- |
| 1 | Perfect Model |
| 0.8 | Excellent Model |
| 0.5 | Good Model |
| 0 | Random Model |

---

# **Example**

Suppose:

Top 20% customers contain  
80% of all defaulters

This indicates a very strong model.

A random model would capture only:

20%

of defaulters.

---

# **CAP Curve vs ROC Curve**

| Feature | CAP Curve | ROC Curve |
| ----- | ----- | ----- |
| Common Usage | Credit Risk | General Classification |
| X-axis | Population % | False Positive Rate |
| Y-axis | Positive Cases Captured | True Positive Rate |
| Interpretation | Business-friendly | Statistical |

---

# **Advantages of CAP Analysis**

* Easy to understand  
* Useful for business decisions  
* Popular in banking and credit scoring  
* Measures model effectiveness visually

---

# **Interview Summary**

**False Positive (FP)** occurs when the model predicts a positive outcome for an actually negative case (false alarm), while **False Negative (FN)** occurs when the model predicts a negative outcome for an actually positive case (missed detection). The importance of minimizing FP or FN depends on the application domain.

A **CAP (Cumulative Accuracy Profile) Curve** is a graphical evaluation technique that compares a model's ability to capture positive cases against a random model and a perfect model. The closer the CAP curve is to the perfect model curve, the better the classifier. The **Accuracy Ratio (AR)** quantifies this performance and is widely used in banking, credit scoring, and risk modeling.

