# 16

# **Logistic Regression in Machine Learning**

Logistic Regression is a supervised Machine Learning algorithm used for **classification problems**.

Despite the name "Regression", it is primarily used to predict **categorical outcomes**, especially binary classes.

Examples:

* Spam / Not Spam  
* Yes / No  
* Fraud / Genuine  
* Disease / No Disease

---

# **Why Not Use Linear Regression for Classification?**

Suppose we want to predict whether a student passes or fails.

| Study Hours | Result |
| ----- | ----- |
| 2 | Fail (0) |
| 4 | Fail (0) |
| 6 | Pass (1) |
| 8 | Pass (1) |

A Linear Regression model can predict values like:

1.4  
\-0.2  
0.7

These are invalid probabilities because probabilities must lie between 0 and 1\.

Therefore, we need Logistic Regression.

---

# **Intuition Behind Logistic Regression**

Instead of predicting a direct class label, Logistic Regression predicts the **probability** that a data point belongs to a class.

For example:

P(Pass) \= 0.85

Since probability \> 0.5,

Prediction \= Pass

---

# **The Sigmoid Function**

Logistic Regression uses the **Sigmoid Function** to convert any value into a probability between 0 and 1\.

Formula:

Where:

\[  
z \= b\_0+b\_1X  
\]

---

# **Why Sigmoid?**

Input can be:

\-100  
0  
100

Output becomes:

0  
0.5  
1

Thus:

* Very negative → Probability close to 0  
* Zero → Probability \= 0.5  
* Very positive → Probability close to 1

---

# **Logistic Regression Equation**

First compute:

Then apply sigmoid:

---

# **Decision Boundary**

After obtaining probability:

### **If**

P(Y=1) ≥ 0.5

Class \= 1

### **Else**

P(Y=1) \< 0.5

Class \= 0

The threshold 0.5 is called the **Decision Boundary**.

---

# **Maximum Likelihood Estimation (MLE)**

One of the most important concepts in Logistic Regression is **Maximum Likelihood Estimation (MLE)**.

---

# **Why Not Use Least Squares?**

Linear Regression uses Least Squares Error:

But Logistic Regression outputs probabilities.

Using least squares creates:

* Non-convex optimization  
* Poor probability estimates

Therefore Logistic Regression uses MLE.

---

# **Intuition of Maximum Likelihood**

Suppose:

| Actual Result | Predicted Probability |
| ----- | ----- |
| Pass (1) | 0.95 |
| Fail (0) | 0.05 |

This model seems good because:

* Actual passes receive high probabilities  
* Actual failures receive low probabilities

MLE tries to find parameter values that maximize the probability of observing the actual training data.

---

# **Likelihood Function**

For one observation:

### **If y \= 1**

Probability:

\[  
P(y=1)=p  
\]

### **If y \= 0**

Probability:

\[  
P(y=0)=1-p  
\]

Combined:

---

# **Likelihood for Entire Dataset**

For n observations:

The goal is:

### **Maximize L**

or equivalently

### **Maximize Log-Likelihood**

---

# **Log-Likelihood Function**

Taking logarithm simplifies multiplication into addition:

The algorithm finds coefficients that maximize this value.

---

# **Cost Function in Logistic Regression**

Instead of maximizing likelihood directly, we minimize the negative log-likelihood:

This leads to **Cross Entropy Loss**.

---

# **Advantages of Logistic Regression**

* Simple and interpretable  
* Fast training  
* Produces probabilities  
* Works well for linearly separable data  
* Less prone to overfitting

---

# **Limitations**

* Assumes linear decision boundary  
* Struggles with highly non-linear data  
* Sensitive to outliers  
* Requires feature engineering for complex problems

---

# **Applications**

* Disease diagnosis  
* Spam detection  
* Credit risk analysis  
* Customer churn prediction  
* Fraud detection

---

# **Python Example**

from sklearn.linear\_model import LogisticRegression

classifier \= LogisticRegression()

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

# **Difference Between Linear and Logistic Regression**

| Basis | Linear Regression | Logistic Regression |
| ----- | ----- | ----- |
| Output | Continuous value | Probability/Class |
| Problem Type | Regression | Classification |
| Function | Linear Line | Sigmoid Curve |
| Range | (-∞, \+∞) | (0,1) |
| Optimization | Least Squares | Maximum Likelihood |

---

# **Conclusion**

Logistic Regression is one of the most important classification algorithms in Machine Learning. It predicts probabilities using the Sigmoid function and classifies data using a decision boundary. Instead of Least Squares, it uses **Maximum Likelihood Estimation (MLE)** to find the best parameters, making it highly effective for binary classification tasks such as spam detection, fraud detection, and disease prediction.

