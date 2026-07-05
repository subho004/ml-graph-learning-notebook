# 45

# **Logistic Regression**

Despite its name, **Logistic Regression is a Classification Algorithm**, not a regression algorithm.

It is primarily used for:

* Binary Classification  
* Probability Prediction  
* Spam Detection  
* Disease Prediction  
* Customer Churn Prediction

---

# **Intuition Behind Logistic Regression**

Suppose we want to predict:

Will a student pass or fail?

Output:

Pass \= 1  
Fail \= 0

---

# **Why Not Use Linear Regression?**

Suppose Linear Regression predicts:

| Hours Studied | Predicted Output |
| ----- | ----- |
| 2 | 0.3 |
| 5 | 0.8 |
| 10 | 1.5 |
| 15 | 2.2 |

Problem:

Probabilities cannot exceed 1  
Probabilities cannot be negative

But Linear Regression can produce:

\-2  
3.5  
5

which makes no sense for classification.

---

# **Solution: Logistic Regression**

Instead of predicting directly:

0 or 1

Logistic Regression predicts:

Probability

between:

0 and 1

Then converts probability into a class label.

---

# **The Sigmoid Function**

Logistic Regression uses the **Sigmoid Function**.

It converts any value into a probability between 0 and 1\.

Formula:

\[  
P(Y=1)=\\frac{1}{1+e^{-z}}  
\]

where

\[  
z=b\_0+b\_1x\_1+b\_2x\_2+\\cdots+b\_nx\_n  
\]

---

# **Intuition of Sigmoid**

Input:

\-100

Output:

≈ 0

Input:

0

Output:

0.5

Input:

100

Output:

≈ 1

---

# **Sigmoid Curve**

Probability  
1.0 |                \*\*\*\*  
    |             \*\*\*  
0.5 |---------\*\*\*  
    |      \*\*\*  
0.0 |\*\*\*\*\*\*  
    \+------------------  
         z

S-shaped curve.

---

# **How Logistic Regression Works**

Suppose features:

Age  
Salary  
Experience

Step 1:

Calculate

\[  
z=b\_0+b\_1x\_1+b\_2x\_2+b\_3x\_3  
\]

---

Step 2:

Apply Sigmoid.

\[  
P(Y=1)=\\frac{1}{1+e^{-z}}  
\]

---

Step 3:

Convert probability into class.

Usually:

P ≥ 0.5 → Class 1

P \< 0.5 → Class 0

---

# **Example**

Prediction:

P(Buy Product)=0.85

Decision:

Class \= Buy

---

# **Odds**

Logistic Regression works using odds.

Odds represent:

Probability of Success  
\----------------------  
Probability of Failure

Formula:

\[  
Odds=\\frac{p}{1-p}  
\]

---

# **Example**

Suppose:

p \= 0.8

Odds:

\[  
\\frac{0.8}{0.2}=4  
\]

Meaning:

Success is 4 times more likely  
than failure.

---

# **Log-Odds (Logit)**

Taking logarithm of odds:

\[  
log\\left(\\frac{p}{1-p}\\right)  
\]

called:

Logit Function

Logistic Regression assumes:

# **\[**

# **log\\left(\\frac{p}{1-p}\\right)**

b\_0+b\_1x\_1+b\_2x\_2+\\cdots+b\_nx\_n  
\]

This creates a linear relationship.

---

# **Decision Boundary**

The boundary separating classes.

Example:

Probability \= 0.5

Above:

Class 1

Below:

Class 0

---

# **Maximum Likelihood Estimation (MLE)**

One of the most important interview topics.

---

# **Why Not Use Least Squares?**

Linear Regression uses:

Minimize Squared Error

But Logistic Regression outputs probabilities.

Squared Error is not ideal.

---

# **Intuition of MLE**

Suppose:

Actual Output \= 1

Model predicts:

0.9

Good prediction.

---

Another model predicts:

0.1

Bad prediction.

MLE asks:

Which parameters make the observed data most likely?

Choose those parameters.

---

# **Likelihood Function**

For one observation:

If:

y \= 1

Likelihood:

p

If:

y \= 0

Likelihood:

1-p

Combined:

\[  
L=\\prod p\_i^{y\_i}(1-p\_i)^{1-y\_i}  
\]

Goal:

Maximize Likelihood

---

# **Log-Likelihood**

Instead of multiplying many probabilities:

Take logarithm.

\[  
LL=  
\\sum  
\\left\[  
y\_i\\log(p\_i)  
\+  
(1-y\_i)\\log(1-p\_i)  
\\right\]  
\]

This simplifies optimization.

---

# **Cost Function (Log Loss)**

Logistic Regression minimizes:

\[  
Loss=  
\-\\frac{1}{n}  
\\sum  
\\left\[  
y\_i\\log(p\_i)  
\+  
(1-y\_i)\\log(1-p\_i)  
\\right\]  
\]

Also called:

Cross Entropy Loss  
Log Loss

---

# **Why Log Loss?**

Correct predictions:

Small Loss

Wrong confident predictions:

Huge Loss

Encourages better probability estimates.

---

# **Training Logistic Regression**

### **Step 1**

Initialize weights.

---

### **Step 2**

Compute probabilities.

Using sigmoid.

---

### **Step 3**

Calculate Log Loss.

---

### **Step 4**

Use Gradient Descent.

Update weights:

\[  
w=w-\\eta\\frac{\\partial L}{\\partial w}  
\]

---

### **Step 5**

Repeat until convergence.

---

# **Binary Classification Example**

Predict:

Pass / Fail

Outputs:

P(Pass)=0.87

Decision:

Pass

---

# **Multiclass Logistic Regression**

For more than two classes:

Cat  
Dog  
Bird

Use:

Softmax Regression

instead of sigmoid.

---

# **Advantages of Logistic Regression**

### **Simple**

Easy to implement.

### **Fast Training**

Works well on large datasets.

### **Interpretable**

Coefficients are understandable.

### **Probability Output**

Provides confidence scores.

---

# **Disadvantages**

### **Linear Decision Boundary**

Struggles with complex patterns.

### **Sensitive to Outliers**

Can be affected by extreme values.

### **Requires Feature Engineering**

For complex relationships.

---

# **Logistic Regression vs Linear Regression**

| Feature | Linear Regression | Logistic Regression |
| ----- | ----- | ----- |
| Output | Continuous Value | Probability |
| Problem Type | Regression | Classification |
| Function | Linear Line | Sigmoid Curve |
| Range | (-∞,+∞) | (0,1) |
| Cost Function | MSE | Log Loss |

---

# **Applications**

* Spam Detection  
* Disease Diagnosis  
* Credit Risk Prediction  
* Customer Churn Prediction  
* Fraud Detection  
* Marketing Response Prediction

---

# **Interview Summary**

**Logistic Regression** is a supervised machine learning algorithm used for **classification**, especially binary classification. It predicts probabilities using the **Sigmoid Function**, which maps any real value into the range 0 to 1\. The model assumes a linear relationship between features and the **log-odds (logit)** of the target variable. Parameters are estimated using **Maximum Likelihood Estimation (MLE)**, and training minimizes **Log Loss (Cross-Entropy Loss)** using Gradient Descent. Logistic Regression is simple, interpretable, fast, and highly effective for many real-world classification problems.

