# 19

# **Kernel SVM**

Kernel SVM is an extension of SVM used when data is **not linearly separable**.

A normal SVM tries to separate classes using a straight line (2D), plane (3D), or hyperplane (higher dimensions).

However, many real-world datasets are non-linear.

Examples:

* Face recognition  
* Image classification  
* Fraud detection  
* Medical diagnosis

Kernel SVM solves this problem using **kernel functions**.

---

# **Why Do We Need Kernel SVM?**

Consider this dataset:

     ● ● ●

   ●       ●

      ▲

   ●       ●

      ● ● ●

* Blue point in center  
* Red points surrounding it

No straight line can separate them.

A Linear SVM will fail.

---

# **Intuition of Kernel SVM**

Instead of separating data in the current dimension, Kernel SVM maps data into a **higher-dimensional space** where separation becomes possible.

---

# **Mapping to Higher Dimensions**

Suppose we have 2D coordinates:

(X1, X2)

Data forms a circle.

Now create a new feature:

The data is transformed from:

2D → 3D

In 3D, the classes may become linearly separable.

A plane can now separate them.

---

# **The Problem**

Directly transforming every point into higher dimensions can be computationally expensive.

For large datasets:

Thousands of features  
Millions of points

This becomes impractical.

---

# **Kernel Trick**

The Kernel Trick allows SVM to work in higher dimensions **without explicitly performing the transformation**.

Instead of computing:

\[  
\\phi(x)  
\]

for every point, SVM directly computes similarity using a kernel function.

---

# **Kernel Function**

A kernel computes:

Where:

* (x\_i) \= point i  
* (x\_j) \= point j  
* (\\phi) \= higher-dimensional mapping

Thus SVM gets the benefit of higher dimensions without actually computing them.

---

# **Gaussian RBF Kernel (Most Important)**

RBF stands for:

Radial Basis Function

It is the most widely used kernel.

---

# **Formula**

Where:

* (\\gamma) controls influence of nearby points  
* (||x\_i-x\_j||^2) \= squared distance

---

# **Intuition of RBF Kernel**

Think of every training point creating a "bubble of influence."

### **Nearby points**

Distance small

Similarity ≈ 1

### **Far points**

Distance large

Similarity ≈ 0

Thus RBF naturally creates curved decision boundaries.

---

# **Understanding Gamma (γ)**

Gamma determines how far the influence of a training point reaches.

---

## **Small Gamma**

Large influence region

Smooth decision boundary.

Better generalization.

May underfit.

---

## **Large Gamma**

Small influence region

Complex decision boundary.

May overfit.

---

# **Why RBF Is Popular**

* Handles highly non-linear data  
* Works well without knowing data structure  
* Flexible decision boundaries  
* Often performs best by default

---

# **Types of Kernel Functions**

---

# **1\. Linear Kernel**

Used when data is linearly separable.

Formula:

### **Characteristics**

* Fastest kernel  
* Simple  
* Good for high-dimensional datasets

### **Applications**

* Text classification  
* Document categorization

---

# **2\. Polynomial Kernel**

Creates curved boundaries.

Formula:

Where:

* (d) \= polynomial degree

---

### **Example**

Degree \= 2

Produces:

Quadratic curves

Degree \= 3

Produces:

Cubic curves

---

### **Applications**

* Image processing  
* Pattern recognition

---

# **3\. Gaussian RBF Kernel**

Formula:

### **Applications**

* Face recognition  
* Medical diagnosis  
* Complex classification tasks

---

# **4\. Sigmoid Kernel**

Inspired by neural networks.

Formula:

---

### **Applications**

* Neural-network-like decision boundaries

---

# **Comparison of Kernel Functions**

| Kernel | Best For |
| ----- | ----- |
| Linear | Linear data |
| Polynomial | Moderately non-linear data |
| RBF | Highly non-linear data |
| Sigmoid | Neural-network style problems |

---

# **Non-Linear Support Vector Regression (Non-Linear SVR)**

Earlier SVR used:

which creates a straight regression line.

But many datasets are non-linear.

Examples:

* Stock prices  
* Population growth  
* Weather forecasting

---

# **Intuition of Non-Linear SVR**

Suppose data looks like:

     \*  
   \*  
 \*  
      \*  
          \*

A straight line cannot fit it properly.

Non-Linear SVR maps data into higher dimensions using kernels and then finds the best regression hyperplane.

---

# **Kernelized SVR**

Instead of:

\[  
x  
\]

SVR uses:

\[  
\\phi(x)  
\]

and solves regression in higher-dimensional space.

Common kernels:

* RBF  
* Polynomial  
* Sigmoid

---

# **Why RBF Kernel Is Commonly Used in SVR**

RBF can:

* Capture complex patterns  
* Fit smooth curves  
* Handle non-linear relationships

Hence most practical SVR implementations use:

SVR(kernel='rbf')

---

# **Parameters in SVR**

## **1\. C (Regularization)**

Controls trade-off between:

* Margin size  
* Prediction error

### **Large C**

* Low training error  
* Risk of overfitting

### **Small C**

* Better generalization  
* More tolerance to errors

---

## **2\. Epsilon (ε)**

Width of the SVR tube.

Points inside this tube are ignored.

---

## **3\. Gamma (γ)**

Only for RBF kernel.

Controls influence region of support vectors.

---

# **Difference Between Linear SVR and Non-Linear SVR**

| Basis | Linear SVR | Non-Linear SVR |
| ----- | ----- | ----- |
| Relationship | Linear | Non-linear |
| Kernel | Linear | RBF/Polynomial/Sigmoid |
| Complexity | Lower | Higher |
| Accuracy on Complex Data | Lower | Higher |

---

# **Conclusion**

Kernel SVM extends traditional SVM to solve non-linear problems using kernel functions. The **Kernel Trick** allows SVM to operate in higher-dimensional spaces without explicitly performing expensive transformations. Among all kernels, the **Gaussian RBF Kernel** is the most popular because it can model highly complex decision boundaries.

Similarly, **Non-Linear SVR** uses kernels to perform regression on non-linear datasets, making it highly effective for forecasting, prediction, and curve-fitting tasks where linear models fail.

# **Logistic Regression**

Logistic Regression is a **Supervised Machine Learning Classification Algorithm** used when the dependent variable is categorical.

Despite the name "Regression", it is mainly used for **classification problems**.

Examples:

* Spam / Not Spam  
* Pass / Fail  
* Disease / No Disease  
* Fraud / Genuine

---

# **Why Not Use Linear Regression for Classification?**

Suppose we want to predict whether a student passes or fails.

| Study Hours | Result |
| ----- | ----- |
| 2 | 0 |
| 4 | 0 |
| 6 | 1 |
| 8 | 1 |

A Linear Regression model may predict:

\-0.3  
1.2  
0.7

These outputs are invalid because probabilities must lie between:

0 ≤ Probability ≤ 1

Therefore we need a function that always produces values between 0 and 1\.

---

# **Intuition Behind Logistic Regression**

Instead of predicting a class directly, Logistic Regression predicts the **probability** of belonging to a class.

Example:

P(Pass) \= 0.92

Since probability \> 0.5

Prediction:

Pass

---

# **The Sigmoid Function**

Logistic Regression uses the **Sigmoid Function**.

The sigmoid converts any real number into a probability between 0 and 1\.

Formula:

\\sigma(z)=\\frac{1}{1+e^{-z}}

where

\[  
z=b\_0+b\_1X\_1+b\_2X\_2+\\cdots+b\_nX\_n  
\]

---

# **Behavior of Sigmoid Function**

| z | Sigmoid Output |
| ----- | ----- |
| \-∞ | 0 |
| 0 | 0.5 |
| \+∞ | 1 |

Thus:

* Large negative values → Class 0  
* Large positive values → Class 1  
* Near zero → Uncertain

---

# **Logistic Regression Equation**

First calculate:

z=b\_0+b\_1X\_1+b\_2X\_2+\\cdots+b\_nX\_n

Then apply sigmoid:

P(Y=1)=\\frac{1}{1+e^{-z}}

This gives the probability that the observation belongs to Class 1\.

---

# **Decision Boundary**

Usually:

If P ≥ 0.5 → Class 1  
If P \< 0.5 → Class 0

The value 0.5 is called the **Decision Boundary**.

---

# **Odds and Log-Odds**

Logistic Regression models the **log-odds** rather than the probability directly.

### **Odds**

Odds=\\frac{P}{1-P}

Example:

P \= 0.8

Odds:

0.8 / 0.2 \= 4

Meaning success is 4 times more likely than failure.

---

### **Log-Odds (Logit)**

\\log\\left(\\frac{P}{1-P}\\right)=b\_0+b\_1X\_1+\\cdots+b\_nX\_n

This is the actual equation Logistic Regression learns.

---

# **Maximum Likelihood Estimation (MLE)**

One of the most important concepts in Logistic Regression is **Maximum Likelihood Estimation (MLE)**.

Unlike Linear Regression, Logistic Regression does not use Least Squares.

---

# **Why Not Least Squares?**

Linear Regression minimizes:

\\sum(y\_i-\\hat{y\_i})^2

This works well for continuous outputs.

But Logistic Regression predicts probabilities.

Using least squares:

* Produces non-convex optimization  
* Gives poor probability estimates  
* Can lead to inefficient training

Therefore Logistic Regression uses MLE.

---

# **Intuition Behind Maximum Likelihood**

Suppose we have two models.

### **Model A**

| Actual | Predicted Probability |
| ----- | ----- |
| 1 | 0.95 |
| 0 | 0.03 |
| 1 | 0.90 |

### **Model B**

| Actual | Predicted Probability |
| ----- | ----- |
| 1 | 0.55 |
| 0 | 0.45 |
| 1 | 0.60 |

Clearly Model A is better.

MLE tries to find parameter values that make the observed training data **most likely**.

---

# **Likelihood for One Observation**

If:

y \= 1

Probability:

\[  
P(y=1)=p  
\]

If:

y \= 0

Probability:

\[  
P(y=0)=1-p  
\]

Combined:

L=p^y(1-p)^{(1-y)}

---

# **Likelihood for Entire Dataset**

For n observations:

L=\\prod\_{i=1}^{n}p\_i^{y\_i}(1-p\_i)^{(1-y\_i)}

where:

* (p\_i) \= predicted probability  
* (y\_i) \= actual label

Goal:

Maximize L

---

# **Log-Likelihood**

Products are difficult to optimize.

Taking logarithms converts multiplication into addition.

\\log L=\\sum\_{i=1}^{n}\\left\[y\_i\\log(p\_i)+(1-y\_i)\\log(1-p\_i)\\right\]

Now optimization becomes easier.

---

# **Cost Function (Cross Entropy Loss)**

Instead of maximizing log-likelihood, we minimize its negative.

This gives the Logistic Regression Cost Function:

J(\\theta)=-\\frac{1}{n}\\sum\_{i=1}^{n}\\left\[y\_i\\log(p\_i)+(1-y\_i)\\log(1-p\_i)\\right\]

This is also called **Binary Cross Entropy Loss**.

---

# **Advantages of Logistic Regression**

* Simple and interpretable  
* Fast training  
* Outputs probabilities  
* Works well for binary classification  
* Less prone to overfitting

---

# **Limitations**

* Assumes linear decision boundary  
* Struggles with highly non-linear data  
* Sensitive to outliers  
* Performance depends on feature engineering

---

# **Applications**

* Spam Detection  
* Fraud Detection  
* Disease Prediction  
* Customer Churn Prediction  
* Credit Risk Analysis  
* Loan Approval Systems

---

# **Python Example**

from sklearn.linear\_model import LogisticRegression

classifier \= LogisticRegression()

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

# **Difference Between Linear and Logistic Regression**

| Feature | Linear Regression | Logistic Regression |
| ----- | ----- | ----- |
| Output | Continuous Value | Probability/Class |
| Problem Type | Regression | Classification |
| Function | Straight Line | Sigmoid Curve |
| Range | (-∞, \+∞) | (0,1) |
| Optimization | Least Squares | Maximum Likelihood |

---

# **Conclusion**

Logistic Regression is one of the most widely used classification algorithms. It predicts class probabilities using the **Sigmoid Function** and classifies observations using a decision boundary. The model parameters are learned through **Maximum Likelihood Estimation (MLE)**, which finds the coefficients that maximize the probability of observing the training data. This makes Logistic Regression highly effective for binary classification tasks such as spam detection, fraud detection, and medical diagnosis.

