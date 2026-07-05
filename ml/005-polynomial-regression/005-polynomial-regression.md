# 8

# **Polynomial Regression**

Polynomial Regression is a supervised Machine Learning algorithm used to model non-linear relationships between variables.

It is an extension of Linear Regression where the relationship between the independent variable and dependent variable is represented using polynomial terms.

Although called “linear regression,” the curve fitted is non-linear.

---

# **Why Polynomial Regression?**

Simple Linear Regression works well only when data follows a straight-line relationship.

However, many real-world datasets are non-linear.

### **Example**

* Salary vs Experience  
* Population growth  
* Temperature variation  
* Stock market trends

In such cases, Polynomial Regression gives better predictions.

---

# **Example**

Suppose we want to predict salary based on experience.

| Experience | Salary |
| ----- | ----- |
| 1 | 30K |
| 2 | 35K |
| 3 | 50K |
| 4 | 80K |
| 5 | 130K |

The relationship is curved, not linear.

A straight line cannot fit the data properly.

---

# **Polynomial Regression Equation**

The equation is:

Where:

* (Y) \= Dependent variable  
* (X) \= Independent variable  
* (b\_0, b\_1, b\_2 ... b\_n) \= Coefficients  
* (n) \= Degree of polynomial

---

# **Understanding Polynomial Degree**

## **Degree 1**

This is Simple Linear Regression.

---

## **Degree 2 (Quadratic)**

Produces a curved graph.

---

## **Degree 3 (Cubic)**

Can model more complex curves.

---

# **Working of Polynomial Regression**

1. Import dataset  
2. Convert features into polynomial terms  
3. Train Linear Regression model on transformed data  
4. Predict results  
5. Plot regression curve

---

# **Advantages**

* Handles non-linear data  
* Better fitting for curved relationships  
* Higher prediction accuracy for complex datasets

---

# **Limitations**

* High-degree polynomial may overfit  
* Computationally expensive  
* Sensitive to outliers

---

# **Applications**

* Stock market analysis  
* Population prediction  
* Weather forecasting  
* Sales trend prediction  
* Growth rate analysis

---

# **Difference Between Linear and Polynomial Regression**

| Basis | Linear Regression | Polynomial Regression |
| ----- | ----- | ----- |
| Relationship | Linear | Non-linear |
| Graph | Straight line | Curved line |
| Complexity | Simple | More complex |
| Accuracy on Curved Data | Lower | Higher |

---

# **Python Example**

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear\_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures

\# Polynomial transformation  
poly\_reg \= PolynomialFeatures(degree=4)  
X\_poly \= poly\_reg.fit\_transform(X)

\# Train model  
lin\_reg \= LinearRegression()  
lin\_reg.fit(X\_poly, y)

\# Prediction  
y\_pred \= lin\_reg.predict(X\_poly)

---

# **Graph Representation**

Polynomial Regression produces a curve that better fits non-linear data compared to a straight regression line.

Example curve:

---

# **Conclusion**

Polynomial Regression is used when data has a non-linear relationship. It transforms linear models into curved models using polynomial terms.

It provides better accuracy for complex datasets and is widely used in forecasting, trend analysis, and predictive modeling.

