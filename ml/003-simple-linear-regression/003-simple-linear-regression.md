# 6

# **Simple Linear Regression**

Simple Linear Regression is a supervised Machine Learning algorithm used to predict a dependent variable using a single independent variable. It establishes a linear relationship between the variables.

It is mainly used when the relationship between variables is approximately linear.

---

# **Example**

Suppose we want to predict a student’s marks based on study hours.

| Study Hours (X) | Marks (Y) |
| ----- | ----- |
| 1 | 20 |
| 2 | 40 |
| 3 | 50 |
| 4 | 70 |
| 5 | 90 |

Here:

* **Independent Variable (X)** → Study Hours  
* **Dependent Variable (Y)** → Marks

As study hours increase, marks also increase.

---

# **Equation of Simple Linear Regression**

The relationship between variables is represented by:

Y=b\_0+b\_1X

Where:

* (Y) \= Dependent variable  
* (X) \= Independent variable  
* (b\_0) \= Intercept  
* (b\_1) \= Slope (coefficient)

---

# **Understanding the Equation**

## **1\. Intercept ((b\_0))**

The value of (Y) when (X \= 0).

## **2\. Slope ((b\_1))**

Represents how much (Y) changes when (X) increases by 1 unit.

* Positive slope → Increasing relationship  
* Negative slope → Decreasing relationship

---

# **Working of Simple Linear Regression**

1. Collect dataset  
2. Plot data points  
3. Find the best-fit line  
4. Minimize prediction error  
5. Use the line for prediction

The best-fit line is obtained using the **Least Squares Method**, which minimizes the error between actual and predicted values.

---

# **Graph Representation**

The regression line can be represented as:

y=mx+c![][image2]

Where:

* (m) \= slope  
* (c) \= intercept

The line passes through the data points as closely as possible.

---

# **Advantages of Simple Linear Regression**

* Easy to understand  
* Fast to train  
* Useful for prediction  
* Helps identify relationships between variables

---

# **Limitations**

* Works only with linear relationships  
* Sensitive to outliers  
* Cannot model complex patterns

---

# **Applications**

* Predicting house prices  
* Sales forecasting  
* Stock trend analysis  
* Student performance prediction  
* Salary prediction based on experience

---

# **Python Example**

import pandas as pd  
from sklearn.model\_selection import train\_test\_split  
from sklearn.linear\_model import LinearRegression

\# Dataset  
dataset \= pd.read\_csv('data.csv')

X \= dataset.iloc\[:, :-1\].values  
y \= dataset.iloc\[:, \-1\].values

\# Splitting dataset  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
    X, y, test\_size=0.2, random\_state=0  
)

\# Training model  
regressor \= LinearRegression()  
regressor.fit(X\_train, y\_train)

\# Prediction  
y\_pred \= regressor.predict(X\_test)

---

# **Conclusion**

Simple Linear Regression is one of the most basic and important Machine Learning algorithms. It predicts values using a straight-line relationship between one independent variable and one dependent variable.

It is widely used for prediction, trend analysis, and understanding relationships in data.

