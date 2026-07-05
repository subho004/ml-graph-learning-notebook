# 12

# **R-Squared ((R^2)) and Adjusted R-Squared in Machine Learning**

R-Squared and Adjusted R-Squared are evaluation metrics used mainly in Regression models to measure how well the model fits the data.

They indicate how much variation in the dependent variable is explained by the independent variables.

---

# **1\. R-Squared ((R^2))**

R-Squared is also called the **Coefficient of Determination**.

It measures the proportion of variance in the dependent variable that is explained by the regression model.

---

# **Formula of R-Squared**

Where:

* (SS\_{res}) \= Sum of Squared Residuals (errors)  
* (SS\_{tot}) \= Total Sum of Squares

---

# **Understanding R-Squared**

| (R^2) Value | Meaning |
| ----- | ----- |
| 0 | Model explains nothing |
| 1 | Perfect prediction |
| 0.8 | Model explains 80% variance |

---

# **Example**

Suppose we predict student marks using study hours.

If:

then:

* 85% variation in marks is explained by study hours.  
* Remaining 15% depends on other factors.

---

# **Intuition Behind R-Squared**

Higher (R^2) means:

* Better fitting model  
* Predictions are closer to actual values

Lower (R^2) means:

* Poor fitting model  
* Model cannot explain data properly

---

# **Problem with R-Squared**

R-Squared always increases when new features are added, even if those features are irrelevant.

This can lead to overfitting.

---

# **2\. Adjusted R-Squared**

Adjusted R-Squared solves the limitation of R-Squared.

It penalizes unnecessary features.

It increases only when a new feature actually improves the model.

---

# **Formula of Adjusted R-Squared**

Where:

* (n) \= Number of observations  
* (k) \= Number of independent variables

---

# **Intuition Behind Adjusted R-Squared**

* If a new feature improves the model significantly:  
  * Adjusted (R^2) increases  
* If the feature is irrelevant:  
  * Adjusted (R^2) decreases

Thus, it helps select meaningful variables.

---

# **Example**

Suppose:

| Model | R² | Adjusted R² |
| ----- | ----- | ----- |
| 2 Features | 0.85 | 0.84 |
| 10 Features | 0.90 | 0.78 |

Although R² increased, Adjusted R² decreased because many added features were unnecessary.

---

# **Difference Between R-Squared and Adjusted R-Squared**

| Basis | R-Squared | Adjusted R-Squared |
| ----- | ----- | ----- |
| Considers Number of Features | No | Yes |
| Always Increases with New Features | Yes | No |
| Penalizes Irrelevant Variables | No | Yes |
| Better for Multiple Regression | No | Yes |

---

# **When to Use**

## **Use R-Squared**

* Simple Linear Regression  
* Quick performance understanding

## **Use Adjusted R-Squared**

* Multiple Linear Regression  
* Feature selection problems  
* Comparing models with different numbers of features

---

# **Python Example**

from sklearn.metrics import r2\_score

\# R-Squared  
r2 \= r2\_score(y\_test, y\_pred)

print(r2)

For Adjusted R²:

adjusted\_r2 \= 1 \- (  
    (1 \- r2) \* (len(y\_test) \- 1\)  
    / (len(y\_test) \- X\_test.shape\[1\] \- 1\)  
)

print(adjusted\_r2)

---

# **Advantages of R-Squared Metrics**

* Easy interpretation  
* Helps evaluate regression models  
* Useful for model comparison

---

# **Limitations**

* Does not indicate causation  
* Cannot detect overfitting alone  
* High R² does not always mean good model

---

# **Conclusion**

* **R-Squared** measures how well the regression model explains variance in the data.  
* **Adjusted R-Squared** improves upon R² by penalizing unnecessary features.

Adjusted R² is generally preferred for Multiple Linear Regression because it provides a more reliable measure of model quality.

