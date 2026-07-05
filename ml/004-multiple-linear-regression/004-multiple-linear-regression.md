# 7

# **Multiple Linear Regression**

Multiple Linear Regression is a supervised Machine Learning algorithm used to predict a dependent variable using two or more independent variables.

It extends Simple Linear Regression by considering multiple factors that influence the output.

---

# **Example**

Suppose we want to predict house prices based on:

* Area  
* Number of bedrooms  
* Age of house

| Area (sq ft) | Bedrooms | Age | Price |
| ----- | ----- | ----- | ----- |
| 1200 | 2 | 5 | 50L |
| 1500 | 3 | 3 | 65L |
| 1800 | 4 | 2 | 80L |

Here:

* **Independent Variables (X)**:  
  * Area  
  * Bedrooms  
  * Age  
* **Dependent Variable (Y)**:  
  * House Price

---

# **Equation of Multiple Linear Regression**

The relationship is represented by:

Where:

* (Y) \= Dependent variable  
* (X\_1, X\_2, X\_3 ... X\_n) \= Independent variables  
* (b\_0) \= Intercept  
* (b\_1, b\_2, ... b\_n) \= Coefficients

---

# **Understanding the Equation**

Each coefficient represents the effect of one feature on the output while keeping other features constant.

Example:

* (b\_1) → Effect of area on house price  
* (b\_2) → Effect of bedrooms  
* (b\_3) → Effect of house age

---

# **Assumptions of Multiple Linear Regression**

1. Linear relationship between variables  
2. No multicollinearity between independent variables  
3. Homoscedasticity (constant variance)  
4. Normally distributed residuals  
5. Independence of observations

---

# **Working of Multiple Linear Regression**

1. Collect dataset  
2. Preprocess data  
3. Split into training and testing sets  
4. Train regression model  
5. Find best-fit hyperplane  
6. Predict output values

The model tries to minimize prediction error using the Least Squares Method.

---

# **Advantages**

* Handles multiple features  
* Better prediction accuracy  
* Easy to interpret  
* Useful for trend analysis

---

# **Limitations**

* Sensitive to outliers  
* Assumes linear relationship  
* Multicollinearity can reduce performance  
* Complex with too many variables

---

# **Applications**

* House price prediction  
* Sales forecasting  
* Weather prediction  
* Risk analysis  
* Financial forecasting

---

# **Python Example**

import pandas as pd  
from sklearn.model\_selection import train\_test\_split  
from sklearn.linear\_model import LinearRegression

\# Import dataset  
dataset \= pd.read\_csv('data.csv')

\# Independent variables  
X \= dataset.iloc\[:, :-1\].values

\# Dependent variable  
y \= dataset.iloc\[:, \-1\].values

\# Split dataset  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
    X, y, test\_size=0.2, random\_state=0  
)

\# Train model  
regressor \= LinearRegression()  
regressor.fit(X\_train, y\_train)

\# Predict results  
y\_pred \= regressor.predict(X\_test)

---

# **Difference Between Simple and Multiple Linear Regression**

| Basis | Simple Linear Regression | Multiple Linear Regression |
| ----- | ----- | ----- |
| Number of Inputs | One | Two or more |
| Equation | (Y=b\_0+b\_1X) | (Y=b\_0+b\_1X\_1+b\_2X\_2+\\cdots) |
| Complexity | Simple | More complex |
| Prediction Accuracy | Lower | Higher |

---

# **Conclusion**

Multiple Linear Regression is used when multiple independent variables affect the dependent variable. It helps improve prediction accuracy by considering several influencing factors together.

It is widely used in business analytics, finance, healthcare, and prediction systems.

# **Five Methods of Building a Machine Learning Model**

In Multiple Linear Regression, different methods are used to select the best set of independent variables (features) for building an efficient model.

These methods help:

* Improve accuracy  
* Reduce overfitting  
* Remove unnecessary variables  
* Simplify the model

---

# **1\. All-in Method**

In the All-in method, all independent variables are included in the model at once.

No feature selection is performed.

---

## **Equation**

---

## **Example**

Suppose we want to predict salary using:

* Experience  
* Education  
* Age  
* Skills

All features are directly used in the model.

---

## **Advantages**

* Simple implementation  
* No feature removal required  
* Useful when domain knowledge confirms all variables are important

---

## **Limitations**

* May include irrelevant variables  
* Can reduce model performance  
* Higher computational cost

---

# **2\. Backward Elimination**

Backward Elimination starts with all features and removes the least significant feature step by step.

It uses p-values to determine feature importance.

---

## **Steps**

1. Train model with all features  
2. Calculate p-values  
3. Remove feature with highest p-value  
4. Retrain model  
5. Repeat until all p-values are below significance level

---

## **Decision Rule**

Where:

* (SL) \= Significance Level  
* Usually (SL \= 0.05)

---

## **Example**

| Feature | p-value |
| ----- | ----- |
| Age | 0.60 |
| Experience | 0.01 |

Remove **Age** because p-value \> 0.05.

---

## **Advantages**

* Removes irrelevant variables  
* Produces optimized model

## **Limitations**

* Computationally expensive  
* Slow for large datasets

---

# **3\. Forward Selection**

Forward Selection starts with no features and adds the most significant features one by one.

---

## **Steps**

1. Start with empty model  
2. Add best feature  
3. Evaluate model performance  
4. Continue adding useful features

---

## **Example**

Step 1:

* Experience gives highest accuracy → Add it

Step 2:

* Add Education if performance improves

Final selected features:

* Experience  
* Education

---

## **Advantages**

* Faster than backward elimination  
* Simpler model

## **Limitations**

* May miss best feature combinations

---

# **4\. Bidirectional Elimination**

Bidirectional Elimination is a combination of:

* Forward Selection  
* Backward Elimination

Features can be both added and removed during model building.

---

## **Working**

1. Add important feature  
2. Check significance of existing features  
3. Remove insignificant features  
4. Repeat process

---

## **Example**

* Add Experience  
* Add Education  
* Later remove Age if it becomes insignificant

---

## **Advantages**

* More flexible  
* Better feature optimization  
* Often produces more accurate models

---

## **Limitations**

* More computationally expensive  
* Complex implementation

---

# **5\. Score Comparison**

In Score Comparison, different combinations of features are tested and compared using evaluation metrics.

The model with the best score is selected.

---

## **Common Evaluation Metrics**

* Accuracy  
* R² Score  
* Adjusted R²  
* Mean Squared Error (MSE)  
* AIC/BIC

---

## **Example**

| Model | R² Score |
| ----- | ----- |
| Model A | 0.78 |
| Model B | 0.85 |
| Model C | 0.81 |

Select **Model B** because it has highest score.

---

## **Advantages**

* Helps find best-performing model  
* Useful for comparing multiple models

---

## **Limitations**

* Time-consuming  
* Requires repeated training

---

# **Difference Between the Methods**

| Method | Starting Point | Action |
| ----- | ----- | ----- |
| All-in | All features | Use everything |
| Backward Elimination | All features | Remove features |
| Forward Selection | No features | Add features |
| Bidirectional Elimination | Mixed | Add & remove |
| Score Comparison | Multiple models | Compare scores |

---

# **Conclusion**

The five methods used in model building help select the best features and improve model performance.

* **All-in** uses every feature.  
* **Backward Elimination** removes unimportant features.  
* **Forward Selection** adds useful features gradually.  
* **Bidirectional Elimination** combines adding and removing.  
* **Score Comparison** selects the best model using evaluation metrics.

These techniques are widely used in Multiple Linear Regression and Machine Learning model optimization.

