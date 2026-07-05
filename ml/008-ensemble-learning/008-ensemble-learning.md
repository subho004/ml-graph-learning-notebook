# 11

# **Ensemble Learning in Machine Learning**

Ensemble Learning is a Machine Learning technique where multiple models are combined to improve overall prediction accuracy and performance.

Instead of relying on a single model, ensemble methods combine predictions from several models to produce a stronger and more reliable model.

---

# **Intuition Behind Ensemble Learning**

Suppose a student asks five teachers the answer to a difficult question.

* One teacher may make mistakes.  
* But combining opinions from all teachers usually gives a better answer.

Similarly, Ensemble Learning combines multiple weak or average models to create a strong model.

---

# **Why Ensemble Learning?**

A single model may:

* Overfit  
* Underfit  
* Make unstable predictions

Ensemble methods help by:

* Reducing variance  
* Reducing bias  
* Improving accuracy  
* Increasing robustness

---

# **Types of Ensemble Learning**

## **1\. Bagging (Bootstrap Aggregating)**

* Multiple models are trained independently on random subsets of data.  
* Final prediction is obtained by averaging or voting.

Example:

* Random Forest

---

## **2\. Boosting**

* Models are trained sequentially.  
* Each new model corrects errors of previous models.

Examples:

* AdaBoost  
* Gradient Boosting  
* XGBoost

---

## **3\. Stacking**

* Combines predictions from multiple different models.  
* A meta-model gives final prediction.

---

# **Random Forest**

Random Forest is an Ensemble Learning algorithm based on Decision Trees.

It combines multiple Decision Trees and produces predictions using:

* Majority voting (classification)  
* Averaging (regression)

It is one of the most popular and powerful ML algorithms.

---

# **Intuition Behind Random Forest**

A single Decision Tree may overfit.

Random Forest solves this by creating many Decision Trees on different subsets of data and combining their predictions.

Instead of depending on one tree, the forest makes collective decisions.

---

# **Working of Random Forest**

## **Step 1: Bootstrap Sampling**

Random subsets of training data are created.

Example:

* Dataset of 1000 rows  
* Each tree gets a random sample

---

## **Step 2: Build Multiple Decision Trees**

Each tree is trained independently using:

* Different data samples  
* Random subset of features

---

## **Step 3: Prediction**

### **For Classification**

Use majority voting.

Example:

| Tree | Prediction |
| ----- | ----- |
| Tree 1 | Yes |
| Tree 2 | No |
| Tree 3 | Yes |

Final prediction → **Yes**

---

### **For Regression**

Take average prediction.

Example:

| Tree | Prediction |
| ----- | ----- |
| Tree 1 | 50 |
| Tree 2 | 55 |
| Tree 3 | 60 |

Final prediction:

---

# **Why Random Forest Works Well**

Different trees make different mistakes.

Combining them reduces:

* Variance  
* Overfitting  
* Noise sensitivity

This improves generalization.

---

# **Important Parameters in Random Forest**

## **1\. Number of Trees**

More trees usually improve performance but increase computation.

---

## **2\. Maximum Depth**

Controls tree complexity.

---

## **3\. Number of Features**

Random subset of features used during splitting.

---

# **Advantages of Random Forest**

* High accuracy  
* Reduces overfitting  
* Handles large datasets  
* Works for classification and regression  
* Handles missing values well  
* Robust to noise

---

# **Limitations**

* Computationally expensive  
* Slower training for large forests  
* Less interpretable than single Decision Tree

---

# **Difference Between Decision Tree and Random Forest**

| Basis | Decision Tree | Random Forest |
| ----- | ----- | ----- |
| Number of Trees | Single | Multiple |
| Overfitting | High | Lower |
| Accuracy | Lower | Higher |
| Stability | Less stable | More stable |
| Training Speed | Faster | Slower |

---

# **Random Forest for Classification**

Used for:

* Spam detection  
* Fraud detection  
* Disease prediction

Prediction is based on majority voting.

---

# **Random Forest for Regression**

Used for:

* Stock prediction  
* House price prediction  
* Sales forecasting

Prediction is based on averaging outputs.

---

# **Python Example**

## **Random Forest Classification**

from sklearn.ensemble import RandomForestClassifier

classifier \= RandomForestClassifier(  
    n\_estimators=100,  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

---

## **Random Forest Regression**

from sklearn.ensemble import RandomForestRegressor

regressor \= RandomForestRegressor(  
    n\_estimators=100,  
    random\_state=0  
)

regressor.fit(X\_train, y\_train)

---

# **Applications of Random Forest**

* Recommendation systems  
* Medical diagnosis  
* Fraud detection  
* Stock market analysis  
* Customer churn prediction

---

# **Conclusion**

Ensemble Learning improves Machine Learning performance by combining multiple models.

Random Forest is a powerful ensemble method that combines many Decision Trees to achieve:

* Better accuracy  
* Reduced overfitting  
* More stable predictions

It is widely used in both classification and regression problems due to its robustness and effectiveness.

