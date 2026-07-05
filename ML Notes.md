# 2

## **Data Preprocessing in Machine Learning**

Data preprocessing is the process of cleaning and transforming raw data before feeding it into a Machine Learning model. Real-world data is often incomplete, noisy, inconsistent, or in different formats, so preprocessing helps improve model accuracy and performance.

### **Common Steps in Data Preprocessing**

1. **Handling Missing Values**  
   * Missing data can be removed or filled using mean, median, mode, or interpolation.  
   * Example:  
     * Age column: `[21, 22, null, 24]`  
     * Replace `null` with average age → `[21, 22, 22.3, 24]`  
2. **Removing Duplicates**  
   * Duplicate rows can bias the model.

Example:  
ID  Name  
1   Ram  
1   Ram

* Remove one duplicate entry.  
3. **Encoding Categorical Data**  
   * ML models work with numbers, so text labels are converted into numeric form.  
   * Example:  
     * Gender: `Male, Female`  
     * Encoded as: `0, 1`  
4. **Feature Scaling**  
   * Different features may have different ranges.  
   * Techniques:  
     * Normalization  
     * Standardization  
   * Example:  
     * Salary: `50000`  
     * Age: `22`  
       Without scaling, salary dominates the model.  
5. **Handling Outliers**  
   * Extreme values can negatively affect model training.  
   * Example:  
     * Salaries: `[30000, 35000, 40000, 1000000]`  
     * `1000000` is an outlier and may be removed or capped.  
6. **Data Transformation**  
   * Convert data into a suitable format.  
   * Example:  
     * Convert date `"17-05-2026"` into separate day, month, year columns.  
7. **Feature Selection**  
   * Selecting important columns/features for training.  
   * Example:  
     * For house price prediction, useful features may be:  
       * Area  
       * Number of rooms  
       * Location

---

## **Example Workflow**

Suppose we want to predict student exam performance.

### **Raw Data**

| Name | Age | Gender | Study Hours | Marks |
| ----- | ----- | ----- | ----- | ----- |
| A | 20 | Male | 5 | 80 |
| B | null | Female | 3 | 65 |
| C | 21 | Male | 50 | 90 |

### **After Preprocessing**

* Replace missing age with average.  
* Encode gender:  
  * Male → 0  
  * Female → 1  
* Detect outlier in study hours (`50`).  
* Scale numerical values.

### **Final Processed Data**

| Age | Gender | Study Hours | Marks |
| ----- | ----- | ----- | ----- |
| 20 | 0 | 0.5 | 80 |
| 20.5 | 1 | 0.3 | 65 |
| 21 | 0 | 1.0 | 90 |

---

## **Importance of Data Preprocessing**

* Improves model accuracy  
* Reduces training time  
* Handles noisy/incomplete data  
* Prevents bias in prediction  
* Makes data suitable for ML algorithms

In Machine Learning, good preprocessing often matters as much as choosing the right algorithm.

## **Importance of Training-Test Split in ML Model Evaluation**

In Machine Learning, the dataset is divided into two parts:

1. **Training Data**  
   * Used to train the model.  
   * The model learns patterns, relationships, and trends from this data.  
2. **Test Data**  
   * Used to evaluate the model’s performance on unseen data.  
   * Helps measure how well the model generalizes.

This process is called the **Training-Test Split**.

---

## **Why is Training-Test Split Important?**

### **1\. Prevents Overfitting**

If the model is tested on the same data it was trained on, it may memorize the data instead of learning patterns.

* Example:  
  * A student memorizes answers instead of understanding concepts.  
  * Scores high in practice but performs poorly in a new exam.

Similarly, an ML model may show high accuracy on training data but fail on new data.

---

### **2\. Measures Real Performance**

Test data simulates real-world unseen data.

* Example:  
  * Train model using old customer data.  
  * Test it using new customer data.  
  * This shows whether predictions will work in practice.

---

### **3\. Helps Compare Models**

Different ML algorithms can be evaluated fairly using the same test dataset.

* Example:  
  * Compare Decision Tree vs Random Forest.  
  * Choose the one with better test accuracy.

---

### **4\. Detects Underfitting and Overfitting**

| Condition | Training Accuracy | Test Accuracy |
| ----- | ----- | ----- |
| Good Fit | High | High |
| Overfitting | Very High | Low |
| Underfitting | Low | Low |

Training-test split helps identify these problems.

---

## **Common Split Ratios**

| Training Data | Test Data |
| ----- | ----- |
| 80% | 20% |
| 70% | 30% |
| 75% | 25% |

The ratio depends on dataset size.

---

## **Example**

Suppose we have **1000 student records**.

* Training Set \= 800 records  
* Test Set \= 200 records

### **Process**

1. Train the model on 800 records.  
2. Evaluate on 200 unseen records.  
3. Calculate accuracy, precision, recall, etc.

If the model performs well on test data, it is likely to work well in real-world scenarios.

---

## **Python Example**

from sklearn.model\_selection import train\_test\_split

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
    X, y, test\_size=0.2, random\_state=42  
)

Here:

* `test_size=0.2` means 20% data is used for testing.  
* `random_state=42` ensures reproducible results.

---

## **Conclusion**

Training-test split is an essential step in Machine Learning because it:

* Evaluates model generalization  
* Prevents overfitting  
* Helps compare models  
* Ensures reliable and unbiased performance measurement

Without proper splitting, model evaluation may give misleading results.

# **Feature Scaling in Machine Learning: Normalization vs Standardization**

Feature Scaling is a preprocessing technique used to bring numerical features into a similar range. Many Machine Learning algorithms perform better when input features are scaled properly.

For example, consider:

| Feature | Value |
| ----- | ----- |
| Age | 22 |
| Salary | 500000 |

Here, salary values are much larger than age values, so the model may give more importance to salary. Feature scaling solves this problem.

---

# **Why Feature Scaling is Important**

* Improves model accuracy  
* Speeds up training  
* Prevents features with large values from dominating  
* Important for algorithms like:  
  * KNN  
  * SVM  
  * Logistic Regression  
  * Neural Networks  
  * K-Means

---

# **Types of Feature Scaling**

## **1\. Normalization**

Normalization scales data into a fixed range, usually between **0 and 1**.

### **Formula**

X\_{norm}=\\frac{X-X\_{min}}{X\_{max}-X\_{min}}

### **Example**

Suppose values are:

\[10, 20, 30, 40, 50\]

For value `30`:

* Min \= 10  
* Max \= 50

Result:

(30 \- 10\) / (50 \- 10\) \= 0.5

### **Characteristics**

* Output range: 0 to 1  
* Sensitive to outliers  
* Works well when data distribution is not Gaussian

### **Applications**

* Image processing  
* Deep learning  
* Neural networks

---

## **2\. Standardization**

Standardization transforms data so that:

* Mean \= 0  
* Standard deviation \= 1

### **Formula**

Z=\\frac{X-\\mu}{\\sigma}	

![][image1]

Where:

* ( \\mu ) \= mean  
* ( \\sigma ) \= standard deviation

### **Example**

Suppose:

* Mean \= 50  
* Standard deviation \= 10  
* Value \= 70

Result:

(70 \- 50\) / 10 \= 2

### **Characteristics**

* Data is centered around zero  
* Less affected by outliers than normalization  
* Commonly used in ML algorithms

### **Applications**

* Logistic Regression  
* SVM  
* PCA  
* Linear Regression

---

# **Difference Between Normalization and Standardization**

| Basis | Normalization | Standardization |
| ----- | ----- | ----- |
| Range | 0 to 1 | No fixed range |
| Formula | Min-Max Scaling | Z-score Scaling |
| Outlier Sensitivity | High | Lower |
| Distribution | Non-Gaussian data | Gaussian-like data |
| Common Use | Deep Learning | Traditional ML models |

---

# **Python Example**

from sklearn.preprocessing import MinMaxScaler, StandardScaler

\# Normalization  
minmax \= MinMaxScaler()  
X\_norm \= minmax.fit\_transform(X)

\# Standardization  
standard \= StandardScaler()  
X\_std \= standard.fit\_transform(X)

---

# **Conclusion**

Feature scaling is an important preprocessing step in Machine Learning.

* **Normalization** scales values between 0 and 1\.  
* **Standardization** centers data around mean 0 with standard deviation 1\.

Choosing the correct scaling technique helps improve model performance and training efficiency.

# 3

# **Data Preprocessing Tools in Machine Learning**

Data preprocessing is an important step in Machine Learning used to clean, organize, and transform raw data into a suitable format for model training. Real-world datasets often contain missing values, categorical data, inconsistent formats, and features with different scales. Proper preprocessing improves model accuracy and efficiency.

---

# **Steps in Data Preprocessing**

## **1\. Importing the Libraries**

Libraries provide predefined functions for data analysis and Machine Learning.

### **Common Python Libraries**

* **NumPy** → Numerical operations  
* **Pandas** → Data handling  
* **Matplotlib** → Visualization  
* **Scikit-learn** → ML preprocessing and models

### **Example**

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt

---

# **2\. Importing the Dataset**

Datasets are usually imported from CSV or Excel files.

### **Example**

dataset \= pd.read\_csv('data.csv')  
print(dataset.head())

This loads the dataset into a dataframe.

---

# **3\. Taking Care of Missing Data**

Datasets may contain missing values represented by:

* Null  
* NaN  
* Empty cells

Missing values can affect model performance.

### **Methods**

* Remove rows/columns  
* Replace using:  
  * Mean  
  * Median  
  * Mode

### **Example**

from sklearn.impute import SimpleImputer

imputer \= SimpleImputer(strategy='mean')

### **Example Dataset**

| Age | Salary |
| ----- | ----- |
| 22 | 30000 |
| NaN | 40000 |
| 25 | 50000 |

After replacing missing age with mean:

| Age | Salary |
| ----- | ----- |
| 22 | 30000 |
| 23.5 | 40000 |
| 25 | 50000 |

---

# **4\. Encoding Categorical Data**

Machine Learning models work with numerical data, so categorical text values must be converted into numbers.

---

## **A. Encoding the Independent Variable**

Independent variables are input features.

### **Example**

| Country |
| ----- |
| India |
| USA |
| UK |

After One-Hot Encoding:

| India | USA | UK |
| ----- | ----- | ----- |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

### **Python Example**

from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import OneHotEncoder

---

## **B. Encoding the Dependent Variable**

Dependent variable is the target/output variable.

### **Example**

| Purchased |
| ----- |
| Yes |
| No |

Encoded as:

| Purchased |
| ----- |
| 1 |
| 0 |

### **Python Example**

from sklearn.preprocessing import LabelEncoder

le \= LabelEncoder()  
y \= le.fit\_transform(y)

---

# **5\. Splitting the Dataset into Training Set and Test Set**

The dataset is divided into:

* **Training Set** → Used for learning  
* **Test Set** → Used for evaluation

### **Common Ratios**

* 80% Training, 20% Testing  
* 70% Training, 30% Testing

### **Importance**

* Prevents overfitting  
* Evaluates real-world performance  
* Helps compare models

### **Python Example**

from sklearn.model\_selection import train\_test\_split

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
    X, y, test\_size=0.2, random\_state=42  
)

---

# **6\. Feature Scaling**

Features may have different ranges.

### **Example**

| Feature | Value |
| ----- | ----- |
| Age | 22 |
| Salary | 500000 |

Salary values dominate due to larger magnitude.

Feature scaling brings all values into a similar range.

---

## **Types of Feature Scaling**

### **A. Normalization**

Scales values between 0 and 1\.

Formula:

X\_{norm}=\\frac{X-X\_{min}}{X\_{max}-X\_{min}}

### **B. Standardization**

Centers data around mean 0 and standard deviation 1\.

Formula:

genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"Z=\\frac{X-\\mu}{\\sigma}"}}

### **Python Example**

from sklearn.preprocessing import StandardScaler

sc \= StandardScaler()  
X\_train \= sc.fit\_transform(X\_train)  
X\_test \= sc.transform(X\_test)

---

# **Conclusion**

Data preprocessing is a critical step in Machine Learning because it:

* Cleans raw data  
* Handles missing values  
* Converts categorical data into numerical form  
* Splits data for proper evaluation  
* Scales features for better performance

Proper preprocessing improves model accuracy, training efficiency, and overall reliability of Machine Learning systems.

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

# 9

# **Support Vector Machine (SVM) Regression**

Support Vector Machine Regression, also called **Support Vector Regression (SVR)**, is a supervised Machine Learning algorithm used for predicting continuous values.

Unlike Linear Regression, SVR tries to fit the best line within a certain error margin instead of minimizing only prediction error.

SVR works well for:

* Linear data  
* Non-linear data  
* High-dimensional datasets

---

# **Intuition Behind SVR**

In Linear Regression, the goal is to minimize the distance between actual and predicted points.

In SVR, the goal is different:

* Fit a line such that most points lie within a margin (boundary)  
* Ignore small errors inside the margin  
* Penalize only large errors outside the margin

---

# **SVR Concept**

SVR tries to create a tube or margin around the regression line.

This margin is controlled by:

called the **epsilon-insensitive region**.

Points inside the margin are ignored.

Only points outside the margin affect the model.

---

# **SVR Regression Function**

The regression equation is:

Where:

* (w) \= weight vector  
* (x) \= input feature vector  
* (b) \= bias

---

# **Key Idea of SVR**

The objective is:

* Maximize margin  
* Keep prediction error within epsilon  
* Use only important data points called **support vectors**

Support vectors are the points closest to the boundary.

---

# **Advantages of SVR**

* Handles high-dimensional data  
* Effective for non-linear problems  
* Good generalization performance  
* Robust against overfitting

---

# **Limitations**

* Computationally expensive for large datasets  
* Sensitive to kernel and parameter selection  
* Slower training time

---

# **Non-Linear SVR**

Sometimes data is not linearly separable.

Example:

* Curved stock trends  
* Population growth  
* Temperature prediction

A straight line cannot fit such data properly.

In these cases, **Non-Linear SVR** is used.

---

# **Intuition of Non-Linear SVR**

Instead of fitting a straight line in lower dimensions, SVR transforms data into a higher-dimensional space where it becomes linearly separable.

---

# **Example**

Suppose data forms a circular pattern in 2D.

It cannot be separated using a straight line.

After mapping into 3D space, the data may become separable using a plane.

---

# **Mapping to a Higher Dimension**

Data transformation can be represented as:

Where:

* (n) \= lower dimension  
* (m) \= higher dimension

This transformation helps model complex relationships.

---

# **Kernel SVM Intuition**

Computing higher-dimensional transformations directly is expensive.

To solve this, SVM uses the **Kernel Trick**.

A kernel function computes similarity between points without explicitly transforming them into higher dimensions.

This makes computations efficient.

---

# **The Kernel Trick**

Instead of calculating:

SVM directly computes:

Where:

* (K) \= Kernel Function

This avoids expensive computations.

---

# **Types of Kernel Functions**

## **1\. Linear Kernel**

Used when data is linearly separable.

Formula:

### **Applications**

* Text classification  
* High-dimensional data

---

# **2\. Polynomial Kernel**

Used for curved relationships.

Formula:

Where:

* (d) \= degree of polynomial

### **Applications**

* Image processing  
* Pattern recognition

---

# **3\. Radial Basis Function (RBF) Kernel**

Most commonly used kernel.

Handles highly non-linear data.

Formula:

Where:

* (\\gamma) controls influence of points

### **Applications**

* Complex classification problems  
* Medical diagnosis  
* Recommendation systems

---

# **4\. Sigmoid Kernel**

Behaves similarly to neural networks.

Formula:

### **Applications**

* Neural-network-like learning

---

# **Difference Between Linear SVR and Non-Linear SVR**

| Basis | Linear SVR | Non-Linear SVR |
| ----- | ----- | ----- |
| Data Type | Linear | Non-linear |
| Kernel | Linear | RBF/Polynomial/etc. |
| Complexity | Lower | Higher |
| Accuracy on Complex Data | Lower | Higher |

---

# **Python Example of SVR**

from sklearn.svm import SVR  
from sklearn.preprocessing import StandardScaler

\# Feature scaling  
sc\_X \= StandardScaler()  
sc\_y \= StandardScaler()

X \= sc\_X.fit\_transform(X)  
y \= sc\_y.fit\_transform(y.reshape(-1,1))

\# SVR model  
regressor \= SVR(kernel='rbf')  
regressor.fit(X, y.ravel())

---

# **Applications of SVR**

* Stock price prediction  
* Weather forecasting  
* Energy demand prediction  
* Risk analysis  
* Time-series forecasting

---

# **Conclusion**

Support Vector Regression (SVR) is a powerful regression algorithm that works for both linear and non-linear datasets.

Key concepts include:

* Margin-based prediction  
* Support vectors  
* Higher-dimensional mapping  
* Kernel trick  
* Kernel functions

SVR is widely used when complex relationships exist in data and high prediction accuracy is required.

# 10

# **Decision Tree in Machine Learning**

A Decision Tree is a supervised Machine Learning algorithm used for:

* Classification problems  
* Regression problems

It works like a tree structure where:

* Each internal node represents a decision/condition  
* Each branch represents an outcome  
* Each leaf node represents the final prediction

Decision Trees are easy to understand and visualize.

---

# **Intuition Behind Decision Tree**

A Decision Tree works similarly to human decision-making.

---

# **Example**

Suppose we want to decide whether a person will buy a car.

Questions may be:

1. Is salary greater than 50K?  
2. Is age less than 30?  
3. Does the person own a house?

Based on answers, the model reaches a final decision:

* Buy  
* Not Buy

The algorithm continuously splits the dataset into smaller groups based on the most important feature.

---

# **Structure of a Decision Tree**

## **Root Node**

The topmost node representing the entire dataset.

## **Decision Node**

A node where data is split based on a condition.

## **Leaf Node**

Final output or prediction.

---

# **Working of Decision Tree**

1. Select the best feature  
2. Split dataset based on that feature  
3. Repeat splitting recursively  
4. Stop when:  
   * Pure nodes are formed  
   * Maximum depth is reached  
   * No further improvement occurs

---

# **How Best Feature is Selected**

Decision Trees use measures like:

* Information Gain  
* Gini Index  
* Variance Reduction

to select the best split.

---

# **Decision Tree for Regression (Regression Trees)**

Regression Trees are used when the target variable is continuous.

Example:

* Predicting salary  
* House price prediction  
* Temperature forecasting

---

# **Intuition of Regression Trees**

Instead of predicting categories, Regression Trees divide data into regions and predict the average value of that region.

---

# **Example**

Suppose we predict salary based on experience.

| Experience | Salary |
| ----- | ----- |
| 1 | 30K |
| 2 | 35K |
| 3 | 50K |
| 4 | 80K |

The tree creates intervals like:

* Experience \< 2.5 → Salary \= 32K  
* Experience ≥ 2.5 → Salary \= 65K

Predictions are averages of regions.

---

# **Regression Tree Splitting**

Regression Trees minimize variance within each split.

The goal is to reduce:

where:

* (y\_i) \= actual values  
* (\\bar{y}) \= mean of region

---

# **Advantages of Regression Trees**

* Handles non-linear relationships  
* No feature scaling required  
* Easy interpretation

---

# **Limitations**

* Can overfit easily  
* Sensitive to small data changes  
* Lower accuracy compared to ensemble methods

---

# **Decision Tree for Classification (Classification Trees)**

Classification Trees are used when the target variable is categorical.

Example:

* Spam detection  
* Disease prediction  
* Fraud detection

---

# **Intuition of Classification Trees**

The tree splits data into classes based on feature conditions.

Example:

* Age \< 30 → Buy  
* Age ≥ 30 → Not Buy

Final output is a category/class label.

---

# **Measures Used in Classification Trees**

## **1\. Entropy**

Measures randomness or impurity.

Formula:

Lower entropy means purer nodes.

---

# **2\. Information Gain**

Measures reduction in entropy after splitting.

Formula:

The feature with highest Information Gain is selected.

---

# **3\. Gini Index**

Measures impurity of nodes.

Formula:

Lower Gini value indicates better split.

---

# **Difference Between Regression Tree and Classification Tree**

| Basis | Regression Tree | Classification Tree |
| ----- | ----- | ----- |
| Output | Continuous value | Category/Class |
| Example | Salary prediction | Spam detection |
| Split Criterion | Variance reduction | Entropy/Gini |
| Leaf Node | Numeric value | Class label |

---

# **Advantages of Decision Trees**

* Easy to understand  
* No feature scaling needed  
* Handles numerical and categorical data  
* Works for both regression and classification

---

# **Limitations of Decision Trees**

* Overfitting problem  
* Unstable with small data changes  
* Can become very complex

---

# **Python Example**

## **Classification Tree**

from sklearn.tree import DecisionTreeClassifier

classifier \= DecisionTreeClassifier()  
classifier.fit(X\_train, y\_train)

---

## **Regression Tree**

from sklearn.tree import DecisionTreeRegressor

regressor \= DecisionTreeRegressor()  
regressor.fit(X\_train, y\_train)

---

# **Applications of Decision Trees**

* Medical diagnosis  
* Fraud detection  
* Customer segmentation  
* Loan approval systems  
* Sales prediction

---

# **Conclusion**

Decision Trees are powerful and interpretable Machine Learning algorithms used for both classification and regression tasks.

* **Classification Trees** predict categories/classes.  
* **Regression Trees** predict continuous numerical values.

They work by recursively splitting data into smaller groups using the most informative features.

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

# 17

# **K-Nearest Neighbors (KNN) Algorithm**

K-Nearest Neighbors (KNN) is a **Supervised Machine Learning** algorithm used for both:

* Classification  
* Regression

It is one of the simplest and most intuitive Machine Learning algorithms.

KNN makes predictions based on the similarity between data points.

---

# **Intuition Behind KNN**

The basic idea is:

"Similar data points tend to belong to the same category."

Suppose you move into a new neighborhood.

To predict your interests, one could look at your closest neighbors:

* If most neighbors like cricket, you probably do too.  
* If most neighbors like football, you may belong to that group.

KNN follows the same principle.

---

# **Example of KNN Classification**

Suppose we want to classify a fruit.

Training Data:

| Weight | Fruit |
| ----- | ----- |
| 100g | Apple |
| 110g | Apple |
| 120g | Apple |
| 180g | Orange |
| 190g | Orange |

New Fruit:

130g

KNN checks the nearest fruits.

Nearest neighbors:

120g → Apple  
110g → Apple  
180g → Orange

Majority class:

Apple

Prediction:

Apple

---

# **How KNN Works**

## **Step 1: Choose K**

Select the number of neighbors.

Example:

K \= 3

means consider the 3 nearest points.

---

## **Step 2: Calculate Distance**

Distance is calculated between the new point and all training points.

Most commonly using Euclidean Distance.

---

# **Euclidean Distance Formula**

For two points:

\[  
(x\_1,y\_1)  
\]

and

\[  
(x\_2,y\_2)  
\]

Distance:

---

# **Step 3: Find K Nearest Neighbors**

Sort distances.

Select the K smallest distances.

---

# **Step 4: Make Prediction**

### **Classification**

Use majority voting.

Example:

| Neighbor | Class |
| ----- | ----- |
| N1 | Red |
| N2 | Red |
| N3 | Blue |

Prediction:

Red

---

### **Regression**

Take average value.

Example:

| Neighbor | Price |
| ----- | ----- |
| N1 | 100 |
| N2 | 120 |
| N3 | 140 |

Prediction:

---

# **Choosing the Value of K**

## **Small K**

Example:

K \= 1

Advantages:

* Captures local patterns

Disadvantages:

* Sensitive to noise  
* Overfitting

---

## **Large K**

Example:

K \= 20

Advantages:

* More stable

Disadvantages:

* Underfitting  
* May ignore local structure

---

# **Rule for Choosing K**

A common choice:

Where:

* (N) \= Number of training samples

Usually choose an odd value for binary classification to avoid ties.

---

# **Why Feature Scaling is Important in KNN**

KNN is distance-based.

Consider:

| Feature | Value |
| ----- | ----- |
| Age | 25 |
| Salary | 500000 |

Salary dominates distance calculations.

Therefore normalization or standardization is usually required before applying KNN.

---

# **Advantages of KNN**

* Simple and easy to understand  
* No training phase (lazy learner)  
* Works for classification and regression  
* Handles non-linear decision boundaries

---

# **Disadvantages of KNN**

* Slow for large datasets  
* Requires feature scaling  
* Sensitive to irrelevant features  
* High memory usage

---

# **KNN Classification vs KNN Regression**

| Feature | KNN Classification | KNN Regression |
| ----- | ----- | ----- |
| Output | Class Label | Continuous Value |
| Prediction | Majority Vote | Average |
| Example | Spam Detection | House Price Prediction |

---

# **Applications of KNN**

* Recommendation Systems  
* Image Classification  
* Pattern Recognition  
* Medical Diagnosis  
* Fraud Detection  
* Customer Segmentation

---

# **Python Example**

from sklearn.neighbors import KNeighborsClassifier

classifier \= KNeighborsClassifier(  
    n\_neighbors=5,  
    metric='minkowski',  
    p=2  
)

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

Where:

* `n_neighbors=5` → K \= 5  
* `p=2` → Euclidean distance

---

# **Decision Boundary in KNN**

Unlike Logistic Regression, which creates a straight decision boundary, KNN can create highly flexible and non-linear boundaries because predictions depend on nearby points.

This makes KNN powerful for complex datasets.

---

# **Conclusion**

K-Nearest Neighbors (KNN) is a simple yet powerful supervised learning algorithm that classifies or predicts data based on the nearest training examples. It works by measuring distances, selecting the K closest neighbors, and making predictions through majority voting (classification) or averaging (regression). KNN is easy to implement and effective for many real-world classification and regression problems.

# 18

# **Support Vector Machine (SVM)**

Support Vector Machine (SVM) is a powerful supervised Machine Learning algorithm used primarily for **classification** and sometimes for **regression**.

The main objective of SVM is to find the **best boundary (hyperplane)** that separates different classes with the **maximum margin**.

---

# **Intuition Behind SVM**

Suppose we have two classes:

* Red points  
* Blue points

Many lines can separate them.

Example:

Red ● ● ●

\-------------  
Line 1

Blue ▲ ▲ ▲

But SVM does not choose just any separating line.

It chooses the line that leaves the **maximum distance from both classes**.

This line is called the **Optimal Hyperplane**.

---

# **Why Maximum Margin?**

Imagine two roads:

* One narrow road between two mountains  
* One wide road between two mountains

Driving through the wider road is safer.

Similarly, a larger margin makes the model:

* More robust  
* Less sensitive to noise  
* Better at generalization

---

# **Important Terminologies**

## **1\. Hyperplane**

A decision boundary that separates classes.

For 2D data:

represents a line.

For higher dimensions it becomes a hyperplane.

---

## **2\. Margin**

The distance between the hyperplane and the nearest data points from each class.

SVM tries to maximize this margin.

---

## **3\. Support Vectors**

The data points closest to the hyperplane.

These points determine the position of the decision boundary.

Without support vectors, the hyperplane would change.

---

# **Working of SVM**

## **Step 1**

Plot the training data.

Example:

● ● ● ●

     \---- Hyperplane \----

▲ ▲ ▲ ▲

---

## **Step 2**

Identify support vectors.

These are the nearest points from both classes.

---

## **Step 3**

Find the hyperplane with maximum margin.

---

## **Step 4**

Use the hyperplane to classify new data points.

---

# **Mathematical Representation**

Hyperplane:

Decision rule:

If:

Class \= 1

If:

Class \= \-1

---

# **Hard Margin SVM**

Used when data is perfectly separable.

Example:

● ● ● ●

\------------

▲ ▲ ▲ ▲

No misclassification allowed.

---

# **Soft Margin SVM**

Real-world data usually contains noise.

Example:

● ● ● ▲ ●

One blue point appears inside the red region.

Soft Margin SVM allows some classification errors while still maximizing margin.

This is controlled using parameter:

---

# **Understanding Parameter C**

## **Large C**

* Fewer classification errors  
* Smaller margin  
* Risk of overfitting

---

## **Small C**

* Larger margin  
* Allows some errors  
* Better generalization

---

# **Linear SVM**

Used when data can be separated using a straight line.

Example:

Red Points

\----------

Blue Points

A linear hyperplane is sufficient.

---

# **Non-Linear SVM**

Sometimes data cannot be separated using a straight line.

Example:

    ● ● ●  
   ●       ●

      ▲

   ●       ●  
     ● ● ●

No straight line can separate classes.

In such cases, SVM uses kernels.

---

# **Mapping to Higher Dimensions**

The idea is:

Transform data into a higher-dimensional space where separation becomes possible.

Example:

2D Data:

Circle inside another circle

Not linearly separable.

Transform to 3D:

Data becomes separable by a plane

---

# **The Kernel Trick**

Computing higher-dimensional transformations directly is expensive.

SVM avoids this using the **Kernel Trick**.

Instead of explicitly transforming data, a kernel function computes similarity directly.

Kernel Function:

This provides the benefits of higher-dimensional mapping without the computational cost.

---

# **Types of Kernel Functions**

## **1\. Linear Kernel**

Used when data is linearly separable.

Formula:

Applications:

* Text classification  
* Sentiment analysis

---

## **2\. Polynomial Kernel**

Creates curved decision boundaries.

Formula:

Where:

* (d) \= polynomial degree

Applications:

* Pattern recognition  
* Image processing

---

## **3\. Radial Basis Function (RBF) Kernel**

Most commonly used kernel.

Formula:

Where:

* (\\gamma) controls influence of nearby points

Applications:

* Medical diagnosis  
* Fraud detection  
* Image classification

---

## **4\. Sigmoid Kernel**

Similar to neural network activation functions.

Formula:

Applications:

* Neural-network-like classification tasks

---

# **Advantages of SVM**

* Effective in high-dimensional spaces  
* Works well with small datasets  
* Robust against overfitting  
* Can handle non-linear data using kernels  
* Uses only support vectors for decision making

---

# **Disadvantages of SVM**

* Computationally expensive for large datasets  
* Sensitive to kernel selection  
* Difficult to interpret  
* Requires feature scaling

---

# **Applications of SVM**

* Face Recognition  
* Image Classification  
* Spam Detection  
* Text Categorization  
* Bioinformatics  
* Handwriting Recognition  
* Fraud Detection

---

# **Python Example**

from sklearn.svm import SVC

classifier \= SVC(  
    kernel='rbf',  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

# **Difference Between Logistic Regression and SVM**

| Basis | Logistic Regression | SVM |
| ----- | ----- | ----- |
| Goal | Predict Probability | Maximize Margin |
| Boundary | Linear | Linear / Non-linear |
| Output | Probability | Class |
| Small Datasets | Good | Excellent |
| Non-linear Problems | Limited | Very Effective |

---

# **Conclusion**

Support Vector Machine (SVM) is one of the most powerful classification algorithms. It works by finding the optimal hyperplane that maximizes the margin between classes. The nearest points, called **support vectors**, determine this boundary. For complex datasets, SVM uses **kernel functions** and the **kernel trick** to create non-linear decision boundaries efficiently. This makes SVM highly effective for tasks such as image recognition, spam filtering, and fraud detection.

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

# 20

# **Naive Bayes and Naive Bayes Classifier**

Naive Bayes is a **Supervised Machine Learning Classification Algorithm** based on **Bayes' Theorem**.

It is widely used for:

* Spam Detection  
* Sentiment Analysis  
* Text Classification  
* Recommendation Systems  
* Medical Diagnosis

The word **"Naive"** comes from the assumption that all features are independent of each other.

---

# **Intuition Behind Naive Bayes**

Suppose you receive an email containing words:

Free  
Lottery  
Prize  
Winner

Based on previous emails, these words are very common in spam emails.

Naive Bayes calculates:

P(Spam | Email Features)

and

P(Not Spam | Email Features)

Whichever probability is higher becomes the prediction.

---

# **Bayes' Theorem**

Naive Bayes is built on Bayes' Theorem.

Formula:

genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"P(A|B)=\\frac{P(B|A)P(A)}{P(B)}"}}

Where:

* (P(A|B)) \= Posterior Probability  
* (P(B|A)) \= Likelihood  
* (P(A)) \= Prior Probability  
* (P(B)) \= Evidence

---

# **Understanding Bayes' Theorem**

Suppose:

* 1% of people have a disease.  
* Test detects disease correctly 99% of the time.

If a person's test is positive, Bayes' theorem helps calculate:

Probability that the person actually has the disease

rather than simply assuming they are sick.

---

# **Components of Bayes' Theorem**

## **Prior Probability**

Probability before seeing the data.

Example:

P(Spam) \= 0.3

30% of emails are spam.

---

## **Likelihood**

Probability of observing features given a class.

Example:

P("Free"|Spam)

Probability that the word "Free" appears in spam emails.

---

## **Posterior Probability**

Probability after observing the data.

Example:

P(Spam|"Free")

Probability that an email is spam after seeing the word "Free".

---

## **Evidence**

Overall probability of observing the feature.

Example:

P("Free")

---

# **Why is it Called "Naive"?**

Naive Bayes assumes:

Every feature is independent of every other feature.

Example:

Suppose a house prediction dataset contains:

* Area  
* Number of rooms  
* Price

In reality:

Area and Rooms are related

But Naive Bayes assumes:

Area ⊥ Rooms

(independent)

This assumption is often unrealistic but surprisingly works well in practice.

---

# **Naive Bayes Classifier Formula**

For multiple features:

\[  
X=(x\_1,x\_2,x\_3,\\ldots,x\_n)  
\]

The classifier computes:

genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"P(C|X)=\\frac{P(X|C)P(C)}{P(X)}"}}

Using independence assumption:

P(X|C)=P(x\_1|C)P(x\_2|C)\\cdots P(x\_n|C)

Therefore:

P(C|X)\\propto P(C)\\prod\_{i=1}^{n}P(x\_i|C)

The class with the highest posterior probability is selected.

---

# **Example of Naive Bayes Classification**

Suppose we classify an email.

Features:

Free  
Prize  
Winner

Calculate:

P(Spam | Email)

and

P(Not Spam | Email)

If:

P(Spam | Email) \= 0.92  
P(Not Spam | Email) \= 0.08

Prediction:

Spam

---

# **Types of Naive Bayes Classifiers**

---

# **1\. Gaussian Naive Bayes**

Used when features are continuous and normally distributed.

Examples:

* Height  
* Weight  
* Salary  
* Age

Assumes data follows a Gaussian (Normal) distribution.

Formula:

P(x|C)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}

Where:

* (\\mu) \= mean  
* (\\sigma^2) \= variance

---

## **Example Applications**

* Medical diagnosis  
* Credit scoring  
* Risk prediction

---

# **2\. Multinomial Naive Bayes**

Used for count-based features.

Examples:

* Word counts  
* Number of clicks  
* Frequency of terms

---

## **Example**

Email:

Free Free Win Prize

Word frequencies are used as features.

---

## **Applications**

* Spam filtering  
* Document classification  
* News categorization

---

# **3\. Bernoulli Naive Bayes**

Used when features are binary.

Possible values:

0 or 1

Example:

| Feature | Value |
| ----- | ----- |
| Contains "Free" | 1 |
| Contains "Lottery" | 0 |

---

## **Applications**

* Text classification  
* Binary feature datasets

---

# **Working of Naive Bayes Classifier**

### **Step 1**

Calculate prior probabilities.

Example:

P(Spam)  
P(Not Spam)

---

### **Step 2**

Calculate likelihood probabilities.

Example:

P("Free"|Spam)  
P("Prize"|Spam)

---

### **Step 3**

Apply Bayes' theorem.

---

### **Step 4**

Compute posterior probabilities.

---

### **Step 5**

Choose class with highest probability.

---

# **Advantages of Naive Bayes**

* Simple and easy to implement  
* Very fast training  
* Works well on small datasets  
* Excellent for text classification  
* Requires less computational power

---

# **Disadvantages of Naive Bayes**

* Strong independence assumption  
* May perform poorly when features are highly correlated  
* Probability estimates may not be very accurate

---

# **Naive Bayes vs Logistic Regression**

| Feature | Naive Bayes | Logistic Regression |
| ----- | ----- | ----- |
| Based On | Bayes Theorem | Sigmoid Function |
| Assumption | Independent Features | Linear Decision Boundary |
| Training Speed | Very Fast | Slower |
| Small Dataset Performance | Excellent | Good |
| Probability Estimates | Less Accurate | More Accurate |

---

# **Python Example**

from sklearn.naive\_bayes import GaussianNB

classifier \= GaussianNB()

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

# **Applications of Naive Bayes**

* Spam Detection  
* Sentiment Analysis  
* News Classification  
* Recommendation Systems  
* Medical Diagnosis  
* Fraud Detection

---

# **Conclusion**

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem**. It assumes that features are independent of one another, which makes it computationally efficient and easy to implement. Despite its "naive" assumption, it performs remarkably well in many real-world tasks, especially **text classification**, **spam filtering**, and **sentiment analysis**. The three main variants are **Gaussian Naive Bayes**, **Multinomial Naive Bayes**, and **Bernoulli Naive Bayes**, each suited for different types of data.

# 21

# **Decision Tree**

Decision Tree is a **Supervised Machine Learning Algorithm** used for both:

* Classification Problems  
* Regression Problems

It works by recursively splitting data into smaller subsets based on feature values, forming a tree-like structure.

The final prediction is obtained by following a path from the root node to a leaf node.

---

# **Intuition Behind Decision Trees**

A Decision Tree mimics human decision-making.

### **Example: Should a Bank Approve a Loan?**

Instead of making a decision directly, the bank may ask:

Is Income \> ₹50,000?  
        |  
       Yes  
        |  
Is Credit Score \> 700?  
        |  
       Yes  
        |  
   Approve Loan

A Decision Tree follows the same process by asking a sequence of questions.

---

# **Real-Life Example**

Suppose we want to predict whether a customer will buy a product.

Features:

* Age  
* Salary  
* Previous Purchases

Decision Tree:

                Age \< 30?  
                /         \\  
              Yes         No  
              /             \\  
     Salary \> 50K?       Buy  
          /   \\  
       Yes    No  
       Buy   Don't Buy

Each decision splits the dataset into more homogeneous groups.

---

# **Structure of a Decision Tree**

### **1\. Root Node**

Topmost node.

Represents the entire dataset.

Example:

Age \< 30?

---

### **2\. Decision Node**

Intermediate nodes where splitting occurs.

Example:

Salary \> 50K?

---

### **3\. Branch**

Represents outcomes of a decision.

Example:

Yes / No

---

### **4\. Leaf Node**

Final prediction.

Example:

Buy  
Don't Buy

---

# **How Does a Decision Tree Learn?**

The tree repeatedly chooses the best feature for splitting.

Goal:

Create groups that are as pure as possible.

A pure node contains observations belonging mostly to one class.

---

# **Example Dataset**

| Age | Purchased |
| ----- | ----- |
| 22 | Yes |
| 24 | Yes |
| 45 | No |
| 50 | No |

A good split:

Age \< 30

Produces:

Yes Yes

No No

Perfect separation.

---

# **Classification Trees**

Used when target variable is categorical.

Examples:

* Spam / Not Spam  
* Disease / No Disease  
* Buy / Not Buy

---

# **How Classification Trees Choose Splits?**

Common criteria:

1. Entropy  
2. Information Gain  
3. Gini Index

---

# **Entropy**

Entropy measures impurity or randomness.

Formula:

Entropy=-\\sum p\_i\\log\_2(p\_i)

Where:

* (p\_i) \= probability of class i

---

## **Understanding Entropy**

### **Pure Node**

100% Yes

Entropy:

0

No uncertainty.

---

### **Mixed Node**

50% Yes  
50% No

Entropy:

1

Maximum uncertainty.

---

# **Information Gain**

Measures reduction in entropy after splitting.

Formula:

Information\\ Gain=Entropy(parent)-Entropy(children)

The feature with highest Information Gain is selected.

---

## **Example**

Before split:

Entropy \= 1

After split:

Entropy \= 0.2

Information Gain:

1 \- 0.2 \= 0.8

Large gain → Good split.

---

# **Gini Index**

Another impurity measure used by CART (Classification and Regression Trees).

Formula:

Gini=1-\\sum p\_i^2

---

## **Interpretation**

### **Pure Node**

Gini \= 0

### **Mixed Node**

Higher Gini value.

The algorithm chooses the split with the lowest weighted Gini.

---

# **Regression Trees**

Used when the target variable is continuous.

Examples:

* House Price Prediction  
* Salary Prediction  
* Temperature Forecasting

---

# **Intuition of Regression Trees**

Instead of predicting classes, Regression Trees predict numerical values.

The tree divides data into regions.

Each leaf predicts the average value of observations in that region.

---

# **Example**

| Experience | Salary |
| ----- | ----- |
| 1 | 30K |
| 2 | 35K |
| 3 | 50K |
| 4 | 80K |

Tree:

Experience \< 2.5  
      /      \\  
    Yes      No

Salary      Salary  
\=32.5K      \=65K

---

# **Splitting Criterion for Regression Trees**

Regression Trees use **Variance Reduction**.

Goal:

Minimize prediction variance within each node.

Formula:

Variance=\\frac{1}{n}\\sum (y\_i-\\bar{y})^2

The split that reduces variance the most is chosen.

---

# **Overfitting in Decision Trees**

Decision Trees can grow very deep.

Example:

Every training point gets its own leaf.

This memorizes training data and causes overfitting.

---

# **Pruning**

Pruning removes unnecessary branches.

Two approaches:

### **1\. Pre-Pruning**

Stop tree growth early.

Parameters:

* max\_depth  
* min\_samples\_split  
* min\_samples\_leaf

---

### **2\. Post-Pruning**

Build full tree first, then remove weak branches.

Produces simpler trees.

---

# **Advantages of Decision Trees**

### **Easy to Understand**

Can be visualized like a flowchart.

### **No Feature Scaling Required**

Works directly on raw features.

### **Handles Numerical and Categorical Data**

Both types can be used.

### **Non-Linear Relationships**

Can learn complex decision boundaries.

### **Feature Importance**

Automatically identifies important features.

---

# **Disadvantages of Decision Trees**

### **Overfitting**

Most common issue.

### **Unstable**

Small changes in data can create different trees.

### **Greedy Nature**

Chooses locally optimal splits.

May not find globally optimal tree.

---

# **Decision Tree vs Random Forest**

| Feature | Decision Tree | Random Forest |
| ----- | ----- | ----- |
| Trees | One | Many |
| Accuracy | Lower | Higher |
| Overfitting | High | Lower |
| Interpretability | Easy | Harder |
| Speed | Faster | Slower |

---

# **Python Example**

## **Classification Tree**

from sklearn.tree import DecisionTreeClassifier

classifier \= DecisionTreeClassifier(  
    criterion='entropy',  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

---

## **Regression Tree**

from sklearn.tree import DecisionTreeRegressor

regressor \= DecisionTreeRegressor(  
    random\_state=0  
)

regressor.fit(X\_train, y\_train)

---

# **Applications of Decision Trees**

* Loan Approval Systems  
* Medical Diagnosis  
* Fraud Detection  
* Customer Churn Prediction  
* Risk Assessment  
* Product Recommendation  
* Credit Scoring

---

# **Interview Summary**

**Decision Tree** is a supervised learning algorithm that recursively splits data based on feature values to create a tree-like model. For **classification**, it uses measures such as **Entropy, Information Gain, and Gini Index** to create pure class groups. For **regression**, it uses **variance reduction** and predicts the average value in each leaf node. Decision Trees are easy to interpret and require minimal preprocessing, but they are prone to overfitting, which can be controlled through pruning techniques. They form the foundation of powerful ensemble algorithms such as **Random Forest** and **Gradient Boosting**.

# 22

# **Random Forest**

Random Forest is a powerful **Ensemble Learning** algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

It can be used for:

* Classification  
* Regression

Random Forest is one of the most widely used Machine Learning algorithms because of its high accuracy and robustness.

---

# **Intuition Behind Random Forest**

Suppose you want to decide whether to invest in a company.

Instead of asking one expert, you ask:

Expert 1 → Invest  
Expert 2 → Don't Invest  
Expert 3 → Invest  
Expert 4 → Invest  
Expert 5 → Don't Invest

Majority opinion:

Invest

This collective decision is usually more reliable than relying on a single expert.

Random Forest works in the same way.

Instead of one Decision Tree, it uses many Decision Trees and combines their predictions.

---

# **Why Not Use a Single Decision Tree?**

A Decision Tree can easily overfit.

Example:

Training Accuracy \= 100%  
Testing Accuracy \= 70%

The tree memorizes the training data.

Random Forest reduces this problem by averaging multiple trees.

---

# **What is a Forest?**

A Forest is simply:

Collection of Decision Trees

Example:

Tree 1  
Tree 2  
Tree 3  
Tree 4  
Tree 5  
...  
Tree N

Together they form a Random Forest.

---

# **Why "Random"?**

Two sources of randomness are introduced:

### **1\. Random Sampling of Data**

Each tree is trained on a random subset of observations.

---

### **2\. Random Sampling of Features**

Each split considers only a random subset of features.

This makes trees different from one another.

---

# **Working of Random Forest**

---

## **Step 1: Bootstrap Sampling**

Random Forest creates multiple datasets from the original dataset using sampling with replacement.

This is called:

Bootstrap Sampling

Example:

Original Dataset

A B C D E

Possible Bootstrap Sample

A C C D E

Notice:

C repeated  
B missing

This is allowed.

---

## **Step 2: Build Multiple Decision Trees**

Each bootstrap sample trains one Decision Tree.

Example:

Dataset 1 → Tree 1  
Dataset 2 → Tree 2  
Dataset 3 → Tree 3

Each tree sees slightly different data.

---

## **Step 3: Random Feature Selection**

Suppose we have 10 features.

At each split:

Decision Tree → Uses all 10 features

Random Forest:

Choose only random subset

Example:

Feature 2  
Feature 5  
Feature 9

This creates diversity among trees.

---

## **Step 4: Make Predictions**

---

# **Random Forest Classification**

Each tree votes.

Example:

| Tree | Prediction |
| ----- | ----- |
| Tree 1 | Yes |
| Tree 2 | No |
| Tree 3 | Yes |
| Tree 4 | Yes |
| Tree 5 | No |

Majority Vote:

Yes

Final Prediction:

Yes

---

# **Random Forest Regression**

Each tree predicts a value.

Example:

| Tree | Prediction |
| ----- | ----- |
| Tree 1 | 100 |
| Tree 2 | 120 |
| Tree 3 | 110 |

Final prediction is the average:

\\frac{100+120+110}{3}=110

---

# **Bagging (Bootstrap Aggregating)**

Random Forest is based on a technique called:

Bagging

Bagging Steps:

1. Create bootstrap samples  
2. Train multiple models independently  
3. Combine predictions

---

# **Why Does Random Forest Work So Well?**

Decision Trees have:

Low Bias  
High Variance

They fit training data very well but overfit.

Random Forest reduces variance by averaging many trees.

Result:

Lower Variance  
Better Generalization  
Higher Accuracy

---

# **Out-of-Bag (OOB) Error**

Since bootstrap sampling leaves some observations unused:

Example:

Dataset Size \= 1000

A tree may use:

630 samples

and leave:

370 samples

unused.

These unused samples are called:

Out-of-Bag Samples

They can be used to estimate model performance without needing a separate validation set.

---

# **Feature Importance**

Random Forest can rank features according to importance.

Example:

| Feature | Importance |
| ----- | ----- |
| Salary | 45% |
| Age | 30% |
| Experience | 15% |
| City | 10% |

This helps in feature selection.

---

# **Important Hyperparameters**

---

## **1\. n\_estimators**

Number of trees.

Example:

n\_estimators=100

More trees:

✔ Better performance

✖ More computation

---

## **2\. max\_depth**

Maximum depth of each tree.

Controls overfitting.

---

## **3\. min\_samples\_split**

Minimum samples required to split a node.

---

## **4\. min\_samples\_leaf**

Minimum samples in a leaf node.

---

## **5\. max\_features**

Number of random features considered at each split.

---

# **Advantages of Random Forest**

### **High Accuracy**

Often outperforms a single Decision Tree.

---

### **Reduces Overfitting**

Averaging reduces variance.

---

### **Handles Missing Values**

Can work with incomplete data.

---

### **Feature Importance**

Provides importance rankings.

---

### **Works with Large Datasets**

Scales well.

---

### **Handles Non-linear Data**

Learns complex relationships.

---

# **Disadvantages of Random Forest**

### **Slower Training**

Many trees must be trained.

---

### **Higher Memory Usage**

Stores many trees.

---

### **Less Interpretable**

Harder to explain than a single Decision Tree.

---

# **Decision Tree vs Random Forest**

| Feature | Decision Tree | Random Forest |
| ----- | ----- | ----- |
| Number of Trees | 1 | Many |
| Overfitting | High | Low |
| Accuracy | Lower | Higher |
| Variance | High | Low |
| Interpretability | Easy | Hard |
| Training Speed | Fast | Slower |

---

# **Random Forest vs Bagging**

| Feature | Bagging | Random Forest |
| ----- | ----- | ----- |
| Bootstrap Sampling | Yes | Yes |
| Random Feature Selection | No | Yes |
| Diversity of Trees | Lower | Higher |
| Accuracy | Good | Better |

---

# **Python Example**

## **Random Forest Classification**

from sklearn.ensemble import RandomForestClassifier

classifier \= RandomForestClassifier(  
    n\_estimators=100,  
    criterion='entropy',  
    random\_state=0  
)

classifier.fit(X\_train, y\_train)

y\_pred \= classifier.predict(X\_test)

---

## **Random Forest Regression**

from sklearn.ensemble import RandomForestRegressor

regressor \= RandomForestRegressor(  
    n\_estimators=100,  
    random\_state=0  
)

regressor.fit(X\_train, y\_train)

y\_pred \= regressor.predict(X\_test)

---

# **Applications of Random Forest**

* Fraud Detection  
* Disease Diagnosis  
* Credit Risk Assessment  
* Customer Churn Prediction  
* Recommendation Systems  
* Stock Market Prediction  
* Image Classification  
* Spam Detection

---

# **Interview Summary**

**Random Forest** is an Ensemble Learning algorithm based on **Bagging** that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting. Each tree is trained on a bootstrap sample of the dataset and uses a random subset of features during splitting. For classification, predictions are made using **majority voting**, while for regression, predictions are made using **averaging**. Random Forest is highly accurate, robust to noise, resistant to overfitting, and capable of providing feature importance, making it one of the most popular algorithms in Machine Learning.

# 23

# **Confusion Matrix in Machine Learning**

A **Confusion Matrix** is a table used to evaluate the performance of a **Classification Model**.

It compares:

* Actual values  
* Predicted values

and shows how many predictions were correct and incorrect.

It is one of the most important evaluation tools for classification problems.

---

# **Why is it Called a "Confusion Matrix"?**

Because it shows where the model gets **confused** between classes.

Example:

A disease detection model may:

* Correctly identify sick patients  
* Mistakenly classify healthy patients as sick

The confusion matrix helps us understand these mistakes.

---

# **Structure of a Confusion Matrix**

For Binary Classification:

|  | Predicted Positive | Predicted Negative |
| ----- | ----- | ----- |
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

---

# **The Four Components**

## **1\. True Positive (TP)**

Actual class is Positive and prediction is also Positive.

Example:

Actual: Disease  
Predicted: Disease

Correct prediction.

---

## **2\. True Negative (TN)**

Actual class is Negative and prediction is also Negative.

Example:

Actual: Healthy  
Predicted: Healthy

Correct prediction.

---

## **3\. False Positive (FP)**

Actual class is Negative but prediction is Positive.

Also called:

Type I Error

Example:

Actual: Healthy  
Predicted: Disease

The model incorrectly raises an alarm.

---

## **4\. False Negative (FN)**

Actual class is Positive but prediction is Negative.

Also called:

Type II Error

Example:

Actual: Disease  
Predicted: Healthy

This is often the most dangerous error in healthcare.

---

# **Example Confusion Matrix**

Suppose 100 patients were tested.

|  | Predicted Disease | Predicted Healthy |
| ----- | ----- | ----- |
| Actual Disease | 40 | 10 |
| Actual Healthy | 5 | 45 |

Therefore:

TP \= 40  
FN \= 10  
FP \= 5  
TN \= 45

---

# **Accuracy**

Accuracy measures overall correctness.

Formula:

Accuracy=\\frac{TP+TN}{TP+TN+FP+FN}

Using the example:

Accuracy=\\frac{40+45}{40+45+5+10}=0.85

Accuracy:

85%

---

# **Precision**

Precision measures:

Out of all predicted positives, how many were actually positive?

Formula:

Precision=\\frac{TP}{TP+FP}

Using example:

Precision=\\frac{40}{40+5}=0.889

Precision:

88.9%

---

# **Recall (Sensitivity)**

Recall measures:

Out of all actual positives, how many were identified correctly?

Formula:

Recall=\\frac{TP}{TP+FN}

Using example:

Recall=\\frac{40}{40+10}=0.80

Recall:

80%

---

# **Specificity**

Specificity measures:

Out of all actual negatives, how many were identified correctly?

Formula:

Specificity=\\frac{TN}{TN+FP}

Using example:

Specificity=\\frac{45}{45+5}=0.90

Specificity:

90%

---

# **F1 Score**

F1 Score balances Precision and Recall.

Formula:

F1=2\\times\\frac{Precision\\times Recall}{Precision+Recall}

Using example:

Precision \= 0.889  
Recall \= 0.80

F1 Score:

≈ 0.842

or

84.2%

---

# **When Accuracy Can Be Misleading**

Suppose:

| Class | Count |
| ----- | ----- |
| Healthy | 990 |
| Disease | 10 |

Model predicts:

Everyone is Healthy

Accuracy:

\\frac{990}{1000}=99%

Looks excellent.

But:

Recall \= 0%

because all diseased patients were missed.

Therefore accuracy alone is not sufficient for imbalanced datasets.

---

# **Which Metric Should Be Used?**

## **Precision Important When**

False Positives are costly.

Examples:

* Spam detection  
* Fraud detection

Goal:

Avoid false alarms

---

## **Recall Important When**

False Negatives are costly.

Examples:

* Cancer detection  
* Disease diagnosis

Goal:

Don't miss positive cases

---

## **F1 Score Important When**

Need balance between Precision and Recall.

Examples:

* Medical diagnosis  
* Fraud detection

---

# **Multi-Class Confusion Matrix**

For more than two classes:

Example:

| Actual / Predicted | Cat | Dog | Bird |
| ----- | ----- | ----- | ----- |
| Cat | 40 | 3 | 2 |
| Dog | 5 | 35 | 4 |
| Bird | 1 | 2 | 38 |

The same concepts extend to each class.

---

# **Python Example**

from sklearn.metrics import confusion\_matrix

cm \= confusion\_matrix(y\_test, y\_pred)

print(cm)

Output:

\[\[45 5\]  
 \[10 40\]\]

Meaning:

TN \= 45  
FP \= 5  
FN \= 10  
TP \= 40

---

# **Visual Representation**

                   Predicted  
                 Positive Negative

Actual Positive     TP       FN

Actual Negative     FP       TN

Remember:

TP → Correct Positive  
TN → Correct Negative  
FP → False Alarm  
FN → Missed Detection

---

# **Interview Summary**

A **Confusion Matrix** is a performance evaluation tool for classification models that compares actual and predicted classes. It consists of **True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)**. From these values, important metrics such as **Accuracy, Precision, Recall, Specificity, and F1 Score** are calculated. While Accuracy measures overall correctness, Precision evaluates the quality of positive predictions, Recall measures the ability to identify actual positives, and F1 Score balances both Precision and Recall. Confusion matrices are especially important for evaluating models on imbalanced datasets where accuracy alone may be misleading.

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

# 26

# **K-Means Clustering**

K-Means Clustering is an **Unsupervised Machine Learning Algorithm** used to group similar data points into clusters.

Unlike classification algorithms, K-Means does not use labeled data.

Its goal is:

Group similar observations together and separate dissimilar observations.

---

# **Real-Life Example**

Suppose a shopping mall has customer data:

| Customer | Annual Income | Spending Score |
| ----- | ----- | ----- |
| A | High | High |
| B | High | Low |
| C | Low | High |

The mall wants to identify groups of customers for targeted marketing.

K-Means can automatically create customer segments such as:

Cluster 1 → Premium Customers  
Cluster 2 → Budget Customers  
Cluster 3 → Occasional Buyers

---

# **Intuition Behind K-Means**

Imagine dropping several magnets on a table full of iron particles.

Each particle gets attracted to the nearest magnet.

Eventually:

Particles around Magnet 1 → Cluster 1  
Particles around Magnet 2 → Cluster 2  
Particles around Magnet 3 → Cluster 3

K-Means works similarly.

---

# **What Does K Mean?**

K represents:

Number of Clusters

Example:

K \= 3

means create 3 clusters.

---

# **Key Terminology**

## **Cluster**

A group of similar observations.

---

## **Centroid**

The center point of a cluster.

Example:

Cluster A  
● ● ● ● ●

Center \= Centroid

The centroid represents the average location of all points in that cluster.

---

# **Working of K-Means**

---

## **Step 1: Choose K**

Suppose:

K \= 3

We want 3 clusters.

---

## **Step 2: Initialize K Centroids**

Randomly select K points as initial centroids.

Example:

Centroid 1  
Centroid 2  
Centroid 3

---

## **Step 3: Assign Points**

For every data point:

Calculate distance to each centroid.

Usually using Euclidean Distance.

Assign point to nearest centroid.

---

### **Euclidean Distance**

For points:

\[  
(x\_1,y\_1)  
\]

and

\[  
(x\_2,y\_2)  
\]

Distance:

genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"d=\\sqrt{(x\_2-x\_1)^2+(y\_2-y\_1)^2}"}}

---

## **Step 4: Update Centroids**

Compute new centroid as the average of all points in that cluster.

Formula:

Centroid=\\frac{\\sum x\_i}{n}

---

## **Step 5: Repeat**

Repeat:

Assign Points  
Update Centroids

until centroids stop changing.

This is called:

Convergence

---

# **Objective Function of K-Means**

K-Means tries to minimize:

### **Within Cluster Sum of Squares (WCSS)**

Also called Inertia.

Formula:

WCSS=\\sum\_{i=1}^{K}\\sum\_{x\\in C\_i}||x-\\mu\_i||^2

Where:

* (C\_i) \= Cluster  
* (\\mu\_i) \= Centroid

Goal:

Minimize WCSS

---

# **Elbow Method**

One major challenge:

How do we choose K?

The most common solution is the Elbow Method.

---

# **Intuition of Elbow Method**

As K increases:

WCSS decreases

because clusters become smaller.

Example:

| K | WCSS |
| ----- | ----- |
| 1 | 1000 |
| 2 | 600 |
| 3 | 300 |
| 4 | 200 |
| 5 | 180 |
| 6 | 170 |

Notice:

Big drop until K=3  
Small drop afterward

---

# **Elbow Curve**

Plot:

X-axis → Number of Clusters (K)  
Y-axis → WCSS

Graph:

WCSS  
|  
|\\  
| \\  
|  \\  
|   \\\_\_  
|      \\\_  
|  
\+----------------  
     K

The bend point resembles an elbow.

Choose K at that point.

Example:

Elbow at K=3

Therefore:

Optimal K \= 3

---

# **Limitation of K-Means**

K-Means is sensitive to initial centroid selection.

Different initial centroids may produce:

Different Clusters

This problem leads to local optima.

---

# **K-Means++**

K-Means++ is an improved initialization technique for K-Means.

It selects better starting centroids.

Result:

✔ Faster convergence

✔ Better clustering

✔ Lower WCSS

✔ More stable results

---

# **Why K-Means++?**

Suppose all initial centroids are chosen close together:

●●●      ○○○○○○○○

Poor clustering occurs.

K-Means++ spreads centroids apart before clustering starts.

---

# **Working of K-Means++**

---

## **Step 1**

Choose first centroid randomly.

---

## **Step 2**

Calculate distance of every point from chosen centroid.

---

## **Step 3**

Choose next centroid with probability proportional to squared distance.

Farther points have higher chance of selection.

---

## **Step 4**

Repeat until K centroids are selected.

---

## **Step 5**

Run standard K-Means.

---

# **K-Means vs K-Means++**

| Feature | K-Means | K-Means++ |
| ----- | ----- | ----- |
| Centroid Initialization | Random | Smart Initialization |
| Convergence Speed | Slower | Faster |
| Clustering Quality | Lower | Higher |
| Stability | Lower | Higher |
| Risk of Poor Clusters | High | Low |

---

# **Advantages of K-Means**

* Simple and easy to implement  
* Fast for large datasets  
* Scalable  
* Easy interpretation

---

# **Disadvantages of K-Means**

* Need to choose K manually  
* Sensitive to outliers  
* Sensitive to initialization  
* Assumes spherical clusters  
* Struggles with clusters of varying sizes

---

# **Applications of K-Means**

### **Customer Segmentation**

Group customers by behavior.

---

### **Recommendation Systems**

Identify similar users.

---

### **Image Compression**

Reduce image colors.

---

### **Document Clustering**

Group similar documents.

---

### **Market Basket Analysis**

Identify purchasing patterns.

---

# **Python Example**

from sklearn.cluster import KMeans

kmeans \= KMeans(  
    n\_clusters=5,  
    init='k-means++',  
    random\_state=0  
)

y\_kmeans \= kmeans.fit\_predict(X)

---

# **Interview Summary**

**K-Means Clustering** is an unsupervised learning algorithm that partitions data into K clusters by assigning each point to its nearest centroid and iteratively updating centroids until convergence. The algorithm minimizes the **Within Cluster Sum of Squares (WCSS)**. The **Elbow Method** helps determine the optimal number of clusters by identifying the point where adding more clusters yields diminishing reductions in WCSS. **K-Means++** improves standard K-Means by selecting smarter initial centroids that are well separated, leading to faster convergence and better clustering performance.

# 27

# **Hierarchical Clustering**

Hierarchical Clustering is an **Unsupervised Machine Learning Algorithm** used to group similar data points into clusters.

Unlike K-Means:

* No need to specify K initially  
* Produces a hierarchy (tree structure) of clusters  
* Useful when the number of clusters is unknown

---

# **Intuition Behind Hierarchical Clustering**

Imagine organizing books in a library.

You first group:

* Science books  
* Literature books

Then Science books are further grouped into:

* Physics  
* Chemistry  
* Biology

Then Physics books may be grouped further.

This creates a hierarchy.

Hierarchical Clustering works in a similar way.

---

# **Types of Hierarchical Clustering**

There are two approaches:

## **1\. Agglomerative Hierarchical Clustering (Bottom-Up)**

Starts with:

Each data point \= One cluster

Then repeatedly merges the closest clusters.

Most commonly used.

---

## **2\. Divisive Hierarchical Clustering (Top-Down)**

Starts with:

All points \= One cluster

Then repeatedly splits clusters.

Less commonly used.

---

# **Agglomerative Hierarchical Clustering (AHC)**

This is the method most interviewers refer to when discussing Hierarchical Clustering.

---

# **Intuition**

Suppose we have 5 points:

A B C D E

Initially:

{A}  
{B}  
{C}  
{D}  
{E}

Each point is its own cluster.

---

## **Step 1**

Find closest clusters.

Example:

A and B

Merge them.

{AB}  
{C}  
{D}  
{E}

---

## **Step 2**

Find next closest clusters.

D and E

Merge them.

{AB}  
{C}  
{DE}

---

## **Step 3**

Continue merging until one cluster remains.

---

# **Dendrogram**

A dendrogram is a tree-like diagram showing cluster merging.

Example:

       \---------  
       |         |  
     \-----      \---  
    |     |    |   |  
   A      B   C   D E

---

# **Why Dendrogram is Important?**

It helps determine:

Optimal Number of Clusters

without using the Elbow Method.

---

# **Choosing Number of Clusters from Dendrogram**

Rule:

Draw a horizontal line through the largest vertical gap.

Example:

      |  
       |  
\-------|---------  
       |  
   \----|----  
   |       |

Number of branches cut \= Number of clusters.

---

# **Distance Between Clusters (Linkage Criteria)**

The most important concept in Hierarchical Clustering is:

How do we measure distance between clusters?

Different answers create different clustering results.

---

# **1\. Single Linkage (Nearest Neighbor)**

Distance between two clusters is the distance between their closest points.

Formula:

D(A,B)=\\min(d(x,y))

---

## **Example**

Cluster A:

A1 A2

Cluster B:

B1 B2

Use:

Closest(A,B)

---

## **Characteristics**

Advantages:

* Can find irregular shapes

Disadvantages:

* Chaining effect  
* Sensitive to noise

---

# **2\. Complete Linkage (Farthest Neighbor)**

Distance between two clusters is the distance between their farthest points.

Formula:

D(A,B)=\\max(d(x,y))

---

## **Example**

Use:

Farthest(A,B)

distance.

---

## **Characteristics**

Advantages:

* Produces compact clusters

Disadvantages:

* Sensitive to outliers

---

# **3\. Average Linkage**

Distance between clusters is the average distance between all pairs of points.

Formula:

D(A,B)=\\frac{1}{mn}\\sum d(x,y)

where:

* m \= points in cluster A  
* n \= points in cluster B

---

## **Characteristics**

Advantages:

* Balanced clustering  
* Less sensitive to noise

Most commonly used after Ward linkage.

---

# **4\. Centroid Linkage**

Distance is measured between cluster centroids.

Formula:

D(A,B)=d(C\_A,C\_B)

Where:

* (C\_A) \= centroid of cluster A  
* (C\_B) \= centroid of cluster B

---

## **Characteristics**

Advantages:

* Computationally efficient

Disadvantages:

* Can produce inversions in dendrogram

---

# **5\. Ward Linkage (Most Important)**

Most commonly used in Machine Learning.

Instead of distance, it minimizes the increase in variance after merging clusters.

Goal:

Minimize Within Cluster Variance

---

# **Intuition of Ward Linkage**

Suppose two clusters:

Cluster A  
Cluster B

Ward asks:

How much does total variance increase  
if we merge them?

Choose the merge causing the smallest increase.

---

## **Characteristics**

Advantages:

* Produces compact clusters  
* Usually best results  
* Most common in practice

Disadvantages:

* Prefers spherical clusters

---

# **Distance Metrics Used Inside Clustering**

Apart from linkage criteria, we also need point-to-point distances.

---

# **Euclidean Distance (Most Common)**

Formula:

d=\\sqrt{\\sum (x\_i-y\_i)^2}

Best for:

* Continuous numerical features

---

# **Manhattan Distance**

Formula:

d=\\sum |x\_i-y\_i|

Best for:

* Grid-based movement  
* High-dimensional data

---

# **Minkowski Distance**

Generalized distance measure.

Formula:

d=\\left(\\sum |x\_i-y\_i|^p\\right)^{1/p}

Special cases:

* p=1 → Manhattan  
* p=2 → Euclidean

---

# **Agglomerative HC Algorithm**

### **Step 1**

Start with:

n clusters

(one point per cluster)

---

### **Step 2**

Compute distance matrix.

---

### **Step 3**

Merge nearest clusters.

---

### **Step 4**

Update distance matrix.

---

### **Step 5**

Repeat until one cluster remains.

---

# **Hierarchical Clustering vs K-Means**

| Feature | Hierarchical Clustering | K-Means |
| ----- | ----- | ----- |
| Need K Initially | No | Yes |
| Output | Dendrogram | Clusters |
| Cluster Shape | Flexible | Mostly Spherical |
| Speed | Slower | Faster |
| Large Datasets | Poor | Better |
| Interpretability | High | Moderate |

---

# **Advantages of Hierarchical Clustering**

* No need to choose K initially  
* Produces dendrogram  
* Easy visualization  
* Works for complex cluster structures

---

# **Disadvantages**

* Computationally expensive  
* Not suitable for very large datasets  
* Sensitive to noise and outliers  
* Cannot undo merges once made

---

# **Python Example**

from sklearn.cluster import AgglomerativeClustering

hc \= AgglomerativeClustering(  
    n\_clusters=5,  
    linkage='ward'  
)

y\_pred \= hc.fit\_predict(X)

---

# **Interview Summary**

**Hierarchical Clustering** is an unsupervised learning algorithm that creates a hierarchy of clusters. The most common approach is **Agglomerative Hierarchical Clustering (AHC)**, which starts with each point as its own cluster and repeatedly merges the closest clusters. The clustering process is visualized using a **dendrogram**, which helps determine the optimal number of clusters. Different linkage methods define how cluster distances are measured: **Single Linkage (nearest neighbor), Complete Linkage (farthest neighbor), Average Linkage, Centroid Linkage, and Ward Linkage**. Among these, **Ward Linkage** is the most commonly used because it minimizes within-cluster variance and generally produces compact, well-separated clusters.

# 29

# **Association Rule Learning**

Association Rule Learning is an **Unsupervised Machine Learning Technique** used to discover relationships, patterns, and associations between items in large datasets.

It is widely used in:

* Market Basket Analysis  
* Recommendation Systems  
* E-commerce  
* Retail Analytics  
* Web Usage Mining

---

# **Intuition Behind Association Rule Learning**

Suppose a supermarket analyzes customer purchases.

Transactions:

| Transaction ID | Items Purchased |
| ----- | ----- |
| T1 | Bread, Milk |
| T2 | Bread, Diapers, Beer |
| T3 | Milk, Diapers, Beer |
| T4 | Bread, Milk, Diapers, Beer |
| T5 | Bread, Milk |

After analysis, we may discover:

People who buy Bread  
often buy Milk

or

People who buy Diapers  
often buy Beer

These relationships are called **Association Rules**.

---

# **Basic Form of an Association Rule**

A rule is represented as:

\[  
X \\rightarrow Y  
\]

Meaning:

If X occurs,  
Y is likely to occur

Example:

Bread → Milk

Interpretation:

Customers buying Bread  
tend to buy Milk

---

# **Key Metrics in Association Rule Learning**

To determine whether a rule is useful, we use:

1. Support  
2. Confidence  
3. Lift

---

# **1\. Support**

Support measures:

How frequently an itemset appears in the dataset.

Formula:

Support(X)=\\frac{\\text{Transactions containing X}}{\\text{Total Transactions}}

---

## **Example**

Transactions:

Total \= 5

Bread appears in:

T1, T2, T4, T5

Count:

4

Support:

Support(Bread)=\\frac{4}{5}=0.8

Support \= 80%

---

# **Support of a Rule**

For:

Bread → Milk

Support is:

\[  
Support(Bread \\cap Milk)  
\]

Formula:

Support(X\\rightarrow Y)=\\frac{Count(X\\cap Y)}{Total\\ Transactions}

---

# **2\. Confidence**

Confidence measures:

How often Y occurs when X occurs.

Formula:

Confidence(X\\rightarrow Y)=\\frac{Support(X\\cap Y)}{Support(X)}

---

## **Example**

Transactions containing Bread:

4

Transactions containing Bread and Milk:

3

Confidence:

\\frac{3}{4}=0.75

Confidence:

75%

Interpretation:

75% of customers who buy Bread  
also buy Milk

---

# **Limitation of Confidence**

Suppose Milk is purchased by almost everyone.

Then confidence becomes naturally high.

Confidence alone can be misleading.

This is why Lift is used.

---

# **3\. Lift**

Lift measures:

How much more likely Y occurs when X occurs compared to random chance.

Formula:

Lift(X\\rightarrow Y)=\\frac{Confidence(X\\rightarrow Y)}{Support(Y)}

---

# **Interpretation of Lift**

### **Lift \= 1**

No relationship

X and Y are independent.

---

### **Lift \> 1**

Positive Association

X increases likelihood of Y.

---

### **Lift \< 1**

Negative Association

X reduces likelihood of Y.

---

# **Example**

Suppose:

Confidence(Bread→Milk)=0.75  
Support(Milk)=0.60

Lift:

Lift=\\frac{0.75}{0.60}=1.25

Interpretation:

Customers buying Bread are  
1.25 times more likely  
to buy Milk.

---

# **Apriori Algorithm**

Apriori is the most famous algorithm for Association Rule Learning.

It finds:

Frequent Itemsets

and then generates association rules.

---

# **Why Is It Called Apriori?**

It uses prior knowledge:

If an itemset is frequent, all of its subsets must also be frequent.

This is called the **Apriori Property**.

---

# **Apriori Property**

If:

{Bread, Milk, Butter}

is frequent,

then:

{Bread, Milk}  
{Bread, Butter}  
{Milk, Butter}

must also be frequent.

---

# **Key Idea of Apriori**

Instead of checking every possible combination:

Millions of combinations

Apriori eliminates impossible candidates early.

This greatly reduces computation.

---

# **Working of Apriori Algorithm**

---

## **Step 1**

Find all individual items.

Example:

Bread  
Milk  
Beer  
Diapers  
Butter

---

## **Step 2**

Calculate support for each item.

Remove items below minimum support threshold.

---

## **Step 3**

Generate candidate pairs.

Example:

Bread, Milk  
Bread, Beer  
Milk, Beer

---

## **Step 4**

Calculate support for each pair.

Remove infrequent pairs.

---

## **Step 5**

Generate larger itemsets.

Example:

Bread, Milk, Beer

---

## **Step 6**

Repeat until no more frequent itemsets exist.

---

## **Step 7**

Generate association rules.

Calculate:

* Confidence  
* Lift

---

# **Example of Apriori**

Transactions:

| TID | Items |
| ----- | ----- |
| T1 | Bread, Milk |
| T2 | Bread, Beer |
| T3 | Milk, Beer |
| T4 | Bread, Milk, Beer |
| T5 | Bread, Milk |

Minimum Support:

60%

Apriori identifies:

Bread  
Milk  
Bread \+ Milk

as frequent itemsets.

---

# **Advantages of Apriori**

### **Easy to Understand**

Simple algorithm.

---

### **Effective for Market Basket Analysis**

Widely used in retail.

---

### **Generates Interpretable Rules**

Rules are easy for business users.

---

# **Disadvantages of Apriori**

### **Computationally Expensive**

Large datasets create many candidate itemsets.

---

### **Multiple Database Scans**

Requires repeated passes through data.

---

### **Slow for Large Itemsets**

Performance decreases as items increase.

---

# **Apriori vs Eclat vs FP-Growth**

| Feature | Apriori | Eclat | FP-Growth |
| ----- | ----- | ----- | ----- |
| Candidate Generation | Yes | Yes | No |
| Speed | Slowest | Faster | Fastest |
| Database Scans | Multiple | Fewer | Two |
| Memory Usage | High | Medium | Efficient |

---

# **Applications of Association Rule Learning**

### **Market Basket Analysis**

Bread → Milk

---

### **Product Recommendation**

Bought Phone  
→ Buy Charger

---

### **E-Commerce**

Cross-selling products.

---

### **Healthcare**

Disease association analysis.

---

### **Web Usage Mining**

User navigation patterns.

---

# **Python Example**

from mlxtend.frequent\_patterns import apriori  
from mlxtend.frequent\_patterns import association\_rules

frequent\_itemsets \= apriori(  
    dataset,  
    min\_support=0.05,  
    use\_colnames=True  
)

rules \= association\_rules(  
    frequent\_itemsets,  
    metric="lift",  
    min\_threshold=1  
)

print(rules)

---

# **Interview Summary**

**Association Rule Learning** is an unsupervised learning technique used to discover relationships among items in transactional data. Rules are represented as **X → Y**, meaning if X occurs, Y is likely to occur. The three key evaluation metrics are **Support** (frequency of occurrence), **Confidence** (conditional probability of Y given X), and **Lift** (strength of association beyond random chance). The **Apriori Algorithm** identifies frequent itemsets using the Apriori property that all subsets of a frequent itemset must also be frequent. It is widely used in **market basket analysis**, recommendation systems, and retail analytics to uncover meaningful purchasing patterns.

# 30

# **Eclat Algorithm (Equivalence Class Clustering and Bottom-Up Lattice Traversal)**

Eclat is an **Association Rule Learning Algorithm** used to find **frequent itemsets** in transactional databases.

Like Apriori, it is used for:

* Market Basket Analysis  
* Recommendation Systems  
* Retail Analytics  
* Web Usage Mining

However, Eclat is generally **faster than Apriori** because it uses a different approach to calculate support.

---

# **Why Do We Need Eclat?**

Apriori works using:

Bread  
Milk  
Bread \+ Milk  
Bread \+ Beer  
...

and repeatedly scans the database.

Problems:

* Multiple database scans  
* Large number of candidate itemsets  
* Slow for large datasets

Eclat solves this by using a **vertical data format**.

---

# **Intuition Behind Eclat**

Instead of storing:

| Transaction | Items |
| ----- | ----- |
| T1 | Bread, Milk |
| T2 | Bread, Beer |
| T3 | Milk, Beer |

Eclat stores:

| Item | Transactions |
| ----- | ----- |
| Bread | T1, T2 |
| Milk | T1, T3 |
| Beer | T2, T3 |

This is called the **Transaction ID Set (TID Set)** representation.

---

# **Core Idea of Eclat**

Eclat finds support by performing:

Set Intersections

instead of repeatedly scanning the database.

---

# **Example Dataset**

| Transaction | Items |
| ----- | ----- |
| T1 | Bread, Milk |
| T2 | Bread, Beer |
| T3 | Milk, Beer |
| T4 | Bread, Milk, Beer |

---

# **Vertical Representation**

### **Bread**

{T1, T2, T4}

### **Milk**

{T1, T3, T4}

### **Beer**

{T2, T3, T4}

---

# **Eclat Support Calculation**

Suppose we want support of:

Bread \+ Milk

Instead of scanning transactions:

### **Bread**

{T1, T2, T4}

### **Milk**

{T1, T3, T4}

Intersection:

{T1, T4}

Count:

2

Therefore:

Support Count:

2

Support:

Support(Bread,Milk)=\\frac{2}{4}=0.5

Support \= 50%

---

# **Eclat Algorithm Steps**

---

## **Step 1**

Convert transaction data into TID sets.

Example:

Bread → {T1,T2,T4}  
Milk  → {T1,T3,T4}  
Beer  → {T2,T3,T4}

---

## **Step 2**

Calculate support for single items.

Remove items below minimum support.

---

## **Step 3**

Generate larger itemsets using set intersections.

Example:

Bread ∩ Milk

---

## **Step 4**

Calculate support from intersection size.

---

## **Step 5**

Continue recursively until no more frequent itemsets remain.

---

# **Support in Eclat**

Support is calculated exactly the same way as Apriori.

Formula:

Support(X)=\\frac{Count(X)}{Total\\ Transactions}

The difference is:

### **Apriori**

Support → Database Scan

### **Eclat**

Support → TID Set Intersection

---

# **Example**

Transactions:

| Transaction | Items |
| ----- | ----- |
| T1 | Bread Milk |
| T2 | Bread Beer |
| T3 | Bread Milk Beer |
| T4 | Milk Beer |

---

### **Bread**

{T1,T2,T3}

### **Milk**

{T1,T3,T4}

---

### **Bread ∩ Milk**

{T1,T3}

Count:

2

Support:

Support(Bread,Milk)=\\frac{2}{4}=0.5

---

# **How Eclat Generates Rules**

After finding frequent itemsets:

Example:

{Bread, Milk}

Generate:

Bread → Milk  
Milk → Bread

Then compute:

* Confidence  
* Lift  
* Support

just like Apriori.

---

# **Example of Confidence**

Formula:

Confidence(X\\rightarrow Y)=\\frac{Support(X\\cap Y)}{Support(X)}

---

# **Example of Lift**

Formula:

Lift(X\\rightarrow Y)=\\frac{Confidence(X\\rightarrow Y)}{Support(Y)}

---

# **Apriori vs Eclat**

| Feature | Apriori | Eclat |
| ----- | ----- | ----- |
| Data Format | Horizontal | Vertical |
| Support Calculation | Database Scan | Set Intersection |
| Candidate Generation | Heavy | Less |
| Speed | Slower | Faster |
| Memory Usage | Lower | Higher |
| Large Datasets | Moderate | Better |

---

# **Why Eclat Is Faster**

Suppose:

1 Million Transactions

Apriori repeatedly scans all transactions.

Eclat simply performs:

Set Intersections

which is much faster.

---

# **Advantages of Eclat**

### **Faster than Apriori**

Uses TID intersections.

---

### **Fewer Database Scans**

Database converted once.

---

### **Efficient for Dense Datasets**

Works well when transactions contain many items.

---

### **Recursive Search**

Efficient frequent itemset generation.

---

# **Disadvantages of Eclat**

### **High Memory Usage**

Must store TID sets.

---

### **Large TID Lists**

Can become expensive for massive datasets.

---

### **Less Popular Than FP-Growth**

FP-Growth is often preferred for very large databases.

---

# **Eclat vs FP-Growth**

| Feature | Eclat | FP-Growth |
| ----- | ----- | ----- |
| Uses TID Sets | Yes | No |
| Uses Tree Structure | No | Yes |
| Speed | Fast | Faster |
| Memory Usage | Higher | Lower |
| Large Databases | Good | Excellent |

---

# **Applications of Eclat**

### **Market Basket Analysis**

Bread → Milk

---

### **Product Recommendations**

Laptop → Mouse

---

### **E-Commerce**

Cross-selling products.

---

### **Healthcare**

Disease association discovery.

---

### **Web Usage Mining**

Page visit associations.

---

# **Interview Summary**

**Eclat (Equivalence Class Clustering and Bottom-Up Lattice Traversal)** is an association rule learning algorithm used to find frequent itemsets. Unlike Apriori, which uses a horizontal transaction format and repeated database scans, Eclat uses a **vertical TID-set representation** and computes support through **set intersections**. This makes it significantly faster for many datasets. Support is calculated by counting the size of intersected transaction ID sets and dividing by the total number of transactions. After discovering frequent itemsets, association rules are generated using **Support, Confidence, and Lift**, similar to Apriori. Eclat is generally faster than Apriori but may require more memory because it stores transaction ID lists.

# 31

# **Reinforcement Learning (RL)**

Reinforcement Learning (RL) is a type of Machine Learning in which an **Agent learns by interacting with an Environment** and receiving rewards or penalties based on its actions.

The goal of the agent is:

Learn the best sequence of actions that maximizes the total reward over time.

Unlike Supervised Learning:

* No labeled dataset is provided.  
* Learning happens through trial and error.

---

# **Real-Life Intuition**

Imagine teaching a dog a trick.

### **Action**

Dog sits.

### **Reward**

Give a treat.

### **Result**

Dog learns:

Sit → Reward

and repeats the behavior.

Similarly, RL agents learn actions that lead to higher rewards.

---

# **Example: Learning to Play Chess**

Agent:

Chess AI

Environment:

Chess Board

Actions:

Move pieces

Reward:

Win \= \+100  
Lose \= \-100  
Draw \= 0

The AI gradually learns which moves increase its chances of winning.

---

# **Components of Reinforcement Learning**

---

## **1\. Agent**

The learner or decision maker.

Examples:

* Robot  
* Self-driving car  
* Chess AI

---

## **2\. Environment**

The world in which the agent operates.

Examples:

* Chess board  
* Road  
* Video game

---

## **3\. State (S)**

Current situation of the environment.

Example:

Chess board position

or

Car location

---

## **4\. Action (A)**

A decision made by the agent.

Examples:

Move Left  
Move Right  
Accelerate  
Brake

---

## **5\. Reward (R)**

Feedback received after taking an action.

Example:

\+10 → Correct move  
\-10 → Wrong move

Rewards guide learning.

---

# **RL Interaction Cycle**

Agent  
   |  
 Action  
   ↓  
Environment  
   |  
 Reward \+ New State  
   ↓  
Agent

The cycle continues repeatedly.

---

# **Goal of Reinforcement Learning**

The objective is not to maximize immediate reward.

Instead:

Maximize Long-Term Reward

Example:

A chess player may sacrifice a piece now to win later.

---

# **Return (Cumulative Reward)**

Total future reward is called Return.

Formula:

G\_t=R\_{t+1}+R\_{t+2}+R\_{t+3}+\\cdots

---

# **Discounted Reward**

Future rewards are often discounted.

Formula:

G\_t=R\_{t+1}+\\gamma R\_{t+2}+\\gamma^2R\_{t+3}+\\cdots

Where:

* (\\gamma) \= Discount Factor  
* (0 \\leq \\gamma \\leq 1\)

---

## **Interpretation of γ**

### **γ \= 0**

Only immediate reward matters

---

### **γ ≈ 1**

Future rewards are important

---

# **Policy**

A Policy determines what action the agent should take.

Represented as:

\\pi(a|s)

Meaning:

Probability of taking action a  
in state s

---

# **Value Function**

Measures how good a state is.

Formula:

V(s)=E\[G\_t|S=s\]

Meaning:

Expected future reward starting from state (s).

---

# **Q-Value (Action Value)**

Measures how good an action is in a state.

Formula:

Q(s,a)=E\[G\_t|S=s,A=a\]

This is the foundation of Q-Learning.

---

# **Exploration vs Exploitation**

One of the most important RL concepts.

---

## **Exploration**

Try new actions.

Example:

Try a new route to work

May discover something better.

---

## **Exploitation**

Use known good actions.

Example:

Take your usual fastest route

---

# **Trade-Off**

Too much exploration:

Never settles on a good strategy

Too much exploitation:

May miss better opportunities

Good RL balances both.

---

# **Types of Reinforcement Learning**

---

# **1\. Model-Based RL**

Agent knows environment dynamics.

Example:

Chess rules are known

Agent can plan ahead.

---

# **2\. Model-Free RL**

Agent learns purely through experience.

Example:

Game-playing AI

Most practical RL algorithms are model-free.

---

# **Popular RL Algorithms**

---

## **1\. Q-Learning**

Most famous RL algorithm.

Learns:

Best Action  
for every State

Uses Q-Table.

Update Equation:

Q(s,a)=Q(s,a)+\\alpha\[r+\\gamma\\max Q(s',a')-Q(s,a)\]

Where:

* (\\alpha) \= Learning Rate  
* (\\gamma) \= Discount Factor

---

## **2\. SARSA**

Similar to Q-Learning.

Uses actual next action instead of best possible action.

---

## **3\. Deep Q-Network (DQN)**

Uses Neural Networks to approximate Q-values.

Used by DeepMind to play Atari games.

---

## **4\. Policy Gradient Methods**

Learn policies directly.

Examples:

* REINFORCE  
* PPO  
* TRPO

---

## **5\. Actor-Critic Methods**

Combine:

* Policy Learning (Actor)  
* Value Learning (Critic)

Examples:

* A2C  
* A3C  
* PPO

---

# **Applications of Reinforcement Learning**

---

## **Self-Driving Cars**

Agent:

Car

Actions:

Steer  
Brake  
Accelerate

Reward:

Safe driving

---

## **Robotics**

Robot learns tasks through interaction.

---

## **Game Playing**

Examples:

* AlphaGo  
* Chess AI  
* Atari Games

---

## **Recommendation Systems**

Optimize user engagement.

---

## **Finance**

Portfolio optimization.

---

## **Resource Allocation**

Cloud computing and networking.

---

# **Advantages of Reinforcement Learning**

* Learns from experience  
* Handles sequential decision-making  
* No labeled data required  
* Can achieve superhuman performance

---

# **Disadvantages of Reinforcement Learning**

* Requires large amounts of training  
* Computationally expensive  
* Reward design is difficult  
* Slow convergence

---

# **Reinforcement Learning vs Other ML Types**

| Feature | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
| ----- | ----- | ----- | ----- |
| Training Data | Labeled | Unlabeled | Reward Signal |
| Goal | Predict Output | Discover Patterns | Maximize Reward |
| Example | Spam Detection | Clustering | Game Playing |
| Feedback | Immediate | None | Delayed Reward |

---

# **Python Example (Q-Learning Concept)**

import numpy as np

Q \= np.zeros((states, actions))

alpha \= 0.1  
gamma \= 0.9

Q\[state, action\] \= Q\[state, action\] \+ alpha \* (  
    reward \+  
    gamma \* np.max(Q\[next\_state\]) \-  
    Q\[state, action\]  
)

---

# **Interview Summary**

**Reinforcement Learning (RL)** is a Machine Learning paradigm in which an **Agent** learns by interacting with an **Environment** and receiving **Rewards** or **Penalties**. The goal is to learn a policy that maximizes cumulative future rewards. Key concepts include **State, Action, Reward, Policy, Value Function, Q-Value, Discount Factor, and Exploration vs Exploitation**. Popular algorithms include **Q-Learning, SARSA, Deep Q-Networks (DQN), Policy Gradient Methods, and Actor-Critic Methods**. RL is widely used in robotics, game playing, self-driving cars, recommendation systems, and resource optimization.

# 32

# **Multi-Armed Bandit Problem**

The **Multi-Armed Bandit (MAB)** problem is a classic Reinforcement Learning problem that models the trade-off between:

* **Exploration** → Trying new options to gather information.  
* **Exploitation** → Using the option that currently seems best.

The name comes from a casino scenario.

---

# **Intuition Behind the Multi-Armed Bandit**

Imagine entering a casino with multiple slot machines.

Each slot machine is called a:

One-Armed Bandit

because it has a single lever (arm).

Suppose there are:

10 Slot Machines

Each machine gives rewards with different unknown probabilities.

Example:

| Machine | Probability of Winning |
| ----- | ----- |
| A | 20% |
| B | 60% |
| C | 35% |
| D | 80% |

The catch:

You don't know these probabilities.

Your goal:

Maximize total reward.

---

# **The Core Challenge**

Suppose Machine B has given good rewards.

Should you:

### **Exploit**

Keep using Machine B

or

### **Explore**

Try other machines

which may be even better?

This is called the:

Exploration vs Exploitation Trade-Off

---

# **Real-Life Examples**

## **Online Advertising**

Different ads:

Ad A  
Ad B  
Ad C

Goal:

Show the ad with highest CTR

while still testing new ads.

---

## **Recommendation Systems**

Netflix:

Recommend known favorites

or

Try new content

---

## **Clinical Trials**

Try different treatments and learn which one works best.

---

# **Exploration vs Exploitation**

---

## **Exploration**

Trying less-tested options.

Example:

Try Machine D

even though Machine B currently looks best.

Purpose:

Gather Information

---

## **Exploitation**

Choose the machine currently believed to be best.

Purpose:

Maximize Immediate Reward

---

# **Why Exploration Is Necessary**

Suppose after 3 plays:

| Machine | Average Reward |
| ----- | ----- |
| A | 8 |
| B | 6 |

You might think:

Choose A forever

But perhaps:

B is actually better

and you simply got unlucky samples.

Therefore some exploration is essential.

---

# **Common Multi-Armed Bandit Strategies**

1. Greedy  
2. ε-Greedy  
3. Upper Confidence Bound (UCB)  
4. Thompson Sampling

---

# **Upper Confidence Bound (UCB)**

UCB is one of the most famous solutions to the Multi-Armed Bandit problem.

---

# **Intuition Behind UCB**

Suppose two ads have:

| Ad | Average Reward |
| ----- | ----- |
| A | 8 |
| B | 7 |

Normally we'd choose:

A

But suppose:

A shown 1000 times  
B shown only 5 times

There is much more uncertainty about B.

UCB says:

Don't look only at average reward.

Also consider uncertainty.

---

# **Main Idea of UCB**

Choose the option with:

High Average Reward  
\+  
High Uncertainty

This naturally balances:

Exploration  
and  
Exploitation

---

# **UCB Formula**

For ad (i):

UCB\_i=\\bar{x}\_i+\\sqrt{\\frac{3}{2}\\frac{\\ln n}{N\_i}}

Where:

* (\\bar{x}\_i) \= Average reward of ad i  
* (n) \= Total number of rounds  
* (N\_i) \= Number of times ad i was selected

---

# **Understanding the Formula**

The formula has two parts.

---

## **Part 1: Average Reward**

\[  
\\bar{x}\_i  
\]

Represents:

Exploitation

Higher reward → more likely to choose.

---

## **Part 2: Confidence Interval**

\[  
\\sqrt{\\frac{3}{2}\\frac{\\ln n}{N\_i}}  
\]

Represents:

Exploration

---

# **Behavior of Confidence Term**

---

## **Small (N\_i)**

Ad chosen few times.

Confidence term:

Large

Result:

More Exploration

---

## **Large (N\_i)**

Ad chosen many times.

Confidence term:

Small

Result:

More Exploitation

---

# **Why It Is Called Upper Confidence Bound**

Suppose estimated reward:

8

But actual reward could be:

7 to 9

UCB considers:

Upper Bound \= 9

Optimistically assuming the arm might be better.

This encourages exploration.

---

# **UCB Algorithm**

---

## **Step 1**

Initialize.

Select each arm once.

Example:

Ad 1  
Ad 2  
Ad 3  
...  
Ad N

---

## **Step 2**

Calculate average reward for each arm.

---

## **Step 3**

Compute UCB value.

Formula:

UCB\_i=\\bar{x}\_i+\\sqrt{\\frac{3}{2}\\frac{\\ln n}{N\_i}}

---

## **Step 4**

Choose arm with largest UCB.

---

## **Step 5**

Observe reward.

---

## **Step 6**

Update:

Number of selections  
Average reward

---

## **Step 7**

Repeat.

---

# **Example**

Suppose:

| Ad | Avg Reward | Times Selected |
| ----- | ----- | ----- |
| A | 8 | 100 |
| B | 7 | 5 |

Although:

A \> B

UCB may still choose B because:

B has high uncertainty

and might turn out to be better.

---

# **Advantages of UCB**

### **Automatic Exploration**

No manual exploration parameter needed.

---

### **Strong Theoretical Guarantees**

Provably near-optimal.

---

### **Simple Implementation**

Easy to code.

---

### **Efficient**

Balances exploration and exploitation naturally.

---

# **Disadvantages of UCB**

### **Assumes Stationary Rewards**

Reward probabilities should not change rapidly.

---

### **Sensitive to Early Observations**

Initial bad luck can affect estimates.

---

### **Less Flexible Than Thompson Sampling**

For highly complex environments.

---

# **UCB vs ε-Greedy**

| Feature | ε-Greedy | UCB |
| ----- | ----- | ----- |
| Exploration | Random | Intelligent |
| Uses Uncertainty | No | Yes |
| Performance | Good | Better |
| Theoretical Guarantee | Weak | Strong |
| Efficiency | Lower | Higher |

---

# **Applications of UCB**

### **Online Advertising**

Choose highest CTR ad.

---

### **Recommendation Systems**

Select best recommendation.

---

### **Clinical Trials**

Select best treatment.

---

### **Dynamic Pricing**

Find optimal product prices.

---

### **A/B Testing**

Determine best website version.

---

# **Python-Style Pseudocode**

for round in range(N):

    ad \= argmax(UCB)

    reward \= get\_reward(ad)

    update\_statistics(ad, reward)

---

# **Interview Summary**

The **Multi-Armed Bandit Problem** models the challenge of balancing **exploration** (trying new actions) and **exploitation** (choosing the currently best action). A popular solution is the **Upper Confidence Bound (UCB)** algorithm, which selects actions based on both their average reward and the uncertainty in their estimates. UCB chooses the action with the highest upper confidence bound:

UCB\_i=\\bar{x}\_i+\\sqrt{\\frac{3}{2}\\frac{\\ln n}{N\_i}}

This allows the algorithm to intelligently explore less-tested options while exploiting high-performing ones. UCB is widely used in online advertising, recommendation systems, A/B testing, and decision-making systems where rewards are initially unknown.

# 33

# **Thompson Sampling**

Thompson Sampling is a **probabilistic solution to the Multi-Armed Bandit Problem** that balances:

* Exploration  
* Exploitation

Unlike UCB, which uses confidence intervals, Thompson Sampling uses **Bayesian Probability** to decide which arm (ad, action, recommendation, etc.) to select.

It is one of the most popular and effective bandit algorithms used in industry.

---

# **Intuition Behind Thompson Sampling**

Suppose we have 3 ads:

Ad A  
Ad B  
Ad C

Initially we don't know:

Which ad has the highest CTR?

Instead of using a fixed confidence formula like UCB, Thompson Sampling says:

"Let's maintain a probability distribution for each ad's success rate and sample from those distributions."

The ad that looks most promising after sampling gets selected.

---

# **Bayesian View**

Thompson Sampling treats the click-through rate (CTR) as an unknown random variable.

Example:

CTR of Ad A \= ?

We don't know the true value.

Instead we maintain our belief about it.

As more clicks arrive:

Belief gets updated

---

# **Successes and Failures**

For online advertising:

### **Success**

User clicked ad  
Reward \= 1

### **Failure**

User did not click ad  
Reward \= 0

For each ad we store:

Number of Successes  
Number of Failures

---

# **Beta Distribution**

Thompson Sampling uses the **Beta Distribution**.

Formula:

Beta(\\alpha,\\beta)

Where:

* (\\alpha) \= Successes \+ 1  
* (\\beta) \= Failures \+ 1

---

# **Why Beta Distribution?**

Because:

* Outputs values between 0 and 1  
* Perfect for modeling probabilities  
* Easy Bayesian updates

---

# **Initial State**

For every ad:

Successes \= 0  
Failures \= 0

Therefore:

Beta(1,1)

This represents complete uncertainty.

All CTR values are equally likely.

---

# **Learning Process**

Suppose Ad A gets:

10 Clicks  
2 Non-clicks

Updated distribution:

Beta(11,3)

Now the algorithm believes:

CTR likely around 0.78

with reduced uncertainty.

---

# **Thompson Sampling Algorithm**

---

## **Step 1**

For each arm:

Generate a random sample from:

Beta(\\alpha\_i,\\beta\_i)

---

## **Step 2**

Compare all sampled values.

Example:

| Ad | Sampled Value |
| ----- | ----- |
| A | 0.74 |
| B | 0.82 |
| C | 0.69 |

---

## **Step 3**

Select highest sample.

Choose B

---

## **Step 4**

Observe reward.

Click \= 1  
No Click \= 0

---

## **Step 5**

Update distribution.

If clicked:

α \= α \+ 1

If not clicked:

β \= β \+ 1

---

## **Step 6**

Repeat.

---

# **Why Thompson Sampling Works**

Suppose:

### **Ad A**

Many observations:

CTR ≈ 0.7

Very certain.

---

### **Ad B**

Only a few observations.

Could be:

0.4  
or  
0.9

Very uncertain.

The Beta distribution for B is wider.

Occasionally it generates a large sample.

Thus:

B gets explored automatically

without forcing exploration.

---

# **Exploration and Exploitation in Thompson Sampling**

Unlike ε-Greedy:

Exploration is not random.

Unlike UCB:

No explicit confidence interval.

Exploration emerges naturally through probability sampling.

---

# **Example**

Suppose:

### **Ad A**

Successes \= 100  
Failures \= 50

Distribution:

Narrow

Very certain.

---

### **Ad B**

Successes \= 5  
Failures \= 2

Distribution:

Wide

More uncertain.

Sometimes B may generate larger samples and get selected.

Thus exploration occurs automatically.

---

# **Advantages of Thompson Sampling**

### **Excellent Exploration-Exploitation Balance**

Automatically balances both.

---

### **Often Outperforms UCB**

Especially in real-world advertising systems.

---

### **Bayesian Framework**

Naturally incorporates uncertainty.

---

### **Simple Updates**

Only successes and failures are needed.

---

# **Disadvantages**

### **More Mathematical**

Requires understanding probability distributions.

---

### **Sampling Overhead**

Need to sample from distributions repeatedly.

---

### **Assumes Probabilistic Reward Model**

Typically Bernoulli rewards.

---

# **UCB vs Thompson Sampling**

| Feature | UCB | Thompson Sampling |
| ----- | ----- | ----- |
| Approach | Deterministic | Probabilistic |
| Exploration Method | Confidence Bound | Bayesian Sampling |
| Formula Based | Yes | No explicit confidence formula |
| Uses Probability Distribution | No | Yes |
| Exploration Quality | Good | Excellent |
| Practical Performance | Very Good | Often Better |
| Industry Usage | High | Very High |

---

# **Intuition Comparison**

## **UCB**

Chooses:

Average\\ Reward+Confidence\\ Bonus

Example:

Ad A \= 0.8 \+ 0.1 \= 0.9  
Ad B \= 0.7 \+ 0.25 \= 0.95

Choose:

B

because uncertainty bonus is higher.

---

## **Thompson Sampling**

Chooses:

Random Sample from Belief Distribution

Example:

| Ad | Sample |
| ----- | ----- |
| A | 0.83 |
| B | 0.91 |

Choose:

B

---

# **Why Thompson Sampling Often Wins**

UCB:

Exploration is systematic

Every uncertain arm gets explored.

---

Thompson Sampling:

Exploration is proportional to uncertainty

More promising arms get explored more often.

Less promising arms gradually disappear.

This usually leads to:

Higher cumulative rewards

---

# **Applications**

* Online Advertising (Google Ads, Meta Ads)  
* News Recommendation Systems  
* Netflix Recommendations  
* A/B Testing  
* Clinical Trials  
* Dynamic Pricing  
* Website Optimization

---

# **Python Pseudocode**

for round in range(N):

    sampled\_values \= \[\]

    for ad in ads:  
        sample \= Beta(alpha\[ad\], beta\[ad\])  
        sampled\_values.append(sample)

    chosen\_ad \= argmax(sampled\_values)

    reward \= observe\_reward()

    if reward \== 1:  
        alpha\[chosen\_ad\] \+= 1  
    else:  
        beta\[chosen\_ad\] \+= 1

---

# **Interview Summary**

**Thompson Sampling** is a Bayesian solution to the Multi-Armed Bandit problem. It models the uncertainty of each arm using probability distributions (typically Beta distributions for binary rewards) and selects actions by sampling from those distributions. Arms with high expected rewards and high uncertainty naturally receive more exploration. Compared to **UCB**, which uses deterministic confidence bounds, Thompson Sampling uses probabilistic sampling and often achieves better empirical performance. It is widely used in recommendation systems, online advertising, A/B testing, and adaptive decision-making systems because it provides an efficient balance between exploration and exploitation.

# 34

# **Natural Language Processing (NLP)**

Natural Language Processing (NLP) is a branch of **Artificial Intelligence (AI)** that enables computers to understand, interpret, process, and generate human language.

It combines:

* Computer Science  
* Linguistics  
* Machine Learning  
* Deep Learning

The goal is to make machines communicate with humans in natural language.

---

# **Intuition Behind NLP**

Humans understand:

"I am feeling happy today."

instantly.

For a computer, this is just:

Characters  
Words  
Tokens

NLP converts human language into a format machines can understand and process.

---

# **Real-Life Examples of NLP**

### **ChatGPT**

Question → Understand → Generate Response

---

### **Google Translate**

English → Hindi

---

### **Siri / Alexa**

Speech → Text → Meaning → Action

---

### **Gmail Spam Filter**

Email → Spam / Not Spam

---

# **NLP Pipeline**

Typically NLP involves:

Text  
 ↓  
Preprocessing  
 ↓  
Feature Extraction  
 ↓  
Model Training  
 ↓  
Prediction

---

# **Types of NLP Tasks**

---

## **1\. Text Classification**

Assign text to categories.

Examples:

Spam Detection  
Sentiment Analysis  
News Classification

Input:

"I love this movie."

Output:

Positive

---

## **2\. Sentiment Analysis**

Determines emotional tone.

Example:

"This phone is amazing."

Output:

Positive

---

## **3\. Machine Translation**

Converts one language into another.

Example:

English → French

---

## **4\. Named Entity Recognition (NER)**

Identifies important entities.

Example:

Elon Musk founded SpaceX.

Entities:

Elon Musk → Person  
SpaceX → Organization

---

## **5\. Question Answering**

Example:

Q: Capital of India?  
A: New Delhi

---

## **6\. Text Summarization**

Converts long documents into short summaries.

---

## **7\. Speech Recognition**

Example:

Voice → Text

Used in:

* Siri  
* Alexa  
* Google Assistant

---

## **8\. Text Generation**

Example:

Prompt → Story

Used by:

* ChatGPT  
* Gemini  
* Claude

---

# **NLP Preprocessing**

Before training models, text must be cleaned.

---

## **Tokenization**

Split text into words.

Example:

"I love NLP"

becomes

\["I","love","NLP"\]

---

## **Stop Word Removal**

Remove common words.

Example:

the  
is  
are  
was

---

## **Stemming**

Reduces words to their root form.

Example:

| Word | Stem |
| ----- | ----- |
| Playing | Play |
| Played | Play |
| Plays | Play |

---

### **Example**

running  
runner  
runs

becomes

run

Popular stemmer:

Porter Stemmer

---

## **Lemmatization**

More advanced than stemming.

Example:

better

becomes

good

Produces meaningful root words.

---

# **Bag of Words (BoW)**

One of the most important classical NLP techniques.

---

# **Intuition of Bag of Words**

Suppose we have sentences:

I love AI  
AI is awesome

Create vocabulary:

\[I, Love, AI, Is, Awesome\]

Now count word occurrences.

---

## **Sentence 1**

I love AI

Vector:

\[1,1,1,0,0\]

---

## **Sentence 2**

AI is awesome

Vector:

\[0,0,1,1,1\]

---

# **Advantages of BoW**

* Simple  
* Fast  
* Easy implementation

---

# **Limitations of BoW**

Ignores:

Word order  
Grammar  
Context  
Meaning

Example:

Dog bites man  
Man bites dog

Same BoW representation.

---

# **Classical NLP Models**

Traditional NLP approaches before Deep Learning.

---

## **Common Feature Extraction Methods**

### **Bag of Words (BoW)**

Counts words.

---

### **TF-IDF**

Measures importance of words.

---

### **N-Grams**

Captures neighboring words.

Example:

Machine Learning

Bigram:

Machine Learning

---

# **Classical Machine Learning Models**

Used with BoW or TF-IDF.

Examples:

### **Naive Bayes**

Spam Detection.

---

### **Logistic Regression**

Sentiment Analysis.

---

### **SVM**

Text Classification.

---

### **Random Forest**

Document Classification.

---

# **Deep Learning in NLP**

Deep Learning automatically learns text representations without manual feature engineering.

Instead of:

BoW  
TF-IDF

it learns:

Embeddings  
Semantic Meaning  
Context

---

# **Why Deep Learning?**

Classical NLP struggles with:

Context  
Polysemy  
Long Dependencies  
Meaning

Example:

Apple released a new phone.

vs

I ate an apple.

Deep Learning understands the difference.

---

# **Word Embeddings**

Words represented as dense vectors.

Example:

King  
Queen  
Man  
Woman

similar words appear close together.

Popular embeddings:

* Word2Vec  
* GloVe  
* FastText

---

# **Deep Learning Models for NLP**

---

## **1\. RNN (Recurrent Neural Networks)**

Processes sequence one word at a time.

Example:

I love Machine Learning

Reads words sequentially.

---

## **Problem**

RNN suffers from:

Vanishing Gradient  
Long-Term Dependency Issues

---

## **2\. LSTM**

Long Short-Term Memory Network.

Designed to remember long-term information.

Used in:

* Translation  
* Speech Recognition

---

## **3\. GRU**

Simplified version of LSTM.

Faster training.

---

## **4\. Transformers**

Modern NLP architecture.

Examples:

* BERT  
* GPT  
* Gemini  
* Claude

Most state-of-the-art NLP systems use Transformers.

---

# **Sequence-to-Sequence (Seq2Seq) Models**

Seq2Seq models convert one sequence into another sequence.

---

# **Intuition**

Input:

English Sentence

Output:

French Sentence

Example:

Hello World

↓

Bonjour le Monde

---

# **Architecture**

### **Encoder**

Reads input sequence.

---

### **Decoder**

Generates output sequence.

---

Diagram:

Input Sentence  
      ↓  
   Encoder  
      ↓  
Context Vector  
      ↓  
   Decoder  
      ↓  
Output Sentence

---

# **Applications of Seq2Seq**

### **Machine Translation**

English → French

---

### **Text Summarization**

Long Article → Summary

---

### **Chatbots**

Question → Response

---

### **Speech Recognition**

Audio → Text

---

# **Classical NLP vs Deep Learning NLP**

| Feature | Classical NLP | Deep Learning NLP |
| ----- | ----- | ----- |
| Feature Engineering | Manual | Automatic |
| Context Understanding | Poor | Excellent |
| Data Requirement | Small | Large |
| Training Time | Fast | Slow |
| Accuracy | Lower | Higher |
| Interpretability | Higher | Lower |
| Examples | BoW, TF-IDF, SVM | LSTM, BERT, GPT |

---

# **Use Cases of NLP**

## **Healthcare**

* Medical report analysis  
* Disease prediction

---

## **Finance**

* Fraud detection  
* Sentiment analysis

---

## **E-Commerce**

* Product recommendations  
* Review analysis

---

## **Customer Support**

* Chatbots  
* Virtual assistants

---

## **Search Engines**

* Query understanding  
* Ranking results

---

## **Social Media**

* Toxic comment detection  
* Sentiment analysis

---

# **Interview Summary**

**Natural Language Processing (NLP)** is the field of AI that enables machines to understand and generate human language. Traditional NLP relies on preprocessing techniques such as **Tokenization, Stop Word Removal, Stemming, Lemmatization**, and feature extraction methods like **Bag of Words (BoW), TF-IDF, and N-Grams**, combined with machine learning algorithms such as **Naive Bayes, Logistic Regression, and SVM**. Modern NLP uses **Deep Learning**, including **RNNs, LSTMs, GRUs, and Transformers**, which automatically learn contextual representations of language. **Sequence-to-Sequence (Seq2Seq)** models use an Encoder-Decoder architecture for tasks like machine translation, summarization, and chatbots. Today, Transformer-based models such as **BERT** and **GPT** dominate NLP due to their superior ability to capture context and meaning.

# 36

# **Artificial Neural Networks (ANN)**

Artificial Neural Networks are Machine Learning models inspired by the structure and working of the human brain.

They are the foundation of:

* Deep Learning  
* Computer Vision  
* NLP  
* Generative AI  
* Speech Recognition

The basic idea is:

Learn patterns from data by adjusting connections between artificial neurons.

---

# **Intuition Behind Neural Networks**

Suppose we want to predict whether a customer will buy a product.

Inputs:

Age  
Salary  
Experience

A neural network learns:

Input Features  
      ↓  
Hidden Processing  
      ↓  
Prediction

without explicitly programming the rules.

---

# **The Artificial Neuron**

The basic building block of a neural network is a neuron.

Biological analogy:

Dendrites → Inputs  
Cell Body → Processing  
Axon → Output

Artificial neuron:

Inputs → Weighted Sum → Activation Function → Output

---

# **Structure of a Neuron**

Suppose inputs:

x1 \= Age  
x2 \= Salary  
x3 \= Experience

Each input has a weight:

w1  
w2  
w3

The neuron computes:

\[  
z=w\_1x\_1+w\_2x\_2+w\_3x\_3+b  
\]

where:

* (w) \= weights  
* (b) \= bias

---

# **Why Do We Need Weights?**

Weights determine importance.

Example:

Salary → Very Important  
Weight \= 0.8

Age → Less Important  
Weight \= 0.2

The network learns these automatically.

---

# **Bias**

Bias shifts the activation threshold.

Without bias:

Output always passes through origin

Bias gives flexibility.

---

# **Activation Function**

After calculating the weighted sum, we apply an activation function.

Purpose:

Introduce non-linearity.

Without activation functions:

Many layers \= One Linear Equation

No benefit from deep networks.

---

# **Common Activation Functions**

---

## **1\. Sigmoid**

Maps values between 0 and 1\.

Used in:

* Binary classification output layer

Advantages:

* Produces probabilities

Disadvantages:

* Vanishing gradient problem

---

## **2\. Tanh**

Maps values between \-1 and 1\.

\[  
tanh(x)=\\frac{e^x-e^{-x}}{e^x+e^{-x}}  
\]

Advantages:

* Zero-centered

Disadvantages:

* Still suffers vanishing gradients

---

## **3\. ReLU (Most Popular)**

Formula:

\[  
f(x)=max(0,x)  
\]

Intuition:

If x \> 0 → Output x  
If x ≤ 0 → Output 0

Advantages:

* Fast  
* Solves vanishing gradient problem  
* Used in most hidden layers

---

## **4\. Softmax**

Used for multi-class classification.

Formula:

\[  
P(y\_i)=\\frac{e^{z\_i}}{\\sum e^{z\_j}}  
\]

Outputs probabilities that sum to 1\.

---

# **How Neural Networks Work**

Suppose:

Input Layer  
    ↓  
Hidden Layer  
    ↓  
Output Layer

Example:

Age  
Salary  
Experience  
     ↓  
 Hidden Neurons  
     ↓  
 Buy / Not Buy

---

# **Layers in Neural Networks**

## **Input Layer**

Receives features.

Example:

Age  
Salary  
Experience

---

## **Hidden Layer**

Learns patterns and relationships.

Example:

Customer Wealth  
Customer Interest

(not explicitly defined)

---

## **Output Layer**

Produces prediction.

Examples:

0 or 1

or

Cat  
Dog  
Bird

---

# **Deep Learning**

When a neural network contains multiple hidden layers, it becomes a **Deep Neural Network**.

Example:

Input  
 ↓  
Hidden Layer 1  
 ↓  
Hidden Layer 2  
 ↓  
Hidden Layer 3  
 ↓  
Output

This is called Deep Learning.

---

# **Why Deep Learning Works**

Each layer learns increasingly complex features.

Example: Image Recognition

Layer 1:

Edges

Layer 2:

Shapes

Layer 3:

Eyes  
Nose

Layer 4:

Face

---

# **Forward Propagation**

Forward Propagation is the process of making predictions.

Steps:

### **Step 1**

Input enters network.

---

### **Step 2**

Weighted sums are computed.

\[  
z=w^Tx+b  
\]

---

### **Step 3**

Activation function applied.

---

### **Step 4**

Output generated.

Example:

Prediction \= 0.8

---

# **Loss Function**

The prediction is compared with actual value.

Example:

Predicted \= 0.8  
Actual \= 1

Error exists.

Loss function measures this error.

---

# **Gradient Descent**

Gradient Descent is the optimization algorithm used to minimize loss.

Goal:

Find weights producing minimum error

---

# **Intuition of Gradient Descent**

Imagine standing on a mountain.

Goal:

Reach the lowest point

You move in the direction of steepest descent.

Similarly:

High Loss  
 ↓  
Lower Loss  
 ↓  
Minimum Loss

---

# **Gradient Descent Update Rule**

Where:

* (w) \= weight  
* (\\eta) \= learning rate  
* (L) \= loss

---

# **Learning Rate**

Controls step size.

### **Large Learning Rate**

May overshoot minimum

---

### **Small Learning Rate**

Very slow learning

---

# **Types of Gradient Descent**

---

## **1\. Batch Gradient Descent**

Uses entire dataset.

Compute error on all samples  
Update once

Advantages:

* Stable

Disadvantages:

* Slow

---

## **2\. Stochastic Gradient Descent (SGD)**

Updates after every training sample.

One Sample  
↓  
Update Weights

Advantages:

* Fast  
* Escapes local minima

Disadvantages:

* Noisy updates

---

## **3\. Mini-Batch Gradient Descent**

Most common approach.

Uses small batches.

Example:

Batch Size \= 32

Advantages:

* Fast  
* Stable

---

# **Backpropagation**

Backpropagation is the learning algorithm of neural networks.

Purpose:

Determine how much each weight contributed to the error.

---

# **Intuition of Backpropagation**

Suppose:

Prediction \= 0.9  
Actual \= 1

Small error exists.

Backpropagation asks:

Which weights caused the error?

and adjusts them accordingly.

---

# **How Backpropagation Works**

### **Step 1**

Perform Forward Propagation.

Input → Output

---

### **Step 2**

Compute Loss.

Actual vs Predicted

---

### **Step 3**

Calculate Gradients.

Find:

\[  
\\frac{\\partial L}{\\partial w}  
\]

for every weight.

---

### **Step 4**

Propagate Error Backwards.

Output Layer  
↓  
Hidden Layer  
↓  
Input Layer

---

### **Step 5**

Update Weights.

Using Gradient Descent:

---

### **Step 6**

Repeat for many epochs.

Network gradually learns.

---

# **Complete Learning Cycle**

Input Data  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Calculate Loss  
      ↓  
Backpropagation  
      ↓  
Update Weights  
      ↓  
Repeat

---

# **Classical ML vs Neural Networks**

| Feature | Classical ML | Neural Networks |
| ----- | ----- | ----- |
| Feature Engineering | Manual | Automatic |
| Data Requirement | Small | Large |
| Training Time | Fast | Slow |
| Non-linearity | Limited | Excellent |
| Accuracy | Good | Very High |
| Images & NLP | Weak | Excellent |

---

# **Applications of Neural Networks**

### **Computer Vision**

* Face Recognition  
* Object Detection

### **NLP**

* ChatGPT  
* Translation  
* Sentiment Analysis

### **Speech Recognition**

* Siri  
* Alexa

### **Healthcare**

* Disease Detection

### **Finance**

* Fraud Detection

### **Recommendation Systems**

* Netflix  
* YouTube

---

# **Interview Summary**

An **Artificial Neural Network (ANN)** is a machine learning model inspired by the human brain. It consists of **neurons** connected through weighted links organized into **input, hidden, and output layers**. Each neuron computes a weighted sum of inputs, adds a bias, and applies an **activation function** such as Sigmoid, ReLU, or Softmax. Learning occurs through **Forward Propagation**, where predictions are generated, and **Backpropagation**, where errors are propagated backward to compute gradients. The weights are then updated using **Gradient Descent** or **Stochastic Gradient Descent (SGD)** to minimize the loss function. Networks with multiple hidden layers are called **Deep Neural Networks**, forming the foundation of modern Deep Learning applications in computer vision, NLP, speech recognition, and generative AI.

# 37

# **Convolutional Neural Networks (CNN)**

A **Convolutional Neural Network (CNN)** is a specialized Deep Learning architecture primarily used for:

* Image Classification  
* Object Detection  
* Face Recognition  
* Medical Imaging  
* Video Analysis

CNNs are designed to automatically learn spatial features from images.

---

# **Why Not Use a Regular Neural Network for Images?**

Suppose an image size is:

256 × 256 × 3

(Number of pixels × RGB channels)

Total inputs:

196,608 features

A fully connected neural network would require millions of parameters.

Problems:

* Huge memory usage  
* Slow training  
* Overfitting

CNN solves this using:

Convolution  
Parameter Sharing  
Feature Extraction

---

# **Intuition Behind CNN**

Imagine recognizing a face.

You don't look at every pixel individually.

You first detect:

Edges  
↓  
Shapes  
↓  
Eyes  
↓  
Nose  
↓  
Face

CNN learns features similarly.

---

# **CNN Architecture**

Typical CNN:

Input Image  
      ↓  
Convolution Layer  
      ↓  
ReLU Layer  
      ↓  
Pooling Layer  
      ↓  
Convolution Layer  
      ↓  
ReLU Layer  
      ↓  
Pooling Layer  
      ↓  
Flattening  
      ↓  
Fully Connected Layer  
      ↓  
Softmax  
      ↓  
Prediction

---

# **1\. Convolution Operation**

The most important operation in CNN.

---

# **What is Convolution?**

A small matrix called a:

Filter  
Kernel  
Feature Detector

slides across the image.

Example:

Image:

1 1 1  
0 1 0  
0 1 0

Filter:

1 0  
0 1

The filter performs element-wise multiplication and summation.

This produces a new value.

The process repeats across the image.

---

# **Convolution Formula**

Mathematically:

\[  
(I\*K)(x,y)  
\]

where:

* (I) \= Image  
* (K) \= Kernel

The output is called a Feature Map.

---

# **Why Convolution?**

Instead of learning:

Millions of pixels

CNN learns:

Edges  
Corners  
Textures  
Patterns

which are much more useful.

---

# **Feature Detectors (Kernels)**

A kernel is a small matrix.

Example:

3 × 3  
5 × 5  
7 × 7

Different kernels learn different features.

---

# **Edge Detector**

Example kernel:

\-1 \-1 \-1  
 0  0  0  
 1  1  1

Detects horizontal edges.

---

# **Vertical Edge Detector**

\-1 0 1  
\-1 0 1  
\-1 0 1

Detects vertical edges.

---

# **Feature Maps**

After convolution, we obtain:

Feature Map  
Activation Map

A feature map highlights locations where a feature appears.

Example:

Original Image  
      ↓  
Edge Detector  
      ↓  
Edge Feature Map

---

# **Multiple Feature Maps**

CNN does not use one filter.

It learns many filters:

Filter 1 → Edges

Filter 2 → Corners

Filter 3 → Textures

Filter 4 → Shapes

Each creates its own feature map.

---

# **2\. ReLU Layer**

After convolution we apply:

ReLU  
(Rectified Linear Unit)

Formula:

\[  
f(x)=max(0,x)  
\]

---

# **Intuition**

Positive Value → Keep  
Negative Value → 0

Example:

Input:

\[-3, 4, \-2, 8\]

Output:

\[0, 4, 0, 8\]

---

# **Why ReLU?**

Advantages:

* Introduces non-linearity  
* Faster training  
* Reduces vanishing gradients

---

# **3\. Pooling Layer**

Pooling reduces feature map size.

Purpose:

Reduce Computation  
Reduce Overfitting  
Retain Important Features

---

# **Max Pooling**

Most common pooling method.

Example:

Input:

1 3  
4 2

Output:

4

Largest value selected.

---

# **Intuition**

Instead of remembering every pixel:

Remember strongest feature

---

# **Max Pooling Example**

Feature Map:

1 2 3 4  
5 6 7 8  
1 2 3 4  
5 6 7 8

Using:

2 × 2 Pooling

Output:

6 8  
6 8

---

# **Average Pooling (Mean Pooling)**

Instead of maximum value:

Take average.

Input:

1 3  
4 2

Output:

2.5

---

# **Max Pooling vs Average Pooling**

| Feature | Max Pooling | Average Pooling |
| ----- | ----- | ----- |
| Keeps Strongest Feature | Yes | No |
| Most Common | Yes | Less Common |
| Accuracy | Higher | Lower |

---

# **4\. Flattening**

After several:

Convolution  
Pooling

we obtain feature maps.

Example:

2 × 2 × 3

CNN cannot directly feed this into dense layers.

Flattening converts it into:

\[1,2,3,4,5,6,7,8,9,10,11,12\]

a single vector.

---

# **Example**

Before:

\[  
 \[1 2\]  
 \[3 4\]  
\]

After Flattening:

\[1,2,3,4\]

---

# **Why Flatten?**

Transforms spatial features into a format suitable for classification.

---

# **5\. Fully Connected Layer**

Also called:

Dense Layer

Works like a regular neural network.

Receives flattened features.

Example:

Edges  
Shapes  
Textures

and combines them to make decisions.

---

# **Intuition**

Earlier layers learn:

Eyes  
Nose  
Mouth

Fully connected layer learns:

Eyes \+ Nose \+ Mouth  
\=  
Face

---

# **6\. Softmax Layer**

Used in multi-class classification.

Converts outputs into probabilities.

Formula:

\[  
P(y\_i)=\\frac{e^{z\_i}}{\\sum e^{z\_j}}  
\]

---

# **Example**

Network output:

\[3.5, 1.2, 0.5\]

Softmax:

\[0.85, 0.10, 0.05\]

Interpretation:

Class 1 → 85%  
Class 2 → 10%  
Class 3 → 5%

Prediction:

Class 1

---

# **Cross Entropy Loss**

Used to measure prediction error.

For classification:

\[  
Loss=-\\sum y\_i \\log(\\hat{y\_i})  
\]

---

# **Intuition**

Suppose:

Actual:

Cat

Prediction:

Cat \= 0.95

Loss:

Very Small

---

Prediction:

Cat \= 0.10

Loss:

Very Large

Cross Entropy penalizes wrong confident predictions heavily.

---

# **Complete CNN Workflow**

Image  
 ↓  
Convolution  
 ↓  
Feature Maps  
 ↓  
ReLU  
 ↓  
Pooling  
 ↓  
Convolution  
 ↓  
ReLU  
 ↓  
Pooling  
 ↓  
Flattening  
 ↓  
Fully Connected Layer  
 ↓  
Softmax  
 ↓  
Prediction  
 ↓  
Cross Entropy Loss  
 ↓  
Backpropagation  
 ↓  
Update Filters

---

# **Why CNN Works So Well**

Early Layers:

Edges  
Corners  
Textures

Middle Layers:

Shapes  
Patterns

Deep Layers:

Objects  
Faces  
Cars  
Animals

This hierarchical feature learning makes CNNs extremely powerful.

---

# **Applications of CNN**

### **Computer Vision**

* Image Classification  
* Object Detection

### **Face Recognition**

* Face Unlock  
* Surveillance

### **Medical Imaging**

* Tumor Detection  
* X-Ray Analysis

### **Self-Driving Cars**

* Lane Detection  
* Traffic Sign Recognition

### **OCR**

* Handwriting Recognition  
* Document Digitization

---

# **Interview Summary**

A **Convolutional Neural Network (CNN)** is a deep learning architecture designed for image and spatial data processing. It consists of **Convolution Layers**, which use learnable filters (feature detectors) to extract features such as edges, textures, and shapes, producing **Feature Maps**. The **ReLU Layer** introduces non-linearity, while **Pooling Layers** (Max Pooling and Average Pooling) reduce spatial dimensions and computational complexity. The resulting feature maps are converted into a one-dimensional vector through **Flattening**, which is fed into **Fully Connected Layers** for classification. The final layer typically uses **Softmax** to generate class probabilities, and training is guided by **Cross Entropy Loss** and **Backpropagation**. CNNs form the backbone of modern computer vision systems and are widely used in image recognition, object detection, facial recognition, and medical image analysis.

# 39

# **Principal Component Analysis (PCA)**

Principal Component Analysis (PCA) is an **Unsupervised Dimensionality Reduction Technique** used to reduce the number of features while preserving as much information (variance) as possible.

It is widely used for:

* Feature Reduction  
* Data Visualization  
* Noise Reduction  
* Faster Model Training  
* Handling Multicollinearity

---

# **Intuition Behind PCA**

Suppose a dataset contains:

Age  
Salary  
Experience  
Education  
City  
Credit Score

Some features may be highly correlated.

Example:

Experience ↑  
Salary ↑

These features contain overlapping information.

PCA combines such features into new variables called:

Principal Components

which retain most of the important information.

---

# **Why PCA?**

Problems with high-dimensional data:

* More computation  
* More storage  
* Overfitting  
* Curse of Dimensionality

PCA reduces dimensions while retaining maximum variance.

---

# **Intuition Using a 2D Dataset**

Suppose data points form an elongated cloud:

     \*  
    \*   \*  
  \*      \*  
 \*         \*

Most variation occurs along one direction.

Instead of using:

X-axis  
Y-axis

PCA finds a new axis that captures maximum variance.

This new axis becomes:

Principal Component 1 (PC1)

---

# **Principal Components**

Principal Components are:

* New variables  
* Linear combinations of original features  
* Uncorrelated with each other

---

# **PC1 (First Principal Component)**

Captures:

Maximum Variance

among all possible directions.

---

# **PC2 (Second Principal Component)**

Captures:

Remaining Maximum Variance

and is perpendicular to PC1.

---

# **Example**

Suppose:

Original Features \= 100

PCA may reduce them to:

10 Principal Components

while preserving:

95% Information

---

# **PCA Algorithm**

---

## **Step 1: Standardize the Dataset**

PCA is affected by feature scales.

Example:

| Feature | Range |
| ----- | ----- |
| Age | 20–60 |
| Salary | 10,000–1,000,000 |

Salary dominates.

Therefore standardization is required.

Formula:

\[  
z=\\frac{x-\\mu}{\\sigma}  
\]

Result:

Mean \= 0  
Variance \= 1

---

## **Step 2: Compute Covariance Matrix**

Covariance measures relationships between features.

For two variables:

\[  
Cov(X,Y)  
\]

Interpretation:

### **Positive Covariance**

X increases  
Y increases

---

### **Negative Covariance**

X increases  
Y decreases

---

### **Zero Covariance**

No linear relationship

Covariance Matrix:

\[  
C=  
\\begin{bmatrix}  
Cov(X\_1,X\_1) & Cov(X\_1,X\_2) \\  
Cov(X\_2,X\_1) & Cov(X\_2,X\_2)  
\\end{bmatrix}  
\]

---

## **Step 3: Compute Eigenvalues and Eigenvectors**

This is the heart of PCA.

From the covariance matrix we calculate:

### **Eigenvectors**

Represent directions.

These become:

Principal Components

---

### **Eigenvalues**

Represent importance.

Higher eigenvalue:

More Variance Explained

---

# **Example**

| Component | Eigenvalue |
| ----- | ----- |
| PC1 | 8 |
| PC2 | 3 |
| PC3 | 1 |

PC1 captures most information.

---

## **Step 4: Sort Eigenvalues**

Arrange in descending order.

Example:

PC1 \= 8  
PC2 \= 3  
PC3 \= 1

Order:

PC1  
PC2  
PC3

---

## **Step 5: Select Top K Components**

Choose components explaining most variance.

Example:

| Component | Variance Explained |
| ----- | ----- |
| PC1 | 70% |
| PC2 | 20% |
| PC3 | 5% |
| PC4 | 5% |

Choose:

PC1 \+ PC2

Total:

90% Variance

---

## **Step 6: Transform Data**

Project original data onto selected principal components.

Result:

Original Features  
↓  
Principal Components

Reduced-dimensional dataset obtained.

---

# **Explained Variance Ratio**

Measures information retained.

Formula:

\[  
\\frac{\\lambda\_i}{\\sum \\lambda}  
\]

where:

* (\\lambda\_i) \= Eigenvalue

---

# **Example**

Eigenvalues:

8, 3, 1

Total:

12

PC1:

8/12 \= 66.7%

PC2:

3/12 \= 25%

PC3:

1/12 \= 8.3%

---

# **Choosing Number of Components**

### **Explained Variance Rule**

Keep enough components to explain:

90%  
95%  
99%

variance.

---

### **Scree Plot**

Plot:

X-axis → Components  
Y-axis → Eigenvalues

Look for the elbow point.

---

# **Geometric Interpretation**

PCA performs:

Rotation of Coordinate System

to align axes with maximum variance directions.

The data itself does not change.

Only the coordinate system changes.

---

# **Advantages of PCA**

### **Dimensionality Reduction**

Fewer features.

---

### **Faster Training**

Smaller datasets.

---

### **Noise Reduction**

Removes low-variance components.

---

### **Visualization**

High-dimensional data visualized in 2D or 3D.

---

### **Removes Multicollinearity**

Principal components are orthogonal.

---

# **Disadvantages of PCA**

### **Reduced Interpretability**

Principal components lose original meaning.

---

### **Linear Technique**

Cannot capture complex non-linear relationships.

---

### **Information Loss**

Some variance is discarded.

---

# **PCA Example**

Suppose:

100 Features

After PCA:

10 Components

Explained Variance:

95%

Result:

90% reduction in dimensions  
Only 5% information lost

---

# **PCA vs Feature Selection**

| Feature | PCA | Feature Selection |
| ----- | ----- | ----- |
| Creates New Features | Yes | No |
| Removes Features | No | Yes |
| Interpretability | Lower | Higher |
| Handles Correlation | Excellent | Moderate |

---

# **Applications of PCA**

### **Face Recognition**

Eigenfaces.

---

### **Image Compression**

Reduce image dimensions.

---

### **Data Visualization**

100D → 2D.

---

### **Genomics**

Gene expression analysis.

---

### **Finance**

Stock market analysis.

---

### **Preprocessing for ML**

Reduce feature count before training.

---

# **Python Example**

from sklearn.decomposition import PCA

pca \= PCA(n\_components=2)

X\_pca \= pca.fit\_transform(X)

---

# **Interview Summary**

**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated variables called **Principal Components**. The first principal component captures the maximum variance in the data, while each subsequent component captures the remaining variance and is orthogonal to previous components. The PCA algorithm consists of **standardizing the data, computing the covariance matrix, finding eigenvalues and eigenvectors, sorting them by explained variance, selecting the top components, and projecting the data onto those components**. PCA is widely used for dimensionality reduction, visualization, noise reduction, and handling multicollinearity while retaining most of the original information.

# 40

# **Linear Discriminant Analysis (LDA)**

**Linear Discriminant Analysis (LDA)** is a **Supervised Dimensionality Reduction Technique** used to reduce features while preserving maximum class separability.

It is commonly used for:

* Classification Problems  
* Feature Extraction  
* Pattern Recognition  
* Face Recognition  
* Preprocessing before Classification

---

# **Intuition Behind LDA**

Suppose we have two classes:

Class A → Red Points  
Class B → Blue Points

with two features:

Height  
Weight

PCA would find directions with maximum variance.

But variance does not necessarily separate classes.

LDA asks:

Which projection best separates different classes?

---

# **PCA vs LDA Intuition**

Imagine students from two departments:

Computer Science  
Mechanical Engineering

---

### **PCA**

Looks for:

Maximum Spread

ignoring class labels.

---

### **LDA**

Looks for:

Maximum Class Separation

using class labels.

This is why LDA is supervised while PCA is unsupervised.

---

# **Goal of LDA**

LDA tries to:

### **Maximize**

Distance Between Classes

and

### **Minimize**

Spread Within Classes

---

# **Visual Intuition**

Suppose two classes overlap:

○ ○ ○ ○  
○ ○ ○ ○

△ △ △ △  
△ △ △ △

LDA finds a line where:

Projected classes  
are as far apart as possible

---

# **Key Idea**

LDA finds new axes called:

Linear Discriminants

that maximize class separability.

---

# **Terminology**

---

## **Within-Class Scatter**

Measures how spread out points are inside each class.

Small within-class scatter is desirable.

Example:

Class A points tightly grouped

Good.

---

## **Between-Class Scatter**

Measures distance between class centers.

Large between-class scatter is desirable.

Example:

Class A center far from Class B center

Good.

---

# **Objective Function of LDA**

LDA maximizes:

\[  
\\frac{Between\\ Class\\ Scatter}{Within\\ Class\\ Scatter}  
\]

In matrix form:

\[  
J(w)=\\frac{w^TS\_Bw}{w^TS\_Ww}  
\]

Where:

* (S\_B) \= Between-class scatter matrix  
* (S\_W) \= Within-class scatter matrix

---

# **Intuition of the Formula**

Large numerator:

Classes far apart

Small denominator:

Classes compact

Result:

Excellent separation

---

# **LDA Algorithm**

---

## **Step 1: Standardize Data**

Normalize features so large-scale variables do not dominate.

---

## **Step 2: Compute Mean of Each Class**

Example:

Class A Mean  
Class B Mean  
Class C Mean

---

## **Step 3: Compute Within-Class Scatter Matrix**

Measures variability inside each class.

\[  
S\_W=\\sum (x-\\mu\_c)(x-\\mu\_c)^T  
\]

---

## **Step 4: Compute Between-Class Scatter Matrix**

Measures distance between class means.

\[  
S\_B=\\sum n\_c(\\mu\_c-\\mu)(\\mu\_c-\\mu)^T  
\]

where:

* (\\mu\_c) \= Class mean  
* (\\mu) \= Overall mean

---

## **Step 5: Solve Eigenvalue Problem**

Find:

\[  
S\_W^{-1}S\_B  
\]

Then compute:

* Eigenvalues  
* Eigenvectors

---

## **Step 6: Select Top Discriminants**

Choose eigenvectors corresponding to largest eigenvalues.

These become:

LD1  
LD2  
LD3  
...

---

## **Step 7: Project Data**

Transform original features onto discriminant axes.

Result:

High-Dimensional Data  
          ↓  
     LDA Space  
          ↓  
 Better Separation

---

# **Number of Components in LDA**

This is a common interview question.

For:

C Classes

Maximum number of LDA components:

\[  
C-1  
\]

---

## **Example**

2 Classes:

Max Components \= 1

---

3 Classes:

Max Components \= 2

---

5 Classes:

Max Components \= 4

---

# **Why C \- 1?**

Only (C-1) independent directions are needed to separate (C) class means.

---

# **Example**

Suppose:

Features \= 100  
Classes \= 3

LDA can produce at most:

2 Discriminant Components

even though there are 100 features.

---

# **PCA vs LDA**

| Feature | PCA | LDA |
| ----- | ----- | ----- |
| Type | Unsupervised | Supervised |
| Uses Labels | No | Yes |
| Goal | Maximize Variance | Maximize Class Separation |
| Components | Up to Features | Up to C−1 |
| Classification Performance | Moderate | Better |
| Dimensionality Reduction | Yes | Yes |

---

# **Example**

Dataset:

| Feature 1 | Feature 2 | Class |
| ----- | ----- | ----- |
| 2 | 3 | A |
| 3 | 4 | A |
| 7 | 8 | B |
| 8 | 9 | B |

LDA finds a projection where:

Class A  
     |  
     |  
     |  
\-----|----------------  
     |  
     |  
Class B

Classes become maximally separated.

---

# **Advantages of LDA**

### **Better Class Separation**

Designed specifically for classification.

---

### **Reduces Overfitting**

Fewer dimensions.

---

### **Faster Training**

Smaller feature space.

---

### **Works Well with Small Datasets**

Unlike deep learning.

---

# **Disadvantages of LDA**

### **Assumes Linear Separation**

Classes should be linearly separable.

---

### **Assumes Normal Distribution**

Works best when features are Gaussian.

---

### **Assumes Equal Covariance**

All classes should have similar covariance matrices.

---

### **Sensitive to Outliers**

Outliers can distort class means.

---

# **Applications of LDA**

### **Face Recognition**

Eigenfaces and Fisherfaces.

---

### **Medical Diagnosis**

Disease classification.

---

### **Customer Segmentation**

Classifying customer groups.

---

### **Speech Recognition**

Speaker classification.

---

### **Bioinformatics**

Gene classification.

---

# **LDA as a Classifier**

Besides dimensionality reduction, LDA itself can be used directly as a classification algorithm.

It assumes:

Each class follows  
a Gaussian distribution  
with common covariance matrix.

Then predicts the most likely class.

---

# **PCA vs LDA Interview Answer**

**PCA** finds directions of maximum variance without considering class labels, whereas **LDA** finds directions that maximize separation between classes using class labels. PCA is unsupervised and useful for general dimensionality reduction, while LDA is supervised and particularly useful as a preprocessing step for classification tasks.

---

# **Interview Summary**

**Linear Discriminant Analysis (LDA)** is a supervised dimensionality reduction technique that projects data onto lower-dimensional spaces while maximizing class separability. It achieves this by maximizing **between-class scatter** and minimizing **within-class scatter**. The algorithm computes class means, within-class scatter matrices, and between-class scatter matrices, then solves an eigenvalue problem to find optimal discriminant directions. Unlike PCA, which focuses on preserving variance, LDA focuses on improving classification performance. For a dataset with **C classes**, LDA can produce at most **C−1 discriminant components** and is widely used in classification, face recognition, medical diagnosis, and pattern recognition.

# 41

# **Kernel PCA (Kernel Principal Component Analysis)**

**Kernel PCA** is an extension of **Principal Component Analysis (PCA)** that can capture **non-linear relationships** in data.

While PCA works well when data is linearly separable, Kernel PCA can uncover complex structures by mapping data into a higher-dimensional feature space.

---

# **Why Do We Need Kernel PCA?**

Regular PCA assumes:

Important patterns lie along  
straight-line directions.

This works for many datasets but fails when data has non-linear structures.

---

# **Example Where PCA Fails**

Suppose data forms concentric circles:

    ○ ○ ○ ○ ○

   ○           ○

       ● ● ●

   ○           ○

     ○ ○ ○ ○ ○

Classes are clearly different.

However:

No straight line  
can separate them.

Regular PCA cannot discover this structure.

---

# **Intuition Behind Kernel PCA**

Kernel PCA says:

Instead of performing PCA in the original space, first map data into a higher-dimensional space where separation becomes easier.

Example:

Original Space:

2D

Mapped Space:

100D  
1000D  
Infinite Dimensions

In this new space:

Non-linear patterns  
become linear.

Then standard PCA is applied.

---

# **The Core Idea**

Regular PCA:

Original Data  
     ↓  
Compute Covariance  
     ↓  
Eigenvectors  
     ↓  
Principal Components

Kernel PCA:

Original Data  
     ↓  
Kernel Transformation  
     ↓  
Higher-Dimensional Space  
     ↓  
PCA  
     ↓  
Kernel Principal Components

---

# **The Problem with Explicit Mapping**

Suppose:

x \= (x₁,x₂)

Mapping:

\[  
\\phi(x)=(x\_1^2,x\_2^2,\\sqrt2x\_1x\_2)  
\]

For small dimensions this is manageable.

But imagine:

100 Features

Mapping could create:

Thousands  
or Millions  
of Features

Computationally expensive.

---

# **The Kernel Trick**

The genius idea behind Kernel PCA:

Compute inner products in the higher-dimensional space without explicitly performing the transformation.

Instead of computing:

\[  
\\phi(x)  
\]

we compute:

\[  
K(x\_i,x\_j)  
\]

called the Kernel Function.

This is known as:

Kernel Trick

---

# **Kernel Function**

A kernel measures similarity between two points.

General form:

\[  
K(x\_i,x\_j)=\\phi(x\_i)^T\\phi(x\_j)  
\]

This allows computations in the high-dimensional space indirectly.

---

# **Steps of Kernel PCA**

---

## **Step 1**

Choose a kernel.

Examples:

* RBF (Gaussian)  
* Polynomial  
* Sigmoid

---

## **Step 2**

Compute Kernel Matrix.

\[  
K=  
\\begin{bmatrix}  
K(x\_1,x\_1) & K(x\_1,x\_2) \\  
K(x\_2,x\_1) & K(x\_2,x\_2)  
\\end{bmatrix}  
\]

---

## **Step 3**

Center the Kernel Matrix.

Equivalent to centering data in feature space.

---

## **Step 4**

Compute Eigenvalues and Eigenvectors.

Just like PCA.

---

## **Step 5**

Select Top Components.

Choose components with largest eigenvalues.

---

## **Step 6**

Project Data.

Transform original data into Kernel PCA space.

---

# **Types of Kernel Functions**

---

# **1\. Linear Kernel**

Formula:

\[  
K(x\_i,x\_j)=x\_i^Tx\_j  
\]

---

### **Characteristics**

* Equivalent to ordinary PCA  
* No non-linear transformation

Used when:

Data already linear

---

# **2\. Polynomial Kernel**

Formula:

\[  
K(x\_i,x\_j)=(x\_i^Tx\_j+c)^d  
\]

Where:

* (d) \= Degree

---

### **Example**

Degree \= 2

Creates quadratic relationships

---

### **Intuition**

Can separate curved boundaries.

---

# **3\. Gaussian RBF Kernel (Most Important)**

Formula:

\[  
K(x\_i,x\_j)=e^{-\\gamma||x\_i-x\_j||^2}  
\]

---

### **Intuition**

Measures similarity based on distance.

If points are close:

Similarity ≈ 1

If far apart:

Similarity ≈ 0

---

### **Why Popular?**

Can model extremely complex non-linear patterns.

Most commonly used Kernel PCA kernel.

---

# **4\. Sigmoid Kernel**

Formula:

\[  
K(x\_i,x\_j)=tanh(ax\_i^Tx\_j+c)  
\]

Inspired by neural networks.

Less commonly used.

---

# **Visual Intuition**

Original Space:

Concentric Circles

○ ○ ○ ○

  ● ●

○ ○ ○ ○

Not linearly separable.

---

After Kernel PCA:

● ● ● ●

\-----------

○ ○ ○ ○

Now separation becomes easy.

---

# **PCA vs Kernel PCA**

| Feature | PCA | Kernel PCA |
| ----- | ----- | ----- |
| Linear Data | Excellent | Excellent |
| Non-linear Data | Poor | Excellent |
| Complexity | Lower | Higher |
| Kernel Needed | No | Yes |
| Computation | Fast | Slower |
| Interpretability | Higher | Lower |

---

# **Advantages of Kernel PCA**

### **Captures Non-linear Structure**

Major advantage.

---

### **Better Feature Extraction**

For complex datasets.

---

### **Works with Arbitrary Kernels**

Highly flexible.

---

### **Powerful Visualization Tool**

Useful for reducing complex data to 2D or 3D.

---

# **Disadvantages of Kernel PCA**

### **Computationally Expensive**

Requires kernel matrix computation.

---

### **Kernel Selection Required**

Wrong kernel can hurt performance.

---

### **Less Interpretable**

Features lose physical meaning.

---

### **Poor Scalability**

Large datasets become expensive.

---

# **Applications of Kernel PCA**

### **Image Recognition**

Extract non-linear image features.

---

### **Face Recognition**

Improve class separability.

---

### **Bioinformatics**

Gene expression analysis.

---

### **Anomaly Detection**

Detect unusual patterns.

---

### **Financial Modeling**

Capture non-linear market relationships.

---

# **PCA vs LDA vs Kernel PCA**

| Feature | PCA | LDA | Kernel PCA |
| ----- | ----- | ----- | ----- |
| Supervised | No | Yes | No |
| Linear | Yes | Yes | No |
| Uses Labels | No | Yes | No |
| Goal | Max Variance | Max Class Separation | Non-linear Variance |
| Non-linear Data | Poor | Poor | Excellent |

---

# **Python Example**

from sklearn.decomposition import KernelPCA

kpca \= KernelPCA(  
    n\_components=2,  
    kernel='rbf',  
    gamma=15  
)

X\_kpca \= kpca.fit\_transform(X)

---

# **Interview Summary**

**Kernel PCA (Kernel Principal Component Analysis)** is a non-linear dimensionality reduction technique that extends PCA using the **Kernel Trick**. Instead of performing PCA directly in the original feature space, it implicitly maps data into a higher-dimensional space where non-linear structures become linearly separable. It then performs PCA in that transformed space using a **Kernel Matrix**. Common kernels include the **Linear Kernel**, **Polynomial Kernel**, **Gaussian RBF Kernel**, and **Sigmoid Kernel**, with the **RBF Kernel** being the most widely used. Kernel PCA is especially useful for datasets with complex non-linear patterns where traditional PCA fails, making it valuable in image processing, pattern recognition, bioinformatics, and anomaly detection.

# 43

# **K-Fold Cross Validation**

K-Fold Cross Validation is a model evaluation technique used to estimate how well a Machine Learning model will perform on unseen data.

Instead of using a single train-test split, the dataset is divided into multiple parts and the model is trained and tested multiple times.

---

# **Why Do We Need Cross Validation?**

Suppose we split data:

80% Training  
20% Testing

Problem:

The result depends heavily on which samples ended up in the test set.

Example:

Accuracy \= 95%

Another random split:

Accuracy \= 87%

Performance becomes unstable.

Cross Validation provides a more reliable estimate.

---

# **Intuition**

Suppose we have:

100 Samples

Instead of testing once:

Train Once  
Test Once

we test multiple times using different portions of the data.

This gives a better estimate of real-world performance.

---

# **What is K in K-Fold?**

K represents:

Number of Folds

Example:

K \= 5

Dataset divided into:

Fold 1  
Fold 2  
Fold 3  
Fold 4  
Fold 5

---

# **K-Fold Cross Validation Process**

For:

K \= 5

---

### **Iteration 1**

Test  → Fold 1  
Train → Fold 2,3,4,5

---

### **Iteration 2**

Test  → Fold 2  
Train → Fold 1,3,4,5

---

### **Iteration 3**

Test  → Fold 3  
Train → Fold 1,2,4,5

---

### **Iteration 4**

Test  → Fold 4  
Train → Fold 1,2,3,5

---

### **Iteration 5**

Test  → Fold 5  
Train → Fold 1,2,3,4

---

# **Final Accuracy**

Suppose:

| Fold | Accuracy |
| ----- | ----- |
| 1 | 91% |
| 2 | 89% |
| 3 | 93% |
| 4 | 90% |
| 5 | 92% |

Final Score:

\[  
\\frac{91+89+93+90+92}{5}  
\]

\= 91%

This average is considered the model's performance.

---

# **Why K-Fold is Better**

Every observation:

Used for Training  
Used for Testing

at least once.

Result:

* More reliable  
* Less variance  
* Better model evaluation

---

# **Common Values of K**

| K | Usage |
| ----- | ----- |
| 5 | Most Common |
| 10 | Very Common |
| n | Leave-One-Out CV |

---

# **Leave-One-Out Cross Validation (LOOCV)**

Special case:

K \= Number of Samples

Example:

100 Samples  
→ 100 Folds

Train on:

99 Samples

Test on:

1 Sample

Repeat 100 times.

Very accurate but computationally expensive.

---

# **Stratified K-Fold**

Used for classification.

Maintains class distribution in every fold.

Example:

Dataset:

90% Class A  
10% Class B

Each fold preserves:

90% A  
10% B

Preferred for imbalanced datasets.

---

# **Advantages of K-Fold Cross Validation**

* Better performance estimation  
* Reduces overfitting risk during evaluation  
* Uses data efficiently  
* More reliable than a single train-test split

---

# **Disadvantages**

* Slower than train-test split  
* Computationally expensive for large datasets  
* Training repeated K times

---

# **Python Example**

from sklearn.model\_selection import cross\_val\_score  
from sklearn.ensemble import RandomForestClassifier

scores \= cross\_val\_score(  
    estimator=model,  
    X=X,  
    y=y,  
    cv=10  
)

print(scores.mean())

---

# **Bias-Variance Tradeoff**

The Bias-Variance Tradeoff is one of the most important concepts in Machine Learning.

It explains:

Why models underfit or overfit and how to find the right balance.

---

# **Intuition**

Suppose you are learning archery.

### **High Bias**

Always shoot left of the target.

Consistently wrong.

---

### **High Variance**

Shots scattered everywhere.

Sometimes correct, sometimes very wrong.

---

### **Balanced Model**

Shots clustered near the center.

Accurate and consistent.

---

# **What is Bias?**

Bias is the error caused by overly simplistic assumptions.

High Bias means:

Model too simple

Cannot learn underlying patterns.

---

# **Example**

Using Linear Regression for a highly curved dataset.

Actual Relationship  
      ∩

Linear Model  
      /

Poor fit.

---

# **Characteristics of High Bias**

* Underfitting  
* Low training accuracy  
* Low test accuracy  
* Model too simple

---

# **Examples**

High Bias Models:

* Linear Regression (for complex data)  
* Very shallow Decision Trees

---

# **What is Variance?**

Variance measures how sensitive the model is to training data.

High Variance means:

Model too complex

Memorizes training data.

---

# **Example**

Decision Tree:

One leaf per sample

Training Accuracy:

100%

Testing Accuracy:

70%

Overfitting occurs.

---

# **Characteristics of High Variance**

* Overfitting  
* High training accuracy  
* Low testing accuracy  
* Sensitive to data changes

---

# **Examples**

High Variance Models:

* Deep Decision Trees  
* Large Neural Networks  
* High-degree Polynomial Regression

---

# **Bias vs Variance**

| Property | High Bias | High Variance |
| ----- | ----- | ----- |
| Complexity | Low | High |
| Training Error | High | Low |
| Test Error | High | High |
| Underfitting | Yes | No |
| Overfitting | No | Yes |

---

# **The Tradeoff**

As model complexity increases:

Bias ↓  
Variance ↑

Simple Models:

High Bias  
Low Variance

Complex Models:

Low Bias  
High Variance

Goal:

Find the Sweet Spot

---

# **Visualization**

Test Error  
     |  
High |\\  
     | \\  
     |  \\  
     |   \\\_\_  
     |      \\\_  
     |  
     \+----------------  
       Model Complexity

The minimum point represents the optimal balance.

---

# **How to Reduce High Bias**

### **Use More Complex Models**

Example:

Linear Regression  
→ Polynomial Regression

---

### **Add More Features**

Provide more information.

---

### **Reduce Regularization**

Allow model to learn more.

---

# **How to Reduce High Variance**

### **More Training Data**

Most effective solution.

---

### **Feature Selection**

Remove irrelevant features.

---

### **Regularization**

Examples:

* Lasso (L1)  
* Ridge (L2)

---

### **Ensemble Methods**

Examples:

* Random Forest  
* Bagging

---

# **Bias-Variance Decomposition**

The prediction error can be viewed as:

Total Error \=  
Bias² \+ Variance \+ Irreducible Error

Irreducible error is noise in the data that cannot be eliminated.

---

# **Interview Summary**

**K-Fold Cross Validation** is a model evaluation technique where the dataset is divided into K folds. The model is trained K times, each time using one fold for testing and the remaining folds for training. The final performance is the average across all folds, providing a more reliable estimate than a single train-test split.

The **Bias-Variance Tradeoff** describes the balance between underfitting and overfitting. **Bias** measures errors due to overly simple assumptions and leads to underfitting, while **Variance** measures sensitivity to training data and leads to overfitting. As model complexity increases, bias decreases but variance increases. The goal is to find an optimal balance that minimizes overall test error and achieves good generalization.

# 44

# **XGBoost (Extreme Gradient Boosting)**

XGBoost stands for:

Extreme Gradient Boosting

It is one of the most powerful and widely used Machine Learning algorithms for:

* Classification  
* Regression  
* Ranking Problems

It is an advanced implementation of **Gradient Boosting** designed for:

Higher Accuracy  
Faster Training  
Better Generalization

Many Kaggle competition winners use XGBoost.

---

# **Intuition Behind XGBoost**

Suppose a student scores:

Math \= 40/100

A teacher identifies mistakes and gives feedback.

Student studies those mistakes and improves to:

60/100

Again mistakes are identified.

Student improves further:

75/100

Then:

85/100

Each new attempt focuses on correcting previous mistakes.

This is exactly how XGBoost works.

---

# **Ensemble Learning**

XGBoost is an **Ensemble Learning Algorithm**.

Instead of one model:

One Weak Learner

it combines:

Many Weak Learners

to create a strong model.

---

# **What is a Weak Learner?**

Usually:

Small Decision Tree

Example:

Tree 1  
Tree 2  
Tree 3  
Tree 4  
...

Each tree is individually weak.

Together they become very powerful.

---

# **Boosting**

Boosting means:

Build models sequentially so that each new model corrects errors made by previous models.

Unlike Random Forest:

Trees built independently

XGBoost:

Trees built sequentially

---

# **Gradient Boosting Intuition**

Suppose actual house prices:

| House | Actual |
| ----- | ----- |
| A | 100 |
| B | 200 |
| C | 300 |

---

## **First Tree Prediction**

| House | Predicted |
| ----- | ----- |
| A | 120 |
| B | 180 |
| C | 250 |

Errors:

| House | Residual |
| ----- | ----- |
| A | \-20 |
| B | 20 |
| C | 50 |

---

## **Second Tree**

Instead of learning original prices:

Learn Residuals

Tree 2 predicts:

\-15  
15  
40

---

## **Updated Prediction**

Prediction \=  
Tree1 \+ Tree2

Error decreases.

---

## **Third Tree**

Again learns remaining errors.

Process repeats.

---

# **Core Idea of XGBoost**

Each new tree learns:

Residual Errors

from previous trees.

Final Prediction:

\[  
\\hat y \= Tree\_1 \+ Tree\_2 \+ Tree\_3 \+ ...  
\]

---

# **Why Is It Called Gradient Boosting?**

Because error reduction is based on:

Gradient Descent

The algorithm minimizes loss using gradients.

---

# **Objective Function of XGBoost**

XGBoost optimizes:

\[  
Objective \= Loss \+ Regularization  
\]

---

## **Loss Function**

Measures prediction error.

Examples:

### **Regression**

Mean Squared Error

### **Classification**

Log Loss

---

## **Regularization Term**

Penalizes complex trees.

Purpose:

Reduce Overfitting

This is one major improvement over traditional Gradient Boosting.

---

# **How XGBoost Increases Accuracy**

This is the most important interview question.

---

# **1\. Sequential Error Correction**

Each tree learns:

Mistakes of Previous Trees

Error continuously decreases.

---

# **2\. Gradient Optimization**

Uses:

First-Order Derivatives

and

Second-Order Derivatives

(Hessian Information)

Traditional Gradient Boosting often uses only first derivatives.

XGBoost uses both.

Result:

More Precise Optimization

---

# **3\. Regularization**

Objective:

\[  
Loss \+ Regularization  
\]

Regularization prevents:

Overfitting

leading to better test accuracy.

---

# **4\. Shrinkage (Learning Rate)**

Instead of fully trusting every tree:

Tree Contribution × Learning Rate

Formula:

# **\[**

# **Prediction\_{new}**

Prediction\_{old}  
\+  
\\eta(Tree)  
\]

where:

* (\\eta) \= learning rate

Small updates improve generalization.

---

# **5\. Tree Pruning**

Normal Decision Trees:

Grow Fully  
Then Prune

XGBoost:

Smart Pruning During Training

Unnecessary branches are removed.

Result:

Less Overfitting

---

# **6\. Handling Missing Values**

XGBoost automatically learns:

Best Direction  
for Missing Data

No heavy preprocessing required.

---

# **7\. Feature Importance**

XGBoost automatically identifies:

Important Features

and focuses on them.

---

# **8\. Parallel Processing**

Unlike classical Gradient Boosting:

Sequential \+ Slow

XGBoost performs parts of tree construction in parallel.

Result:

Faster Training

without sacrificing accuracy.

---

# **XGBoost Algorithm**

### **Step 1**

Train first tree.

---

### **Step 2**

Calculate residual errors.

Actual \- Predicted

---

### **Step 3**

Train second tree on residuals.

---

### **Step 4**

Update prediction.

# **\[**

# **New Prediction**

Old Prediction  
\+  
Learning Rate × Tree  
\]

---

### **Step 5**

Repeat until convergence.

---

# **Important Hyperparameters**

---

## **n\_estimators**

Number of trees.

Example:

n\_estimators=500

More trees:

✔ Better learning

✖ More computation

---

## **learning\_rate**

Controls contribution of each tree.

Example:

learning\_rate=0.01

Small value:

✔ Better generalization

✖ Longer training

---

## **max\_depth**

Maximum depth of trees.

Controls complexity.

---

## **subsample**

Fraction of data used per tree.

Helps reduce overfitting.

---

## **colsample\_bytree**

Fraction of features used per tree.

Adds randomness.

---

# **XGBoost vs Random Forest**

| Feature | Random Forest | XGBoost |
| ----- | ----- | ----- |
| Tree Building | Parallel | Sequential |
| Error Correction | No | Yes |
| Boosting | No | Yes |
| Bagging | Yes | No |
| Accuracy | High | Usually Higher |
| Overfitting Control | Moderate | Strong |
| Training Speed | Faster | Slower |

---

# **XGBoost vs Gradient Boosting**

| Feature | Gradient Boosting | XGBoost |
| ----- | ----- | ----- |
| Regularization | No/Minimal | Yes |
| Second Derivatives | No | Yes |
| Missing Values | Limited | Native Support |
| Speed | Slower | Faster |
| Accuracy | High | Higher |

---

# **Advantages of XGBoost**

### **Very High Accuracy**

One of the strongest tabular-data algorithms.

---

### **Handles Missing Values**

Automatically.

---

### **Built-in Regularization**

Reduces overfitting.

---

### **Fast Training**

Optimized implementation.

---

### **Feature Importance**

Easy interpretation.

---

# **Disadvantages**

### **Many Hyperparameters**

Needs tuning.

---

### **Computationally Expensive**

For very large datasets.

---

### **Less Interpretable**

Compared to a single Decision Tree.

---

# **Applications**

### **Credit Risk Scoring**

Banks heavily use XGBoost.

---

### **Fraud Detection**

Credit card fraud.

---

### **Customer Churn Prediction**

Telecom and SaaS.

---

### **Recommendation Systems**

E-commerce.

---

### **Healthcare**

Disease prediction.

---

### **Kaggle Competitions**

One of the most successful algorithms.

---

# **Interview Summary**

**XGBoost (Extreme Gradient Boosting)** is an advanced ensemble learning algorithm based on Gradient Boosting. It builds multiple decision trees sequentially, where each new tree learns the residual errors made by previous trees. XGBoost improves accuracy through **gradient-based optimization, second-order derivatives (Hessians), regularization, shrinkage (learning rate), tree pruning, feature subsampling, and handling missing values**. Its objective function combines prediction loss with regularization, helping achieve both high accuracy and strong generalization. Compared to Random Forest and traditional Gradient Boosting, XGBoost usually provides superior performance on structured/tabular datasets and is one of the most widely used algorithms in industry and machine learning competitions.

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

# 46

# **Artificial Neural Networks (ANN)**

Artificial Neural Networks (ANNs) are Machine Learning models inspired by the human brain. They consist of interconnected **neurons** that learn patterns from data by adjusting connection weights.

They are the foundation of **Deep Learning** and power applications such as image recognition, speech recognition, NLP (ChatGPT), recommendation systems, and autonomous vehicles.

---

# **Intuition Behind Neural Networks**

Imagine identifying whether an image is a **cat or a dog**.

Instead of manually writing rules like:

If ears are pointed and whiskers exist → Cat

the neural network automatically learns these patterns from thousands of training examples.

It learns:

Pixels  
   ↓  
Edges  
   ↓  
Shapes  
   ↓  
Eyes  
   ↓  
Face  
   ↓  
Cat / Dog

---

# **Biological vs Artificial Neuron**

Biological neuron:

Dendrites → Receive Signals

Cell Body → Process Signals

Axon → Send Signal

Artificial neuron:

Inputs  
   ↓  
Weighted Sum  
   ↓  
Activation Function  
   ↓  
Output

---

# **The Artificial Neuron**

Suppose we have three inputs:

x1 \= Age

x2 \= Salary

x3 \= Experience

Each input has a weight:

w1

w2

w3

The neuron computes:

\[  
z=w\_1x\_1+w\_2x\_2+w\_3x\_3+b  
\]

where:

* **Weights (w)** determine the importance of each feature.  
* **Bias (b)** shifts the output, giving the neuron flexibility.

---

# **Why Do We Need Weights?**

Weights indicate how important each feature is.

Example:

Salary → Very Important

Weight \= 0.9

Age → Less Important

Weight \= 0.2

During training, the network automatically learns the optimal weights.

---

# **Bias**

Bias allows the neuron to make predictions even when all inputs are zero.

Without bias:

Decision boundary always passes through origin.

With bias:

Decision boundary can shift anywhere.

---

# **Activation Function**

After computing the weighted sum, an activation function is applied.

Purpose:

* Introduce **non-linearity**  
* Enable learning of complex patterns

Without activation functions:

Multiple layers  
\=  
One Linear Equation

The network would be no more powerful than linear regression.

---

# **Common Activation Functions**

---

## **1\. Sigmoid**

Maps output between **0 and 1**.

\[  
\\sigma(x)=\\frac{1}{1+e^{-x}}  
\]

Range:

0 to 1

Uses:

* Binary Classification Output Layer

Advantages:

* Produces probabilities

Disadvantages:

* Vanishing Gradient Problem  
* Slower convergence

---

## **2\. Tanh**

Maps values between **−1 and 1**.

\[  
tanh(x)=\\frac{e^x-e^{-x}}{e^x+e^{-x}}  
\]

Advantages:

* Zero-centered  
* Better than Sigmoid for hidden layers

Disadvantages:

* Still suffers from vanishing gradients

---

## **3\. ReLU (Rectified Linear Unit)**

Most commonly used activation.

\[  
f(x)=\\max(0,x)  
\]

Behavior:

x \> 0 → x

x ≤ 0 → 0

Example:

Input:

\[-3, 5, \-2, 7\]

Output:

\[0, 5, 0, 7\]

Advantages:

* Fast computation  
* Reduces vanishing gradients  
* Sparse activation  
* Used in most hidden layers

---

## **4\. Softmax**

Used for **multi-class classification**.

Converts outputs into probabilities that sum to 1\.

Example:

Input Scores:  
\[3.2, 1.1, 0.4\]

Output Probabilities:  
\[0.82, 0.13, 0.05\]

Prediction:

Class 1

---

# **Neural Network Architecture**

A typical neural network consists of:

Input Layer  
      ↓  
Hidden Layer(s)  
      ↓  
Output Layer

Example:

Age  
Salary  
Experience  
      ↓  
Hidden Layer  
      ↓  
Buy / Don't Buy

---

# **Types of Layers**

### **Input Layer**

Receives the input features.

Example:

Age  
Salary  
Experience

---

### **Hidden Layers**

Extract patterns from the data.

First hidden layer may learn:

Income Level

Second hidden layer may learn:

Buying Power

These features are learned automatically.

---

### **Output Layer**

Produces the final prediction.

Examples:

0 / 1

or

Cat / Dog / Bird

---

# **How Neural Networks Work (Forward Propagation)**

Forward propagation is the prediction phase.

Steps:

1. Input data enters the network.  
2. Each neuron computes a weighted sum.  
3. Activation functions are applied.  
4. Output is produced.

Flow:

Input  
   ↓  
Weighted Sum  
   ↓  
Activation  
   ↓  
Output

---

# **Loss Function**

The network compares:

Predicted Output

vs

Actual Output

Example:

Predicted \= 0.75

Actual \= 1

Difference between them is called **Loss (Error)**.

Common loss functions:

* Mean Squared Error (Regression)  
* Cross Entropy Loss (Classification)

---

# **How Neural Networks Learn?**

Neural networks learn by continuously adjusting their weights to reduce the loss.

Learning cycle:

Input  
   ↓  
Prediction  
   ↓  
Calculate Error  
   ↓  
Adjust Weights  
   ↓  
Better Prediction

This process is repeated for many iterations (epochs).

---

# **Gradient Descent**

Gradient Descent is the optimization algorithm used to minimize the loss.

### **Intuition**

Imagine standing on top of a mountain.

Goal:

Reach the lowest point.

You repeatedly move in the direction of the steepest downward slope.

Similarly, Gradient Descent updates the weights to move toward the minimum loss.

Weight update rule:

\[  
w=w-\\eta\\frac{\\partial L}{\\partial w}  
\]

Where:

* (w) \= Weight  
* (\\eta) \= Learning Rate  
* (L) \= Loss Function

---

# **Learning Rate**

The learning rate controls how large each weight update is.

### **Very Small Learning Rate**

Slow learning

Many iterations required

---

### **Very Large Learning Rate**

May overshoot the minimum

Training becomes unstable

---

### **Good Learning Rate**

Fast convergence

Stable training

---

# **Types of Gradient Descent**

---

## **1\. Batch Gradient Descent**

Uses the **entire training dataset** before updating weights.

Advantages:

* Stable updates

Disadvantages:

* Slow for large datasets

---

## **2\. Stochastic Gradient Descent (SGD)**

Updates weights after **every single training example**.

Process:

One Sample  
     ↓  
Compute Error  
     ↓  
Update Weights

Advantages:

* Faster  
* Uses less memory  
* Can escape local minima

Disadvantages:

* Noisy updates  
* Less stable

---

## **3\. Mini-Batch Gradient Descent**

Uses small batches (e.g., 32 or 64 samples).

Advantages:

* Faster than Batch GD  
* More stable than SGD

Most modern deep learning models use **Mini-Batch Gradient Descent**.

---

# **Backpropagation**

Backpropagation is the algorithm that enables neural networks to learn.

Purpose:

Determine how much each weight contributed to the prediction error and update it accordingly.

---

# **Intuition Behind Backpropagation**

Suppose:

Actual \= 1

Predicted \= 0.7

Error:

0.3

Backpropagation asks:

Which weights caused this error?

How should each weight change?

Using calculus (the chain rule), it computes the gradient of the loss with respect to every weight.

---

# **Steps of Backpropagation**

### **Step 1: Forward Propagation**

Generate prediction.

Input  
   ↓  
Prediction

---

### **Step 2: Compute Loss**

Compare:

Predicted

vs

Actual

---

### **Step 3: Compute Gradients**

Calculate how much each weight contributed to the error.

---

### **Step 4: Propagate Error Backwards**

Output Layer  
      ↑  
Hidden Layer  
      ↑  
Input Layer

Error flows backward through the network.

---

### **Step 5: Update Weights**

Gradient Descent updates weights:

\[  
w=w-\\eta\\frac{\\partial L}{\\partial w}  
\]

---

### **Step 6: Repeat**

Repeat for many epochs until the loss becomes very small.

---

# **Complete Neural Network Learning Cycle**

Input Data  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Gradient Descent  
      ↓  
Weight Update  
      ↓  
Repeat

---

# **Deep Learning**

When a neural network contains **multiple hidden layers**, it is called a **Deep Neural Network (DNN)**.

Example:

Input  
   ↓  
Hidden Layer 1  
   ↓  
Hidden Layer 2  
   ↓  
Hidden Layer 3  
   ↓  
Output

Each layer learns increasingly abstract features.

---

# **Applications of Neural Networks**

* Image Recognition  
* Speech Recognition  
* Natural Language Processing (ChatGPT, Translation)  
* Recommendation Systems  
* Fraud Detection  
* Medical Diagnosis  
* Autonomous Vehicles

---

# **Interview Summary**

An **Artificial Neural Network (ANN)** is a machine learning model inspired by the human brain, consisting of interconnected **neurons** organized into **input, hidden, and output layers**. Each neuron computes a weighted sum of its inputs, adds a bias, and applies an **activation function** such as **Sigmoid, Tanh, ReLU, or Softmax**. During **forward propagation**, the network generates predictions, which are compared to the true values using a **loss function**. The **backpropagation algorithm** computes the gradients of the loss with respect to every weight using the chain rule, and **Gradient Descent** (or **Stochastic/Mini-Batch Gradient Descent**) updates the weights to minimize the loss. Repeating this process over many epochs allows the network to learn increasingly accurate representations, making neural networks the foundation of modern deep learning applications.

# encoder decoder

# **Encoder-Decoder Transformer Architecture**

The **Encoder-Decoder Transformer** is the original Transformer architecture introduced in the paper:

**"Attention Is All You Need" (2017)**

It is designed for **sequence-to-sequence (Seq2Seq)** tasks, where an input sequence is transformed into an output sequence.

Examples:

* Machine Translation  
* Text Summarization  
* Question Answering  
* Speech Recognition

Unlike RNNs and LSTMs, Transformers process the entire sequence in parallel using **Self-Attention**, making them much faster and better at capturing long-range dependencies.

---

# **Why Was the Transformer Introduced?**

Earlier Seq2Seq models used:

RNN

↓

LSTM

↓

Encoder-Decoder

Problems:

* Sequential processing (slow)  
* Vanishing gradients  
* Difficulty remembering long sequences

Transformers replace recurrence with:

Self-Attention

allowing all words to interact simultaneously.

---

# **High-Level Architecture**

Input Sentence  
      ↓  
Input Embedding  
      ↓  
Positional Encoding  
      ↓  
Encoder Stack  
      ↓  
Context Representation  
      ↓  
Decoder Stack  
      ↓  
Linear Layer  
      ↓  
Softmax  
      ↓  
Output Sentence

The architecture consists of:

* Encoder  
* Decoder

---

# **Overall Transformer Architecture**

               INPUT

                   │

         Token Embeddings

                   │

       Positional Encoding

                   │

          ┌───────────────┐  
          │    Encoder    │  
          └───────────────┘  
                   │  
         Context Embeddings  
                   │  
          ┌───────────────┐  
          │    Decoder    │  
          └───────────────┘  
                   │  
             Linear Layer  
                   │  
               Softmax  
                   │  
               Prediction

---

# **Input Embeddings**

Words are first converted into vectors.

Example:

Sentence:

I love AI

↓

Embeddings

I

↓

\[0.2,0.4,...\]

Each word becomes a dense numerical vector.

---

# **Positional Encoding**

Unlike RNNs,

Transformers have **no notion of word order**.

Therefore,

positional information is added.

Example:

Word

\+

Position

↓

Final Embedding

Example:

I

Position 1

Love

Position 2

AI

Position 3

This allows the model to distinguish:

Dog bites man

≠

Man bites dog

---

# **Encoder**

The encoder reads the entire input sentence and produces contextual representations.

It consists of multiple identical layers.

Each encoder layer contains:

Multi-Head Self Attention

↓

Add & Normalize

↓

Feed Forward Network

↓

Add & Normalize

Usually:

* 6 encoder layers (original paper)  
* 12 or more in modern models

---

# **Step 1: Multi-Head Self-Attention**

Each word attends to every other word.

Example:

Sentence:

The animal didn't cross the road because it was tired.

The word:

it

learns that it refers to:

animal

instead of:

road

This captures context.

---

# **Multi-Head Attention**

Instead of one attention mechanism,

multiple attention heads operate in parallel.

Example:

Head 1

↓

Grammar

Head 2

↓

Meaning

Head 3

↓

Long-distance Relation

Each head learns different relationships.

---

# **Feed Forward Network (FFN)**

Each token independently passes through a small neural network.

Purpose:

* Increase representation power  
* Learn nonlinear transformations

Pipeline:

Attention Output

↓

Linear

↓

ReLU/GELU

↓

Linear

---

# **Add & Normalize**

Residual (skip) connections are added:

Input

\+

Layer Output

Benefits:

* Stable gradients  
* Faster training  
* Enables deep Transformers

Layer Normalization stabilizes activations.

---

# **Encoder Output**

After all encoder layers,

every input word becomes a **context-aware embedding**.

Example:

Apple

In:

Apple released a new phone.

means the company.

In:

Apple is healthy.

means the fruit.

The encoder captures this difference.

---

# **Decoder**

The decoder generates the output sequence **one token at a time**.

Each decoder layer has:

Masked Multi-Head Self Attention

↓

Add & Normalize

↓

Encoder-Decoder Attention

↓

Add & Normalize

↓

Feed Forward Network

↓

Add & Normalize

---

# **1\. Masked Self-Attention**

The decoder cannot "see the future."

Example:

Generating:

I love AI

When predicting:

love

It cannot access:

AI

Masking ensures autoregressive generation.

---

# **2\. Encoder-Decoder Attention**

The decoder attends to the encoder outputs.

Example:

English:

I love AI

Encoder creates context.

Decoder attends to it while generating:

Je

↓

aime

↓

IA

(French)

This connects input and output sequences.

---

# **3\. Feed Forward Network**

Same as encoder.

Each token is processed independently.

---

# **Linear Layer**

Converts decoder output into vocabulary scores.

Example:

Vocabulary:

Cat

Dog

Car

Tree

Each word receives a score.

---

# **Softmax**

Softmax converts scores into probabilities.

Example:

Cat

0.05

Dog

0.85

Tree

0.10

Highest probability becomes the predicted token.

---

# **Training**

Training uses:

Teacher Forcing

The decoder receives the correct previous token during training.

Loss:

Cross Entropy Loss

Optimizer:

* Adam  
* AdamW

---

# **Inference**

During inference:

Step 1

Generate first word.

↓

Step 2

Feed generated word back into decoder.

↓

Generate next word.

↓

Repeat until:

\<END\>

token.

---

# **Encoder-Only vs Decoder-Only vs Encoder-Decoder**

| Architecture | Components | Example Models | Main Use |
| ----- | ----- | ----- | ----- |
| Encoder Only | Encoder | BERT, RoBERTa | Understanding, classification |
| Decoder Only | Decoder | GPT, LLaMA | Text generation |
| Encoder-Decoder | Both | T5, BART, Original Transformer | Translation, summarization, Seq2Seq |

---

# **Applications**

### **Machine Translation**

English → French

---

### **Text Summarization**

Long article → Short summary

---

### **Question Answering**

Question → Answer

---

### **Speech Recognition**

Speech → Text

---

### **Image Captioning**

Image → Sentence

---

### **Code Generation**

Description → Source Code

---

# **Advantages**

* Parallel processing  
* Captures long-range dependencies  
* Faster than RNNs/LSTMs  
* Highly scalable  
* State-of-the-art performance on Seq2Seq tasks

---

# **Limitations**

* High memory usage (attention is (O(n^2)) with sequence length)  
* Expensive for very long sequences  
* Requires large amounts of training data

---

# **Complete Transformer Pipeline**

Input Tokens  
      ↓  
Embeddings  
      ↓  
Positional Encoding  
      ↓  
Encoder Stack  
      ↓  
Context Representation  
      ↓  
Masked Self-Attention  
      ↓  
Encoder-Decoder Attention  
      ↓  
Feed Forward Network  
      ↓  
Linear Layer  
      ↓  
Softmax  
      ↓  
Output Tokens

---

# **Interview Summary**

The **Encoder-Decoder Transformer** is the original Transformer architecture introduced in **"Attention Is All You Need" (2017)** for **sequence-to-sequence (Seq2Seq)** tasks such as **machine translation, summarization, and question answering**. The **encoder** processes the entire input sequence using **multi-head self-attention**, **feed-forward networks**, **residual connections**, and **layer normalization** to produce context-aware embeddings. The **decoder** generates the output sequence one token at a time using **masked self-attention** (to prevent looking at future tokens), **encoder-decoder attention** (to attend to the encoder's output), and a **feed-forward network**, followed by a **linear layer** and **softmax** to predict the next token. **Positional encoding** is added to input embeddings because Transformers process tokens in parallel and do not inherently capture word order. Compared to RNNs and LSTMs, Transformers provide **parallel computation**, **better long-range dependency modeling**, and **higher scalability**. Modern Transformer models fall into three categories: **encoder-only models** (e.g., BERT) for understanding tasks, **decoder-only models** (e.g., GPT, LLaMA) for text generation, and **encoder-decoder models** (e.g., T5, BART) for sequence-to-sequence applications.

# Graph ml 1

# **Graph Neural Networks (GNN)**

A **Graph Neural Network (GNN)** is a Deep Learning model specifically designed to work with **graph-structured data**.

Unlike CNNs (for images) and RNNs (for sequences), GNNs learn from:

* Nodes (Vertices)  
* Edges (Connections)  
* Graph Structure

They capture both **feature information** and **relationships** between entities.

---

# **Why Do We Need GNNs?**

Many real-world datasets are naturally represented as graphs.

Examples:

Facebook

Users connected by friendships.

Google Maps

Cities connected by roads.

Molecules

Atoms connected by chemical bonds.

Bank Transactions

Accounts connected by money transfers.

Traditional ML treats every sample independently.

GNNs learn using:

Node Features  
\+  
Neighborhood Information  
\+  
Graph Structure

---

# **What is a Graph?**

A graph is represented as:

\[  
G=(V,E)  
\]

Where:

* **V** \= Set of Nodes (Vertices)  
* **E** \= Set of Edges (Connections)

Example:

A \----- B  
|       |  
|       |  
C \----- D

Nodes:

A B C D

Edges:

AB AC BD CD

---

# **Components of a Graph**

## **Node (Vertex)**

Represents an entity.

Examples:

* Person  
* Bank Account  
* Molecule Atom  
* Product

---

## **Edge**

Represents a relationship.

Examples:

* Friendship  
* Money Transfer  
* Chemical Bond  
* Road

---

## **Node Features**

Each node may contain attributes.

Example:

Person:

Age  
Salary  
City

---

## **Edge Features**

Each edge may contain information.

Example:

Bank Transfer:

Amount  
Date  
Currency

---

# **Types of Graphs**

---

# **1\. Directed Graph**

Edges have direction.

Example:

A → B

Meaning:

A follows B

Examples:

* Twitter  
* Citation Networks  
* Bank Transactions

---

# **2\. Undirected Graph**

Edges have no direction.

Example:

A \----- B

Examples:

* Facebook Friendship  
* Collaboration Networks

---

# **3\. Weighted Graph**

Edges have weights.

Example:

A \----5---- B

Weight may represent:

* Distance  
* Cost  
* Similarity  
* Transaction Amount

---

# **4\. Unweighted Graph**

Every edge has equal importance.

Example:

A \----- B

---

# **5\. Bipartite Graph**

Nodes belong to two different sets.

Connections occur only across sets.

Example:

Students \---- Courses

No:

Student → Student

Course → Course

Applications:

* Recommendation Systems  
* Job Matching

---

# **6\. Cyclic Graph**

Contains cycles.

Example:

A → B

↑   ↓

D ← C

---

# **7\. Acyclic Graph**

Contains no cycles.

Example:

Decision Trees.

---

# **Connectivity**

Connectivity determines whether nodes are reachable from one another.

---

## **Connected Graph**

Every node can reach every other node.

Example:

A \----- B  
|       |  
C \----- D

---

## **Disconnected Graph**

Contains isolated groups.

Example:

A \----- B

C \----- D

---

# **Connected Components**

A **Connected Component** is a group of nodes where every node is reachable from every other node.

Example:

A \---- B

C \---- D \---- E

F

Connected Components:

{A,B}

{C,D,E}

{F}

---

# **Adjacency Matrix**

A graph can be represented using an **Adjacency Matrix**.

Suppose:

A \---- B

A \---- C

B \---- C

Matrix:

|  | A | B | C |
| ----- | ----- | ----- | ----- |
| A | 0 | 1 | 1 |
| B | 1 | 0 | 1 |
| C | 1 | 1 | 0 |

Meaning:

1 → Edge Exists

0 → No Edge

---

# **Adjacency List**

Alternative representation.

Example:

A → B,C

B → A,C

C → A,B

Memory efficient for sparse graphs.

---

# **Intuition Behind GNN**

Suppose we want to predict whether a user is interested in a product.

Instead of only using:

User Age

Salary

GNN also considers:

Friends

Purchases

Interactions

The prediction depends on both the node and its neighbors.

---

# **How GNN Works**

The key idea is:

A node learns from its neighbors.

This process is called **Message Passing** or **Neighborhood Aggregation**.

---

# **Message Passing**

Suppose:

     B  
      |  
A \---- C \---- D

To update node **C**:

Step 1:

Receive information from:

A

B

D

Step 2:

Aggregate neighbor information.

Step 3:

Combine with C's own features.

Step 4:

Generate updated representation.

---

# **Graph Representation (Embeddings)**

After several message-passing layers, each node is represented by a dense vector called a **Node Embedding**.

Similar nodes have similar embeddings.

---

# **Types of GNN Prediction Tasks**

---

# **1\. Node-Level Prediction**

Predict properties of individual nodes.

Example:

Facebook:

Predict User Interests

Fraud Detection:

Fraudulent Bank Account?

Examples:

* Node Classification  
* Node Regression

---

# **2\. Edge-Level Prediction**

Predict whether an edge exists or estimate an edge property.

Examples:

Friend Recommendation:

Will A connect to B?

Knowledge Graph Completion.

Drug Interaction Prediction.

---

# **3\. Graph-Level Prediction**

Predict a label for the entire graph.

Example:

Molecule:

Will this molecule cure cancer?

Protein Classification.

Document Graph Classification.

---

# **Types of Graph Representations (Embeddings)**

---

## **Node Embedding**

Each node gets its own vector.

Used for:

* Node Classification  
* Clustering

---

## **Edge Embedding**

Each relationship gets a vector.

Used for:

* Link Prediction  
* Recommendation

---

## **Graph Embedding**

Entire graph represented as one vector.

Used for:

* Graph Classification  
* Molecule Prediction

---

# **Common GNN Architectures**

---

## **Graph Convolutional Network (GCN)**

Most popular GNN.

Idea:

Similar to CNN convolution.

Instead of neighboring pixels:

Neighboring Nodes

are aggregated.

---

## **GraphSAGE**

Learns by sampling neighbors.

Useful for:

Large graphs.

---

## **Graph Attention Network (GAT)**

Uses attention mechanism.

Important neighbors receive higher weights.

---

## **Graph Isomorphism Network (GIN)**

Powerful for graph classification.

---

# **Applications of GNN**

---

## **Social Networks**

* Friend Recommendation  
* Community Detection  
* User Classification

---

## **Recommendation Systems**

Amazon:

User → Product

Netflix:

Movie Recommendation.

---

## **Fraud Detection**

Banks detect suspicious transaction networks.

---

## **Drug Discovery**

Predict molecular properties.

---

## **Knowledge Graphs**

Google Search.

Question Answering.

---

## **Traffic Networks**

Road prediction.

Navigation.

---

## **Cybersecurity**

Attack graph analysis.

---

## **Supply Chain Networks**

Route optimization.

---

# **Advantages of GNN**

* Naturally models relationships  
* Captures graph structure  
* Works on irregular data  
* State-of-the-art for graph problems

---

# **Disadvantages**

* Computationally expensive for very large graphs  
* Training can be slow  
* Oversmoothing in deep GNNs (node representations become too similar)  
* Difficult to scale to billion-node graphs

---

# **CNN vs RNN vs GNN**

| Feature | CNN | RNN | GNN |
| ----- | ----- | ----- | ----- |
| Data Type | Images | Sequences | Graphs |
| Input Structure | Grid | Ordered Sequence | Nodes & Edges |
| Learns From | Nearby Pixels | Previous Time Steps | Neighboring Nodes |
| Examples | Image Classification | Language Translation | Social Networks |

---

# **GNN Workflow**

Graph Data  
(Nodes \+ Edges)  
        ↓  
Initialize Node Features  
        ↓  
Message Passing  
        ↓  
Neighbor Aggregation  
        ↓  
Update Node Embeddings  
        ↓  
Repeat for Multiple Layers  
        ↓  
Node / Edge / Graph Embeddings  
        ↓  
Prediction

---

# **Interview Summary**

A **Graph Neural Network (GNN)** is a deep learning model designed for **graph-structured data**, where information is represented as **nodes (entities)** and **edges (relationships)**. Unlike traditional neural networks, GNNs learn by **aggregating information from neighboring nodes** through a process called **message passing**, producing rich node, edge, or graph embeddings. Graphs can be **directed, undirected, weighted, unweighted, bipartite, cyclic, or acyclic**, and can be represented using **adjacency matrices** or **adjacency lists**. GNNs support **node-level tasks** (e.g., node classification), **edge-level tasks** (e.g., link prediction), and **graph-level tasks** (e.g., molecule classification). Popular GNN architectures include **GCN**, **GraphSAGE**, **Graph Attention Networks (GAT)**, and **Graph Isomorphism Networks (GIN)**. They are widely used in **social networks, fraud detection, recommendation systems, knowledge graphs, drug discovery, cybersecurity, and traffic network analysis**.

# graph ml 2

# **Node-Level, Edge-Level, and Graph-Level Features & Predictions in GNN**

One of the key strengths of **Graph Neural Networks (GNNs)** is that they can learn representations (embeddings) at **three different levels**:

1. **Node-Level**  
2. **Edge (Link)-Level**  
3. **Graph-Level**

Each level has its own **features**, **embeddings**, and **prediction tasks**.

---

# **Overview**

| Level | What is Learned? | Prediction Target |
| ----- | ----- | ----- |
| Node Level | Representation of each node | Predict node properties |
| Edge (Link) Level | Representation of relationships | Predict edge existence or attributes |
| Graph Level | Representation of the entire graph | Predict graph properties |

---

# **1\. Node-Level Features**

Node features describe the characteristics of an individual node.

Every node has its own feature vector.

Example:

### **Social Network**

Node \= User

Features:

Age

Gender

Location

Followers

Interests

---

### **Banking Network**

Node \= Bank Account

Features:

Account Age

Balance

Number of Transactions

Account Type

---

### **Molecular Graph**

Node \= Atom

Features:

Atomic Number

Charge

Hybridization

Atomic Mass

---

# **Node Embedding**

During message passing, each node learns information from its neighbors.

After training:

Node  
↓

Dense Vector (Embedding)

↓

Prediction

Example:

User A

↓

\[0.34, 1.25, \-0.81, ...\]

This vector captures:

* Node features  
* Neighbor information  
* Graph structure

---

# **Node-Level Prediction**

Goal:

Predict a property of an individual node.

---

## **Examples**

### **Social Networks**

Predict:

User Interests

---

### **Fraud Detection**

Predict:

Fraudulent Account?

---

### **Citation Networks**

Predict:

Research Paper Topic

---

### **Recommendation Systems**

Predict:

Customer Preference

---

# **Examples of Node-Level Tasks**

* Node Classification  
* Node Regression  
* Node Clustering

---

# **2\. Edge-Level (Link-Level) Features**

Edges also contain useful information.

Example:

### **Facebook**

Edge:

Friendship

Feature:

Friend Since \= 2018

---

### **Banking**

Edge:

Money Transfer

Features:

Amount

Date

Currency

Transaction Type

---

### **Transportation**

Edge:

Road

Features:

Distance

Travel Time

Traffic

---

# **Edge Embedding**

An edge can also have its own embedding.

Example:

(Account A)

↓

(Transaction)

↓

(Account B)

The embedding represents:

* Relationship strength  
* Interaction frequency  
* Similarity  
* Transaction history

---

# **Edge-Level Prediction**

Goal:

Predict information about relationships.

---

## **Examples**

### **Friend Recommendation**

Predict:

Will User A become friends with User B?

---

### **LinkedIn**

Predict:

Connection Recommendation

---

### **Drug Discovery**

Predict:

Will Drug A interact with Protein B?

---

### **Banking**

Predict:

Will Account A send money to Account B?

---

# **Examples of Edge-Level Tasks**

* Link Prediction  
* Edge Classification  
* Edge Regression

---

# **3\. Graph-Level Features**

Sometimes the entire graph represents one object.

Instead of learning individual nodes,

learn one embedding for the entire graph.

---

### **Example 1**

One molecule

Entire graph:

Atoms  
\+  
Chemical Bonds

↓

Predict:

Toxic or Non-Toxic

---

### **Example 2**

Social Community

Entire graph:

Users  
Connections

↓

Predict:

Community Type

---

### **Example 3**

Protein Structure

Entire graph:

Protein Network

↓

Predict:

Disease Related?

---

# **Graph Embedding**

All node embeddings are combined using:

Pooling

Readout

Aggregation

to create one vector.

Graph

↓

Graph Embedding

↓

Prediction

---

# **Graph-Level Prediction**

Goal:

Predict a property of the whole graph.

---

## **Examples**

### **Chemistry**

Predict:

Molecule Toxicity

---

### **Drug Discovery**

Predict:

Drug Effectiveness

---

### **Social Networks**

Predict:

Community Category

---

### **Cybersecurity**

Predict:

Network Under Attack?

---

# **Graph Readout (Pooling)**

To obtain graph-level embeddings,

node embeddings are aggregated.

Common methods:

---

## **Mean Pooling**

Take average.

Graph Embedding

\=

Average(Node Embeddings)

---

## **Sum Pooling**

Add all embeddings.

---

## **Max Pooling**

Take maximum value from each dimension.

---

## **Attention Pooling**

More important nodes receive higher weights.

---

# **Comparison of Prediction Types**

| Prediction Type | Input | Output | Example |
| ----- | ----- | ----- | ----- |
| Node Prediction | One Node | Node Label | Fraudulent Account? |
| Edge Prediction | Two Nodes \+ Edge | Edge Label | Friend Recommendation |
| Graph Prediction | Entire Graph | Graph Label | Molecule Toxicity |

---

# **Real-World Examples**

---

## **Node-Level**

### **Facebook**

Input:

User Node

Prediction:

Political Interest

---

### **Banking**

Input:

Account Node

Prediction:

Fraud Score

---

## **Edge-Level**

### **LinkedIn**

Input:

User A

User B

Prediction:

Should Connect?

---

### **Amazon**

Input:

Customer

Product

Prediction:

Will Purchase?

---

## **Graph-Level**

### **Chemistry**

Input:

Entire Molecule

Prediction:

Toxic?

---

### **Cybersecurity**

Input:

Computer Network

Prediction:

Malicious Network?

---

# **Feature Summary**

| Level | Example Entity | Features |
| ----- | ----- | ----- |
| Node | User | Age, Salary, Interests |
| Edge | Transaction | Amount, Date, Distance |
| Graph | Molecule | Entire Structure |

---

# **Embedding Summary**

| Embedding | Represents | Used For |
| ----- | ----- | ----- |
| Node Embedding | Individual Node | Node Classification |
| Edge Embedding | Relationship | Link Prediction |
| Graph Embedding | Entire Graph | Graph Classification |

---

# **Interview Summary**

Graph Neural Networks learn representations at **three levels**:

1. **Node-Level**, where each node has its own features (e.g., age, balance, atomic number) and the model predicts properties of individual nodes such as **fraud detection, user classification, or paper topic classification**.  
2. **Edge-Level (Link-Level)**, where edges represent relationships with features like **transaction amount, friendship duration, or road distance**. GNNs perform **link prediction, edge classification, and edge regression**, such as recommending friends or predicting future transactions.  
3. **Graph-Level**, where the entire graph represents a single object, such as a **molecule, protein, or social community**. Node embeddings are aggregated using **graph pooling/readout** techniques (mean, sum, max, or attention pooling) to produce a **graph embedding**, which is used for tasks like **molecule toxicity prediction, drug discovery, protein classification, and network anomaly detection**. These three levels of representation make GNNs highly versatile for learning from complex relational data.

# graph ml 3

# **Node Embeddings**

A **Node Embedding** is a low-dimensional dense vector representation of a node that captures:

* Node features  
* Graph structure  
* Neighborhood information  
* Connectivity patterns

Instead of representing a node as an ID or one-hot vector:

Node A \= \[0,0,1,0,0,0,...\]

it is represented as:

Node A \= \[0.42, \-0.31, 1.27, 0.88, ...\]

Nodes that are similar in the graph have similar embeddings.

---

# **Why Do We Need Node Embeddings?**

Machine learning algorithms cannot directly work with graph structures.

Graphs contain:

* Nodes  
* Edges  
* Neighborhoods

We convert every node into a fixed-length vector.

Then traditional ML or Deep Learning models can use these vectors.

---

# **Intuition**

Suppose we have a social network:

Alice \---- Bob \---- Charlie  
   |  
 David

Alice and David are friends.

Alice and Bob share similar neighborhoods.

Their embeddings should be close together.

Example:

Alice

↓

\[0.51, \-0.22, 1.4\]

Bob

↓

\[0.48, \-0.18, 1.3\]

Charlie may have a different embedding.

---

# **What Should a Good Node Embedding Capture?**

A good embedding captures:

### **Graph Structure**

Nearby nodes have similar vectors.

---

### **Node Attributes**

Age

Location

Interests

etc.

---

### **Connectivity**

Who is connected to whom.

---

### **Community Information**

Nodes in the same community have similar embeddings.

---

# **Types of Node Embedding Methods**

There are two major categories.

Node Embeddings  
      │  
      ├── Unsupervised  
      │  
      └── Supervised

---

# **1\. Unsupervised Node Embeddings**

No labels are required.

Goal:

Learn graph structure automatically.

---

Popular methods:

* DeepWalk  
* Node2Vec  
* LINE  
* Graph Autoencoders

---

# **Intuition**

Suppose we have Facebook.

Nobody tells the algorithm:

Person A likes sports.

Instead it learns:

People with similar friends  
↓

Probably similar.

---

# **Advantages**

* No labels needed.  
* Works on massive graphs.  
* Learns useful representations.

---

# **2\. Supervised Node Embeddings**

Uses node labels.

Example:

Fraud

Normal

The embedding is optimized for prediction.

---

Example:

Bank accounts.

Input:

Account Features

\+

Graph Structure

Output:

Fraud Probability

During training, embeddings become better for fraud detection.

---

# **Comparison**

| Feature | Unsupervised | Supervised |
| ----- | ----- | ----- |
| Labels Needed | No | Yes |
| Learns | Structure | Structure \+ Task |
| Examples | Node2Vec, DeepWalk | GCN, GraphSAGE, GAT |

---

# **Node Similarity**

The goal of node embeddings is:

Similar Nodes

↓

Similar Vectors

---

Suppose:

Doctor A

Doctor B

Both:

* Work at same hospital  
* Have same specialization  
* Collaborate often

Their embeddings should be close.

---

# **Measuring Node Similarity**

After obtaining embeddings:

A \= \[0.4, 0.8\]

B \= \[0.5, 0.7\]

Similarity can be measured using:

---

## **Cosine Similarity**

Most popular.

Measures angle between vectors.

Range:

\-1 to 1

Interpretation:

1

↓

Very Similar

---

## **Euclidean Distance**

Measures physical distance.

Smaller distance:

↓

More Similar

---

## **Dot Product**

Often used during training.

Higher value:

↓

More Similar

---

# **Random Walk**

Random Walk is one of the most important ideas behind node embeddings.

---

# **Intuition**

Imagine standing on a node.

At every step:

Choose one neighboring node randomly.

Example:

A \---- B \---- C  
      |  
      D

Random walk:

A

↓

B

↓

D

↓

B

↓

C

Produces a sequence:

A B D B C

This is similar to a sentence in NLP.

---

# **Why Random Walk?**

Words appearing together in a sentence are related.

Similarly,

nodes appearing together in random walks are related.

This idea allows graph problems to be solved using NLP techniques.

---

# **DeepWalk**

DeepWalk combines:

Random Walk

\+

Word2Vec

Steps:

1. Perform many random walks.  
2. Treat each walk as a sentence.  
3. Train Word2Vec.  
4. Obtain node embeddings.

---

# **Node2Vec**

Node2Vec improves DeepWalk by making random walks more flexible.

Instead of purely random movement,

it controls how the walk explores the graph.

---

# **Intuition**

Suppose:

A

|

B

/ \\

C  D

Node2Vec decides whether to:

Stay Nearby

or

Explore Far Away

---

It introduces two parameters:

### **p (Return Parameter)**

Controls probability of returning.

Large p:

Don't go back.

Small p:

Return frequently.

---

### **q (In-Out Parameter)**

Controls exploration.

Small q:

Explore far.

Large q:

Stay local.

---

# **BFS vs DFS Behavior**

Node2Vec can imitate both.

---

## **BFS-like Walk**

Focuses nearby nodes.

Learns:

Community Similarity

Example:

Friends in same group.

---

## **DFS-like Walk**

Explores distant nodes.

Learns:

Structural Similarity

Example:

Managers from different departments.

---

# **Encoder-Decoder Framework**

Most node embedding methods follow an Encoder-Decoder framework.

Graph

↓

Encoder

↓

Embedding

↓

Decoder

↓

Prediction

---

# **Encoder**

Converts node into embedding.

Example:

Node

↓

128-D Vector

Examples:

* GCN  
* GraphSAGE  
* Node2Vec

---

# **Decoder**

Uses embeddings to solve tasks.

Examples:

Node Classification

Link Prediction

Similarity

Graph Classification

---

# **Node Similarity Measure**

The decoder often predicts similarity using:

Dot Product

or

Cosine Similarity

If:

Embedding(A)

≈

Embedding(B)

Then:

Nodes are similar.

---

# **Extension to Graph Embedding**

Instead of embedding one node,

we can embed the entire graph.

---

Suppose a molecule:

Atoms

\+

Chemical Bonds

↓

One vector.

Graph

↓

\[0.31, \-1.4, 2.8, ...\]

---

# **Why Graph Embeddings?**

Sometimes:

Entire graph

\=

One data sample.

Examples:

* Molecule  
* Protein  
* Social Network  
* Knowledge Graph

---

# **How Are Graph Embeddings Created?**

Step 1

Compute node embeddings.

Node A

↓

Vector

---

Step 2

Aggregate all node embeddings.

Methods:

### **Mean Pooling**

Average.

---

### **Sum Pooling**

Add vectors.

---

### **Max Pooling**

Take maximum values.

---

### **Attention Pooling**

Important nodes receive larger weights.

---

Result:

Entire Graph

↓

Single Embedding

---

# **Applications of Graph Embeddings**

### **Molecule Classification**

Predict:

Cancer Drug?

---

### **Protein Classification**

Predict:

Protein Function

---

### **Social Networks**

Predict:

Community Type

---

### **Cybersecurity**

Predict:

Malicious Network?

---

# **Node Embedding vs Graph Embedding**

| Feature | Node Embedding | Graph Embedding |
| ----- | ----- | ----- |
| Represents | One Node | Entire Graph |
| Output | Node Vector | Graph Vector |
| Applications | Node Classification, Link Prediction | Molecule Classification, Graph Classification |

---

# **Workflow of Node2Vec**

Graph  
   ↓  
Random Walks  
   ↓  
Generate Node Sequences  
   ↓  
Word2Vec (Skip-Gram)  
   ↓  
Node Embeddings  
   ↓  
Similarity / Classification / Link Prediction

---

# **Applications of Node Embeddings**

### **Social Networks**

Friend recommendation.

---

### **Banking**

Fraud detection.

---

### **Recommendation Systems**

User-product similarity.

---

### **Knowledge Graphs**

Entity representation.

---

### **Biological Networks**

Gene prediction.

---

# **Interview Summary**

A **Node Embedding** is a dense, low-dimensional vector representation of a graph node that captures its **features, neighborhood information, and graph topology**. Node embeddings can be learned using **unsupervised methods** such as **DeepWalk** and **Node2Vec**, which generate random walks and apply **Word2Vec** to learn node representations, or **supervised methods** such as **GCN, GraphSAGE, and GAT**, where embeddings are optimized for tasks like node classification. **Node2Vec** extends DeepWalk by introducing biased random walks through the **return parameter (p)** and **in-out parameter (q)**, allowing exploration of both local communities (BFS-like behavior) and structurally similar nodes (DFS-like behavior). Most graph representation learning follows an **encoder-decoder framework**, where the encoder generates embeddings and the decoder performs tasks such as node classification, link prediction, or similarity estimation using measures like **cosine similarity, dot product, or Euclidean distance**. The same idea extends to **graph embeddings**, where node embeddings are aggregated using **mean, sum, max, or attention pooling** to obtain a single vector representing the entire graph for tasks such as **molecule classification, protein function prediction, and graph classification**.

# graph ml 4

# **PageRank**

**PageRank** is a graph algorithm originally developed by **Google founders Larry Page and Sergey Brin** to rank web pages.

The idea is:

A webpage is important if many important webpages link to it.

PageRank assigns every node (webpage) a numerical importance score based on the graph structure.

---

# **Intuition Behind PageRank**

Imagine each webpage is a person voting for another webpage.

Example:

A → B

C → B

D → B

Page **B** receives three votes.

So B appears more important.

However,

not all votes are equal.

A vote from an important webpage is worth more than a vote from an unimportant webpage.

---

# **Example**

Suppose:

A → B

B → C

D → C

Page C receives links from:

* B  
* D

If B itself is highly important,

then C becomes even more important.

---

# **Random Surfer Model**

PageRank is based on the **Random Surfer Model**.

Imagine a person browsing the internet.

They:

1. Open a webpage.  
2. Randomly click one of its links.  
3. Continue forever.

Pages visited more frequently receive higher PageRank.

---

# **Basic PageRank Formula**

The PageRank of page A is:

\[  
PR(A)=\\sum\_{B\\in In(A)}\\frac{PR(B)}{L(B)}  
\]

Where:

* (PR(B)) \= PageRank of page B  
* (L(B)) \= Number of outgoing links from B  
* (In(A)) \= Pages pointing to A

Meaning:

A page receives importance from all pages linking to it.

---

# **Damping Factor**

Real users don't click forever.

Sometimes they type a new URL.

This is modeled using the **Damping Factor**.

Usually:

d \= 0.85

Meaning:

* 85% probability → Follow a link  
* 15% probability → Jump to a random page

---

# **Complete PageRank Formula**

\[  
PR(A)=\\frac{1-d}{N}+d\\sum\_{B\\in In(A)}\\frac{PR(B)}{L(B)}  
\]

Where:

* (N) \= Number of pages  
* (d) \= Damping factor (typically 0.85)

---

# **Solving PageRank (Power Iteration)**

Initially:

Every page receives equal rank.

Example:

4 Pages

PR \= 0.25 each

Then repeatedly:

1. Compute new PageRanks.  
2. Update all nodes.  
3. Repeat until values stop changing (convergence).

This iterative method is called **Power Iteration**.

---

# **Example**

Graph:

A → B

B → C

C → A

Initial:

A \= 1/3

B \= 1/3

C \= 1/3

After several iterations,

the PageRank values stabilize.

---

# **Applications of PageRank**

* Google Search Ranking  
* Citation Networks  
* Social Network Influence  
* Recommendation Systems  
* Fraud Detection

---

# **Random Walk and Embeddings**

Random Walk is the foundation of many graph embedding methods.

---

# **Intuition**

Suppose:

A — B — C  
     |  
     D

Start at A.

Walk randomly:

A

↓

B

↓

D

↓

B

↓

C

Sequence generated:

A B D B C

This sequence behaves like a sentence in NLP.

---

# **Why Random Walk?**

Words occurring together are related.

Similarly,

nodes visited together during random walks are likely related.

Embedding methods exploit this idea.

---

# **Random Walk → Node Embeddings**

Pipeline:

Graph  
   ↓  
Random Walks  
   ↓  
Node Sequences  
   ↓  
Word2Vec (Skip-Gram)  
   ↓  
Node Embeddings

Examples:

* DeepWalk  
* Node2Vec

---

# **Random Walk with Restarts (RWR)**

Standard random walk may wander far away from the starting node.

Sometimes we want to remain close to the starting node.

Solution:

**Random Walk with Restarts.**

---

# **Intuition**

Start from node A.

Each step:

* With probability **c** → Return to A.  
* With probability **1−c** → Continue walking.

Example:

A

↓

B

↓

C

↓

Restart

↓

A

↓

D

---

# **Why Restart?**

Keeps exploration centered around the source node.

Useful for:

* Personalized recommendations  
* Similarity search  
* Local influence analysis

---

# **Personalized PageRank (PPR)**

Personalized PageRank is an extension of PageRank.

Instead of jumping to **any random page**, the surfer jumps back to a **specific preferred node or set of nodes**.

---

# **Intuition**

Suppose you're interested in:

Machine Learning

Search results should prioritize pages related to ML.

Instead of random jumps:

Restart

↓

Machine Learning Page

This creates rankings personalized to your interests.

---

# **Difference Between PageRank and Personalized PageRank**

| PageRank | Personalized PageRank |
| ----- | ----- |
| Random restart anywhere | Restart to chosen node(s) |
| Global importance | Personalized importance |
| Same ranking for everyone | Different rankings for different users |

---

# **Applications of Personalized PageRank**

* Personalized Search  
* Friend Recommendation  
* Product Recommendation  
* Social Networks  
* Knowledge Graph Search

---

# **Matrix Factorization and Node Embeddings**

Another way to learn node embeddings is through **Matrix Factorization**.

Instead of random walks,

we factorize matrices that capture node relationships.

---

# **Intuition**

Suppose we have an adjacency matrix:

|  | A | B | C |
| ----- | ----- | ----- | ----- |
| A | 0 | 1 | 1 |
| B | 1 | 0 | 0 |
| C | 1 | 0 | 0 |

Rather than using this large matrix directly,

we approximate it as:

Adjacency Matrix

≈

Embedding Matrix × Embedding Matrixᵀ

Each row of the embedding matrix becomes a node embedding.

---

# **Why Matrix Factorization?**

Nearby nodes should have similar embeddings.

The factorization learns latent features that preserve graph structure.

---

# **Matrix Factorization Workflow**

Adjacency Matrix  
        ↓  
Factorization  
        ↓  
Low-Dimensional Matrix  
        ↓  
Node Embeddings

---

# **Matrix Factorization vs Random Walk**

| Matrix Factorization | Random Walk |
| ----- | ----- |
| Uses adjacency/similarity matrix | Uses graph traversal |
| Global graph structure | Local neighborhoods |
| Mathematical decomposition | Sampling-based |
| Examples: GraRep, NetMF | DeepWalk, Node2Vec |

---

# **Extension to Graph Embeddings**

Instead of embedding each node,

we embed the **entire graph**.

---

# **Intuition**

Example:

One molecule:

Atoms

\+

Chemical Bonds

↓

One vector.

Entire Molecule

↓

Graph Embedding

---

# **How Graph Embeddings are Created**

### **Step 1**

Compute node embeddings.

Node A

↓

Embedding

---

### **Step 2**

Aggregate all node embeddings.

Methods include:

* Mean Pooling  
* Sum Pooling  
* Max Pooling  
* Attention Pooling

---

### **Step 3**

Produce one graph embedding.

Graph

↓

Single Vector

---

# **Applications of Graph Embeddings**

### **Molecule Classification**

Predict:

Drug or Non-Drug

---

### **Protein Classification**

Predict:

Protein Function

---

### **Social Networks**

Predict:

Community Type

---

### **Cybersecurity**

Predict:

Malicious Network

---

# **Relationship Between These Concepts**

Graph  
   │  
   ├── PageRank  
   │      ↓  
   │   Node Importance  
   │  
   ├── Random Walk  
   │      ↓  
   │   Node Sequences  
   │      ↓  
   │   DeepWalk / Node2Vec  
   │      ↓  
   │   Node Embeddings  
   │  
   ├── Matrix Factorization  
   │      ↓  
   │   Node Embeddings  
   │  
   └── Pooling  
          ↓  
      Graph Embedding

---

# **Interview Summary**

**PageRank** is a graph algorithm that ranks nodes based on the importance of incoming links using the **Random Surfer Model**. It iteratively computes scores using the **PageRank equation** with a **damping factor** to model random jumps. **Random Walks** generate sequences of connected nodes, which can be treated like sentences and used by algorithms such as **DeepWalk** and **Node2Vec** to learn node embeddings. **Random Walk with Restarts (RWR)** periodically returns to a starting node, making it suitable for local similarity and recommendation tasks. **Personalized PageRank (PPR)** extends PageRank by restarting from user-specific nodes, producing personalized rankings. Another approach to learning embeddings is **Matrix Factorization**, which decomposes graph matrices (such as adjacency or similarity matrices) into low-dimensional latent vectors representing nodes. Finally, **Graph Embeddings** are obtained by aggregating node embeddings using pooling techniques, enabling graph-level tasks such as molecule classification, protein function prediction, and network classification.

# graph ml 5

# **Message Passing in Graph Neural Networks (GNN)**

**Message Passing** is the core mechanism of Graph Neural Networks.

The main idea is:

Every node learns by collecting information (messages) from its neighboring nodes.

Instead of making predictions using only its own features, a node also uses information from connected nodes.

---

# **Intuition**

Suppose we have a social network:

     Bob  
       |  
Alice--John--David  
       |  
     Emily

Suppose we want to classify **John**.

Using only John's information:

Age  
Profession  
Location

may not be enough.

Instead, GNN also considers:

Alice  
Bob  
David  
Emily

The neighbors send information to John.

John combines:

Own Features

\+

Neighbor Features

to produce a better representation.

---

# **Message Passing Process**

Each GNN layer performs three steps:

Neighbor Nodes  
      ↓  
Message Generation  
      ↓  
Aggregation  
      ↓  
Node Update

---

## **Step 1: Message Generation**

Each neighboring node creates a message.

Example:

Alice → John's Message

Bob → John's Message

Emily → John's Message

Usually, the message is simply the node's embedding.

---

## **Step 2: Aggregation**

Combine all incoming messages.

Common aggregation methods:

### **Sum**

m \= Alice \+ Bob \+ Emily

---

### **Mean**

m \= Average(Messages)

---

### **Max**

Take the maximum value from each feature.

---

### **Attention**

More important neighbors receive larger weights.

Used in:

Graph Attention Networks (GAT)

---

## **Step 3: Node Update**

Combine:

Current Node Features

\+

Aggregated Messages

to compute a new node embedding.

Example:

Old John Embedding

↓

Combine Neighbor Information

↓

New John Embedding

After several layers:

John indirectly learns from:

Friends

Friends of Friends

Entire Neighborhood

---

# **Message Passing Formula**

One GNN layer updates node (v) as:

\[  
h\_v^{(k+1)}=  
Update  
\\left(  
h\_v^{(k)},  
Aggregate({h\_u^{(k)}:u\\in N(v)})  
\\right)  
\]

Where:

* (h\_v) \= Node embedding  
* (N(v)) \= Neighbor set  
* Aggregate \= Mean, Sum, Max, Attention

---

# **Why Multiple Message Passing Layers?**

Layer 1:

Learn from Immediate Neighbors

Layer 2:

Learn from Neighbors of Neighbors

Layer 3:

Learn from Larger Neighborhood

---

# **Node Classification**

Node Classification predicts labels for individual nodes.

Input:

Graph

\+

Node Features

Output:

Label for each Node

---

# **Examples**

### **Social Network**

Predict:

User Interests

---

### **Banking**

Predict:

Fraudulent Account

---

### **Citation Network**

Predict:

Research Topic

---

### **Healthcare**

Predict:

Disease Gene

---

# **Workflow of Node Classification**

Graph  
      ↓  
Message Passing  
      ↓  
Node Embeddings  
      ↓  
Classifier  
      ↓  
Node Labels

---

# **Relational Classification**

Relational Classification predicts node labels by considering both:

* Node features  
* Neighbor labels

Instead of assuming samples are independent,

it exploits relationships.

---

# **Intuition**

Suppose we want to classify a student.

Known:

Student A

↓

Friends with

↓

Five Computer Science Students

Even if A's features are incomplete,

neighbors strongly suggest:

A probably belongs to Computer Science.

---

# **Example**

Email Network

Nodes:

Employees

Edges:

Email Communication

Task:

Predict department.

Employees communicating frequently usually belong to the same department.

---

# **Collective Classification**

Collective Classification classifies **multiple connected nodes simultaneously**.

Instead of classifying each node independently,

the labels influence one another.

---

# **Intuition**

Suppose:

A

↓

Friend

↓

B

↓

Friend

↓

C

Initially:

Unknown Labels

As the algorithm predicts A,

it improves prediction for B,

which improves prediction for C.

Predictions evolve together.

---

# **Why Collective Classification?**

Traditional ML assumes:

Samples Independent

Graphs violate this assumption.

Connected nodes are often correlated.

Collective Classification leverages these dependencies.

---

# **Iterative Classification Algorithm (ICA)**

ICA is a simple algorithm for Collective Classification.

Instead of predicting once,

it repeatedly updates predictions.

---

# **ICA Steps**

### **Step 1**

Predict labels using only node features.

Example:

A → Student

B → Unknown

C → Unknown

---

### **Step 2**

Use predicted neighbor labels.

Example:

If most neighbors are:

Computer Science

increase probability for Computer Science.

---

### **Step 3**

Update predictions.

---

### **Step 4**

Repeat until labels stop changing.

---

# **ICA Workflow**

Initial Prediction  
        ↓  
Neighbor Labels  
        ↓  
Update Prediction  
        ↓  
Repeat  
        ↓  
Convergence

---

# **Belief Propagation**

Belief Propagation is another inference algorithm for graphical models.

Instead of exchanging embeddings,

nodes exchange **probabilities (beliefs)**.

---

# **Intuition**

Suppose:

A

↓

B

↓

C

A believes:

90% Fraud

This belief influences B.

Then B updates its belief and influences C.

Eventually,

beliefs stabilize.

---

# **Belief Propagation Process**

Each node maintains:

Probability Distribution

Example:

Node A

Fraud \= 0.8

Normal \= 0.2

Nodes repeatedly send beliefs to neighbors.

Neighbors update their own beliefs.

Repeat until convergence.

---

# **Message Passing vs Belief Propagation**

| Message Passing | Belief Propagation |
| ----- | ----- |
| Exchanges embeddings | Exchanges probabilities (beliefs) |
| Used in GNNs | Used in Probabilistic Graphical Models |
| Learns node representations | Performs probabilistic inference |
| Neural network training | Bayesian inference |

---

# **Relational vs Collective Classification**

| Relational Classification | Collective Classification |
| ----- | ----- |
| Uses neighbor information | Uses neighbor predictions jointly |
| Predicts one node at a time | Predicts all nodes together |
| Simpler | More accurate but computationally heavier |

---

# **Node Classification Pipeline in GNN**

Graph  
      ↓  
Initialize Node Features  
      ↓  
Message Passing  
      ↓  
Aggregate Neighbor Information  
      ↓  
Updated Node Embeddings  
      ↓  
Node Classifier  
      ↓  
Predicted Node Labels

---

# **Real-World Applications**

### **Fraud Detection**

Node:

Bank Account

Prediction:

Fraud / Normal

Neighbors:

Transaction Partners

---

### **Social Networks**

Node:

User

Prediction:

Community

Interest

Fake Account

---

### **Citation Networks**

Node:

Research Paper

Prediction:

Research Area

---

### **Healthcare**

Node:

Gene

Prediction:

Disease Association

---

# **Comparison of Methods**

| Method | Main Idea | Uses |
| ----- | ----- | ----- |
| Message Passing | Aggregate neighbor embeddings | GNN Training |
| Node Classification | Predict node labels | Fraud Detection, Recommendation |
| Relational Classification | Use node features \+ neighbor information | Social Networks |
| Collective Classification | Predict labels jointly | Citation Networks |
| Iterative Classification (ICA) | Repeatedly update predictions | Collective Classification |
| Belief Propagation | Exchange probabilities between nodes | Bayesian Networks, Error Correction |

---

# **Interview Summary**

**Message Passing** is the fundamental operation in Graph Neural Networks, where each node gathers information from its neighbors, aggregates it (using mean, sum, max, or attention), and updates its own embedding. This process enables nodes to learn from both their own features and the graph structure. **Node Classification** uses these learned embeddings to predict labels for individual nodes, such as fraud detection or user classification. **Relational Classification** improves predictions by incorporating relationships between connected nodes, while **Collective Classification** jointly predicts labels for multiple interconnected nodes, allowing predictions to reinforce each other. A common approach is the **Iterative Classification Algorithm (ICA)**, which repeatedly updates node labels based on neighbors until convergence. **Belief Propagation**, in contrast, is a probabilistic inference algorithm where nodes exchange probability distributions (beliefs) instead of embeddings, iteratively refining their estimates until they stabilize. Together, these methods exploit the relational structure of graph data to achieve more accurate predictions than treating nodes independently.

# graph ml 6

# **Introduction to Graph Neural Networks (GNN)**

A **Graph Neural Network (GNN)** is a Deep Learning model designed to work with **graph-structured data**, where data consists of **nodes (entities)** connected by **edges (relationships)**.

Unlike CNNs (images) and RNNs (sequences), GNNs learn from both:

* Features of individual nodes  
* Relationships between connected nodes

The key idea is:

**Connected nodes often have similar properties, so each node should learn from its neighbors.**

---

# **Why Deep Learning for Graphs?**

Traditional Deep Learning works well for structured data.

| Data Type | Model |
| ----- | ----- |
| Images | CNN |
| Text | RNN / Transformers |
| Audio | RNN / CNN |
| Graphs | GNN |

Graphs are challenging because:

* Variable number of neighbors  
* No fixed ordering of nodes  
* Irregular structure  
* Complex relationships

A graph can represent:

People → Friendships  
Roads → Cities  
Proteins → Chemical Bonds  
Bank Accounts → Transactions

Traditional Neural Networks cannot naturally process these relationships.

---

# **Why Not Use Traditional Machine Learning?**

Suppose we want to detect fraud.

Traditional ML uses:

Balance  
Age  
Income

But ignores:

Who transferred money to whom?

Fraud often depends on **connections**, not just individual features.

GNNs solve this problem.

---

# **Intuition Behind GNN**

Suppose we have:

     Bob  
       |  
Alice--John--David  
       |  
     Emily

Suppose John's profession is unknown.

His neighbors are:

Alice → Engineer

Bob → Engineer

Emily → Engineer

Even without John's features,

we can infer:

John is probably an Engineer.

This is the intuition of GNNs.

---

# **Graph Representation**

A graph is represented as:

\[  
G=(V,E)  
\]

Where:

* **V** \= Nodes  
* **E** \= Edges

Each node contains:

Features

Example:

Person

↓

Age

Salary

Location

Each edge represents:

Relationship

Example:

Friendship

Transaction

Road

---

# **Basic GNN Architecture**

Graph  
      ↓  
Node Features  
      ↓  
Message Passing  
      ↓  
Neighbor Aggregation  
      ↓  
Updated Node Embeddings  
      ↓  
Prediction

---

# **Deep Learning for Graphs**

Just like CNNs stack multiple convolution layers,

GNNs stack multiple **Graph Convolution Layers**.

Layer 1:

Learn from Immediate Neighbors

Layer 2:

Neighbors of Neighbors

Layer 3:

Larger Neighborhood

This allows the network to capture both local and global graph structure.

---

# **Forward Propagation in GNN**

Forward propagation is the prediction phase.

Each layer performs three operations:

### **Step 1: Message Passing**

Neighbors send information.

A

↓

C

←

B

Messages come from neighboring node embeddings.

---

### **Step 2: Aggregation**

Combine neighbor information.

Common methods:

* Mean  
* Sum  
* Max  
* Attention

Example:

Neighbor Features

↓

Average

---

### **Step 3: Update**

Combine:

Own Features

\+

Aggregated Neighbor Features

to compute a new node embedding.

Mathematically:

# **\[**

# **h\_v^{(k+1)}**

Update  
\\left(  
h\_v^{(k)},  
Aggregate(N(v))  
\\right)  
\]

where:

* (h\_v) \= Node embedding  
* (N(v)) \= Neighbor set

---

# **Example of Forward Propagation**

Suppose:

A \---- B \---- C

Initial embeddings:

A \= \[1,0\]

B \= \[0,1\]

C \= \[1,1\]

Update B:

Average(A,C)

↓

Combine with B

↓

New Embedding

After several layers,

each node contains information from a larger neighborhood.

---

# **Loss Function**

Suppose the task is:

Fraud Detection

Predictions:

Node A

↓

Fraud \= 0.82

Actual:

Fraud \= 1

Loss is computed using:

* Cross Entropy (Classification)  
* MSE (Regression)

---

# **Backward Propagation**

After computing the loss,

errors are propagated backward.

Purpose:

Determine how much each parameter contributed to the prediction error.

Backpropagation computes gradients for:

* Weight matrices  
* Aggregation functions  
* Node transformation layers

---

# **Steps of Backpropagation**

### **Step 1**

Forward propagation.

---

### **Step 2**

Compute loss.

---

### **Step 3**

Calculate gradients.

---

### **Step 4**

Update weights.

---

### **Step 5**

Repeat.

---

# **Stochastic Gradient Descent (SGD)**

Weights are updated using Gradient Descent.

Update rule:

\[  
w=w-\\eta\\frac{\\partial L}{\\partial w}  
\]

where:

* (w) \= Weight  
* (\\eta) \= Learning Rate

---

## **Batch Gradient Descent**

Uses:

Entire Dataset

Very stable but slow.

---

## **Stochastic Gradient Descent**

Uses:

One Training Sample

Advantages:

* Faster  
* Less memory  
* Can escape local minima

---

## **Mini-Batch SGD**

Uses small batches:

32

64

128

Most commonly used.

---

# **GraphSAGE (Graph Sample and Aggregate)**

GraphSAGE stands for:

Graph Sample and Aggregate

It is one of the most popular GNN architectures, proposed to efficiently learn node embeddings on **large graphs**.

Unlike standard GCNs, GraphSAGE **does not require the entire graph during training**. Instead, it samples a fixed number of neighbors, making it scalable.

---

# **Why GraphSAGE?**

Suppose a Facebook user has:

10 Million Friends

Aggregating all neighbors is impossible.

GraphSAGE solves this by:

Randomly Sample

↓

Fixed Number of Neighbors

↓

Aggregate

Example:

10 Million Friends

↓

Sample 25

↓

Use Only Those

This dramatically reduces computation.

---

# **GraphSAGE Architecture**

Graph  
      ↓  
Sample Neighbors  
      ↓  
Aggregate Neighbor Features  
      ↓  
Combine with Node Features  
      ↓  
Apply Neural Network  
      ↓  
New Node Embedding

---

# **GraphSAGE Algorithm**

### **Step 1: Neighbor Sampling**

Instead of using every neighbor,

sample a fixed number.

Example:

Node A

↓

Neighbors

↓

Randomly Pick 10

---

### **Step 2: Aggregation**

Aggregate sampled neighbor embeddings.

Common aggregators:

* Mean Aggregator  
* Max Pooling Aggregator  
* LSTM Aggregator

---

### **Step 3: Concatenation**

Combine:

Node Embedding

\+

Aggregated Neighbor Embedding

---

### **Step 4: Linear Transformation**

Pass through a neural network layer.

Result:

Updated Node Embedding

---

### **Step 5: Repeat**

Repeat for multiple layers.

Each layer captures information from farther nodes.

---

# **GraphSAGE Aggregators**

## **1\. Mean Aggregator**

Average neighbor embeddings.

Most common.

---

## **2\. Max Pooling Aggregator**

Apply a neural network to each neighbor, then take the maximum value across neighbors.

Captures the strongest features.

---

## **3\. LSTM Aggregator**

Uses an LSTM to combine neighbor information.

Can model complex interactions but depends on neighbor ordering, making it less commonly used.

---

# **GraphSAGE Workflow**

Graph  
      ↓  
Sample Neighbors  
      ↓  
Aggregate  
      ↓  
Concatenate  
      ↓  
Linear Layer  
      ↓  
ReLU  
      ↓  
Updated Embedding  
      ↓  
Repeat  
      ↓  
Node Prediction

---

# **GCN vs GraphSAGE**

| Feature | GCN | GraphSAGE |
| ----- | ----- | ----- |
| Uses All Neighbors | Yes | No |
| Neighbor Sampling | No | Yes |
| Scalability | Moderate | Excellent |
| Large Graphs | Difficult | Very Good |
| Inductive Learning | Limited | Yes |

---

# **Applications of GraphSAGE**

### **Social Networks**

Friend recommendation.

---

### **Banking**

Fraud detection.

---

### **Recommendation Systems**

User-product recommendations.

---

### **Knowledge Graphs**

Entity prediction.

---

### **Biology**

Protein interaction prediction.

---

# **Complete GNN Training Pipeline**

Graph Data  
      ↓  
Initialize Node Features  
      ↓  
Neighbor Sampling (GraphSAGE)  
      ↓  
Message Passing  
      ↓  
Aggregation  
      ↓  
Updated Node Embeddings  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Stochastic Gradient Descent  
      ↓  
Update Weights  
      ↓  
Repeat Until Convergence

---

# **Interview Summary**

A **Graph Neural Network (GNN)** is a deep learning model designed for **graph-structured data**, where nodes learn by exchanging information with their neighbors through **message passing**. During **forward propagation**, each node aggregates information from neighboring nodes, combines it with its own features, and generates updated node embeddings, which are used for tasks such as node, edge, or graph prediction. The prediction error is measured using a loss function, and **backpropagation** computes gradients to update the network parameters using **Stochastic Gradient Descent (SGD)** or Mini-Batch SGD. **GraphSAGE (Graph Sample and Aggregate)** is a scalable GNN architecture that addresses large graphs by **sampling a fixed number of neighbors** instead of aggregating information from all neighbors. It then aggregates the sampled neighbor embeddings (using mean, max pooling, or LSTM aggregators), combines them with the node's own features, and applies a neural network to produce updated embeddings. GraphSAGE supports **inductive learning**, making it highly effective for large-scale social networks, recommendation systems, fraud detection, and biological network analysis.

# graph ml 7

# **General Perspective on Graph Neural Networks (GNN)**

Graph Neural Networks (GNNs) are Deep Learning models designed to learn from **graph-structured data**, where objects are represented as **nodes** and their relationships as **edges**.

Unlike traditional neural networks that treat samples independently, GNNs exploit the fact that **connected nodes often influence each other**.

The fundamental principle is:

**A node's representation should be learned using both its own features and information from its neighboring nodes.**

This idea is called **Neighborhood Aggregation** or **Message Passing**.

---

# **Why Do We Need GNNs?**

Traditional Machine Learning assumes:

Each sample is independent.

However, many real-world problems involve relationships.

Examples:

Facebook → Friendships

Google Maps → Roads

Banks → Money Transfers

Molecules → Chemical Bonds

Ignoring these relationships results in loss of valuable information.

GNNs learn from:

Node Features

\+

Graph Structure

\+

Neighbor Information

---

# **Intuition Behind GNN**

Suppose we have:

     Bob  
       |  
Alice--John--David  
       |  
     Emily

Suppose John's occupation is unknown.

Neighbors:

Alice → Engineer

Bob → Engineer

Emily → Engineer

A GNN learns:

John is likely an Engineer.

because neighboring nodes provide useful context.

---

# **General GNN Workflow**

Graph  
      ↓  
Node Features  
      ↓  
Message Passing  
      ↓  
Neighbor Aggregation  
      ↓  
Updated Node Embeddings  
      ↓  
Prediction

---

# **Single Layer of a GNN**

A **single GNN layer** allows each node to collect information from its **immediate (1-hop) neighbors**.

One GNN layer performs three operations:

---

## **Step 1: Message Passing**

Each neighboring node sends its embedding.

Example:

A

↓

C

←

B

Messages arrive from:

Neighbor Embeddings

---

## **Step 2: Aggregation**

Combine neighbor information.

Common methods:

Mean

Sum

Max

Attention

Example:

Neighbor Features

↓

Average

---

## **Step 3: Update**

Combine:

Node's Own Features

\+

Neighbor Information

to create a new embedding.

Mathematically:

# **\[**

# **h\_v^{(1)}**

Update  
\\left(  
h\_v^{(0)},  
Aggregate(N(v))  
\\right)  
\]

Where:

* (h\_v^{(0)}) \= Initial node features  
* (N(v)) \= Neighbor set

---

# **Example**

Graph:

A \---- B \---- C

Initially:

A \= Blue

B \= Green

C \= Red

After one GNN layer,

Node B learns from:

A

\+

C

Only immediate neighbors are considered.

---

# **What Does One Layer Learn?**

One GNN layer captures:

1-Hop Neighborhood

Example:

     A  
      |  
B \---- C \---- D  
      |  
      E

Node C learns only from:

A

B

D

E

---

# **Stacking Multiple GNN Layers**

One layer only captures local information.

To capture larger neighborhoods,

multiple GNN layers are stacked.

Example:

Input  
   ↓  
GNN Layer 1  
   ↓  
GNN Layer 2  
   ↓  
GNN Layer 3  
   ↓  
Prediction

---

# **Why Stack Layers?**

Suppose:

A — B — C — D

---

### **One Layer**

Node B learns:

A

C

---

### **Two Layers**

Node B now learns:

A

C

D

because C already contains information from D.

---

### **Three Layers**

Node B indirectly learns from:

Entire Graph

---

# **Receptive Field**

The **Receptive Field** is the portion of the graph a node can access.

After:

### **One Layer**

1-Hop Neighbors

---

### **Two Layers**

2-Hop Neighbors

---

### **Three Layers**

3-Hop Neighbors

The receptive field grows with network depth.

---

# **Advantages of Stacking Layers**

More layers allow the model to:

* Learn larger graph context  
* Capture long-range dependencies  
* Discover global graph patterns

---

# **Problems with Deep GNNs**

Stacking too many layers introduces challenges.

---

## **1\. Over-Smoothing**

The most common problem.

After many layers,

neighbor information is repeatedly averaged.

Eventually:

All Node Embeddings

↓

Become Nearly Identical

The model cannot distinguish between nodes.

---

### **Example**

Initially:

A \= Engineer

B \= Doctor

C \= Lawyer

After many layers:

A ≈ B ≈ C

Unique information is lost.

---

## **2\. Vanishing Gradients**

Gradients become very small during backpropagation,

making learning difficult.

---

## **3\. Overfitting**

Deep GNNs may memorize training graphs instead of generalizing.

---

# **Skip Connections (Residual Connections)**

Skip Connections solve several problems in deep GNNs.

Instead of passing information only layer by layer,

they create shortcut connections.

---

# **Intuition**

Traditional network:

Layer 1

↓

Layer 2

↓

Layer 3

Skip Connection:

Layer 1  
   │  
   ├──────────►  
   │           │  
   ▼           ▼  
Layer 2 → Layer 3

The original information bypasses intermediate layers.

---

# **Why Skip Connections?**

They help preserve:

Original Node Features

while still learning new information.

---

# **Residual Update**

Instead of:

New Embedding

\=

Layer Output

Use:

New Embedding

\=

Old Embedding

\+

Layer Output

The model learns only the residual (difference).

---

# **Advantages of Skip Connections**

### **Prevent Over-Smoothing**

Original node information is preserved.

---

### **Better Gradient Flow**

Gradients can travel through shortcut paths.

---

### **Easier Training**

Deep GNNs become more stable.

---

### **Improved Accuracy**

Allows training of deeper architectures.

---

# **Example**

Without Skip Connection:

Layer1

↓

Layer2

↓

Layer3

↓

Prediction

Important information may disappear.

---

With Skip Connection:

Layer1  
   │  
   ▼  
Layer2  
   │  
   ▼  
Layer3  
   ▲  
   │  
Layer1 Output

Original node representation is retained.

---

# **Skip Connections in Modern GNNs**

Used in:

* DeepGCN  
* Residual GCN  
* GAT Variants  
* Graph Transformers

---

# **Complete Deep GNN Pipeline**

Graph  
      ↓  
Initial Node Features  
      ↓  
GNN Layer 1  
      ↓  
Skip Connection  
      ↓  
GNN Layer 2  
      ↓  
Skip Connection  
      ↓  
GNN Layer 3  
      ↓  
Node Embeddings  
      ↓  
Prediction

---

# **Single Layer vs Multiple Layers**

| Feature | Single GNN Layer | Multiple GNN Layers |
| ----- | ----- | ----- |
| Information Learned | Immediate neighbors (1-hop) | Multi-hop neighbors |
| Graph Context | Local | Local \+ Global |
| Computational Cost | Low | Higher |
| Typical Use | Simple graphs | Complex graphs |

---

# **Stacking Layers vs Skip Connections**

| Stacking Layers | Skip Connections |
| ----- | ----- |
| Increases receptive field | Preserves original information |
| Captures larger neighborhoods | Prevents over-smoothing |
| May suffer vanishing gradients | Improves gradient flow |
| Can overfit if very deep | Enables deeper GNNs |

---

# **Interview Summary**

A **Graph Neural Network (GNN)** is a deep learning model for graph-structured data that learns node representations by aggregating information from neighboring nodes. A **single GNN layer** performs **message passing**, **neighbor aggregation**, and **node embedding updates**, allowing each node to learn from its **1-hop neighbors**. By **stacking multiple GNN layers**, the receptive field expands so that nodes can capture information from **2-hop, 3-hop, and more distant neighbors**, enabling the model to learn broader graph structure. However, very deep GNNs suffer from **over-smoothing**, where node embeddings become too similar, and **vanishing gradients**, making training difficult. **Skip connections (Residual Connections)** address these issues by directly adding earlier layer representations to later layers, preserving original node features, improving gradient flow, reducing over-smoothing, and enabling the successful training of deeper GNN architectures.

# graph ml 8

# **GNN Augmentation and Training**

Training a **Graph Neural Network (GNN)** is similar to training other deep learning models but includes additional steps to handle graph structure, neighborhood sampling, and message passing.

The overall pipeline is:

Graph Data  
      ↓  
Graph Augmentation  
      ↓  
Neighbor Sampling  
      ↓  
Forward Propagation  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Gradient Descent  
      ↓  
Model Evaluation

---

# **Why is GNN Training Different?**

Unlike CNNs or MLPs, every node depends on its neighbors.

Example:

Node A

↓

Needs Features of

↓

Neighbors

↓

Neighbors of Neighbors

Therefore, training must efficiently handle graph connectivity.

---

# **GNN Augmentation**

Graph Augmentation refers to modifying the graph during training to improve **generalization**, reduce **overfitting**, and make the model more robust.

It is similar to **image augmentation** in CNNs (rotation, flipping, cropping), but for graph data.

---

# **Why Augment Graphs?**

Real-world graphs may contain:

* Missing edges  
* Noisy connections  
* Incomplete node features  
* Small labeled datasets

Augmentation helps the model learn more robust representations.

---

# **Types of Graph Augmentation**

---

## **1\. Edge Dropping**

Randomly remove some edges.

Example:

Before:

A \----- B \----- C

After:

A     B \----- C

Benefits:

* Prevents over-reliance on specific edges.  
* Reduces overfitting.

---

## **2\. Node Dropping**

Randomly remove nodes during training.

Useful for:

Large Graphs

Makes the model robust to missing nodes.

---

## **3\. Feature Masking**

Hide some node features.

Example:

Before:

Age \= 25

Salary \= 50K

After:

Age \= ?

Salary \= 50K

The model learns to rely on neighborhood information.

---

## **4\. Subgraph Sampling**

Instead of using the entire graph,

train on smaller subgraphs.

Benefits:

* Faster training  
* Lower memory usage  
* Better scalability

---

## **5\. Edge Addition**

Randomly add edges.

Improves robustness against missing links.

---

# **Neighborhood Sampling**

One of the biggest challenges in GNNs is that some nodes have **thousands or millions of neighbors**.

Example:

Facebook User

↓

10 Million Friends

Aggregating all neighbors is computationally impossible.

Solution:

Neighborhood Sampling

---

# **Intuition**

Instead of using:

1,000,000 Neighbors

Sample:

20

50

100

neighbors.

The sampled neighbors approximate the full neighborhood.

---

# **Types of Neighborhood Sampling**

---

## **1\. Uniform Random Sampling**

Every neighbor has equal probability.

Example:

100 Neighbors

↓

Randomly Select 25

Simple and efficient.

---

## **2\. Fixed-Size Sampling**

Always sample a fixed number.

Example:

25 Neighbors

Used by:

GraphSAGE

---

## **3\. Importance Sampling**

More important neighbors have higher probability.

Importance may depend on:

* Edge weight  
* Degree  
* Attention score

---

## **4\. Layer-wise Sampling**

Instead of sampling neighbors for every node,

sample once for an entire layer.

Used by:

* FastGCN  
* LADIES

---

# **GraphSAGE Neighbor Sampling**

GraphSAGE uses:

Sample

↓

Aggregate

↓

Update

Example:

Node

↓

1000 Neighbors

↓

Sample 20

↓

Aggregate

↓

Update Embedding

Makes GNNs scalable.

---

# **Training a GNN**

Training follows the same basic idea as other neural networks.

---

## **Step 1**

Initialize:

* Node Features  
* Edge Features  
* Weights

---

## **Step 2**

Sample neighbors (if needed).

---

## **Step 3**

Perform Message Passing.

Each node gathers information.

---

## **Step 4**

Aggregate neighbor embeddings.

Example:

Mean

Sum

Max

Attention

---

## **Step 5**

Update node embeddings.

---

## **Step 6**

Forward Propagation.

Generate predictions.

---

## **Step 7**

Compute Loss.

Examples:

Classification:

Cross Entropy Loss

Regression:

Mean Squared Error

---

## **Step 8**

Backpropagation.

Compute gradients.

---

## **Step 9**

Update weights using:

* SGD  
* Adam  
* RMSProp

---

## **Step 10**

Repeat until convergence.

---

# **GNN Training Pipeline**

Input Graph  
      ↓  
Node Features  
      ↓  
Neighbor Sampling  
      ↓  
Message Passing  
      ↓  
Aggregation  
      ↓  
Update Node Embeddings  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Function  
      ↓  
Backpropagation  
      ↓  
Optimizer  
      ↓  
Updated Parameters

---

# **Setting Up GNN Prediction Tasks**

A GNN can solve different types of prediction problems.

---

# **1\. Node-Level Prediction**

Predict properties of individual nodes.

Examples:

* Fraud Detection  
* User Classification  
* Disease Gene Prediction

Input:

Node \+ Neighborhood

Output:

Node Label

---

# **2\. Edge-Level Prediction**

Predict properties of relationships.

Examples:

* Friend Recommendation  
* Link Prediction  
* Drug Interaction Prediction

Input:

Two Nodes

\+

Edge Features

Output:

Edge Label

---

# **3\. Graph-Level Prediction**

Predict a label for the entire graph.

Examples:

* Molecule Classification  
* Protein Function Prediction  
* Malware Detection

Input:

Entire Graph

Output:

Graph Label

---

# **Train, Validation and Test Split**

Like traditional ML,

graph datasets are divided into:

Training Set

Validation Set

Test Set

---

For node classification:

Nodes

↓

Train

Validation

Test

---

For graph classification:

Entire graphs are split.

---

# **Loss Functions**

---

## **Node Classification**

Cross Entropy Loss

---

## **Graph Classification**

Cross Entropy Loss

---

## **Regression**

Mean Squared Error

---

## **Link Prediction**

Binary Cross Entropy

---

# **Optimizers**

Common optimizers:

* SGD  
* Adam (Most Popular)  
* RMSProp

Adam usually converges faster than plain SGD.

---

# **Evaluation Metrics**

Evaluation depends on the task.

---

# **Node Classification**

### **Accuracy**

Most common.

\[  
Accuracy=  
\\frac{Correct}{Total}  
\]

---

### **Precision**

Useful for imbalanced datasets.

\[  
Precision=  
\\frac{TP}{TP+FP}  
\]

---

### **Recall**

Measures ability to detect positives.

\[  
Recall=  
\\frac{TP}{TP+FN}  
\]

---

### **F1 Score**

Balances Precision and Recall.

\[  
F1=  
\\frac{2PR}{P+R}  
\]

---

### **ROC-AUC**

Measures ranking quality.

Useful for binary classification.

---

# **Link Prediction**

Common metrics:

* ROC-AUC  
* Average Precision (AP)  
* Hits@K  
* Mean Reciprocal Rank (MRR)

---

# **Graph Classification**

Common metrics:

* Accuracy  
* Precision  
* Recall  
* F1 Score  
* ROC-AUC

---

# **Regression Tasks**

Metrics:

* Mean Absolute Error (MAE)  
* Mean Squared Error (MSE)  
* Root Mean Squared Error (RMSE)  
* R² Score

---

# **Early Stopping**

To prevent overfitting:

Monitor:

Validation Loss

If it stops improving for several epochs,

stop training.

---

# **Hyperparameters in GNN Training**

Common hyperparameters:

* Number of GNN layers  
* Hidden dimension  
* Learning rate  
* Batch size  
* Number of sampled neighbors  
* Dropout rate  
* Weight decay  
* Number of epochs

---

# **Complete GNN Workflow**

Graph Dataset  
      ↓  
Graph Augmentation  
      ↓  
Train / Validation / Test Split  
      ↓  
Neighbor Sampling  
      ↓  
Message Passing  
      ↓  
Aggregation  
      ↓  
Updated Embeddings  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Loss Calculation  
      ↓  
Backpropagation  
      ↓  
Optimizer (Adam/SGD)  
      ↓  
Parameter Update  
      ↓  
Repeat Until Convergence  
      ↓  
Evaluation Metrics

---

# **Interview Summary**

**GNN training** extends standard deep learning by incorporating graph structure through **message passing** and **neighbor aggregation**. To improve robustness, **graph augmentation** techniques such as **edge dropping, node dropping, feature masking, subgraph sampling, and edge addition** are used during training. Since large graphs make full-neighborhood aggregation computationally expensive, **neighbor sampling** methods such as **uniform sampling, fixed-size sampling (GraphSAGE), importance sampling, and layer-wise sampling** are employed. A typical GNN training pipeline consists of **graph preprocessing, augmentation, train/validation/test splitting, neighbor sampling, message passing, forward propagation, loss computation, backpropagation, and optimization using SGD or Adam**. GNNs can be configured for **node-level, edge-level, or graph-level prediction tasks**, with evaluation metrics chosen accordingly: **Accuracy, Precision, Recall, F1-score, and ROC-AUC** for classification tasks; **Hits@K** and **MRR** for link prediction; and **MSE, RMSE, MAE, and R²** for regression tasks. Proper augmentation, sampling, optimization, and evaluation are essential for building scalable and accurate graph neural networks.

# graph ml 9

# **Expressiveness of Graph Neural Networks (GNNs)**

One of the most important theoretical questions in Graph Neural Networks is:

**How powerful is a GNN at distinguishing different graph structures?**

This ability is called the **expressiveness** of a GNN.

A highly expressive GNN can distinguish graphs that have different structures, while a less expressive GNN may incorrectly produce the same embedding for different graphs.

---

# **What Does Expressiveness Mean?**

Expressiveness refers to:

**The ability of a GNN to generate different embeddings for structurally different graphs or nodes.**

A more expressive GNN can capture:

* Graph topology  
* Node neighborhoods  
* Complex structural patterns  
* Different graph structures

---

# **Intuition**

Suppose we have two graphs.

Graph 1:

A  
|  
B  
|  
C

Graph 2:

  A  
 / | \\  
B  C  D

A powerful GNN should assign:

Different Embeddings

because the structures are clearly different.

A weak GNN might incorrectly produce:

Same Embedding

making the graphs indistinguishable.

---

# **Why Does Expressiveness Matter?**

Many applications depend on graph structure.

Examples:

* Molecule Classification  
* Fraud Detection  
* Protein Analysis  
* Social Networks

If the GNN cannot distinguish different graph structures,

predictions become inaccurate.

---

# **Why Are Some GNNs Less Expressive?**

Most GNNs update node embeddings using:

Mean

Sum

Max

aggregation.

Sometimes different neighborhoods produce identical aggregated values.

Example:

Neighborhood 1:

2

4

Mean:

3

Neighborhood 2:

3

3

Mean:

3

Different neighborhoods,

same mean.

The GNN cannot distinguish them.

---

# **Designing More Expressive GNNs**

To improve expressiveness, researchers focus on:

### **Better Aggregation**

Instead of simple averaging,

learn richer aggregation functions.

---

### **Better Update Functions**

Use deeper neural networks.

---

### **Preserve Structural Information**

Include:

* Degree  
* Position  
* Edge Features  
* Attention

---

### **Avoid Over-Smoothing**

Maintain unique node information.

---

# **Weisfeiler-Lehman (WL) Test**

The **Weisfeiler-Lehman (WL) Test** is the standard benchmark used to measure the expressive power of GNNs.

It is also called:

1-WL Test

Color Refinement Algorithm

---

# **Purpose of WL Test**

Determine whether:

Two Graphs

↓

Are Structurally Different

If WL distinguishes them,

a highly expressive GNN should also distinguish them.

---

# **Intuition Behind WL Test**

Initially,

every node receives the same color.

Example:

Gray

---

Iteration 1:

Each node collects colors from neighbors.

Example:

Node

↓

Neighbor Colors

↓

Generate New Color

---

Iteration 2:

Repeat.

Eventually,

different graph structures receive different color patterns.

---

# **Example**

Graph A:

A

|

B

|

C

Graph B:

A

/ \\

B  C

After several WL iterations,

nodes receive different colors,

indicating different graph structures.

---

# **Relationship Between WL and GNNs**

Most Message Passing GNNs perform operations similar to the WL Test.

Both:

Collect Neighbor Information

↓

Aggregate

↓

Update Representation

Therefore,

the expressive power of many GNNs is bounded by the **1-WL Test**.

---

# **Limitation of Standard GNNs**

Standard GCNs and GraphSAGE cannot distinguish some non-isomorphic graphs.

They are:

No More Expressive

Than

1-WL

---

# **Graph Isomorphism**

Two graphs are **isomorphic** if they have identical structures but different node labels.

Example:

Graph 1:

A \----- B

|

C

Graph 2:

X \----- Y

|

Z

Structurally,

they are identical.

A good GNN should assign similar embeddings.

---

# **Graph Isomorphism Network (GIN)**

The **Graph Isomorphism Network (GIN)** was proposed to achieve the maximum expressive power possible for Message Passing GNNs.

It is theoretically proven to be:

As Powerful

As

1-WL Test

---

# **Intuition Behind GIN**

Instead of averaging neighbors,

GIN uses:

Sum Aggregation

Why?

Because:

Sum

↓

Preserves More Information

Than

Mean

---

# **GIN Update Rule**

Node embedding update:

# **\[**

# **h\_v^{(k)}**

MLP  
\\left(  
(1+\\epsilon)h\_v^{(k-1)}  
\+  
\\sum\_{u\\in N(v)}h\_u^{(k-1)}  
\\right)  
\]

Where:

* (h\_v) \= Node embedding  
* (N(v)) \= Neighbor set  
* MLP \= Multi-Layer Perceptron  
* (\\epsilon) \= Learnable or fixed parameter

---

# **Why Sum Aggregation?**

Suppose:

Neighborhood A:

1

2

Sum:

3

Neighborhood B:

3

Sum:

3

Although sums can occasionally match, **combined with an MLP and repeated message passing, sum aggregation is injective over multisets under suitable conditions**, making it more expressive than mean or max in theory.

---

# **Why Is GIN More Expressive?**

GIN preserves:

* Number of neighbors  
* Node multiplicities  
* Structural differences

Mean aggregation may lose this information.

---

# **GCN vs GraphSAGE vs GIN**

| Feature | GCN | GraphSAGE | GIN |
| ----- | ----- | ----- | ----- |
| Aggregation | Mean/Normalized Sum | Mean, Max, LSTM | Sum |
| Expressiveness | Moderate | Moderate | Highest among standard message-passing GNNs |
| Neighbor Sampling | No | Yes | No |
| Large Graphs | Moderate | Excellent | Moderate |
| Theoretical Power | ≤ 1-WL | ≤ 1-WL | Matches 1-WL |

---

# **Can GIN Beat WL?**

No.

A standard GIN **matches** the expressive power of the **1-Weisfeiler-Lehman (1-WL)** test but does **not exceed it**.

If the WL test cannot distinguish two graphs,

GIN also cannot.

To go beyond WL,

researchers use more advanced methods such as:

* Higher-Order GNNs  
* k-WL-based GNNs  
* Graph Transformers  
* Subgraph GNNs

---

# **Designing More Expressive GNNs**

Researchers improve expressiveness by:

### **Better Aggregation**

Use injective aggregation (e.g., sum).

---

### **Richer Node Features**

Include:

* Degree  
* Position  
* Edge attributes  
* Structural encodings

---

### **Attention Mechanisms**

Learn which neighbors are more important.

Example:

Graph Attention Networks (GAT)

---

### **Deeper Architectures with Skip Connections**

Capture larger neighborhoods while reducing over-smoothing.

---

### **Higher-Order Message Passing**

Capture interactions among groups of nodes instead of only pairwise neighbors.

---

# **Applications Requiring High Expressiveness**

### **Drug Discovery**

Different molecular structures require distinct embeddings.

---

### **Protein Analysis**

Small structural changes affect protein function.

---

### **Fraud Detection**

Transaction patterns matter more than individual accounts.

---

### **Social Networks**

Community structures should be distinguished accurately.

---

# **Interview Summary**

The **expressiveness of a Graph Neural Network (GNN)** refers to its ability to distinguish different graph structures and produce unique node or graph embeddings. Most message-passing GNNs update node representations by aggregating information from neighbors, but simple aggregation methods such as **mean** or **max** can lose structural information, limiting expressive power. The **Weisfeiler-Lehman (1-WL) Test**, also known as the **Color Refinement Algorithm**, is the standard benchmark for evaluating GNN expressiveness. Most traditional GNNs, including **GCN** and **GraphSAGE**, are at most as expressive as the **1-WL test**. The **Graph Isomorphism Network (GIN)** improves expressiveness by using **sum aggregation** together with a **Multi-Layer Perceptron (MLP)**, and it has been proven to match the expressive power of the **1-WL test**, making it one of the most expressive standard message-passing GNN architectures. However, GIN cannot distinguish graphs that 1-WL itself cannot distinguish; surpassing 1-WL requires more advanced architectures such as **higher-order GNNs**, **k-WL-based models**, or **Graph Transformers**.

# graph ml 10

# **Heterogeneous Graphs**

A **Heterogeneous Graph (Heterograph)** is a graph that contains **multiple types of nodes and/or multiple types of edges (relations)**.

Unlike homogeneous graphs, where all nodes and edges are of the same type, heterogeneous graphs model much richer real-world systems.

---

# **Homogeneous vs Heterogeneous Graph**

### **Homogeneous Graph**

Only one node type and one edge type.

Example:

Person \---- Friend \---- Person

Everything is the same type.

---

### **Heterogeneous Graph**

Multiple node types and edge types.

Example:

User \------ Buys \------ Product  
  |                          |  
Follows                  Belongs To  
  |                          |  
 User \------ Sold By \------ Seller

Node types:

User

Product

Seller

Edge types:

Buys

Follows

Sold By

---

# **Formal Definition**

A heterogeneous graph is represented as

\[  
G=(V,E,T\_V,T\_E)  
\]

Where:

* (V) \= Nodes  
* (E) \= Edges  
* (T\_V) \= Node Types  
* (T\_E) \= Edge Types

---

# **Examples of Heterogeneous Graphs**

## **E-Commerce**

Nodes:

* User  
* Product  
* Seller

Edges:

* Buys  
* Reviews  
* Sells

---

## **Academic Network**

Nodes:

* Author  
* Paper  
* Conference

Edges:

* Writes  
* Published In  
* Cites

---

## **Healthcare**

Nodes:

* Patient  
* Disease  
* Drug  
* Doctor

Edges:

* Diagnosed With  
* Prescribed  
* Treats

---

# **Why Ordinary GNNs Fail**

Standard GCN assumes:

Every edge has the same meaning.

But in a heterogeneous graph:

User \----Buys---- Product

User \----Follows---- User

These relations have different semantics.

Treating them identically loses information.

---

# **Knowledge Graph (KG)**

A **Knowledge Graph** is a special type of heterogeneous graph where information is stored as **entities** connected by **typed relationships**.

Knowledge is represented as **triples**.

---

# **Triple Representation**

Each fact is represented as:

(Head, Relation, Tail)

or

\[  
(h,r,t)  
\]

Example:

(Paris, CapitalOf, France)

(Albert Einstein, BornIn, Germany)

(ChatGPT, DevelopedBy, OpenAI)

---

# **Visual Example**

Einstein  
     |  
   BornIn  
     |  
Germany  
     |  
Capital  
     |  
Berlin

---

# **Applications of Knowledge Graphs**

* Google Search  
* Recommendation Systems  
* Question Answering  
* Drug Discovery  
* Chatbots  
* Knowledge Retrieval

---

# **Knowledge Graph Embeddings (KGE)**

Knowledge Graph Embeddings convert:

Entities

Relations

into dense vectors.

Instead of storing:

Paris

France

store:

Paris → \[0.3,0.7,-1.2\]

France → \[1.4,-0.6,0.8\]

Relations also receive embeddings.

---

# **Why KG Embeddings?**

They enable:

* Link Prediction  
* Missing Fact Prediction  
* Recommendation  
* Reasoning

using vector operations.

---

# **Knowledge Graph Completion**

Knowledge Graphs are often incomplete.

Example:

Known:

(Paris, CapitalOf, France)

Missing:

(France, LocatedIn, Europe)

Goal:

Predict missing triples.

This task is called:

Knowledge Graph Completion (KGC)

---

# **KG Completion Using Embeddings**

Pipeline:

Knowledge Graph  
        ↓  
Learn Entity Embeddings  
        ↓  
Learn Relation Embeddings  
        ↓  
Score Candidate Triples  
        ↓  
Predict Missing Links

---

# **Relational GCN (R-GCN)**

**Relational Graph Convolutional Network (R-GCN)** extends GCN to heterogeneous graphs.

Instead of one weight matrix,

R-GCN learns:

One Weight Matrix

Per Relation Type

---

# **Intuition**

Suppose:

User \--Buys--\> Product

User \--Follows--\> User

Different relations require different transformations.

R-GCN performs:

Buy Relation

↓

Transformation W₁

Follow Relation

↓

Transformation W₂

---

# **R-GCN Workflow**

Node  
     ↓  
Group Neighbors by Relation  
     ↓  
Apply Relation-specific Weights  
     ↓  
Aggregate  
     ↓  
Updated Embedding

---

# **Advantages of R-GCN**

* Handles multiple relation types  
* Preserves relation semantics  
* Excellent for knowledge graphs

---

# **Limitations of R-GCN**

Many relation types require:

Many Weight Matrices

Large KGs become memory intensive.

---

# **Relational GraphSAGE (R-GraphSAGE)**

GraphSAGE samples neighbors.

R-GraphSAGE extends it to heterogeneous graphs.

Pipeline:

Sample Neighbors  
        ↓  
Group by Relation  
        ↓  
Relation-wise Aggregation  
        ↓  
Combine  
        ↓  
Embedding

Benefits:

* Scalable  
* Suitable for very large heterogeneous graphs

---

# **Relational Graph Attention Network (R-GAT)**

GAT uses attention.

R-GAT extends attention to heterogeneous graphs.

Instead of equal importance,

attention is computed separately for each relation.

Example:

User → Bought → Product

Attention \= 0.9

User → Viewed → Product

Attention \= 0.3

The model learns:

Bought

\>

Viewed

for purchase prediction.

---

# **Comparison**

| Model | Supports Relations | Neighbor Sampling | Attention |
| ----- | ----- | ----- | ----- |
| GCN | No | No | No |
| GraphSAGE | No | Yes | No |
| GAT | No | No | Yes |
| R-GCN | Yes | No | No |
| R-GraphSAGE | Yes | Yes | No |
| R-GAT | Yes | Optional | Yes |

---

# **Knowledge Graph Embedding Models**

Several models learn embeddings differently.

---

# **1\. TransE**

Most famous KG embedding model.

Core idea:

Head

\+

Relation

≈

Tail

Formula:

\[  
h+r\\approx t  
\]

---

## **Intuition**

Example:

(Paris, CapitalOf, France)

Embedding:

Paris

\+

CapitalOf

≈

France

Relation acts like a translation vector.

Hence the name:

Translation Embedding

---

## **Advantages**

* Simple  
* Fast  
* Easy to train

---

## **Limitations**

Poor for:

* One-to-many  
* Many-to-one  
* Many-to-many relations

Example:

Person

BornIn

City

Many people share one city.

TransE struggles.

---

# **2\. TransR**

Improves TransE.

Idea:

Entities and relations belong to different spaces.

Instead of:

One Embedding Space

TransR learns:

Entity Space

Relation Space

Entities are projected into relation-specific spaces before translation.

Advantages:

* Better modeling of complex relations  
* Handles many-to-many relationships better

---

# **3\. DistMult**

Instead of translation,

DistMult uses multiplicative interactions.

Score function:

\[  
score(h,r,t)=h^Tdiag(r)t  
\]

Intuition:

Each relation scales the dimensions of entity embeddings.

Advantages:

* Fast  
* Simple

Limitation:

The score is symmetric:

ParentOf

≈

ChildOf

which is incorrect for asymmetric relations.

---

# **4\. ComplEx**

ComplEx extends DistMult using **complex-valued embeddings**.

Entities and relations are represented in the complex number space.

Advantages:

* Handles symmetric relations  
* Handles asymmetric relations  
* Better than DistMult for many KGs

Example:

ParentOf

≠

ChildOf

ComplEx can distinguish these.

---

# **Comparison of KG Embedding Models**

| Model | Main Idea | Strength | Limitation |
| ----- | ----- | ----- | ----- |
| TransE | (h+r\\approx t) | Simple, Fast | Poor for one-to-many & many-to-many |
| TransR | Separate entity and relation spaces | Better complex relations | More parameters |
| DistMult | Bilinear multiplication | Efficient | Cannot model asymmetric relations |
| ComplEx | Complex-valued embeddings | Handles asymmetric relations | More computationally expensive |

---

# **Applications of Knowledge Graph Embeddings**

### **Search Engines**

Google Knowledge Graph.

---

### **Recommendation Systems**

Netflix

Amazon

Spotify

---

### **Drug Discovery**

Predict unknown drug-disease interactions.

---

### **Question Answering**

Chatbots

Virtual Assistants

---

### **Fraud Detection**

Detect suspicious entity relationships.

---

### **Social Networks**

Friend recommendation.

---

# **Complete Pipeline**

Knowledge Graph  
      ↓  
Entities \+ Relations  
      ↓  
Knowledge Graph Embeddings  
      ↓  
R-GCN / R-GraphSAGE / R-GAT  
      ↓  
Node Embeddings  
      ↓  
Knowledge Graph Completion  
      ↓  
Predict Missing Triples

---

# **Interview Summary**

A **heterogeneous graph** contains **multiple types of nodes and relations**, making it suitable for modeling real-world systems such as e-commerce platforms, healthcare networks, and knowledge graphs. A **Knowledge Graph (KG)** represents facts as **triples (head, relation, tail)**, and **Knowledge Graph Embeddings (KGE)** map entities and relations into dense vectors for tasks such as **link prediction** and **knowledge graph completion**, where missing facts are inferred. Standard GNNs are extended to heterogeneous graphs through **Relational GCN (R-GCN)**, which learns **relation-specific weight matrices**, **R-GraphSAGE**, which combines neighbor sampling with relation-aware aggregation for scalability, and **R-GAT**, which applies **relation-aware attention** to weigh different edge types. Popular knowledge graph embedding models include **TransE**, which models relations as translations ((h+r\\approx t)); **TransR**, which projects entities into relation-specific spaces; **DistMult**, which uses bilinear multiplicative interactions but assumes symmetric relations; and **ComplEx**, which uses complex-valued embeddings to model both symmetric and asymmetric relations effectively. These techniques are widely used in search engines, recommendation systems, biomedical knowledge discovery, question answering, and fraud detection.

# graph ml 11

# **Reasoning in Knowledge Graphs using Embeddings**

A **Knowledge Graph (KG)** stores facts as triples:

(Head, Relation, Tail)

(h, r, t)

Example:

(Paris, CapitalOf, France)

(France, LocatedIn, Europe)

However, real-world Knowledge Graphs are **incomplete**.

Many facts are missing.

Instead of manually adding them, we use **Knowledge Graph Embeddings (KGE)** to automatically infer new facts.

This process is called:

Knowledge Graph Reasoning

---

# **What is Reasoning?**

Reasoning means:

**Inferring new facts from existing knowledge.**

Example:

Known facts:

(Paris, CapitalOf, France)

(France, LocatedIn, Europe)

Reasoning allows us to infer:

(Paris, LocatedIn, Europe)

Even though this triple was never explicitly stored.

---

# **Why Do We Need Reasoning?**

Knowledge Graphs are usually:

* Incomplete  
* Sparse  
* Missing relationships

Reasoning helps:

* Discover hidden knowledge  
* Answer complex queries  
* Recommend entities  
* Improve search engines

---

# **Embedding-Based Reasoning**

Instead of symbolic logic,

modern systems use embeddings.

Every entity becomes a vector.

Example:

Paris

↓

\[0.32,1.5,-0.7\]

Relation:

CapitalOf

↓

\[-0.2,0.8,1.1\]

Reasoning becomes a geometric problem in vector space.

---

# **Intuition**

Suppose:

Paris

\+

CapitalOf

≈

France

Now ask:

France

\+

LocatedIn

\=

?

The nearest embedding may be:

Europe

Thus,

the model predicts:

(France, LocatedIn, Europe)

---

# **Types of Reasoning**

---

# **1\. Link Prediction**

Predict missing entities.

Example:

(Paris, CapitalOf, ?)

Answer:

France

---

# **2\. Relation Prediction**

Predict missing relation.

Example:

(Paris, ?, France)

Answer:

CapitalOf

---

# **3\. Entity Prediction**

Predict missing head or tail.

Example:

(?, CapitalOf, France)

Answer:

Paris

---

# **Predictive Queries**

A **Predictive Query** asks the model to predict missing information instead of retrieving existing facts.

Example:

Database Query:

Who is the president of France?

Requires stored data.

---

Predictive Query:

Who is likely to become friends with Alice?

Requires inference.

---

# **Examples**

Social Network:

(User A

Friend

?)

Recommendation:

Suggest New Friend

---

Healthcare:

(Drug A

Treats

?)

Predict unknown disease.

---

E-commerce:

(User

Will Buy

?)

Predict future purchases.

---

# **Answering Predictive Queries**

Pipeline:

Knowledge Graph  
       ↓  
Learn Embeddings  
       ↓  
Encode Query  
       ↓  
Score Candidate Answers  
       ↓  
Choose Highest Score

---

# **Example**

Query:

(Paris, LocatedIn, ?)

Candidates:

Germany

Europe

Asia

Embedding model computes scores.

Highest score:

Europe

Predicted answer.

---

# **Scoring Functions**

Most embedding models compute:

Score(h,r,t)

Higher score:

More Likely Triple

Lower score:

Unlikely Triple

---

# **Multi-Hop Reasoning**

Some queries require multiple reasoning steps.

Example:

Known:

(Alice, LivesIn, Paris)

(Paris, CapitalOf, France)

Question:

Where does Alice live?

↓

France?

Need:

Alice

↓

Paris

↓

France

Multiple relations are combined.

---

# **Limitation of Traditional Embeddings**

Models like:

* TransE  
* DistMult  
* ComplEx

are excellent for:

Single-Hop Reasoning

But struggle with:

Complex Logical Queries

Example:

LivesIn

AND

WorksIn

---

# **Reasoning Using Box Embeddings**

A newer approach represents entities and queries as **boxes** instead of points.

Instead of:

Point Embedding

Use:

Hyper-Rectangle (Box)

---

# **Intuition**

Traditional embedding:

Paris

↓

•

Box embedding:

\+---------+  
|  Paris  |  
\+---------+

The box represents a **region** in embedding space.

---

# **Why Boxes?**

Logical concepts often represent **sets**, not individual points.

Example:

Concept:

European Cities

Contains:

* Paris  
* Berlin  
* Madrid

Instead of one point,

represent the entire set as a box.

---

# **Example**

Box:

European Cities

Contains:

Paris

Berlin

Rome

Another box:

Capital Cities

Contains:

Paris

London

Tokyo

Intersection:

European Capitals

Only:

Paris

Berlin

---

# **Why Box Embeddings are Powerful**

Boxes naturally support logical operations.

---

## **Intersection (AND)**

Query:

European

AND

Capital

Intersection of boxes.

---

## **Union (OR)**

Query:

Doctor

OR

Professor

Union of boxes.

---

## **Negation (NOT)**

Approximate complement.

Although harder than AND/OR,

modern methods support it.

---

# **Reasoning with Boxes**

Suppose:

Query:

Cities

↓

LocatedIn Europe

AND

Population \> 1 Million

Each condition is a box.

Intersection:

Valid Cities

Entities inside the final box satisfy the query.

---

# **Multi-Hop Queries with Boxes**

Example:

Known:

Alice

LivesIn

Paris

LocatedIn

France

MemberOf

EU

Question:

Which EU country does Alice live in?

Box embeddings combine all constraints geometrically.

---

# **Point Embeddings vs Box Embeddings**

| Point Embeddings | Box Embeddings |
| ----- | ----- |
| Entity → Point | Entity/Query → Region (Box) |
| Good for link prediction | Good for logical reasoning |
| Hard to represent sets | Naturally represent sets |
| Limited AND/OR reasoning | Excellent AND/OR reasoning |
| TransE, DistMult, ComplEx | Query2Box, BoxE, BetaE (extensions) |

---

# **Applications**

### **Search Engines**

Answer complex semantic queries.

---

### **Recommendation Systems**

Recommend items satisfying multiple conditions.

---

### **Drug Discovery**

Reason over:

Drug

↓

Protein

↓

Disease

relationships.

---

### **Knowledge Graph Completion**

Infer missing facts.

---

### **Question Answering**

Answer complex multi-hop questions.

---

### **Scientific Knowledge Graphs**

Infer hidden relationships.

---

# **Complete Pipeline**

Knowledge Graph  
      ↓  
Entity & Relation Embeddings  
      ↓  
Embedding Model  
(TransE / ComplEx / Box Embeddings)  
      ↓  
Logical Reasoning  
      ↓  
Predictive Queries  
      ↓  
Knowledge Graph Completion  
      ↓  
Missing Facts

---

# **Interview Summary**

**Reasoning in Knowledge Graphs** is the process of **inferring new facts from existing triples** rather than simply retrieving stored information. Modern approaches perform reasoning using **Knowledge Graph Embeddings (KGE)**, where entities and relations are mapped into a continuous vector space and missing facts are predicted by scoring candidate triples. This enables **predictive queries**, such as predicting missing entities, relations, or future links, which are essential for **knowledge graph completion**. Traditional embedding models such as **TransE**, **DistMult**, and **ComplEx** are effective for **single-hop reasoning** and link prediction but are limited when answering complex logical or multi-hop queries. To overcome this limitation, **Box Embeddings** represent entities or query results as **hyper-rectangles (boxes)** rather than points. Boxes naturally represent **sets of entities** and support logical operations such as **intersection (AND)**, **union (OR)**, and approximate **negation (NOT)**, making them highly effective for **multi-hop reasoning, predictive query answering, and complex knowledge graph completion**. These techniques are widely used in search engines, recommendation systems, question answering, biomedical knowledge graphs, and scientific discovery.

# graph ml 12

# **Fast Neural Subgraph Matching and Counting**

Subgraph matching is one of the most fundamental problems in graph mining.

The objective is:

**Find whether a small query graph exists inside a larger graph, and if so, identify all occurrences.**

Traditional algorithms are computationally expensive because the problem is **NP-complete**.

Recent **Neural Subgraph Matching** methods use Graph Neural Networks (GNNs) to learn graph representations, making matching much faster and scalable.

---

# **What is a Subgraph?**

A **subgraph** is a smaller graph formed from a subset of nodes and edges of a larger graph.

Example:

Large Graph

A \----- B \----- C  
|       |  
D \----- E

Possible subgraph

A \----- B  
|  
D

---

# **What is Subgraph Matching?**

Subgraph matching asks:

**Does a query graph exist inside a larger graph?**

Example

Query Graph

A \---- B  
|  
C

Target Graph

P \---- Q \---- R  
|  
S

If the structures match,

the algorithm returns:

Match Found

---

# **Applications**

* Fraud Detection  
* Chemical Compound Search  
* Protein Structure Matching  
* Social Network Analysis  
* Knowledge Graph Search  
* Cybersecurity

---

# **Why is Subgraph Matching Difficult?**

Suppose

Query Graph

5 Nodes

Target Graph

1 Million Nodes

The algorithm may need to compare millions of possible node combinations.

The problem belongs to the class of **NP-complete** problems.

---

# **Traditional Subgraph Matching**

Classical algorithms include:

* Ullmann Algorithm  
* VF2 Algorithm  
* TurboISO  
* GraphQL

These methods perform:

Graph Traversal

\+

Backtracking

\+

Constraint Checking

Advantages:

* Exact matching

Disadvantages:

* Slow on large graphs  
* High computational cost

---

# **Neural Subgraph Matching**

Neural methods use **Graph Neural Networks** to learn embeddings of graphs.

Instead of comparing graphs node by node,

they compare learned embeddings.

Pipeline:

Query Graph  
      ↓  
GNN Encoder  
      ↓  
Embedding

Target Graph  
      ↓  
GNN Encoder  
      ↓  
Embedding

↓

Similarity Score

↓

Match Prediction

---

# **Intuition**

Suppose two graphs are structurally similar.

Traditional method:

Compare Every Node

Neural method:

Compare Embeddings

Embedding comparison is much faster.

---

# **Fast Neural Subgraph Matching**

Modern approaches improve speed using:

### **Graph Embeddings**

Convert graphs into vectors.

---

### **Node Embeddings**

Compare only candidate nodes.

---

### **Attention Mechanisms**

Focus on important nodes.

---

### **Neighborhood Sampling**

Avoid exploring the whole graph.

---

### **Approximate Matching**

Instead of exact matching,

find the most similar subgraph.

---

# **Workflow**

Query Graph  
      ↓  
Graph Encoder (GNN)  
      ↓  
Query Embedding  
      ↓  
Target Graph Encoder  
      ↓  
Node/Graph Embeddings  
      ↓  
Similarity Search  
      ↓  
Top Matching Subgraphs

---

# **Subgraph Counting**

Sometimes we don't want to know **where** the subgraph exists.

Instead, we ask:

**How many times does it appear?**

---

# **Example**

Graph

A \---- B \---- C  
 \\    /  
   D

Query

Triangle

Result

Triangle Count \= 1

---

# **Applications**

* Social Network Analysis  
* Fraud Detection  
* Biological Networks  
* Motif Discovery  
* Chemistry

---

# **Neural Subgraph Counting**

Instead of enumerating every subgraph,

GNNs estimate counts using learned graph representations.

Advantages:

* Faster  
* Scalable  
* Suitable for large graphs

---

# **Finding Frequent Subgraphs**

A **Frequent Subgraph** is a subgraph that appears repeatedly across one or more graphs.

---

# **Example**

Suppose we have 100 molecular graphs.

If a chemical structure appears in:

75 Molecules

it is considered a frequent subgraph.

---

# **Why Find Frequent Subgraphs?**

Frequent patterns often reveal:

* Common molecular structures  
* Repeated fraud patterns  
* Frequently purchased product combinations  
* Common social interaction patterns

---

# **Workflow**

Graph Dataset  
      ↓  
Extract Candidate Subgraphs  
      ↓  
Count Frequency  
      ↓  
Keep Frequent Ones

---

# **Traditional Algorithms**

Popular methods:

* gSpan  
* FSG  
* AGM  
* Gaston

These enumerate candidate subgraphs and prune infrequent ones using support thresholds.

---

# **Neural Approaches**

GNNs learn embeddings of subgraphs.

Similar subgraphs cluster together,

making frequent pattern discovery much faster.

---

# **Community Detection in Networks**

A **Community** is a group of nodes that are **more densely connected to each other than to the rest of the graph**.

---

# **Intuition**

Social Network

Students

↓

Friends

↓

Same College

These students form one community.

---

# **Example**

A \----- B  
|       |  
C \----- D

E \----- F  
|       |  
G \----- H

Two communities:

Community 1

A B C D

Community 2

E F G H

---

# **Why Detect Communities?**

Communities help identify:

* Friend Groups  
* Criminal Networks  
* Protein Modules  
* Customer Segments  
* Research Collaborations

---

# **Applications**

## **Social Networks**

Friend recommendation

Interest groups

---

## **Banking**

Fraud rings

Money laundering networks

---

## **Biology**

Protein interaction modules

Gene networks

---

## **Marketing**

Customer segmentation

---

## **Cybersecurity**

Attack groups

Bot networks

---

# **Traditional Community Detection Algorithms**

### **1\. Louvain Algorithm**

Optimizes **modularity** to find communities.

Fast and widely used for large networks.

---

### **2\. Label Propagation**

Nodes adopt the majority label among neighbors.

Simple and scalable.

---

### **3\. Spectral Clustering**

Uses eigenvectors of the graph Laplacian to partition the graph.

Suitable for medium-sized graphs.

---

### **4\. Girvan-Newman Algorithm**

Removes edges with the highest **edge betweenness centrality** to reveal communities.

Produces hierarchical community structures but is computationally expensive.

---

# **GNN-based Community Detection**

Instead of relying only on graph topology,

GNNs learn node embeddings.

Pipeline:

Graph  
      ↓  
GNN  
      ↓  
Node Embeddings  
      ↓  
Clustering  
(K-Means / Spectral)  
      ↓  
Communities

Nodes with similar embeddings are grouped into the same community.

---

# **Advantages of Neural Community Detection**

* Learns both node features and graph structure  
* Handles attributed and heterogeneous graphs  
* More robust on noisy networks  
* Scales better to very large graphs with sampling techniques

---

# **Comparison**

| Task | Goal | Output |
| ----- | ----- | ----- |
| Subgraph Matching | Find a query graph inside a larger graph | Matching subgraph(s) |
| Neural Subgraph Matching | Use GNN embeddings for efficient matching | Similar subgraphs |
| Subgraph Counting | Count occurrences of a query subgraph | Frequency count |
| Frequent Subgraph Mining | Discover recurring patterns | Frequent subgraphs |
| Community Detection | Group densely connected nodes | Communities |

---

# **Real-World Applications**

| Domain | Application |
| ----- | ----- |
| Google Search | Knowledge graph pattern matching |
| Facebook | Community detection, friend recommendation |
| Banking | Fraud ring detection, transaction pattern mining |
| Healthcare | Protein motif discovery, drug design |
| Cybersecurity | Attack pattern matching, malware family detection |
| E-commerce | Customer segmentation, recommendation systems |

---

# **Interview Summary**

**Subgraph matching** is the task of determining whether a **query graph** exists within a larger graph, while **subgraph counting** determines how many times a given subgraph appears. Traditional algorithms such as **VF2**, **Ullmann**, and **gSpan** provide exact solutions but become computationally expensive on large graphs because subgraph isomorphism is **NP-complete**. **Neural subgraph matching** addresses this challenge by using **Graph Neural Networks (GNNs)** to learn node and graph embeddings, allowing similarity comparisons in embedding space instead of exhaustive structural matching. This significantly improves scalability and supports approximate matching. **Frequent subgraph mining** identifies recurring graph patterns that occur above a specified support threshold and is widely used in chemistry, fraud detection, and pattern discovery. **Community detection** partitions a graph into groups of densely connected nodes, enabling applications such as social network analysis, customer segmentation, fraud ring detection, and biological network analysis. Classical community detection methods include **Louvain**, **Label Propagation**, **Spectral Clustering**, and **Girvan-Newman**, while modern GNN-based approaches learn node embeddings and cluster them to discover communities more effectively.

# graph ml 13

# **Community Detection in Networks**

**Community Detection** is the process of identifying groups of nodes that are **more densely connected to each other than to the rest of the network**.

A community is also called a:

* Cluster  
* Module  
* Group

The objective is:

**Find naturally occurring groups in a network without predefined labels.**

---

# **Intuition**

Consider a social network.

Alice \---- Bob  
 |         |  
Charlie \--- David

Emma \---- Frank  
 |         |  
Grace \---- Henry

Clearly,

Alice, Bob, Charlie, David

form one community,

while

Emma, Frank, Grace, Henry

form another.

Nodes inside a community interact more frequently than with nodes outside the community.

---

# **Why Community Detection?**

Real-world networks naturally form communities.

Examples:

### **Social Networks**

* Friend groups  
* College classmates  
* Families

---

### **Banking**

* Fraud rings  
* Money laundering groups

---

### **Biology**

* Protein interaction modules  
* Gene regulatory networks

---

### **E-commerce**

* Customer segments  
* Product groups

---

# **Network Communities**

A **Network Community** is a subset of nodes satisfying:

Strong Internal Connections

Weak External Connections

Mathematically,

nodes inside a community have:

* High edge density  
* Frequent interactions

compared to nodes outside the community.

---

# **Characteristics of Good Communities**

A good community has:

* Many internal edges  
* Few external edges  
* Strong structural similarity  
* High connectivity

---

# **Example**

A \----- B  
|       |  
C \----- D

      E \----- F  
      |       |  
      G \----- H

Communities:

Community 1

A B C D

Community 2

E F G H

---

# **Granovetter's Theory (Strength of Weak Ties)**

One of the most influential theories in network science was proposed by **Mark Granovetter**.

It is called:

Strength of Weak Ties

---

# **Main Idea**

**Weak connections between communities are often more valuable than strong connections within communities.**

---

# **Intuition**

Suppose:

Family

↓

Strong Relationships

Everyone already knows similar information.

Now imagine:

Friend from Another Company

This is a **weak tie**.

That friend may provide:

* New job opportunities  
* New ideas  
* New information

Weak ties connect different communities.

---

# **Example**

A \----- B \----- C  
|       |  
D       E \----- F

Inside:

A B C D

Strong connections.

Between:

B

↓

E

Weak connection.

This single edge connects two communities.

---

# **Importance of Weak Ties**

Weak ties help in:

* Information diffusion  
* Disease spread modeling  
* Viral marketing  
* Friend recommendations  
* Knowledge transfer

---

# **Strong vs Weak Ties**

| Strong Tie | Weak Tie |
| ----- | ----- |
| Family | Acquaintance |
| Close Friends | Colleagues |
| Frequent Interaction | Rare Interaction |
| Within Community | Between Communities |

---

# **Modularity**

Modularity is the most common metric for evaluating community quality.

It measures:

**How well a network is divided into communities.**

---

# **Intuition**

Suppose we randomly divide a graph.

Communities may look poor.

If nodes inside communities have significantly more connections than expected by chance,

modularity becomes high.

---

# **Modularity Formula**

## **\[**

## **Q=**

## **\\frac{1}{2m}**

## **\\sum\_{ij}**

## **\\left(**

## **A\_{ij}**

\\frac{k\_i k\_j}{2m}  
\\right)  
\\delta(c\_i,c\_j)  
\]

Where:

* (A\_{ij}) \= Edge exists between i and j  
* (k\_i) \= Degree of node i  
* (m) \= Total number of edges  
* (\\delta) \= 1 if both nodes belong to the same community

---

# **Interpretation**

Modularity ranges approximately from:

\-1 to 1

Higher values indicate stronger community structure.

Typical interpretation:

Q ≈ 0

↓

Weak Communities

Q \> 0.3

↓

Good Communities

Q \> 0.7

↓

Very Strong Communities

---

# **Louvain Algorithm**

The **Louvain Algorithm** is one of the most popular algorithms for community detection.

Goal:

Maximize Modularity

It is:

* Fast  
* Scalable  
* Widely used

---

# **Intuition**

Initially,

every node forms its own community.

Example:

A

B

C

D

Each node:

Own Community

---

Then,

nodes are moved to neighboring communities

only if modularity increases.

---

# **Louvain Algorithm Steps**

---

## **Step 1**

Initialize.

Every node is its own community.

---

## **Step 2**

For each node,

evaluate moving it to each neighboring community.

Choose the move that gives the largest modularity gain.

---

## **Step 3**

Repeat until no move increases modularity.

---

## **Step 4**

Collapse each detected community into a single "super-node."

Example:

Community A

↓

Single Node

---

## **Step 5**

Repeat the process on the new graph.

Continue until modularity no longer improves.

---

# **Louvain Workflow**

Graph  
      ↓  
Each Node \= Community  
      ↓  
Move Nodes  
      ↓  
Increase Modularity  
      ↓  
Merge Communities  
      ↓  
Build New Graph  
      ↓  
Repeat

---

# **Advantages of Louvain**

* Very fast  
* Handles millions of nodes  
* Produces hierarchical communities  
* Maximizes modularity automatically

---

# **Limitation**

Louvain assumes:

One Node

↓

One Community

Real-world networks often violate this assumption.

---

# **Overlapping Communities**

In reality,

people belong to multiple communities.

Example:

Student

↓

College Friends

↓

Family

↓

Football Team

↓

Office

One person belongs to multiple groups simultaneously.

These are called:

Overlapping Communities

---

# **Why Overlapping Communities?**

Examples:

Social Networks

Person

↓

Family

\+

Office

\+

Friends

Protein Networks

One protein participates in multiple biological processes.

---

# **BigCLAM**

**BigCLAM** stands for:

Cluster Affiliation Model for Big Networks

It is one of the most popular algorithms for detecting overlapping communities.

---

# **Intuition**

Instead of assigning:

One Node

↓

One Community

BigCLAM learns:

Node

↓

Membership Scores

↓

Multiple Communities

---

# **Example**

Person A

Membership:

Family

0.9

Office

0.8

Football Club

0.6

Person belongs to all three communities.

---

# **BigCLAM Model**

Every node learns a **community affiliation vector**.

Example:

Node A

↓

\[0.8,0.1,0.7\]

Each dimension corresponds to one community.

Higher values indicate stronger membership.

---

# **Edge Probability**

Nodes sharing stronger community memberships are more likely to be connected.

Thus,

BigCLAM predicts edges based on shared affiliations.

---

# **BigCLAM Workflow**

Network  
      ↓  
Initialize Community Scores  
      ↓  
Learn Membership Strengths  
      ↓  
Optimize Likelihood  
      ↓  
Detect Multiple Communities

---

# **Advantages of BigCLAM**

* Detects overlapping communities  
* Scalable to large graphs  
* Suitable for social networks  
* Handles multiple memberships naturally

---

# **Louvain vs BigCLAM**

| Feature | Louvain | BigCLAM |
| ----- | ----- | ----- |
| Community Type | Non-overlapping | Overlapping |
| Objective | Maximize Modularity | Learn Community Memberships |
| Output | One community per node | Multiple communities per node |
| Scalability | Very High | High |
| Common Applications | General clustering | Social networks, biological networks |

---

# **Applications**

### **Social Networks**

* Friend group detection  
* Interest communities  
* Influencer analysis

---

### **Banking**

* Fraud rings  
* Money laundering detection

---

### **Biology**

* Protein complexes  
* Gene modules

---

### **Recommendation Systems**

* User communities  
* Product communities

---

### **Cybersecurity**

* Botnet detection  
* Attack groups

---

# **Complete Community Detection Pipeline**

Network  
     ↓  
Node & Edge Analysis  
     ↓  
Community Detection  
(Louvain / BigCLAM)  
     ↓  
Community Assignment  
     ↓  
Evaluate using Modularity  
     ↓  
Applications

---

# **Interview Summary**

**Community Detection** is the task of identifying groups of densely connected nodes within a network. A **network community** contains many internal connections and relatively few external ones. **Granovetter's Strength of Weak Ties Theory** states that **weak ties**, which connect different communities, play a crucial role in spreading information and connecting otherwise separate groups. The quality of a community partition is commonly measured using **modularity**, which compares the density of edges within communities against what would be expected in a random network. The **Louvain Algorithm** is a fast, hierarchical method that detects **non-overlapping communities** by iteratively maximizing modularity and merging communities into super-nodes. However, many real-world networks have **overlapping communities**, where a node belongs to multiple groups simultaneously. **BigCLAM (Cluster Affiliation Model for Big Networks)** addresses this by learning a **community affiliation vector** for each node, allowing multiple community memberships. Louvain is ideal when each node belongs to a single community, while BigCLAM is preferred for networks such as social or biological systems where overlapping memberships are common.

# graph ml 14

# **Generative Models for Graphs**

A **Graph Generative Model** is a model that **learns the structure of existing graphs and generates new graphs with similar properties**.

Instead of predicting labels or links, these models answer:

**How can we generate realistic networks that resemble real-world graphs?**

Examples:

* Generate synthetic social networks  
* Simulate internet topology  
* Create molecular structures  
* Generate protein interaction networks

---

# **Why Do We Need Graph Generative Models?**

Real-world graphs are often:

* Huge  
* Sensitive (privacy issues)  
* Expensive to collect

Instead of sharing real data,

we generate synthetic graphs that preserve important network properties.

---

# **Applications**

* Social Network Simulation  
* Drug Discovery  
* Recommendation Systems  
* Cybersecurity  
* Benchmarking Graph Algorithms  
* Privacy-Preserving Data Generation

---

# **Types of Graph Generative Models**

Classical Models:

* Erdős-Rényi Random Graph  
* Small World Model  
* Barabási-Albert Model  
* Kronecker Graph Model

Deep Learning Models:

* GraphVAE  
* GraphGAN  
* GraphRNN  
* Graph Diffusion Models

---

# **Key Network Properties**

A good graph generator should preserve important characteristics of real-world networks.

---

## **1\. Degree Distribution**

Degree \= Number of neighbors of a node.

Example:

A

↓

Degree \= 4

Real social networks often follow a **power-law distribution**:

Few nodes have many connections (hubs), while most have only a few.

---

## **2\. Clustering Coefficient**

Measures how well neighbors of a node are connected.

Example:

A

/ \\

B---C

Neighbors of A are also connected.

High clustering is common in social networks.

---

## **3\. Average Path Length**

The average number of hops required to travel between two nodes.

Small values imply efficient communication.

---

## **4\. Connected Components**

The number of disconnected groups in the graph.

Many real networks contain one **giant connected component** plus a few isolated groups.

---

## **5\. Diameter**

The maximum shortest-path distance between any two nodes.

---

## **6\. Community Structure**

Real graphs naturally contain densely connected communities.

---

## **7\. Power-Law Degree Distribution**

Many real networks follow:

\[  
P(k)\\propto k^{-\\gamma}  
\]

Meaning:

Few Hub Nodes

Many Low-Degree Nodes

---

# **Erdős-Rényi Random Graph (G(n,p))**

One of the earliest graph generation models.

Developed by:

* Paul Erdős  
* Alfréd Rényi

---

# **Intuition**

Suppose we have:

n Nodes

Every possible pair of nodes has:

Probability p

of being connected.

Every edge is generated independently.

---

# **Example**

Suppose:

n \= 5

p \= 0.4

Each possible edge has a 40% chance of existing.

Result:

A random graph.

---

# **G(n,p) Model**

Notation:

\[  
G(n,p)  
\]

Where:

* (n) \= Number of nodes  
* (p) \= Probability of edge creation

---

# **Properties**

Expected number of edges:

\[  
E=\\frac{n(n-1)}{2}p  
\]

Expected degree:

\[  
(n-1)p  
\]

---

# **Advantages**

* Very simple  
* Easy to analyze mathematically  
* Fast to generate

---

# **Limitations**

Real networks do **not** behave randomly.

Problems:

* Low clustering  
* No hubs  
* Degree distribution follows a binomial/Poisson distribution instead of a power law  
* Weak community structure

---

# **Real Networks vs G(n,p)**

| Property | Real Networks | Erdős-Rényi |
| ----- | ----- | ----- |
| Degree Distribution | Power Law | Binomial / Poisson |
| Hubs | Yes | Rare |
| Clustering | High | Low |
| Communities | Strong | Weak |
| Average Path Length | Small | Small |
| Realism | High | Low |

---

# **The Small World Model**

Proposed by:

**Watts and Strogatz (1998)**

Goal:

Create graphs having:

* High clustering  
* Small average path length

---

# **Motivation**

Real-world networks exhibit:

Friends of Friends

↓

Usually Friends

and yet

Any Person

↓

Few Hops Away

This is known as the **Small World Phenomenon** or **"Six Degrees of Separation."**

---

# **Construction**

### **Step 1**

Create a regular ring lattice.

Example:

Each node connected to its 4 nearest neighbors.

---

### **Step 2**

Randomly rewire each edge with probability:

β

---

### **Step 3**

Some long-range shortcuts appear.

Result:

High Clustering

\+

Short Paths

---

# **Properties**

* High clustering coefficient  
* Small average path length  
* Better reflects social networks than Erdős-Rényi

---

# **Applications**

* Social Networks  
* Brain Networks  
* Transportation Networks  
* Communication Systems

---

# **Kronecker Graph Generative Model**

The **Kronecker Graph Model** is designed to generate **large realistic networks** while preserving important structural properties.

Instead of randomly adding edges,

it repeatedly expands a small graph using the **Kronecker Product**.

---

# **Intuition**

Start with a tiny graph.

Example:

2 Nodes

Repeatedly duplicate and connect it.

After several iterations:

Tiny Graph

↓

Medium Graph

↓

Large Graph

↓

Realistic Network

---

# **Kronecker Product**

Suppose the initiator matrix is

\[  
K\_1=  
\\begin{bmatrix}  
0.9 & 0.5\\  
0.5 & 0.2  
\\end{bmatrix}  
\]

Each iteration computes the **Kronecker product** of the matrix with itself.

This produces progressively larger probability matrices from which edges are sampled.

---

# **Advantages**

* Generates massive graphs efficiently  
* Preserves self-similar (fractal-like) structure  
* Produces realistic degree distributions  
* Captures densification and shrinking diameter observed in many evolving networks

---

# **Applications**

* Internet topology modeling  
* Citation networks  
* Social media simulation  
* Biological networks

---

# **Comparison of Graph Generative Models**

| Model | Main Idea | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| Erdős-Rényi | Random edge creation | Simple, mathematically tractable | Unrealistic for many real networks |
| Small World (Watts-Strogatz) | Rewire a regular lattice | High clustering, short paths | Does not naturally produce hubs |
| Kronecker Graph | Recursive graph expansion | Scalable, preserves many real network properties | More complex to fit parameters |

---

# **Real Networks vs Classical Models**

| Property | Real Networks | Erdős-Rényi | Small World | Kronecker |
| ----- | ----- | ----- | ----- | ----- |
| Power-law Degree Distribution | ✔ | ✖ | ✖ | ✔ |
| High Clustering | ✔ | ✖ | ✔ | ✔ (approx.) |
| Short Average Path | ✔ | ✔ | ✔ | ✔ |
| Community Structure | ✔ | Weak | Moderate | Better |
| Hubs | ✔ | ✖ | Limited | ✔ |
| Scalability | High | High | High | Very High |

---

# **Modern Deep Generative Models for Graphs**

Recent approaches use deep learning to generate graphs.

Examples:

* **GraphVAE** – Variational Autoencoder for graphs  
* **GraphRNN** – Sequential graph generation  
* **GraphGAN** – Adversarial graph generation  
* **Graph Diffusion Models** – State-of-the-art graph generation using diffusion processes

These models learn directly from graph data and can generate realistic molecules, social networks, and knowledge graphs.

---

# **Complete Workflow**

Real Network  
      ↓  
Learn Network Properties  
      ↓  
Choose Generative Model  
(Erdős-Rényi / Small World / Kronecker / Deep Models)  
      ↓  
Generate New Graph  
      ↓  
Evaluate  
(Degree Distribution,  
Clustering,  
Path Length,  
Communities)

---

# **Interview Summary**

A **Graph Generative Model** learns the structural characteristics of existing graphs and generates **synthetic graphs** with similar properties. Important network characteristics include the **degree distribution**, **clustering coefficient**, **average path length**, **diameter**, **connected components**, and **community structure**. The **Erdős-Rényi Random Graph (G(n,p))** creates edges independently with probability (p), making it mathematically simple but unrealistic because it lacks hubs, strong clustering, and power-law degree distributions. The **Watts-Strogatz Small World Model** improves realism by rewiring a regular lattice to produce networks with **high clustering** and **short path lengths**, capturing the small-world phenomenon observed in social and biological networks. The **Kronecker Graph Generative Model** recursively expands a small initiator graph using the **Kronecker product**, producing scalable synthetic graphs that preserve many properties of real evolving networks, including heavy-tailed degree distributions and self-similar structure. Modern graph generation increasingly relies on deep learning models such as **GraphVAE**, **GraphRNN**, **GraphGAN**, and **Graph Diffusion Models**, which can learn complex graph distributions directly from data.

# graph ml 15

# **Deep Generative Models for Graphs**

Classical graph generation models (Erdős-Rényi, Small World, Kronecker) rely on **hand-designed rules**.

Deep Generative Models learn these rules **automatically from data**.

Instead of specifying how graphs should be generated,

the model learns:

**The probability distribution of real graphs and generates new graphs that resemble them.**

---

# **Why Deep Generative Models?**

Real-world graphs are highly complex.

They contain:

* Communities  
* Hubs  
* Rich node attributes  
* Different edge types  
* Complex dependencies

Classical models cannot capture all these characteristics.

Deep learning models can.

---

# **Applications**

* Molecule Generation  
* Drug Discovery  
* Protein Design  
* Social Network Simulation  
* Circuit Design  
* Knowledge Graph Generation

---

# **Types of Deep Graph Generative Models**

The main categories are:

Auto-Regressive Models

Variational Autoencoders (GraphVAE)

Generative Adversarial Networks (GraphGAN)

Diffusion Models

Among these,

Auto-Regressive models are the easiest to understand.

---

# **Auto-Regressive Models**

An **Auto-Regressive (AR) Model** generates a graph **one step at a time**, conditioning each step on everything generated so far.

General idea:

**Generate the next node or edge based on the current partial graph.**

---

# **Intuition**

Imagine drawing a graph.

You don't draw everything simultaneously.

Instead:

Start

↓

Add Node

↓

Connect Edge

↓

Add Another Node

↓

Connect More Edges

↓

Repeat

The graph grows sequentially.

---

# **Auto-Regressive Generation**

Instead of predicting:

Entire Graph

Predict:

Next Node

or

Next Edge

The probability is factorized as:

\[  
P(G)=\\prod\_i P(G\_i|G\_{\<i})  
\]

Meaning:

Each generation step depends on previous steps.

---

# **Advantages**

* Produces valid graphs  
* Captures sequential dependencies  
* Flexible generation

---

# **Limitations**

* Slow generation  
* Difficult for very large graphs  
* Sensitive to node ordering

---

# **GraphRNN**

**GraphRNN** is one of the most influential autoregressive graph generation models.

It generates graphs:

Node-by-Node

Edge-by-Edge

using Recurrent Neural Networks (RNNs).

---

# **Intuition**

Instead of generating the whole graph,

GraphRNN asks:

### **Step 1**

Should we create a new node?

↓

Yes

---

### **Step 2**

Which previous nodes should this new node connect to?

↓

Predict edges

---

### **Step 3**

Repeat.

---

# **GraphRNN Architecture**

GraphRNN uses **two RNNs**.

---

## **1\. Node-Level RNN**

Responsible for:

Generate New Node

---

## **2\. Edge-Level RNN**

Responsible for:

Predict Edges

for that Node

---

# **GraphRNN Workflow**

Empty Graph  
      ↓  
Node RNN  
      ↓  
Generate Node  
      ↓  
Edge RNN  
      ↓  
Predict Connections  
      ↓  
Updated Graph  
      ↓  
Repeat

---

# **Example**

Initially

No Nodes

↓

Generate:

Node A

↓

Generate:

Node B

↓

Predict:

A \----- B

↓

Generate:

Node C

↓

Predict:

A \----- C

B \----- C

Continue until the graph is complete.

---

# **Why GraphRNN Works Well**

Instead of learning an enormous graph distribution,

it learns many small decisions:

* Add node?  
* Connect edge?  
* Stop generation?

This makes training easier.

---

# **Generating Realistic Graphs**

A generated graph should resemble the real dataset.

It should preserve:

* Degree distribution  
* Clustering coefficient  
* Community structure  
* Shortest path distribution  
* Number of connected components  
* Motif frequencies

---

# **Training Deep Graph Generative Models**

Pipeline:

Graph Dataset  
      ↓  
Graph Encoder  
      ↓  
Learn Graph Distribution  
      ↓  
Train Neural Generator  
      ↓  
Generate New Graphs

---

# **Challenges**

Graph generation is difficult because:

* Graphs have variable sizes  
* Variable numbers of edges  
* Different node orderings  
* Complex dependencies  
* Large search space

---

# **Scaling Up Graph Generation**

Large graphs may contain:

Millions of Nodes

Generating every node sequentially becomes slow.

Modern methods improve scalability using:

---

## **1\. Hierarchical Generation**

Generate communities first.

Then generate nodes inside communities.

---

## **2\. Parallel Generation**

Generate multiple nodes simultaneously.

Common in diffusion models.

---

## **3\. Graph Partitioning**

Split large graphs into smaller subgraphs.

Generate each independently.

---

## **4\. Sparse Generation**

Generate only likely edges.

Ignore improbable ones.

---

# **Evaluating Graph Generation**

After generating graphs,

we compare them with real graphs.

---

# **Structural Metrics**

## **Degree Distribution**

Generated graph should have similar degree statistics.

---

## **Clustering Coefficient**

Measures local connectivity.

---

## **Average Path Length**

Generated graphs should preserve small-world behavior.

---

## **Graph Diameter**

Maximum shortest-path distance.

---

## **Connected Components**

Should match the real graph.

---

## **Community Structure**

Generated communities should resemble real communities.

---

## **Motif Distribution**

Small recurring patterns (e.g., triangles, stars) should appear with similar frequencies.

---

# **Statistical Evaluation**

Common measures include:

* Maximum Mean Discrepancy (MMD)  
* Precision and Recall for Graphs  
* Validity  
* Novelty  
* Uniqueness

---

# **Applications to Molecule Generation**

One of the biggest successes of graph generation is in **drug discovery**.

---

# **Why Molecules are Graphs**

A molecule naturally forms a graph.

Nodes:

Atoms

Edges:

Chemical Bonds

Example:

Carbon

↓

Connected to

↓

Hydrogen

---

# **Goal**

Instead of predicting properties,

generate entirely new molecules.

---

# **Pipeline**

Known Molecules  
      ↓  
Train Graph Generator  
      ↓  
Learn Chemical Rules  
      ↓  
Generate New Molecules  
      ↓  
Test Drug Properties

---

# **Constraints**

Generated molecules must satisfy:

* Valid chemical bonds  
* Correct atom valency  
* Chemical stability  
* Synthesizability  
* Desired biological activity

---

# **Example**

Training molecules:

Aspirin

Ibuprofen

Paracetamol

Model learns common chemical patterns.

Then generates:

Completely New Molecule

with similar medicinal properties.

---

# **Applications**

### **Drug Discovery**

Design new medicines.

---

### **Material Science**

Generate new polymers and materials.

---

### **Protein Engineering**

Design proteins with desired functions.

---

### **Chemistry**

Generate catalysts and reactions.

---

### **Battery Research**

Design better electrolytes.

---

# **Deep Graph Generative Models Comparison**

| Model | Main Idea | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| GraphRNN | Sequential node and edge generation | Flexible, realistic | Slow for very large graphs |
| GraphVAE | Encode and decode entire graphs | Fast inference | Limited graph size |
| GraphGAN | Adversarial graph generation | Produces realistic graphs | Training instability |
| Graph Diffusion Models | Gradually denoise random graphs | State-of-the-art quality | Computationally expensive |

---

# **Complete Workflow**

Training Graphs  
      ↓  
Graph Encoder  
      ↓  
Learn Graph Distribution  
      ↓  
Deep Graph Generator  
(GraphRNN / GraphVAE / GraphGAN / Diffusion)  
      ↓  
Generate New Graphs  
      ↓  
Evaluate  
(Degree Distribution,  
Clustering,  
Communities,  
Motifs,  
Validity)  
      ↓  
Applications  
(Molecule Generation,  
Drug Discovery,  
Network Simulation)

---

# **Interview Summary**

**Deep Generative Models for Graphs** learn the probability distribution of real-world graphs and generate new graphs with similar structural properties. Unlike classical graph generators, they automatically capture complex characteristics such as **communities, hubs, motifs, and node attributes**. **Auto-regressive models** generate graphs sequentially by predicting the next node or edge based on the partially generated graph. **GraphRNN** is a landmark autoregressive model that uses a **Node RNN** to generate new nodes and an **Edge RNN** to predict connections for each node, producing realistic graphs one step at a time. The quality of generated graphs is evaluated using structural metrics such as **degree distribution, clustering coefficient, average path length, diameter, connected components, community structure, and motif frequencies**, along with statistical measures like **Maximum Mean Discrepancy (MMD)**, **validity**, **novelty**, and **uniqueness**. Scaling graph generation to large networks is achieved through **hierarchical generation, graph partitioning, sparse edge generation, and parallel generation**. One of the most important applications is **molecule generation**, where atoms are represented as nodes and chemical bonds as edges. Deep graph generative models learn chemical structures from existing molecules and generate novel, chemically valid molecules for **drug discovery, material science, protein engineering, and catalyst design**, significantly accelerating scientific research.

# graph ml 16

# **Limitations of Graph Neural Networks (GNNs)**

Although Graph Neural Networks (GNNs) have achieved remarkable success, they also face several theoretical and practical limitations.

Understanding these limitations has led to newer architectures such as **Position-Aware GNNs**, **Identity-Aware GNNs**, and **Robust GNNs**.

---

# **Major Limitations of GNNs**

---

## **1\. Over-Smoothing**

The most common limitation.

As more GNN layers are stacked,

node embeddings become increasingly similar.

Eventually,

All Nodes

↓

Nearly Identical Embeddings

The model can no longer distinguish between different nodes.

---

### **Intuition**

Initially:

Node A \= Engineer

Node B \= Doctor

Node C \= Student

After many message-passing layers,

neighbor information keeps getting averaged.

Eventually:

A ≈ B ≈ C

Unique information disappears.

---

## **2\. Over-Squashing**

Information from many distant nodes must pass through a few intermediate nodes.

This compresses a large amount of information into small embeddings.

---

### **Example**

Many Nodes

↓

One Bottleneck Node

↓

Target Node

Important information is lost during aggregation.

---

## **3\. Limited Expressiveness**

Most Message Passing GNNs are at most as expressive as the **1-Weisfeiler-Lehman (1-WL) Test**.

They cannot distinguish certain graph structures.

Example:

Two different graphs may receive identical embeddings.

---

## **4\. Scalability**

Large graphs may contain:

Millions of Nodes

Aggregating all neighbors becomes computationally expensive.

Solutions:

* Neighbor Sampling  
* GraphSAGE  
* Mini-batching

---

## **5\. Dynamic Graphs**

Many real networks change continuously.

Example:

* Social networks  
* Financial transactions  
* Communication networks

Traditional GNNs assume static graphs.

---

## **6\. Heterogeneous Graphs**

Standard GNNs assume one node type and one edge type.

Real-world graphs often contain multiple node and relation types.

Extensions:

* R-GCN  
* R-GAT  
* Heterogeneous GNNs

---

## **7\. Sensitivity to Noise**

Incorrect edges or node features can significantly affect predictions because information propagates through the graph.

---

# **Position-Aware GNNs**

One limitation of standard GNNs is that they know **who is connected**, but not **where a node is located within the graph**.

---

# **Why Position Matters**

Consider two graphs:

A \----- B \----- C

and

X \----- Y \----- Z

Suppose B and Y have identical features.

A standard GNN may generate the same embedding,

even if their structural positions differ in a larger graph.

---

# **Intuition**

Imagine two employees.

Both have:

* Same age  
* Same salary  
* Same department

However,

one is the **CEO**,

the other is an **intern**.

Their position within the organization matters.

---

# **How Position-Aware GNNs Work**

They augment node features with **positional information**.

Examples include:

* Shortest-path distances  
* Graph Laplacian eigenvectors (Laplacian Positional Encoding)  
* Random walk positional encodings  
* Centrality measures

Updated node features become:

Node Features

\+

Position Encoding

---

# **Benefits**

* Distinguishes structurally similar nodes  
* Improves graph-level learning  
* Essential for Graph Transformers

---

# **Applications**

* Molecule generation  
* Graph Transformers  
* Traffic networks  
* Knowledge graphs

---

# **Identity-Aware GNNs**

Standard GNNs often fail to uniquely identify nodes with similar neighborhoods.

Identity-aware GNNs explicitly encode **node identity**.

---

# **Problem**

Suppose:

Node A

Neighbors:

B

C

and

Node X

Neighbors:

Y

Z

If all node features are identical,

standard GNNs may produce identical embeddings.

---

# **Solution**

Assign each node an identity representation.

Example:

Node A

↓

ID Embedding

or use structural identifiers such as:

* One-hot node IDs (small graphs)  
* Learnable ID embeddings  
* Hash encodings  
* Structural role encodings

The model learns:

Node Features

\+

Identity Features

---

# **Advantages**

* Better node discrimination  
* Improved expressiveness  
* Distinguishes symmetric structures  
* Useful for link prediction

---

# **Applications**

* Social networks  
* Fraud detection  
* Knowledge graphs  
* Recommendation systems

---

# **Robustness of GNNs**

Robustness refers to:

**The ability of a GNN to maintain good performance even when the graph contains noise, missing data, or adversarial changes.**

---

# **Why Robustness Matters**

Real-world graphs are rarely perfect.

They may contain:

* Missing edges  
* Fake nodes  
* Incorrect labels  
* Noisy features  
* Adversarial attacks

---

# **Example**

Fraud Detection

Attacker creates:

Fake Transactions

↓

Mislead GNN

Without robustness,

the GNN may classify fraudulent accounts as legitimate.

---

# **Types of Robustness Challenges**

---

## **1\. Noisy Node Features**

Example:

Age

Incorrect

Salary

Missing

---

## **2\. Missing Edges**

Some relationships are absent.

---

## **3\. Spurious Edges**

Fake or irrelevant connections are added.

---

## **4\. Adversarial Attacks**

Small changes to nodes or edges are intentionally designed to fool the model.

Example:

Add One Edge

↓

Prediction Changes

---

# **Improving Robustness**

Several techniques improve GNN robustness.

---

## **1\. Graph Augmentation**

Randomly:

* Drop edges  
* Mask features  
* Sample subgraphs

Helps the model generalize better.

---

## **2\. Attention Mechanisms**

Graph Attention Networks (GAT) learn to assign lower weights to noisy neighbors.

---

## **3\. Adversarial Training**

During training,

intentionally perturb the graph.

The model learns to resist attacks.

---

## **4\. Regularization**

Use:

* Dropout  
* Weight decay  
* Feature smoothing

to reduce overfitting.

---

## **5\. Robust Aggregation**

Instead of simple mean aggregation,

use methods less sensitive to outliers, such as attention or robust statistical aggregators.

---

# **Comparison**

| Aspect | Standard GNN | Improved GNN |
| ----- | ----- | ----- |
| Position Awareness | No | Yes (Position-aware GNNs) |
| Identity Awareness | Limited | Explicit node identities |
| Noise Handling | Moderate | Robust training techniques |
| Over-Smoothing | Common | Mitigated with residual/skip connections |
| Large Graphs | Challenging | Neighbor sampling, GraphSAGE |
| Heterogeneous Graphs | Limited | R-GCN, R-GAT |

---

# **Applications**

### **Autonomous Driving**

Robust road-network understanding.

---

### **Banking**

Fraud detection despite fake transactions.

---

### **Social Networks**

Handle fake accounts and noisy friendships.

---

### **Healthcare**

Work with incomplete biological networks.

---

### **Cybersecurity**

Detect attacks despite adversarial modifications.

---

# **Complete Workflow**

Graph  
      ↓  
Position Encoding  
      ↓  
Identity Encoding  
      ↓  
Robust Graph Augmentation  
      ↓  
Message Passing  
      ↓  
Neighbor Aggregation  
      ↓  
Prediction  
      ↓  
Robust Evaluation

---

# **Interview Summary**

Graph Neural Networks (GNNs) have several important limitations, including **over-smoothing** (node embeddings become indistinguishable with increasing depth), **over-squashing** (large amounts of information are compressed through bottleneck nodes), **limited expressiveness** (bounded by the **1-Weisfeiler-Lehman test**), scalability challenges on large graphs, difficulty handling dynamic and heterogeneous graphs, and sensitivity to noisy or adversarial graph structures. **Position-Aware GNNs** address the inability of standard GNNs to recognize a node's location within a graph by incorporating positional encodings such as **Laplacian eigenvectors, shortest-path distances, or random-walk encodings**. **Identity-Aware GNNs** improve expressiveness by assigning explicit identity or structural role information to nodes, allowing the model to distinguish nodes that otherwise have identical neighborhoods. **Robust GNNs** aim to maintain reliable performance despite missing edges, noisy features, fake nodes, or adversarial attacks by using techniques such as **graph augmentation, attention mechanisms, adversarial training, robust aggregation, regularization, and neighbor sampling**. Together, these advances make modern GNNs more expressive, scalable, and reliable for real-world applications.

# graph ml 17

# **Scaling Up GNNs to Large Graphs**

One of the biggest challenges in **Graph Neural Networks (GNNs)** is scalability.

Real-world graphs can contain:

* Millions of nodes  
* Billions of edges  
* Highly connected hub nodes

Examples:

* Facebook Social Graph  
* Google Knowledge Graph  
* Twitter/X Network  
* Financial Transaction Networks

A standard GNN cannot process the entire graph efficiently because each node aggregates information from all of its neighbors.

---

# **Why Do Standard GNNs Not Scale?**

Suppose one node has:

1 Million Neighbors

One GNN layer requires aggregating information from all of them.

Now consider 3 GNN layers:

Layer 1 → 1-hop neighbors

Layer 2 → Neighbors of neighbors

Layer 3 → 3-hop neighbors

The number of accessed nodes grows exponentially.

This is called the **Neighborhood Explosion Problem**.

---

# **Neighborhood Explosion**

Suppose:

Each node has 100 neighbors

Then

1 Layer

↓

100 nodes

2 Layers

↓

100 × 100

\=

10,000 nodes

3 Layers

↓

100³

\=

1,000,000 nodes

Clearly impossible for large graphs.

---

# **Approaches for Scaling GNNs**

Several techniques have been proposed:

1. Neighbor Sampling (GraphSAGE)  
2. Cluster-based Training (Cluster-GCN)  
3. Graph Sampling  
4. Mini-batch Training  
5. Simplified GNNs (SGC)  
6. Layer Sampling (FastGCN, LADIES)

---

# **GraphSAGE: Neighbor Sampling**

GraphSAGE (**Graph Sample and Aggregate**) was one of the first scalable GNNs.

Instead of aggregating **all neighbors**, GraphSAGE samples only a fixed number.

---

# **Intuition**

Suppose a node has:

1000 neighbors

Instead of using all:

1000

↓

Sample 25

↓

Aggregate

The sampled neighbors approximate the full neighborhood.

---

# **GraphSAGE Neighbor Sampling**

Suppose:

Node A

↓

100 Neighbors

Sampling:

Randomly Choose

↓

20 Neighbors

Then aggregate only these 20\.

---

# **Multi-Layer Sampling**

Suppose:

2 GNN Layers

Layer 1:

Sample

25 Neighbors

Layer 2:

Each sampled neighbor again samples

10 Neighbors

Instead of visiting:

Thousands

↓

Only

25 × 10

\=

250 Nodes

Huge computational savings.

---

# **GraphSAGE Pipeline**

Target Node  
      ↓  
Sample Neighbors  
      ↓  
Aggregate Features  
      ↓  
Combine with Own Features  
      ↓  
Updated Embedding

---

# **Advantages of GraphSAGE**

* Works on massive graphs  
* Supports mini-batches  
* Memory efficient  
* Inductive learning  
* Faster training

---

# **Limitations**

Neighbor sampling introduces approximation.

Some important neighbors may not be sampled.

---

# **Cluster-GCN**

GraphSAGE samples **neighbors**.

Cluster-GCN samples **subgraphs (clusters)**.

---

# **Intuition**

Large graphs naturally contain communities.

Example:

Community A

Community B

Community C

Instead of training on the entire graph,

Cluster-GCN trains on one cluster at a time.

---

# **Step 1**

Partition graph.

Algorithms such as:

* METIS

produce:

Large Graph

↓

Cluster 1

Cluster 2

Cluster 3

---

# **Step 2**

Train on one cluster.

Example:

Only Cluster 2

↓

Forward Pass

↓

Backward Pass

---

# **Step 3**

Repeat for remaining clusters.

Eventually,

all clusters are seen.

---

# **Cluster-GCN Pipeline**

Large Graph  
      ↓  
Graph Partitioning  
      ↓  
Clusters  
      ↓  
Mini-batch Training  
      ↓  
GNN  
      ↓  
Prediction

---

# **Why Cluster-GCN Works**

Nodes inside a community are highly connected.

Most useful information remains inside each cluster.

Therefore,

training on clusters preserves graph structure while reducing computation.

---

# **Advantages**

* Very memory efficient  
* Supports huge graphs  
* Fast mini-batch training  
* Better locality than random sampling

---

# **Limitations**

Quality depends on graph partitioning.

Poor clusters may reduce accuracy.

---

# **Scaling Up by Simplifying GNNs**

Instead of changing training,

another idea is:

**Simplify the GNN architecture itself.**

Observation:

Many GNNs repeatedly perform:

Aggregation

↓

Linear Layer

↓

ReLU

↓

Aggregation

↓

Linear Layer

↓

ReLU

Research showed that some nonlinearities contribute less than expected for certain tasks.

---

# **Simplified Graph Convolution (SGC)**

**Simplified Graph Convolution (SGC)** removes unnecessary operations.

Idea:

Instead of repeatedly performing:

Aggregate

↓

Transform

↓

ReLU

SGC performs aggregation first,

then applies one classifier.

---

# **Standard GCN**

Aggregation

↓

Linear Layer

↓

ReLU

↓

Aggregation

↓

Linear Layer

↓

Prediction

---

# **Simplified GCN**

Repeated Aggregation

↓

One Linear Classifier

↓

Prediction

---

# **Intuition**

Suppose we need information from

3-Hop Neighbors

Instead of:

Aggregate

↓

ReLU

↓

Aggregate

↓

ReLU

↓

Aggregate

Simply perform:

Aggregate

3 Times

↓

Classifier

---

# **Advantages of SGC**

* Extremely fast  
* Much lower memory usage  
* Easy implementation  
* Similar accuracy on many benchmark datasets

---

# **Limitations**

Removing nonlinearities reduces model capacity.

Less suitable for highly complex graphs requiring expressive representations.

---

# **Comparison of Scaling Techniques**

| Technique | Main Idea | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| GraphSAGE | Sample neighbors | Simple, scalable, inductive | Approximate neighbors |
| Cluster-GCN | Train on graph clusters | Efficient mini-batching, preserves locality | Depends on partition quality |
| SGC (Simplified GCN) | Remove intermediate transformations and nonlinearities | Very fast, lightweight | Lower expressiveness |
| FastGCN | Sample graph nodes layer-wise | Faster training | Sampling approximation |
| LADIES | Layer-dependent importance sampling | Better sampling quality | More complex implementation |

---

# **GraphSAGE vs Cluster-GCN**

| Feature | GraphSAGE | Cluster-GCN |
| ----- | ----- | ----- |
| Sampling Unit | Neighbor | Graph Cluster |
| Mini-batching | Yes | Yes |
| Inductive Learning | Yes | Primarily designed for efficient training |
| Memory Usage | Low | Very Low |
| Large Graphs | Excellent | Excellent |

---

# **Standard GCN vs SGC**

| Feature | Standard GCN | Simplified GCN |
| ----- | ----- | ----- |
| Nonlinear Layers | Multiple | Removed |
| Training Speed | Moderate | Very Fast |
| Computational Cost | Higher | Lower |
| Expressiveness | Higher | Lower |
| Suitable For | Complex tasks | Large-scale node classification |

---

# **Complete Scalable GNN Pipeline**

Large Graph  
      ↓  
Choose Scaling Strategy  
      │  
      ├── Neighbor Sampling (GraphSAGE)  
      │  
      ├── Graph Clustering (Cluster-GCN)  
      │  
      ├── Layer Sampling (FastGCN/LADIES)  
      │  
      └── Simplified GCN (SGC)  
              ↓  
Mini-Batch Training  
              ↓  
Message Passing  
              ↓  
Forward Propagation  
              ↓  
Backpropagation  
              ↓  
Prediction

---

# **Interview Summary**

Scaling Graph Neural Networks (GNNs) to large graphs is challenging because of the **neighborhood explosion problem**, where the number of nodes involved in message passing grows exponentially with the number of layers. **GraphSAGE** addresses this by using **neighbor sampling**, where each node aggregates information from only a fixed number of randomly sampled neighbors, enabling scalable mini-batch and inductive learning. **Cluster-GCN** takes a different approach by partitioning the graph into **clusters (communities)** and training on one cluster at a time, preserving local graph structure while greatly reducing memory usage. Another strategy is **simplifying the GNN architecture** itself. **Simplified Graph Convolution (SGC)** removes intermediate nonlinearities and weight matrices between propagation steps, performing repeated feature propagation followed by a single linear classifier. This significantly speeds up training while maintaining competitive accuracy on many node classification tasks. Other scalable methods include **FastGCN** and **LADIES**, which perform layer-wise node sampling. Together, these approaches make GNNs practical for graphs with millions of nodes and billions of edges.

# graph ml 18

# **Pre-Training Graph Neural Networks (Pre-trained GNNs)**

Training Graph Neural Networks (GNNs) from scratch requires **large amounts of labeled graph data**, which is often expensive to obtain.

Inspired by NLP (BERT) and Computer Vision (ImageNet pretraining), researchers introduced **Pre-trained Graph Neural Networks (Pre-trained GNNs)**.

The idea is:

**First learn general graph representations from large unlabeled graphs, then fine-tune on a specific downstream task.**

---

# **Why Pre-train GNNs?**

Many graph datasets have:

* Very few labeled nodes  
* Limited labeled graphs  
* Expensive annotations

Examples:

* Drug molecules  
* Protein networks  
* Knowledge graphs

Instead of learning from scratch,

the model first learns:

General Graph Knowledge

then transfers it.

---

# **Analogy**

Just as:

BERT

↓

Learns Language

↓

Fine-tuned for QA

Similarly,

Pre-trained GNN

↓

Learns Graph Structure

↓

Fine-tuned for Node Classification

---

# **Pre-training Pipeline**

Large Unlabeled Graph Dataset  
            ↓  
Self-Supervised Learning  
            ↓  
Pre-trained GNN  
            ↓  
Fine-Tuning  
            ↓  
Task-Specific Model

---

# **Types of Pre-training Tasks**

---

## **1\. Node-Level Pre-training**

Learn node embeddings.

Example tasks:

* Predict masked node attributes  
* Contrastive learning  
* Context prediction

---

## **2\. Edge-Level Pre-training**

Predict:

Missing Edges

Useful for:

* Link prediction  
* Recommendation

---

## **3\. Graph-Level Pre-training**

Learn embeddings for entire graphs.

Used in:

* Molecule classification  
* Protein prediction

---

# **Self-Supervised Learning**

No labels are required.

Instead,

the graph itself provides supervision.

Popular methods:

* Masked node prediction  
* Contrastive learning  
* Graph reconstruction  
* Edge prediction

---

# **Fine-Tuning**

After pre-training,

replace the final prediction layer.

Train only on the target dataset.

Benefits:

* Faster convergence  
* Better accuracy  
* Requires fewer labeled examples

---

# **Advantages**

* Better generalization  
* Reduced labeling cost  
* Faster training  
* Improved transfer learning

---

# **Applications**

* Drug discovery  
* Protein prediction  
* Fraud detection  
* Recommendation systems  
* Knowledge graphs

---

# **Hyperbolic Graph Embeddings**

Traditional graph embeddings represent nodes in **Euclidean space**.

However,

many real-world graphs have **hierarchical structures**.

Examples:

Company Hierarchy

↓

CEO

↓

Managers

↓

Employees

Euclidean embeddings struggle to represent such hierarchies efficiently.

---

# **Hyperbolic Space**

Hyperbolic space is a **non-Euclidean geometry** with **negative curvature**.

Unlike a flat plane,

it expands exponentially as you move away from the center.

---

# **Intuition**

Imagine:

Flat paper:

Euclidean Space

versus

A continuously expanding surface:

Hyperbolic Space

The hyperbolic space has much more "room" for representing branching structures.

---

# **Why Hyperbolic Space?**

Tree-like graphs grow exponentially.

Example:

CEO

↓

2 Managers

↓

4 Employees

↓

8 Interns

↓

16 ...

Hyperbolic geometry naturally models this exponential growth.

---

# **Euclidean vs Hyperbolic Embeddings**

Euclidean:

Root

↓

Children

↓

Crowded

Hyperbolic:

Root

↓

Children

↓

Plenty of Space

---

# **Advantages**

* Better represents trees  
* Preserves hierarchical distances  
* Requires lower embedding dimensions  
* Improves hierarchical classification

---

# **Applications**

* Knowledge Graphs  
* Taxonomies  
* Organizational Charts  
* Protein Hierarchies  
* Recommendation Systems

---

# **Limitations**

* More difficult optimization  
* Complex distance computations  
* Higher implementation complexity

---

# **Design Space of GNNs**

The **Design Space of GNNs** refers to all the architectural choices involved in building a Graph Neural Network.

A GNN is composed of several independent design components.

Choosing different options leads to different GNN architectures.

---

# **General GNN Architecture**

Input Graph  
      ↓  
Message Construction  
      ↓  
Message Aggregation  
      ↓  
Node Update  
      ↓  
Readout  
      ↓  
Prediction

Each stage has multiple design choices.

---

# **1\. Message Construction**

Defines **what information is sent** from neighboring nodes.

Possible inputs:

* Node features  
* Edge features  
* Positional encodings  
* Structural features

---

# **2\. Aggregation Function**

Determines **how messages from neighbors are combined**.

Common methods:

Sum

Mean

Max

Attention

Examples:

* GCN → Mean/normalized sum  
* GraphSAGE → Mean, Max, LSTM  
* GAT → Attention  
* GIN → Sum

---

# **3\. Update Function**

Updates the node embedding after aggregation.

Common choices:

* MLP  
* Linear layer  
* GRU/LSTM  
* Residual updates

---

# **4\. Number of Layers**

Each GNN layer aggregates information from one additional hop.

* 2 layers → 2-hop neighborhood  
* 3 layers → 3-hop neighborhood

Too many layers may cause:

* Over-smoothing  
* Over-squashing

---

# **5\. Skip Connections**

Residual or skip connections help preserve original node information.

Example:

Input  
   ↓  
Layer 1  
   ↓  
Layer 2  
   ↓  
Output

↘────────────↗

Benefits:

* Easier optimization  
* Reduces over-smoothing  
* Enables deeper GNNs

---

# **6\. Readout Function**

Combines node embeddings into a graph embedding.

Methods:

* Sum Pooling  
* Mean Pooling  
* Max Pooling  
* Attention Pooling

Used for graph-level prediction.

---

# **7\. Loss Function**

Depends on the task.

Examples:

* Cross Entropy (Classification)  
* MSE (Regression)  
* Contrastive Loss (Self-supervised)  
* Ranking Loss (Link Prediction)

---

# **8\. Training Strategy**

Possible choices:

* Full-batch training  
* Mini-batch training  
* Neighbor sampling  
* Cluster-GCN  
* GraphSAINT

---

# **9\. Positional and Structural Encodings**

Modern GNNs often include:

* Laplacian eigenvectors  
* Random walk encodings  
* Degree encodings  
* Centrality measures

These improve expressiveness.

---

# **10\. Graph Type**

Design also depends on graph characteristics:

* Homogeneous graphs  
* Heterogeneous graphs  
* Dynamic graphs  
* Knowledge graphs

Different architectures are appropriate for different graph types.

---

# **Popular GNN Designs**

| Model | Aggregation | Special Feature |
| ----- | ----- | ----- |
| GCN | Mean/Normalized Sum | Simple and efficient |
| GraphSAGE | Mean/Max/LSTM | Neighbor sampling |
| GAT | Attention | Learns neighbor importance |
| GIN | Sum | Maximum expressiveness among standard message-passing GNNs |
| R-GCN | Relation-specific aggregation | Heterogeneous graphs |
| Graph Transformer | Global attention | Long-range dependencies |

---

# **Complete Design Workflow**

Input Graph  
      ↓  
Choose Graph Type  
      ↓  
Message Function  
      ↓  
Aggregation Method  
      ↓  
Update Function  
      ↓  
Number of Layers  
      ↓  
Readout  
      ↓  
Loss Function  
      ↓  
Training Strategy  
      ↓  
Prediction

---

# **Interview Summary**

**Pre-trained Graph Neural Networks (Pre-trained GNNs)** follow the same philosophy as BERT in NLP and ImageNet pretraining in computer vision: a GNN is first trained on **large unlabeled graph datasets using self-supervised objectives** such as masked node prediction, contrastive learning, edge prediction, or graph reconstruction, and is then **fine-tuned** for downstream tasks like node classification, graph classification, or link prediction. This improves accuracy and reduces the need for labeled data. **Hyperbolic Graph Embeddings** represent nodes in **hyperbolic (negatively curved) space** instead of Euclidean space, making them particularly effective for **hierarchical and tree-like graphs** because hyperbolic space naturally accommodates exponential branching while using lower-dimensional embeddings. The **Design Space of GNNs** refers to the collection of architectural choices involved in building a GNN, including **message construction, aggregation function (sum, mean, max, attention), update function, number of layers, skip connections, readout function, loss function, training strategy, positional encodings, and graph type**. Different combinations of these components lead to architectures such as **GCN, GraphSAGE, GAT, GIN, R-GCN, and Graph Transformers**, each optimized for different graph learning scenarios.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAloAAADtCAYAAABnC9sOAAAdM0lEQVR4Xu3da4xdZb3HcaK+8IWJMSb6QhNTeXEinV5G22nLgECddiwWIpRKLPYUOVXAE8AQMFH0NFwCyeHaRE7wEhIJEY8oNWkEIqQXSIlgRWoioWiCUAikVS4dJTElXSfPOnk2a37/teey9vPfe621vzv5ZO/5rbWmMy37v388a82eE0444YQMAAAALkwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEwAAACANEzgpuym+3g7fvx4dvHFF5scAADAgQncDKJYKYoWAADoIxO4malohQIUvfDCC9Py73//+/l9/PiRRx7p7KvHFo/Tzx/vY9Hatm1b6bE7duzILr300mnHAwAAVGACN6HMTExMdCxatCjPn3nmmWzDhg2d/Q4ePJitXLmyc8z73//+aZ9DP2d8vG/fvuwDH/iAyYsfF4vWhRde2Nm+cOHC7Dvf+U7+eNeuXdnVV1897XgAAIAKTOCmuHoU3H///Z28uN9JJ52Ul6aybTN9HFaoVqxYYfLix8WipR588EGTAQAA9MAEbrT8dMvf8573ZC+++GLptpk+nm/RCo/vvvvu7Pzzz8/Gxsayhx56aNoxAAAAPTKBGy0/0Z///OdsfHy88/ETTzyRn1osO2amj7sVrU984hOlReuFwrVg119/PUULAACkZgI3WpJ0W/Tmm292PWamj4tF69ChQ53Pd/nll5cWreKf+aEPfahz6jB8/M4770z7cwAAACowAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwAQAAANIwQd+deOKJ2UknnQQA83bKKadkn/70pwFgmjAftG8MiAn6RgcmAMwXRQvAbLR/9JkJ+kKHJQBUQdECMBfaQ/rIBO50UAJAVRQtAHOlfaRPTOBKhyQA9IKiBWA+tJf0gQncvPe97zVDEgB6QdECMB/ve9/7TD9xZgI3OiABoFcULQDzpf3EmQlcfPSjHzUDEgB6RdECMF8f//jHTU9xZAIXOhwBDK87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>

[image2]: <data:image/png;base64,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>