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

