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

