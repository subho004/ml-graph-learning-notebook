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

