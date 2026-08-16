# Fintech ML Interview — Revision Notes

Quick-refresh notes for ML fundamentals commonly asked in fintech interviews (fraud detection, credit scoring, risk models). Written so a first-year engineer can follow: **what it is → why it matters → intuition → code → equation (only where it helps)**.

Related deep-dive files already in this repo (folder numbers under `ml/`):
`001` scaling · `002` preprocessing · `007` decision tree · `008` ensemble · `011/035` logistic regression · `012` KNN · `013/014` SVM · `015` naive bayes · `017` random forest · `018/019` confusion matrix · `020` k-means · `030` PCA · `033` k-fold CV · `034` xgboost · `039` regularization · `040` DBSCAN/GMM · `041` adaboost/gradient boosting · `042` anomaly detection (useful for fraud).

---

## Table of Contents

1. [Data Processing](#1-data-processing)
2. [Evaluation Metrics](#2-evaluation-metrics)
3. [Algorithms — Intuition Only](#3-algorithms--intuition-only)
4. [Bias-Variance, Overfitting & Regularization](#4-bias-variance-overfitting--regularization)
5. [Handling Imbalanced Data (fraud/credit-risk staple)](#5-handling-imbalanced-data)
6. [Model Explainability — SHAP & friends](#6-model-explainability--shap--friends)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Fintech-Specific Gotchas](#8-fintech-specific-gotchas)
9. [Cheat-Sheet: "Which metric / model when?"](#9-cheat-sheet)

---

## 1. Data Processing

### 1.1 Train / Test / Validation Split

**Why:** you need data the model has never seen to know if it actually generalizes, not just memorized.

- **Train** → model learns weights/splits from this.
- **Validation** → used to tune hyperparameters / pick the best model (still "seen" indirectly).
- **Test** → touched exactly once, at the end, to report final performance.

Typical split: 60/20/20 or 70/15/15. In fintech, also consider a **time-based split** (train on past, test on future) instead of random split, because fraud/credit patterns drift over time and random splitting leaks future information into training ("look-ahead bias").

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y)
```

`stratify=y` keeps class ratios the same in each split — important when fraud is e.g. 1% of data.

### 1.2 Cross Validation

**Why:** a single train/test split can be lucky or unlucky. CV averages performance over multiple splits → more reliable estimate.

**K-Fold CV intuition:** split data into K equal chunks ("folds"). Train on K-1 folds, test on the remaining 1, repeat K times so every fold is the test set once. Average the K scores.

```
Fold1 Fold2 Fold3 Fold4 Fold5
[test][train][train][train][train]  -> score 1
[train][test][train][train][train]  -> score 2
... etc, then average
```

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(scores.mean(), scores.std())
```

For **time-series / transaction data**, use `TimeSeriesSplit` instead — it never trains on future data to predict the past.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### 1.3 Feature Engineering

**Why:** models are only as good as the signal you hand them. In fintech this is often *more* impactful than model choice.

Common fintech feature ideas:
- **Aggregations**: avg transaction amount over last 7/30/90 days, count of transactions per merchant category.
- **Ratios**: debt-to-income, transaction amount vs. account average.
- **Time features**: hour of day, day of week, time since last transaction (fraud often happens at odd hours / in bursts).
- **Velocity features**: number of transactions in last 10 minutes (classic fraud signal).
- **Domain flags**: is_foreign_transaction, is_new_device, is_new_merchant.

```python
df['txn_hour'] = df['timestamp'].dt.hour
df['amount_to_avg_ratio'] = df['amount'] / df.groupby('user_id')['amount'].transform('mean')
df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
```

### 1.4 Missing Values

**Why:** most ML algorithms (except tree-based ones like XGBoost/LightGBM) can't handle NaNs directly, and missingness itself can be informative (e.g., a missing income field might correlate with risk).

| Strategy | When to use |
|---|---|
| Drop rows | Missing is rare (<5%) and random |
| Mean/Median imputation | Numeric, roughly symmetric/skewed distribution |
| Mode imputation | Categorical |
| Forward/backward fill | Time series |
| Model-based (KNN/iterative imputer) | When missingness is complex |
| **Add "was_missing" flag column** | When missingness itself is a signal (very common in credit scoring) |

```python
from sklearn.impute import SimpleImputer

df['income_missing'] = df['income'].isna().astype(int)  # keep the signal
imputer = SimpleImputer(strategy='median')
df['income'] = imputer.fit_transform(df[['income']])
```

⚠️ **Gotcha:** fit the imputer (and any scaler/encoder) only on the **training set**, then `.transform()` the val/test sets — never `.fit_transform()` on the whole dataset, or you leak information.

### 1.5 One-Hot Encoding

**Why:** converts a categorical column with no inherent order (e.g., `merchant_category: grocery, travel, electronics`) into binary columns, so the model doesn't assume a false numeric order (electronics > travel > grocery would be nonsense).

```
merchant_category      grocery  travel  electronics
grocery         --->     1        0         0
travel                    0        1         0
electronics                0        0         1
```

```python
import pandas as pd
pd.get_dummies(df, columns=['merchant_category'], drop_first=True)  # drop_first avoids redundant column (multicollinearity)
```

**Downside:** blows up dimensionality if the category has many unique values (e.g. `merchant_id` with 50,000 values) — consider target encoding or embeddings instead.

### 1.6 Label Encoding

**Why:** converts categories to integers `0, 1, 2, ...`. Use only when categories have a **natural order** (ordinal), e.g. `credit_grade: Poor < Fair < Good < Excellent`. Using it on non-ordinal data tricks linear/distance-based models into thinking `2 > 1`.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['credit_grade_enc'] = le.fit_transform(df['credit_grade'])  # careful: alphabetical order by default, reorder manually if needed
```

Tree-based models (Random Forest, XGBoost) are fairly robust to label-encoded nominal data too, since they split on thresholds rather than assuming linear distance.

### 1.7 Standardization vs Normalization

**Why:** features on different scales (e.g. `age: 18-90` vs `income: 20,000-500,000`) dominate distance-based or gradient-based models unless rescaled.

| | Standardization (Z-score) | Normalization (Min-Max) |
|---|---|---|
| Formula | $x' = \dfrac{x - \mu}{\sigma}$ | $x' = \dfrac{x - x_{min}}{x_{max} - x_{min}}$ |
| Output range | mean 0, std 1 (unbounded) | fixed [0, 1] |
| Sensitive to outliers? | Less | Very (min/max shift with outliers) |
| Use with | Linear/Logistic Regression, SVM, KNN, PCA, Neural Nets | Neural nets (images), algorithms needing bounded input |
| Not needed for | Tree-based models (Decision Tree, Random Forest, XGBoost) — splits are scale-invariant |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # transform only, never re-fit on test
```

---

## 2. Evaluation Metrics

### 2.1 Confusion Matrix (base for classification metrics)

```
                 Predicted Positive   Predicted Negative
Actual Positive        TP                    FN
Actual Negative        FP                    TN
```

In fraud detection: **Positive = fraud**. FN = fraud you missed (money lost). FP = legit transaction flagged as fraud (annoyed customer).

### 2.2 Classification Metrics — when to use which

| Metric | Formula | Use when |
|---|---|---|
| **Accuracy** | $(TP+TN)/(TP+TN+FP+FN)$ | Classes are balanced. **Misleading on imbalanced data** — a model predicting "not fraud" always gets 99% accuracy on 1%-fraud data while catching zero fraud. |
| **Precision** | $TP/(TP+FP)$ | False positives are costly — e.g. blocking a legit customer's card causes churn/complaints. "Of all flagged, how many were really fraud?" |
| **Recall (Sensitivity)** | $TP/(TP+FN)$ | False negatives are costly — e.g. missing fraud = direct financial loss. "Of all actual fraud, how many did we catch?" |
| **F1 Score** | $2 \cdot \dfrac{Precision \cdot Recall}{Precision + Recall}$ | Need one number balancing precision & recall, especially on imbalanced data. |
| **ROC AUC** | area under TPR vs FPR curve | Comparing/ranking models overall, class balance is moderate. Says "how well does the model rank positives above negatives" regardless of threshold. |
| **PR AUC** | area under Precision vs Recall curve | **Preferred over ROC AUC for heavily imbalanced problems** (fraud, rare disease) — ROC AUC can look great even when precision is poor, because TNs (huge in number) dominate FPR. |
| **Log Loss** | $-\frac{1}{N}\sum y\log(\hat p)+(1-y)\log(1-\hat p)$ | You care about **calibrated probabilities**, not just the class label — e.g. a risk score used for pricing/decisioning, not just approve/deny. Penalizes confident-wrong predictions heavily. |

**Precision-Recall tradeoff:** raising the classification threshold → higher precision, lower recall (and vice versa). Pick the threshold based on business cost of FP vs FN, not a default 0.5.

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score, log_loss)

precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
roc_auc_score(y_test, y_proba)          # needs probabilities, not hard labels
average_precision_score(y_test, y_proba) # this IS the PR-AUC
log_loss(y_test, y_proba)
```

### 2.3 Regression Metrics

| Metric | Formula | Use when |
|---|---|---|
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum \lvert y-\hat y \rvert$ | Want an interpretable "average error in the same unit," robust to outliers |
| **MSE** (Mean Squared Error) | $\frac{1}{n}\sum (y-\hat y)^2$ | Want to penalize large errors more (a $10,000 credit-limit error matters way more than a $10 one) |
| **RMSE** | $\sqrt{MSE}$ | Same as MSE but back in original units — most commonly reported |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

MAE treats all errors equally; MSE/RMSE punishes big misses much harder (because of squaring) — use MSE when large errors are disproportionately bad (e.g., under-predicting credit risk loss).

---

## 3. Algorithms — Intuition Only

*(No need to memorize derivations — interviewers want to see you understand the "story" and know when to use each.)*

### Linear Regression
Draws the best straight line through the data by minimizing squared error between predicted and actual values. Use for: predicting a continuous number (e.g. transaction amount, loan payoff amount) when relationship is roughly linear.
$\hat y = w_0 + w_1x_1 + \dots + w_nx_n$, fit by minimizing $\sum(y-\hat y)^2$ (Ordinary Least Squares).

### Logistic Regression
Same idea as linear regression, but squashes the output through a **sigmoid** function so it outputs a probability between 0 and 1, for classification (e.g. fraud / not fraud).
$P(y=1) = \dfrac{1}{1+e^{-(w_0+w_1x_1+\dots)}}$
Fast, interpretable (coefficients ≈ feature importance direction), a great **baseline** for credit scoring since regulators like interpretable models.

### Decision Tree
Repeatedly splits data on the feature/threshold that best separates classes (using **Gini impurity** or **entropy/information gain**), forming a tree of if/else rules. Very interpretable ("if income < 30k AND age < 25 → high risk"), but prone to overfitting if grown too deep.

### Random Forest
Trains **many decision trees**, each on a random subset of rows (bagging) and random subset of features, then averages/votes their predictions. Reduces overfitting of a single tree, more robust, still gives feature importances. Good general-purpose fintech model when interpretability isn't as critical as accuracy.

### XGBoost (Gradient Boosting)
Instead of building trees independently (like Random Forest), builds trees **sequentially**, where each new tree focuses on correcting the errors (residuals) of the previous ones. Usually the strongest performer on structured/tabular data — the go-to for fraud/credit-risk Kaggle-style problems. Slightly less interpretable, more prone to overfitting if untuned, but SHAP fixes the interpretability gap (see §6).

### SVM (Support Vector Machine)
Finds the line/hyperplane that separates two classes with the **maximum margin** (widest possible street between the classes). Support vectors = the closest points that define this margin. With a **kernel trick**, it can separate non-linear data by implicitly projecting it into higher dimensions. Works well on smaller, high-dimensional datasets; slow on large datasets.

### KNN (K-Nearest Neighbors)
"Tell me who your neighbors are, and I'll tell you what you are." To classify a new point, look at its K closest points (by distance) in the training data and take a majority vote. No real "training" — all computation happens at prediction time (lazy learner). Sensitive to feature scale (must standardize!) and slow on large datasets.

### Naive Bayes
Uses Bayes' theorem to compute the probability of each class given the features, **assuming features are independent** (the "naive" part — rarely true, but works surprisingly well anyway). Extremely fast, great baseline for text/spam-style classification, works well even with limited data.
$P(class\mid features) \propto P(class)\prod_i P(feature_i \mid class)$

### K-Means
Unsupervised clustering: pick K cluster centers, assign every point to its nearest center, recompute centers as the mean of assigned points, repeat until stable. Used for customer segmentation (e.g. grouping users by spending behavior). You must choose K upfront (use the "elbow method").

### DBSCAN
Density-based clustering: groups points that are closely packed together, and marks isolated points as **outliers/noise** — no need to specify number of clusters upfront. Great for **anomaly/fraud detection**, since fraud often looks like a sparse "noise" cluster far from normal transaction density.

### PCA (Principal Component Analysis)
Dimensionality reduction: finds the directions (principal components) along which the data varies the most, and projects data onto the top few directions while losing the least information. Used to reduce hundreds of correlated features (e.g. many transaction stats) down to a handful, speed up training, remove multicollinearity, or visualize high-dimensional data in 2D/3D.

---

## 4. Bias-Variance, Overfitting & Regularization

- **Bias** = error from overly simplistic assumptions (underfitting) — model too dumb to capture the pattern.
- **Variance** = error from being overly sensitive to training data noise (overfitting) — model memorizes training data, fails on new data.
- **Goal:** find the sweet spot — low bias AND low variance.

```
High Bias (underfit)     Good Fit          High Variance (overfit)
   straight line       reasonable curve      wiggly line hitting every point
```

**Regularization** = penalize large model weights to reduce overfitting.

| Type | Penalty added to loss | Effect |
|---|---|---|
| **L1 (Lasso)** | $\lambda\sum\lvert w_i\rvert$ | Shrinks some weights to exactly 0 → automatic feature selection |
| **L2 (Ridge)** | $\lambda\sum w_i^2$ | Shrinks all weights smaller, but rarely to 0 |
| **Elastic Net** | mix of L1+L2 | Best of both |

```python
from sklearn.linear_model import Ridge, Lasso
Ridge(alpha=1.0).fit(X_train, y_train)
Lasso(alpha=0.1).fit(X_train, y_train)
```

---

## 5. Handling Imbalanced Data

Extremely common in fintech: fraud is often <1% of transactions, defaults might be 5-10% of loans.

- **Don't trust accuracy** — use PR-AUC, F1, recall on the minority class.
- **Resampling:**
  - Oversample minority class (**SMOTE** — generates synthetic minority samples by interpolating between real ones, rather than just duplicating).
  - Undersample majority class (risk: throwing away useful data).
- **Class weights:** tell the model to penalize misclassifying the minority class more.
  ```python
  from sklearn.linear_model import LogisticRegression
  LogisticRegression(class_weight='balanced')  # or pass a custom {0: 1, 1: 10}
  ```
- **Threshold tuning:** instead of default 0.5, pick the probability cutoff that optimizes business cost (F-beta score, cost-sensitive analysis).
- **Anomaly detection framing:** if fraud is extremely rare (<0.1%), sometimes better treated as an outlier-detection problem (Isolation Forest, One-Class SVM, DBSCAN) rather than standard classification.

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

---

## 6. Model Explainability — SHAP & friends

**Why it matters in fintech specifically:** regulations (e.g. Fair Lending / ECOA in the US, GDPR "right to explanation" in EU) often *require* you to explain why a loan/credit decision was made — you can't just say "the model said so."

### SHAP (SHapley Additive exPlanations)
Based on game theory (Shapley values): treats each feature as a "player" contributing to the final prediction, and fairly distributes credit for how much each feature pushed the prediction up or down from the baseline (average prediction).

- **Global explainability:** which features matter most overall across the whole dataset.
- **Local explainability:** for *this one* customer, why did the model deny their loan? (e.g. "low income contributed -0.3, high existing debt contributed -0.2, long credit history contributed +0.1")

```python
import shap

explainer = shap.TreeExplainer(model)          # fast path for tree models (XGBoost/RF)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)          # global: feature importance + direction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])  # local: explain one prediction
```

Key property: SHAP values for a prediction **sum up exactly** to (prediction − average prediction) — unlike generic "feature importance," it's mathematically consistent per-row.

### Other explainability tools worth knowing the name of
- **LIME** — approximates the model locally with a simple interpretable model around one prediction (older, less rigorous than SHAP, but conceptually similar).
- **Permutation Feature Importance** — shuffle one feature's values, see how much performance drops; simple, model-agnostic, but can mislead with correlated features.
- **Partial Dependence Plots (PDP)** — shows how the prediction changes as one feature varies, averaging out the others.

---

## 7. Hyperparameter Tuning

Since it will likely come up alongside XGBoost/Random Forest in a fintech interview.

| Method | Idea |
|---|---|
| **Grid Search** | Try every combination of a predefined parameter grid — exhaustive, slow. |
| **Random Search** | Sample random combinations — often nearly as good, much faster. |
| **Bayesian Optimization (Optuna)** | Uses past trial results to intelligently pick the next parameters to try — most efficient for expensive models. |

```python
from sklearn.model_selection import GridSearchCV

params = {'max_depth': [3, 5, 7], 'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1]}
grid = GridSearchCV(xgb_model, params, cv=5, scoring='average_precision')
grid.fit(X_train, y_train)
```

Key XGBoost knobs to know: `max_depth` (tree complexity), `learning_rate`/`eta` (step size — smaller = slower but often better), `n_estimators` (number of trees), `subsample`/`colsample_bytree` (row/column sampling to prevent overfitting), `scale_pos_weight` (handles class imbalance directly).

---

## 8. Fintech-Specific Gotchas

- **Data leakage**: never let a feature "know the future" — e.g. using "chargeback_flag" (only known after the fact) to predict fraud at transaction time. Always ask: "would I have known this value *at prediction time*?"
- **Concept drift**: fraud patterns, spending habits, and macroeconomic conditions change — models decay in production and need periodic retraining/monitoring.
- **Calibration matters**: if the output feeds into a business decision like "expected loss = P(default) × loan amount," you need well-calibrated probabilities (check with a **calibration curve** / `CalibratedClassifierCV`), not just a good ranking metric.
- **Interpretability vs. performance tradeoff**: regulated credit decisions often favor Logistic Regression / shallow trees over black-box models, or require SHAP explanations if using XGBoost.
- **Cost-sensitive evaluation**: a false negative (missed fraud) and false positive (blocked legit customer) rarely cost the same — build a custom cost matrix instead of optimizing a generic metric when possible.

---

## 9. Cheat-Sheet

**"Which metric when?"**
- Balanced classes, care about both errors equally → **Accuracy**
- FP is expensive (annoying customers) → **Precision**
- FN is expensive (missed fraud/default) → **Recall**
- Need one balanced number, imbalanced classes → **F1**
- Comparing model ranking ability, roughly balanced → **ROC AUC**
- Heavily imbalanced (fraud) → **PR AUC**
- Need calibrated probabilities for downstream $ decisions → **Log Loss**
- Predicting a number, outlier-robust → **MAE**
- Predicting a number, punish big misses → **MSE / RMSE**

**"Which model when?"**
- Need interpretability / regulatory sign-off → Logistic Regression, shallow Decision Tree
- Best raw tabular performance → XGBoost / LightGBM (+ SHAP for explainability)
- Small dataset, complex boundary → SVM (with kernel)
- Quick baseline, mostly independent features → Naive Bayes
- Customer segmentation (no labels) → K-Means
- Anomaly/fraud detection (no labels, weird-shaped clusters) → DBSCAN / Isolation Forest
- Too many correlated features → PCA first, then any model
