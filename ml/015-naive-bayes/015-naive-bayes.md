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

