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

