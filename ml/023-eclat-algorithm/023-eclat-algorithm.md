# 30

# **Eclat Algorithm (Equivalence Class Clustering and Bottom-Up Lattice Traversal)**

Eclat is an **Association Rule Learning Algorithm** used to find **frequent itemsets** in transactional databases.

Like Apriori, it is used for:

- Market Basket Analysis
- Recommendation Systems
- Retail Analytics
- Web Usage Mining

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

- Multiple database scans
- Large number of candidate itemsets
- Slow for large datasets

Eclat solves this by using a **vertical data format**.

---

# **Intuition Behind Eclat**

Instead of storing:

| Transaction | Items       |
| ----------- | ----------- |
| T1          | Bread, Milk |
| T2          | Bread, Beer |
| T3          | Milk, Beer  |

Eclat stores:

| Item  | Transactions |
| ----- | ------------ |
| Bread | T1, T2       |
| Milk  | T1, T3       |
| Beer  | T2, T3       |

This is called the **Transaction ID Set (TID Set)** representation.

---

# **Core Idea of Eclat**

Eclat finds support by performing:

Set Intersections

instead of repeatedly scanning the database.

---

# **Example Dataset**

| Transaction | Items             |
| ----------- | ----------------- |
| T1          | Bread, Milk       |
| T2          | Bread, Beer       |
| T3          | Milk, Beer        |
| T4          | Bread, Milk, Beer |

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
Milk → {T1,T3,T4}
Beer → {T2,T3,T4}

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

| Transaction | Items           |
| ----------- | --------------- |
| T1          | Bread Milk      |
| T2          | Bread Beer      |
| T3          | Bread Milk Beer |
| T4          | Milk Beer       |

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

- Confidence
- Lift
- Support

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

| Feature              | Apriori       | Eclat            |
| -------------------- | ------------- | ---------------- |
| Data Format          | Horizontal    | Vertical         |
| Support Calculation  | Database Scan | Set Intersection |
| Candidate Generation | Heavy         | Less             |
| Speed                | Slower        | Faster           |
| Memory Usage         | Lower         | Higher           |
| Large Datasets       | Moderate      | Better           |

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

| Feature             | Eclat  | FP-Growth |
| ------------------- | ------ | --------- |
| Uses TID Sets       | Yes    | No        |
| Uses Tree Structure | No     | Yes       |
| Speed               | Fast   | Faster    |
| Memory Usage        | Higher | Lower     |
| Large Databases     | Good   | Excellent |

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
