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

