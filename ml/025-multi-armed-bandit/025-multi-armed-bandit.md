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

