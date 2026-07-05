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

