# 36

# **Artificial Neural Networks (ANN)**

Artificial Neural Networks are Machine Learning models inspired by the structure and working of the human brain.

They are the foundation of:

* Deep Learning  
* Computer Vision  
* NLP  
* Generative AI  
* Speech Recognition

The basic idea is:

Learn patterns from data by adjusting connections between artificial neurons.

---

# **Intuition Behind Neural Networks**

Suppose we want to predict whether a customer will buy a product.

Inputs:

Age  
Salary  
Experience

A neural network learns:

Input Features  
      ↓  
Hidden Processing  
      ↓  
Prediction

without explicitly programming the rules.

---

# **The Artificial Neuron**

The basic building block of a neural network is a neuron.

Biological analogy:

Dendrites → Inputs  
Cell Body → Processing  
Axon → Output

Artificial neuron:

Inputs → Weighted Sum → Activation Function → Output

---

# **Structure of a Neuron**

Suppose inputs:

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

* (w) \= weights  
* (b) \= bias

---

# **Why Do We Need Weights?**

Weights determine importance.

Example:

Salary → Very Important  
Weight \= 0.8

Age → Less Important  
Weight \= 0.2

The network learns these automatically.

---

# **Bias**

Bias shifts the activation threshold.

Without bias:

Output always passes through origin

Bias gives flexibility.

---

# **Activation Function**

After calculating the weighted sum, we apply an activation function.

Purpose:

Introduce non-linearity.

Without activation functions:

Many layers \= One Linear Equation

No benefit from deep networks.

---

# **Common Activation Functions**

---

## **1\. Sigmoid**

Maps values between 0 and 1\.

Used in:

* Binary classification output layer

Advantages:

* Produces probabilities

Disadvantages:

* Vanishing gradient problem

---

## **2\. Tanh**

Maps values between \-1 and 1\.

\[  
tanh(x)=\\frac{e^x-e^{-x}}{e^x+e^{-x}}  
\]

Advantages:

* Zero-centered

Disadvantages:

* Still suffers vanishing gradients

---

## **3\. ReLU (Most Popular)**

Formula:

\[  
f(x)=max(0,x)  
\]

Intuition:

If x \> 0 → Output x  
If x ≤ 0 → Output 0

Advantages:

* Fast  
* Solves vanishing gradient problem  
* Used in most hidden layers

---

## **4\. Softmax**

Used for multi-class classification.

Formula:

\[  
P(y\_i)=\\frac{e^{z\_i}}{\\sum e^{z\_j}}  
\]

Outputs probabilities that sum to 1\.

---

# **How Neural Networks Work**

Suppose:

Input Layer  
    ↓  
Hidden Layer  
    ↓  
Output Layer

Example:

Age  
Salary  
Experience  
     ↓  
 Hidden Neurons  
     ↓  
 Buy / Not Buy

---

# **Layers in Neural Networks**

## **Input Layer**

Receives features.

Example:

Age  
Salary  
Experience

---

## **Hidden Layer**

Learns patterns and relationships.

Example:

Customer Wealth  
Customer Interest

(not explicitly defined)

---

## **Output Layer**

Produces prediction.

Examples:

0 or 1

or

Cat  
Dog  
Bird

---

# **Deep Learning**

When a neural network contains multiple hidden layers, it becomes a **Deep Neural Network**.

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

This is called Deep Learning.

---

# **Why Deep Learning Works**

Each layer learns increasingly complex features.

Example: Image Recognition

Layer 1:

Edges

Layer 2:

Shapes

Layer 3:

Eyes  
Nose

Layer 4:

Face

---

# **Forward Propagation**

Forward Propagation is the process of making predictions.

Steps:

### **Step 1**

Input enters network.

---

### **Step 2**

Weighted sums are computed.

\[  
z=w^Tx+b  
\]

---

### **Step 3**

Activation function applied.

---

### **Step 4**

Output generated.

Example:

Prediction \= 0.8

---

# **Loss Function**

The prediction is compared with actual value.

Example:

Predicted \= 0.8  
Actual \= 1

Error exists.

Loss function measures this error.

---

# **Gradient Descent**

Gradient Descent is the optimization algorithm used to minimize loss.

Goal:

Find weights producing minimum error

---

# **Intuition of Gradient Descent**

Imagine standing on a mountain.

Goal:

Reach the lowest point

You move in the direction of steepest descent.

Similarly:

High Loss  
 ↓  
Lower Loss  
 ↓  
Minimum Loss

---

# **Gradient Descent Update Rule**

Where:

* (w) \= weight  
* (\\eta) \= learning rate  
* (L) \= loss

---

# **Learning Rate**

Controls step size.

### **Large Learning Rate**

May overshoot minimum

---

### **Small Learning Rate**

Very slow learning

---

# **Types of Gradient Descent**

---

## **1\. Batch Gradient Descent**

Uses entire dataset.

Compute error on all samples  
Update once

Advantages:

* Stable

Disadvantages:

* Slow

---

## **2\. Stochastic Gradient Descent (SGD)**

Updates after every training sample.

One Sample  
↓  
Update Weights

Advantages:

* Fast  
* Escapes local minima

Disadvantages:

* Noisy updates

---

## **3\. Mini-Batch Gradient Descent**

Most common approach.

Uses small batches.

Example:

Batch Size \= 32

Advantages:

* Fast  
* Stable

---

# **Backpropagation**

Backpropagation is the learning algorithm of neural networks.

Purpose:

Determine how much each weight contributed to the error.

---

# **Intuition of Backpropagation**

Suppose:

Prediction \= 0.9  
Actual \= 1

Small error exists.

Backpropagation asks:

Which weights caused the error?

and adjusts them accordingly.

---

# **How Backpropagation Works**

### **Step 1**

Perform Forward Propagation.

Input → Output

---

### **Step 2**

Compute Loss.

Actual vs Predicted

---

### **Step 3**

Calculate Gradients.

Find:

\[  
\\frac{\\partial L}{\\partial w}  
\]

for every weight.

---

### **Step 4**

Propagate Error Backwards.

Output Layer  
↓  
Hidden Layer  
↓  
Input Layer

---

### **Step 5**

Update Weights.

Using Gradient Descent:

---

### **Step 6**

Repeat for many epochs.

Network gradually learns.

---

# **Complete Learning Cycle**

Input Data  
      ↓  
Forward Propagation  
      ↓  
Prediction  
      ↓  
Calculate Loss  
      ↓  
Backpropagation  
      ↓  
Update Weights  
      ↓  
Repeat

---

# **Classical ML vs Neural Networks**

| Feature | Classical ML | Neural Networks |
| ----- | ----- | ----- |
| Feature Engineering | Manual | Automatic |
| Data Requirement | Small | Large |
| Training Time | Fast | Slow |
| Non-linearity | Limited | Excellent |
| Accuracy | Good | Very High |
| Images & NLP | Weak | Excellent |

---

# **Applications of Neural Networks**

### **Computer Vision**

* Face Recognition  
* Object Detection

### **NLP**

* ChatGPT  
* Translation  
* Sentiment Analysis

### **Speech Recognition**

* Siri  
* Alexa

### **Healthcare**

* Disease Detection

### **Finance**

* Fraud Detection

### **Recommendation Systems**

* Netflix  
* YouTube

---

# **Interview Summary**

An **Artificial Neural Network (ANN)** is a machine learning model inspired by the human brain. It consists of **neurons** connected through weighted links organized into **input, hidden, and output layers**. Each neuron computes a weighted sum of inputs, adds a bias, and applies an **activation function** such as Sigmoid, ReLU, or Softmax. Learning occurs through **Forward Propagation**, where predictions are generated, and **Backpropagation**, where errors are propagated backward to compute gradients. The weights are then updated using **Gradient Descent** or **Stochastic Gradient Descent (SGD)** to minimize the loss function. Networks with multiple hidden layers are called **Deep Neural Networks**, forming the foundation of modern Deep Learning applications in computer vision, NLP, speech recognition, and generative AI.

