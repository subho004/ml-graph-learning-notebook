# 37

# **Convolutional Neural Networks (CNN)**

A **Convolutional Neural Network (CNN)** is a specialized Deep Learning architecture primarily used for:

* Image Classification  
* Object Detection  
* Face Recognition  
* Medical Imaging  
* Video Analysis

CNNs are designed to automatically learn spatial features from images.

---

# **Why Not Use a Regular Neural Network for Images?**

Suppose an image size is:

256 × 256 × 3

(Number of pixels × RGB channels)

Total inputs:

196,608 features

A fully connected neural network would require millions of parameters.

Problems:

* Huge memory usage  
* Slow training  
* Overfitting

CNN solves this using:

Convolution  
Parameter Sharing  
Feature Extraction

---

# **Intuition Behind CNN**

Imagine recognizing a face.

You don't look at every pixel individually.

You first detect:

Edges  
↓  
Shapes  
↓  
Eyes  
↓  
Nose  
↓  
Face

CNN learns features similarly.

---

# **CNN Architecture**

Typical CNN:

Input Image  
      ↓  
Convolution Layer  
      ↓  
ReLU Layer  
      ↓  
Pooling Layer  
      ↓  
Convolution Layer  
      ↓  
ReLU Layer  
      ↓  
Pooling Layer  
      ↓  
Flattening  
      ↓  
Fully Connected Layer  
      ↓  
Softmax  
      ↓  
Prediction

---

# **1\. Convolution Operation**

The most important operation in CNN.

---

# **What is Convolution?**

A small matrix called a:

Filter  
Kernel  
Feature Detector

slides across the image.

Example:

Image:

1 1 1  
0 1 0  
0 1 0

Filter:

1 0  
0 1

The filter performs element-wise multiplication and summation.

This produces a new value.

The process repeats across the image.

---

# **Convolution Formula**

Mathematically:

\[  
(I\*K)(x,y)  
\]

where:

* (I) \= Image  
* (K) \= Kernel

The output is called a Feature Map.

---

# **Why Convolution?**

Instead of learning:

Millions of pixels

CNN learns:

Edges  
Corners  
Textures  
Patterns

which are much more useful.

---

# **Feature Detectors (Kernels)**

A kernel is a small matrix.

Example:

3 × 3  
5 × 5  
7 × 7

Different kernels learn different features.

---

# **Edge Detector**

Example kernel:

\-1 \-1 \-1  
 0  0  0  
 1  1  1

Detects horizontal edges.

---

# **Vertical Edge Detector**

\-1 0 1  
\-1 0 1  
\-1 0 1

Detects vertical edges.

---

# **Feature Maps**

After convolution, we obtain:

Feature Map  
Activation Map

A feature map highlights locations where a feature appears.

Example:

Original Image  
      ↓  
Edge Detector  
      ↓  
Edge Feature Map

---

# **Multiple Feature Maps**

CNN does not use one filter.

It learns many filters:

Filter 1 → Edges

Filter 2 → Corners

Filter 3 → Textures

Filter 4 → Shapes

Each creates its own feature map.

---

# **2\. ReLU Layer**

After convolution we apply:

ReLU  
(Rectified Linear Unit)

Formula:

\[  
f(x)=max(0,x)  
\]

---

# **Intuition**

Positive Value → Keep  
Negative Value → 0

Example:

Input:

\[-3, 4, \-2, 8\]

Output:

\[0, 4, 0, 8\]

---

# **Why ReLU?**

Advantages:

* Introduces non-linearity  
* Faster training  
* Reduces vanishing gradients

---

# **3\. Pooling Layer**

Pooling reduces feature map size.

Purpose:

Reduce Computation  
Reduce Overfitting  
Retain Important Features

---

# **Max Pooling**

Most common pooling method.

Example:

Input:

1 3  
4 2

Output:

4

Largest value selected.

---

# **Intuition**

Instead of remembering every pixel:

Remember strongest feature

---

# **Max Pooling Example**

Feature Map:

1 2 3 4  
5 6 7 8  
1 2 3 4  
5 6 7 8

Using:

2 × 2 Pooling

Output:

6 8  
6 8

---

# **Average Pooling (Mean Pooling)**

Instead of maximum value:

Take average.

Input:

1 3  
4 2

Output:

2.5

---

# **Max Pooling vs Average Pooling**

| Feature | Max Pooling | Average Pooling |
| ----- | ----- | ----- |
| Keeps Strongest Feature | Yes | No |
| Most Common | Yes | Less Common |
| Accuracy | Higher | Lower |

---

# **4\. Flattening**

After several:

Convolution  
Pooling

we obtain feature maps.

Example:

2 × 2 × 3

CNN cannot directly feed this into dense layers.

Flattening converts it into:

\[1,2,3,4,5,6,7,8,9,10,11,12\]

a single vector.

---

# **Example**

Before:

\[  
 \[1 2\]  
 \[3 4\]  
\]

After Flattening:

\[1,2,3,4\]

---

# **Why Flatten?**

Transforms spatial features into a format suitable for classification.

---

# **5\. Fully Connected Layer**

Also called:

Dense Layer

Works like a regular neural network.

Receives flattened features.

Example:

Edges  
Shapes  
Textures

and combines them to make decisions.

---

# **Intuition**

Earlier layers learn:

Eyes  
Nose  
Mouth

Fully connected layer learns:

Eyes \+ Nose \+ Mouth  
\=  
Face

---

# **6\. Softmax Layer**

Used in multi-class classification.

Converts outputs into probabilities.

Formula:

\[  
P(y\_i)=\\frac{e^{z\_i}}{\\sum e^{z\_j}}  
\]

---

# **Example**

Network output:

\[3.5, 1.2, 0.5\]

Softmax:

\[0.85, 0.10, 0.05\]

Interpretation:

Class 1 → 85%  
Class 2 → 10%  
Class 3 → 5%

Prediction:

Class 1

---

# **Cross Entropy Loss**

Used to measure prediction error.

For classification:

\[  
Loss=-\\sum y\_i \\log(\\hat{y\_i})  
\]

---

# **Intuition**

Suppose:

Actual:

Cat

Prediction:

Cat \= 0.95

Loss:

Very Small

---

Prediction:

Cat \= 0.10

Loss:

Very Large

Cross Entropy penalizes wrong confident predictions heavily.

---

# **Complete CNN Workflow**

Image  
 ↓  
Convolution  
 ↓  
Feature Maps  
 ↓  
ReLU  
 ↓  
Pooling  
 ↓  
Convolution  
 ↓  
ReLU  
 ↓  
Pooling  
 ↓  
Flattening  
 ↓  
Fully Connected Layer  
 ↓  
Softmax  
 ↓  
Prediction  
 ↓  
Cross Entropy Loss  
 ↓  
Backpropagation  
 ↓  
Update Filters

---

# **Why CNN Works So Well**

Early Layers:

Edges  
Corners  
Textures

Middle Layers:

Shapes  
Patterns

Deep Layers:

Objects  
Faces  
Cars  
Animals

This hierarchical feature learning makes CNNs extremely powerful.

---

# **Applications of CNN**

### **Computer Vision**

* Image Classification  
* Object Detection

### **Face Recognition**

* Face Unlock  
* Surveillance

### **Medical Imaging**

* Tumor Detection  
* X-Ray Analysis

### **Self-Driving Cars**

* Lane Detection  
* Traffic Sign Recognition

### **OCR**

* Handwriting Recognition  
* Document Digitization

---

# **Interview Summary**

A **Convolutional Neural Network (CNN)** is a deep learning architecture designed for image and spatial data processing. It consists of **Convolution Layers**, which use learnable filters (feature detectors) to extract features such as edges, textures, and shapes, producing **Feature Maps**. The **ReLU Layer** introduces non-linearity, while **Pooling Layers** (Max Pooling and Average Pooling) reduce spatial dimensions and computational complexity. The resulting feature maps are converted into a one-dimensional vector through **Flattening**, which is fed into **Fully Connected Layers** for classification. The final layer typically uses **Softmax** to generate class probabilities, and training is guided by **Cross Entropy Loss** and **Backpropagation**. CNNs form the backbone of modern computer vision systems and are widely used in image recognition, object detection, facial recognition, and medical image analysis.

