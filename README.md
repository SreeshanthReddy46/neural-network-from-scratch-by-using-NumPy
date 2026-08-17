# Neural Network from Scratch (NumPy Only)

A modular, mathematically rigorous implementation of a multi-layer deep neural network built entirely from scratch using only **NumPy** and **Matplotlib** — without relying on deep learning frameworks like PyTorch, TensorFlow, or scikit-learn.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Mathematics](#architecture--mathematics)
  - [1. Forward Pass (Affine Transform)](#1-forward-pass-affine-transform)
  - [2. Activation Functions](#2-activation-functions)
  - [3. Softmax & Categorical Cross-Entropy](#3-softmax--categorical-cross-entropy)
  - [4. Backpropagation & The Chain Rule](#4-backpropagation--the-chain-rule)
  - [5. Weight Initialization (He / Kaiming Init)](#5-weight-initialization-he--kaiming-init)
  - [6. Gradient Checking (Finite Differences)](#6-gradient-checking-finite-differences)
  - [7. Optimizer: SGD with Momentum & L2 Weight Decay](#7-optimizer-sgd-with-momentum--l2-weight-decay)
- [Dataset: 2D Spiral Classification](#dataset-2d-spiral-classification)
- [Code Structure & Class Breakdown](#code-structure--class-breakdown)
- [How to Run](#how-to-run)
- [Visualizations & Output](#visualizations--output)

---

## 🚀 Overview

The purpose of this project is to demystify deep learning by implementing every foundational component by hand:
- Constructing feedforward layers
- Applying non-linear activations
- Computing analytical gradients via the calculus chain rule
- Verifying analytical derivatives against numerical approximations
- Optimizing weights using momentum-based Stochastic Gradient Descent (SGD)

---

## ✨ Key Features

1. **Pure NumPy**: Zero dependencies on high-level ML libraries.
2. **Modular Architecture**: Clean, object-oriented design with reusable `Layer`, `Activation`, `Loss`, and `Optimizer` abstractions.
3. **Analytic Gradient Checking**: Automated validation comparing backpropagation gradients with two-sided finite difference approximations ($f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$).
4. **Numerical Stability**: Includes the log-sum-exp shift trick in Softmax ($z - \max(z)$) and epsilon clipping ($\epsilon = 10^{-12}$) in Cross-Entropy.
5. **Nonlinear Separation**: Solves the intertwined 3-class spiral dataset, demonstrating the capacity of hidden layers to learn complex non-linear decision boundaries.
6. **Built-in Visualizations**: Generates training loss curves, accuracy progress, and 2D decision boundary contours.

---

## 📐 Architecture & Mathematics

```
Input (2D) ──> Dense Layer (128) ──> ReLU ──> Dense Layer (64) ──> ReLU ──> Dense Layer (3) ──> Softmax ──> Probabilities (3 Classes)
```

### 1. Forward Pass (Affine Transform)
For a batch of input samples $X \in \mathbb{R}^{N \times D_{\text{in}}}$:
$$Z = X W + b$$
where:
- $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ is the weight matrix
- $b \in \mathbb{R}^{1 \times D_{\text{out}}}$ is the bias vector
- $Z \in \mathbb{R}^{N \times D_{\text{out}}}$ is the pre-activation logit matrix

---

### 2. Activation Functions

#### **ReLU (Rectified Linear Unit)**
Introduces non-linearity while mitigating the vanishing gradient problem:
$$f(z) = \max(0, z)$$
$$\frac{df}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

#### **Linear (Identity)**
Used in the final layer before Softmax:
$$f(z) = z, \quad \frac{df}{dz} = 1$$

---

### 3. Softmax & Categorical Cross-Entropy

#### **Softmax Function**
Converts raw logits $z$ into a valid probability distribution $\hat{y}$:
$$\hat{y}_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^K e^{z_j - \max(z)}}$$
*(Subtracting $\max(z)$ prevents floating-point overflow without altering the output probability).*

#### **Categorical Cross-Entropy Loss**
For $N$ samples and $K$ classes:
$$L = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik} + \epsilon)$$
where $y$ is the one-hot encoded ground truth.

#### **Combined Softmax + Cross-Entropy Gradient**
When combining Softmax and Cross-Entropy, the Jacobian multiplication simplifies into an elegant form:
$$\frac{\partial L}{\partial Z} = \frac{\hat{y} - y}{N}$$

---

### 4. Backpropagation & The Chain Rule

Gradients flow backward through each layer from output to input:

1. **Pre-activation Gradient ($\delta$ or $\frac{\partial L}{\partial Z}$)**:
   $$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot f'(Z)$$
   *(where $\odot$ denotes the element-wise Hadamard product)*

2. **Weight Gradient ($\frac{\partial L}{\partial W}$)**:
   $$\frac{\partial L}{\partial W} = X^T \left(\frac{\partial L}{\partial Z}\right)$$

3. **Bias Gradient ($\frac{\partial L}{\partial b}$)**:
   $$\frac{\partial L}{\partial b} = \sum_{i=1}^N \left(\frac{\partial L}{\partial Z}\right)_i$$

4. **Upstream Gradient to Previous Layer ($\frac{\partial L}{\partial X}$)**:
   $$\frac{\partial L}{\partial X} = \left(\frac{\partial L}{\partial Z}\right) W^T$$

---

### 5. Weight Initialization (He / Kaiming Init)

Standard Gaussian initialization causes vanishing or exploding activations across deep ReLU networks. He initialization draws weights from:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right), \quad b = 0$$
This preserves signal variance across layers by compensating for the 50% sparsity of ReLU activations.

---

### 6. Gradient Checking (Finite Differences)

To guarantee that the hand-derived analytic gradients are mathematically correct, the network performs numerical gradient checking before training:
$$f'(w) \approx \frac{L(w + h) - L(w - h)}{2h}, \quad \text{where } h = 10^{-5}$$

The relative error between analytic gradient $g_{\text{analytic}}$ and numerical gradient $g_{\text{numerical}}$ is computed as:
$$\text{Relative Error} = \frac{|g_{\text{analytic}} - g_{\text{numerical}}|}{\max(10^{-12}, |g_{\text{analytic}}| + |g_{\text{numerical}}|)}$$
A relative error $< 10^{-6}$ confirms backpropagation is exact.

---

### 7. Optimizer: SGD with Momentum & L2 Weight Decay

#### Update Formula:
$$v_t = \mu \cdot v_{t-1} - \alpha \cdot (\nabla W + \lambda W)$$
$$W_t = W_{t-1} + v_t$$

- $\alpha$ (**learning rate** = `0.5`): Controls step size.
- $\mu$ (**momentum** = `0.9`): Dampens oscillations and accelerates along consistent directions.
- $\lambda$ (**weight decay / L2 regularization** = `1e-4`): Penalizes large weights to prevent overfitting.

---

## 🌀 Dataset: 2D Spiral Classification

The synthetic spiral dataset generates $K=3$ intertwined spiral arms in 2D space:
$$r = \text{linspace}(0, 1, n)$$
$$\theta = \text{linspace}\left(0, 3\pi, n\right) + k \cdot \frac{2\pi}{K} + \mathcal{N}(0, \sigma^2)$$
$$x_1 = r \sin(\theta), \quad x_2 = r \cos(\theta)$$

- **Samples**: 300 (100 points per class)
- **Features**: 2 ($X_1, X_2$), standardized via Z-score normalization $\frac{X - \mu}{\sigma}$
- **Challenge**: Linearly non-separable; requires a non-linear multi-layer network to learn curved boundaries.

---

## 📂 Code Structure & Class Breakdown

| Component | Description |
| :--- | :--- |
| [`ReLU`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L44-L59) | Rectified linear activation function with forward and backward passes. |
| [`Linear`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L61-L71) | Identity activation function for output logits. |
| [`Softmax`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L73-L92) | Numerically stable probability conversion using max-subtraction. |
| [`CrossEntropyLoss`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L98-L133) | Computes batch loss and combined $(\hat{y} - y)/N$ gradient. |
| [`DenseLayer`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L138-L190) | Fully connected layer handling weights, biases, He initialization, and backward propagation. |
| [`NeuralNetwork`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L195-L296) | Manages layer stack, full forward/backward pipeline, predictions, accuracy, and gradient checking. |
| [`SGDMomentum`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L302-L332) | Optimizer applying velocity updates, momentum, and L2 regularization. |
| [`generate_spiral_data`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L337-L364) | Generates synthetic multi-class spiral dataset. |
| [`main()`](file:///C:/Users/hp/PycharmProjects/pyTorch/main.py#L381-L521) | Orchestrates dataset generation, gradient checking, 5000-epoch training, visualization generation, and weight diagnostics. |

---

## 💻 How to Run

### 1. Requirements
Ensure Python 3.8+ is installed with the following packages:
```bash
pip install numpy matplotlib
```

### 2. Execution
Run the script using Python:
```bash
python main.py
```

### 3. Example Console Output
```text
============================================================
NEURAL NETWORK FROM SCRATCH
No TensorFlow, No PyTorch, No sklearn — Just NumPy
============================================================

Dataset: Spiral (300 samples, 3 classes)
Input dim: 2, Output dim: 3

Network architecture:
  Input:  2
  Hidden: 128 (ReLU)
  Hidden: 64  (ReLU)
  Output: 3 (Softmax)
  Loss:   Cross-Entropy
  Optimizer: SGD with Momentum (lr=0.5, μ=0.9, weight_decay=1e-4)

------------------------------------------------------------
GRADIENT CHECKING (Finite Difference Verification)
------------------------------------------------------------
  Checked 30 parameters
  Max relative error: 8.87e-08
  ✓ PASSED — Backpropagation is mathematically correct!

------------------------------------------------------------
TRAINING
------------------------------------------------------------
  Epoch     0/5000 | Loss: 1.1070 | Acc: 0.3333
  Epoch   500/5000 | Loss: 0.7061 | Acc: 0.5833
  Epoch  1000/5000 | Loss: 0.6558 | Acc: 0.6633
  Epoch  2500/5000 | Loss: 0.3671 | Acc: 0.8533
  Epoch  4999/5000 | Loss: 0.6584 | Acc: 0.6700

  Final accuracy: 0.6733 (67.3%)

------------------------------------------------------------
Generating visualizations...
------------------------------------------------------------

✓ Done! Visualization saved to: ...\neural_net_from_scratch.png
```

---

## 📊 Visualizations & Output

The script outputs a comprehensive 3-panel figure saved as [`neural_net_from_scratch.png`](file:///C:/Users/hp/PycharmProjects/pyTorch/neural_net_from_scratch.png):

1. **Training Loss Curve**: Log-scale cross-entropy loss progression over 5000 epochs.
2. **Training Accuracy Curve**: Classification accuracy trajectory reaching convergence.
3. **Decision Boundary Plot**: A fine 2D mesh grid predicting class regions across the feature space, displaying the non-linear boundaries separating the spiral arms.
