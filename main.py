""""
A NEURAL NETWORK FROM SCRATCH — No Deep Learning Frameworks
============================================================
Implements a fully-connected multi-layer neural network using ONLY NumPy.

Features:
  1. Modular Layer/Network architecture (forward/backward interfaces)
  2. Custom activation functions (ReLU, Softmax) with analytic gradients
  3. Cross-entropy loss with numerical stability (log-sum-exp trick)
  4. Full backpropagation via the chain rule — computed by hand
  5. Gradient checking via finite differences — PROVES the math is correct
  6. Momentum-based SGD optimizer with L2 regularization
  7. He weight initialization (for ReLU networks)
  8. Input normalization for training stability
  9. Trains on the spiral dataset — a hard nonlinear classification problem

Mathematics involved:
  - Linear algebra:  z = Wx + b  (affine transform)
  - Calculus:  chain rule across composite functions
  - Softmax + cross-entropy:  ∂L/∂z = ŷ - y  (elegant simplification)
  - Finite differences:  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
  - Probability:  softmax turns logits into a probability distribution
  - Optimization:  gradient descent with momentum + regularization

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass



# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================
# Each activation provides:
#   forward(z)  → applies the nonlinearity
#   backward(z) → returns the derivative w.r.t. its input z
#
# The backward pass uses the *local gradient* — the derivative
# of the activation function evaluated at z — which gets
# multiplied by the upstream gradient during backpropagation.

class ReLU:
    """Rectified Linear Unit:  f(z) = max(0, z)

    The gradient is 1 where z > 0, and 0 where z ≤ 0.
    This sparsity is what makes deep networks trainable —
    it avoids the vanishing gradient problem of sigmoid/tanh.
    """
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        return (z > 0).astype(float)

    def __repr__(self):
        return "ReLU"


class Linear:
    """Identity activation: f(z) = z. Used on the output layer before softmax."""
    def forward(self, z):
        return z

    def backward(self, z):
        return np.ones_like(z)

    def __repr__(self):
        return "Linear"


class Softmax:
    """Softmax:  f(z)_i = e^{z_i} / Σ_j e^{z_j}

    Converts raw logits into a probability distribution.
    The gradient of softmax alone is a full Jacobian matrix, but when
    combined with cross-entropy loss, it simplifies to just (ŷ - y).
    So we apply softmax in the forward pass and handle the combined
    gradient in the loss function.

    Numerical stability: subtract max(z) before exponentiating.
    softmax(z) = softmax(z - max(z))  — prevents overflow in e^z.
    """
    def forward(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __repr__(self):
        return "Softmax"


# ============================================================
# LOSS FUNCTION
# ============================================================

class CrossEntropyLoss:
    """Categorical cross-entropy loss.

    L = -Σ_i y_i * log(ŷ_i)   (averaged over the batch)

    where y is the one-hot true label and ŷ is the predicted probability.

    THE MAGIC OF SOFTMAX + CROSS-ENTROPY:
    When you differentiate L through the softmax, the chain rule
    collapses beautifully:

        ∂L/∂z = ŷ - y

    Derivation:
      ∂L/∂ŷ_i = -y_i / ŷ_i
      ∂ŷ_i/∂z_j = ŷ_i(δ_ij - ŷ_j)        (softmax Jacobian)

    Multiplying:
      ∂L/∂z_j = Σ_i (-y_i/ŷ_i) · ŷ_i(δ_ij - ŷ_j)
               = Σ_i -y_i(δ_ij - ŷ_j)
               = -y_j + ŷ_j · Σ_i y_i
               = ŷ_j - y_j          (since Σ_i y_i = 1 for one-hot)

    This is why softmax + cross-entropy is the standard for classification.
    """
    def forward(self, probs, y_onehot):
        """Compute the average cross-entropy loss."""
        n = probs.shape[0]
        return -np.sum(y_onehot * np.log(probs + 1e-12)) / n

    def backward(self, probs, y_onehot):
        """Combined gradient of softmax + cross-entropy: (ŷ - y) / n."""
        n = probs.shape[0]
        return (probs - y_onehot) / n


# ============================================================
# DENSE LAYER
# ============================================================

class DenseLayer:
    """A fully-connected (dense) layer:  z = xW + b,  a = activation(z).

    Forward:   z = x @ W + b    →    a = activation(z)
    Backward:  propagates ∂L/∂a upstream and computes ∂L/∂W, ∂L/∂b

    He weight initialization:  W ~ N(0, sqrt(2/fan_in))
    This is critical for ReLU networks:
      - Too large → exploding gradients
      - Too small → vanishing gradients
      - He init keeps signal variance stable across layers by
        compensating for ReLU's 50% sparsity (2x factor).
    """
    def __init__(self, input_dim, output_dim, activation, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)
        self.W = rng.normal(0, scale, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.activation = activation

        # Gradients (filled during backward pass)
        self.dW = None
        self.db = None

        # Momentum buffers (for the optimizer)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        # Cache for backward pass
        self._z = None   # pre-activation
        self._x = None   # layer input

    def forward(self, x):
        """Forward pass: z = xW + b, then a = activation(z)."""
        self._x = x
        self._z = x @ self.W + self.b
        return self.activation.forward(self._z)

    def backward(self, grad_output):
        """Backward pass: given ∂L/∂a, compute ∂L/∂W, ∂L/∂b, ∂L/∂x.

        Chain rule through this layer:
            ∂L/∂z = ∂L/∂a · ∂a/∂z         (activation derivative)
            ∂L/∂W = x^T @ ∂L/∂z            (weight gradient)
            ∂L/∂b = Σ_batch ∂L/∂z           (bias gradient)
            ∂L/∂x = ∂L/∂z @ W^T             (propagate upstream)
        """
        dz = grad_output * self.activation.backward(self._z)
        self.dW = self._x.T @ dz
        self.db = np.sum(dz, axis=0)
        return dz @ self.W.T


# ============================================================
# NEURAL NETWORK
# ============================================================

class NeuralNetwork:
    """A multi-layer perceptron with softmax output.

    Architecture:  [input] → Dense(ReLU) → Dense(ReLU) → ... → Dense → Softmax

    Forward flows left-to-right through layers, then softmax.
    Backward flows right-to-left (reverse order), propagating gradients.
    """
    def __init__(self, layer_sizes, seed=42):
        self.loss_fn = CrossEntropyLoss()
        self.softmax = Softmax()
        self.layers = []

        rng_seed = seed
        for i in range(len(layer_sizes) - 1):
            # Hidden layers use ReLU; output layer is linear (softmax
            # is applied separately in the loss, so no activation here)
            if i < len(layer_sizes) - 2:
                act = ReLU()
            else:
                act = Linear()

            layer = DenseLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                activation=act,
                seed=rng_seed + i
            )
            self.layers.append(layer)

    def forward(self, x):
        """Forward pass through all layers, then softmax → probabilities."""
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return self.softmax.forward(a)

    def backward(self, probs, y_onehot):
        """Backpropagation: start from (ŷ - y) and chain-rule backward."""
        grad = self.loss_fn.backward(probs, y_onehot)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def compute_loss(self, probs, y_onehot):
        return self.loss_fn.forward(probs, y_onehot)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def accuracy(self, x, y):
        return np.mean(self.predict(x) == y)

    # --------------------------------------------------------
    # GRADIENT CHECKING
    # --------------------------------------------------------
    # Compares analytic gradients (from backprop) against numerical
    # gradients from central finite differences:
    #
    #   ∂L/∂w ≈ [L(w + h) - L(w - h)] / (2h)
    #
    # Relative error < 1e-6 means backprop is correct.
    # This is the gold-standard validation used in real deep learning libraries.

    def gradient_check(self, x, y_onehot, num_checks=50, h=1e-5):
        """Verify backprop gradients against finite differences."""
        probs = self.forward(x)
        self.backward(probs, y_onehot)

        details = []
        max_rel_error = 0.0
        rng = np.random.default_rng(123)

        for layer_idx, layer in enumerate(self.layers):
            for param_name, param, grad in [('W', layer.W, layer.dW),
                                             ('b', layer.b, layer.db)]:
                flat_size = param.size
                check_indices = rng.choice(flat_size,
                                           size=min(num_checks, flat_size),
                                           replace=False)
                for idx in check_indices:
                    orig = param.flat[idx]

                    param.flat[idx] = orig + h
                    loss_plus = self.compute_loss(self.forward(x), y_onehot)

                    param.flat[idx] = orig - h
                    loss_minus = self.compute_loss(self.forward(x), y_onehot)

                    param.flat[idx] = orig

                    num_grad = (loss_plus - loss_minus) / (2 * h)
                    ana_grad = grad.flat[idx]

                    denom = max(1e-12, abs(num_grad) + abs(ana_grad))
                    rel_error = abs(num_grad - ana_grad) / denom

                    max_rel_error = max(max_rel_error, rel_error)
                    details.append((layer_idx, param_name, ana_grad,
                                    num_grad, rel_error))

        return max_rel_error, details


# ============================================================
# SGD OPTIMIZER WITH MOMENTUM + L2 REGULARIZATION
# ============================================================

class SGDMomentum:
    """SGD with momentum and optional L2 weight decay.

    Update rule:
        v = μ·v - lr·(grad + λ·W)
        W = W + v

    Parameters:
        lr: learning rate
        momentum: μ ∈ [0, 1), typically 0.9
            Accelerates in consistent gradient directions and damps oscillation.
        weight_decay: λ — L2 regularization strength
            Adds λ·W to the gradient, pulling weights toward zero.
    """
    def __init__(self, lr=0.1, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self, network):
        for layer in network.layers:
            grad_W = layer.dW + self.weight_decay * layer.W
            grad_b = layer.db

            layer.vW = self.momentum * layer.vW - self.lr * grad_W
            layer.vb = self.momentum * layer.vb - self.lr * grad_b

            layer.W += layer.vW
            layer.b += layer.vb


# ============================================================
# SPIRAL DATASET
# ============================================================

def generate_spiral_data(n_points_per_class=100, n_classes=3,
                         noise_std=0.10, seed=42):
    """Generate a spiral classification dataset.

    Each class is a spiral arm:
        r = linspace(0, 1, n)
        θ = linspace(0, 3π, n) + i·(2π/K)
        x = r·sin(θ) + noise
        y = r·cos(θ) + noise

    This is a classic benchmark — classes are deeply intertwined and
    no linear separator exists. The model must learn curved boundaries.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_points_per_class * n_classes, 2))
    y = np.zeros(n_points_per_class * n_classes, dtype=int)

    for i in range(n_classes):
        start = i * n_points_per_class
        end = start + n_points_per_class
        r = np.linspace(0.0, 1.0, n_points_per_class)
        t = np.linspace(0, 3*np.pi, n_points_per_class) + i*(2*np.pi/n_classes)
        X[start:end, 0] = r * np.sin(t) + rng.normal(0, noise_std, n_points_per_class)
        X[start:end, 1] = r * np.cos(t) + rng.normal(0, noise_std, n_points_per_class)
        y[start:end] = i

    return X, y


def one_hot_encode(y, n_classes):
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y] = 1
    return onehot


def normalize(X):
    """Standardize to zero mean, unit variance — critical for training."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Data ---
    N_CLASSES = 3
    X, y = generate_spiral_data(n_points_per_class=100, n_classes=N_CLASSES,
                                noise_std=0.10, seed=42)
    y_onehot = one_hot_encode(y, N_CLASSES)
    X = normalize(X)

    print("=" * 60)
    print("NEURAL NETWORK FROM SCRATCH")
    print("No TensorFlow, No PyTorch, No sklearn — Just NumPy")
    print("=" * 60)
    print(f"\nDataset: Spiral ({len(y)} samples, {N_CLASSES} classes)")
    print(f"Input dim: {X.shape[1]}, Output dim: {N_CLASSES}")

    # --- Network: [2 → 128 → 64 → 3] ---
    net = NeuralNetwork(layer_sizes=[2, 128, 64, N_CLASSES], seed=42)

    print(f"\nNetwork architecture:")
    print(f"  Input:  {X.shape[1]}")
    print(f"  Hidden: 128 (ReLU)")
    print(f"  Hidden: 64  (ReLU)")
    print(f"  Output: {N_CLASSES} (Softmax)")
    print(f"  Loss:   Cross-Entropy")
    print(f"  Optimizer: SGD with Momentum (lr=0.5, μ=0.9, weight_decay=1e-4)")

    # --- GRADIENT CHECKING ---
    print("\n" + "-" * 60)
    print("GRADIENT CHECKING (Finite Difference Verification)")
    print("-" * 60)

    check_size = min(20, len(y))
    max_err, details = net.gradient_check(
        X[:check_size], y_onehot[:check_size], num_checks=30, h=1e-5)

    print(f"  Checked {len(details)} parameters")
    print(f"  Max relative error: {max_err:.2e}")
    if max_err < 1e-6:
        print(f"  ✓ PASSED — Backpropagation is mathematically correct!")
    elif max_err < 1e-4:
        print(f"  ~ ACCEPTABLE — Small numerical error (float precision)")
    else:
        print(f"  ✗ FAILED — Backpropagation has a bug!")

    print(f"\n  Sample checks (first 5):")
    print(f"  {'Layer':>5} {'Param':>5} {'Analytic':>12} {'Numerical':>12} {'Rel Error':>12}")
    for layer_idx, p_name, ana, num, rel in details[:5]:
        print(f"  {layer_idx:>5} {p_name:>5} {ana:>12.6f} {num:>12.6f} {rel:>12.2e}")

    # --- TRAINING ---
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)

    optimizer = SGDMomentum(lr=0.5, momentum=0.9, weight_decay=1e-4)
    EPOCHS = 5000
    losses = []
    train_accs = []

    for epoch in range(EPOCHS):
        # Full-batch gradient descent
        probs = net.forward(X)
        loss = net.compute_loss(probs, y_onehot)
        acc = net.accuracy(X, y)

        losses.append(loss)
        train_accs.append(acc)

        net.backward(probs, y_onehot)
        optimizer.step(net)

        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:>5d}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    final_acc = net.accuracy(X, y)
    print(f"\n  Final accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")

    # --- VISUALIZATION ---
    print("\n" + "-" * 60)
    print("Generating visualizations...")
    print("-" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Training Loss
    ax = axes[0]
    ax.plot(losses, color='royalblue', lw=1.2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training Loss\n(SGD with Momentum)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Training Accuracy
    ax = axes[1]
    ax.plot(train_accs, color='forestgreen', lw=1.2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Training Accuracy\n(Final: {final_acc*100:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 3: Decision Boundary
    ax = axes[2]
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_preds = net.predict(mesh_points).reshape(xx.shape)

    cmap_bg = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_pts = ListedColormap(['#FF0000', '#00AA00', '#0000FF'])
    ax.contourf(xx, yy, mesh_preds, cmap=cmap_bg, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
               edgecolors='white', s=30, linewidth=0.5)
    ax.set_xlabel('X₁', fontsize=12)
    ax.set_ylabel('X₂', fontsize=12)
    ax.set_title('Learned Decision Boundary\n(Nonlinear spiral separation)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout(pad=2.0)
    output_filename = 'neural_net_from_scratch.png'
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    plt.savefig(output_path,
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Done! Visualization saved to: {output_path}")

    # --- Weight Statistics ---
    print("\n" + "-" * 60)
    print("WEIGHT STATISTICS (post-training)")
    print("-" * 60)
    for i, layer in enumerate(net.layers):
        print(f"  Layer {i}: W mean={layer.W.mean():.4f}, "
              f"std={layer.W.std():.4f}, "
              f"min={layer.W.min():.4f}, max={layer.W.max():.4f}")


if __name__ == "__main__":
    main()
