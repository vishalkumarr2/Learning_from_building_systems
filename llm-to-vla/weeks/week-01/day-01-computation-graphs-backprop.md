# Day 1: Computation Graphs & Backpropagation

> **Phase I · Week 1 · Day 1 of 112 · 2.5 hours**

*"If you really understand backpropagation, everything else — transformers, diffusion, VLAs — is 'just' clever graph design."*

---

## Navigation

| | |
|---|---|
| **Previous** | — Start of curriculum |
| **Next** | [Day 2: CNN & ResNets →](day-02-cnn-resnets.md) |
| **Week** | [Week 1: DL Foundations](README.md) |
| **Phase** | [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md) |
| **Curriculum** | [Full Curriculum](../../CURRICULUM.md) |

---

## Why This Matters

Every neural network — from a 3-layer MLP to a 70B-parameter VLA — is a computation graph. Training it means computing gradients through that graph. Today you'll understand *exactly* how gradients flow, why reverse-mode AD is the only viable option for deep learning, and build your own autograd engine from scratch.

By the end of Day 112, you'll be working with graphs that take camera images, process them through a vision encoder, fuse them with language instructions in a transformer, and output robot joint commands. The gradient must flow through *all* of that. It starts here.

---

## 1. Theory: Computation Graphs as DAGs

### 1.1 What Is a Computation Graph?

A neural network computation is a **directed acyclic graph (DAG)** where:
- **Nodes** = operations (add, multiply, relu, matmul)
- **Edges** = tensors (data flowing between operations)
- **Leaves** = inputs and parameters
- **Root** = the loss scalar

```
         ┌─────┐
    x ──▶│  *  │──▶ z ──▶ ┌─────┐
         │     │          │     │
    w ──▶│     │     b ──▶│  +  │──▶ a ──▶ ┌──────┐──▶ L
         └─────┘          │     │          │ Loss │
                          └─────┘          └──────┘
                                    y ────▶│      │

    z = w * x
    a = z + b
    L = loss(a, y)
```

**Key insight**: The graph records *what happened* during the forward pass, creating a recipe for computing gradients in the backward pass.

### 1.2 The Chain Rule Through Graphs

For a simple chain $L = f(g(h(x)))$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

When a node has **multiple paths** to the loss (e.g., a parameter used twice), gradients **sum**:

$$\frac{\partial L}{\partial x} = \sum_{i \in \text{paths}} \frac{\partial L}{\partial x}\bigg|_{\text{path } i}$$

This is the multivariate chain rule. It's why `grad` accumulates, not replaces.

### 1.3 Forward Mode vs Reverse Mode AD

| Property | Forward Mode | Reverse Mode (Backprop) |
|----------|-------------|------------------------|
| Computes | $\frac{\partial \text{all outputs}}{\partial \text{one input}}$ | $\frac{\partial \text{one output}}{\partial \text{all inputs}}$ |
| Direction | Input → Output | Output → Input |
| Cost per pass | One Jacobian column | One Jacobian row |
| # passes needed | One per input | One per output |
| When optimal | Few inputs, many outputs | **Few outputs (=1 loss), many inputs (=millions of params)** |

**Why reverse mode wins for deep learning**: We have 1 scalar loss and millions of parameters. Reverse mode computes *all* gradients in a single backward pass. Forward mode would require one pass per parameter — impossibly expensive.

$$\text{Forward mode cost: } O(n_{\text{params}}) \quad \text{vs} \quad \text{Reverse mode cost: } O(1)$$

### 1.4 The Backward Pass — Local Gradients

Each node stores its **local gradient** — the derivative of its output w.r.t. its inputs. During backprop, it receives the **upstream gradient** (from the loss side) and computes:

$$\text{downstream gradient} = \text{upstream gradient} \times \text{local gradient}$$

For common operations:

| Operation | Forward | Local gradient w.r.t. inputs |
|-----------|---------|------------------------------|
| $c = a + b$ | sum | $\frac{\partial c}{\partial a} = 1, \quad \frac{\partial c}{\partial b} = 1$ |
| $c = a \times b$ | product | $\frac{\partial c}{\partial a} = b, \quad \frac{\partial c}{\partial b} = a$ |
| $c = \text{ReLU}(a)$ | max(0,a) | $\frac{\partial c}{\partial a} = \mathbb{1}[a > 0]$ |
| $c = \sigma(a)$ | sigmoid | $\frac{\partial c}{\partial a} = \sigma(a)(1 - \sigma(a))$ |
| $C = A \cdot B$ | matmul | $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T, \quad \frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$ |

### 1.5 PyTorch Autograd Mechanics

PyTorch builds the computation graph dynamically (define-by-run):

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Forward pass — graph is built implicitly
z = w * x          # MulBackward node created
a = z + b          # AddBackward node created
L = (a - 5.0)**2   # PowBackward, SubBackward

# Backward pass — gradients computed
L.backward()

print(f"dL/dw = {w.grad}")  # 2 * (w*x + b - 5) * x = 2*(7-5)*2 = 8
print(f"dL/dx = {x.grad}")  # 2 * (w*x + b - 5) * w = 2*(7-5)*3 = 12
print(f"dL/db = {b.grad}")  # 2 * (w*x + b - 5) * 1 = 2*(7-5)*1 = 4
```

**Critical detail**: `requires_grad=True` tells PyTorch to track this tensor in the graph. Tensors without it are treated as constants.

---

## 2. Implementation: Build a Micrograd Engine

Build a scalar-valued autograd engine from scratch (inspired by Karpathy's micrograd). This is the single most illuminating exercise in all of deep learning.

### 2.1 The Value Class

```python
import math

class Value:
    """A scalar value with automatic differentiation."""
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # dL/d(self)
        self._backward = lambda: None  # backward function
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad   # d(a+b)/da = 1
            other.grad += 1.0 * out.grad  # d(a+b)/db = 1
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad  # d(a*b)/da = b
            other.grad += self.data * out.grad  # d(a*b)/db = a
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'σ')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        """Topological sort + reverse-order backward pass."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1.0  # dL/dL = 1
        for node in reversed(topo):
            node._backward()
```

### 2.2 Build a Neuron, Layer, and MLP

```python
import random

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(n_in)]
        self.b = Value(0.0, label='b')
    
    def __call__(self, x):
        # w · x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### 2.3 Training Loop

```python
# Create a simple dataset: learn XOR-like function
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4, 1 output

for epoch in range(100):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Zero gradients
    for p in model.parameters():
        p.grad = 0.0
    
    # Backward pass
    loss.backward()
    
    # Update (SGD)
    lr = 0.01
    for p in model.parameters():
        p.data -= lr * p.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.4f}")
```

### 2.4 Verify Against PyTorch

```python
import torch

# Same computation in PyTorch
x_pt = torch.tensor([2.0, 3.0], requires_grad=True)
w_pt = torch.tensor([0.5, -1.0], requires_grad=True)
b_pt = torch.tensor([0.1], requires_grad=True)

z = (x_pt * w_pt).sum() + b_pt
a = torch.relu(z)
L = (a - 1.0)**2
L.backward()

# Same computation in our micrograd
x0, x1 = Value(2.0), Value(3.0)
w0, w1 = Value(0.5), Value(-1.0)
b = Value(0.1)

z_mg = x0*w0 + x1*w1 + b
a_mg = z_mg.relu()
L_mg = (a_mg - Value(1.0))**2
L_mg.backward()

# Compare
print(f"PyTorch:  dL/dw0={w_pt.grad[0].item():.4f}  dL/dw1={w_pt.grad[1].item():.4f}")
print(f"Micrograd: dL/dw0={w0.grad:.4f}  dL/dw1={w1.grad:.4f}")
# Should match exactly!
```

---

## 3. Exercises

### Exercise 1: Draw the Computation Graph

Draw the full computation graph for a 3-layer MLP with ReLU activations and MSE loss:

```
Input: x ∈ R²
Layer 1: h₁ = ReLU(W₁x + b₁)     W₁ ∈ R³ˣ², b₁ ∈ R³
Layer 2: h₂ = ReLU(W₂h₁ + b₂)    W₂ ∈ R²ˣ³, b₂ ∈ R²
Layer 3: ŷ  = W₃h₂ + b₃          W₃ ∈ R¹ˣ², b₃ ∈ R¹
Loss:    L  = (ŷ - y)²
```

Label every intermediate node. Count the total number of trainable parameters.

**Answer**: $3 \times 2 + 3 + 2 \times 3 + 2 + 1 \times 2 + 1 = 6 + 3 + 6 + 2 + 2 + 1 = 20$ parameters.

### Exercise 2: Manual Gradient Computation

Given $x = 2, w_1 = 3, w_2 = -1, b = 0.5$:

$$z_1 = w_1 \cdot x = 6$$
$$z_2 = z_1 + b = 6.5$$
$$z_3 = \text{ReLU}(z_2) = 6.5$$
$$z_4 = w_2 \cdot z_3 = -6.5$$
$$L = z_4^2 = 42.25$$

Compute $\frac{\partial L}{\partial w_1}$, $\frac{\partial L}{\partial w_2}$, $\frac{\partial L}{\partial b}$ by hand.

<details>
<summary>Solution</summary>

Working backward:
- $\frac{\partial L}{\partial z_4} = 2 z_4 = -13.0$
- $\frac{\partial z_4}{\partial w_2} = z_3 = 6.5 \implies \frac{\partial L}{\partial w_2} = -13.0 \times 6.5 = -84.5$
- $\frac{\partial z_4}{\partial z_3} = w_2 = -1 \implies \frac{\partial L}{\partial z_3} = -13.0 \times (-1) = 13.0$
- $z_2 > 0$ so $\frac{\partial z_3}{\partial z_2} = 1 \implies \frac{\partial L}{\partial z_2} = 13.0$
- $\frac{\partial z_2}{\partial b} = 1 \implies \frac{\partial L}{\partial b} = 13.0$
- $\frac{\partial z_2}{\partial z_1} = 1 \implies \frac{\partial L}{\partial z_1} = 13.0$
- $\frac{\partial z_1}{\partial w_1} = x = 2 \implies \frac{\partial L}{\partial w_1} = 13.0 \times 2 = 26.0$

</details>

### Exercise 3: Gradient Accumulation

Add a `cross_entropy` method to the `Value` class for binary classification:

$$L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

where $\hat{y} = \sigma(z)$ is the model output after sigmoid.

### Exercise 4: Gradient Flow Visualization

Using PyTorch, create a 10-layer MLP (no skip connections). After a forward+backward pass, plot the gradient norms $\|\frac{\partial L}{\partial W_\ell}\|$ for each layer $\ell$. What pattern do you see? (This previews tomorrow's lesson on ResNets.)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(10)])
x = torch.randn(32, 64)
y = torch.randn(32, 64)

out = model(x)
loss = ((out - y)**2).mean()
loss.backward()

norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None and p.dim() == 2]
plt.plot(norms)
plt.xlabel("Layer")
plt.ylabel("Gradient Norm")
plt.title("Gradient norms across 10 layers (no skip connections)")
plt.show()
```

---

## 4. Key Takeaways

- A neural network is a **DAG** of differentiable operations
- **Reverse-mode AD** (backprop) computes all parameter gradients in one backward pass — $O(1)$ passes regardless of parameter count
- Each node applies the chain rule locally: `downstream_grad = upstream_grad × local_grad`
- When paths merge, gradients **sum** (multivariate chain rule)
- The `backward()` function is just a **topological sort** followed by reverse-order local gradient multiplication
- PyTorch's autograd builds the graph dynamically — every forward pass creates a fresh graph
- Zeroing gradients before each backward pass is critical (otherwise they accumulate across iterations)

---

## 5. The Thread: Compression = Prediction = Intelligence

Today's lesson is about *infrastructure* — the machinery that makes learning possible. But notice: backpropagation is itself a form of **information compression**. Instead of storing the full Jacobian matrix ($n_{\text{outputs}} \times n_{\text{params}}$), reverse-mode AD compresses it into a single backward pass by exploiting the chain rule's factorization structure.

This is a recurring theme: **the most powerful algorithms find compressed representations** of expensive computations. Transformers will compress sequential dependency into parallel attention. VLAs will compress camera images + language into action distributions. The gradient computation you just built is the engine that drives all of it.

---

## 6. Further Reading

- **Karpathy, micrograd** — [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd) — The original inspiration
- **Karpathy, "The spelled-out intro to neural networks"** — YouTube lecture building micrograd live
- **Baydin et al., "Automatic Differentiation in Machine Learning: a Survey"** (2018) — The definitive AD reference
- **PyTorch Autograd docs** — `torch.autograd` internals and hooks
- **Griewank & Walther, "Evaluating Derivatives"** (2008) — The textbook on AD theory
