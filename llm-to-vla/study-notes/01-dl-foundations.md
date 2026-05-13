# 01 — Deep Learning Foundations & Information Theory

> **Phase I · Days 1–7 · ~17.5 hours**
> Prerequisites: Basic ML (loss functions, backprop, SGD/Adam), Python + PyTorch, linear algebra
> Learning Objectives: Solidify DL foundations, understand WHY attention was invented, grasp compression = prediction = intelligence

---

## The Thread: Compression = Prediction = Intelligence

Before we begin, understand the thread that connects **everything** in this curriculum:

> "Prediction is compression. And compression is understanding."
> — Ilya Sutskever

If you can predict the next token perfectly, you have **perfectly compressed** the data — which means you have captured every statistical regularity, every causal pattern, every abstract concept embedded in the training distribution. A model that achieves optimal compression of internet text **must** learn syntax, semantics, world knowledge, reasoning, and even theory of mind — because all of these help predict the next word.

This idea — that intelligence emerges from compression — is the deep reason why:
- Cross-entropy is the right loss function (it measures compression efficiency)
- Scale works (more parameters = better compressor)
- Language models exhibit emergent capabilities (compression forces abstraction)
- The same architecture works for vision, language, and action (they're all sequences to compress)

Keep this thread in mind through every section. We'll trace it explicitly.

---

## Table of Contents

1. [Computation Graphs & Backprop (Day 1)](#1-computation-graphs--backprop-day-1)
2. [CNNs & ResNets (Day 2)](#2-cnns--resnets-day-2)
3. [RNN/LSTM Essentials (Day 3)](#3-rnnlstm-essentials-day-3)
4. [Seq2Seq & The Bottleneck (Day 4)](#4-seq2seq--the-bottleneck-day-4)
5. [Information Theory & Compression (Day 5)](#5-information-theory--compression-day-5)
6. [Embeddings & Representation Learning (Day 6)](#6-embeddings--representation-learning-day-6)
7. [Training Stability Cookbook (Day 7)](#7-training-stability-cookbook-day-7)
8. [Key Takeaways](#8-key-takeaways)
9. [Paper References](#9-paper-references)
10. [Connection to the Thread](#10-connection-to-the-thread)

---

## 1. Computation Graphs & Backprop (Day 1)

**~2.5 hours** · Goal: understand how gradients actually flow through a neural network

### 1.1 Computation Graph Representation

Every neural network computation can be represented as a **directed acyclic graph (DAG)** where:
- **Nodes** are operations (add, multiply, relu, matmul, …)
- **Edges** carry tensors (data flows forward, gradients flow backward)
- **Leaves** are inputs and parameters

Example: a simple 2-layer MLP computing $L = \text{loss}(\sigma(xW_1 + b_1)W_2 + b_2, y)$

```
         x ──→ [matmul] ──→ [add] ──→ [σ] ──→ [matmul] ──→ [add] ──→ [loss] ──→ L
                  ↑            ↑                   ↑            ↑          ↑
                 W₁           b₁                  W₂           b₂         y
```

The computation graph makes the chain rule **mechanical**: to find $\frac{\partial L}{\partial W_1}$, you just multiply the local gradients along every path from $L$ back to $W_1$.

### 1.2 The Chain Rule Through Graphs

For a composition $L = f(g(h(x)))$, the chain rule gives:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

In a computation graph with **multiple paths** (e.g., skip connections), gradients **sum** over all paths:

$$\frac{\partial L}{\partial x} = \sum_{\text{paths } p \text{ from } x \text{ to } L} \prod_{\text{edges } e \in p} \frac{\partial e_{\text{out}}}{\partial e_{\text{in}}}$$

This summation property is crucial — it's why ResNets work (more paths = better gradient flow).

### 1.3 Forward Mode vs Reverse Mode AD

There are two ways to evaluate the chain rule through a computation graph:

| | Forward Mode | Reverse Mode (Backprop) |
|---|---|---|
| **Direction** | Input → Output | Output → Input |
| **Computes** | $\frac{\partial \text{all outputs}}{\partial \text{one input}}$ | $\frac{\partial \text{one output}}{\partial \text{all inputs}}$ |
| **Cost per pass** | O(1) per input variable | O(1) per output variable |
| **Good when** | Few inputs, many outputs | Few outputs (loss!), many inputs |

**Why deep learning uses reverse mode**: We have ONE scalar output (the loss) but MILLIONS of parameters. Reverse mode computes ALL gradients in ONE backward pass — the cost is ~2-3× the forward pass regardless of parameter count.

$$\text{Forward mode cost} = O(n_{\text{params}}) \quad \text{vs} \quad \text{Reverse mode cost} = O(n_{\text{outputs}}) = O(1)$$

> 💡 **Key Insight**: Backpropagation IS reverse-mode automatic differentiation applied to a computation graph. It's not a heuristic — it's an exact, efficient algorithm for computing all partial derivatives of a scalar loss.

### 1.4 PyTorch Autograd

PyTorch builds the computation graph dynamically (define-by-run):

```python
import torch

# Parameters (leaves with grad tracking)
W = torch.randn(3, 2, requires_grad=True)
b = torch.zeros(2, requires_grad=True)

# Forward pass — graph is built as you compute
x = torch.randn(4, 3)
y_true = torch.randint(0, 2, (4,))

logits = x @ W + b                    # graph: matmul → add
loss = torch.nn.functional.cross_entropy(logits, y_true)  # graph: → softmax → log → nll

# Backward pass — reverse-mode AD through the recorded graph
loss.backward()

# Gradients are now populated
print(W.grad.shape)  # (3, 2) — same shape as W
print(b.grad.shape)  # (2,)   — same shape as b

# IMPORTANT: gradients accumulate! Zero them before next step.
W.grad.zero_()
b.grad.zero_()

# Use no_grad() for inference (no graph recording = faster, less memory)
with torch.no_grad():
    predictions = (x @ W + b).argmax(dim=1)
```

Key mechanics:
- `.backward()` — triggers reverse-mode AD, populates `.grad` on all leaf tensors
- `.grad` — accumulates gradients (must zero between steps!)
- `torch.no_grad()` — disables graph recording for inference
- `.detach()` — creates a tensor that shares data but stops gradient flow
- `retain_graph=True` — keeps graph for multiple backward passes (rare, expensive)

### 1.5 Micrograd: A Tiny Autograd Engine

Understanding autograd deeply means building one. Here's the core idea (inspired by Karpathy's micrograd):

```python
class Value:
    """A scalar value with automatic differentiation."""
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0  # dL/d(self)
        self._backward = lambda: None  # local gradient propagation function
        self._prev = set(_children)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad   # += because gradients accumulate!
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        
        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        """Reverse-mode AD: topological sort then propagate."""
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

# Usage
x = Value(2.0)
w = Value(-3.0)
b = Value(1.0)
y = (x * w + b).relu()
y.backward()
print(f"dy/dx = {x.grad}")  # -3.0 (because relu gate is open)
print(f"dy/dw = {w.grad}")  #  2.0
```

The key pattern: each operation records a `_backward` function that computes local gradients and propagates them. The `backward()` method does a topological sort and calls these in reverse order.

> ⚠️ **Common Pitfall**: Note the `+=` in gradient accumulation. If a variable is used multiple times in the graph (like in `x * x`), gradients from both uses must be **summed**. Forgetting this is the #1 bug in custom autograd.

### 1.6 What This Means for VLAs

VLA models have computation graphs spanning **vision encoders → language models → action decoders** — billions of parameters, trained end-to-end with backprop. Understanding gradient flow through these massive graphs is essential for:
- Knowing which parts to freeze vs fine-tune
- Diagnosing training instabilities (where do gradients vanish/explode?)
- Understanding why certain architectural choices (residual connections, normalization) are non-negotiable at scale

---

## 2. CNNs & ResNets (Day 2)

**~2.5 hours** · Goal: understand spatial feature hierarchies and the residual revolution

### 2.1 Convolution as Learnable Feature Extraction

A convolution layer slides a small kernel (typically 3×3 or 5×5) across an image, computing dot products at each position. This gives two critical properties:

1. **Translation equivariance**: a cat detected in the top-left uses the same features as one in the bottom-right
2. **Parameter sharing**: one 3×3×C kernel has only 9C parameters regardless of image size

```
Input Image (H×W×C)    Kernel (k×k×C)      Feature Map (H'×W'×1)
┌─────────────┐        ┌───┐
│ . . . . . . │        │w w│               ┌───────────┐
│ . [a b] . . │   ×    │w w│      →        │ . . Σ . . │
│ . [c d] . . │        └───┘               │ . . . . . │
│ . . . . . . │                            └───────────┘
└─────────────┘     Σ = aw₁+bw₂+cw₃+dw₄
```

With $C_{\text{out}}$ kernels, the output has $C_{\text{out}}$ channels — each channel is a different learned feature detector.

### 2.2 Receptive Field, Stride, Padding, Dilation

**Receptive field**: the region of the input image that influences a particular output pixel.

```
Layer 1 (3×3):  receptive field = 3×3
Layer 2 (3×3):  receptive field = 5×5
Layer 3 (3×3):  receptive field = 7×7
...
Each layer adds 2 pixels (for 3×3 kernels) to the receptive field.
```

**Stride**: skip positions during convolution. Stride 2 halves spatial dimensions (efficient downsampling).

**Padding**: add zeros around borders. `padding=1` with 3×3 kernel preserves spatial dimensions.

**Dilation**: spread kernel elements apart. A 3×3 kernel with dilation 2 has an effective 5×5 receptive field with only 9 parameters.

Output size formula:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2 \cdot \text{padding} - \text{dilation} \cdot (k - 1) - 1}{\text{stride}} + 1 \right\rfloor$$

### 2.3 Feature Hierarchy

CNNs learn a **hierarchy of increasingly abstract features**:

```
Layer 1:  Edges, gradients, colors
Layer 2:  Corners, textures, simple shapes  
Layer 3:  Parts — eyes, wheels, windows
Layer 4:  Objects — faces, cars, buildings
Layer 5+: Scenes, relationships, context
```

This hierarchy is not designed — it **emerges** from training. Early layers learn low-level features because they're useful for ALL visual tasks; deeper layers specialize.

> 💡 **Key Insight**: This hierarchical feature extraction is the visual analog of how language models learn syntax (early layers) → semantics (middle layers) → reasoning (deep layers). Both are **compression**: finding efficient representations that capture the structure of the input distribution.

### 2.4 The Depth Problem

Naively stacking more layers should give better features — more abstraction, larger receptive fields. But in practice, deeper networks (>20 layers) trained **worse** than shallow ones:

```
Training error vs depth (pre-ResNet):
Layers:  20      56
Error:   6.0%    8.0%   ← WORSE! Not just overfitting — training error is higher.
```

This isn't overfitting (training error is higher too). It's an **optimization problem**: deep networks are harder to train because gradients must flow through many multiplicative stages.

### 2.5 Residual Connections: Gradient Highways

ResNets (He et al., 2015) add **skip connections** that create shortcut paths for gradient flow:

$$y = F(x) + x$$

where $F(x)$ is the "residual" — the network only needs to learn the **difference** from identity.

```
         ┌──────────────────────┐
         │                      │  (skip / identity)
         │                      ▼
x ──→ [Conv] ──→ [BN] ──→ [ReLU] ──→ [Conv] ──→ [BN] ──→ (+) ──→ [ReLU] ──→ y
         │                                                  ▲
         └── This is F(x) ─────────────────────────────────┘
                                                     y = F(x) + x
```

Why this works — three perspectives:

1. **Gradient flow**: $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$. The identity term means gradients always have a direct path, preventing vanishing.

2. **Ensemble view**: A ResNet with $n$ blocks can be unrolled into an ensemble of $2^n$ paths of different lengths. Most gradient signal flows through shorter paths.

3. **Compression view**: Skip connections let the network learn **refinements** to existing representations rather than entirely new ones. This is more parameter-efficient — a better compressor.

### 2.6 ResNet Block in PyTorch

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Standard ResNet basic block with pre-activation."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # THE skip connection


class SimpleResNet(nn.Module):
    """Minimal ResNet for CIFAR-10."""
    def __init__(self, num_blocks: int = 4, channels: int = 64):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 10),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

# Count gradient paths: 2^4 = 16 paths through 4 blocks!
model = SimpleResNet(num_blocks=4)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

> ⚠️ **Common Pitfall**: The skip connection requires input and output dimensions to match. When changing spatial dimensions or channel count, use a 1×1 conv "projection shortcut": `self.shortcut = nn.Conv2d(c_in, c_out, 1, stride=2)`.

### 2.7 What This Means for VLAs

VLA models use **vision encoders** (ViT, SigLIP) that are direct descendants of CNN ideas:
- ViT replaces convolution with attention but keeps the hierarchical feature extraction concept
- Many VLAs (RT-2, Octo) use pretrained vision backbones — transfer learning from ImageNet features
- The residual connection is **universal** — every transformer layer uses it. It's not optional at scale.

---

## 3. RNN/LSTM Essentials (Day 3)

**~2.5 hours** · Goal: understand sequential processing and the vanishing gradient problem that DEMANDS attention

### 3.1 Vanilla RNN

An RNN processes a sequence one element at a time, maintaining a hidden state:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$
$$y_t = W_{hy} \cdot h_t + b_y$$

```
          y₁         y₂         y₃         y₄
          ↑          ↑          ↑          ↑
         [W_hy]     [W_hy]     [W_hy]     [W_hy]
          ↑          ↑          ↑          ↑
h₀ ──→ [RNN] ──→ [RNN] ──→ [RNN] ──→ [RNN] ──→ h₄
          ↑          ↑          ↑          ↑
         x₁         x₂         x₃         x₄
```

Training uses **Backpropagation Through Time (BPTT)**: unroll the RNN for $T$ steps and apply standard backprop.

### 3.2 The Vanishing Gradient Problem

This is THE fundamental limitation that motivated the invention of attention. Understand it deeply.

The gradient of the loss at time $T$ with respect to hidden state at time $t$ involves:

$$\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Each factor is:

$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(\tanh'(z_k)) \cdot W_{hh}$$

where $\tanh'(z) \in (0, 1]$. The product of $T - t$ such matrices either:

- **Vanishes**: if the largest singular value of $W_{hh}$ is < 1, the product shrinks exponentially → gradients from distant past vanish → the model cannot learn long-range dependencies
- **Explodes**: if > 1, the product grows exponentially → training becomes unstable (solved by gradient clipping, but vanishing is harder)

```
Gradient magnitude vs distance:

For σ_max(W_hh) = 0.9:
  Distance 10:  0.9^10  ≈ 0.35   (still ok)
  Distance 50:  0.9^50  ≈ 0.005  (nearly zero)
  Distance 100: 0.9^100 ≈ 0.00003 (effectively zero)

The model CANNOT learn that token 1 affects token 100.
```

> 💡 **Key Insight**: The vanishing gradient problem isn't a bug — it's a **fundamental limitation** of propagating information through multiplicative chains. This is why attention was invented: it creates **direct connections** between any two positions, bypassing the multiplicative chain entirely.

### 3.3 LSTM: Gradient Highways Through Time

Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) introduces a **cell state** $c_t$ that flows through time with **additive** updates (not multiplicative!), plus **gates** to control information flow:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$  — **Forget gate**: what to erase from cell
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$  — **Input gate**: what new info to write
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$  — **Candidate values**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$  — **Cell update** (ADDITIVE!)
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$  — **Output gate**: what to expose
$$h_t = o_t \odot \tanh(c_t)$$

```
LSTM Cell:
                                    c_{t-1}
                                      │
                    ┌─────────────────│─────────────────────┐
                    │                 ▼                     │
                    │  ┌─────┐    ┌─────┐                  │
                    │  │  ×  │←── │ f_t │ (forget gate)    │
                    │  └──┬──┘    └─────┘                  │
                    │     │          ↑                     │
                    │     │    ┌────────┐                  │
                    │     ▼    │        │                  │
                    │  ┌─────┐ │  ┌─────┐   ┌──────┐      │
                    │  │  +  │←┤  │  ×  │←──│ i_t  │      │
                    │  └──┬──┘ │  └──┬──┘   └──────┘      │
                    │     │    │     ↑          ↑         │
                    │     │    │  ┌──────┐     │         │
                    │     │    │  │ c̃_t  │     │         │
                    │     │    │  └──────┘     │         │
                    │     │    │     ↑          │         │
                    │     ▼    │     │          │         │   c_t
                    │  ┌──────┐│  ┌──────────────┐       │───→
                    │  │ tanh ││  │  [h_{t-1}, x_t] │    │
                    │  └──┬───┘│  └──────────────┘       │
                    │     │    │     ↑          ↑         │
                    │  ┌─────┐ │  ┌──────┐               │
                    │  │  ×  │←┤  │ o_t  │ (output gate) │
                    │  └──┬──┘    └──────┘               │
                    │     │                               │
                    └─────│───────────────────────────────┘
                          ▼
                         h_t
```

**Why LSTM helps**: The cell state gradient is:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

When $f_t \approx 1$ (forget gate open), gradients flow through **unchanged** — no vanishing. The network **learns** which information to preserve vs forget.

### 3.4 The Remaining Limitation

Even LSTMs hit a wall:

```
LSTM effective dependency range:
  Reliable:    ~50-100 tokens
  Degraded:    ~100-200 tokens
  Unreliable:  >200 tokens
  
Compare to attention:
  Reliable:    ALL positions (up to context length)
  Cost:        O(n²) — but that's the price of direct access
```

The problem is that LSTM still processes **sequentially** — information from position 1 must pass through positions 2, 3, ..., n to reach position n. Even with gating, this is lossy.

> 💡 **Key Insight**: LSTMs are like a **telephone game** — information must be whispered through every intermediate position. Attention is like **everyone being in the same room** — any position can directly query any other position. This is the architectural leap that enables modern AI.

### 3.5 What This Means for VLAs

- RNNs teach us that sequential processing creates information bottlenecks
- VLA action sequences (robot trajectories) are temporal — but we don't process them with RNNs
- Instead, VLAs use **attention** over discretized action tokens, giving every action access to the full visual and language context
- Understanding WHY RNNs fail makes the transformer architecture feel **inevitable** rather than arbitrary

---

## 4. Seq2Seq & The Bottleneck (Day 4)

**~2.5 hours** · Goal: understand the specific failure that attention was designed to fix

### 4.1 Encoder-Decoder Architecture

Seq2Seq (Sutskever et al., 2014) uses one RNN to **encode** a variable-length input into a fixed-size vector, and another RNN to **decode** that vector into a variable-length output:

```
Encoder:
  "The cat sat on the mat" → [h₁] → [h₂] → [h₃] → [h₄] → [h₅] → [h₆]
                                                                      │
                                                              context vector c
                                                                      │
Decoder:                                                              ▼
  c → [d₁] → "Le"   [d₂] → "chat"   [d₃] → "s'est"   [d₄] → "assis" ...
```

### 4.2 The Bottleneck Problem

Here's the critical failure: the ENTIRE source sentence is compressed into ONE fixed-size vector $c = h_T^{\text{enc}}$.

For short sentences (5-10 words), this works reasonably well. But:

```
BLEU score vs sentence length (Cho et al., 2014):

Length:   5    10    15    20    25    30    35    40
BLEU:   34    30    25    20    16    13    11     9
         ↑good              gradually                ↑terrible
                            collapsing
```

Think about what this means: you're asking a 512-dimensional vector to encode:
- Every word and its meaning
- Syntactic structure and dependencies  
- Word order and relationships
- Nuance, negation, conditionals

For a 40-word sentence with complex structure, this is **information-theoretically impossible** with a fixed-size bottleneck.

> 💡 **Key Insight**: The bottleneck problem is really a **compression** problem. A fixed-size vector is a lossy compressor, and for complex inputs, the information loss becomes catastrophic. Attention solves this by letting the decoder look at **all** encoder hidden states — the full "uncompressed" representation.

### 4.3 Teacher Forcing

During training, the decoder receives the **true** previous token as input (not its own prediction):

```
Training (teacher forcing):
  Input to decoder at time t: y_{t-1}  (ground truth)
  
Inference (autoregressive):
  Input to decoder at time t: ŷ_{t-1}  (model's own prediction)
```

This creates a **train-test mismatch** (exposure bias): the model never sees its own errors during training, so error accumulation during inference can be severe.

### 4.4 Beam Search

Greedy decoding (always pick highest-probability token) is suboptimal. Beam search maintains the top-$k$ candidates at each step:

```
Beam width = 3:

Step 1:  "The" (0.4)   "A" (0.3)    "One" (0.1)
Step 2:  "The cat" (0.35)  "A cat" (0.25)  "The dog" (0.20)
Step 3:  ...continue expanding top-3 at each step...
```

Beam search finds better translations but increases compute linearly with beam width.

### 4.5 The Setup for Attention

Let's be explicit about what we need:

1. ❌ **Fixed bottleneck** — compressing everything into one vector loses information
2. ❌ **Sequential processing** — information from early tokens degrades through the chain
3. ✅ **What we want**: the decoder should be able to **look at any part of the input** at any time, weighted by **relevance** to the current output position

This is exactly what attention provides. Bahdanau et al. (2014) proposed:

$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})} \quad \text{where} \quad e_{t,s} = a(s_{t-1}, h_s)$$

$$c_t = \sum_s \alpha_{t,s} \cdot h_s$$

Instead of one context vector, we compute a **different** context vector for each decoder step — a weighted combination of ALL encoder states, where the weights are learned based on relevance.

This is the bridge to Study Note 02 (Attention & Transformers).

### 4.6 What This Means for VLAs

- VLAs face the same bottleneck if you try to compress a visual scene into a single vector
- Modern VLAs (RT-2, OpenVLA) use attention between visual tokens and language tokens, so the action decoder can attend to the **specific** visual features relevant to each action
- The seq2seq bottleneck is why we DON'T compress a camera image into one embedding — we keep it as spatial tokens

---

## 5. Information Theory & Compression (Day 5) — CRITICAL SECTION

**~2.5 hours** · Goal: understand the mathematical foundation of "compression = prediction = intelligence"

This section is the **intellectual backbone** of the entire curriculum. Every major advance in deep learning — from better architectures to scaling laws — can be understood through the lens of compression.

### 5.1 Shannon Entropy

Claude Shannon (1948) defined the fundamental quantity of information theory:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x) \quad \text{[bits]}$$

Entropy measures the **minimum average bits needed** to encode samples from distribution $X$:

```
Example — English characters:
  If all 26 letters equally likely: H = log₂(26) = 4.7 bits/char
  With real English frequencies:     H ≈ 4.0 bits/char
  With digram statistics:            H ≈ 3.3 bits/char
  With word-level patterns:          H ≈ 1.5 bits/char
  Shannon's estimate for English:    H ≈ 0.6-1.3 bits/char

Each level captures MORE STRUCTURE → requires FEWER BITS → better COMPRESSION.
A model that achieves H ≈ 1.0 bits/char has captured most of English structure.
```

> 💡 **Key Insight**: Entropy is the **theoretical compression limit**. No lossless coding scheme can do better than $H(X)$ bits per symbol on average. Any compressor that approaches this limit must have **learned the distribution** $p(x)$.

### 5.2 Cross-Entropy: The Loss Function IS Compression Efficiency

Cross-entropy between the true distribution $p$ and our model $q$:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

This measures: **how many bits our model $q$ uses, on average, to encode data from true distribution $p$**.

In the language modeling setting:
- $p$ = the true next-token distribution (one-hot in practice: the actual next token)
- $q$ = our model's predicted distribution over next tokens
- $H(p, q)$ = the cross-entropy loss we minimize

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log q(x_i | x_{<i})$$

This is literally the negative log-likelihood — the standard training objective. And it's literally measuring **how well our model compresses the data**.

### 5.3 KL Divergence: The Compression Gap

KL divergence measures how much EXTRA encoding cost our model incurs compared to the optimal:

$$D_{KL}(p \| q) = H(p, q) - H(p) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} \geq 0$$

```
Cross-entropy = Entropy + KL divergence
    H(p,q)    =  H(p)   +  D_KL(p||q)
      ↑            ↑           ↑
  our model's   theoretical    gap = how much
  compression   minimum        worse we are
  cost          (irreducible)  than optimal
```

When we minimize cross-entropy loss, we're minimizing KL divergence — pushing our model $q$ toward the true distribution $p$. A model with $D_{KL} = 0$ has **perfectly learned** the data distribution.

### 5.4 The Deep Connection: Prediction = Compression = Understanding

Here's the core argument that connects everything:

**Claim**: Optimal next-token prediction and optimal data compression are **the same task**.

**Proof sketch**:
1. By Shannon's source coding theorem, optimal lossless compression requires $H(p)$ bits per symbol
2. Arithmetic coding achieves this bound given access to the true distribution $p$
3. Therefore: a model that predicts $q = p$ (perfect next-token prediction) enables **optimal compression**
4. Conversely: a compressor that achieves the entropy bound has implicitly **learned the distribution**

$$\text{Better prediction} \Leftrightarrow \text{Lower cross-entropy} \Leftrightarrow \text{Better compression}$$

**Why this implies understanding**:

To predict the next word in "The capital of France is ___", you must know:
- Syntax (noun phrase expected)
- Semantics (capital = major city)
- World knowledge (France's capital is Paris)
- Context tracking (what was asked)

A model that achieves low cross-entropy on diverse text **must** have internal representations that capture all of these — because they're all useful for prediction.

> 💡 **Key Insight**: This is Ilya Sutskever's central argument for why scale works. As you compress harder (lower cross-entropy), you're forced to discover **deeper** structure in the data. First you learn word frequencies, then syntax, then semantics, then world knowledge, then reasoning. Each layer of understanding provides diminishing but non-zero compression gains. Intelligence is what you get when you push compression far enough.

### 5.5 Cross-Entropy as THE Loss Function

Why do we use cross-entropy rather than MSE, hinge loss, or something else?

$$\mathcal{L}_{\text{CE}} = -\log q(x_{\text{true}})$$

1. **Information-theoretically optimal**: minimizing CE = minimizing KL divergence = approaching the true distribution as fast as possible

2. **Proper scoring rule**: the unique minimum of expected CE is achieved when $q = p$ (the true distribution). Other loss functions can have minima at wrong distributions.

3. **Gradient quality**: $\frac{\partial \mathcal{L}}{\partial z_k} = q_k - \mathbb{1}[k = y]$ (for softmax output). This gradient is simple, well-scaled, and non-zero even when the prediction is confident. Compare to MSE on probabilities, which has vanishing gradients when predictions are near 0 or 1.

4. **Bits-per-character**: $\text{BPC} = \frac{\mathcal{L}_{\text{CE}}}{\ln 2}$ gives a universal, interpretable metric. A model achieving 1.0 BPC on English text is near-human compression.

### 5.6 Perplexity and Bits-Per-Character

Two common metrics, both derived from cross-entropy:

$$\text{Perplexity} = 2^{H(p,q)} = \exp(\mathcal{L}_{\text{CE}})$$

Perplexity = the effective vocabulary size the model is "confused" among. PPL = 10 means the model is, on average, equally uncertain among 10 candidates.

$$\text{Bits-per-character (BPC)} = \frac{H(p,q)}{\text{avg chars per token}}$$

BPC is the most fundamental metric — it measures raw compression efficiency in the universal unit of bits.

```
Compression benchmarks on English text:
  Unigram model:     ~4.5 BPC
  Bigram model:      ~3.5 BPC
  LSTM (2016):       ~1.3 BPC
  GPT-2 (2019):      ~0.97 BPC
  GPT-4 (est):       ~0.7 BPC
  Shannon's estimate: 0.6-1.3 BPC
  Theoretical limit:  ~0.6 BPC (?)
  
The progression toward the entropy limit IS the history of NLP.
```

### 5.7 The Solomonoff-Hutter Connection (Advanced)

For the theoretically inclined: Marcus Hutter's AIXI framework formalizes this connection. The **Solomonoff prior** assigns probability to sequences based on the length of the shortest program that generates them:

$$p(x) \propto \sum_{\text{programs } \pi : U(\pi) = x} 2^{-|\pi|}$$

This is **Kolmogorov complexity** meets Bayesian inference. The key insight: the universal prior IS optimal compression, and optimal prediction under this prior IS a form of general intelligence.

We can't compute Kolmogorov complexity (it's uncomputable!), but neural networks are **approximating** it — learning compressed representations of the data-generating process.

### 5.8 What This Means for VLAs

- VLAs are trained to predict action tokens — this IS compression of sensorimotor experience
- A robot that can predict "what action comes next given this image and instruction" has learned a compressed model of physical interaction
- The scaling hypothesis for robotics: bigger models = better compression of physical world dynamics = more capable robots
- Cross-entropy on action tokens is not just a convenient loss — it's the **information-theoretically correct** objective for learning to act

---

## 6. Embeddings & Representation Learning (Day 6)

**~2.5 hours** · Goal: understand how discrete symbols become continuous vectors — the foundation of all neural language processing

### 6.1 Why One-Hot Fails

Naive encoding: each word gets a binary vector with one 1 and the rest 0s.

```
Vocabulary of 50,000 words:
  "cat"  → [0, 0, ..., 1, 0, ..., 0]  (50,000-dim, one 1)
  "dog"  → [0, 0, ..., 0, 1, ..., 0]  (50,000-dim, one 1)
  "the"  → [0, 1, ..., 0, 0, ..., 0]  (50,000-dim, one 1)

Problems:
  1. cos("cat", "dog") = 0      — no similarity despite being related
  2. cos("cat", "the") = 0      — same distance as "cat" to "dog"!
  3. 50,000 dimensions for 1 bit of info — absurdly wasteful
  4. No generalization: learning about "cat" tells you nothing about "dog"
```

All vectors are orthogonal. The representation contains ZERO information about relationships between words.

### 6.2 Distributed Representations

The fix: map each word to a **dense, low-dimensional** vector where **geometry encodes meaning**:

```
Embedding space (d=256):
  "cat"  → [0.21, -0.45, 0.78, ..., 0.33]   256 dimensions
  "dog"  → [0.19, -0.42, 0.81, ..., 0.30]   similar to "cat"!
  "the"  → [-0.65, 0.12, -0.03, ..., 0.89]  very different

cos("cat", "dog") ≈ 0.95  — nearby in embedding space
cos("cat", "the") ≈ 0.05  — far apart
```

Each dimension captures a latent feature (not necessarily interpretable). Similar words cluster together, and the **direction** between words encodes relationships.

### 6.3 Word2Vec: Learning Embeddings from Co-occurrence

Mikolov et al. (2013) showed that simple objectives produce remarkably structured embedding spaces:

**Skip-gram**: predict context words from center word
**CBOW**: predict center word from context

$$\mathcal{L}_{\text{skip-gram}} = -\frac{1}{T}\sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$$

where $p(w_o | w_i) = \frac{\exp(v'_{w_o} \cdot v_{w_i})}{\sum_w \exp(v'_w \cdot v_{w_i})}$

### 6.4 Embedding Arithmetic

The most famous result — embeddings capture **relational structure**:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

This works because the embedding space organizes concepts along **consistent directions**:

```
                    "royal" direction
                    ──────────────→
          man ─────────────────── king
           │                       │
  "gender" │                       │ "gender"
  direction│                       │ direction
           │                       │
           ▼                       ▼
         woman ────────────────── queen
                    ──────────────→
                    "royal" direction
```

Other examples:
- Paris - France + Italy ≈ Rome (capital relationship)
- walking - walk + swim ≈ swimming (tense transformation)
- bigger - big + small ≈ smaller (comparative form)

> 💡 **Key Insight**: Embedding arithmetic works because the network discovered that encoding **relationships as directions** in vector space is the most **compressed** representation of word co-occurrence patterns. This is compression in action: instead of memorizing each word independently, learn the structure.

### 6.5 Subword Tokenization (Preview)

Modern LLMs don't embed whole words — they use **subword** units:

**Byte-Pair Encoding (BPE)**:
1. Start with character vocabulary: {a, b, c, ..., z, ...}
2. Repeatedly merge the most frequent adjacent pair
3. "unhappiness" → "un" + "happiness" → "un" + "hap" + "piness"

```
BPE vocabulary building:
  Iteration 1: merge "t" + "h" → "th"     (most frequent pair)
  Iteration 2: merge "th" + "e" → "the"   (next most frequent)
  Iteration 3: merge "i" + "n" → "in"
  ...continue until vocabulary reaches target size (32k-100k)
```

**WordPiece** (used by BERT): similar but uses likelihood instead of frequency for merging.

**Why subwords**: handles unknown words, morphology, and reduces vocabulary size. "Unhappiness" is rare but "un" + "happi" + "ness" are common — the model can compose meaning from parts.

### 6.6 The Embedding Matrix

In practice, embeddings are a simple lookup table:

```python
import torch
import torch.nn as nn

# Embedding layer: vocabulary size × embedding dimension
embed = nn.Embedding(num_embeddings=50000, embedding_dim=256)

# Input: token IDs (integers)
token_ids = torch.tensor([42, 1337, 7, 42])  # batch of 4 tokens

# Output: dense vectors
vectors = embed(token_ids)  # shape: (4, 256)

# Under the hood, this is just:
# vectors = embed.weight[token_ids]  — a fancy indexing operation!

# The embedding matrix has shape (50000, 256)
# It's a learnable parameter, trained end-to-end with the rest of the model
print(f"Embedding parameters: {embed.weight.numel():,}")  # 12,800,000
```

> ⚠️ **Common Pitfall**: Embeddings for rare tokens are poorly trained (few gradient updates). This is why subword tokenization helps — even rare words are composed of well-trained subword pieces.

### 6.7 Connection to Compression

Embeddings are literally **compression**: mapping a sparse, 50,000-dimensional one-hot space to a dense, 256-dimensional space. The information-theoretic minimum for encoding 50,000 symbols is $\log_2(50000) \approx 15.6$ bits. A 256-dim float32 vector uses 8,192 bits — seemingly wasteful! But those extra dimensions encode **relationships** — the structure of language — which makes downstream processing much more efficient.

The embedding layer is the first compression step in the neural pipeline:

```
One-hot (50,000 dim, 1 bit of info)
    → Embedding (256 dim, ~16 bits of symbol + relationship info)
        → Hidden layers (progressive compression of context)
            → Output distribution (compressed prediction)
```

### 6.8 What This Means for VLAs

- VLAs embed MULTIPLE modalities: vision patch tokens, language tokens, and discretized action tokens all live in the **same** embedding space
- The magic of VLAs is that a single transformer can attend across visual embeddings and language embeddings because they share a geometry
- Discretized robot actions (position bins, rotation bins) are embedded just like words — the model learns that "move left a little" and "move left a lot" have nearby embeddings

---

## 7. Training Stability Cookbook (Day 7)

**~2.5 hours** · Goal: practical knowledge to train deep networks without exploding/crashing

### 7.1 Common Training Failures

```
Failure Modes:

1. NaN loss           — gradient explosion → overflow → NaN → unrecoverable
2. Loss spike         — sudden jump, may recover or cascade to NaN  
3. Loss plateau       — stuck at suboptimal point, not learning
4. Gradient vanishing — loss decreases but agonizingly slowly
5. Oscillation        — loss bounces up and down, no convergence
6. Divergence         — loss steadily increases
```

### 7.2 Gradient Norm Monitoring — The Most Important Diagnostic

The single most informative training diagnostic is the **gradient norm** over time:

$$\|\nabla\| = \sqrt{\sum_i \left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)^2}$$

```python
# Monitor gradient norms during training
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# In training loop:
loss.backward()
grad_norm = get_grad_norm(model)

# Healthy training: grad norm is stable, slowly decreasing
# Trouble signs:
#   - Spikes (>10× normal):     explosion starting
#   - Drops to ~0:              vanishing gradients
#   - Steady increase:          learning rate too high
#   - Wild oscillation:         batch size too small or LR too high

# Gradient clipping — insurance against explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

> 💡 **Key Insight**: If you monitor only ONE thing during training, monitor gradient norms. NaN losses, training instabilities, and learning failures almost always show up in gradient norms **before** they show up in the loss curve.

### 7.3 Learning Rate Schedules

The learning rate is the single most impactful hyperparameter.

**Linear Warmup** (critical for transformers):

```python
def warmup_schedule(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps  # linear ramp
    return max_lr
```

Why warmup? At initialization, model outputs are random → gradients point in arbitrary directions → a large LR amplifies this noise → unstable. Warmup lets the model find a reasonable region first.

**Cosine Decay** (standard after warmup):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}} \cdot \pi\right)\right)$$

```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Combined warmup + cosine decay
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=100000,
    pct_start=0.05,      # 5% warmup
    anneal_strategy='cos',
    div_factor=25,        # initial_lr = max_lr / 25
    final_div_factor=1e4, # final_lr  = initial_lr / 10000
)

# In training loop:
for step, batch in enumerate(dataloader):
    loss = compute_loss(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

```
Learning rate over training:

LR │      ╱─────────╲
   │     ╱            ╲
   │    ╱               ╲
   │   ╱                  ╲
   │  ╱                     ╲
   │ ╱                        ╲___
   │╱                              
   └──────────────────────────────── step
   warmup    cosine decay      final
```

### 7.4 Mixed Precision Training

Modern GPUs are 2-8× faster in FP16/BF16 than FP32. But lower precision introduces risks:

```
Precision comparison:
  FP32:  1 sign + 8 exp + 23 mantissa  → range ±3.4e38,  precision ~7 decimal digits
  FP16:  1 sign + 5 exp + 10 mantissa  → range ±65504,   precision ~3 decimal digits
  BF16:  1 sign + 8 exp +  7 mantissa  → range ±3.4e38,  precision ~2 decimal digits
```

**FP16 dangers**:
- Overflow: gradients > 65504 → Inf → NaN (common in early training!)
- Underflow: gradients < 6e-8 → rounded to 0 → training stalls

**Loss scaling** (for FP16):
```python
# Manual loss scaling
scale = 2**16  # 65536
(loss * scale).backward()               # scale up before backward
for p in model.parameters():
    if p.grad is not None:
        p.grad.data /= scale            # scale down gradients
optimizer.step()

# PyTorch AMP (automatic mixed precision) — preferred approach
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**BF16** (preferred when available): same range as FP32 (no overflow!), less precision. Available on A100, H100, and newer GPUs. Almost always preferred over FP16 for training.

> ⚠️ **Common Pitfall**: BF16 is safe for most training, but FP16 requires loss scaling. If you see NaN losses on a new GPU, check whether you're accidentally using FP16 without a GradScaler.

### 7.5 Weight Initialization

Bad initialization → bad gradient flow from step 0.

**Xavier (Glorot) initialization** — for linear layers with tanh/sigmoid:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

Maintains variance of activations through layers: $\text{Var}(h_l) \approx \text{Var}(h_{l-1})$.

**Kaiming (He) initialization** — for layers with ReLU:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

The $\sqrt{2}$ factor compensates for ReLU zeroing out half the activations.

```python
# PyTorch applies sensible defaults, but for custom layers:
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)

# For transformer layers, common practice:
# Scale residual branch initialization by 1/sqrt(2*num_layers)
# This prevents residual stream magnitude from growing with depth
```

### 7.6 Training Stability Checklist

When training goes wrong, check in this order:

```
□ 1. Learning rate
     - Too high? → loss oscillates, spikes, or diverges
     - Too low?  → loss decreases agonizingly slowly
     - Try: sweep LR from 1e-6 to 1e-1, plot loss vs LR

□ 2. Gradient norms
     - Exploding (>100)?  → reduce LR, add gradient clipping
     - Vanishing (<1e-7)? → check residual connections, initialization
     - Steady growth?     → weight decay may be too low

□ 3. NaN detection
     - Where does NaN first appear? (use torch.autograd.detect_anomaly())
     - Common causes: log(0), division by zero, FP16 overflow

□ 4. Data pipeline
     - Are labels correct? (visualize a batch!)
     - Are inputs normalized? (mean ≈ 0, std ≈ 1)
     - Any NaN in the data?

□ 5. Architecture
     - Missing residual connections?
     - Missing LayerNorm/BatchNorm?
     - Activation function mismatch with initialization?

□ 6. Warmup
     - Transformers NEED warmup (500-4000 steps typical)
     - Without warmup, attention weights are random → huge gradients

□ 7. Batch size
     - Too small (<16)? → noisy gradients, unstable
     - Too large (>4096)? → may need LR scaling, more warmup
     - Linear scaling rule: double batch → double LR (approximately)

□ 8. Weight decay
     - Typical: 0.01-0.1 for AdamW
     - DON'T apply to biases and LayerNorm parameters
```

### 7.7 AdamW: The Standard Optimizer

```python
# AdamW (decoupled weight decay) — standard for transformers
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # peak learning rate
    betas=(0.9, 0.999), # momentum and second moment
    eps=1e-8,           # numerical stability
    weight_decay=0.1,   # L2 regularization (decoupled!)
)

# Why AdamW over Adam?
# Adam applies weight decay to the gradient, NOT the weight directly.
# This couples regularization with the adaptive learning rate.
# AdamW decouples them: update = adam_update + λ * weight
# This gives better generalization in practice.
```

### 7.8 What This Means for VLAs

- VLA training combines vision, language, and action losses — each can have different gradient scales
- Mixed precision is **essential** for training multi-billion parameter VLAs (memory and speed)
- Training stability at scale is an active research problem — techniques like μP (maximal update parameterization) help
- Gradient monitoring across modalities reveals which parts of the model are actually learning

---

## 8. Key Takeaways

### The Compression Thread Through Phase I

```
Day 1: Backprop         → The engine that optimizes compression efficiency
Day 2: CNNs/ResNets     → Hierarchical spatial compression with gradient highways
Day 3: RNNs/LSTMs       → Sequential compression with vanishing gradient limits
Day 4: Seq2Seq          → The bottleneck IS a compression failure
Day 5: Info Theory       → Compression = Prediction = Intelligence (formalized)
Day 6: Embeddings        → Discrete → continuous compression of symbols
Day 7: Training          → Practical tools to make compression training work
```

### Most Important Insights (Ranked)

1. **Prediction IS compression** — minimizing cross-entropy IS maximizing compression. A perfect predictor is a perfect compressor, and vice versa. This is not metaphor — it's mathematical equivalence.

2. **The vanishing gradient problem is the fundamental motivation for attention** — RNNs and even LSTMs cannot reliably propagate information across long distances. This isn't a minor limitation — it's a fundamental architectural ceiling.

3. **Residual connections are mandatory at scale** — they provide gradient highways and enable training networks with hundreds of layers. Every transformer layer uses them.

4. **The seq2seq bottleneck is what attention solves** — compressing a variable-length input into a fixed-size vector loses information catastrophically. Attention provides dynamic, content-dependent access to the full input.

5. **Cross-entropy is the right loss function** — it's not arbitrary. It's the information-theoretically optimal objective for learning to predict (compress) data.

6. **Embeddings are the bridge between discrete and continuous** — they compress sparse symbols into dense vectors where geometry encodes meaning. This enables neural networks to process language (and actions).

7. **Training stability is practical craft** — gradient norm monitoring, warmup, mixed precision, and proper initialization are not optional. They're the difference between a model that trains and one that produces NaN.

### What You Should Be Able To Do After Phase I

- [ ] Explain WHY backprop uses reverse-mode AD (complexity argument)
- [ ] Derive why gradients vanish in deep RNNs (product of Jacobians)
- [ ] Explain the seq2seq bottleneck and why attention fixes it
- [ ] State the connection: cross-entropy ↔ compression ↔ prediction
- [ ] Read the gradient norm curve and diagnose common failures
- [ ] Implement a ResNet block with proper skip connections
- [ ] Reason about when/why LSTMs fail on long sequences

---

## 9. Paper References

### Foundational

1. **Backpropagation**: Rumelhart, Hinton & Williams (1986). "Learning representations by back-propagating errors." *Nature*.
   — The paper that made backprop practical.

2. **ResNets**: He et al. (2015). ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385).
   — Skip connections enabling 152-layer networks. Won ILSVRC 2015.

3. **LSTM**: Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." *Neural Computation*.
   — Gradient highways through time with learned gates.

4. **Seq2Seq**: Sutskever et al. (2014). ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215).
   — Encoder-decoder with the bottleneck we'll fix with attention.

### Information Theory & Compression

5. **Shannon (1948)**. "A Mathematical Theory of Communication." *Bell System Technical Journal*.
   — The founding document of information theory.

6. **Hutter Prize**: Marcus Hutter. [The Hutter Prize for Lossless Compression of Human Knowledge](http://prize.hutter1.net/).
   — Prizes for text compression, explicitly linking compression to intelligence.

7. **Scaling Laws**: Kaplan et al. (2020). ["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361).
   — Empirical laws connecting model size, data, and compression efficiency (loss).

### Embeddings

8. **Word2Vec**: Mikolov et al. (2013). ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781).
   — Distributed word representations with arithmetic properties.

9. **BPE**: Sennrich et al. (2015). ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909).
   — Subword tokenization enabling open-vocabulary NLP.

### Training Stability

10. **BatchNorm**: Ioffe & Szegedy (2015). ["Batch Normalization: Accelerating Deep Network Training"](https://arxiv.org/abs/1502.03167).

11. **Adam**: Kingma & Ba (2014). ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980).

12. **Decoupled Weight Decay**: Loshchilov & Hutter (2017). ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101).
    — AdamW — the standard optimizer for transformers.

13. **Kaiming Init**: He et al. (2015). ["Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852).

### Compression = Intelligence (The Thread)

14. **Sutskever's Lecture**: Ilya Sutskever. "An Observation on Generalization" (various talks, 2023).
    — The compression-prediction-intelligence argument presented informally.

15. **Deletang et al. (2024)**. ["Language Modeling Is Compression"](https://arxiv.org/abs/2309.10668).
    — Formal demonstration that language models are general-purpose compressors.

---

## 10. Connection to the Thread

### Compression = Prediction = Intelligence

Phase I establishes the **foundation** of this thread:

```
                    PHASE I ESTABLISHES
                    ══════════════════
                    
Shannon Entropy ──→ Cross-entropy loss IS compression efficiency
       │
       ▼
KL Divergence  ──→ Training minimizes the compression gap
       │
       ▼
Prediction     ──→ Better prediction = better compression = more understanding
       │
       ▼
Scaling        ──→ More parameters = better compressor = emergent capabilities
       │
       ▼
VLAs           ──→ Compressing vision + language + action jointly
                   = understanding the physical world
```

**Where we go next** (Phase II preview):

Study Note 02 will introduce **attention** — the mechanism that replaced the seq2seq bottleneck. And it will introduce **transformers** — the architecture that made attention practical at scale.

The key question attention answers: "How do I let every position directly access every other position, with learned relevance weights, without the multiplicative chain that kills RNNs?"

The answer — and it's beautiful in its simplicity — is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

But that's for next time.

---

> **Navigation**: [← Curriculum Overview](../CURRICULUM.md) · **01 DL Foundations** · [02 Attention & Transformers →](02-attention-transformers.md)
