# Day 19: Normalization + Activations — The Hidden Architecture Decisions

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"The devil is in the details — and the details are in the normalization layer."*

## Navigation
- **Previous:** [Day 18: KV Cache](day-18-kv-cache.md)
- **Next:** [Day 20: Mixture of Experts](day-20-mixture-of-experts.md)
- **Week:** [Week 3 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/03-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 19.1 Why Normalization Matters

Without normalization, transformer training is fragile. Activations grow or shrink exponentially across layers:

```
Layer 1 output:  mean ≈ 0.0, std ≈ 1.0   ← fine
Layer 8 output:  mean ≈ 0.3, std ≈ 4.2   ← drifting
Layer 16 output: mean ≈ 1.7, std ≈ 18.1  ← unstable
Layer 32 output: mean ≈ ???, std ≈ ???    ← NaN city
```

**LayerNorm** (Ba et al., 2016) normalizes across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$

Key: normalizes each token's feature vector independently — no batch dependency.

### 19.2 Post-LN vs Pre-LN

The original transformer (Vaswani et al., 2017) used **Post-LN**:

```
Post-LN (original):                    Pre-LN (modern):
┌──────────┐                           ┌──────────┐
│    x      │                           │    x      │
├──────────┤                           ├──────────┤
│  Attn(x)  │                           │  LN(x)   │ ← normalize FIRST
├──────────┤                           ├──────────┤
│  x + Attn │ ← residual               │  Attn(·)  │
├──────────┤                           ├──────────┤
│  LN(·)    │ ← normalize AFTER        │  x + Attn │ ← residual
├──────────┤                           ├──────────┤
│  FFN(·)   │                           │  LN(·)   │ ← normalize FIRST
├──────────┤                           ├──────────┤
│  + resid  │                           │  FFN(·)   │
├──────────┤                           ├──────────┤
│  LN(·)    │ ← normalize AFTER        │  + resid  │
└──────────┘                           └──────────┘
```

**Why Pre-LN won:**

| Property | Post-LN | Pre-LN |
|----------|---------|--------|
| Gradient flow | Gradients pass through LN → can vanish | Gradients flow directly through residual |
| Warmup needed | Yes, critical | Often unnecessary |
| Training stability | Fragile at depth >12 | Stable at depth >100 |
| Final quality | Slightly better IF you get it to converge | Slightly worse but reliable |
| Used by | Original Transformer, BERT | GPT-2/3, LLaMA, all modern LLMs |

**The key insight:** In Pre-LN, the residual path $x + f(\text{LN}(x))$ creates a "gradient highway" — gradients can flow unimpeded through the addition, bypassing the nonlinear sublayers entirely.

### 19.3 RMSNorm — Simpler Is Better

LLaMA, Gemma, and most modern LLMs use **RMSNorm** instead of LayerNorm:

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

**What's different:** No mean subtraction, no learned bias $\beta$. Just scale by the root-mean-square.

**Why it works:**
- The re-centering (mean subtraction) in LayerNorm is largely redundant after learning
- RMSNorm is ~10-15% faster (saves one reduction operation)
- Empirically matches LayerNorm quality

```python
# LayerNorm: 2 reductions + 2 learned params
mean = x.mean(-1, keepdim=True)       # reduction 1
var  = x.var(-1, keepdim=True)        # reduction 2
out  = gamma * (x - mean) / sqrt(var + eps) + beta

# RMSNorm: 1 reduction + 1 learned param
rms  = sqrt(mean(x^2, dim=-1))        # reduction 1
out  = gamma * x / (rms + eps)         # no beta!
```

### 19.4 Activation Functions: ReLU → GELU → SwiGLU

**The evolution of transformer FFN activations:**

**ReLU** (original transformers):
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$
$$\text{ReLU}(x) = \max(0, x)$$

Problem: dead neurons (permanently zero for negative inputs).

**GELU** (GPT-2, BERT):
$$\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$$

Smooth approximation of ReLU — allows small negative gradients. Used in GPT-2, BERT, ViT.

**SwiGLU** (LLaMA, PaLM, Gemma):
$$\text{SwiGLU}(x) = (\text{Swish}(xW_{\text{gate}})) \odot (xW_1) \cdot W_2$$

where $\text{Swish}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication.

```
Standard FFN:
  x ──→ [W₁] ──→ [ReLU/GELU] ──→ [W₂] ──→ out
  Params: d×4d + 4d×d = 8d²

SwiGLU FFN:
  x ──→ [W_gate] ──→ [Swish] ──┐
  x ──→ [W₁]    ─────────────→ ⊙ ──→ [W₂] ──→ out
  Params: d×(8d/3) × 3 ≈ 8d²  (adjusted to match param count)
```

**Why GLU variants work:** The gate $\text{Swish}(xW_{\text{gate}})$ learns **which features to pass through**. This is a form of learned feature selection — the model can decide to suppress irrelevant information rather than relying on a fixed nonlinearity.

### 19.5 Parameter Count with SwiGLU

Standard FFN with hidden dim $4d$:
$$\text{Params} = d \times 4d + 4d \times d = 8d^2$$

SwiGLU adds a gate projection. To keep total params ≈ same, LLaMA uses hidden dim $\frac{8d}{3}$:
$$\text{Params} = d \times \frac{8d}{3} + d \times \frac{8d}{3} + \frac{8d}{3} \times d = 3 \times \frac{8d^2}{3} = 8d^2$$

In practice, LLaMA rounds to a multiple of 256, so the FFN hidden dim for LLaMA-7B with $d=4096$ is $11008$ (not $\frac{8 \times 4096}{3} = 10923$).

---

## Implementation (60 min)

### 19.6 RMSNorm Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA)."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# Verify equivalence on centered data
x = torch.randn(2, 10, 512)
ln = nn.LayerNorm(512, elementwise_affine=False)
rms = RMSNorm(512)
rms.weight.data.fill_(1.0)

# For zero-mean data, RMSNorm ≈ LayerNorm
x_centered = x - x.mean(dim=-1, keepdim=True)
ln_out = ln(x_centered)
rms_out = rms(x_centered)
print(f"Max diff (centered data): {(ln_out - rms_out).abs().max():.6f}")
```

### 19.7 SwiGLU Implementation

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA-style)."""

    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # LLaMA uses 8/3 × d_model, rounded to multiple of 256
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish = SiLU
        up = self.w1(x)
        return self.w2(gate * up)


class StandardFFN(nn.Module):
    """Standard FFN with configurable activation."""

    def __init__(self, d_model, hidden_dim=None, activation='relu'):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d_model)
        self.act = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        }[activation]

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

### 19.8 Pre-LN vs Post-LN Transformer Block

```python
class TransformerBlock(nn.Module):
    """Transformer block supporting both Pre-LN and Post-LN."""

    def __init__(self, d_model=256, n_heads=8, norm_type='pre',
                 norm_class='layernorm', ffn_type='standard'):
        super().__init__()
        self.norm_type = norm_type

        # Normalization
        NormClass = nn.LayerNorm if norm_class == 'layernorm' else RMSNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)

        # Attention (simplified — using PyTorch MHA)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # FFN
        if ffn_type == 'swiglu':
            self.ffn = SwiGLUFFN(d_model)
        else:
            self.ffn = StandardFFN(d_model, activation='relu')

    def forward(self, x):
        if self.norm_type == 'pre':
            # Pre-LN: normalize → sublayer → residual
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out

            normed = self.norm2(x)
            x = x + self.ffn(normed)
        else:
            # Post-LN: sublayer → residual → normalize
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)

            x = self.norm2(x + self.ffn(x))

        return x


def compare_training_stability(norm_type, n_layers=24, n_steps=200):
    """Train a deep transformer and track gradient norms."""
    d_model = 256
    blocks = nn.Sequential(
        *[TransformerBlock(d_model, norm_type=norm_type) for _ in range(n_layers)]
    )
    optimizer = torch.optim.Adam(blocks.parameters(), lr=1e-4)

    losses = []
    grad_norms = []
    for step in range(n_steps):
        x = torch.randn(4, 32, d_model)
        target = torch.randn(4, 32, d_model)

        out = blocks(x)
        loss = F.mse_loss(out, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm of first layer
        total_norm = 0.0
        for p in blocks[0].parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norms.append(total_norm ** 0.5)

        optimizer.step()

    return losses, grad_norms


# Run comparison
print("Training 24-layer Pre-LN transformer...")
pre_losses, pre_grads = compare_training_stability('pre', n_layers=24)

print("Training 24-layer Post-LN transformer...")
post_losses, post_grads = compare_training_stability('post', n_layers=24)
```

### 19.9 Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training loss
axes[0, 0].plot(pre_losses, label='Pre-LN', alpha=0.7)
axes[0, 0].plot(post_losses, label='Post-LN', alpha=0.7)
axes[0, 0].set_title('Training Loss (24 layers)')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')

# Gradient norms
axes[0, 1].plot(pre_grads, label='Pre-LN', alpha=0.7)
axes[0, 1].plot(post_grads, label='Post-LN', alpha=0.7)
axes[0, 1].set_title('Gradient Norm (Layer 0)')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Gradient Norm')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')

# Activation comparison
x = torch.linspace(-3, 3, 200)
axes[1, 0].plot(x, F.relu(x), label='ReLU')
axes[1, 0].plot(x, F.gelu(x), label='GELU')
axes[1, 0].plot(x, F.silu(x), label='SiLU/Swish')
axes[1, 0].set_title('Activation Functions')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

# SwiGLU gating illustration
gate_input = torch.randn(1000)
gate_output = F.silu(gate_input)
axes[1, 1].scatter(gate_input.numpy(), gate_output.numpy(), alpha=0.3, s=5)
axes[1, 1].set_title('SiLU/Swish Gating')
axes[1, 1].set_xlabel('Input')
axes[1, 1].set_ylabel('Gate Value')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('norm_activation_comparison.png', dpi=150)
plt.show()
```

---

## Exercise (45 min)

### E19.1 Parameter Count Comparison

For $d_{\text{model}} = 4096$ (LLaMA-7B), compute the FFN parameter counts:

| FFN Type | Hidden Dim | Parameters | Formula |
|----------|-----------|------------|---------|
| Standard (ReLU) | 16,384 | ? | $2 \times d \times 4d$ |
| SwiGLU | 11,008 | ? | $3 \times d \times h$ |

Do they have approximately the same number of parameters?

### E19.2 Training Stability Experiment

Run the `compare_training_stability` function with varying depths:
- 6 layers, 12 layers, 24 layers, 48 layers

For each, compare Pre-LN vs Post-LN. At what depth does Post-LN fail to converge?

### E19.3 Activation Ablation

Replace the activation in a small transformer (6 layers, d=256) and train on a language modeling task:
1. ReLU FFN
2. GELU FFN
3. SwiGLU FFN

```python
# Starter: create a simple LM training setup
configs = [
    ('ReLU', StandardFFN(256, activation='relu')),
    ('GELU', StandardFFN(256, activation='gelu')),
    ('SwiGLU', SwiGLUFFN(256)),
]

for name, ffn in configs:
    param_count = sum(p.numel() for p in ffn.parameters())
    print(f"{name}: {param_count:,} parameters")
    # Train each variant for 500 steps, plot loss curves
```

Which converges fastest? Which reaches the lowest loss?

---

## Key Takeaways

1. **Pre-LN is universally preferred** — normalizing before the sublayer creates a gradient highway through residual connections
2. **RMSNorm drops mean subtraction** with no quality loss — simpler and ~15% faster than LayerNorm
3. **SwiGLU uses learned gating** to decide which features pass through the FFN, outperforming fixed nonlinearities
4. **These "boring" details matter enormously** — the difference between a model that trains and one that diverges often comes down to normalization placement
5. **Modern LLM recipe:** Pre-RMSNorm + SwiGLU + no bias terms (LLaMA, Gemma, Mistral all use this)

## Connection to the Thread

You've built the transformer (Day 14), learned to train it (Day 15), and optimized attention (Day 17) and inference (Day 18). Today fills in the remaining architectural decisions that separate a toy transformer from a production LLM. These choices — Pre-LN, RMSNorm, SwiGLU — are the accumulated wisdom of years of scaling experiments. Tomorrow, you'll see the most dramatic architectural innovation in recent LLM history: Mixture of Experts.

## Further Reading

- **[RMSNorm Paper](https://arxiv.org/abs/1910.07467)** — Zhang & Sennrich, 2019
- **[GLU Variants Paper](https://arxiv.org/abs/2002.05202)** — Shazeer, 2020
- **[On Layer Normalization](https://arxiv.org/abs/2002.04745)** — Xiong et al., 2020 (Pre-LN analysis)
- **[LLaMA Paper](https://arxiv.org/abs/2302.13971)** — Touvron et al., 2023 (see architecture section)
- **[Karpathy's Building GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — LayerNorm discussion
