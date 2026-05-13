# 03 — The Transformer Architecture

> Phase II · Days 13–16 · ~10 hours
> Prerequisites: 02-attention-mechanism
> Learning Objectives: Build a complete transformer, understand positional encoding, internalize why this simple architecture dominates

---

## Table of Contents

1. [Positional Encoding (Day 13)](#1-positional-encoding-day-13)
2. [The Full Transformer (Day 14)](#2-the-full-transformer-day-14)
3. [Training a Transformer (Day 15)](#3-training-a-transformer-day-15)
4. [🛑 STOP AND REFLECT #1 (Day 16)](#4--stop-and-reflect-1-day-16)
5. [Key Equations Summary](#5-key-equations-summary)
6. [Architecture Comparison Table](#6-architecture-comparison-table)
7. [Key Takeaways + Paper References](#7-key-takeaways--paper-references)
8. [Connection to Thread](#8-connection-to-thread)

---

## 1. Positional Encoding (Day 13)

### 1.1 The Permutation Problem

Attention is **permutation-equivariant**: if you shuffle the input tokens, the output is shuffled in exactly the same way. This means a vanilla attention mechanism has **no notion of order**.

Consider:
- "The cat sat on the mat"
- "mat the on sat cat the"

Without positional information, attention treats these identically. For language (and definitely for robot trajectories), **order matters**.

> 💡 **Key Insight**: Self-attention operates on **sets**, not **sequences**. Positional encoding is what converts set processing into sequence processing.

### 1.2 Sinusoidal Positional Encoding

The original "Attention Is All You Need" paper proposed injecting position information through sinusoidal functions:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in the sequence (0, 1, 2, ...)
- $i$ = dimension index (0, 1, ..., $d_{model}/2 - 1$)
- $d_{model}$ = model dimension

**Why sinusoids?**

1. **Unique encoding**: Each position gets a unique vector
2. **Bounded**: Values always in $[-1, 1]$
3. **Relative positions via linear projection**: $PE(pos + k)$ can be expressed as a linear function of $PE(pos)$ — meaning the model can learn to attend to relative positions

```python
import torch
import math

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding.
    
    Args:
        max_len: Maximum sequence length
        d_model: Model dimension (must be even)
    
    Returns:
        PE matrix of shape (max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
    
    # Compute the division term: 10000^(2i/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)
    
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    
    return pe
```

**Frequency interpretation**: Each pair of dimensions $(2i, 2i+1)$ oscillates at a different frequency. Low dimensions = low frequency (long wavelength, captures global position), high dimensions = high frequency (fine-grained local position).

```
Position:     0    1    2    3    4    5    6    7
dim 0 (sin):  0.00 0.84 0.91 0.14 -0.76 -0.96 -0.28 0.66
dim 1 (cos):  1.00 0.54 -0.42 -0.99 -0.65 0.28 0.96 0.75
dim 2 (sin):  0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07  ← much slower
dim 3 (cos):  1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00  ← barely changes
```

> ⚠️ **Common Confusion**: PE is **added** to the token embeddings, not concatenated. This means $x_{input} = \text{TokenEmbed}(token) + PE(pos)$. The model must learn to "undo" this addition to separate content from position.

### 1.3 Learned Positional Encoding

Instead of fixed sinusoids, learn a position embedding matrix:

```python
class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)
```

**Pros**: Can learn arbitrary position-dependent patterns
**Cons**: Cannot extrapolate beyond `max_len` seen during training

GPT-2 and BERT both used learned PE. In practice, learned PE performs similarly to sinusoidal for typical sequence lengths.

### 1.4 RoPE — Rotary Position Embeddings (Modern Standard)

RoPE (Su et al., 2021) is used in LLaMA, Mistral, Qwen, and most modern LLMs. Instead of adding positional information to embeddings, RoPE **rotates** the query and key vectors.

**Core idea**: Encode position by rotating pairs of dimensions:

$$f_q(x_m, m) = x_m e^{im\theta}$$

In real numbers, for each pair of dimensions $(2j, 2j+1)$:

$$\begin{pmatrix} q_{2j}' \\ q_{2j+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix}$$

Where $\theta_j = 10000^{-2j/d}$ and $m$ is the position.

**Why RoPE is elegant**: The dot product $q_m \cdot k_n$ depends only on the **relative position** $m - n$, not the absolute positions. This gives relative position encoding for free.

```python
def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding.
    
    Args:
        x: Input tensor (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed complex frequencies (seq_len, head_dim/2)
    """
    # Reshape x to pairs of dimensions
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    # Multiply by rotation (complex multiplication = rotation)
    x_rotated = x_complex * freqs_cis
    # Convert back to real
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0):
    """Precompute the rotation frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64
```

### 1.5 ALiBi — Attention with Linear Biases

ALiBi (Press et al., 2022) takes a completely different approach: add a **position-dependent bias** directly to the attention scores:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot [-(i-j)]\right) V$$

Where $m$ is a head-specific slope and $(i-j)$ is the distance between positions. Closer tokens get higher attention scores.

**Key advantage**: Excellent length extrapolation — trained on 1024 tokens, works on 2048+.

### 1.6 Comparison Table

| Method | Type | Relative Pos? | Extrapolation | Used In |
|--------|------|:---:|:---:|---------|
| Sinusoidal | Fixed, additive | Theoretically | Limited | Original Transformer |
| Learned | Learned, additive | No | None | GPT-2, BERT |
| RoPE | Rotation on Q,K | ✅ Yes | Good (with NTK) | LLaMA, Mistral, Qwen |
| ALiBi | Bias on attention | ✅ Yes | ✅ Excellent | BLOOM, MPT |
| Relative PE | Learned bias | ✅ Yes | Limited | T5, Transformer-XL |

> 💡 **For robotics/VLA**: RoPE is the default choice. For multi-modal models processing images + text + actions, positional encoding must handle different "position spaces" — more on this in later notes.

---

## 2. The Full Transformer (Day 14)

### 2.1 Architecture Overview

```
                    ┌─────────────────────────┐
                    │       OUTPUT PROBS       │
                    │    (softmax over vocab)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      Linear + Softmax    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                      │
    ┌─────────┴─────────┐              ┌─────────────┴──────────┐
    │   ENCODER STACK    │              │    DECODER STACK        │
    │   (N× layers)      │              │    (N× layers)          │
    │                    │              │                         │
    │ ┌────────────────┐ │    K,V       │ ┌─────────────────────┐│
    │ │ Self-Attention  │ │──────────►  │ │ Masked Self-Attn    ││
    │ │ + Add & Norm    │ │             │ │ + Add & Norm        ││
    │ ├────────────────┤ │             │ ├─────────────────────┤│
    │ │ FFN             │ │             │ │ Cross-Attention     ││
    │ │ + Add & Norm    │ │             │ │ + Add & Norm        ││
    │ └────────────────┘ │             │ ├─────────────────────┤│
    │                    │             │ │ FFN                  ││
    └─────────┬─────────┘             │ │ + Add & Norm        ││
              │                        │ └─────────────────────┘│
    ┌─────────┴─────────┐             └─────────────┬──────────┘
    │ Input Embedding    │              ┌────────────┴───────────┐
    │ + Positional Enc   │              │ Output Embedding       │
    └─────────┬─────────┘              │ + Positional Enc       │
              │                        └────────────┬───────────┘
        [Source Tokens]                      [Target Tokens]
              │                              (shifted right)
              │                                    │
         "The cat sat"                    "<sos> Le chat"
```

### 2.2 The Encoder Block

Each encoder layer has two sub-layers:

```
     Input x
        │
        ▼
  ┌───────────┐
  │  Multi-Head │
  │  Self-Attn  │
  └──────┬──────┘
         │
    Add & Norm  ◄── x (residual connection)
         │
         ▼
  ┌───────────┐
  │   FFN      │
  │ (2 linear) │
  └──────┬──────┘
         │
    Add & Norm  ◄── (residual connection)
         │
         ▼
     Output
```

```python
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Sub-layer 1: Multi-Head Self-Attention
        attn_output = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))  # Post-LN
        
        # Sub-layer 2: Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

### 2.3 The Decoder Block

The decoder has **three** sub-layers:

1. **Masked Self-Attention**: Attends to previous decoder outputs (causal mask prevents looking ahead)
2. **Cross-Attention**: Queries = decoder states, Keys/Values = encoder output
3. **FFN**: Same as encoder

```python
class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Sub-layer 1: Masked Self-Attention
        self_attn_out = self.masked_self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Sub-layer 2: Cross-Attention (Q from decoder, K/V from encoder)
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Sub-layer 3: Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x
```

### 2.4 The Causal Mask

For autoregressive generation, position $i$ can only attend to positions $\leq i$:

```python
def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """Generate causal (look-ahead) mask.
    
    Returns:
        Upper triangular mask where True = masked out
        Shape: (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask  # True where attention should NOT go

# For seq_len=4:
# [[False,  True,  True,  True],
#  [False, False,  True,  True],
#  [False, False, False,  True],
#  [False, False, False, False]]
#
# Position 0 sees: [0]
# Position 1 sees: [0, 1]
# Position 2 sees: [0, 1, 2]
# Position 3 sees: [0, 1, 2, 3]
```

### 2.5 Pre-LN vs Post-LN

**Post-LN** (original paper):
$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

**Pre-LN** (modern standard):
$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$$

```python
# Post-LN (original)
x = self.norm(x + self.sublayer(x))

# Pre-LN (modern — better training stability)
x = x + self.sublayer(self.norm(x))
```

> 💡 **Why Pre-LN?** The gradient flows directly through the residual connection without going through LayerNorm. This makes training much more stable, especially for deep models (>24 layers). Almost all modern LLMs use Pre-LN.

> ⚠️ **Gotcha**: Pre-LN requires an **extra final LayerNorm** after the last layer, since the last sub-layer output isn't normalized.

### 2.6 Feed-Forward Network

The FFN is a simple two-layer MLP applied independently to each position:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

Where $\sigma$ is the activation function.

**Classic (ReLU)**:
$$\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2$$

**Modern (SwiGLU)** — used in LLaMA, Mistral:
$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xV) W_2$$

Where $\text{Swish}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication.

```python
class FeedForward(torch.nn.Module):
    """Standard FFN with GELU activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff)
        self.w2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(torch.nn.functional.gelu(self.w1(x))))


class SwiGLUFeedForward(torch.nn.Module):
    """SwiGLU FFN used in LLaMA/Mistral."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = torch.nn.functional.silu(self.w1(x))  # Swish = SiLU
        gate = self.w3(x)
        return self.w2(self.dropout(swish * gate))
```

> 💡 **FFN dimension**: Typically $d_{ff} = 4 \times d_{model}$. For SwiGLU, the hidden dim is often $\frac{8}{3} d_{model}$ rounded to a multiple of 256, to keep parameter count similar to a 4× ReLU FFN.

### 2.7 Residual Connections

Residual connections are **everywhere** in the transformer and are arguably the most important architectural choice:

$$\text{output} = x + f(x)$$

**Why residuals matter**:
1. **Gradient highway**: Gradients flow through the addition unmodified
2. **Identity initialization**: At initialization, sub-layers output ≈ 0, so the model starts as an identity function
3. **Depth**: Without residuals, training even a 6-layer model is unstable. With residuals, we can train 100+ layers

```
Without residuals:           With residuals:
x → f₁ → f₂ → f₃ → y       x ──┬──► (+) ──┬──► (+) ──┬──► (+) → y
                                 │    ▲      │    ▲      │    ▲
                                 └─ f₁┘      └─ f₂┘      └─ f₃┘
                             
Gradient must flow             Gradient has a direct path
through all f's                through additions
```

### 2.8 Putting It All Together

```python
class Transformer(torch.nn.Module):
    """Complete Transformer for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Embeddings
        self.src_embed = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = sinusoidal_positional_encoding(max_len, d_model)
        self.scale = math.sqrt(d_model)
        
        # Encoder
        self.encoder_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = torch.nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = torch.nn.Linear(d_model, tgt_vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        x = self.dropout(self.src_embed(src) * self.scale + self.pos_encoding[:src.size(1)])
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        return x
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.dropout(self.tgt_embed(tgt) * self.scale + self.pos_encoding[:tgt.size(1)])
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.output_proj(output)
```

> ⚠️ **The $\sqrt{d_{model}}$ scaling on embeddings**: The embedding values are typically small (initialized ~$\mathcal{N}(0, 1)$), while PE values are in $[-1, 1]$. Multiplying embeddings by $\sqrt{d_{model}}$ puts them on a similar scale so PE doesn't dominate.

---

## 3. Training a Transformer (Day 15)

### 3.1 Label Smoothing

Instead of hard targets (one-hot), use soft targets:

$$y_{smooth}(k) = \begin{cases} 1 - \epsilon & \text{if } k = \text{target} \\ \frac{\epsilon}{K-1} & \text{otherwise} \end{cases}$$

Where $\epsilon$ is typically 0.1 and $K$ is vocab size.

```python
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch * seq_len, vocab_size) logits
            target: (batch * seq_len,) indices
        """
        pred = pred.log_softmax(dim=-1)
        
        # Uniform distribution for smoothing
        smooth_loss = -pred.sum(dim=-1) / self.vocab_size
        # NLL for correct class
        nll_loss = -pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # Mask padding
        mask = target != self.pad_idx
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
```

**Why label smoothing?** 
- Prevents the model from becoming overconfident
- Acts as a regularizer
- Improves BLEU scores in translation

### 3.2 Learning Rate Warmup — The "Noam" Schedule

The original transformer used a distinctive LR schedule:

$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup^{-1.5})$$

This linearly increases LR for `warmup` steps, then decays proportionally to $1/\sqrt{step}$.

```python
class NoamScheduler:
    """Noam learning rate schedule from 'Attention Is All You Need'."""
    
    def __init__(self, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
    
    def step(self):
        self._step += 1
        return self.get_lr()
    
    def get_lr(self) -> float:
        step = max(self._step, 1)
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
```

```
LR
▲
│        ╱╲
│       ╱  ╲
│      ╱    ╲
│     ╱      ╲╲
│    ╱         ╲╲
│   ╱            ╲╲╲
│  ╱                ╲╲╲╲╲
│ ╱                       ╲╲╲╲╲╲╲
│╱                                  ╲╲╲╲╲
└────────────────────────────────────────► step
     warmup        decay
```

> 💡 **Modern practice**: Most LLMs now use **cosine decay with warmup** instead of the Noam schedule. Warmup is still universal — it stabilizes training when the model's gradients are large and noisy early on.

```python
def cosine_warmup_schedule(step, warmup_steps, total_steps, max_lr, min_lr=0):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### 3.3 Masked Loss

When training on sequences with padding, we must not compute loss on padding tokens:

```python
def masked_cross_entropy(logits, targets, pad_idx):
    """Cross entropy ignoring padding tokens."""
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction='mean',
    )
    return loss
```

### 3.4 Training Stability (from Note 01)

Critical hyperparameters for stable transformer training:

| Hyperparameter | Typical Value | Notes |
|---------------|:---:|-------|
| Warmup steps | 4000–10000 | More for larger models |
| Max LR | 1e-4 to 3e-4 | Scale with $1/\sqrt{d_{model}}$ |
| Dropout | 0.1 | 0.0 for large-scale pretraining |
| Label smoothing | 0.1 | — |
| Weight decay | 0.01–0.1 | Don't apply to biases/norms |
| Gradient clipping | 1.0 | Max norm |
| Batch size | Large (4k+ tokens) | Accumulate gradients if needed |
| β₁, β₂ (Adam) | 0.9, 0.98 | β₂=0.95 also common |
| ε (Adam) | 1e-9 | Prevents division by zero |

> ⚠️ **Critical**: Don't apply weight decay to LayerNorm parameters or bias terms. This is a common source of training instability.

```python
def get_optimizer(model, lr, weight_decay=0.01):
    """Separate parameters for weight decay."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'norm' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.98), eps=1e-9)
```

### 3.5 Training Recipe Summary

```python
# Pseudocode for transformer training loop
model = Transformer(...)
optimizer = get_optimizer(model, lr=3e-4)
scheduler = CosineWarmupScheduler(warmup=4000, total=100000)

for epoch in range(num_epochs):
    for src, tgt in dataloader:
        # Teacher forcing: input = tgt[:-1], target = tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        
        # Forward
        causal_mask = generate_causal_mask(tgt_input.size(1))
        logits = model(src, tgt_input, tgt_mask=causal_mask)
        
        # Loss
        loss = masked_cross_entropy(logits, tgt_target, pad_idx=0)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## 4. 🛑 STOP AND REFLECT #1 (Day 16)

> **This is the most important day in Phase II.** Before moving on, you must be able to reconstruct the transformer from first principles.

### 4.1 The Thesis Statement

> **"Attention is just a soft dictionary lookup. The transformer is embarrassingly simple — just attention + FFN + residuals, stacked. Why does something this simple work so well?"**

Take a moment. The entire model is:
1. **Attention**: weighted average based on similarity (soft lookup)
2. **FFN**: independent per-position nonlinear transformation
3. **Residual connections**: identity + refinement
4. **Layer norm**: scale normalization

That's it. No recurrence. No convolution. No complex routing. Just these four building blocks, stacked 6 to 100+ times.

### 4.2 Re-derive Attention from Scratch

Without looking at notes, derive on paper:

1. Start with the problem: "I have a sequence of vectors. I want each vector to gather information from all others based on relevance."
2. Define relevance as a dot product → $QK^T$
3. Realize you need probabilities → softmax
4. Realize large $d_k$ makes softmax peaked → scale by $\sqrt{d_k}$
5. Apply the probabilities to values → $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
6. Want different "types" of relevance → multi-head

### 4.3 Draw the Architecture from Memory

On paper (or whiteboard), draw:
- [ ] Encoder block with sub-layers
- [ ] Decoder block with all three sub-layers
- [ ] Residual connections and layer norms
- [ ] The full model with embeddings and output projection
- [ ] Data flow between encoder and decoder (cross-attention)
- [ ] Causal mask in the decoder

### 4.4 Write: "Why Does the Transformer Work?"

Write a 1-page essay (handwritten or typed) answering:

1. **Universality**: Attention can represent any pairwise interaction pattern. FFN can represent any per-position function. Together, they're a universal sequence-to-sequence function.

2. **Gradient flow**: Residual connections create an O(1) gradient path from output to any layer. Pre-LN makes this path smooth. The model is effectively an ensemble of paths of different depths.

3. **Parallelism**: Unlike RNNs, all positions are processed simultaneously. This maps perfectly to GPU hardware.

4. **Inductive bias**: The transformer has almost no inductive bias (unlike CNNs with locality or RNNs with sequential). This means it needs more data but can learn any pattern. The implicit inductive bias from positional encoding is mild.

5. **Scaling**: The transformer has a remarkably smooth loss curve as we increase parameters, data, and compute. This is the foundation of scaling laws (later notes).

### 4.5 Check Your Understanding

Answer without looking:

- [ ] What is the computational complexity of self-attention? ($O(n^2 d)$)
- [ ] Why scale by $\sqrt{d_k}$? (Prevent softmax saturation)
- [ ] What does each FFN layer do? (Per-position nonlinear transformation)
- [ ] Why Pre-LN over Post-LN? (Better gradient flow, training stability)
- [ ] What's the difference between encoder self-attention and decoder self-attention? (Causal mask)
- [ ] Where do the encoder outputs go in the decoder? (Cross-attention K, V)
- [ ] Why multiply embeddings by $\sqrt{d_{model}}$? (Match scale with PE)
- [ ] What is teacher forcing? (Feed ground-truth previous tokens during training)

---

## 5. Key Equations Summary

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Positional Encoding
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### RoPE
$$f(x_m, m) = x_m e^{im\theta_j}, \quad \theta_j = 10000^{-2j/d}$$

### Feed-Forward (SwiGLU)
$$\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_3) W_2$$

### Noam Schedule
$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup^{-1.5})$$

### Label Smoothing
$$y_{smooth}(k) = (1 - \epsilon) \cdot \mathbb{1}[k = target] + \frac{\epsilon}{K - 1}$$

### Encoder Sub-layer (Post-LN)
$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

### Encoder Sub-layer (Pre-LN)
$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$$

---

## 6. Architecture Comparison Table

| Property | Encoder-Only | Decoder-Only | Encoder-Decoder |
|----------|:---:|:---:|:---:|
| **Attention** | Bidirectional | Causal (unidirectional) | Both |
| **Example** | BERT, RoBERTa | GPT, LLaMA, Mistral | T5, BART, Original Transformer |
| **Pretraining** | Masked LM | Next token prediction | Span corruption / Denoising |
| **Strengths** | Classification, NLU | Generation, in-context learning | Seq-to-seq, translation |
| **For VLAs?** | Feature extraction | Action generation | Perception → Action |
| **Parameter sharing** | — | — | Sometimes shared embeddings |
| **Cross-attention** | None | None | Decoder attends to encoder |
| **Dominant today** | Legacy (BERT era) | ✅ LLMs, chat, code | Translation, speech, some VLAs |

> 💡 **Why decoder-only won**: With enough scale, decoder-only models can do everything encoder-only and encoder-decoder models can do, through in-context learning. Simpler architecture = easier to scale.

> 💡 **For VLAs**: Many VLAs use encoder-decoder or decoder-only with a vision encoder prefix. RT-2 and OpenVLA use decoder-only transformers where images are tokenized and prepended to the action sequence.

---

## 7. Key Takeaways + Paper References

### Key Takeaways

1. **Positional encoding converts set processing to sequence processing** — without it, the transformer is permutation-equivariant
2. **RoPE is the modern standard** for positional encoding, providing relative position awareness through rotation
3. **The transformer is four building blocks**: attention, FFN, residual connections, layer norm
4. **Pre-LN + residual connections** are the keys to training deep models stably
5. **SwiGLU** has replaced ReLU/GELU in modern LLMs for the FFN
6. **Label smoothing + LR warmup + gradient clipping** are essential training ingredients
7. **The architecture is simple** — the magic comes from scale, data, and the universality of attention

### Paper References

| Paper | Year | Key Contribution |
|-------|:---:|------------------|
| Vaswani et al. — "Attention Is All You Need" | 2017 | The transformer |
| Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding" | 2021 | RoPE |
| Press et al. — "Train Short, Test Long: Attention with Linear Biases" | 2022 | ALiBi |
| Shazeer — "GLU Variants Improve Transformer" | 2020 | SwiGLU, GeGLU |
| Xiong et al. — "On Layer Normalization in the Transformer Architecture" | 2020 | Pre-LN analysis |
| Zhang & Sennrich — "Root Mean Square Layer Normalization" | 2019 | RMSNorm |
| He et al. — "Deep Residual Learning for Image Recognition" | 2015 | Residual connections |
| Szegedy et al. — "Rethinking the Inception Architecture" | 2016 | Label smoothing |

---

## 8. Connection to Thread

> **The Thread**: Information compression → attention as selective retrieval → the transformer as a compression machine

The transformer architecture is an **information compression machine**:

1. **Attention** performs **selective compression**: from all positions, extract only what's relevant to each position. This is lossy compression guided by the query.

2. **FFN** performs **per-position transformation**: each position independently processes its compressed representation. Think of this as a "lookup table" in the model's learned knowledge.

3. **Stacking layers** means applying **iterative refinement**: each layer further compresses and transforms the representation. Early layers capture local patterns; later layers capture global, abstract concepts.

4. **Residual connections** preserve the **raw signal**: the model can always access less-processed information if needed, preventing catastrophic information loss.

**Why is the transformer the architecture for VLAs?**
- Robot perception is information compression: camera images → relevant features
- Robot decision-making is selective retrieval: from all knowledge, retrieve the right action
- Both are exactly what attention does

> Next: **04-transformer-variants** — efficiency improvements, KV cache, MoE, BERT, and tokenization
