# Day 13: Positional Encoding

> **Phase II — Attention, Transformers & Scaling | Week 2 | 2.5 hours**
> *"Attention treats the input as a set, not a sequence. We must inject the notion of order."*

## Navigation
- **Previous:** [Day 12: Multi-Head Attention](day-12-multi-head-attention.md)
- **Next:** [Day 14: The Full Transformer](day-14-full-transformer.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 13.1 The Permutation Problem

Self-attention is **permutation-equivariant**: if you shuffle the input tokens, the output gets shuffled the same way, but each token's representation is unchanged.

$$\text{Attention}(\pi(X)) = \pi(\text{Attention}(X))$$

where $\pi$ is any permutation. This means attention **doesn't know word order**. "The cat sat on the mat" and "mat the on sat cat the" produce the same attention weights (up to permutation).

But word order matters! We need to somehow encode position into the representation.

### 13.2 Sinusoidal Positional Encoding (Vaswani et al., 2017)

The original transformer uses deterministic sinusoidal functions at different frequencies:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the position index and $i$ is the dimension index.

**Why sinusoids?**
- For any fixed offset $k$, $PE_{pos+k}$ can be expressed as a **linear function** of $PE_{pos}$:

$$\begin{bmatrix} \sin(\omega \cdot (pos+k)) \\ \cos(\omega \cdot (pos+k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix} \begin{bmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{bmatrix}$$

This means the model can learn to attend to **relative positions** using linear transformations.

- Different dimensions use different frequencies (like Fourier basis), spanning from $2\pi$ (dimension 0) to $10000 \cdot 2\pi$ (last dimension). Low-frequency components capture coarse position, high-frequency components capture fine position.

```
Position →
Dim 0: ∿∿∿∿∿∿∿∿∿∿   (high frequency — changes rapidly)
Dim 1: ∿∿∿∿∿         (medium frequency)
Dim 2: ∿∿             (low frequency — changes slowly)
...
Dim d: ∿              (very low — almost constant)
```

### 13.3 Learned Positional Embeddings (GPT-2)

Instead of a fixed formula, learn a position embedding table:

$$\text{input} = \text{token\_embed}(x) + \text{pos\_embed}(pos)$$

where $\text{pos\_embed} \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$ is a trainable parameter.

| Property | Sinusoidal | Learned |
|---|---|---|
| Extrapolation | Can generalize beyond training length (in theory) | Cannot — unseen positions have random embeddings |
| Parameters | 0 extra | $L_{\max} \times d_{\text{model}}$ extra |
| Performance | Slightly worse on fixed-length tasks | Slightly better within training range |
| Used in | Original Transformer | GPT-2, BERT, ViT |

### 13.4 Rotary Position Embeddings — RoPE (Su et al., 2021)

The modern standard, used in LLaMA, Mistral, Qwen, Gemma. RoPE encodes position by **rotating** query and key vectors in 2D subspaces.

**Core idea:** Instead of adding position information, multiply by a rotation matrix. For position $m$ and dimension pair $(2i, 2i+1)$:

$$\begin{bmatrix} q_{2i}^{(m)} \\ q_{2i+1}^{(m)} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$$

where $\theta_i = 10000^{-2i/d}$ (same frequency base as sinusoidal PE).

**Why RoPE is powerful:**

The dot product between rotated queries and keys at positions $m$ and $n$ depends only on the **relative position** $m - n$:

$$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = f(q, k, m-n)$$

This gives relative position encoding *for free* through the attention score computation, without adding any parameters.

### 13.5 ALiBi — Attention with Linear Biases (Press et al., 2022)

A simpler alternative: don't modify embeddings at all. Instead, add a position-dependent **bias** directly to the attention scores:

$$\text{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|$$

where $m$ is a head-specific slope. Closer positions get higher scores; distant positions are penalized linearly. Each head uses a different slope $m$, creating a geometric series:

$$m \in \left\{\frac{1}{2^1}, \frac{1}{2^2}, \ldots, \frac{1}{2^h}\right\}$$

**Advantages:** Zero extra parameters, excellent length extrapolation, trivial to implement.

---

## Implementation (60 min)

### 13.6 Sinusoidal Positional Encoding

```python
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from 'Attention Is All You Need'."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute the positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### 13.7 Rotary Position Embeddings (RoPE)

```python
def precompute_rope_frequencies(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.
    
    Returns: (max_len, dim//2) complex tensor
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    positions = torch.arange(max_len, dtype=torch.float)
    angles = torch.outer(positions, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex exponentials


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to query or key tensor.
    
    x: (batch, n_heads, seq_len, d_k)
    freqs: (seq_len, d_k//2) complex
    """
    # View real tensor as complex: pair up consecutive dimensions
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Reshape freqs for broadcasting: (1, 1, seq_len, d_k//2)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    
    # Multiply by rotation (complex multiplication = rotation in 2D)
    x_rotated = x_complex * freqs[:, :, :x.shape[2], :]
    
    # Back to real: (batch, n_heads, seq_len, d_k)
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


# Usage in attention:
# freqs = precompute_rope_frequencies(d_k, max_len)
# Q_rotated = apply_rope(Q, freqs)
# K_rotated = apply_rope(K, freqs)
# scores = Q_rotated @ K_rotated.transpose(-2, -1) / sqrt(d_k)
```

### 13.8 Visualizing Positional Encodings

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_sinusoidal_pe(d_model: int = 128, max_len: int = 100):
    """Visualize sinusoidal PE as heatmap and similarity matrix."""
    pe = SinusoidalPositionalEncoding(d_model, max_len)
    pe_matrix = pe.pe.squeeze(0).numpy()  # (max_len, d_model)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. PE heatmap
    axes[0].imshow(pe_matrix[:50, :64], aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    axes[0].set_title('Sinusoidal PE Values')
    
    # 2. Cosine similarity between positions
    norms = np.linalg.norm(pe_matrix, axis=1, keepdims=True)
    similarity = (pe_matrix @ pe_matrix.T) / (norms @ norms.T)
    axes[1].imshow(similarity[:50, :50], cmap='viridis')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Position')
    axes[1].set_title('Position Similarity (cosine)')
    
    # 3. Individual dimension waveforms
    for dim in [0, 1, 10, 20, 50]:
        axes[2].plot(pe_matrix[:, dim], label=f'dim {dim}')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('PE Value')
    axes[2].set_title('Selected Dimensions')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('positional_encoding_viz.png', dpi=150)
    plt.show()


visualize_sinusoidal_pe()
```

---

## Exercise (45 min)

### E13.1 Relative Position via Dot Products

For sinusoidal PE with $d_{\text{model}} = 4$:
1. Compute $PE(0), PE(1), PE(5), PE(6)$ by hand (use the formula with $10000^{2i/4}$)
2. Compute the dot products: $PE(0) \cdot PE(1)$ and $PE(5) \cdot PE(6)$
3. Are they similar? Why? (Hint: relative distance is the same)
4. Compute $PE(0) \cdot PE(5)$ — how does it compare?

### E13.2 Comparing Positional Encoding Approaches

Implement all three approaches and compare on a sequence ordering task:
1. **Sinusoidal PE** (fixed, added to embeddings)
2. **Learned PE** (trainable embedding table)
3. **RoPE** (rotary, applied to Q and K)

Task: classify whether a permuted sequence [3, 1, 4, 2] is in ascending order.
- Compare accuracy, especially on sequence lengths *longer* than training (extrapolation test).

### E13.3 ALiBi Implementation

Implement ALiBi as a simple bias matrix:
```python
def alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """Return (n_heads, seq_len, seq_len) bias tensor."""
    # Your implementation here
    pass
```
Test: verify that slopes form a geometric series, plot the bias patterns.

---

## Key Takeaways

- Attention is **permutation-equivariant** — it needs external position information
- **Sinusoidal PE**: deterministic, uses different frequency sines/cosines, enables relative position via linear transforms
- **Learned PE**: trainable table, better within training range, can't extrapolate
- **RoPE**: rotates Q and K by position, makes dot products depend on relative position, used in all modern LLMs
- **ALiBi**: adds linear distance penalty to attention scores, best extrapolation
- The choice of positional encoding significantly affects length generalization

## Connection to the Thread

Position encoding bridges two fundamental representations: **sets** (what attention naturally operates on) and **sequences** (what language actually is). Each approach represents a different compression of positional information: sinusoidal PE uses a Fourier basis (compact but fixed), learned PE uses a lookup table (flexible but finite), RoPE embeds position into the geometry of the attention computation itself. The evolution from absolute to relative to rotary encodings mirrors a broader ML pattern: make inductive biases *softer* and let the model discover the right structure.

## Further Reading

- Vaswani et al. **"Attention Is All You Need"** (2017), Section 3.5. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Su et al. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (2021). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Press, Smith, Lewis. **"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"** (2022). [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Kazemnejad et al. **"The Impact of Positional Encoding on Length Generalization in Transformers"** (2023). [arXiv:2305.19466](https://arxiv.org/abs/2305.19466)
