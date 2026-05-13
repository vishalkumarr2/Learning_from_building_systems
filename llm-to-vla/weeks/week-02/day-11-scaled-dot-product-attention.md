# Day 11: Scaled Dot-Product Attention

> **Phase II — Attention, Transformers & Scaling | Week 2 | 2.5 hours**
> *"The dot product is nature's similarity measure — but without scaling, it breaks softmax."*

## Navigation
- **Previous:** [Day 10: Bahdanau Attention](day-10-bahdanau-attention.md)
- **Next:** [Day 12: Multi-Head Attention](day-12-multi-head-attention.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 11.1 From Additive to Multiplicative Attention

Yesterday's Bahdanau attention uses a learned feedforward network to compute alignment scores. This works but is slow — for each (query, key) pair, we run a neural network forward pass.

**Luong et al. (2015)** proposed a simpler alternative: just take the **dot product** between query and key vectors.

| Attention type | Score function | Complexity |
|---|---|---|
| Additive (Bahdanau) | $e = v^T \tanh(W_s s + W_h h)$ | $O(d_a)$ per pair, sequential |
| Dot-product (Luong) | $e = s^T h$ | $O(d)$ per pair, parallelizable |
| Scaled dot-product | $e = \frac{s^T h}{\sqrt{d_k}}$ | $O(d)$ per pair, parallelizable |

The dot product is computationally cheaper and can be computed in parallel via matrix multiplication.

### 11.2 The Query-Key-Value Framework

Vaswani et al. (2017) formalized attention using three projections:

- **Query** ($Q$): "What am I looking for?" — each position projects its state into a query vector
- **Key** ($K$): "What do I contain?" — each position projects into a key vector for matching
- **Value** ($V$): "What do I offer?" — the actual content to be retrieved

Given input sequence $X \in \mathbb{R}^{n \times d}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$.

### 11.3 The Attention Equation

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Step by step:

```
1. Compute scores:   S = Q K^T          shape: (n, n)
2. Scale:            S = S / √d_k       prevents softmax saturation
3. Softmax:          A = softmax(S)      row-wise, each row sums to 1
4. Weighted sum:     O = A V             shape: (n, d_v)
```

In matrix form, this is beautifully parallelizable:

```
Q: (n × d_k)     ┐
                  ├──→ QK^T: (n × n) ──→ softmax ──→ A: (n × n) ──→ AV: (n × d_v)
K: (n × d_k)     ┘                                                    ↑
V: (n × d_v)  ────────────────────────────────────────────────────────┘
```

### 11.4 Why Scale by $\sqrt{d_k}$?

This is the critical insight that makes dot-product attention work in practice.

**Problem:** For large $d_k$, the dot products $q \cdot k$ grow in magnitude. If $q$ and $k$ have components drawn from $\mathcal{N}(0, 1)$, then:

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

So the standard deviation of dot products scales as $\sqrt{d_k}$. For $d_k = 64$, typical dot products have magnitude $\sim 8$.

**What happens without scaling:** Large values push softmax into regions where gradients are extremely small:

$$\text{softmax}([10, 0, 0]) = [0.9999, 0.00005, 0.00005]$$

The gradient $\frac{\partial}{\partial x_i} \text{softmax}(x)_i = p_i(1 - p_i)$ vanishes when $p_i \approx 1$ or $p_i \approx 0$.

**Scaling by $\sqrt{d_k}$** restores the variance to $\sim 1$:

$$\text{Var}\!\left[\frac{q \cdot k}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1$$

This keeps softmax in a regime with healthy gradients.

### 11.5 Attention as Matrix Operations

The elegance of scaled dot-product attention is that it's just three matrix multiplications:

$$\underbrace{Q K^T}_{\text{all pairwise similarities}} \rightarrow \underbrace{\text{softmax}}_{\text{normalize}} \rightarrow \underbrace{\times V}_{\text{weighted retrieval}}$$

This means **every position attends to every other position simultaneously** — no recurrence, no sequential bottleneck. An $n$-token sequence requires $O(n^2 \cdot d)$ compute but has $O(1)$ sequential depth.

| Property | RNN | Attention |
|---|---|---|
| Sequential operations | $O(n)$ | $O(1)$ |
| Total compute | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| Max path length | $O(n)$ | $O(1)$ |

For $n < d$ (typical in practice), attention wins on everything.

---

## Implementation (60 min)

### 11.6 Implementing Scaled Dot-Product Attention from Scratch

```python
import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,    # (batch, n_q, d_k)
    key: torch.Tensor,      # (batch, n_k, d_k)
    value: torch.Tensor,    # (batch, n_k, d_v)
    mask: torch.Tensor | None = None,  # (batch, n_q, n_k) or broadcastable
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    Returns:
        output: (batch, n_q, d_v) — weighted value vectors
        weights: (batch, n_q, n_k) — attention weights
    """
    d_k = query.size(-1)
    
    # Step 1: Compute raw scores via matrix multiplication
    # (batch, n_q, d_k) × (batch, d_k, n_k) → (batch, n_q, n_k)
    scores = torch.bmm(query, key.transpose(-2, -1))
    
    # Step 2: Scale by √d_k
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (e.g., causal mask for autoregressive decoding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax over keys dimension
    weights = F.softmax(scores, dim=-1)
    
    # Step 5: Weighted sum of values
    output = torch.bmm(weights, value)
    
    return output, weights
```

### 11.7 Testing Dimensions and Properties

```python
def test_attention_shapes():
    """Verify output shapes match expectations."""
    batch, n_q, n_k, d_k, d_v = 2, 5, 7, 64, 32
    
    Q = torch.randn(batch, n_q, d_k)
    K = torch.randn(batch, n_k, d_k)
    V = torch.randn(batch, n_k, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch, n_q, d_v), f"Expected ({batch}, {n_q}, {d_v}), got {output.shape}"
    assert weights.shape == (batch, n_q, n_k), f"Expected ({batch}, {n_q}, {n_k}), got {weights.shape}"
    
    # Attention weights should sum to 1 along key dimension
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sums}"
    
    print("✓ All shape tests passed")


def test_scaling_effect():
    """Demonstrate why scaling matters."""
    d_k = 512
    Q = torch.randn(1, 1, d_k)
    K = torch.randn(1, 10, d_k)
    V = torch.randn(1, 10, d_k)
    
    # Without scaling
    scores_unscaled = torch.bmm(Q, K.transpose(-2, -1))
    weights_unscaled = F.softmax(scores_unscaled, dim=-1)
    
    # With scaling
    scores_scaled = scores_unscaled / math.sqrt(d_k)
    weights_scaled = F.softmax(scores_scaled, dim=-1)
    
    print(f"d_k = {d_k}")
    print(f"Score std (unscaled): {scores_unscaled.std():.2f}")
    print(f"Score std (scaled):   {scores_scaled.std():.2f}")
    print(f"Max weight (unscaled): {weights_unscaled.max():.4f}")
    print(f"Max weight (scaled):   {weights_scaled.max():.4f}")
    print(f"Entropy (unscaled): {-(weights_unscaled * weights_unscaled.log()).sum():.4f}")
    print(f"Entropy (scaled):   {-(weights_scaled * weights_scaled.log()).sum():.4f}")


test_attention_shapes()
test_scaling_effect()
```

### 11.8 Causal (Autoregressive) Masking

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create lower-triangular mask for autoregressive attention.
    
    Position i can only attend to positions ≤ i.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = attend, 0 = block


# Example: 4-token sequence
mask = create_causal_mask(4)
print(mask)
# tensor([[1, 0, 0, 0],    ← token 0 sees only itself
#         [1, 1, 0, 0],    ← token 1 sees 0,1
#         [1, 1, 1, 0],    ← token 2 sees 0,1,2
#         [1, 1, 1, 1]])   ← token 3 sees everything

Q = K = V = torch.randn(1, 4, 64)
output, weights = scaled_dot_product_attention(Q, K, V, mask=mask.unsqueeze(0))
print(f"Causal attention weights:\n{weights.squeeze()}")
# Upper triangle should be zero
```

---

## Exercise (45 min)

### E11.1 Hand Computation

Given a 3-token sequence with $d_k = 2$:

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix}$$

1. Compute $QK^T$ (the raw scores matrix)
2. Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$
3. Apply softmax row-wise to get the attention weight matrix $A$
4. Compute the output $AV$
5. Interpret: which query attends most uniformly? Which is most focused?

### E11.2 Saturation Demonstration

Write code that:
1. Fixes $Q$ and $K$ and varies $d_k$ from 1 to 1024
2. For each $d_k$, computes attention weights **with and without** scaling
3. Plots the **entropy** of the attention distribution vs $d_k$
4. Shows that without scaling, entropy collapses (attention becomes one-hot) as $d_k$ grows

### E11.3 Comparing Bahdanau vs Scaled Dot-Product

Using the same encoder-decoder architecture:
1. Replace Bahdanau attention with scaled dot-product attention
2. Train both on the same data
3. Compare: training speed (wall time per epoch), final BLEU, attention heatmaps
4. Where do they agree? Where do they differ?

---

## Key Takeaways

- **Dot-product attention** replaces the learned alignment network with a simple dot product: $\text{score} = q \cdot k$
- **Scaling by $\sqrt{d_k}$** prevents softmax saturation as dimensionality grows
- The full equation: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) \, V$
- Everything is **matrix multiplications** — embarrassingly parallel on GPUs
- Attention has $O(1)$ path length between any two positions (vs $O(n)$ for RNNs)

## Connection to the Thread

The dot product $q \cdot k = \|q\| \|k\| \cos\theta$ measures **similarity** in the embedding space. Attention weights are a similarity-normalized information retrieval. The model learns to place queries and keys in a space where geometrically close = semantically relevant. This is the same principle behind compression: identify redundancy (similarity), then use it to reconstruct (retrieve values). The transformer will build on this to create an *entire architecture* out of nothing but learned similarity lookups.

## Further Reading

- Vaswani et al. **"Attention Is All You Need"** (2017), Section 3.2.1. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Luong, Pham, Manning. **"Effective Approaches to Attention-based Neural Machine Translation"** (2015). [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- Lilian Weng. **"Attention? Attention!"** (2018). [lilianweng.github.io](https://lilianweng.github.io/posts/2018-06-24-attention/)
