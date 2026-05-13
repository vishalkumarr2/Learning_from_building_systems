# 02 — The Attention Mechanism
> Phase II · Days 10–12 · ~7.5 hours
> Prerequisites: 01-dl-foundations (especially seq2seq bottleneck, information theory)
> Learning Objectives: Understand attention from Bahdanau to multi-head, derive equations, implement from scratch

---

## Table of Contents
1. [Motivation: Why Attention?](#1-motivation-why-attention)
2. [Bahdanau Attention (Day 10)](#2-bahdanau-attention-day-10)
3. [Scaled Dot-Product Attention (Day 11)](#3-scaled-dot-product-attention-day-11)
4. [Multi-Head Attention (Day 12)](#4-multi-head-attention-day-12)
5. [Self-Attention vs Cross-Attention](#5-self-attention-vs-cross-attention)
6. [Key Equations Summary](#6-key-equations-summary)
7. [Key Takeaways](#7-key-takeaways)
8. [Paper References](#8-paper-references)
9. [Connection to the Thread](#9-connection-to-the-thread)

---

## 1. Motivation: Why Attention?

### 1.1 The Seq2Seq Bottleneck (Recap)

In note 01, we saw that encoder-decoder (seq2seq) models compress an entire input sequence into a single fixed-size vector $\mathbf{c}$:

```
┌──────────────────────────────────────┐
│  Encoder                              │
│  x₁ → h₁ → h₂ → ... → hₙ → [c]    │  ← entire input squashed here
│                                       │
│  Decoder                              │
│  [c] → s₁ → s₂ → ... → sₘ          │  ← must reconstruct from [c]
└──────────────────────────────────────┘
```

This is an **information bottleneck**: the vector $\mathbf{c} \in \mathbb{R}^d$ must encode everything about the source sentence — word order, semantics, syntax, coreferences — regardless of whether the source is 5 words or 50 words.

Empirically, Cho et al. (2014) showed that seq2seq translation quality **degrades sharply** on sentences longer than ~20 tokens. The fixed-size vector simply can't hold it all.

> 💡 **Key Insight**: The bottleneck is not about model capacity — it's about *forcing* all information through a single vector. Even a very large $d$ doesn't help because the decoder has no way to *selectively access* different parts of the input at different decoding steps.

### 1.2 What If the Decoder Could Look Back?

The core idea behind attention is deceptively simple:

**Instead of reading only $\mathbf{c}$, let the decoder look at ALL encoder hidden states $h_1, h_2, \ldots, h_n$ at every decoding step.**

```
Encoder states:  h₁   h₂   h₃   h₄   h₅
                  ↑    ↑    ↑    ↑    ↑
                  └────┴────┴────┴────┘
                         ↓ weighted sum
                    context vector cᵢ
                         ↓
Decoder step i:  ────→  sᵢ  ────→  yᵢ
```

At each decoder step $i$, we compute a **weighted combination** of encoder states. The weights tell the decoder *where to look* for the most relevant information for generating the current output token.

### 1.3 Attention as the Solution

Attention replaces the single bottleneck vector with a **dynamic, step-specific** context:

| Without Attention | With Attention |
|---|---|
| One fixed $\mathbf{c}$ for all steps | Different $\mathbf{c}_i$ for each step |
| Decoder "remembers" through its state | Decoder "looks back" at the source |
| Information must be memorized | Information is retrieved on demand |
| Degrades on long sequences | Scales to longer sequences |

> 💡 **Key Insight**: Attention is fundamentally a **retrieval mechanism**. Instead of compressing everything into memory, we keep the raw data accessible and learn to retrieve what we need. This is the same principle behind databases, caches, and search engines — applied to neural networks.

---

## 2. Bahdanau Attention (Day 10)

### 2.1 The Paper

**"Neural Machine Translation by Jointly Learning to Align and Translate"** (Bahdanau, Cho, Bengio, 2014)

This paper introduced the attention mechanism for NMT and is one of the most influential papers in deep learning. It reframed the decoder as performing **alignment** between source and target — hence "jointly learning to align and translate."

### 2.2 Setup

Given:
- Encoder hidden states: $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_n \in \mathbb{R}^d$ (one per source token)
- Previous decoder state: $\mathbf{s}_{i-1} \in \mathbb{R}^d$

We want: a context vector $\mathbf{c}_i$ that tells the decoder which parts of the source are relevant for generating token $y_i$.

### 2.3 The Additive Attention Mechanism

**Step 1 — Compute alignment scores (energies):**

$$e_{ij} = \mathbf{v}^T \tanh(\mathbf{W}_s \mathbf{s}_{i-1} + \mathbf{W}_h \mathbf{h}_j)$$

where:
- $\mathbf{W}_s \in \mathbb{R}^{a \times d}$ projects the decoder state
- $\mathbf{W}_h \in \mathbb{R}^{a \times d}$ projects the encoder state
- $\mathbf{v} \in \mathbb{R}^a$ produces a scalar score
- $a$ is the attention hidden dimension (a hyperparameter)

The $\tanh$ makes this an **additive** (or concatenative) attention — the two projections are added, not multiplied.

**Step 2 — Normalize to get attention weights:**

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})} = \text{softmax}(e_{ij})$$

Properties of $\alpha_{ij}$:
- $\alpha_{ij} \geq 0$ (non-negative)
- $\sum_j \alpha_{ij} = 1$ (sums to 1 over source positions)
- Interpretable: $\alpha_{ij}$ = "how much attention does decoder step $i$ pay to source position $j$"

**Step 3 — Compute context vector:**

$$\mathbf{c}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{h}_j$$

This is a weighted average of encoder states, where the weights are learned and step-specific.

### 2.4 The QKV Lens

Even though Bahdanau didn't use this terminology, we can retroactively view the mechanism through the **Query-Key-Value** framework:

| Concept | Bahdanau Mapping | Role |
|---------|-----------------|------|
| **Query** | Decoder state $\mathbf{s}_{i-1}$ | "What am I looking for?" |
| **Key** | Encoder states $\mathbf{h}_j$ | "What do I contain?" |
| **Value** | Encoder states $\mathbf{h}_j$ | "What information do I carry?" |

> 💡 **Key Insight**: **Attention is a soft dictionary lookup.** Think of a Python dictionary: `dict[key] → value`. Attention does the same thing, but *softly* — instead of exact matching one key, it matches all keys with different weights and returns a weighted blend of values. This is THE fundamental insight for understanding all attention variants.

```
Hard lookup:     query → exact match → one value
                 "cat" → {"cat": 🐱} → 🐱

Soft attention:  query → similarity scores → weighted blend of all values
                 "cat" → {0.7: 🐱, 0.2: 🐕, 0.1: 🐦} → mostly 🐱
```

### 2.5 Visualization: Attention as a Heatmap

Attention weights $\alpha_{ij}$ form an $m \times n$ matrix (target length × source length) that can be visualized as a heatmap:

```
                 Source: The  cat  sat  on  the  mat
Target:
Le               [0.8  0.1  0.0  0.0  0.1  0.0]
chat             [0.0  0.9  0.0  0.0  0.0  0.1]
était            [0.0  0.1  0.8  0.0  0.1  0.0]
assis            [0.0  0.0  0.9  0.0  0.0  0.1]
sur              [0.0  0.0  0.0  0.9  0.0  0.1]
le               [0.0  0.0  0.0  0.0  0.9  0.0]
tapis            [0.0  0.0  0.0  0.0  0.0  1.0]
```

For French-English, you'd expect a roughly diagonal pattern (monotonic alignment). For Japanese-English, the pattern would be much more complex due to word order differences.

### 2.6 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism.
    
    Computes: e_ij = v^T tanh(W_s @ s_{i-1} + W_h @ h_j)
    """
    
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    
    def forward(
        self,
        decoder_state: torch.Tensor,   # (batch, hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (batch, hidden_dim)
            weights: (batch, src_len)
        """
        # decoder_state: (batch, hidden_dim) -> (batch, 1, attn_dim)
        query = self.W_s(decoder_state).unsqueeze(1)
        
        # encoder_outputs: (batch, src_len, hidden_dim) -> (batch, src_len, attn_dim)
        keys = self.W_h(encoder_outputs)
        
        # Alignment scores: (batch, src_len, 1) -> (batch, src_len)
        energy = self.v(torch.tanh(query + keys)).squeeze(-1)
        
        # Attention weights: (batch, src_len)
        weights = F.softmax(energy, dim=-1)
        
        # Context vector: (batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights
```

### 2.7 Complexity Analysis

For Bahdanau attention at each decoder step:
- Alignment score: $O(n \cdot a)$ — project each of $n$ encoder states, add, tanh, project to scalar
- Softmax: $O(n)$
- Context: $O(n \cdot d)$ — weighted sum of $n$ vectors of dimension $d$
- Total per decoder step: $O(n \cdot (a + d))$
- Total for full sequence: $O(m \cdot n \cdot (a + d))$ — still sequential over $m$ decoder steps!

> ⚠️ **Common Pitfall**: Bahdanau attention is still **sequential** over decoder steps because each step depends on the previous decoder state $s_{i-1}$. The attention only lets us look at all encoder states — it doesn't parallelize the decoder. This sequential bottleneck motivates the transformer.

---

## 3. Scaled Dot-Product Attention (Day 11)

### 3.1 From Additive to Multiplicative

Bahdanau's additive attention works well but has a learnable alignment function with parameters $\mathbf{W}_s$, $\mathbf{W}_h$, $\mathbf{v}$. Luong et al. (2015) proposed a simpler alternative:

**Dot-product attention:**

$$e_{ij} = \mathbf{s}_i^T \mathbf{h}_j$$

No learned parameters! Just a dot product measuring similarity directly.

| Property | Additive (Bahdanau) | Dot-Product (Luong) |
|----------|-------------------|-------------------|
| Formula | $\mathbf{v}^T \tanh(\mathbf{W}_s \mathbf{s} + \mathbf{W}_h \mathbf{h})$ | $\mathbf{s}^T \mathbf{h}$ |
| Parameters | $\mathbf{W}_s, \mathbf{W}_h, \mathbf{v}$ | None (or linear projections) |
| Computation | Slower (tanh, two matmuls) | Faster (single matmul) |
| Expressiveness | More flexible | Simpler |

### 3.2 The Scaling Problem

With high-dimensional vectors, dot products can become very large in magnitude. If $\mathbf{q}$ and $\mathbf{k}$ are independent random vectors with components drawn from $\mathcal{N}(0, 1)$:

$$\mathbb{E}[\mathbf{q}^T \mathbf{k}] = 0, \quad \text{Var}[\mathbf{q}^T \mathbf{k}] = d_k$$

As $d_k$ grows, the variance of the dot product grows, pushing some scores to extreme values. When fed through softmax, this creates **very peaked** distributions:

```
d_k = 16:   softmax scores ≈ [0.15, 0.20, 0.25, 0.22, 0.18]  ← useful gradient
d_k = 512:  softmax scores ≈ [0.00, 0.00, 0.99, 0.01, 0.00]  ← gradient vanishes!
```

The softmax saturates, gradients vanish, and the model can't learn useful attention patterns.

> ⚠️ **Common Pitfall**: Without scaling, attention becomes essentially a hard argmax for large $d_k$, defeating the purpose of a *soft* weighted average. Students often skip the scaling factor and wonder why attention doesn't train well.

### 3.3 The Core Equation

Vaswani et al. (2017) introduced the **scaled dot-product attention** that fixes this:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$ — Query matrix (one row per query position)
- $K \in \mathbb{R}^{m \times d_k}$ — Key matrix (one row per key position)
- $V \in \mathbb{R}^{m \times d_v}$ — Value matrix (one row per value position)
- $d_k$ — dimension of queries/keys

The $\frac{1}{\sqrt{d_k}}$ scaling ensures the dot products have unit variance regardless of $d_k$, keeping softmax in its useful gradient regime.

### 3.4 Step-by-Step Matrix View

Let's trace through the computation for a concrete example.

Suppose we have a sequence of 4 tokens with $d_k = d_v = 3$:

```
Q = [[1, 0, 1],     K = [[1, 1, 0],     V = [[1, 0, 0],
     [0, 1, 1],          [0, 1, 1],          [0, 1, 0],
     [1, 1, 0],          [1, 0, 1],          [0, 0, 1],
     [0, 0, 1]]          [0, 1, 0]]          [1, 1, 0]]
```

**Step 1: Compute $QK^T$ (similarity matrix)**

$$QK^T = \begin{bmatrix} 1 & 1 & 2 & 0 \\ 1 & 2 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$

Shape: $(4, 3) \times (3, 4) = (4, 4)$. Entry $(i, j)$ = how much query $i$ matches key $j$.

**Step 2: Scale by $\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{3}} \approx 0.577$**

$$\frac{QK^T}{\sqrt{d_k}} = \begin{bmatrix} 0.58 & 0.58 & 1.15 & 0.00 \\ 0.58 & 1.15 & 0.58 & 0.58 \\ 0.58 & 0.58 & 0.58 & 0.58 \\ 0.00 & 0.58 & 0.58 & 0.00 \end{bmatrix}$$

**Step 3: Softmax (row-wise)**

Each row becomes a probability distribution.

**Step 4: Multiply by $V$**

Weighted combination of value vectors according to the attention weights.

### 3.5 The Power of Matrices: Full Parallelism

The critical insight: **the entire attention operation is a sequence of matrix multiplications.** No recurrence, no sequential dependence.

```
  Q       K^T         Scores        Softmax       V         Output
(n,d)  x (d,m)   →  (n,m)    →    (n,m)    x  (m,d)   →  (n,d)
  │       │            │             │           │           │
  └───matmul───→  ┌────┘     ┌──────┘    ┌──────┘    ┌──────┘
                  scale      softmax     matmul      done!
```

Every position computes attention simultaneously. This is why transformers are so much faster than RNNs on modern GPUs — they exploit massive parallelism.

> 💡 **Key Insight**: Attention converts a sequential problem (process tokens one-by-one) into a parallel problem (compute all pairwise interactions at once). The cost shifts from $O(n)$ sequential steps to $O(n^2)$ parallel computation — a favorable trade when you have GPUs.

### 3.6 FLOP Analysis

For scaled dot-product attention:
- $QK^T$: $O(n^2 \cdot d_k)$ — each of $n^2$ entries requires a $d_k$-dimensional dot product
- Softmax: $O(n^2)$
- Multiply by $V$: $O(n^2 \cdot d_v)$
- **Total: $O(n^2 \cdot d)$** where $d = \max(d_k, d_v)$

Memory for the attention matrix: $O(n^2)$.

This quadratic cost in sequence length $n$ is the **key limitation** of standard attention and motivates efficient attention variants (linear attention, sparse attention, FlashAttention, etc.) — but that's for later notes.

### 3.7 PyTorch Implementation from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,    # (batch, n_q, d_k)
    key: torch.Tensor,      # (batch, n_kv, d_k)
    value: torch.Tensor,    # (batch, n_kv, d_v)
    mask: torch.Tensor | None = None,  # (batch, n_q, n_kv) or broadcastable
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.
    
    Returns:
        output: (batch, n_q, d_v)
        weights: (batch, n_q, n_kv)
    """
    d_k = query.size(-1)
    
    # (batch, n_q, d_k) @ (batch, d_k, n_kv) -> (batch, n_q, n_kv)
    scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (e.g., causal mask for autoregressive generation)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax over key dimension
    weights = F.softmax(scores, dim=-1)
    
    # (batch, n_q, n_kv) @ (batch, n_kv, d_v) -> (batch, n_q, d_v)
    output = torch.bmm(weights, value)
    
    return output, weights
```

### 3.8 Causal Mask

For autoregressive models (language models, decoders), position $i$ should only attend to positions $\leq i$:

```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular mask for autoregressive attention."""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

# Example for seq_len = 4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

Positions with mask = 0 get $-\infty$ before softmax, which becomes 0 after softmax — effectively invisible.

---

## 4. Multi-Head Attention (Day 12)

### 4.1 Why One Head Isn't Enough

A single attention head computes one set of attention weights — one "way of looking at" the input. But language requires multiple simultaneous relationships:

```
"The animal didn't cross the street because it was too tired."

Head 1 (coreference):  "it" → "animal"     (what does "it" refer to?)
Head 2 (syntax):       "cross" → "animal"   (who is the subject?)
Head 3 (locality):     "tired" → "too"      (local modifier)
Head 4 (long-range):   "tired" → "didn't"   (negation scope)
```

A single head must compress all these different relationship types into one set of weights. Multiple heads let the model learn **different attention patterns in parallel**.

> 💡 **Key Insight**: Multi-head attention is like having multiple "eyes" that each look at the sequence differently. One head might focus on syntactic structure, another on semantic similarity, another on positional proximity. The outputs are combined to give the model a rich, multi-faceted view of the input.

### 4.2 The Mechanism

**Multi-head attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

And the learned projection matrices are:
- $W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$ — project queries for head $i$
- $W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$ — project keys for head $i$
- $W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$ — project values for head $i$
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_\text{model}}$ — output projection

Typically: $d_k = d_v = d_\text{model} / h$

```
Input (d_model=512)
     │
     ├──→ Head 1: project to (d_k=64), attend, output (d_v=64)
     ├──→ Head 2: project to (d_k=64), attend, output (d_v=64)
     ├──→ ...
     └──→ Head 8: project to (d_k=64), attend, output (d_v=64)
                      │
                      ├── Concat → (512)
                      ├── W^O   → (512)
                      └── Output  (d_model=512)
```

### 4.3 Parameter Count

For $h$ heads with $d_\text{model}$-dimensional input/output:

| Component | Shape | Params |
|-----------|-------|--------|
| $W_i^Q$ (all heads) | $d_\text{model} \times d_\text{model}$ | $d_\text{model}^2$ |
| $W_i^K$ (all heads) | $d_\text{model} \times d_\text{model}$ | $d_\text{model}^2$ |
| $W_i^V$ (all heads) | $d_\text{model} \times d_\text{model}$ | $d_\text{model}^2$ |
| $W^O$ | $d_\text{model} \times d_\text{model}$ | $d_\text{model}^2$ |
| **Total** | | $4 \times d_\text{model}^2$ |

For $d_\text{model} = 512$: $4 \times 512^2 = 1{,}048{,}576 \approx 1$M parameters per multi-head attention layer.

> ⚠️ **Common Pitfall**: Students often think multi-head attention has $h$ times more parameters than single-head. It doesn't! Each head operates in a *lower-dimensional* space ($d_k = d_\text{model}/h$), so the total is the same as a single-head attention with $d_k = d_\text{model}$. You get multiple attention patterns for free (in terms of parameters).

### 4.4 What Different Heads Learn

Researchers have found that different heads in trained transformers specialize in different patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Positional** | Attend to adjacent positions | "The" → "cat" (next word) |
| **Syntactic** | Attend to syntactically related tokens | verb → subject |
| **Coreference** | Attend to antecedents | "it" → "the animal" |
| **Rare token** | Attend to infrequent words | Copy rare names |
| **Separator** | Attend to punctuation/delimiters | Attend to period at end |
| **BOS/global** | Attend to first token (aggregator) | All positions → [BOS] |

Clark et al. (2019) "What Does BERT Look At?" systematically analyzed BERT's attention patterns and found these consistent specializations across layers.

### 4.5 PyTorch Implementation

```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism (Vaswani et al., 2017)."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Combined projections for all heads (more efficient than separate)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self,
        query: torch.Tensor,    # (batch, seq_q, d_model)
        key: torch.Tensor,      # (batch, seq_kv, d_model)
        value: torch.Tensor,    # (batch, seq_kv, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 1. Linear projections: (batch, seq, d_model) -> (batch, seq, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape to (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled dot-product attention per head
        # scores: (batch, n_heads, seq_q, seq_kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask: (batch, 1, seq_q, seq_kv) or (1, 1, seq_q, seq_kv)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # (batch, n_heads, seq_q, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # 4. Concat heads: (batch, seq_q, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        
        # 5. Final linear projection
        return self.W_o(attn_output)
```

### 4.6 Implementation Details That Matter

**Why `transpose` and `view` instead of `reshape`?**

The reshape from `(batch, seq, n_heads, d_k)` to `(batch, n_heads, seq, d_k)` groups all positions for each head together. This means the matmul `Q @ K^T` is batched over both `batch` and `n_heads` dimensions — extremely efficient on GPUs.

**Why `contiguous()` before the final `view`?**

After `transpose`, the tensor may not be contiguous in memory. `view` requires contiguous memory layout, so we call `contiguous()` (or use `reshape` which handles this automatically).

**Dropout on attention weights:**

In practice, dropout is applied to attention weights during training:

```python
attn_weights = F.softmax(scores, dim=-1)
attn_weights = self.attn_dropout(attn_weights)  # e.g., p=0.1
```

This prevents the model from relying too heavily on any single attention pattern.

---

## 5. Self-Attention vs Cross-Attention

### 5.1 Self-Attention

In self-attention, $Q$, $K$, $V$ all come from the **same** sequence:

```python
# Self-attention: sequence attends to itself
output = multi_head_attention(
    query=x,   # same input
    key=x,     # same input
    value=x,   # same input
)
```

Use cases:
- **Encoder**: each token attends to all other tokens in the input
- **Decoder (masked)**: each token attends to previous tokens only

```
Self-attention in encoder (bidirectional):
  The  cat  sat  on  the  mat
   ↕    ↕    ↕   ↕    ↕    ↕    (every token sees every other token)

Self-attention in decoder (causal/masked):
  I    love  cats
  ↓    ↙↓    ↙↙↓   (each token sees only previous tokens + itself)
```

### 5.2 Cross-Attention

In cross-attention, $Q$ comes from one sequence and $K$, $V$ come from **another**:

```python
# Cross-attention: decoder attends to encoder
output = multi_head_attention(
    query=decoder_states,    # from decoder
    key=encoder_outputs,     # from encoder
    value=encoder_outputs,   # from encoder
)
```

This is Bahdanau attention generalized — the decoder "looks back" at the encoder. It's used in:
- Encoder-decoder transformers (translation, summarization)
- Vision-language models (text queries attending to image patches)
- Retrieval-augmented generation (queries attending to retrieved documents)

### 5.3 Causal Masking for Autoregressive Models

In autoregressive (left-to-right) generation, position $i$ must not see future positions $j > i$. This is enforced with a causal mask:

```python
# Causal mask: lower triangular
# Position 0 sees: [0]
# Position 1 sees: [0, 1]
# Position 2 sees: [0, 1, 2]
# ...

causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq_len, seq_len)
# Add batch and head dimensions for broadcasting:
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)      # (1, 1, seq_len, seq_len)
```

Without causal masking, a language model could "cheat" by looking at the answer during training. The mask ensures the model can only use past context, matching how it generates text at inference time.

> 💡 **Key Insight**: Self-attention with causal masking is the core mechanism of GPT-style models. It's simple, parallelizable, and powerful — just one type of attention doing all the work.

### 5.4 Summary Table

| Type | Q source | K, V source | Masking | Example |
|------|----------|-------------|---------|---------|
| Self (bidirectional) | Input $X$ | Input $X$ | None | BERT encoder |
| Self (causal) | Input $X$ | Input $X$ | Causal | GPT decoder |
| Cross | Decoder $Y$ | Encoder $X$ | None | T5 decoder layer |

---

## 6. Key Equations Summary

### Additive (Bahdanau) Attention
$$e_{ij} = \mathbf{v}^T \tanh(\mathbf{W}_s \mathbf{s}_{i-1} + \mathbf{W}_h \mathbf{h}_j)$$
$$\alpha_{ij} = \text{softmax}(e_{ij})$$
$$\mathbf{c}_i = \sum_j \alpha_{ij} \mathbf{h}_j$$

### Dot-Product (Luong) Attention
$$e_{ij} = \mathbf{s}_i^T \mathbf{h}_j$$

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Complexity
- Scaled dot-product: $O(n^2 \cdot d)$ time, $O(n^2)$ memory
- Multi-head (total params): $4 \times d_\text{model}^2$

---

## 7. Key Takeaways

- **Attention solves the bottleneck**: Instead of forcing all information through a fixed-size vector, attention lets the model *retrieve* relevant information dynamically.
- **"Attention is a soft dictionary lookup"**: Query matches keys, returns weighted blend of values. This is the single most important intuition.
- **Scaling by $\sqrt{d_k}$ is essential**: Without it, softmax saturates and gradients vanish for large dimensions.
- **Matrix form enables parallelism**: Unlike RNNs, attention computes all pairwise interactions simultaneously — critical for GPU efficiency.
- **Multi-head = multiple perspectives**: Different heads learn different relationship types (syntax, coreference, position) at no extra parameter cost.
- **Self-attention vs cross-attention**: Self = same sequence, cross = different sequences. Causal masking prevents "cheating" in autoregressive models.
- **Quadratic cost is the tradeoff**: $O(n^2)$ in sequence length is the price for full pairwise interactions. This limits standard transformers to ~2K–8K tokens (without efficiency tricks).

---

## 8. Paper References

| Paper | Key Contribution |
|-------|------------------|
| Bahdanau, Cho, Bengio (2014). *"Neural Machine Translation by Jointly Learning to Align and Translate"* | Introduced additive attention for NMT |
| Luong, Pham, Manning (2015). *"Effective Approaches to Attention-based Neural Machine Translation"* | Dot-product and general attention variants |
| Vaswani et al. (2017). *"Attention Is All You Need"* | Scaled dot-product, multi-head attention, the Transformer |
| Clark et al. (2019). *"What Does BERT Look At? An Analysis of BERT's Attention"* | Systematic analysis of attention head specialization |

---

## 9. Connection to the Thread

> **Thread**: compression → prediction → attention → transformer → LLM → VLA

In Phase I, we established that **learning = compression** and **prediction = compression**. The LSTM bottleneck is a compression failure — trying to squeeze an arbitrarily long sequence into a fixed-size vector.

Attention is the breakthrough that says: **don't compress everything — learn WHAT to compress**. The model learns a *selective compression* function — the attention weights — that dynamically chooses which parts of the input are relevant for each output position.

This is a fundamentally different compression strategy:
- **RNN**: compress sequentially, lose information along the way
- **Attention**: keep everything, retrieve selectively

The transformer (next note) takes this insight to its logical conclusion: **replace recurrence entirely with attention**. If attention can selectively access any position, why bother with sequential processing at all?

> 💡 **Key Insight**: Attention is the mechanism that makes the compression *selective*. The model doesn't just predict — it learns where to look for the information it needs to predict well. This is the key enabler for scaling to longer contexts and larger models.
