# Day 12: Multi-Head Attention

> **Phase II — Attention, Transformers & Scaling | Week 2 | 2.5 hours**
> *"Instead of performing a single attention function, we found it beneficial to linearly project the queries, keys and values h times."*
> — Vaswani et al., 2017

## Navigation
- **Previous:** [Day 11: Scaled Dot-Product Attention](day-11-scaled-dot-product-attention.md)
- **Next:** [Day 13: Positional Encoding](day-13-positional-encoding.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 12.1 Why One Head Isn't Enough

A single attention head computes one set of attention weights — one "view" of how positions relate to each other. But language has many types of relationships:

- **Syntactic:** subject–verb agreement ("The cat **sits**")
- **Semantic:** coreference ("Alice picked up the book. **She** read it.")
- **Positional:** nearby words tend to interact ("the big **red** car")

A single head must compromise between all of these. **Multi-head attention** runs multiple attention functions in parallel, each free to learn a different relationship type.

### 12.2 The Multi-Head Attention Equation

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O$$

where each head is an independent scaled dot-product attention:

$$\text{head}_i = \text{Attention}(Q W_Q^{(i)}, \; K W_K^{(i)}, \; V W_V^{(i)})$$

**Dimensions:**
- Input: $Q, K, V \in \mathbb{R}^{n \times d_{\text{model}}}$
- Per-head projections: $W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- Typically: $d_k = d_v = d_{\text{model}} / h$
- Output projection: $W_O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$

```
Input X ──┬──→ [W_Q¹, W_K¹, W_V¹] ──→ Attention ──→ head₁ ──┐
          ├──→ [W_Q², W_K², W_V²] ──→ Attention ──→ head₂ ──┤
          ├──→ [W_Q³, W_K³, W_V³] ──→ Attention ──→ head₃ ──┼──→ Concat ──→ W_O ──→ Output
          └──→ [W_Q⁴, W_K⁴, W_V⁴] ──→ Attention ──→ head₄ ──┘
```

### 12.3 Parameter Count Analysis

The key insight: multi-head attention has **approximately the same parameter count** as single-head attention with the same $d_{\text{model}}$.

For $h$ heads with $d_k = d_{\text{model}} / h$:

$$\text{Params (multi-head)} = h \times (d_{\text{model}} \times d_k + d_{\text{model}} \times d_k + d_{\text{model}} \times d_v) + h \cdot d_v \times d_{\text{model}}$$

$$= h \times 3 \times d_{\text{model}} \times \frac{d_{\text{model}}}{h} + d_{\text{model}}^2 = 3 d_{\text{model}}^2 + d_{\text{model}}^2 = 4 d_{\text{model}}^2$$

For single-head with $d_k = d_{\text{model}}$:

$$\text{Params (single-head)} = 3 d_{\text{model}}^2 + d_{\text{model}}^2 = 4 d_{\text{model}}^2$$

Same! Multiple heads are **free** — you get diversity without extra parameters.

### 12.4 What Different Heads Learn

Empirical studies (Clark et al., 2019; Voita et al., 2019) reveal that heads specialize:

| Head type | What it attends to | Example |
|---|---|---|
| **Positional** | Previous/next token | Local context |
| **Syntactic** | Verb from subject, object from verb | Dependency parsing |
| **Separator** | [SEP] or [CLS] tokens | "No-op" attention |
| **Rare token** | Infrequent or important words | Content words |
| **Coreference** | Pronouns to their referents | Long-range links |

Not all heads are equally important — **head pruning** shows that many can be removed with minimal accuracy loss (Michel et al., 2019).

### 12.5 Efficient Implementation: Reshape, Don't Loop

Naively looping over heads is slow. The standard trick: reshape the $d_{\text{model}}$ dimension into $(h, d_k)$, compute all heads simultaneously, then reshape back.

```
(batch, seq, d_model) 
    → project Q, K, V: (batch, seq, d_model) each
    → reshape: (batch, seq, h, d_k)
    → transpose: (batch, h, seq, d_k)
    → attention: (batch, h, seq, d_v)
    → transpose: (batch, seq, h, d_v)
    → reshape: (batch, seq, h * d_v)
    → project out: (batch, seq, d_model)
```

---

## Implementation (60 min)

### 12.6 Multi-Head Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        
        # Projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, d_model) → (batch, n_heads, seq, d_k)."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_k)
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, n_heads, seq, d_k) → (batch, seq, d_model)."""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # (batch, seq, n_heads, d_k)
        return x.view(batch, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,   # (batch, n_q, d_model)
        key: torch.Tensor,     # (batch, n_k, d_model)
        value: torch.Tensor,   # (batch, n_k, d_model)
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Project Q, K, V
        Q = self.split_heads(self.W_q(query))   # (batch, h, n_q, d_k)
        K = self.split_heads(self.W_k(key))     # (batch, h, n_k, d_k)
        V = self.split_heads(self.W_v(value))   # (batch, h, n_k, d_k)
        
        # Scaled dot-product attention (all heads in parallel)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask shape: (batch, 1, 1, n_k) or (batch, 1, n_q, n_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (batch, h, n_q, n_k)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch, h, n_q, d_k)
        
        # Merge heads and project
        output = self.W_o(self.merge_heads(attn_output))  # (batch, n_q, d_model)
        
        return output, attn_weights
```

### 12.7 Testing and Verification

```python
def test_multi_head_attention():
    batch, seq, d_model, n_heads = 2, 10, 512, 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch, seq, d_model)
    
    # Self-attention: Q=K=V=x
    output, weights = mha(x, x, x)
    
    assert output.shape == (batch, seq, d_model), f"Output shape: {output.shape}"
    assert weights.shape == (batch, n_heads, seq, seq), f"Weights shape: {weights.shape}"
    
    # Verify parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    expected = 4 * d_model * d_model  # W_q + W_k + W_v + W_o (no bias)
    assert total_params == expected, f"Params: {total_params} != {expected}"
    
    # Attention weights should sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Weights shape: {weights.shape}")
    print(f"✓ Parameter count: {total_params:,} = 4 × {d_model}² = {expected:,}")
    print(f"✓ d_k per head: {d_model // n_heads}")


test_multi_head_attention()
```

### 12.8 Visualizing Per-Head Attention Patterns

```python
import matplotlib.pyplot as plt


def visualize_heads(attn_weights: torch.Tensor, tokens: list[str], n_cols: int = 4):
    """Plot attention patterns for each head side by side.
    
    attn_weights: (n_heads, seq, seq) — weights from one example
    """
    n_heads = attn_weights.shape[0]
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for h in range(n_heads):
        ax = axes[h]
        ax.matshow(attn_weights[h].detach().numpy(), cmap='Blues')
        ax.set_title(f'Head {h}', fontsize=10)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, fontsize=7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
    
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)
    
    plt.suptitle('Attention Patterns per Head', fontsize=14)
    plt.tight_layout()
    plt.savefig('multi_head_patterns.png', dpi=150)
    plt.show()


# After training, you'd see heads specializing:
# Head 0: mostly diagonal (positional)
# Head 3: attends to verbs from subjects (syntactic)
# Head 7: attends to [CLS] everywhere (aggregation)
```

---

## Exercise (45 min)

### E12.1 Parameter Count Verification

For the base transformer ($d_{\text{model}} = 512$, $h = 8$):
1. Compute $d_k = d_v$ per head
2. Count parameters in $W_Q, W_K, W_V, W_O$ (with and without bias)
3. Verify that single-head ($h=1, d_k=512$) has the same count
4. What fraction of a full transformer's parameters are in MHA vs FFN?

### E12.2 Head Count Comparison

Using the same self-attention task (e.g., copy task or simple sequence classification):
1. Train with $h \in \{1, 2, 4, 8, 16\}$ heads, keeping $d_{\text{model}} = 256$ fixed
2. Compare: convergence speed, final accuracy, attention entropy
3. Plot attention entropy per head — do more heads lead to more specialization?

### E12.3 Head Pruning Experiment

After training a multi-head attention model:
1. Zero out one head at a time and measure accuracy drop
2. Rank heads by importance (biggest accuracy drop = most important)
3. Iteratively prune least important heads — how many can you remove before accuracy drops > 1%?
4. Visualize the surviving heads — what patterns do they capture?

---

## Key Takeaways

- **Multi-head attention** runs $h$ parallel attention functions, each with its own $W_Q, W_K, W_V$
- Heads split $d_{\text{model}}$ into $h$ subspaces of dimension $d_k = d_{\text{model}} / h$
- **Same parameter count** as single-head attention — diversity is free
- Different heads learn different relationship types: positional, syntactic, semantic
- Efficient implementation uses **reshape + transpose**, not loops

## Connection to the Thread

Multi-head attention is **multi-view compression**. Each head finds a different kind of redundancy in the input: one discovers syntactic patterns, another semantic ones, another positional regularities. The concat + projection step merges these views into a unified representation. This parallels how good compression algorithms use multiple models — e.g., PNG uses both horizontal and vertical prediction. The transformer gets its power not from one attention, but from many simultaneous perspectives on the same data.

## Further Reading

- Vaswani et al. **"Attention Is All You Need"** (2017), Section 3.2.2. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Clark et al. **"What Does BERT Look At? An Analysis of BERT's Attention"** (2019). [arXiv:1906.04341](https://arxiv.org/abs/1906.04341)
- Voita et al. **"Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting"** (2019). [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)
- Michel et al. **"Are Sixteen Heads Really Better than One?"** (2019). [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)
