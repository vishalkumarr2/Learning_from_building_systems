# 04 — Transformer Variants & Efficiency

> Phase II · Days 17–22 · ~15 hours
> Prerequisites: 03-transformer-architecture
> Learning Objectives: Understand efficiency improvements, KV cache, MoE, BERT, tokenization

---

## Table of Contents

1. [Efficient Attention (Day 17)](#1-efficient-attention-day-17)
2. [KV Cache (Day 18)](#2-kv-cache-day-18)
3. [Normalization & Activations (Day 19)](#3-normalization--activations-day-19)
4. [Mixture of Experts (Day 20)](#4-mixture-of-experts-day-20)
5. [BERT & Masked Language Modeling (Day 21)](#5-bert--masked-language-modeling-day-21)
6. [Tokenization Deep Dive (Day 22)](#6-tokenization-deep-dive-day-22)
7. [Key Takeaways + Paper References](#7-key-takeaways--paper-references)
8. [Connection to Thread](#8-connection-to-thread)

---

## 1. Efficient Attention (Day 17)

### 1.1 The O(n²) Problem

Standard self-attention computes a full $n \times n$ attention matrix:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Complexity**: $O(n^2 d)$ time, $O(n^2 + nd)$ memory

For context length $n$:
| n | Attention matrix size | Memory (FP16) |
|---|:---:|:---:|
| 512 | 262K | 512 KB |
| 2048 | 4.2M | 8 MB |
| 8192 | 67M | 128 MB |
| 32768 | 1.07B | 2 GB |
| 131072 | 17.2B | 32 GB |

This is **per head per layer**. For a 32-head, 32-layer model at 128K context, the attention matrices alone require terabytes.

> ⚠️ **The bottleneck is memory, not compute**: Modern GPUs have plenty of FLOPs but limited HBM bandwidth. The attention matrix must be materialized in memory, and reading/writing it is the real bottleneck.

### 1.2 Sparse Attention

**Idea**: Instead of attending to all $n$ positions, attend to a subset.

**Fixed patterns**:

```
Full Attention         Local Attention        Strided Attention
(every-to-every)       (sliding window)       (local + global)

■ ■ ■ ■ ■ ■          ■ ■ ■ · · ·            ■ ■ ■ · · ■
■ ■ ■ ■ ■ ■          ■ ■ ■ ■ · ·            ■ ■ ■ · ■ ·
■ ■ ■ ■ ■ ■          · ■ ■ ■ ■ ·            ■ ■ ■ ■ · ·
■ ■ ■ ■ ■ ■          · · ■ ■ ■ ■            · · ■ ■ ■ ·
■ ■ ■ ■ ■ ■          · · · ■ ■ ■            · ■ · ■ ■ ■
■ ■ ■ ■ ■ ■          · · · · ■ ■            ■ · · · ■ ■
```

**Longformer** (Beltagy et al., 2020): Combines local windowed attention with global attention on special tokens. Complexity: $O(n \cdot w)$ where $w$ is window size.

**BigBird** (Zaheer et al., 2020): Local + global + random attention patterns. Proven to be a universal approximator.

```python
def local_attention_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create a local attention mask with a sliding window."""
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = False
    return mask  # True = masked out
```

**Mistral's Sliding Window Attention (SWA)**: Uses a fixed window of 4096 tokens. Information propagates across windows through stacked layers — after $L$ layers, the effective receptive field is $L \times W$ tokens.

### 1.3 Flash Attention (IO-Aware Exact Attention)

Flash Attention (Dao et al., 2022) doesn't approximate attention — it computes **exact** attention but reorganizes the computation to minimize memory reads/writes.

**Key insight**: The GPU memory hierarchy has:
- SRAM (on-chip): ~20 MB, ~19 TB/s bandwidth
- HBM (off-chip): ~40-80 GB, ~1.5-3 TB/s bandwidth

Standard attention writes the $n \times n$ matrix to HBM. Flash Attention tiles the computation so everything stays in SRAM.

```
Standard Attention:
  Q, K → HBM → compute QK^T → write S to HBM → softmax → write P to HBM → P×V → write O to HBM
  
Flash Attention:
  Load Q, K, V tiles into SRAM → compute attention in tiles → accumulate → write final O to HBM
  Never materializes the full n×n attention matrix!
```

**Algorithm sketch**:
1. Split Q, K, V into blocks that fit in SRAM
2. For each block of Q:
   - For each block of K, V:
     - Compute block attention scores in SRAM
     - Update running softmax using the online softmax trick
     - Accumulate output
3. Write final output to HBM

```python
# In practice, just use PyTorch's built-in (which uses Flash Attention 2):
import torch.nn.functional as F

# Automatically uses Flash Attention when available
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    is_causal=True,  # For decoder self-attention
    dropout_p=0.0,
)
```

**Flash Attention speedups** (approximate):
| Seq Length | Speedup vs Standard | Memory Reduction |
|:---:|:---:|:---:|
| 1K | 1.5-2× | 5-20× |
| 4K | 2-3× | 20-50× |
| 16K | 3-5× | 50-100× |

> 💡 **Flash Attention is not an approximation** — it computes mathematically identical results to standard attention, just with better hardware utilization. This is why it became universal: no accuracy tradeoff.

### 1.4 Linear Attention

**Idea**: Replace $\text{softmax}(QK^T)V$ with $\phi(Q)(\phi(K)^T V)$ by exploiting associativity:

Standard: $((\text{softmax}(QK^T)) V$ → $O(n^2 d)$
Linear: $\phi(Q) \cdot (\phi(K)^T V)$ → $O(n d^2)$

The trick: compute $\phi(K)^T V$ first (a $d \times d$ matrix), then multiply by $\phi(Q)$.

$$\text{LinearAttn}(Q, K, V) = \phi(Q) \frac{\sum_{j} \phi(K_j)^T V_j}{\sum_{j} \phi(K_j)^T}$$

**Variants**:
- **Linear Transformer** (Katharopoulos et al., 2020): $\phi(x) = \text{elu}(x) + 1$
- **Random Feature Attention**: Approximate softmax kernel with random features
- **Mamba** (Gu & Dao, 2023): State-space model that processes sequences in $O(n)$ with selective state updates — not exactly linear attention but achieves similar goals

> ⚠️ **Trade-off**: Linear attention loses the "sharp" attention patterns that softmax provides. In practice, linear attention models often underperform softmax attention at the same scale. Mamba and hybrid architectures are more promising.

### 1.5 Comparison: Efficient Attention Methods

| Method | Complexity | Exact? | Quality | Used In |
|--------|:---:|:---:|:---:|---------|
| Standard | $O(n^2 d)$ | ✅ | Baseline | All |
| Flash Attention | $O(n^2 d)$ | ✅ | = Baseline | LLaMA, Mistral, GPT-4 |
| Sliding Window | $O(n \cdot w \cdot d)$ | ✅ (local) | Good | Mistral, Longformer |
| Sparse | $O(n \cdot s \cdot d)$ | ✅ (sparse) | Good | BigBird |
| Linear | $O(n \cdot d^2)$ | ❌ | Weaker | Research |
| Mamba | $O(n \cdot d)$ | N/A | Competitive | Mamba, Jamba |

> 💡 **Practical takeaway**: Use Flash Attention (it's free performance). Add sliding window for very long contexts. Don't bother with linear attention unless you're doing research.

---

## 2. KV Cache (Day 18)

### 2.1 The Autoregressive Inference Problem

During generation, a decoder-only model produces one token at a time:

```
Step 1: Input "The" → Predict "cat"
Step 2: Input "The cat" → Predict "sat"
Step 3: Input "The cat sat" → Predict "on"
...
```

**Naive approach**: At step $t$, recompute attention over all $t$ tokens. This means:
- Step 1: Process 1 token
- Step 2: Process 2 tokens
- Step 3: Process 3 tokens
- Step $n$: Process $n$ tokens
- **Total**: $1 + 2 + ... + n = O(n^2)$ forward passes

### 2.2 How KV Cache Works

**Key observation**: At step $t$, the Keys and Values for positions $0, 1, ..., t-1$ haven't changed. Only the new token $t$ needs new K, V computed.

```
Without KV Cache (step 3, generating "on"):

Q = W_Q × [emb("The"), emb("cat"), emb("sat")]   ← Recompute ALL queries
K = W_K × [emb("The"), emb("cat"), emb("sat")]   ← Recompute ALL keys  
V = W_V × [emb("The"), emb("cat"), emb("sat")]   ← Recompute ALL values

With KV Cache (step 3):

cache_K = [k("The"), k("cat")]   ← Cached from steps 1-2
cache_V = [v("The"), v("cat")]   ← Cached from steps 1-2

q_new = W_Q × emb("sat")        ← Only compute new query
k_new = W_K × emb("sat")        ← Only compute new key
v_new = W_V × emb("sat")        ← Only compute new value

K = concat(cache_K, k_new)      ← Append to cache
V = concat(cache_V, v_new)      ← Append to cache

attn = softmax(q_new × K^T / √d) × V  ← One query against all keys
```

```python
class CachedAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x, cache_k=None, cache_v=None):
        """
        Args:
            x: (batch, 1, d_model) — single new token during generation
            cache_k: (batch, n_heads, past_len, head_dim) or None
            cache_v: (batch, n_heads, past_len, head_dim) or None
        """
        batch = x.size(0)
        
        # Compute Q, K, V for new token only
        q = self.w_q(x).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Append to cache
        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)  # (batch, heads, past+1, head_dim)
            v = torch.cat([cache_v, v], dim=2)
        
        # Attention: q is (batch, heads, 1, d) @ k^T is (batch, heads, d, past+1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, 1, -1)
        
        return self.w_o(out), k, v  # Return updated cache
```

### 2.3 KV Cache Memory Analysis

For each layer, the cache stores K and V for all past tokens:

$$\text{KV cache size per token} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times \text{bytes}$$

For LLaMA 2 70B ($n_{layers}=80$, $n_{heads}=64$, $d_{head}=128$, FP16):

$$= 2 \times 80 \times 64 \times 128 \times 2 = 2.6 \text{ MB per token}$$

For a batch of 16 sequences at 4096 tokens:

$$= 16 \times 4096 \times 2.6 \text{ MB} = 170 \text{ GB}$$

> ⚠️ **KV cache is often the memory bottleneck** during inference, not model weights. For LLaMA 70B, the weights are ~140 GB but the KV cache for a large batch can easily exceed this.

### 2.4 KV Cache Optimization: Grouped-Query Attention (GQA)

Instead of separate K, V projections per head, **share** K, V across groups of heads:

```
Multi-Head Attention (MHA):     Grouped-Query Attention (GQA):
Q: h heads                      Q: h heads
K: h heads (1 per query head)   K: g groups (shared among h/g heads)
V: h heads (1 per query head)   V: g groups (shared among h/g heads)

Example: 32 query heads, 8 KV groups
Each K,V head serves 4 query heads
KV cache is 4× smaller!
```

**Multi-Query Attention (MQA)** is the extreme case: all query heads share a single K, V head. GQA is a compromise.

| Method | KV heads | Cache size | Quality |
|--------|:---:|:---:|:---:|
| MHA | $h$ | 1× | Baseline |
| GQA (8 groups) | 8 | $h/8$× | ≈ Baseline |
| MQA | 1 | $1/h$× | Slightly lower |

LLaMA 2 70B uses GQA with 8 KV heads (vs 64 query heads) → 8× smaller KV cache.

### 2.5 Paged Attention (vLLM)

**Problem**: KV cache memory is allocated per-sequence. With variable-length sequences and batching, memory is wasted on fragmentation.

**Paged Attention** (Kwon et al., 2023) borrows from OS virtual memory:
- Divide KV cache into fixed-size **pages** (blocks)
- Allocate pages on-demand as sequences grow
- Share pages across sequences (for beam search, shared prefixes)

```
Traditional allocation:          Paged allocation:
Seq 1: [████████░░░░░░░]        Seq 1: [page1][page3][page7]
Seq 2: [████░░░░░░░░░░░]        Seq 2: [page2][page5]
Seq 3: [██████████████░]        Seq 3: [page4][page6][page8][page9]
        ↑ wasted space                    ↑ no waste, dynamic allocation
```

**Result**: vLLM achieves 2-4× higher throughput than naive serving by eliminating memory waste.

### 2.6 Memory vs Compute Tradeoff Summary

```
                     Memory ◄─────────────────────────► Compute
                     
KV Cache:            Cache everything                   Recompute everything
                     ← Fast generation                  → Low memory
                     
GQA/MQA:             Full MHA                           MQA (1 KV head)
                     ← Best quality                     → Smallest cache
                     
Flash Attention:     Standard (materialize)             Flash (tiled, no materialize)
                     ← Simpler code                     → Less memory, faster
```

---

## 3. Normalization & Activations (Day 19)

### 3.1 Layer Normalization Variants

**Layer Norm (Ba et al., 2016)**: Normalize across the feature dimension:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$

**RMSNorm (Zhang & Sennrich, 2019)**: Skip the mean subtraction, only normalize by root mean square:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

```python
class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Mistral)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**Why RMSNorm?**
- ~10-15% faster than LayerNorm (no mean computation)
- Empirically equivalent quality
- Used in LLaMA, Mistral, Qwen, Gemma

### 3.2 Pre-LN vs Post-LN (Revisited)

```
Post-LN (original, 2017):              Pre-LN (modern):
                                        
x ──► SubLayer ──► Add ──► LN ──►     x ──► LN ──► SubLayer ──► Add ──►
      ▲                                            ▲
      └──────────── x ─────┘                       └──────── x ──────┘
```

**Post-LN issues**:
- Gradient scale grows with depth → unstable training
- Requires careful LR warmup
- Hard to train beyond ~12 layers without tricks

**Pre-LN advantages**:
- Gradient through residual path is always $O(1)$
- Much more stable, can train 100+ layers easily
- Needs final LN after last layer

> 💡 **Sandwich-LN**: Some architectures add norm both before AND after the sub-layer for extra stability. Used in some vision transformers.

### 3.3 Activation Functions

**ReLU** (classic): $\text{ReLU}(x) = \max(0, x)$
- Simple, fast
- "Dead neuron" problem: once a neuron outputs 0, it may never recover

**GELU** (Hendrycks & Gimpel, 2016): $\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$
- Smooth approximation of ReLU
- Used in BERT, GPT-2, many models

**SiLU / Swish**: $\text{SiLU}(x) = x \cdot \sigma(x)$
- Self-gated: the input gates itself
- Used in LLaMA, Mistral (within SwiGLU)

### 3.4 GLU Variants and SwiGLU

**Gated Linear Unit (GLU)** (Dauphin et al., 2017):

$$\text{GLU}(x) = (xW_1 + b_1) \otimes \sigma(xW_3 + b_3)$$

The idea: one linear projection produces the content, another produces a gate that controls information flow.

**SwiGLU** (Shazeer, 2020) — the modern standard:

$$\text{SwiGLU}(x, W_1, W_3, W_2) = (\text{SiLU}(xW_1) \otimes xW_3) W_2$$

```python
class SwiGLU(torch.nn.Module):
    """SwiGLU activation as used in LLaMA."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

> ⚠️ **Parameter count**: SwiGLU has 3 weight matrices ($W_1, W_3, W_2$) instead of 2, so the hidden dimension is typically reduced to $\frac{2}{3} \times 4d_{model}$ (rounded to multiples of 256) to keep total parameters similar.

### 3.5 Comparison Table

| Activation | Formula | Parameters | Used In |
|-----------|---------|:---:|---------|
| ReLU | $\max(0, xW_1)W_2$ | $2 \times d \times d_{ff}$ | Original Transformer |
| GELU | $\text{GELU}(xW_1)W_2$ | $2 \times d \times d_{ff}$ | BERT, GPT-2 |
| SwiGLU | $(\text{SiLU}(xW_1) \otimes xW_3)W_2$ | $3 \times d \times d'_{ff}$ | LLaMA, Mistral, Qwen |

---

## 4. Mixture of Experts (Day 20)

### 4.1 The Core Idea: Sparse Activation

**Problem**: Larger models = better performance, but also = more compute per token.

**MoE insight**: What if we could have a huge model but only use a fraction of it for each token?

$$\text{MoE}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

Where:
- $E_i$ are **expert** networks (typically FFN layers)
- $G(x)$ is a **gating/routing** network that produces sparse weights
- Only the top-$k$ experts are actually computed

```
Input x ──► Router ──► [Expert 1: 0.6] [Expert 2: 0.0] [Expert 3: 0.4] [Expert 4: 0.0]
                              │                                 │
                              ▼                                 ▼
                        FFN₁(x) × 0.6                   FFN₃(x) × 0.4
                              │                                 │
                              └──────────► Add ◄───────────────┘
                                            │
                                         Output
```

### 4.2 Top-k Routing

```python
class MoERouter(torch.nn.Module):
    """Top-k router for Mixture of Experts."""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            indices: (batch, seq_len, top_k) — which experts to use
            weights: (batch, seq_len, top_k) — expert weights (sum to 1)
        """
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        return top_k_indices, top_k_weights
```

### 4.3 Load Balancing

**Problem**: Without constraints, the router may send all tokens to a few experts, wasting the others.

**Auxiliary load balancing loss**:

$$\mathcal{L}_{balance} = N \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$
- $P_i$ = mean routing probability for expert $i$
- $N$ = number of experts

This loss is minimized when load is perfectly balanced ($f_i = 1/N$ for all $i$).

```python
def load_balancing_loss(router_logits, top_k_indices, num_experts):
    """Compute auxiliary load balancing loss."""
    probs = torch.softmax(router_logits, dim=-1)  # (batch*seq, num_experts)
    
    # Fraction of tokens per expert
    one_hot = torch.nn.functional.one_hot(top_k_indices, num_experts).float()
    tokens_per_expert = one_hot.sum(dim=(0, 1)) / one_hot.sum()  # f_i
    
    # Mean probability per expert
    prob_per_expert = probs.mean(dim=0)  # P_i
    
    return num_experts * (tokens_per_expert * prob_per_expert).sum()
```

### 4.4 Switch Transformer (Fedus et al., 2022)

Simplified MoE with top-1 routing:
- Each token goes to exactly **one** expert
- Simpler, faster, easier to scale
- Expert capacity factor limits max tokens per expert
- Achieved 4-7× speedup over dense T5 at similar quality

### 4.5 Mixtral (Jiang et al., 2024)

Mixtral 8x7B: A practical, high-quality MoE model:
- 8 expert FFN layers per MoE layer, top-2 routing
- 46.7B total parameters, but only ~12.9B active per token
- Matches or beats LLaMA 2 70B on most benchmarks
- Much faster inference (only 2/8 experts active)

```
Mixtral Architecture (per layer):
┌─────────────┐
│  Self-Attn   │  ← Standard attention (NOT MoE)
│  + RMSNorm   │
├─────────────┤
│  MoE FFN     │  ← 8 SwiGLU experts, top-2 routing
│  + RMSNorm   │
└─────────────┘
```

> 💡 **Only the FFN is MoE**: Attention layers are shared across all tokens. This makes sense — attention needs global context, while FFN is per-position "knowledge lookup" where specialization helps.

### 4.6 MoE for VLAs

MoE is particularly interesting for VLAs:
- **Different experts for different modalities**: Vision tokens might route to different experts than language tokens
- **Task specialization**: Navigation, manipulation, and language tasks might activate different experts
- **Efficiency**: Robot models need to be fast for real-time control; MoE gives model capacity without proportional compute cost

---

## 5. BERT & Masked Language Modeling (Day 21)

### 5.1 The Encoder-Only Paradigm

BERT (Devlin et al., 2019) showed that a transformer **encoder** alone, trained with masked language modeling (MLM), produces powerful general-purpose representations.

```
Decoder-only (GPT):             Encoder-only (BERT):
Predict next token               Predict masked tokens
Sees only past                   Sees ALL context (bidirectional)

"The [cat] sat" → "on"          "The [MASK] sat on the [MASK]" → "cat", "mat"
   ← ← ← ← ←                      → ← → ← → ← → ← → ← →
```

### 5.2 Masked Language Modeling (MLM)

**Training procedure**:
1. Randomly select 15% of tokens
2. Of selected tokens:
   - 80% → replace with `[MASK]`
   - 10% → replace with random token
   - 10% → keep original
3. Predict the original tokens

```python
def create_mlm_targets(token_ids, vocab_size, mask_token_id, mask_prob=0.15):
    """Create MLM training inputs and targets."""
    labels = token_ids.clone()
    
    # Random mask positions
    probability_matrix = torch.full(token_ids.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Don't mask special tokens (CLS, SEP, PAD)
    labels[~masked_indices] = -100  # Ignore in loss
    
    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(token_ids.shape, 0.8)).bool() & masked_indices
    token_ids[indices_replaced] = mask_token_id
    
    # 10% of the time, replace with random word
    indices_random = (
        torch.bernoulli(torch.full(token_ids.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, token_ids.shape)
    token_ids[indices_random] = random_words[indices_random]
    
    # 10% of the time, keep the original word (implicitly)
    
    return token_ids, labels
```

### 5.3 BERT Architecture Details

```
Input:  [CLS] The cat sat on the mat [SEP]
         ↓    ↓   ↓   ↓  ↓  ↓   ↓   ↓
Token:  [101] 1996 4937 2068 2006 1996 13523 [102]
Segment: 0    0    0    0    0    0    0     0
Position: 0    1    2    3    4    5    6     7
         ↓
Embedding = TokenEmbed + SegmentEmbed + PositionEmbed
         ↓
    [12 × Transformer Encoder Layers]
         ↓
    Contextual representations
```

**Special tokens**:
- `[CLS]`: Classification token — its final representation is used for sentence-level tasks
- `[SEP]`: Separator between segments (for sentence-pair tasks)
- `[MASK]`: Replaces tokens during MLM training

**BERT sizes**:
| Model | Layers | Hidden | Heads | Params |
|-------|:---:|:---:|:---:|:---:|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### 5.4 Fine-tuning BERT

The beauty of BERT: pretrain once, fine-tune for many tasks:

```
Classification:        [CLS] output → Linear → Class label
NER:                   Each token output → Linear → Entity tag
Question Answering:    Token outputs → Start/End span pointers
Sentence Similarity:   [CLS] output → Cosine similarity
```

```python
class BertForClassification(torch.nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model  # Pretrained
        self.classifier = torch.nn.Linear(768, num_classes)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[:, 0, :]  # [CLS] token
        return self.classifier(self.dropout(cls_output))
```

### 5.5 Encoder-Only vs Decoder-Only

| Aspect | Encoder-Only (BERT) | Decoder-Only (GPT) |
|--------|:---:|:---:|
| Context | Bidirectional | Unidirectional (causal) |
| Pretraining | MLM (15% of tokens) | Next token (100% of tokens) |
| Training efficiency | Lower (only predicts masked) | Higher (predicts every token) |
| Generation | Cannot generate naturally | Natural generation |
| Understanding | Excellent | Excellent (at scale) |
| Scaling winner | ❌ | ✅ |

> 💡 **Why decoder-only won**: (1) Training signal from every token, not just 15%. (2) Natural generation without architectural tricks. (3) Simpler to scale. (4) In-context learning emerges. BERT was revolutionary but the field moved to decoder-only.

---

## 6. Tokenization Deep Dive (Day 22)

### 6.1 Why Tokenization Matters

The tokenizer determines the model's **vocabulary** — the atomic units it reads and writes. This is a critical design decision:

```
"unfortunately" as different tokenizations:

Character:    u n f o r t u n a t e l y     → 13 tokens
Word:         unfortunately                  → 1 token (but what about rare words?)
BPE:          un fortunate ly                → 3 tokens
SentencePiece: _unfortunately               → 1 token (with leading space)
```

**Why it matters**:
- Sequence length is proportional to token count → affects attention cost
- Rare words must be representable → subword tokenization
- Tokenization affects what the model can "see" and learn
- Bad tokenization → wasted model capacity

### 6.2 Character-Level Tokenization

**Pros**: Handles any text, no unknown tokens, small vocabulary (~256)
**Cons**: Very long sequences, model must learn character→word mapping from scratch

$$\text{Sequence length} \approx 4\text{×} \text{ word-level}$$

### 6.3 Word-Level Tokenization

**Pros**: Natural units, short sequences
**Cons**: Fixed vocabulary, can't handle new/rare words, huge embedding matrices

### 6.4 BPE — Byte Pair Encoding (Step by Step)

BPE (Sennrich et al., 2016) starts with characters and iteratively merges the most frequent pairs:

**Training the tokenizer**:

```
Corpus: "low lower lowest low"

Step 0 — Character vocabulary:
  l, o, w, e, r, s, t, <space>

Step 0 — Word frequencies (with end-of-word marker):
  "l o w </w>"     → 2
  "l o w e r </w>" → 1  
  "l o w e s t </w>" → 1

Step 1 — Most frequent pair: (l, o) → merge to "lo"
  "lo w </w>"       → 2
  "lo w e r </w>"   → 1
  "lo w e s t </w>" → 1

Step 2 — Most frequent pair: (lo, w) → merge to "low"
  "low </w>"        → 2
  "low e r </w>"    → 1
  "low e s t </w>"  → 1

Step 3 — Most frequent pair: (low, </w>) → merge to "low</w>"
  "low</w>"         → 2
  "low e r </w>"    → 1
  "low e s t </w>"  → 1

Step 4 — Most frequent pair: (e, r) → merge to "er"
  "low</w>"         → 2
  "low er </w>"     → 1
  "low e s t </w>"  → 1

... continue until vocabulary size reached
```

**Tokenizing new text** (apply merges in learned order):

```python
def bpe_tokenize(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply BPE merges to tokenize text."""
    # Start with characters
    tokens = list(text)
    
    for merge_pair in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if (i < len(tokens) - 1 
                and tokens[i] == merge_pair[0] 
                and tokens[i+1] == merge_pair[1]):
                new_tokens.append(merge_pair[0] + merge_pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return tokens
```

### 6.5 WordPiece

WordPiece (Schuster & Nakajima, 2012; used by BERT):
- Similar to BPE but merges maximize **likelihood** rather than frequency
- Uses `##` prefix for continuation tokens

```
"playing" → ["play", "##ing"]
"unhappiness" → ["un", "##happy", "##ness"]
```

### 6.6 SentencePiece / Unigram

**SentencePiece** (Kudo & Richardson, 2018):
- Treats the input as a raw byte stream (language-agnostic)
- Uses `▁` (U+2581) to represent spaces
- Two algorithms: BPE or Unigram

**Unigram Language Model**:
- Start with a large vocabulary
- Iteratively **remove** tokens that least affect the overall likelihood
- More principled than BPE's greedy merging

```
"Hello world" → ["▁Hello", "▁world"]
"tokenization" → ["▁token", "ization"]  or  ["▁to", "ken", "ization"]
```

### 6.7 Modern Tokenizers

**tiktoken** (OpenAI, GPT-4):
- BPE on bytes (byte-level BPE)
- Very fast (Rust implementation)
- ~100K vocabulary
- Handles any UTF-8 text

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

tokens = enc.encode("Hello, world!")
print(tokens)       # [9906, 11, 1917, 0]
print(len(tokens))  # 4

text = enc.decode(tokens)
print(text)          # "Hello, world!"
```

**LLaMA tokenizer** (SentencePiece BPE):
- 32K vocabulary (LLaMA 2) or 128K (LLaMA 3)
- Byte-fallback for unknown characters
- Separate digits into individual tokens (better math)

### 6.8 Tokenization Pitfalls

> ⚠️ **Numbers**: Many tokenizers split numbers inconsistently:
> ```
> "123456" might become ["123", "456"] or ["12", "34", "56"] or ["1", "234", "56"]
> ```
> This makes arithmetic hard for LLMs.

> ⚠️ **Multilingual**: English-centric tokenizers fragment non-English text heavily:
> ```
> "hello" → 1 token
> "こんにちは" → 3-5 tokens (depending on tokenizer)
> ```

> ⚠️ **Leading spaces**: SentencePiece treats " hello" and "hello" differently. The space is part of the token.

> ⚠️ **Code**: Programming languages have different tokenization needs than natural language. Special tokens for indentation, brackets, operators.

### 6.9 Tokenization for Robotics / VLAs

For VLAs, tokenization extends beyond text:
- **Actions**: Discretize continuous actions into bins (e.g., 256 bins per dimension)
- **Images**: Use a vision encoder (ViT) to produce visual tokens
- **Proprioception**: Discretize joint angles, velocities

```
Text tokens:     "Pick up the red block"  → [43, 892, 1034, ...]
Vision tokens:   [image patch features]    → [5001, 5002, 5003, ...]  
Action tokens:   [x, y, z, grip]           → [6128, 6045, 6100, 6200]
```

> 💡 **The entire VLA pipeline**: tokenize everything (text + images + state + actions) into a unified token space, then model the joint distribution with a transformer.

---

## 7. Key Takeaways + Paper References

### Key Takeaways

1. **Flash Attention is free performance** — exact attention with better hardware utilization; no accuracy loss
2. **KV cache is essential for inference** — trades memory for compute; GQA reduces memory; paged attention reduces waste
3. **RMSNorm + Pre-LN + SwiGLU** is the modern standard (LLaMA, Mistral, Qwen)
4. **MoE gives scale without proportional compute** — Mixtral 8x7B runs as fast as a 13B model with near-70B quality
5. **BERT showed the power of bidirectional encoders** but decoder-only models won for scaling
6. **Tokenization is a critical design choice** — BPE is standard, byte-level handles all text, subword is the sweet spot
7. **Everything is tokenizable** — the VLA paradigm tokenizes images, text, states, and actions into one sequence

### Paper References

| Paper | Year | Key Contribution |
|-------|:---:|------------------|
| Dao et al. — "FlashAttention: Fast and Memory-Efficient Exact Attention" | 2022 | IO-aware attention |
| Dao — "FlashAttention-2" | 2023 | Further optimizations |
| Kwon et al. — "Efficient Memory Management for Large Language Model Serving with PagedAttention" | 2023 | vLLM, paged KV cache |
| Ainslie et al. — "GQA: Training Generalized Multi-Query Transformer Models" | 2023 | Grouped-Query Attention |
| Shazeer — "GLU Variants Improve Transformer" | 2020 | SwiGLU |
| Zhang & Sennrich — "Root Mean Square Layer Normalization" | 2019 | RMSNorm |
| Shazeer et al. — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" | 2017 | MoE foundations |
| Fedus et al. — "Switch Transformers: Scaling to Trillion Parameter Models" | 2022 | Top-1 MoE routing |
| Jiang et al. — "Mixtral of Experts" | 2024 | Practical MoE |
| Devlin et al. — "BERT: Pre-training of Deep Bidirectional Transformers" | 2019 | MLM, encoder-only |
| Sennrich et al. — "Neural Machine Translation of Rare Words with Subword Units" | 2016 | BPE |
| Kudo & Richardson — "SentencePiece" | 2018 | Language-agnostic tokenization |
| Beltagy et al. — "Longformer: The Long-Document Transformer" | 2020 | Sparse attention |
| Gu & Dao — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" | 2023 | SSM alternative |

---

## 8. Connection to Thread

> **The Thread**: Efficiency = better compression at lower cost; MoE = conditional compression

The efficiency improvements in this note are all about the same thing: **achieving the same information compression with less resource expenditure**.

1. **Flash Attention**: Same compression, less memory I/O. The attention computation is information compression — selecting what's relevant. Flash Attention just does it more efficiently on hardware.

2. **KV Cache**: Avoid recomputing compressions you've already done. Each K, V vector is a compressed representation of a position — cache it, don't recompute.

3. **MoE**: **Conditional compression** — different tokens need different transformations. Instead of one FFN processing everything, route to specialized experts. This is the same principle as attention (selective processing) applied to the FFN.

4. **Tokenization**: The first compression step. BPE compresses frequent character sequences into single tokens. Better tokenization = better initial compression = shorter sequences = faster attention.

**The theme**: As we scale transformers, every layer of the stack must become more efficient at its compression task. This carries directly to VLAs, where real-time robot control demands both rich representations and fast inference.

> Next: [05-gpt-scaling](05-gpt-scaling.md) — GPT, scaling laws, pretraining, and the emergence of capabilities
