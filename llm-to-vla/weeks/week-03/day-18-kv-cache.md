# Day 18: KV Cache — Why Autoregressive Inference Is Fast

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"The best optimization is the one that eliminates work you never needed to do." — Every systems engineer ever*

## Navigation
- **Previous:** [Day 17: Efficient Attention](day-17-efficient-attention.md)
- **Next:** [Day 19: Normalization + Activations](day-19-normalization-activations.md)
- **Week:** [Week 3 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 18.1 The Autoregressive Inference Problem

During generation, a transformer produces one token at a time:

```
Step 1: "The"           → predict "cat"
Step 2: "The cat"       → predict "sat"
Step 3: "The cat sat"   → predict "on"
...
Step n: "The cat sat on the..." → predict next
```

**Naive approach:** At step $n$, run the FULL transformer on all $n$ tokens.
- Step 1: process 1 token
- Step 2: process 2 tokens
- Step 3: process 3 tokens
- Step $n$: process $n$ tokens

**Total work:** $1 + 2 + 3 + \ldots + n = \frac{n(n+1)}{2} = O(n^2)$ forward passes through the model.

This is catastrophically wasteful — we're recomputing the same representations over and over.

### 18.2 The KV Cache Insight

Look at the attention computation for the **new** token at position $t$:

$$\text{Attention}(q_t, K_{1:t}, V_{1:t}) = \text{softmax}\!\left(\frac{q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

**Key observation:** Only $q_t$ changes. The keys $K_{1:t-1}$ and values $V_{1:t-1}$ for all previous tokens are the same as the last step!

```
Without KV Cache (step 4):
  ┌────────────────────┐
  │ Recompute K,V for  │ ← WASTED: identical to step 3
  │ tokens 1, 2, 3     │
  │ Compute K,V for    │ ← NEW: only this is needed
  │ token 4            │
  └────────────────────┘

With KV Cache (step 4):
  ┌────────────────────┐
  │ Load K,V from cache│ ← FREE: already computed
  │ for tokens 1, 2, 3 │
  │ Compute K,V for    │ ← Only compute this
  │ token 4            │
  │ Append to cache    │
  └────────────────────┘
```

**Speedup:** Each generation step processes 1 token instead of $t$ tokens → total work drops from $O(n^2)$ to $O(n)$.

### 18.3 KV Cache Memory Calculation

The cache stores K and V tensors for all previous tokens across all layers and heads.

$$\text{KV Cache Size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times n_{\text{seq}} \times \text{bytes}$$

The factor of 2 is for K **and** V (both cached).

**Example — LLaMA-7B:**

| Parameter | Value |
|-----------|-------|
| $n_{\text{layers}}$ | 32 |
| $n_{\text{heads}}$ | 32 |
| $d_{\text{head}}$ | 128 |
| Context length | 4,096 |
| Precision | float16 (2 bytes) |

$$\text{Cache} = 2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2,147,483,648 \text{ bytes} = \textbf{2 GB}$$

For **LLaMA-70B** at 4096 context: ~10 GB of KV cache alone!

**This is why batch size matters:** Each request in a batch needs its OWN KV cache. Serving 32 concurrent users with LLaMA-7B needs 32 × 2 GB = 64 GB just for caches.

### 18.4 Paged Attention (vLLM)

The KV cache problem: requests have **variable** lengths. Pre-allocating max-length cache wastes massive memory.

```
Traditional: Pre-allocate max_seq_len for every request
┌──────────────────────────────────────────────────┐
│ Request 1: [used: 256 tokens][wasted: 3840 empty]│
│ Request 2: [used: 2048 tokens][wasted: 2048]     │
│ Request 3: [used: 100 tokens][wasted: 3996 empty]│
└──────────────────────────────────────────────────┘
Utilization: (256+2048+100) / (4096×3) = 19.6% ← terrible!
```

**vLLM's PagedAttention** borrows from OS virtual memory:

```
Physical "pages" of KV cache (e.g., 16 tokens each):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │  Physical pages
└────┴────┴────┴────┴────┴────┴────┴────┘

Page Table (per request):
  Request 1: [P0, P3, P7] → 48 tokens, no waste
  Request 2: [P1, P2, P4, P5, P6] → 80 tokens, no waste
  
New tokens? Allocate a new page on demand.
Request done? Free its pages immediately.
```

**Result:** Near-optimal memory utilization, 2–4× more requests served concurrently.

### 18.5 Multi-Query & Grouped-Query Attention

Reduce KV cache size by sharing K,V across heads:

| Variant | K,V Heads | Cache Size | Quality |
|---------|-----------|------------|---------|
| Multi-Head (MHA) | $h$ (e.g., 32) | 100% | Baseline |
| Multi-Query (MQA) | 1 | $1/h$ (3.1%) | Slight degradation |
| Grouped-Query (GQA) | $g$ (e.g., 8) | $g/h$ (25%) | Near-MHA quality |

LLaMA-2 70B uses GQA with 8 KV heads (vs 64 Q heads) → 8× smaller KV cache.

---

## Implementation (60 min)

### 18.6 KV Cache from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class CausalSelfAttention(nn.Module):
    """Self-attention with optional KV cache for fast inference."""

    def __init__(self, d_model=512, n_heads=8, max_seq_len=1024):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Args:
            x: (batch, seq_len, d_model) — full sequence or single new token
            kv_cache: tuple of (cached_K, cached_V) or None
            use_cache: whether to return updated cache
        Returns:
            output: (batch, seq_len, d_model)
            new_cache: updated (K, V) if use_cache=True
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Append to KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # (B, H, T_total, d)
            v = torch.cat([cached_v, v], dim=2)

        # Standard causal attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask: only attend to past + current
        T_total = k.size(2)
        T_query = q.size(2)
        causal_mask = torch.triu(
            torch.ones(T_query, T_total, device=x.device, dtype=torch.bool),
            diagonal=T_total - T_query + 1
        )
        scores.masked_fill_(causal_mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T_query, C)
        out = self.out_proj(out)

        if use_cache:
            return out, (k, v)
        return out, None


def generate_with_cache(attn, prompt_emb, n_tokens=100):
    """Autoregressive generation WITH KV cache."""
    B, T, C = prompt_emb.shape
    cache = None
    outputs = []

    # Prefill: process entire prompt
    out, cache = attn(prompt_emb, kv_cache=None, use_cache=True)
    outputs.append(out[:, -1:, :])  # last token's output

    # Decode: one token at a time
    x = out[:, -1:, :]  # (B, 1, C) — just the new token
    for _ in range(n_tokens - 1):
        out, cache = attn(x, kv_cache=cache, use_cache=True)
        outputs.append(out)
        x = out  # feed output as next input (simplified)

    return torch.cat(outputs, dim=1)


def generate_without_cache(attn, prompt_emb, n_tokens=100):
    """Autoregressive generation WITHOUT KV cache (naive)."""
    B, T, C = prompt_emb.shape
    all_tokens = prompt_emb.clone()
    outputs = []

    for i in range(n_tokens):
        out, _ = attn(all_tokens, kv_cache=None, use_cache=False)
        next_tok = out[:, -1:, :]
        outputs.append(next_tok)
        all_tokens = torch.cat([all_tokens, next_tok], dim=1)

    return torch.cat(outputs, dim=1)
```

### 18.7 Benchmark: Cache vs No Cache

```python
attn = CausalSelfAttention(d_model=512, n_heads=8)
attn.eval()

prompt = torch.randn(1, 32, 512)  # 32-token prompt
n_gen = [10, 50, 100, 200]

for n in n_gen:
    # With cache
    start = time.perf_counter()
    with torch.no_grad():
        _ = generate_with_cache(attn, prompt, n_tokens=n)
    t_cache = time.perf_counter() - start

    # Without cache
    start = time.perf_counter()
    with torch.no_grad():
        _ = generate_without_cache(attn, prompt, n_tokens=n)
    t_nocache = time.perf_counter() - start

    speedup = t_nocache / t_cache
    print(f"n_tokens={n:3d} | Cache: {t_cache*1000:7.1f}ms | "
          f"No cache: {t_nocache*1000:7.1f}ms | Speedup: {speedup:.1f}x")
```

Expected output pattern:
```
n_tokens= 10 | Cache:    5.2ms | No cache:    8.1ms | Speedup: 1.6x
n_tokens= 50 | Cache:   18.3ms | No cache:   89.4ms | Speedup: 4.9x
n_tokens=100 | Cache:   34.7ms | No cache:  310.2ms | Speedup: 8.9x
n_tokens=200 | Cache:   68.1ms | No cache: 1142.5ms | Speedup: 16.8x
```

---

## Exercise (45 min)

### E18.1 KV Cache Memory for Real Models

Fill in the table (use the formula from §18.3):

| Model | Layers | Heads | $d_{\text{head}}$ | Context | Precision | KV Cache |
|-------|--------|-------|--------------------|---------|-----------|----------|
| GPT-2 (124M) | 12 | 12 | 64 | 1,024 | fp32 | ? |
| LLaMA-7B | 32 | 32 | 128 | 4,096 | fp16 | ? |
| LLaMA-70B (GQA) | 80 | 8 KV | 128 | 4,096 | fp16 | ? |
| GPT-4 (est.) | 120 | 96 | 128 | 128,000 | fp16 | ? |

**Bonus:** How many concurrent LLaMA-7B requests can an A100-80GB serve, assuming model weights take 14GB?

### E18.2 Plot Per-Token Latency

```python
import matplotlib.pyplot as plt

# Measure per-token latency during generation
def measure_per_token_latency(attn, prompt, n_tokens, use_cache):
    latencies = []
    if use_cache:
        cache = None
        out, cache = attn(prompt, kv_cache=None, use_cache=True)
        x = out[:, -1:, :]
        for _ in range(n_tokens):
            start = time.perf_counter()
            out, cache = attn(x, kv_cache=cache, use_cache=True)
            latencies.append(time.perf_counter() - start)
            x = out
    else:
        all_tokens = prompt.clone()
        for _ in range(n_tokens):
            start = time.perf_counter()
            out, _ = attn(all_tokens, kv_cache=None, use_cache=False)
            latencies.append(time.perf_counter() - start)
            all_tokens = torch.cat([all_tokens, out[:, -1:, :]], dim=1)
    return latencies

# Your task: plot latency[i] vs token index for both approaches
# What shape do you expect for each? Why?
```

### E18.3 The Serving Cost Problem

A startup serves LLaMA-7B on 8× A100-80GB GPUs.

1. Model weights (fp16): 14 GB per GPU (tensor parallel across 8)
2. Each user request has average context length 2048, generates 512 tokens
3. Each request needs its own KV cache

**Questions:**
- How much memory per request for KV cache at peak (2048 + 512 = 2560 tokens)?
- How many concurrent requests can the cluster serve?
- What happens if you switch to GQA with 8 KV heads instead of 32?

---

## Key Takeaways

1. **KV cache eliminates redundant computation** — reduces autoregressive generation from $O(n^2)$ to $O(n)$
2. **Memory is the new bottleneck** — the KV cache can consume more memory than the model weights
3. **Paged attention (vLLM)** treats KV cache like virtual memory — allocate pages on demand, eliminate waste
4. **GQA reduces cache size** by sharing K,V across query head groups — LLaMA-2 70B uses this
5. **Cache size scales linearly** with sequence length, layers, and batch size — this fundamentally limits serving capacity

## Connection to the Thread

Yesterday you learned that Flash Attention solves the O(n²) **training** bottleneck. Today's KV cache solves the O(n²) **inference** bottleneck. Together, they make modern LLM deployment viable. But the KV cache introduces a new constraint: memory per request limits how many users you can serve simultaneously. This tension between compute and memory will recur throughout the curriculum — especially when we get to VLA models that must run on robot hardware (Phase VII).

## Further Reading

- **[vLLM Paper](https://arxiv.org/abs/2309.06180)** — Kwon et al., 2023 (PagedAttention)
- **[GQA Paper](https://arxiv.org/abs/2305.13245)** — Ainslie et al., 2023
- **[Multi-Query Attention](https://arxiv.org/abs/1911.02150)** — Shazeer, 2019
- **[Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2312.07104)** — Survey
- **[Karpathy's LLM Inference](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — KV cache explained visually
