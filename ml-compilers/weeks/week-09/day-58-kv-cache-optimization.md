# Day 58: KV Cache Optimization

> **Phase IV · Week 9 · Day 58 of 70 · 2.5 hours**

*"The fastest byte is the one you never read. The cheapest memory is the one you never allocate."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 57: LLM Inference Challenges](day-57-llm-inference-challenges.md) | [Day 59: vLLM & PagedAttention](day-59-vllm-pagedattention.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Yesterday we showed that the KV cache is the dominant memory consumer during LLM inference — a single LLaMA-2 70B request at 4K context uses 20 GB just for cached keys and values. This directly limits how many requests you can serve concurrently, which determines your cost per token. Today we attack this bottleneck. Multi-Query Attention can shrink the cache by 32×. Grouped-Query Attention finds the practical sweet spot at 4-8×. Sliding window attention caps memory regardless of sequence length. These are not theoretical — they're deployed in production at every major LLM provider.

---

## 1. Anatomy of the KV Cache

Let's dissect exactly what's stored in the KV cache and where the memory goes:

```
KV Cache Structure — Per Layer
═══════════════════════════════════════════════════════════════

  Standard Multi-Head Attention (MHA):
  ────────────────────────────────────

  n_heads = 32, head_dim = 128, seq_len = S

  Key cache:   [batch, n_heads, seq_len, head_dim]
               [B,     32,      S,       128     ]

  Value cache: [batch, n_heads, seq_len, head_dim]
               [B,     32,      S,       128     ]

  Memory per layer per token = 2 × n_heads × head_dim × sizeof(dtype)
                             = 2 × 32 × 128 × 2   (FP16)
                             = 16,384 bytes = 16 KB

  ┌────────────────────────────────────────────────────┐
  │  Model          Layers  KV/tok   KV @ 4K   KV @ 128K │
  │  LLaMA-2 7B     32      512 KB   2 GB      64 GB     │
  │  LLaMA-2 70B    80      1.25 MB  5 GB      160 GB    │
  │  Mixtral 8x7B   32      512 KB   2 GB      64 GB     │
  │  GPT-4 (est.)   120     ~4 MB    ~16 GB    ~512 GB   │
  └────────────────────────────────────────────────────┘
```

---

## 2. Multi-Query Attention (MQA)

The key insight of MQA (Shazeer, 2019): **queries need per-head projections, but keys and values can be shared across all heads**.

```
Multi-Head vs Multi-Query Attention
═══════════════════════════════════════════════════════════════

  Multi-Head Attention (MHA):
  ──────────────────────────
  Q: [B, 32, S, 128]    ← 32 separate query heads
  K: [B, 32, S, 128]    ← 32 separate key heads
  V: [B, 32, S, 128]    ← 32 separate value heads

  KV parameters: 2 × 32 × d × d_head


  Multi-Query Attention (MQA):
  ────────────────────────────
  Q: [B, 32, S, 128]    ← 32 separate query heads (unchanged)
  K: [B,  1, S, 128]    ← 1 shared key head  ← SHARED!
  V: [B,  1, S, 128]    ← 1 shared value head ← SHARED!

  KV parameters: 2 × 1 × d × d_head

  Memory reduction: 32× for KV cache!
  Quality impact:   ~0.5-1% degradation on benchmarks
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention: shared KV heads, independent Q heads."""
    
    def __init__(self, d_model: int = 4096, n_heads: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)          # Full
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)    # Single head!
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)    # Single head!
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, kv_cache=None):
        B, S, D = x.shape
        
        # Q has n_heads independent projections
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # K and V have a single shared projection
        k = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        
        # Append to cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        
        # Broadcast K, V across all query heads
        # k: [B, 1, S_total, head_dim] → broadcast with q: [B, n_heads, S, head_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # v broadcasts from [B, 1, ...] to [B, n_heads, ...]
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out), new_cache
```

---

## 3. Grouped-Query Attention (GQA)

GQA (Ainslie et al., 2023) is the practical compromise between MHA and MQA. Instead of 1 KV head (MQA) or $n$ KV heads (MHA), GQA uses $g$ KV head groups:

```
Attention Variants Comparison
═══════════════════════════════════════════════════════════════

  MHA (n_kv_heads = n_heads = 32):
  Q: H₁ H₂ H₃ H₄ H₅ H₆ H₇ H₈ ... H₃₂
  K: H₁ H₂ H₃ H₄ H₅ H₆ H₇ H₈ ... H₃₂    ← 32 KV heads
  V: H₁ H₂ H₃ H₄ H₅ H₆ H₇ H₈ ... H₃₂

  GQA-8 (n_kv_heads = 8, group_size = 4):
  Q: H₁ H₂ H₃ H₄ │ H₅ H₆ H₇ H₈ │ ... │ H₂₉ H₃₀ H₃₁ H₃₂
  K:     G₁        │     G₂        │ ... │      G₈            ← 8 KV heads
  V:     G₁        │     G₂        │ ... │      G₈

  MQA (n_kv_heads = 1):
  Q: H₁ H₂ H₃ H₄ H₅ H₆ H₇ H₈ ... H₃₂
  K:              G₁                         ← 1 KV head
  V:              G₁

  ┌──────────────────────────────────────────────────────┐
  │  Variant   KV heads   Cache reduction   Quality      │
  │  MHA         32       1× (baseline)     Best         │
  │  GQA-8        8       4×                ~Baseline    │
  │  GQA-4        4       8×                Minimal loss │
  │  MQA           1       32×              Slight loss  │
  └──────────────────────────────────────────────────────┘
```

LLaMA-2 70B uses GQA with 8 KV heads (vs 64 query heads), giving an 8× cache reduction. LLaMA-3 uses GQA across all model sizes.

```python
class GroupedQueryAttention(nn.Module):
    """GQA: n_kv_heads < n_heads, each KV head serves a group of Q heads."""
    
    def __init__(self, d_model: int = 4096, n_heads: int = 32, n_kv_heads: int = 8):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # queries per KV group
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        
        # Expand KV heads to match Q heads by repeating
        # k: [B, n_kv_heads, S, head_dim] → [B, n_heads, S, head_dim]
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_cache
```

---

## 4. Sliding Window Attention

Mistral introduced sliding window attention (SWA) where each token attends only to the last $W$ tokens, not the full context:

```
Sliding Window Attention (W = 4)
═══════════════════════════════════════════════════════════════

  Full Attention mask:           Sliding Window mask (W=4):
  t₁ t₂ t₃ t₄ t₅ t₆ t₇ t₈     t₁ t₂ t₃ t₄ t₅ t₆ t₇ t₈
  ✓  .  .  .  .  .  .  .       ✓  .  .  .  .  .  .  .
  ✓  ✓  .  .  .  .  .  .       ✓  ✓  .  .  .  .  .  .
  ✓  ✓  ✓  .  .  .  .  .       ✓  ✓  ✓  .  .  .  .  .
  ✓  ✓  ✓  ✓  .  .  .  .       ✓  ✓  ✓  ✓  .  .  .  .
  ✓  ✓  ✓  ✓  ✓  .  .  .       .  ✓  ✓  ✓  ✓  .  .  .  ← evict t₁
  ✓  ✓  ✓  ✓  ✓  ✓  .  .       .  .  ✓  ✓  ✓  ✓  .  .
  ✓  ✓  ✓  ✓  ✓  ✓  ✓  .       .  .  .  ✓  ✓  ✓  ✓  .
  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓       .  .  .  .  ✓  ✓  ✓  ✓

  KV cache: O(n) growing     KV cache: O(W) constant!

  Key insight: information propagates through layers.
  After L layers with window W, effective receptive field = L × W.
  Mistral 7B: W=4096, L=32 → effective context = 131,072 tokens
```

$$\text{KV cache memory (SWA)} = 2 \times L \times n_{\text{kv\_heads}} \times W \times d_h \times \text{sizeof(dtype)}$$

This is **constant** regardless of actual sequence length — a massive advantage for long-context serving.

---

## 5. KV Cache Compression Techniques

Beyond architectural changes, we can compress the cache post-hoc:

```
KV Cache Compression Strategies
═══════════════════════════════════════════════════════════════

  ┌────────────────────┐
  │ Quantization       │  FP16 → INT8 or INT4 per-token
  │ 2-4× reduction     │  Minimal quality loss with per-channel scales
  └────────────────────┘
  ┌────────────────────┐
  │ Token Eviction     │  Drop least-important tokens from cache
  │ ~50% reduction     │  Keep: recent + high-attention + initial tokens
  └────────────────────┘
  ┌────────────────────┐
  │ Token Merging      │  Combine similar adjacent tokens
  │ ~30% reduction     │  Average similar K/V representations
  └────────────────────┘
  ┌────────────────────┐
  │ Low-Rank Approx.   │  Project KV to lower dimension
  │ 2-4× reduction     │  SVD-based compression per layer
  └────────────────────┘
```

```python
def quantize_kv_cache(k_cache: torch.Tensor, v_cache: torch.Tensor, bits: int = 8):
    """Quantize KV cache to INT8 per-channel for memory reduction.
    
    Args:
        k_cache: [batch, n_kv_heads, seq_len, head_dim] in FP16
        v_cache: same shape
        bits: quantization bits (4 or 8)
    
    Returns:
        Quantized cache + scales for dequantization
    """
    qmax = 2 ** (bits - 1) - 1
    
    def quantize_tensor(t):
        # Per-channel quantization (along head_dim)
        scale = t.abs().amax(dim=-1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)
        t_q = (t / scale).round().clamp(-qmax, qmax).to(torch.int8)
        return t_q, scale
    
    k_q, k_scale = quantize_tensor(k_cache)
    v_q, v_scale = quantize_tensor(v_cache)
    
    original_bytes = (k_cache.numel() + v_cache.numel()) * 2  # FP16
    compressed_bytes = (k_q.numel() + v_q.numel()) * 1 + \
                       (k_scale.numel() + v_scale.numel()) * 2
    
    print(f"Original:   {original_bytes / 1e6:.1f} MB")
    print(f"Quantized:  {compressed_bytes / 1e6:.1f} MB")
    print(f"Ratio:      {original_bytes / compressed_bytes:.2f}×")
    
    return (k_q, k_scale), (v_q, v_scale)


def evict_tokens(k_cache, v_cache, attention_scores, keep_ratio=0.5, protect_recent=64):
    """H2O-style token eviction: keep heavy hitters + recent tokens."""
    B, H, S, D = k_cache.shape
    n_keep = int(S * keep_ratio)
    
    # Always protect the most recent tokens
    # Score the rest by cumulative attention received
    cumulative_attn = attention_scores.sum(dim=2)  # [B, H, S_kv]
    
    # Mask recent tokens with high scores to protect them
    cumulative_attn[:, :, -protect_recent:] = float('inf')
    
    # Keep top-k by attention score
    _, keep_idx = cumulative_attn.topk(n_keep, dim=-1)
    keep_idx = keep_idx.sort(dim=-1).values  # maintain order
    
    # Gather kept tokens
    k_kept = k_cache.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, D))
    v_kept = v_cache.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, D))
    
    return k_kept, v_kept
```

---

## 6. Memory Layout Optimization

How KV cache is laid out in memory affects performance due to GPU cache line behavior:

```
Memory Layout: Heads-First vs Tokens-First
═══════════════════════════════════════════════════════════════

  Heads-first (standard PyTorch): [B, n_heads, seq_len, head_dim]
  ─────────────────────────────────────────────────────────────
  Memory: |H0_T0|H0_T1|H0_T2|...|H1_T0|H1_T1|H1_T2|...|

  Appending token T3 → scatter across n_heads locations
  Good for: attention (contiguous head slices)

  Tokens-first (append-friendly): [B, seq_len, n_heads, head_dim]
  ─────────────────────────────────────────────────────────────
  Memory: |T0_H0|T0_H1|T0_H2|...|T1_H0|T1_H1|T1_H2|...|

  Appending token T3 → single contiguous write
  Good for: token append (contiguous per-token)

  Production systems (vLLM, TensorRT-LLM):
  ────────────────────────────────────────
  Use paged block layout — covered tomorrow in PagedAttention.
  Blocks of [block_size, n_heads, head_dim] indexed via block table.
```

---

## Hands-On Exercise: Measure KV Cache Optimizations

```python
def kv_cache_comparison():
    """Compare memory usage across MHA, GQA, MQA with/without quantization."""
    configs = {
        "LLaMA-2 7B (MHA)":   {"layers": 32, "kv_heads": 32, "head_dim": 128},
        "LLaMA-2 7B (GQA-8)": {"layers": 32, "kv_heads": 8,  "head_dim": 128},
        "LLaMA-2 7B (MQA)":   {"layers": 32, "kv_heads": 1,  "head_dim": 128},
        "LLaMA-3 8B (GQA-8)": {"layers": 32, "kv_heads": 8,  "head_dim": 128},
        "Mistral 7B (SWA)":   {"layers": 32, "kv_heads": 8,  "head_dim": 128,
                                "window": 4096},
    }
    
    seq_len = 32768  # 32K context
    gpu_mem = 80     # A100 80GB
    
    print(f"{'Config':<25} {'FP16':>8} {'INT8':>8} {'INT4':>8} {'Max Batch (FP16)':>16}")
    print("─" * 75)
    
    for name, cfg in configs.items():
        effective_seq = cfg.get("window", seq_len)
        cache_bytes = (2 * cfg["layers"] * cfg["kv_heads"] * effective_seq 
                       * cfg["head_dim"] * 2)  # FP16
        cache_gb = cache_bytes / 1e9
        
        model_gb = 14  # approximate for 7B
        max_batch = int((gpu_mem - model_gb) / cache_gb)
        
        print(f"{name:<25} {cache_gb:>7.1f}G {cache_gb/2:>7.1f}G "
              f"{cache_gb/4:>7.1f}G {max_batch:>16}")

kv_cache_comparison()
```

### Exercise Tasks

1. **Implement GQA from MHA**: Take a trained MHA checkpoint and convert it to GQA by averaging KV head weights within groups. Measure quality change on a small eval set.
2. **Cache eviction benchmark**: Implement H2O-style eviction with different keep ratios (25%, 50%, 75%). Measure perplexity impact on WikiText.
3. **Quantized cache decode**: Implement a decode step that reads INT8-quantized KV cache, dequantizes on-the-fly, and produces correct attention output. Compare speed vs FP16.

---

## Key Takeaways

1. **MQA reduces KV cache by $n_{\text{heads}} \times$** by sharing K/V projections — the simplest and most effective optimization
2. **GQA is the production standard** — LLaMA-2 70B, LLaMA-3, and Mistral all use it, balancing quality and cache size
3. **Sliding window attention caps memory at $O(W)$** regardless of context length, with effective receptive field $L \times W$
4. **INT8 KV cache quantization** gives nearly 2× memory savings with negligible quality loss
5. **Token eviction** (H2O) can drop 50% of cached tokens by removing low-attention tokens
6. **Memory layout matters** — the order of dimensions affects append latency and attention compute efficiency

---

## Further Reading

- **Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need"** (2019) — original MQA paper
- **Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** (2023)
- **Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"** (2023) — token eviction
- **Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"** (2024)
- **Jiang et al., "Mistral 7B"** (2023) — sliding window attention in practice

---

## Tomorrow's Preview

You now know how to shrink the KV cache at the model level. But there's still the systems-level problem: how do you allocate and manage cache memory across hundreds of concurrent requests that arrive and leave at different times? **Day 59: vLLM & PagedAttention** solves this with a brilliant idea borrowed from operating systems — virtual memory for the KV cache.
