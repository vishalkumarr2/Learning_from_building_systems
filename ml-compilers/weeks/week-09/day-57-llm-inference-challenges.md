# Day 57: LLM Inference Challenges

> **Phase IV · Week 9 · Day 57 of 70 · 2.5 hours**

*"Serving one LLM request is easy. Serving a thousand per second without going bankrupt is the hard part."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 56: Mini-Project — Optimization Pipeline](../week-08/day-56-mini-project-optimization-pipeline.md) | [Day 58: KV Cache Optimization](day-58-kv-cache-optimization.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

LLMs broke every assumption traditional inference systems were built on. A ResNet forward pass is **one shot** — fixed input, fixed compute, fixed memory. An LLM generates tokens **one at a time**, each depending on every token before it. This autoregressive loop turns inference from a compute-bound matrix multiply into a **memory-bandwidth-bound serial dependency chain**. The KV cache that stores past attention state can consume more GPU memory than the model weights themselves. Understanding these unique challenges is prerequisite to every optimization in the next two weeks.

---

## 1. Autoregressive Generation — The Fundamental Bottleneck

Unlike image models that produce output in a single forward pass, LLMs generate text token by token:

```
Autoregressive Generation Loop
═══════════════════════════════════════════════════════════════

  Input: "The capital of France is"

  Step 1: Forward(all tokens) ──→ logits ──→ sample ──→ "Paris"
  Step 2: Forward("Paris")    ──→ logits ──→ sample ──→ "."
  Step 3: Forward(".")        ──→ logits ──→ sample ──→ "<eos>"

  ┌─────────────────────────────────────────────────────────┐
  │  Each step MUST wait for the previous token.            │
  │  You cannot parallelize across output tokens.           │
  │  N output tokens = N sequential forward passes.         │
  └─────────────────────────────────────────────────────────┘
```

This creates a serial dependency that fundamentally limits throughput. Generating 512 tokens requires 512 sequential model invocations — no amount of hardware parallelism can break this chain.

---

## 2. Prefill vs Decode — Two Very Different Phases

Every LLM request has two distinct phases with entirely different computational profiles:

```
Request Lifecycle
═══════════════════════════════════════════════════════════════

  ┌─────────────────────┐     ┌─────────────────────────────┐
  │     PREFILL          │     │          DECODE              │
  │                      │     │                              │
  │  Process all input   │     │  Generate tokens one by one  │
  │  tokens in parallel  │────▶│  using cached KV states      │
  │                      │     │                              │
  │  Compute-bound       │     │  Memory-bandwidth-bound      │
  │  High GPU util.      │     │  Low GPU util. (~5-15%)      │
  │  O(n²) attention     │     │  O(n) attention per step     │
  │  One pass            │     │  T passes for T tokens       │
  └─────────────────────┘     └─────────────────────────────┘

  Example: LLaMA-2 7B on A100 (80GB)
  ───────────────────────────────────
  Prefill 2048 tokens:   ~45 ms    (compute-bound, high FLOPS)
  Decode 1 token:        ~12 ms    (memory-bound, reading weights)
  Generate 512 tokens:   ~6.1 s    (512 × 12ms, serial)
```

### Prefill Phase

All input tokens are processed in parallel. The attention computation is a large matrix multiply — $O(n^2 \cdot d)$ where $n$ is sequence length and $d$ is head dimension. This phase looks like traditional inference: compute-bound, high GPU utilization.

### Decode Phase

Each new token requires reading the **entire model weights** from GPU memory but only performs a matrix-vector multiply (batch size = 1 token). The arithmetic intensity drops dramatically:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

$$\text{Prefill: } \frac{2 \cdot n \cdot d^2}{d^2 \cdot 2} = n \quad \text{(high, compute-bound)}$$

$$\text{Decode: } \frac{2 \cdot 1 \cdot d^2}{d^2 \cdot 2} = 1 \quad \text{(low, memory-bound)}$$

With an A100's compute-to-bandwidth ratio of ~312 TFLOPS / 2 TB/s ≈ 156, any operation with arithmetic intensity < 156 is memory-bound. Decode is deeply memory-bound.

---

## 3. KV Cache Memory Scaling

During prefill, each layer computes Key and Value projections for all input tokens. During decode, we must **reuse** these cached K/V tensors to avoid recomputation. This KV cache grows with every token generated:

```
KV Cache Memory Per Request
═══════════════════════════════════════════════════════════════

  Memory = 2 × n_layers × n_heads × seq_len × head_dim × bytes_per_param

  LLaMA-2 7B (FP16):
  ──────────────────
  n_layers  = 32
  n_heads   = 32
  head_dim  = 128
  bytes     = 2 (FP16)

  Per token: 2 × 32 × 32 × 1 × 128 × 2 = 524,288 bytes = 0.5 MB

  For 4096 token sequence:
  2 × 32 × 32 × 4096 × 128 × 2 = 2,147,483,648 bytes ≈ 2 GB

  ┌─────────────────────────────────────────────────────────┐
  │  Model weights (7B × 2 bytes):       14 GB              │
  │  KV cache (1 request, 4K ctx):        2 GB              │
  │  KV cache (32 concurrent requests): 64 GB  ← PROBLEM   │
  │  A100 total memory:                 80 GB               │
  └─────────────────────────────────────────────────────────┘
```

For larger models, the situation worsens:

| Model | Weights (FP16) | KV Cache/request (4K) | Max Concurrent (A100 80GB) |
|-------|-----------|---------------------|--------------------------|
| LLaMA-2 7B | 14 GB | 2 GB | ~33 |
| LLaMA-2 13B | 26 GB | 3.25 GB | ~16 |
| LLaMA-2 70B | 140 GB | 20 GB | Needs multi-GPU |
| GPT-3 175B | 350 GB | 50 GB | Needs 8+ GPUs |

---

## 4. Batch Scheduling Challenges

Traditional inference batches requests trivially — stack inputs into a larger tensor. LLM batching is fundamentally harder:

```
The LLM Batching Problem
═══════════════════════════════════════════════════════════════

  Static Batching (Naïve):
  ────────────────────────
  Request A: [################............]  (16 tokens in, done at step 12)
  Request B: [################################]  (8 tokens in, done at step 30)
  Request C: [####################..........]  (20 tokens in, done at step 20)

  ▲ All requests padded to longest. A and C waste GPU cycles waiting.
  ▲ New requests must wait for entire batch to finish.

  Continuous Batching (Better):
  ─────────────────────────────
  Step 1:  [A₁, B₁, C₁]      ← all active
  Step 12: [  _, B₁₂, C₁₂]   ← A finishes, slot freed
  Step 13: [D₁, B₁₃, C₁₃]   ← D joins immediately!
  Step 20: [D₈, B₂₀,   _]   ← C finishes
  Step 21: [D₉, B₂₁, E₁]   ← E joins

  ▲ No wasted compute. Requests enter/leave independently.
```

The challenge with continuous batching:
- Requests have different **prefill lengths** (new requests need full prefill, existing need single-token decode)
- **KV cache must be managed** per-request (allocate on arrival, free on completion)
- The scheduler must balance **prefill latency** (time-to-first-token) against **decode throughput**

---

## 5. The Economics of LLM Serving

LLM serving cost is dominated by GPU time. Let's compute the cost of serving LLaMA-2 70B:

```python
# Cost analysis for LLM serving
def serving_cost_analysis():
    """Compute cost per million tokens for LLM serving."""
    
    # Hardware assumptions
    a100_cost_per_hour = 3.00        # $/hr (cloud spot pricing)
    gpus_needed = 4                   # 70B needs ~4x A100-80GB
    
    # Performance (measured, LLaMA-2 70B, FP16, batch=1)
    prefill_tokens_per_sec = 2000     # tokens/sec during prefill
    decode_tokens_per_sec = 35        # tokens/sec during decode
    
    # Typical request
    input_tokens = 512
    output_tokens = 256
    
    # Time per request
    prefill_time = input_tokens / prefill_tokens_per_sec  # 0.256 sec
    decode_time = output_tokens / decode_tokens_per_sec    # 7.3 sec
    total_time = prefill_time + decode_time                # 7.56 sec
    
    # Cost per request (no batching)
    cost_per_sec = (a100_cost_per_hour * gpus_needed) / 3600  # $0.0033/sec
    cost_per_request = cost_per_sec * total_time               # $0.025
    
    # Cost per million output tokens
    cost_per_M_tokens = (cost_per_request / output_tokens) * 1e6
    print(f"Prefill time:         {prefill_time:.3f} sec")
    print(f"Decode time:          {decode_time:.3f} sec")
    print(f"Cost per request:     ${cost_per_request:.4f}")
    print(f"Cost per 1M tokens:   ${cost_per_M_tokens:.2f}")
    print(f"Requests/hour:        {3600/total_time:.0f}")
    
    # With batching (batch=32)
    print("\n--- With batch=32 ---")
    batch_size = 32
    # Decode throughput scales ~linearly until memory-bound
    batched_decode_tps = decode_tokens_per_sec * batch_size * 0.7  # 70% efficiency
    effective_time_per_request = output_tokens / (batched_decode_tps / batch_size)
    batched_cost = cost_per_sec * effective_time_per_request
    batched_cost_per_M = (batched_cost / output_tokens) * 1e6
    print(f"Batched cost/req:     ${batched_cost:.4f}")
    print(f"Batched cost/1M tok:  ${batched_cost_per_M:.2f}")
    print(f"Speedup:              {cost_per_M_tokens/batched_cost_per_M:.1f}x cheaper")

serving_cost_analysis()
```

**Key insight**: Batching is the single most important optimization for cost. A batch of 32 can make serving 10-20× cheaper per token by amortizing the weight-reading cost across requests.

---

## 6. Compute vs Memory Bound Analysis — The Roofline Model

The roofline model reveals why decode is so inefficient:

```
Roofline Model for LLM Inference
═══════════════════════════════════════════════════════════════

  Performance (TFLOPS)
  ▲
  │                                    ┌───────────────────
  │                                    │  Peak Compute
  312│- - - - - - - - - - - - - - - - -│  312 TFLOPS (A100)
  │                                  ╱ │
  │                                ╱   │
  │                              ╱     │
  │                            ╱       │
  │                          ╱         │
  │            Memory       ╱          │
  │           Bandwidth   ╱            │
  │             Roof    ╱              │
  │                   ╱                │
  │         ★       ╱                  │
  │       Decode  ╱                    │
  │             ╱     ★ Prefill        │
  │           ╱       (batch=2048)     │
  │         ╱                          │
  └───────┴────────────────────────────▶ Arithmetic Intensity
          1       156                     (FLOPs/Byte)
          ▲        ▲
        Decode   Ridge Point
      (AI ≈ 1)  (compute=memory)

  Decode utilizes only 1/156 = 0.6% of the GPU's compute capability!
```

The arithmetic for a single decode step in a transformer layer:

$$\text{FLOPs per decode step} = 2 \times d_{\text{model}}^2 \times n_{\text{layers}} \times 4$$

$$\text{Bytes read} = d_{\text{model}}^2 \times n_{\text{layers}} \times 4 \times 2 \text{ (FP16)}$$

$$\text{AI} = \frac{2 \times d^2 \times L \times 4}{d^2 \times L \times 4 \times 2} = 1$$

For LLaMA-2 7B ($d=4096, L=32$): FLOPs = $2 \times 4096^2 \times 32 \times 4 \approx 8.6$ GFLOP, but we read 14 GB of weights — giving us only 0.6 FLOP/byte. The GPU sits nearly idle while waiting for memory.

---

## Hands-On Exercise: Profile LLM Inference Phases

```python
import torch
import torch.nn as nn
import time

class SimplifiedTransformerBlock(nn.Module):
    """Minimal transformer block for profiling prefill vs decode."""
    def __init__(self, d_model=4096, n_heads=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_up = nn.Linear(d_model, d_model * 4, bias=False)
        self.ffn_down = nn.Linear(d_model * 4, d_model, bias=False)
    
    def forward(self, x, kv_cache=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k.detach(), v.detach())
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        
        ffn_out = self.ffn_down(torch.relu(self.ffn_up(out)))
        return out + ffn_out, new_cache


def profile_phases(device='cuda'):
    """Profile prefill vs decode to show the fundamental asymmetry."""
    model = SimplifiedTransformerBlock(d_model=4096, n_heads=32).to(device).half()
    
    # Prefill: 512 tokens at once
    x_prefill = torch.randn(1, 512, 4096, device=device, dtype=torch.float16)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        _, kv = model(x_prefill)
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start) / 10 * 1000
    
    # Decode: 1 token at a time
    x_decode = torch.randn(1, 1, 4096, device=device, dtype=torch.float16)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    cache = kv
    for _ in range(100):
        _, cache = model(x_decode, kv_cache=cache)
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) / 100 * 1000
    
    print(f"Prefill 512 tokens:  {prefill_ms:.2f} ms  "
          f"({512/prefill_ms*1000:.0f} tok/s)")
    print(f"Decode  1 token:     {decode_ms:.2f} ms  "
          f"({1/decode_ms*1000:.0f} tok/s)")
    print(f"Decode is {prefill_ms/decode_ms/512:.1%} efficient vs prefill per token")

# Uncomment on GPU:
# profile_phases()
```

### Exercise Tasks

1. **KV cache memory calculator**: Write a function that takes model config (layers, heads, head_dim, dtype) and sequence length, returns total KV cache memory in GB.
2. **Batch size sweeper**: Measure decode throughput at batch sizes 1, 2, 4, 8, 16, 32, 64. Plot tokens/sec vs batch size and identify the memory-bandwidth saturation point.
3. **Roofline plotter**: Calculate arithmetic intensity for prefill (seq_len=512) and decode (seq_len=1) for your GPU. Plot both on a roofline diagram.

---

## Key Takeaways

1. **Autoregressive generation is inherently serial** — each token depends on all previous tokens, creating an unbreakable sequential dependency
2. **Prefill is compute-bound, decode is memory-bound** — they need fundamentally different optimizations
3. **KV cache scales as $O(L \times H \times S \times D)$** — a single LLaMA-70B request at 4K context uses 20 GB just for cache
4. **Decode utilizes <1% of GPU compute** — the GPU waits for memory reads most of the time
5. **Batching is the primary lever** — amortizing weight reads across requests can improve cost by 10-20×
6. **Continuous batching** eliminates head-of-line blocking where finished requests hold up the batch

---

## Further Reading

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023)
- **Pope et al., "Efficiently Scaling Transformer Inference"** (2023) — prefill vs decode analysis
- **NVIDIA, "Mastering LLM Techniques: Inference Optimization"** — roofline analysis for LLMs
- **Aminabadi et al., "DeepSpeed-Inference"** — multi-GPU inference parallelism
- **Dao et al., "FlashAttention-2"** — optimizing the attention computation itself

---

## Tomorrow's Preview

Now that you understand why LLM inference is fundamentally memory-bound and why the KV cache dominates memory, tomorrow we'll attack the cache directly. **Day 58: KV Cache Optimization** covers Multi-Query Attention (MQA), Grouped-Query Attention (GQA), sliding window attention, and cache compression — techniques that can reduce KV cache memory by 4-8× without meaningful quality loss.
