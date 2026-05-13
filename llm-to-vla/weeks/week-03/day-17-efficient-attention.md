# Day 17: Efficient Attention — The O(n²) Problem

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"The art of progress is to preserve order amid change and to preserve change amid order." — Alfred North Whitehead*

## Navigation
- **Previous:** [Day 16: Stop & Reflect #1](day-16-stop-reflect-1.md)
- **Next:** [Day 18: KV Cache](day-18-kv-cache.md)
- **Week:** [Week 3 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 17.1 The Quadratic Wall

Standard self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The $QK^T$ product creates an $n \times n$ attention matrix where $n$ is the sequence length.

| Metric | Complexity | At $n = 4096$, $d = 128$ |
|--------|-----------|--------------------------|
| Time | $O(n^2 d)$ | ~2.1B FLOPs per head |
| Memory | $O(n^2 + nd)$ | ~64MB per head (fp32) |

This means:
- **Doubling** sequence length → **4× cost**
- GPT-4's 128K context would need ~65TB of attention matrices across all heads/layers
- Clearly we need better approaches

```
Standard attention memory layout:

Q (n×d)  ×  K^T (d×n)  =  S (n×n)  →  softmax  →  P (n×n)  ×  V (n×d)  =  O (n×d)
  ↑               ↑            ↑                        ↑              ↑
 small          small       MASSIVE                  MASSIVE         small
                            (this is the problem)
```

### 17.2 Sparse Attention Patterns

Instead of attending to ALL positions, attend to a SUBSET. The key insight: most attention weights are near-zero anyway.

**Local/Sliding Window Attention (Longformer):**
```
Token attends to:  [w tokens left] [self] [w tokens right]
                    ←── window ──→

Full attention matrix (8 tokens, w=2):
  1 2 3 4 5 6 7 8
1 ■ ■ ■ · · · · ·     ■ = attends
2 ■ ■ ■ ■ · · · ·     · = masked
3 ■ ■ ■ ■ ■ · · ·
4 · ■ ■ ■ ■ ■ · ·     Complexity: O(n × w × d)
5 · · ■ ■ ■ ■ ■ ·     where w << n
6 · · · ■ ■ ■ ■ ■
7 · · · · ■ ■ ■ ■
8 · · · · · ■ ■ ■
```

**Strided/Dilated Attention:**
```
Stride = 3:
  1 2 3 4 5 6 7 8 9
1 ■ · · ■ · · ■ · ·    Every 3rd token
4 ■ · · ■ · · ■ · ·    Captures long-range
7 ■ · · ■ · · ■ · ·    dependencies
```

**BigBird (Google, 2020):** Combines three patterns:
1. **Random attention** — attend to $r$ random tokens
2. **Window attention** — attend to $w$ local neighbors
3. **Global tokens** — a few tokens attend to ALL positions (like [CLS])

This gives $O(n)$ complexity while provably approximating full attention (it's a universal approximation of sequence-to-sequence functions).

### 17.3 Flash Attention — The Hardware-Aware Revolution

Flash Attention (Dao et al., 2022) is **not** an approximation. It computes **exact** standard attention but reorganizes the computation to minimize GPU memory transfers.

**The key insight: memory hierarchy matters more than FLOPs.**

```
GPU Memory Hierarchy:
┌──────────────────────────┐
│  HBM (High Bandwidth     │  ~80 GB, ~2 TB/s bandwidth
│  Memory / Global)         │  ← Attention matrices stored here
├──────────────────────────┤
│  L2 Cache                 │  ~40 MB, ~4 TB/s
├──────────────────────────┤
│  SRAM (Shared Memory /    │  ~20 MB, ~19 TB/s bandwidth
│  Registers per SM)        │  ← Want computation HERE
└──────────────────────────┘

Standard attention: writes n×n matrix to HBM, reads it back → IO bound!
Flash attention:    never materializes n×n matrix, tiles computation in SRAM
```

**How Flash Attention works (tiling + online softmax):**

1. **Divide** Q, K, V into blocks that fit in SRAM
2. **For each block of Q:** iterate over blocks of K, V
3. **Compute block attention** in SRAM using online softmax (Milakov & Gimelshein)
4. **Write only the output** O back to HBM — never write the $n \times n$ matrix

$$m_{\text{new}} = \max(m_{\text{old}}, \max(\text{block scores}))$$
$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum e^{s_i - m_{\text{new}}}$$

The online softmax trick maintains running statistics so we never need the full attention matrix.

**Results:**
- **2–4× faster** than standard PyTorch attention
- **5–20× less memory** (linear in $n$ instead of quadratic)
- Enables 16K+ context lengths that were previously OOM
- Now the default in PyTorch 2.0+ via `F.scaled_dot_product_attention`

### 17.4 Linear Attention Approximations

Replace softmax with kernel approximations to get truly $O(n)$ attention:

$$\text{Attention}(Q, K, V) = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) (\phi(K)^T \mathbf{1})}$$

By computing $\phi(K)^T V$ first (a $d \times d$ matrix), we avoid the $n \times n$ product.

**Examples:** Random Feature Attention (Performer), cosine similarity (Linear Transformer).

**Trade-off:** These are approximations — quality degrades, especially for tasks requiring precise positional attention. In practice, Flash Attention (exact but fast) has won over linear approximations.

---

## Implementation (60 min)

### 17.5 Sliding Window Attention

```python
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def standard_attention(Q, K, V):
    """Standard O(n²) attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


def sliding_window_attention(Q, K, V, window_size=256):
    """
    Sliding window attention — each token attends to
    window_size tokens on each side.
    """
    batch, n_heads, seq_len, d_k = Q.shape
    scale = d_k ** -0.5
    output = torch.zeros_like(V)

    for i in range(seq_len):
        # Define window boundaries
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)

        # Extract windowed K, V
        q_i = Q[:, :, i:i+1, :]             # (B, H, 1, d)
        k_win = K[:, :, start:end, :]        # (B, H, w, d)
        v_win = V[:, :, start:end, :]        # (B, H, w, d)

        scores = torch.matmul(q_i, k_win.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        output[:, :, i:i+1, :] = torch.matmul(weights, v_win)

    return output


# --- Benchmark ---
def benchmark_attention(fn, Q, K, V, name, **kwargs):
    """Time an attention function over multiple runs."""
    # Warmup
    for _ in range(3):
        _ = fn(Q, K, V, **kwargs)
    torch.cuda.synchronize() if Q.is_cuda else None

    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = fn(Q, K, V, **kwargs)
        torch.cuda.synchronize() if Q.is_cuda else None
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    print(f"{name}: {avg*1000:.2f} ms (seq_len={Q.size(2)})")
    return avg


# Compare on increasing sequence lengths
seq_lengths = [128, 256, 512, 1024, 2048]
standard_times = []
window_times = []

for n in seq_lengths:
    Q = torch.randn(1, 8, n, 64)
    K = torch.randn(1, 8, n, 64)
    V = torch.randn(1, 8, n, 64)

    t1 = benchmark_attention(standard_attention, Q, K, V, f"Standard (n={n})")
    t2 = benchmark_attention(
        sliding_window_attention, Q, K, V,
        f"Window  (n={n})", window_size=64
    )
    standard_times.append(t1)
    window_times.append(t2)
```

### 17.6 Flash Attention via PyTorch SDPA

```python
# PyTorch 2.0+ provides Flash Attention automatically
def flash_attention_benchmark(Q, K, V):
    """Uses Flash Attention under the hood when possible."""
    return F.scaled_dot_product_attention(Q, K, V)


def naive_attention_benchmark(Q, K, V):
    """Force the naive math implementation."""
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.MATH
    ):
        return F.scaled_dot_product_attention(Q, K, V)


# GPU benchmark (requires CUDA)
if torch.cuda.is_available():
    device = "cuda"
    flash_times = []
    naive_times = []

    for n in [512, 1024, 2048, 4096, 8192]:
        Q = torch.randn(1, 32, n, 128, device=device, dtype=torch.float16)
        K = torch.randn(1, 32, n, 128, device=device, dtype=torch.float16)
        V = torch.randn(1, 32, n, 128, device=device, dtype=torch.float16)

        t_flash = benchmark_attention(
            flash_attention_benchmark, Q, K, V, f"Flash  (n={n})"
        )
        t_naive = benchmark_attention(
            naive_attention_benchmark, Q, K, V, f"Naive  (n={n})"
        )
        flash_times.append(t_flash)
        naive_times.append(t_naive)

    print(f"\nSpeedup at n=8192: {naive_times[-1]/flash_times[-1]:.1f}x")
```

---

## Exercise (45 min)

### E17.1 Memory Usage Analysis

Calculate the peak memory for standard attention at these sequence lengths. Assume `float16`, batch=1, 32 heads, $d_k = 128$:

| Seq Length | Attention Matrix Size | Memory (GB) |
|------------|----------------------|-------------|
| 2,048      | ?                    | ?           |
| 8,192      | ?                    | ?           |
| 32,768     | ?                    | ?           |
| 131,072    | ?                    | ?           |

**Formula:** $\text{mem} = \text{batch} \times \text{heads} \times n^2 \times \text{bytes}$

### E17.2 Plot: Standard vs Flash Attention

Using the benchmarks above, create a plot:
- X-axis: sequence length (log scale)
- Y-axis: time in ms (log scale)
- Two lines: standard vs flash
- Add a vertical line where flash becomes >2× faster

```python
# Starter code
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Time comparison
ax1.plot(seq_lengths, standard_times, 'r-o', label='Standard')
ax1.plot(seq_lengths, window_times, 'b-s', label='Sliding Window (w=64)')
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Time (s)')
ax1.set_title('Attention Time Complexity')
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

# Memory comparison (theoretical)
mem_standard = [n**2 * 32 * 2 / 1e9 for n in seq_lengths]  # fp16
mem_window = [n * 128 * 32 * 2 / 1e9 for n in seq_lengths]
ax2.plot(seq_lengths, mem_standard, 'r-o', label='Standard O(n²)')
ax2.plot(seq_lengths, mem_window, 'b-s', label='Window O(nw)')
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Memory (GB)')
ax2.set_title('Attention Memory Usage')
ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('efficient_attention_comparison.png', dpi=150)
plt.show()
```

### E17.3 Critical Thinking

At what sequence length does flash attention become **essential** (not just nice-to-have)? Consider:
1. A100 GPU with 80GB HBM
2. LLaMA-2 with 32 heads, $d_k = 128$, 32 layers
3. Each layer needs its own attention matrix

Calculate the max sequence length before OOM with standard attention, then with flash attention.

---

## Key Takeaways

1. **Standard attention is O(n²)** in both time and memory — this is THE bottleneck for long sequences
2. **Sparse patterns** (sliding window, BigBird) reduce to O(n) but lose some global connectivity
3. **Flash Attention is not an approximation** — it's the same math with smarter memory access patterns
4. **IO-awareness matters more than FLOP count** — Flash Attention does MORE arithmetic but runs faster because it avoids HBM round-trips
5. **In practice, Flash Attention has won** — it's now the default in PyTorch, and linear approximations are rarely used

## Connection to the Thread

You've been building attention from Bahdanau (Day 10) through multi-head (Day 12) to the full transformer (Day 14). Now you've hit the scaling wall: attention doesn't scale to long sequences naively. Flash Attention is the engineering breakthrough that makes modern LLMs with 128K+ context possible. Tomorrow, you'll see another critical optimization: the KV cache that makes autoregressive generation practical.

## Further Reading

- **[Flash Attention Paper](https://arxiv.org/abs/2205.14135)** — Dao et al., 2022
- **[Flash Attention 2](https://arxiv.org/abs/2307.08691)** — Dao, 2023 (further optimizations)
- **[Longformer](https://arxiv.org/abs/2004.05150)** — Beltagy et al., 2020
- **[BigBird](https://arxiv.org/abs/2007.14062)** — Zaheer et al., 2020
- **[Online Softmax](https://arxiv.org/abs/1805.02867)** — Milakov & Gimelshein, 2018
- **[Karpathy on Flash Attention](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — relevant section in his transformer lecture
