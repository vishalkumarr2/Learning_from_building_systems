# Day 24: Triton Flash Attention

> **Phase II · Week 4 · Day 24 of 70 · 2.5 hours**

*"Flash Attention's key insight isn't algorithmic cleverness — it's respecting the memory hierarchy. By never materializing the full N×N attention matrix, it turns a memory-bound problem into a compute-bound one."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 23: Triton Matrix Multiplication](day-23-triton-matmul.md) | [Day 25: torch.compile Internals](day-25-torch-compile-internals.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Standard attention computes the full $N \times N$ score matrix, writes it to HBM, reads it back for softmax, writes it again, then reads it for the final matmul with $V$. For sequence length $N = 8192$ and batch $\times$ heads $= 96$, that's **24 GB** of intermediate storage — just for one attention layer. Flash Attention eliminates this entirely by fusing Q·K^T, softmax, and the multiplication by V into a single kernel that tiles over the sequence dimension. The result: **2–4× wall-clock speedup** and memory reduction from $O(N^2)$ to $O(N)$. This is the most impactful GPU kernel of the transformer era.

---

## 1. Standard Attention: The Memory Problem

### The Computation

Given $Q, K, V \in \mathbb{R}^{N \times d}$ (for one head):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

### Standard Implementation

```python
# Standard attention — PyTorch
def standard_attention(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)     # (N, N) — materialized in HBM!
    P = torch.softmax(S, dim=-1)   # (N, N) — another HBM round-trip
    O = P @ V                      # (N, d)
    return O
```

### Memory Access Pattern

```
Standard Attention — 4 HBM round trips:

HBM                          SRAM (on-chip)
┌────────────┐               ┌──────────┐
│ Q (N×d)    │──read──────►  │          │
│ K (N×d)    │──read──────►  │ compute  │
│            │               │  QK^T    │
│ S (N×N)    │◄──write─────  │          │  ① Write S to HBM
│            │               └──────────┘
│ S (N×N)    │──read──────►  ┌──────────┐
│            │               │ softmax  │
│ P (N×N)    │◄──write─────  │          │  ② Write P to HBM
│            │               └──────────┘
│ P (N×N)    │──read──────►  ┌──────────┐
│ V (N×d)    │──read──────►  │  P @ V   │
│ O (N×d)    │◄──write─────  │          │  ③ Write O to HBM
└────────────┘               └──────────┘

HBM traffic: O(N²d + N²) — dominated by N×N matrices
Memory: O(N²) for S and P
```

For $N = 4096, d = 64$: $S$ and $P$ are $4096^2 \times 4 = 64$ MB each — per head.

---

## 2. The Flash Attention Algorithm

### Core Ideas

1. **Tile the outer loop** — process $Q$ in blocks of $B_r$ rows
2. **Tile the inner loop** — process $K, V$ in blocks of $B_c$ rows
3. **Online softmax** — compute softmax incrementally without materializing the full row
4. **Never write $S$ or $P$ to HBM** — keep them in SRAM

### Online Softmax (the Mathematical Trick)

Standard softmax requires two passes: one for $\max$, one for $\sum e^{x_i - \max}$. Flash Attention uses the **online softmax** algorithm that updates the running maximum and sum as new blocks arrive:

Given partial results from blocks $1 \ldots j-1$ and a new block $j$:

$$m_j = \max(m_{j-1}, \max(\mathbf{s}_j))$$

$$\ell_j = e^{m_{j-1} - m_j} \cdot \ell_{j-1} + \sum_i e^{s_{j,i} - m_j}$$

$$O_j = \frac{e^{m_{j-1} - m_j} \cdot \ell_{j-1} \cdot O_{j-1} + e^{\mathbf{s}_j - m_j} \cdot V_j}{\ell_j}$$

When the last block is processed, $O$ contains the exact attention output — no approximation.

### Algorithm Pseudocode

```
FLASH-ATTENTION-FORWARD(Q, K, V):
  Input: Q, K, V ∈ ℝ^{N×d}, block sizes Br, Bc
  Output: O ∈ ℝ^{N×d}
  
  for i = 0 to ⌈N/Br⌉ - 1:          # Outer loop: Q blocks
      Qi = Q[i·Br : (i+1)·Br]        # Load Q block (Br × d) from HBM
      Oi = zeros(Br, d)              # Initialize output block
      mi = -∞ · ones(Br)             # Running row-wise max
      li = zeros(Br)                 # Running row-wise sum
      
      for j = 0 to ⌈N/Bc⌉ - 1:      # Inner loop: K,V blocks
          Kj = K[j·Bc : (j+1)·Bc]    # Load K block (Bc × d) from HBM
          Vj = V[j·Bc : (j+1)·Bc]    # Load V block (Bc × d) from HBM
          
          Sij = Qi @ Kj^T / √d       # Compute scores (Br × Bc) in SRAM
          
          # Online softmax update
          mi_new = max(mi, rowmax(Sij))
          Pij = exp(Sij - mi_new)     # Unnormalized attention (Br × Bc)
          li_new = exp(mi - mi_new) * li + rowsum(Pij)
          
          # Rescale previous output and accumulate
          Oi = (exp(mi - mi_new) * li / li_new)[:,None] * Oi
              + (1 / li_new)[:,None] * (Pij @ Vj)
          
          mi = mi_new
          li = li_new
      
      O[i·Br : (i+1)·Br] = Oi        # Write output block to HBM
  return O
```

### Memory Access Pattern

```
Flash Attention — 1 HBM round trip per Q block:

HBM                          SRAM (on-chip, ~20 MB on A100)
┌────────────┐               ┌──────────────────────────┐
│ Q block    │──read──────►  │ Q_i (Br × d)             │
│ K block j  │──read──────►  │ K_j (Bc × d)             │
│ V block j  │──read──────►  │ V_j (Bc × d)             │
│            │               │                          │
│            │  (no write!)  │ S_ij = Q_i @ K_j^T  SRAM │
│            │  (no write!)  │ P_ij = softmax(S_ij) SRAM│
│            │               │ O_i += P_ij @ V_j   SRAM │
│            │               │ m_i, l_i (running stats) │
│            │               └──────────────────────────┘
│ O block    │◄──write─────  (only final output)
└────────────┘

HBM traffic: O(N²d² / M) where M = SRAM size
Memory: O(N) — only Q, K, V, O stored (no N×N matrices)
```

---

## 3. Triton Implementation: Forward Pass

```python
import triton
import triton.language as tl
import torch
import math

@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Scaling factor
    sm_scale,
    # Sequence length
    N,
    # Head dimension
    d: tl.constexpr,
    # Strides
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    # Block sizes
    BLOCK_M: tl.constexpr,    # Q block size (Br)
    BLOCK_N: tl.constexpr,    # K/V block size (Bc)
    IS_CAUSAL: tl.constexpr,
):
    # Which Q block does this program handle?
    pid_m = tl.program_id(0)
    
    # Q block row range
    q_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_range < N
    
    # Load Q block: (BLOCK_M, d)
    q_ptrs = Q_ptr + q_range[:, None] * stride_qn + tl.arange(0, d)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), value=-1e9, dtype=tl.float32)   # row-wise max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)               # row-wise sum
    o_i = tl.zeros((BLOCK_M, d), dtype=tl.float32)             # output accumulator
    
    # Determine K/V iteration range
    if IS_CAUSAL:
        kv_bound = min((pid_m + 1) * BLOCK_M, N)
    else:
        kv_bound = N
    
    # Inner loop: iterate over K/V blocks
    for start_n in range(0, kv_bound, BLOCK_N):
        kv_range = start_n + tl.arange(0, BLOCK_N)
        kv_mask = kv_range < N
        
        # Load K block: (BLOCK_N, d)
        k_ptrs = K_ptr + kv_range[:, None] * stride_kn + tl.arange(0, d)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        
        # Compute QK^T: (BLOCK_M, BLOCK_N)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = q_range[:, None] >= kv_range[None, :]
            s = tl.where(causal_mask, s, -1e9)
        
        # --- Online softmax ---
        # New row-wise max
        m_ij = tl.max(s, axis=1)                      # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_ij)                  # (BLOCK_M,)
        
        # Correction factor for previous accumulations
        alpha = tl.exp(m_i - m_new)                    # (BLOCK_M,)
        
        # Exponentiate current block scores
        p = tl.exp(s - m_new[:, None])                 # (BLOCK_M, BLOCK_N)
        
        # Update running sum
        l_new = alpha * l_i + tl.sum(p, axis=1)        # (BLOCK_M,)
        
        # Load V block: (BLOCK_N, d)
        v_ptrs = V_ptr + kv_range[:, None] * stride_vn + tl.arange(0, d)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        
        # Rescale previous output and accumulate
        o_i = alpha[:, None] * o_i + tl.dot(p.to(v.dtype), v)
        
        # Update state
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    o_i = o_i / l_i[:, None]
    
    # Store output block
    o_ptrs = O_ptr + q_range[:, None] * stride_on + tl.arange(0, d)[None, :] * stride_od
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])
```

---

## 4. The Launcher

```python
def flash_attention_triton(
    Q: torch.Tensor,    # (N, d) or (B, H, N, d)
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Flash Attention forward pass using Triton."""
    # Handle batched input — flatten batch and heads
    orig_shape = Q.shape
    if Q.ndim == 4:
        B, H, N, d = Q.shape
        Q = Q.reshape(B * H, N, d)
        K = K.reshape(B * H, N, d)
        V = V.reshape(B * H, N, d)
    else:
        B_H, N, d = Q.shape
    
    O = torch.empty_like(Q)
    sm_scale = 1.0 / math.sqrt(d)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    # One program per Q block, per batch*head
    grid = (triton.cdiv(N, BLOCK_M), Q.shape[0])
    
    # Launch for each batch*head element
    for bh in range(Q.shape[0]):
        flash_attention_fwd_kernel[(triton.cdiv(N, BLOCK_M),)](
            Q[bh], K[bh], V[bh], O[bh],
            sm_scale, N, d,
            Q.stride(-2), Q.stride(-1),
            K.stride(-2), K.stride(-1),
            V.stride(-2), V.stride(-1),
            O.stride(-2), O.stride(-1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal,
        )
    
    return O.reshape(orig_shape)

# Correctness check
N, d = 1024, 64
Q = torch.randn(N, d, device='cuda', dtype=torch.float16)
K = torch.randn(N, d, device='cuda', dtype=torch.float16)
V = torch.randn(N, d, device='cuda', dtype=torch.float16)

out_flash = flash_attention_triton(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))
out_ref = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0).unsqueeze(0), K.unsqueeze(0).unsqueeze(0), V.unsqueeze(0).unsqueeze(0)
)
print(f"Max error: {(out_flash.squeeze() - out_ref.squeeze()).abs().max():.6f}")
```

---

## 5. Complexity Analysis

### Memory Complexity

| Algorithm | HBM Memory | HBM I/O |
|:--|:--|:--|
| Standard attention | $O(N^2)$ for $S, P$ | $O(N^2 d + N^2)$ |
| Flash Attention | $O(N)$ for $Q, K, V, O$ only | $O(N^2 d^2 / M)$ |

Where $M$ is the SRAM size. Since $d \ll N$ and $M$ is large enough to hold several blocks, Flash Attention is **IO-optimal** — it achieves the minimum possible HBM accesses for the attention computation.

### Compute Complexity

Both algorithms perform the same FLOPs: $O(N^2 d)$. Flash Attention does more FLOP per byte — it's compute-bound on modern GPUs, which is exactly where we want to be.

### Wall-Clock Speedup (A100, FP16)

```
Sequence    Standard      Flash         Speedup
Length      Attention     Attention
──────────────────────────────────────────────
  512       0.12 ms       0.08 ms       1.5×
 1024       0.41 ms       0.18 ms       2.3×
 2048       1.52 ms       0.52 ms       2.9×
 4096       5.89 ms       1.68 ms       3.5×
 8192       23.1 ms       5.92 ms       3.9×
16384         OOM         21.3 ms        ∞
──────────────────────────────────────────────
```

---

## 6. Causal Masking

For autoregressive models (GPT, LLaMA), position $i$ can only attend to positions $\leq i$:

```
Causal attention mask (N=8):
  K₀ K₁ K₂ K₃ K₄ K₅ K₆ K₇
Q₀ [✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗]
Q₁ [✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗]
Q₂ [✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗]
Q₃ [✓  ✓  ✓  ✓  ✗  ✗  ✗  ✗]
Q₄ [✓  ✓  ✓  ✓  ✓  ✗  ✗  ✗]
Q₅ [✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗]
Q₆ [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗]
Q₇ [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓]
```

In Flash Attention, causal masking provides a **free speedup**: for Q block $i$, we only iterate over K/V blocks $j \leq i$. This skips ~50% of the computation:

```
Tiled causal structure (BLOCK_M = BLOCK_N = 2):

  K₀₁  K₂₃  K₄₅  K₆₇
Q₀₁ [COMP  skip  skip  skip]   ← only 1 K/V block
Q₂₃ [COMP  COMP  skip  skip]   ← 2 K/V blocks
Q₄₅ [COMP  COMP  COMP  skip]   ← 3 K/V blocks
Q₆₇ [COMP  COMP  COMP  COMP]   ← 4 K/V blocks

COMP = computed, skip = skipped entirely
Within boundary blocks: element-wise causal_mask applied
```

---

## Hands-On Exercises

### Exercise 1: Verify Online Softmax (20 min)

Implement the online softmax algorithm in pure PyTorch (no Triton) and verify it matches `torch.softmax`:

```python
def online_softmax(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Compute softmax using online algorithm, processing block_size elements at a time."""
    N = x.shape[-1]
    m = torch.full((x.shape[0],), -1e9, device=x.device)
    l = torch.zeros(x.shape[0], device=x.device)
    
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        x_block = x[:, start:end]
        
        m_block = x_block.max(dim=-1).values
        m_new = torch.maximum(m, m_block)
        
        # TODO: update l, compute final softmax
        # Hint: track exp(x_i - m_new) contributions
    
    # TODO: reconstruct full softmax output
```

### Exercise 2: Benchmark Flash vs Standard (30 min)

Compare your Triton Flash Attention against `torch.nn.functional.scaled_dot_product_attention` and a naive PyTorch implementation for $N \in [256, 512, 1024, 2048, 4096]$. Plot throughput (GFLOPS) and peak memory.

### Exercise 3: Add Causal Support (30 min)

If you haven't already, add causal masking to the kernel. Verify that:
1. Causal output matches `scaled_dot_product_attention(..., is_causal=True)`
2. Causal is faster than non-causal (should skip ~50% of inner loop iterations)
3. Future positions have zero influence on the output

---

## Key Takeaways

1. **Memory is the bottleneck, not compute** — standard attention is $O(N^2)$ in memory; Flash Attention is $O(N)$.
2. **Online softmax is exact** — the running max/sum trick produces identical results to two-pass softmax, with no approximation.
3. **Tiling + fusion = IO-optimality** — by fusing QK^T, softmax, and PV into one kernel, Flash Attention achieves the theoretical minimum HBM accesses.
4. **Causal masking is free** — skipping future K/V blocks eliminates ~50% of compute with zero overhead.
5. **This pattern generalizes** — the "tile, accumulate, normalize at end" pattern appears in many fused kernels (layer norm, cross-entropy, etc.).

---

## Further Reading

- **FlashAttention paper:** Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022)
- **FlashAttention-2:** Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)
- **FlashAttention-3:** Shah et al. (2024) — exploits WGMMA and TMA on Hopper
- **Online softmax:** Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018)
- **Triton Flash Attention tutorial:** [triton-lang.org/main/getting-started/tutorials/06-fused-attention.html](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

---

## Tomorrow

**Day 25** steps back from kernel writing to understand the **torch.compile** pipeline that *generates* kernels automatically. We'll trace how `@torch.compile` captures Python bytecode (Dynamo), splits forward and backward graphs (AOTAutograd), and generates Triton kernels (Inductor). Understanding this pipeline connects the Triton skills you've built this week to the compiler infrastructure that ships with PyTorch.
