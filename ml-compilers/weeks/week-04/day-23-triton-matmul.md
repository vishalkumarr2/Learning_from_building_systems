# Day 23: Triton Matrix Multiplication

> **Phase II · Week 4 · Day 23 of 70 · 2.5 hours**

*"Matrix multiplication is the hydrogen atom of GPU programming — simple enough to understand completely, yet rich enough to expose every performance-critical concept: tiling, memory hierarchy, pipelining, and autotuning."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 22: Triton Language Basics](day-22-triton-language-basics.md) | [Day 24: Triton Flash Attention](day-24-triton-flash-attention.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Every transformer forward pass is dominated by GEMMs — `QK^T`, `\text{softmax} \cdot V`, and the two linear projections in each FFN block. A single GPT-3 inference on a 2048-token sequence performs ~300 billion multiply-accumulate operations. The matmul kernel is where Triton's block-based model delivers the most leverage: what takes 200+ lines of CUDA with explicit shared memory staging becomes ~50 lines of Triton with comparable performance. Understanding this kernel deeply is the foundation for everything we'll build this week.

---

## 1. Naive Matmul and Why It's Slow

For $C_{M \times N} = A_{M \times K} \cdot B_{K \times N}$, the naive approach assigns one thread per output element:

```python
# Pseudo-CUDA: one thread per C[i,j]
for k in range(K):
    C[i, j] += A[i, k] * B[k, j]
```

### Performance Analysis

| Metric | Value |
|:--|:--|
| FLOPs | $2MNK$ |
| DRAM reads of $A$ | $MK \cdot N$ (each row read $N$ times) |
| DRAM reads of $B$ | $KN \cdot M$ (each column read $M$ times) |
| Arithmetic intensity | $\frac{2MNK}{4(MKN + KNM)} = \frac{1}{4}$ FLOP/byte |
| GPU peak | ~300 TFLOPS (A100 FP16) |
| DRAM bandwidth | ~2 TB/s (A100 HBM2e) |
| Achievable | $2 \times 10^{12} \times 0.25 = 500$ GFLOPS (0.17% peak) |

**The problem:** arithmetic intensity of $O(1)$ — completely memory-bound.

---

## 2. Tiled GEMM: The Key Insight

We partition $C$ into tiles of size $\text{BLOCK\_M} \times \text{BLOCK\_N}$ and iterate over $K$ in chunks of $\text{BLOCK\_K}$:

```
Matrix C (M×N) tiled:
┌──────┬──────┬──────┬───┐
│ C₀₀  │ C₀₁  │ C₀₂  │...│  Each tile: BLOCK_M × BLOCK_N
├──────┼──────┼──────┼───┤
│ C₁₀  │ C₁₁  │ C₁₂  │...│
├──────┼──────┼──────┼───┤
│ C₂₀  │ C₂₁  │ C₂₂  │...│
└──────┴──────┴──────┴───┘

For tile C[i,j]:
  acc = zeros(BLOCK_M, BLOCK_N)
  for k_block in range(0, K, BLOCK_K):
      a_tile = A[i*BM:(i+1)*BM, k_block:k_block+BK]   # BLOCK_M × BLOCK_K
      b_tile = B[k_block:k_block+BK, j*BN:(j+1)*BN]   # BLOCK_K × BLOCK_N
      acc += a_tile @ b_tile                             # local GEMM
  C[i*BM:(i+1)*BM, j*BN:(j+1)*BN] = acc
```

### Arithmetic Intensity After Tiling

Each $A$ tile ($\text{BM} \times \text{BK}$ floats) is reused across $\lceil N/\text{BN} \rceil$ column tiles. Each $B$ tile is reused across $\lceil M/\text{BM} \rceil$ row tiles.

$$\text{AI}_{\text{tiled}} = \frac{2 \cdot \text{BM} \cdot \text{BN} \cdot \text{BK}}{4 \cdot (\text{BM} \cdot \text{BK} + \text{BK} \cdot \text{BN})} = \frac{\text{BM} \cdot \text{BN}}{2(\text{BM} + \text{BN})}$$

For $\text{BM} = \text{BN} = 128$: AI = $128 \cdot 128 / (2 \cdot 256) = 32$ FLOP/byte — **compute-bound!**

---

## 3. The Triton Matmul Kernel

```python
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (elements, not bytes)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one BLOCK_M × BLOCK_N tile of C = A @ B."""
    # ── Step 1: Identify which tile this program computes ──
    pid_m = tl.program_id(0)  # tile row index
    pid_n = tl.program_id(1)  # tile column index
    
    # ── Step 2: Compute base offsets ──
    # Row indices for the A tile and C tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # shape: (BLOCK_M,)
    # Column indices for the B tile and C tile
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape: (BLOCK_N,)
    # K-dimension indices (first block)
    rk = tl.arange(0, BLOCK_K)                     # shape: (BLOCK_K,)
    
    # ── Step 3: Initialize pointers to first tiles ──
    # A tile: rows rm, columns rk → shape (BLOCK_M, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    # B tile: rows rk, columns rn → shape (BLOCK_K, BLOCK_N)
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    
    # ── Step 4: Accumulator (in registers) ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # ── Step 5: Main loop over K dimension ──
    for k in range(0, K, BLOCK_K):
        # Boundary masks
        a_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        b_mask = (rk[:, None] + k < K) & (rn[None, :] < N)
        
        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Block-level matrix multiply — accumulate
        acc += tl.dot(a, b)
        
        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # ── Step 6: Store result tile ──
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
```

### Kernel Execution Model

```
Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))

Each program instance:
  ┌─────────────────────────────────────────┐
  │ pid_m=1, pid_n=2                        │
  │                                         │
  │  Load A[BM:2BM, 0:BK]                  │
  │  Load B[0:BK, 2BN:3BN]                 │
  │  acc += A_tile @ B_tile                 │
  │                                         │
  │  Load A[BM:2BM, BK:2BK]                │
  │  Load B[BK:2BK, 2BN:3BN]              │
  │  acc += A_tile @ B_tile                 │
  │      ...                                │
  │  (repeat K/BLOCK_K times)               │
  │                                         │
  │  Store acc → C[BM:2BM, 2BN:3BN]        │
  └─────────────────────────────────────────┘
```

---

## 4. The Launcher and Verification

```python
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton matmul launcher."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Tile sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    
    # 2D grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

# Verify correctness
M, N, K = 512, 512, 512
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c_triton = matmul(a, b)
c_torch = torch.matmul(a, b)
print(f"Max error: {(c_triton - c_torch).abs().max():.6f}")
# Expect < 1e-2 for FP16 (due to accumulation order differences)
```

---

## 5. Autotuning with `@triton.autotune`

Different matrix sizes favor different tile configurations. Triton's autotuner searches the space at runtime:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # Retune when these values change
)
@triton.jit
def matmul_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ... same kernel body as above ...
    pass
```

### What `num_stages` and `num_warps` Control

```
num_stages = software pipelining depth
─────────────────────────────────────
Stage 0: Load tile k=0    ──→ Compute tile k=0   ──→ ...
Stage 1:                      Load tile k=1      ──→ Compute tile k=1
Stage 2:                                             Load tile k=2  ...

Higher num_stages → more memory latency hiding
                  → more register pressure
                  → diminishing returns past 4-5

num_warps = parallelism within one program
─────────────────────────────────────────
1 warp  = 32 threads  → good for small tiles
4 warps = 128 threads → balanced
8 warps = 256 threads → good for large tiles, saturates SM
```

---

## 6. Performance vs cuBLAS

### Benchmark Results (A100, FP16, square matrices)

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['cublas', 'triton'],
        line_names=['cuBLAS', 'Triton'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={'N': 4096, 'K': 4096},
    )
)
def bench_matmul(M, N, K, provider):
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    if provider == 'cublas':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    else:
        ms = triton.testing.do_bench(lambda: matmul(a, b))
    
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops
```

Typical results on A100:

```
Matrix Size     cuBLAS (TFLOPS)   Triton (TFLOPS)   Ratio
─────────────────────────────────────────────────────────
  256 × 4096       45.2              38.1            84%
  512 × 4096       98.7              89.3            90%
 1024 × 4096      178.4             165.2            93%
 2048 × 4096      242.6             231.8            96%
 4096 × 4096      280.1             268.5            96%
─────────────────────────────────────────────────────────
```

Triton achieves **84–96% of cuBLAS** — remarkable for ~50 lines of Python vs thousands of lines of hand-tuned SASS in cuBLAS.

---

## 7. Advanced: Swizzled Program Ordering

For large matrices, L2 cache hit rate matters. Adjacent programs should access nearby memory. The **swizzle** technique reorders program IDs:

```python
# Inside the kernel — replace simple pid calculation:
# pid_m = tl.program_id(0)
# pid_n = tl.program_id(1)

# With grouped ordering for better L2 locality:
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
GROUP_SIZE_M: tl.constexpr = 8

num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

```
Without swizzle (row-major):     With swizzle (grouped):
  0  1  2  3  4  5  6  7         0  2  4  6  1  3  5  7
  8  9 10 11 12 13 14 15         8 10 12 14  9 11 13 15
 16 17 18 19 20 21 22 23        16 18 20 22 17 19 21 23

Programs 0-7 hit same A rows    Programs 0,2,4,6 span fewer B
but ALL B columns → L2 thrash   columns → better L2 reuse
```

---

## Hands-On Exercises

### Exercise 1: Complete the Autotuned Kernel (30 min)

Take the autotuned kernel skeleton from Section 5 and fill in the body. Run the benchmark from Section 6 on your GPU and compare against cuBLAS. Record the best config for 4096×4096.

### Exercise 2: Non-Square Matrices (20 min)

Test your kernel with non-square shapes common in transformers:
- $A: (8, 4096) \times B: (4096, 4096)$ — small batch, large hidden
- $A: (2048, 768) \times B: (768, 3072)$ — BERT FFN expansion

Verify correctness and measure throughput. Do the autotuner configs from square matrices still win?

### Exercise 3: Add L2 Swizzle (30 min)

Implement the grouped program ordering from Section 7. Benchmark with and without swizzle for 4096×4096 matmul. On A100, expect 2–5% improvement from better L2 utilization.

---

## Key Takeaways

1. **Tiling transforms memory-bound into compute-bound** — arithmetic intensity goes from $O(1)$ to $O(\text{BLOCK\_SIZE})$, crossing the roofline from memory- to compute-bound.
2. **`tl.dot` is the workhorse** — Triton compiles block matrix multiplies to Tensor Core `HMMA` instructions on supported hardware.
3. **Autotuning is essential** — the optimal tile size depends on matrix shape, GPU architecture, and occupancy; `@triton.autotune` handles this automatically.
4. **`num_stages` controls pipelining** — overlapping memory loads with computation hides DRAM latency, especially for memory-bound configs.
5. **Program ordering matters** — swizzled tile ordering improves L2 cache hit rates by 20–40% for large matrices.

---

## Further Reading

- **Triton matmul tutorial:** [triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance** — Simon Boehm's blog post
- **CUTLASS:** NVIDIA's C++ template library for GEMM — the cuBLAS building blocks ([github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass))
- **Roofline model:** Williams, Waterman & Patterson (2009) — the theory behind arithmetic intensity analysis

---

## Tomorrow

**Day 24** builds on the matmul foundation to implement the most important kernel in modern AI: **Flash Attention**. We'll implement the tiled softmax algorithm that reduces attention's memory complexity from $O(N^2)$ to $O(N)$, making long-sequence transformers practical. If matmul is the hydrogen atom, Flash Attention is the helium — a slightly harder two-body problem that changed everything.
