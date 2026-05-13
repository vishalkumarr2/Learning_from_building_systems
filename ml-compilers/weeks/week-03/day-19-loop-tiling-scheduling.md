# Day 19: Loop Tiling & Scheduling

> **Phase II · Week 3 · Day 19 of 70 · 2.5 hours**

*"Tiling is the single most impactful transformation a compiler can apply — it converts an O(N³) memory traffic problem into an O(N³/T) one, where T is the tile size you choose."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 18: The Polyhedral Model](day-18-polyhedral-model.md) | [Day 20: Auto-Tuning & Search Spaces](day-20-auto-tuning-search.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Yesterday we learned the polyhedral model gives us a **mathematical certificate** for loop transformations. Today we put that to work. Loop tiling (also called blocking) is the dominant technique for exploiting the memory hierarchy in dense linear algebra — the backbone of every neural network. A naive matmul on a 4096×4096 matrix touches ~1 TB of DRAM traffic; a well-tiled version cuts that by 100×. TVM, XLA, Triton, and every serious ML compiler expose tiling as a first-class scheduling concept.

---

## 1. Why Tiling Works: The Memory Wall

Modern processors have a multi-level memory hierarchy:

```
┌───────────────────────────────────────────────┐
│              Registers                        │  ~ 1 cycle, ~KB
├───────────────────────────────────────────────┤
│              L1 Cache                         │  ~ 4 cycles, 32-64 KB
├───────────────────────────────────────────────┤
│              L2 Cache                         │  ~ 12 cycles, 256 KB-1 MB
├───────────────────────────────────────────────┤
│              L3 Cache (shared)                │  ~ 40 cycles, 8-64 MB
├───────────────────────────────────────────────┤
│              DRAM                             │  ~ 200 cycles, 16-256 GB
├───────────────────────────────────────────────┤
│              GPU HBM / Device Memory          │  ~ 400 cycles from core
└───────────────────────────────────────────────┘
```

**The key metric:** *arithmetic intensity* — FLOPs per byte transferred.

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes moved from DRAM}}$$

For naive matmul $C_{M \times N} = A_{M \times K} \cdot B_{K \times N}$:
- FLOPs: $2MNK$
- DRAM traffic: $O(MNK)$ (each element of $B$ is re-read $M$ times)
- Arithmetic intensity: $O(1)$ — **memory-bound**

For tiled matmul with tile size $T$:
- DRAM traffic: $O(MNK / T)$
- Arithmetic intensity: $O(T)$ — **compute-bound** when $T$ is large enough

---

## 2. Single-Level Tiling

### The Transformation

Split each loop into an **outer tile loop** and an **inner point loop**:

```
Original:           Tiled (tile size T):
for i in [0, M):    for io in [0, M/T):       ← tile loop
  for j in [0, N):    for jo in [0, N/T):
    for k in [0, K):    for ii in [0, T):      ← point loop
      C[i][j] +=           for ji in [0, T):
        A[i][k]*B[k][j]     for ki in [0, T):
                               C[io*T+ii][jo*T+ji] +=
                                 A[io*T+ii][ko*T+ki] * B[ko*T+ki][jo*T+ji]
```

Wait — we also need to tile `k`. The full 3-level tiling:

```c
// Tiled matmul: tile sizes Ti, Tj, Tk
for (int io = 0; io < M; io += Ti)
  for (int jo = 0; jo < N; jo += Tj)
    for (int ko = 0; ko < K; ko += Tk)        // ← tile loops
      for (int ii = io; ii < min(io+Ti, M); ii++)
        for (int ji = jo; ji < min(jo+Tj, N); ji++)
          for (int ki = ko; ki < min(ko+Tk, K); ki++)  // ← point loops
            C[ii][ji] += A[ii][ki] * B[ki][ji];
```

### Why This Is Legal

From the polyhedral model (Day 18): the matmul has **no loop-carried dependencies** on `i` and `j` (each `C[i][j]` is independent), and the `k` reduction only requires summing in any order. Tiling preserves these dependence relations because it only **reorders** iterations, never removes them.

### Data Reuse Analysis

For one tile of size $T_i \times T_j \times T_k$:

| Array | Footprint | Accesses | Reuse ratio |
|:------|:----------|:---------|:------------|
| $A$ | $T_i \times T_k$ | $T_i \times T_j \times T_k$ | $T_j$ |
| $B$ | $T_k \times T_j$ | $T_i \times T_j \times T_k$ | $T_i$ |
| $C$ | $T_i \times T_j$ | $T_i \times T_j \times T_k$ | $T_k$ |

**Total working set:** $T_i T_k + T_k T_j + T_i T_j$ elements must fit in cache.

---

## 3. Tile Size Selection

Choosing tile sizes is a constrained optimization problem:

### Cache Capacity Constraint

$$T_i \cdot T_k + T_k \cdot T_j + T_i \cdot T_j \leq \frac{C_{\text{L1}}}{\text{sizeof(float)}}$$

For L1 = 32 KB with float32: capacity = 8192 elements.

### Practical Guidelines

```
┌──────────────────────────────────────────────────┐
│            Tile Size Decision Tree                │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Start with cache capacity constraint         │
│     └─ L1: 32KB → ~8K floats                    │
│     └─ 3 arrays → ~2.7K per array               │
│     └─ T ≈ √2700 ≈ 52 → round to 32 or 64      │
│                                                  │
│  2. Align to vector width                        │
│     └─ AVX-512: 16 floats → T multiple of 16    │
│     └─ NEON: 4 floats → T multiple of 4         │
│                                                  │
│  3. Match GPU warp/wavefront                     │
│     └─ NVIDIA: warp=32, tile often 128×128      │
│     └─ AMD: wavefront=64                        │
│                                                  │
│  4. L2 tiling: repeat for L2 capacity            │
│     └─ Outer tiles target L2 (256KB-1MB)        │
│     └─ Inner tiles target L1 (32KB)             │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Example: Optimal Tile Size Calculation

```python
import math

def optimal_tile_size(M, N, K, cache_bytes, elem_size=4):
    """Compute tile size for square tiles fitting in cache.
    
    Working set: 3 * T^2 elements (A-tile + B-tile + C-tile)
    Constraint: 3 * T^2 * elem_size <= cache_bytes
    """
    capacity = cache_bytes // elem_size
    T_max = int(math.sqrt(capacity / 3))
    
    # Round down to power of 2 for alignment
    T = 1
    while T * 2 <= T_max:
        T *= 2
    
    # Verify
    working_set = 3 * T * T * elem_size
    print(f"Tile size: {T}×{T}")
    print(f"Working set: {working_set} bytes ({working_set/1024:.1f} KB)")
    print(f"Cache: {cache_bytes} bytes ({cache_bytes/1024:.1f} KB)")
    print(f"Utilization: {working_set/cache_bytes*100:.1f}%")
    return T

# L1 = 32KB
T_l1 = optimal_tile_size(4096, 4096, 4096, 32 * 1024)
# → Tile size: 32×32, Working set: 12288 bytes (12.0 KB), 37.5% utilization

# L2 = 256KB  
T_l2 = optimal_tile_size(4096, 4096, 4096, 256 * 1024)
# → Tile size: 128×128, Working set: 196608 bytes (192.0 KB), 75.0% utilization
```

---

## 4. Multi-Level Tiling

Real compilers apply tiling **recursively** to target each cache level:

```
┌─────────────────────────────────────────────────┐
│  Level 3: Thread-block tile (128×128)           │  ← Targets L2 / shared mem
│  ┌─────────────────────────────────────────┐    │
│  │  Level 2: Warp tile (32×64)             │    │  ← Targets register file
│  │  ┌─────────────────────────────────┐    │    │
│  │  │  Level 1: Thread tile (8×8)     │    │    │  ← Targets registers
│  │  │  ┌────────────────────────┐     │    │    │
│  │  │  │  Vectorized (1×4 FMA)  │     │    │    │  ← SIMD unit
│  │  │  └────────────────────────┘     │    │    │
│  │  └─────────────────────────────────┘    │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

In TVM, this maps to:

```python
import tvm
from tvm import te

M, N, K = 1024, 1024, 1024

# Define computation
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

s = te.create_schedule(C.op)

# Level 1: Thread-block tiles (target L2)
block_x = 128
block_y = 128

# Level 2: Warp tiles
warp_x = 32
warp_y = 64

# Level 3: Thread tiles
thread_x = 8
thread_y = 8

# Get loop axes
i, j = s[C].op.axis
k_axis = s[C].op.reduce_axis[0]

# Split i: block → warp → thread
i_block, i_inner = s[C].split(i, factor=block_x)
i_warp, i_thread = s[C].split(i_inner, factor=warp_x)
i_t_outer, i_t_inner = s[C].split(i_thread, factor=thread_x)

# Split j: block → warp → thread
j_block, j_inner = s[C].split(j, factor=block_y)
j_warp, j_thread = s[C].split(j_inner, factor=warp_y)
j_t_outer, j_t_inner = s[C].split(j_thread, factor=thread_y)

# Split k for software pipelining
k_outer, k_inner = s[C].split(k_axis, factor=8)

# Reorder: blocks outermost, then warps, then threads, then k
s[C].reorder(i_block, j_block, i_warp, j_warp,
             k_outer, i_t_outer, j_t_outer,
             k_inner, i_t_inner, j_t_inner)
```

---

## 5. Loop Reordering and Parallelism Extraction

Tiling creates new loop levels. **Reordering** them controls parallelism and locality:

```
After tiling matmul (6 loops):
  io, jo, ko, ii, ji, ki

Option A: io, jo, ko, ii, ji, ki  — ko innermost among tiles
  → Each tile computes partial C, accumulates across ko
  → Good for C locality (stays in registers)

Option B: ko, io, jo, ii, ji, ki  — ko outermost
  → Process one K-slice globally, then next
  → Bad: C evicted between K-tiles

Option C: io, jo, ii, ji, ko, ki  — reduction fully inside
  → Each (io,jo) block completes C-tile independently
  → Best for parallelism: no cross-block synchronization
```

### Vectorization

The innermost loop should match SIMD width:

```python
# TVM: vectorize the innermost dimension
s[C].vectorize(j_t_inner)  # Emit SIMD instructions for j-inner

# This transforms:
#   for j_inner in range(8):
#     C[i][j_base + j_inner] += A[i][k] * B[k][j_base + j_inner]
# Into:
#   C[i][j_base:j_base+8] += A[i][k] * B[k][j_base:j_base+8]  # Single SIMD op
```

### Thread Binding (GPU)

```python
# Bind tile loops to GPU grid/block dimensions
s[C].bind(i_block, te.thread_axis("blockIdx.y"))
s[C].bind(j_block, te.thread_axis("blockIdx.x"))
s[C].bind(i_warp,  te.thread_axis("threadIdx.y"))
s[C].bind(j_warp,  te.thread_axis("threadIdx.x"))
```

---

## 6. Scheduling as an Optimization Space

A **schedule** is a mapping from computation to hardware execution order. The same algorithm can have millions of legal schedules:

```
                    Algorithm
                   (what to compute)
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      Schedule A    Schedule B    Schedule C
      (tiled 32)   (tiled 64)   (tiled 128)
      (vectorize j)(vectorize k)(parallel i)
           │            │            │
           ▼            ▼            ▼
        12 ms         8 ms         15 ms
```

This is the **Halide insight**: separate the algorithm from the schedule.

### TVM Schedule Primitives Summary

| Primitive | Effect | Typical Use |
|:----------|:-------|:------------|
| `split(axis, factor)` | Divide loop into outer/inner | Tiling |
| `fuse(a, b)` | Merge two loops into one | Reducing loop overhead |
| `reorder(a, b, c, ...)` | Change loop nesting order | Locality optimization |
| `bind(axis, thread)` | Map loop to GPU thread | Parallelism |
| `vectorize(axis)` | Emit SIMD instructions | Inner loop vectorization |
| `unroll(axis)` | Fully unroll loop | Small inner loops |
| `compute_at(tensor, axis)` | Inline producer at consumer loop | Fusion / locality |
| `compute_inline()` | Fully inline computation | Element-wise ops |
| `parallel(axis)` | OpenMP parallelize | CPU multi-core |
| `cache_read/write` | Stage data in local buffer | Shared memory / registers |

---

## Hands-On Exercises

### Exercise 1: Tiled Matmul in NumPy (20 min)

Implement and benchmark tiled vs. naive matmul to observe the cache effect:

```python
import numpy as np
import time

def naive_matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C

def tiled_matmul(A, B, T=32):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    for io in range(0, M, T):
        for jo in range(0, N, T):
            for ko in range(0, K, T):
                # Tile boundaries
                i_end = min(io + T, M)
                j_end = min(jo + T, N)
                k_end = min(ko + T, K)
                # Compute tile (using NumPy for inner block)
                C[io:i_end, jo:j_end] += (
                    A[io:i_end, ko:k_end] @ B[ko:k_end, jo:j_end]
                )
    return C

# Benchmark (use small N for pure-Python naive)
N = 256
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# Test correctness
C_ref = A @ B
C_tiled = tiled_matmul(A, B, T=32)
print(f"Max error: {np.max(np.abs(C_ref - C_tiled)):.2e}")

# Benchmark tiled with different tile sizes
for T in [8, 16, 32, 64, 128]:
    start = time.perf_counter()
    for _ in range(10):
        tiled_matmul(A, B, T)
    elapsed = (time.perf_counter() - start) / 10
    print(f"T={T:4d}: {elapsed*1000:.1f} ms")
```

### Exercise 2: TVM Schedule Exploration (25 min)

Write a TVM schedule for 2D convolution with tiling:

```python
import tvm
from tvm import te

# 2D Convolution: O[n,c,h,w] = sum over rc,rh,rw of I[n,rc,h+rh,w+rw] * W[c,rc,rh,rw]
N, CI, H, W = 1, 64, 56, 56
CO, KH, KW = 128, 3, 3

data = te.placeholder((N, CI, H, W), name="data")
weight = te.placeholder((CO, CI, KH, KW), name="weight")

rc = te.reduce_axis((0, CI), name="rc")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")

OH, OW = H - KH + 1, W - KW + 1
conv = te.compute(
    (N, CO, OH, OW),
    lambda n, c, h, w: te.sum(
        data[n, rc, h + rh, w + rw] * weight[c, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="conv"
)

s = te.create_schedule(conv.op)

# TODO: Apply these schedule transforms
# 1. Split the 'c' (output channel) axis by factor 32
# 2. Split the 'h' axis by factor 4  
# 3. Split 'rc' (reduction) by factor 16
# 4. Reorder: n, c_outer, h_outer, c_inner, h_inner, rc_outer, rh, rw, rc_inner, w
# 5. Vectorize 'w'
# 6. Parallel 'c_outer'

# Verify: print(tvm.lower(s, [data, weight, conv], simple_mode=True))
```

### Exercise 3: Tile Size vs. Performance Curve (15 min)

Use `perf stat` (Linux) to measure L1 cache misses for different tile sizes:

```bash
# Compile the tiled matmul C program (provided)
gcc -O2 -o tiled_matmul tiled_matmul.c -lm

# Sweep tile sizes and measure cache misses
for T in 4 8 16 32 64 128 256 512; do
    echo "=== Tile size: $T ==="
    perf stat -e L1-dcache-load-misses,LLC-load-misses \
        ./tiled_matmul $T 2>&1 | grep -E "(misses|seconds)"
done

# Expected: L1 misses drop dramatically from T=4 to T=32,
# then increase again past T=128 (tiles exceed L1 capacity)
```

---

## Key Takeaways

1. **Tiling transforms memory-bound computation into compute-bound** by increasing arithmetic intensity from $O(1)$ to $O(T)$
2. **Tile size selection** is constrained by cache capacity — the working set of all arrays must fit in the target cache level
3. **Multi-level tiling** targets each level of the memory hierarchy (registers → L1 → L2 → shared memory)
4. **Loop reordering** after tiling controls whether parallelism, vectorization, or locality is prioritized
5. **TVM schedule primitives** (`split`, `reorder`, `bind`, `vectorize`, `compute_at`) are direct encodings of polyhedral transformations
6. **The schedule space is combinatorial** — even a single matmul has millions of valid schedules, motivating auto-tuning (tomorrow)

---

## Further Reading

- Lam, M., Rothberg, E., Wolf, M. — *The Cache Performance and Optimizations of Blocked Algorithms* (ASPLOS 1991) — the original tiling paper
- Ragan-Kelley, J. et al. — *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation* (PLDI 2013)
- Chen, T. et al. — *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (OSDI 2018)
- Goto, K. & van de Geijn, R. — *Anatomy of High-Performance Matrix Multiplication* (ACM TOMS 2008)
- Intel Optimization Manual, Chapter 3 — *Cache Blocking Techniques*

---

## Tomorrow's Preview

**Day 20** tackles the question tiling leaves unanswered: **how do you choose the best tile sizes, unroll factors, and loop orders automatically?** We'll explore auto-tuning — from brute-force grid search through genetic algorithms to TVM's learned cost models — and see why ML compilers increasingly use ML to optimize ML.
