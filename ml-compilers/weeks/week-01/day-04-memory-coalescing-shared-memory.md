# Day 4: Memory Coalescing & Shared Memory

> **Phase I — GPU Foundations & CUDA** · Week 1 · Day 4 of 70

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 3: CUDA Programming Basics](day-03-cuda-programming-basics.md) | [Day 5: CUDA Profiling & Roofline](day-05-cuda-profiling-roofline.md) | [Week 1](../week-01/) | [Phase I](../../#phase-i) | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

On an A100, global memory bandwidth is 2 TB/s — but only if threads access memory in the right pattern. A naively-coded kernel can achieve under 100 GB/s, leaving 95% of bandwidth on the table. Memory coalescing and shared memory are the two most impactful optimizations you'll ever learn for GPU code. Compilers like Triton auto-tile into shared memory — understanding the underlying mechanics lets you debug and improve what they generate.

---

## 1. Global Memory Transaction Mechanics

### 1.1 Cache Lines and Sectors

When a warp (32 threads) executes a load instruction, the hardware coalesces their addresses into **memory transactions**:

```
L1 cache line: 128 bytes  (L1 is per-SM)
L2 sector:      32 bytes  (L2 is shared across SMs)

One 128-byte L1 line = 4 × 32-byte L2 sectors
```

For `float` (4 bytes), a perfectly coalesced warp load touches:
$$32 \text{ threads} \times 4 \text{ bytes} = 128 \text{ bytes} = 1 \text{ cache line}$$

### 1.2 Coalesced vs Uncoalesced Access

```
COALESCED — threads access consecutive addresses:

Thread:   t0    t1    t2    t3   ...  t31
Address:  [0]   [1]   [2]   [3]  ...  [31]
          └──────────────────────────────┘
                  1 transaction (128 B)

STRIDED — threads skip elements (stride = 2):

Thread:   t0    t1    t2    t3   ...  t31
Address:  [0]   [2]   [4]   [6]  ...  [62]
          └──────────────────────────────────────┘
                  2 transactions (256 B loaded, 128 B used)
                  → 50% bandwidth utilization

RANDOM — scattered addresses:

Thread:   t0     t1     t2     t3    ...  t31
Address:  [174]  [3]    [891]  [42]  ...  [507]
          └─ up to 32 transactions (1024 B loaded, 128 B used) ─┘
                  → 12.5% bandwidth utilization
```

### 1.3 The Cost Formula

Effective bandwidth utilization:

$$\eta = \frac{\text{bytes actually needed}}{\text{bytes actually transferred}} = \frac{128}{T \times 32}$$

where $T$ is the number of 32-byte L2 sectors touched by the warp.

---

## 2. AoS vs SoA — The Layout That Matters

### 2.1 Array of Structures (AoS)

```c
struct Particle { float x, y, z, w; };  // 16 bytes
Particle particles[N];  // AoS layout
```

Memory layout:
```
[x0 y0 z0 w0 | x1 y1 z1 w1 | x2 y2 z2 w2 | ...]
```

When 32 threads each read `.x`:
```
Thread 0 reads particles[0].x   → byte offset 0
Thread 1 reads particles[1].x   → byte offset 16
Thread 2 reads particles[2].x   → byte offset 32
...
Stride = 16 bytes → 4 transactions instead of 1
Bandwidth utilization: 25%
```

### 2.2 Structure of Arrays (SoA)

```c
struct Particles {
    float *x, *y, *z, *w;
};
```

Memory layout:
```
x: [x0 x1 x2 x3 ... xN]
y: [y0 y1 y2 y3 ... yN]
z: [z0 z1 z2 z3 ... zN]
w: [w0 w1 w2 w3 ... wN]
```

When 32 threads each read `x[tid]`:
```
Thread 0 reads x[0]   → byte offset 0
Thread 1 reads x[1]   → byte offset 4
Thread 2 reads x[2]   → byte offset 8
...
Stride = 4 bytes → 1 transaction. PERFECT coalescing.
```

### 2.3 Performance Comparison

```
Kernel: update position += velocity * dt

AoS:  Effective BW =  410 GB/s  (A100 peak: 2039 GB/s)
SoA:  Effective BW = 1840 GB/s

Speedup from layout change alone: 4.5×
```

---

## 3. Shared Memory

### 3.1 What Is Shared Memory?

Shared memory is a **programmer-managed L1 cache** — fast SRAM on each SM, shared among all threads in a block.

```
┌─────────────── SM ───────────────┐
│                                  │
│  ┌─────────┐   ┌──────────────┐  │
│  │ Warp 0  │   │   Shared     │  │
│  │ Warp 1  │◄─►│   Memory     │  │
│  │ Warp 2  │   │  (up to 164  │  │
│  │  ...    │   │    KB/SM)    │  │
│  └─────────┘   └──────────────┘  │
│         │              │         │
│         └───── L1 ─────┘         │
│                │                 │
└────────────────┼─────────────────┘
                 │
           L2 Cache (global)
                 │
           HBM (global memory)
```

| Property | Global Memory | Shared Memory |
|----------|--------------|---------------|
| Latency | ~400 cycles | ~20 cycles |
| Bandwidth (per SM) | ~100 GB/s | ~4 TB/s |
| Size | 40–80 GB | 48–164 KB/SM |
| Scope | All threads | Block only |

### 3.2 Declaration & Usage

```cuda
// Static allocation (compile-time size)
__shared__ float tile[32][32];

// Dynamic allocation (runtime size)
extern __shared__ float sdata[];
// Launch: kernel<<<grid, block, shared_bytes>>>(...)
```

### 3.3 The Tiling Pattern

The fundamental shared memory pattern:

```
1. Load a tile from global memory into shared memory (coalesced!)
2. __syncthreads()  ← ensure all threads finished loading
3. Compute using shared memory (fast, repeated access OK)
4. __syncthreads()  ← ensure all threads finished computing
5. Write results back to global memory (coalesced!)
```

---

## 4. Bank Conflicts

### 4.1 Shared Memory Banks

Shared memory is divided into **32 banks** (one per warp lane). Successive 4-byte words map to successive banks:

```
Address:  0    4    8   12   ...  124  128  132
Bank:     0    1    2    3   ...   31    0    1
```

- **No conflict**: each thread accesses a different bank → 1 cycle
- **N-way conflict**: N threads hit the same bank → serialized to N cycles
- **Broadcast**: all threads read the *same* address → 1 cycle (special case)

### 4.2 Common Conflict Pattern — Column Access

```cuda
__shared__ float tile[32][32];

// Row access: tile[threadIdx.x][0] → stride 32 → bank 0 always!
// ALL 32 threads hit bank 0 → 32-way conflict!
```

### 4.3 Fix: Padding

```cuda
__shared__ float tile[32][32 + 1];  // +1 padding column

// Now stride between rows = 33 floats
// Thread k accesses bank (k * 33) % 32 = k * 1 % 32 → all different!
// 0-way conflict. Free speedup.
```

Bank conflict effect on a matrix transpose kernel:

```
Without padding:  380 GB/s  (32-way bank conflicts)
With padding:    1590 GB/s  (zero bank conflicts)

Speedup: 4.2×
```

---

## 5. Complete Example: Matrix Transpose

### 5.1 Naive Transpose (Uncoalesced Writes)

```cuda
__global__ void transpose_naive(float *out, const float *in,
                                int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        out[col * height + row] = in[row * width + col];
        //  ^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^
        //  strided writes!        coalesced reads
    }
}
```

### 5.2 Shared Memory Transpose (Coalesced Everything)

```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_smem(float *out, const float *in,
                               int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 for padding

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile — coalesced reads from 'in'
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    __syncthreads();

    // Write transposed tile — coalesced writes to 'out'
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // swapped block indices
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
            //                               ^^^^^^^^^^^ transposed indexing
        }
    }
}
```

### 5.3 Performance Summary (4096 × 4096, A100)

```
Version                  | Effective BW | % of Peak
─────────────────────────┼──────────────┼──────────
Naive (uncoalesced)      |   380 GB/s   |    19%
Shared memory (no pad)   |   920 GB/s   |    45%
Shared memory (+1 pad)   |  1590 GB/s   |    78%
cuBLAS geam              |  1710 GB/s   |    84%
```

---

## 6. Synchronization Deep Dive

### 6.1 `__syncthreads()`

Barrier for **all threads in a block**. Every thread must reach it before any thread proceeds.

```cuda
// WRONG — divergent syncthreads → undefined behavior
if (threadIdx.x < 16) {
    __syncthreads();  // ← only half the block reaches this!
}

// CORRECT — all threads reach the same barrier
__syncthreads();
if (threadIdx.x < 16) {
    // safe to use shared data written by all threads
}
```

### 6.2 `__syncwarp(mask)`

Lightweight sync within a warp (32 threads). Useful for warp-level primitives:

```cuda
// Warp-level reduction (no shared memory needed)
float val = input[tid];
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// Thread 0 of each warp now has the partial sum
```

---

## Hands-On Exercises

1. **Coalescing experiment**: Write a kernel that reads a float array with stride 1, 2, 4, 8, 16, 32. Measure effective bandwidth for each. Plot the curve.

2. **AoS → SoA conversion**: Convert a particle simulation kernel from AoS to SoA layout. Measure the bandwidth improvement with `nsight compute`.

3. **Bank conflict detector**: Write a shared-memory kernel that deliberately creates 2-way, 4-way, 8-way, and 32-way bank conflicts. Use `nsight compute` to verify the `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` metric.

4. **Optimized transpose**: Implement the tiled transpose from Section 5.2. Try tile sizes 16×16 and 32×32. Compare against `cublasSgeam`.

---

## Key Takeaways

1. A warp's memory access pattern determines how many **128-byte cache line transactions** are needed — aim for exactly one.
2. **SoA layout** is almost always better than AoS for GPU kernels — it directly enables coalescing.
3. Shared memory is ~20× faster than global memory and acts as a **programmer-managed cache**.
4. **Padding** (`float tile[32][33]`) eliminates bank conflicts for free.
5. The tiling pattern — load coalesced → sync → compute from shared → sync → store coalesced — is the single most important GPU optimization pattern.
6. `__syncthreads()` must be reached by **all** threads in a block — never place it inside divergent branches.

---

## Further Reading

- [CUDA C++ Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA C++ Best Practices — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- Volkov & Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra" (SC'08)
- Ruetsch & Micikevicius, "Optimizing Matrix Transpose in CUDA" — [NVIDIA Technical Report](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

---

> **Tomorrow**: [Day 5 — CUDA Profiling & Roofline Model](day-05-cuda-profiling-roofline.md). You'll learn to use `nsight compute` and `nsight systems` to measure exactly where your kernels sit on the roofline — and what to do about it.
