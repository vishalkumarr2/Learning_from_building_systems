# Day 6: Matrix Multiply — Naive to Tiled

> **Phase I — GPU Foundations & CUDA** · Week 1 · Day 6 of 70

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 5: CUDA Profiling & Roofline](day-05-cuda-profiling-roofline.md) | [Day 7: Mini-Project — Profile & Optimize a GEMM](day-07-mini-project-gemm.md) | [Week 1](../week-01/) | [Phase I](../../#phase-i) | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

General Matrix Multiply (GEMM) is the computational backbone of deep learning. A single transformer forward pass on GPT-3 executes thousands of GEMMs. cuBLAS SGEMM achieves >90% of peak FLOPS — understanding how it gets there teaches you every GPU optimization technique in one kernel. Today you'll write three versions and watch performance climb from ~2% to ~60% of cuBLAS.

---

## 1. The Math

Matrix multiply $C = A \times B$ where $A$ is $M \times K$, $B$ is $K \times N$, $C$ is $M \times N$:

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

Computational complexity:
- **FLOPs**: $2 \times M \times N \times K$ (one multiply + one add per $k$)
- **Memory**: $(M \times K + K \times N + M \times N) \times 4$ bytes (FP32, read A, B, write C)

Arithmetic intensity for square $N \times N$:

$$\text{AI} = \frac{2N^3}{4 \cdot 3N^2} = \frac{N}{6}$$

At $N = 4096$: AI ≈ 683 → deeply **compute-bound**. We must maximize FLOP throughput.

---

## 2. Version 1: Naive SGEMM

### 2.1 Idea

Each thread computes one element of $C$ by doing a dot product of a row of $A$ with a column of $B$.

```
A (M×K)           B (K×N)           C (M×N)
┌─────────┐      ┌─────────┐      ┌─────────┐
│→→→→→→→→→│      │ ↓       │      │         │
│         │  ×   │ ↓       │  =   │    *    │ ← C[i][j]
│         │      │ ↓       │      │         │
└─────────┘      └─────────┘      └─────────┘
  row i            col j
```

### 2.2 Kernel Code

```cuda
__global__ void sgemm_naive(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

### 2.3 Why It's Slow

Each thread reads $K$ elements from A and $K$ elements from B. Across the entire grid:

- **Total global reads**: $M \times N \times K$ (from A) + $M \times N \times K$ (from B) = $2MNK$
- Each element of A is read $N$ times; each element of B is read $M$ times.
- For $M = N = K = 4096$: that's $2 \times 4096^3 \times 4 = 512$ GB of DRAM traffic!

```
                A[row][k] — same row read by N threads in column
                B[k][col] — same column read by M threads in row

Redundancy factor: O(N) per element
```

**Performance (4096×4096, A100)**: ~350 GFLOPS → **1.8% of peak** (19.5 TFLOPS)

---

## 3. Version 2: Tiled GEMM with Shared Memory

### 3.1 The Tiling Idea

Instead of each thread independently reading from global memory, threads in a block **cooperatively load tiles** of A and B into shared memory, then compute from shared memory.

```
Outer loop: slide a TILE×TILE window across K dimension

Step t=0:                Step t=1:               Step t=2:
A tile     B tile        A tile     B tile       A tile     B tile
┌────┐    ┌────┐        ┌────┐    ┌────┐       ┌────┐    ┌────┐
│ A0 │    │ B0 │        │ A1 │    │ B1 │       │ A2 │    │ B2 │
└────┘    └────┘        └────┘    └────┘       └────┘    └────┘

C_tile += A0 × B0      C_tile += A1 × B1      C_tile += A2 × B2
```

### 3.2 Memory Traffic Reduction

With tile size $T$:
- Each tile of A ($T \times T$) is loaded once per block, shared by $T$ threads in the column direction
- Each tile of B ($T \times T$) is loaded once per block, shared by $T$ threads in the row direction

**Total global reads per block**: $2 \times T^2 \times (K/T) = 2TK$

**Total across all blocks**: $\frac{M}{T} \times \frac{N}{T} \times 2TK = \frac{2MNK}{T}$

Reduction factor vs naive: **$T\times$** less global memory traffic!

With $T = 32$: 32× fewer global memory reads.

### 3.3 Kernel Code

```cuda
#define TILE 32

__global__ void sgemm_tiled(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles along K
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Collaborative load: each thread loads one element
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 3.4 Performance Analysis

For the inner loop (`for k in 0..TILE`):
- 2 shared memory reads + 1 FMA = 2 reads + 2 FLOPs per iteration
- Shared memory latency: ~20 cycles vs global ~400 cycles

**Performance (4096×4096, A100)**: ~3800 GFLOPS → **19.5% of peak**

Major improvement over naive, but still far from cuBLAS. Why?

---

## 4. Version 3: Register Blocking (2×2 Per Thread)

### 4.1 The Problem with Version 2

Each thread computes **1 output element** but loads from shared memory every cycle. The ratio:
- 2 shared memory loads per 2 FLOPs → compute intensity = 1.0
- Shared memory bandwidth becomes the bottleneck!

### 4.2 Register Blocking — More Work Per Thread

Each thread computes a **2×2 (or 4×4) sub-tile** of C, reusing A and B values loaded into registers:

```
Thread computes C[ry:ry+2][cx:cx+2]:

For each k:
  a0 = As[ry+0][k]     ← load once
  a1 = As[ry+1][k]     ← load once
  b0 = Bs[k][cx+0]     ← load once
  b1 = Bs[k][cx+1]     ← load once

  c00 += a0 * b0        ← reuse a0, b0
  c01 += a0 * b1        ← reuse a0
  c10 += a1 * b0        ← reuse b0
  c11 += a1 * b1        ← reuse a1, b1

Shared mem loads: 4      FLOPs: 8     Ratio: 2.0 (vs 1.0 before)
```

For a $R \times R$ register tile:
$$\text{Loads} = 2R, \quad \text{FLOPs} = 2R^2, \quad \text{Ratio} = R$$

### 4.3 Kernel Code (2×2 Register Blocking)

```cuda
#define TILE 32
#define REG  2   // each thread computes REG×REG outputs

__global__ void sgemm_reg_blocked(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // Thread position — note: fewer threads per block needed
    int ty = threadIdx.y;  // 0..TILE/REG-1
    int tx = threadIdx.x;  // 0..TILE/REG-1

    int row_base = blockIdx.y * TILE + ty * REG;
    int col_base = blockIdx.x * TILE + tx * REG;

    // Register accumulators
    float c[REG][REG] = {{0.0f}};

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tile — each thread loads REG elements of A and B
        for (int r = 0; r < REG; r++) {
            int a_row = row_base + r;
            int a_col = t * TILE + tx * REG;
            for (int rr = 0; rr < REG; rr++) {
                As[ty * REG + r][tx * REG + rr] =
                    (a_row < M && (a_col + rr) < K)
                    ? A[a_row * K + a_col + rr] : 0.0f;
            }
            int b_row = t * TILE + ty * REG + r;
            int b_col = col_base;
            for (int rr = 0; rr < REG; rr++) {
                Bs[ty * REG + r][tx * REG + rr] =
                    (b_row < K && (b_col + rr) < N)
                    ? B[b_row * N + b_col + rr] : 0.0f;
            }
        }
        __syncthreads();

        // Compute with register reuse
        for (int k = 0; k < TILE; k++) {
            float a[REG], b[REG];
            for (int r = 0; r < REG; r++) {
                a[r] = As[ty * REG + r][k];
                b[r] = Bs[k][tx * REG + r];
            }
            for (int ri = 0; ri < REG; ri++) {
                for (int rj = 0; rj < REG; rj++) {
                    c[ri][rj] += a[ri] * b[rj];
                }
            }
        }
        __syncthreads();
    }

    // Write results
    for (int ri = 0; ri < REG; ri++) {
        for (int rj = 0; rj < REG; rj++) {
            int r = row_base + ri;
            int cc = col_base + rj;
            if (r < M && cc < N) {
                C[r * N + cc] = c[ri][rj];
            }
        }
    }
}

// Launch: fewer threads since each handles REG×REG outputs
dim3 block(TILE / REG, TILE / REG);  // 16×16 = 256 threads
dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
```

**Performance (4096×4096, A100)**: ~7200 GFLOPS → **37% of peak**

---

## 5. Performance Comparison

### 5.1 Benchmark Results (M = N = K = 4096, FP32, A100)

```
Version                     GFLOPS    % of cuBLAS    Global Reads
──────────────────────────────────────────────────────────────────
V1: Naive                     350        1.9%        2 × M×N×K
V2: Tiled (TILE=32)          3800       20.5%        2 × M×N×K / 32
V3: Reg-blocked (2×2)        7200       38.9%        same as V2
V4: Reg-blocked (4×4)       11500       62.2%        same as V2
cuBLAS                      18500      100.0%        highly optimized
```

### 5.2 What cuBLAS Does Beyond V3

```
cuBLAS additional optimizations:
├── 8×8 or 16×16 register blocking
├── Double buffering (prefetch next tile while computing current)
├── Vectorized loads (float4 = 128-bit loads)
├── Warp-level matrix operations (wmma/mma PTX instructions)
├── Heuristic tile-size selection based on M, N, K
├── Multi-level tiling (thread-block tile → warp tile → thread tile)
└── Assembly-level tuning for specific GPU architectures
```

### 5.3 Scaling With Matrix Size

```
Size          Naive    Tiled    Reg(4×4)  cuBLAS
─────────────────────────────────────────────────
 256×256       180      650      1200      3500
 512×512       250     1800      4200      9800
1024×1024      310     2900      7800     15200
2048×2048      340     3500     10200     17500
4096×4096      350     3800     11500     18500
```

Small matrices are harder — not enough parallelism to saturate the GPU.

---

## 6. Loop Unrolling

The compiler can unroll the inner `k` loop for higher ILP:

```cuda
// Manual unrolling (or use #pragma unroll)
#pragma unroll
for (int k = 0; k < TILE; k++) {
    // compiler generates TILE separate FMA instructions
    // no loop overhead, better instruction scheduling
}
```

Impact on the register-blocked kernel:

```
Without #pragma unroll:  7200 GFLOPS
With    #pragma unroll: 11500 GFLOPS  (+60%)
```

The compiler can also unroll with `-O3`, but `#pragma unroll` is explicit and reliable.

---

## Hands-On Exercises

1. **Implement all three versions**: Naive, tiled (TILE=16 and TILE=32), and register-blocked (REG=2, REG=4). Time each with `cudaEvent_t`.

2. **Verify correctness**: Compare each version against cuBLAS output using max absolute error (should be < $10^{-5}$ for FP32).

3. **Profile with ncu**: For each version, record:
   - SM throughput %
   - Memory throughput %
   - Achieved FLOPS
   - Dominant warp stall reason

4. **Tile size sweep**: Try TILE = 8, 16, 32, 64. Plot GFLOPS vs tile size. Why does 64 sometimes perform worse than 32?

5. **cuBLAS comparison**: Use `cublasSgemm` and measure the gap. How close can you get?

---

## Key Takeaways

1. Naive GEMM achieves ~2% of peak because each element is read $O(N)$ times from slow global memory.
2. **Tiling** with shared memory reduces global reads by $T\times$, where $T$ is the tile size.
3. **Register blocking** increases the compute-to-shared-memory-load ratio from 1 to $R$ (register tile dimension).
4. `#pragma unroll` and vectorized loads provide another 1.5–2× on top of algorithmic improvements.
5. cuBLAS uses multi-level tiling, assembly tuning, and Tensor Core instructions to reach >90% of peak.
6. The progression Naive → Tiled → Register-blocked → cuBLAS mirrors what ML compilers (Triton, TVM) generate automatically.

---

## Further Reading

- Volkov & Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra" (SC'08) — [PDF](https://people.eecs.berkeley.edu/~volkov/volkov08-sc.pdf)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass) — NVIDIA's open-source GEMM library
- Huang et al., "Strassen's Algorithm Reloaded on GPUs" (SC'16)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm's excellent walkthrough

---

> **Tomorrow**: [Day 7 — Mini-Project: Profile & Optimize a GEMM](day-07-mini-project-gemm.md). You'll implement all four GEMM versions, benchmark them against cuBLAS, build a roofline plot, and write up a performance analysis covering everything from Week 1.
