# Day 7: Mini-Project — Profile & Optimize a GEMM

> **Phase I — GPU Foundations & CUDA** · Week 1 · Day 7 of 70

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 6: Matrix Multiply — Naive to Tiled](day-06-matrix-multiply-naive-to-tiled.md) | [Day 8: PyTorch Under the Hood](../week-02/day-08-pytorch-under-the-hood.md) | [Week 1](../week-01/) | [Phase I](../../#phase-i) | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

This is your Week 1 capstone. You'll consolidate every concept — CUDA execution model, memory coalescing, shared memory tiling, register blocking, profiling, and roofline analysis — into a single, complete project. By the end, you'll have a benchmarked portfolio piece and the muscle memory to approach any GPU optimization problem systematically.

---

## 1. Project Overview

### 1.1 Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **4 GEMM kernels** | Naive, tiled, register-blocked, double-buffered |
| 2 | **Benchmarking harness** | Timing, correctness check, GFLOPS calculation |
| 3 | **Nsight Compute profiles** | One `.ncu-rep` per kernel |
| 4 | **Roofline plot** | All 4 kernels + cuBLAS on one chart |
| 5 | **Analysis writeup** | Bottleneck diagnosis per version |

### 1.2 Project Structure

```
gemm-project/
├── Makefile
├── include/
│   └── common.h          # error checking, timing utilities
├── src/
│   ├── naive.cu           # Version 1
│   ├── tiled.cu           # Version 2
│   ├── reg_blocked.cu     # Version 3
│   ├── double_buffer.cu   # Version 4
│   └── benchmark.cu       # Main driver
├── scripts/
│   ├── profile.sh         # ncu profiling commands
│   └── plot_roofline.py   # Roofline visualization
├── results/
│   ├── benchmark.csv      # Timing results
│   └── roofline.png       # Generated plot
└── README.md              # Analysis writeup
```

---

## 2. Common Utilities

### 2.1 include/common.h

```cuda
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", \
                __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

// Check correctness against reference
float check_result(const float *test, const float *ref, int N) {
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(test[i] - ref[i]);
        float rel = err / fmaxf(fabsf(ref[i]), 1e-7f);
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

// GPU timing utility
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { cudaEventRecord(start); }
    float end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Calculate GFLOPS
float calc_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * M * N * K;
    return (float)(flops / (ms * 1e6));  // ms → GFLOPS
}
```

---

## 3. The Four Kernel Versions

### 3.1 Version 1: Naive

```cuda
// src/naive.cu
#include "common.h"

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

void launch_naive(const float *A, const float *B, float *C,
                  int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    sgemm_naive<<<grid, block>>>(A, B, C, M, N, K);
}
```

### 3.2 Version 2: Shared Memory Tiled

```cuda
// src/tiled.cu
#include "common.h"
#define TILE 32

__global__ void sgemm_tiled(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_tiled(const float *A, const float *B, float *C,
                  int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
}
```

### 3.3 Version 3: Register Blocked (4×4)

```cuda
// src/reg_blocked.cu
#include "common.h"
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void sgemm_reg_blocked(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // Thread indices
    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    int row_base = blockIdx.y * BM + ty * TM;
    int col_base = blockIdx.x * BN + tx * TN;

    // Register accumulators
    float c[TM][TN] = {};

    int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; t++) {
        // Load tiles cooperatively
        // Each thread loads TM elements of A and TN elements of B
        for (int i = 0; i < TM; i++) {
            int a_row = row_base + i;
            int a_col = t * BK + tx;
            As[tx < BK ? tx : 0][ty * TM + i] =
                (a_row < M && (t * BK + tx) < K)
                ? A[a_row * K + t * BK + tx] : 0.0f;
        }
        for (int i = 0; i < TN; i++) {
            int b_row = t * BK + ty;
            int b_col = col_base + i;
            Bs[ty < BK ? ty : 0][tx * TN + i] =
                (b_row < K && b_col < N)
                ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        // Compute with register reuse
        for (int k = 0; k < BK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int i = 0; i < TN; i++) b[i] = Bs[k][tx * TN + i];

            #pragma unroll
            for (int ri = 0; ri < TM; ri++) {
                #pragma unroll
                for (int rj = 0; rj < TN; rj++) {
                    c[ri][rj] += a[ri] * b[rj];
                }
            }
        }
        __syncthreads();
    }

    // Store results
    for (int ri = 0; ri < TM; ri++) {
        for (int rj = 0; rj < TN; rj++) {
            int r = row_base + ri;
            int cc = col_base + rj;
            if (r < M && cc < N) {
                C[r * N + cc] = c[ri][rj];
            }
        }
    }
}
```

### 3.4 Version 4: Double Buffering

Double buffering overlaps the **next tile's load** with the **current tile's compute**:

```
Tile t:     LOAD tile t+1   |||   COMPUTE tile t
Tile t+1:   LOAD tile t+2   |||   COMPUTE tile t+1

Pipeline: load and compute happen concurrently
```

```cuda
// src/double_buffer.cu — key difference from Version 3
// Use two sets of shared memory buffers and alternate

__shared__ float As[2][BK][BM];  // double buffer
__shared__ float Bs[2][BK][BN];

int cur = 0;  // current buffer index

// Pre-load first tile into buffer 0
load_tile(As[0], Bs[0], A, B, t=0, ...);
__syncthreads();

for (int t = 0; t < num_tiles - 1; t++) {
    int nxt = 1 - cur;

    // Load next tile into alternate buffer (overlaps with compute)
    load_tile(As[nxt], Bs[nxt], A, B, t+1, ...);

    // Compute from current buffer
    compute_tile(c, As[cur], Bs[cur]);

    __syncthreads();
    cur = nxt;
}

// Compute last tile
compute_tile(c, As[cur], Bs[cur]);
```

---

## 4. Benchmarking Harness

### 4.1 src/benchmark.cu

```cuda
#include "common.h"

// Forward declarations
void launch_naive(const float*, const float*, float*, int, int, int);
void launch_tiled(const float*, const float*, float*, int, int, int);
void launch_reg_blocked(const float*, const float*, float*, int, int, int);
void launch_double_buffer(const float*, const float*, float*, int, int, int);

int main() {
    const int sizes[] = {512, 1024, 2048, 4096};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int warmup = 3, repeats = 10;

    printf("%-6s %-12s %-12s %-12s %-12s %-12s\n",
           "Size", "Naive", "Tiled", "RegBlock", "DblBuffer", "cuBLAS");
    printf("%-6s %-12s %-12s %-12s %-12s %-12s\n",
           "", "(GFLOPS)", "(GFLOPS)", "(GFLOPS)", "(GFLOPS)", "(GFLOPS)");

    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t bytes = M * N * sizeof(float);

        // Allocate
        float *h_A = (float*)malloc(M * K * sizeof(float));
        float *h_B = (float*)malloc(K * N * sizeof(float));
        float *h_C = (float*)malloc(bytes);
        float *h_ref = (float*)malloc(bytes);
        init_matrix(h_A, M, K);
        init_matrix(h_B, K, N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));

        // cuBLAS reference
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f, beta = 0.0f;

        GpuTimer timer;

        // Benchmark cuBLAS
        for (int i = 0; i < warmup; i++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        timer.begin();
        for (int i = 0; i < repeats; i++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        float cublas_ms = timer.end() / repeats;
        float cublas_gf = calc_gflops(M, N, K, cublas_ms);

        // Save reference result
        CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes, cudaMemcpyDeviceToHost));

        // Helper macro for benchmarking each version
        #define BENCH(name, launch_fn) \
            for (int i = 0; i < warmup; i++) launch_fn(d_A, d_B, d_C, M, N, K); \
            timer.begin(); \
            for (int i = 0; i < repeats; i++) launch_fn(d_A, d_B, d_C, M, N, K); \
            float name##_ms = timer.end() / repeats; \
            float name##_gf = calc_gflops(M, N, K, name##_ms); \
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost)); \
            float name##_err = check_result(h_C, h_ref, M * N);

        BENCH(naive, launch_naive);
        BENCH(tiled, launch_tiled);
        BENCH(regbl, launch_reg_blocked);
        BENCH(dblbf, launch_double_buffer);

        printf("%-6d %-12.1f %-12.1f %-12.1f %-12.1f %-12.1f\n",
               M, naive_gf, tiled_gf, regbl_gf, dblbf_gf, cublas_gf);

        // Cleanup
        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C); free(h_ref);
    }
    return 0;
}
```

### 4.2 Makefile

```makefile
NVCC = nvcc
CFLAGS = -O3 -arch=sm_80  # sm_80 for A100
LDFLAGS = -lcublas

SRCS = src/naive.cu src/tiled.cu src/reg_blocked.cu \
       src/double_buffer.cu src/benchmark.cu

all: benchmark

benchmark: $(SRCS) include/common.h
	$(NVCC) $(CFLAGS) -I include $(SRCS) -o $@ $(LDFLAGS)

profile: benchmark
	bash scripts/profile.sh

clean:
	rm -f benchmark results/*.ncu-rep
```

---

## 5. Expected Results

### 5.1 Performance Table (A100, FP32)

```
Size   Naive       Tiled       RegBlock    DblBuffer   cuBLAS
       (GFLOPS)    (GFLOPS)    (GFLOPS)    (GFLOPS)    (GFLOPS)
──────────────────────────────────────────────────────────────────
 512     180         650        1800        2100        3500
1024     250        1800        5500        6800        9800
2048     310        2900        8500       10200       15200
4096     350        3800       11500       13200       18500
```

### 5.2 Profiling Metrics (4096×4096)

```
Metric              Naive    Tiled    RegBlock  DblBuffer  cuBLAS
─────────────────────────────────────────────────────────────────
SM Throughput (%)     12       45        72        81       94
Mem Throughput (%)    95       87        45        38       22
Occupancy (%)        100       50        25        25       varies
Dominant Stall     mem_dep  mem_dep  not_sel   not_sel  math_pipe
```

---

## 6. Roofline Plot Script

### 6.1 scripts/plot_roofline.py

```python
#!/usr/bin/env python3
"""Generate roofline plot from benchmark results."""
import matplotlib.pyplot as plt
import numpy as np
import csv

# A100 specs
PEAK_FLOPS = 19500  # GFLOPS FP32
PEAK_BW = 2039      # GB/s

def plot_roofline(results_csv='results/benchmark.csv'):
    ai = np.logspace(-1, 3, 1000)
    roofline = np.minimum(PEAK_FLOPS, PEAK_BW * ai)
    ridge = PEAK_FLOPS / PEAK_BW

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.loglog(ai, roofline, 'b-', lw=2.5, label='Roofline (FP32)')
    ax.axvline(ridge, color='gray', ls='--', alpha=0.4,
               label=f'Ridge = {ridge:.1f} FLOP/B')

    # For GEMM, AI = N/6 (square matrices FP32)
    kernels = {
        'Naive':       (683, 350,   'rs'),
        'Tiled':       (683, 3800,  'g^'),
        'Reg-Blocked': (683, 11500, 'bD'),
        'Dbl-Buffer':  (683, 13200, 'mp'),
        'cuBLAS':      (683, 18500, 'k*'),
    }

    for name, (k_ai, k_gf, fmt) in kernels.items():
        ax.plot(k_ai, k_gf, fmt, markersize=12, label=name)

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=13)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=13)
    ax.set_title('Roofline Analysis — GEMM Optimization (A100, 4096³)', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.1, 2000)
    ax.set_ylim(100, 25000)
    plt.tight_layout()
    plt.savefig('results/roofline.png', dpi=150)
    print("Saved results/roofline.png")
    plt.show()

if __name__ == '__main__':
    plot_roofline()
```

---

## 7. Reflection Checklist

After completing the project, verify you can answer these questions:

### GPU Architecture (Days 1-2)
- [ ] What is an SM? How many are on the GPU you used?
- [ ] What is a warp? Why does warp size = 32 matter?
- [ ] Draw the GPU memory hierarchy (registers → shared → L1 → L2 → HBM)

### CUDA Programming (Day 3)
- [ ] Write the global thread index formula for 1D and 2D grids
- [ ] Why do we need bounds checking (`if (idx < N)`)?
- [ ] What happens if you forget `cudaFree()`?

### Memory Coalescing (Day 4)
- [ ] What is a 128-byte cache line transaction?
- [ ] Why is SoA better than AoS on GPUs?
- [ ] What does the `+1` padding trick do for bank conflicts?

### Profiling (Day 5)
- [ ] When to use Nsight Systems vs Nsight Compute?
- [ ] What is arithmetic intensity? Calculate it for vector add.
- [ ] If SM throughput is 90% and memory throughput is 20%, is the kernel compute-bound or memory-bound?

### GEMM Optimization (Day 6)
- [ ] Why does the naive GEMM read each A element $N$ times?
- [ ] How does tiling reduce global memory traffic by $T\times$?
- [ ] Why does register blocking help even though total FLOPs don't change?

### Integration (Day 7)
- [ ] Place all 4 kernels + cuBLAS on the roofline. Are they where you expect?
- [ ] What is the dominant bottleneck for each version?
- [ ] What optimization would you try next to close the gap to cuBLAS?

---

## 8. Stretch Goals

If you finish early, try these extensions:

1. **FP16 GEMM**: Convert to `__half` types. How does performance change? What's the new peak?

2. **Tensor Core GEMM**: Use `wmma` (Warp Matrix Multiply-Accumulate) API for 16×16×16 tiles. See [CUDA wmma docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma).

3. **Non-square matrices**: Test with $M=1, N=4096, K=4096$ (GEMV). How does performance degrade? Why?

4. **Autotuning**: Write a Python script that sweeps TILE, BM, BN, BK, TM, TN and finds the best configuration for a given GPU.

---

## Key Takeaways

1. A systematic optimization workflow — **profile → diagnose → optimize → re-profile** — beats guessing every time.
2. The four GEMM versions map to a clean performance ladder: naive (2%) → tiled (20%) → register-blocked (60%) → double-buffered (70%) of cuBLAS.
3. **Roofline analysis** instantly tells you whether to optimize memory access or compute.
4. GPU optimization is a hierarchy: **algorithm > memory access pattern > compute optimization > micro-architecture tricks**.
5. cuBLAS's remaining edge comes from assembly-level tuning, Tensor Cores, and multi-level tiling — the same techniques ML compilers automate.

---

## Further Reading

- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass) — Study `gemm/` directory for production GEMM implementations
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm's walkthrough
- Jia, Maggioni & Staiger, "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" (2018)
- [Programming Massively Parallel Processors (Hwu, Kirk & El Hajj)](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — Chapters 4-6

---

> **Next week**: [Day 8 — PyTorch Under the Hood](../week-02/day-08-pytorch-under-the-hood.md). You'll trace what happens when you call `torch.matmul()` — from Python through the dispatcher, to ATen, to cuBLAS, and finally to the GPU kernels you now understand.
