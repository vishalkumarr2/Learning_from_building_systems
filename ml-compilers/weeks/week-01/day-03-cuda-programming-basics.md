# Day 3: CUDA Programming Basics

> **Phase I — GPU Foundations & CUDA** · Week 1 · Day 3 of 70

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 2: GPU Architecture](day-02-gpu-architecture.md) | [Day 4: Memory Coalescing & Shared Memory](day-04-memory-coalescing-shared-memory.md) | [Week 1](../week-01/) | [Phase I](../../#phase-i) | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

Every ML framework — PyTorch, JAX, TensorFlow — ultimately emits GPU kernels to do the heavy lifting. Understanding CUDA's execution model is the foundation for writing custom operators, debugging performance regressions, and understanding what compilers like Triton and XLA actually generate. Today you go from "GPU is a black box" to "I can launch my own kernels."

---

## 1. The CUDA Execution Model

### 1.1 Host vs Device

CUDA programs are **heterogeneous**: the CPU (host) orchestrates work, and the GPU (device) executes massively parallel kernels.

```
┌──────────────────┐          ┌──────────────────────────────┐
│       HOST       │          │           DEVICE             │
│   (CPU + DRAM)   │  PCIe /  │        (GPU + HBM)           │
│                  │  NVLink  │                              │
│  main()          │ ──────►  │  kernel<<<grid, block>>>()   │
│  cudaMalloc()    │          │  thousands of threads        │
│  cudaMemcpy()    │ ◄──────  │  execute in parallel         │
│  cudaFree()      │          │                              │
└──────────────────┘          └──────────────────────────────┘
```

### 1.2 Kernel Launch Syntax

A **kernel** is a function decorated with `__global__` that runs on the device and is callable from the host:

```cuda
__global__ void my_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}
```

Launch with the **triple-chevron** syntax:

```cuda
int threads_per_block = 256;
int blocks = (N + threads_per_block - 1) / threads_per_block;
my_kernel<<<blocks, threads_per_block>>>(d_data, N);
```

### 1.3 Function Qualifiers

| Qualifier | Runs on | Callable from |
|-----------|---------|---------------|
| `__global__` | Device | Host (or device with dynamic parallelism) |
| `__device__` | Device | Device only |
| `__host__` | Host | Host only |
| `__host__ __device__` | Both | Both |

---

## 2. Grid, Block, and Thread Hierarchy

### 2.1 The Three-Level Hierarchy

```
Grid (1 kernel launch)
├── Block (0,0)          Block (1,0)          Block (2,0)
│   ├── Thread (0,0)     ├── Thread (0,0)     ├── ...
│   ├── Thread (1,0)     ├── Thread (1,0)     │
│   ├── Thread (2,0)     ├── ...              │
│   └── ...              └── ...              └── ...
├── Block (0,1)          Block (1,1)          Block (2,1)
│   └── ...              └── ...              └── ...
└── ...
```

- **Grid**: all blocks launched by a single kernel call
- **Block**: a group of threads that can cooperate via shared memory and `__syncthreads()`
- **Thread**: the smallest unit of execution

### 2.2 Built-in Variables

Every thread knows its position via built-in variables:

| Variable | Type | Meaning |
|----------|------|---------|
| `threadIdx.x/y/z` | `uint3` | Thread index within its block |
| `blockIdx.x/y/z` | `uint3` | Block index within the grid |
| `blockDim.x/y/z` | `dim3` | Number of threads per block |
| `gridDim.x/y/z` | `dim3` | Number of blocks in the grid |

The **global thread index** (1D case):

$$\text{idx} = \texttt{blockIdx.x} \times \texttt{blockDim.x} + \texttt{threadIdx.x}$$

### 2.3 dim3 for Multi-Dimensional Launches

```cuda
// 2D grid of 2D blocks
dim3 block(16, 16);       // 256 threads per block
dim3 grid((width + 15) / 16, (height + 15) / 16);
matrix_kernel<<<grid, block>>>(d_out, d_in, width, height);
```

Inside the kernel:
```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## 3. Memory Management

### 3.1 Allocation & Transfer API

```cuda
float *d_a, *d_b, *d_c;      // device pointers
size_t bytes = N * sizeof(float);

// Allocate on device
cudaMalloc(&d_a, bytes);
cudaMalloc(&d_b, bytes);
cudaMalloc(&d_c, bytes);

// Copy host → device
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

// Launch kernel
add_kernel<<<blocks, threads>>>(d_c, d_a, d_b, N);

// Copy device → host
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

### 3.2 Memory Transfer Directions

```
cudaMemcpyHostToDevice    H → D   (upload input)
cudaMemcpyDeviceToHost    D → H   (download result)
cudaMemcpyDeviceToDevice  D → D   (on-device copy)
cudaMemcpyHostToHost      H → H   (rarely used)
```

### 3.3 Unified Memory (bonus)

```cuda
float *data;
cudaMallocManaged(&data, bytes);  // accessible from both CPU & GPU
// Driver handles page migration automatically
```

---

## 4. Error Handling

CUDA calls are **asynchronous** — errors may surface later. Always check:

```cuda
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

// After kernel launch (asynchronous check):
my_kernel<<<grid, block>>>(d_a, N);
CUDA_CHECK(cudaGetLastError());       // launch config errors
CUDA_CHECK(cudaDeviceSynchronize());  // runtime errors
```

---

## 5. Complete Example: Vector Addition

```cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;  // ~1M elements
    size_t bytes = N * sizeof(float);

    // Host allocation & init
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i) * sinf(i);
        h_b[i] = cosf(i) * cosf(i);
    }

    // Device allocation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // H → D
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vec_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // D → H
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify: sin²(x) + cos²(x) = 1.0
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_c[i] - 1.0f));
    }
    printf("Max error: %e\n", max_err);  // Should be ~0

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

Compile and run:
```bash
nvcc -o vec_add vec_add.cu -O2
./vec_add
```

---

## 6. Simple Parallel Reduction

Summing $N$ elements in $O(\log N)$ parallel steps:

```
Step 0:  [a0  a1  a2  a3  a4  a5  a6  a7]
Step 1:  [a0+a1   a2+a3   a4+a5   a6+a7]     (stride 1)
Step 2:  [a0..a3         a4..a7        ]       (stride 2)
Step 3:  [a0..a7                       ]       (stride 4)
```

```cuda
__global__ void reduce_sum(float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

Launch with dynamic shared memory:
```cuda
int smem_bytes = block_size * sizeof(float);
reduce_sum<<<grid_size, block_size, smem_bytes>>>(d_in, d_out, N);
```

> **Note**: This is a basic reduction. Mark Harris's classic "Optimizing Parallel Reduction in CUDA" shows 7 progressively faster versions. We'll revisit reductions in Week 3.

---

## Hands-On Exercises

1. **Modify vec_add** to compute $c[i] = a[i]^2 + b[i]^2 + 2 \cdot a[i] \cdot b[i]$ and verify it equals $(a[i]+b[i])^2$.

2. **2D kernel**: Write a kernel that takes a 1024×1024 matrix and sets each element to `row * 1024 + col`. Use `dim3` blocks of (16,16).

3. **Grid-stride loop**: Rewrite vec_add so a single block can process arrays larger than `blockDim.x * gridDim.x`:
   ```cuda
   for (int i = idx; i < N; i += blockDim.x * gridDim.x) { ... }
   ```

4. **Reduction**: Extend the reduction kernel to handle arrays larger than a single block (hint: run two passes, or use atomics for the final stage).

---

## Key Takeaways

1. CUDA kernels launch as a **grid of blocks of threads** — pick sizes to maximize occupancy.
2. The global thread index formula $\texttt{blockIdx.x} \times \texttt{blockDim.x} + \texttt{threadIdx.x}$ is the bread and butter of CUDA.
3. **Always bounds-check** (`if (idx < N)`) — grid dimensions rarely divide N exactly.
4. Memory transfers (`cudaMemcpy`) are expensive — minimize host ↔ device traffic.
5. Kernel launches are **asynchronous** — wrap every CUDA call in error-checking macros.
6. Shared memory and `__syncthreads()` enable intra-block cooperation (reduction, tiling).

---

## Further Reading

- [CUDA C++ Programming Guide — Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- Mark Harris, "Optimizing Parallel Reduction in CUDA" — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [CUDA By Example (Sanders & Kandrot)](https://developer.nvidia.com/cuda-example) — Chapters 2-4

---

> **Tomorrow**: [Day 4 — Memory Coalescing & Shared Memory](day-04-memory-coalescing-shared-memory.md). You'll learn why *how* threads access memory matters more than *how much* they compute, and how shared memory turns global memory bottlenecks into local wins.
