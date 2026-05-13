# Day 2: GPU Architecture Deep Dive

> **Phase I · Week 1 · Day 2 of 70 · 2.5 hours**

*"You can't optimize what you don't understand. Every ML compiler decision traces back to hardware constraints."*

---

## Navigation

| | |
|---|---|
| **Previous** | [← Day 1: Why ML Needs Compilers](day-01-why-ml-needs-compilers.md) |
| **Next** | [Day 3: CUDA Programming Basics →](day-03-cuda-programming-basics.md) |
| **Week** | Week 1: GPU Architecture & CUDA |
| **Phase** | Phase I: Hardware & Compute Foundations |
| **Curriculum** | [Full Curriculum](../../curriculum.html) |

---

## Why This Matters

Every decision an ML compiler makes — tile sizes, loop orders, memory placement, thread counts — is constrained by GPU hardware. Understanding the architecture lets you predict whether a kernel is memory-bound or compute-bound, why certain tile sizes are magic numbers, and where performance cliffs hide.

---

## 1. GPU vs CPU Philosophy

### 1.1 Fundamental Design Difference

```
CPU (Latency-optimized):                GPU (Throughput-optimized):
┌────────────────────────┐             ┌────────────────────────┐
│ Big cache (L1/L2/L3)   │             │ Tiny caches per SM     │
│ Complex branch predict  │             │ Simple in-order cores  │
│ Out-of-order execution  │             │ Massive parallelism    │
│ Few powerful cores      │             │ Thousands of threads   │
│ (8-128 cores)          │             │ (10K-100K threads)     │
└────────────────────────┘             └────────────────────────┘
   Optimize: single-thread speed          Optimize: total throughput
```

**Key insight**: GPUs hide memory latency with **parallelism**, not caches. While one warp waits for data, another warp executes.

### 1.2 Why GPUs Dominate ML

Matrix multiplication $C = A \times B$ for $n \times n$ matrices:
- **Operations**: $O(n^3)$ multiplications and additions
- **Data**: $O(n^2)$ elements to read/write
- **Arithmetic intensity**: $O(n)$ — increases with problem size

This high ratio of compute-to-memory makes matmul ideal for GPUs. Neural networks are mostly matmul.

---

## 2. Streaming Multiprocessor (SM) Architecture

### 2.1 NVIDIA A100 Hierarchy

```
GPU (A100)
├── 108 Streaming Multiprocessors (SMs)
│   ├── 4 Processing Blocks per SM
│   │   ├── 16 FP32 CUDA Cores per block (64 per SM)
│   │   ├── 16 FP64 Cores per block
│   │   ├── 8 Tensor Cores per block (using FP16/TF32/INT8)
│   │   ├── 1 Warp Scheduler per block
│   │   └── Register File (16,384 × 32-bit registers per block)
│   ├── 192 KB L1 Cache / Shared Memory (configurable split)
│   └── 65,536 × 32-bit registers total
├── 40 MB L2 Cache
└── 80 GB HBM2e Global Memory
    └── 2039 GB/s bandwidth
```

### 2.2 The Warp: Fundamental Execution Unit

A **warp** is 32 threads executing the **same instruction** in lockstep (SIMT):

```
Warp 0 (32 threads):
Thread 0:  FMUL R1, R2, R3    ← same instruction
Thread 1:  FMUL R1, R2, R3    ← different data (registers)
Thread 2:  FMUL R1, R2, R3
...
Thread 31: FMUL R1, R2, R3

All 32 threads execute FMUL simultaneously
```

**Warp divergence**: If threads in a warp take different branches, both paths execute serially:

```python
# This causes divergence — avoid in hot kernels!
if thread_id % 2 == 0:
    do_path_A()  # Executes while odd threads idle
else:
    do_path_B()  # Executes while even threads idle
# Total time = time(A) + time(B), not max(A, B)
```

### 2.3 Occupancy

**Occupancy** = active warps / maximum warps per SM

An A100 SM supports up to 64 active warps (2048 threads). Resources that limit occupancy:

| Resource | Per SM | Limit |
|----------|--------|-------|
| Registers | 65,536 | Kernel uses 64 regs/thread → 65536/64 = 1024 threads = 32 warps max |
| Shared memory | 164 KB | Kernel uses 48 KB → floor(164/48) = 3 blocks max |
| Thread blocks | 32 max | — |

Higher occupancy = better latency hiding (more warps to switch to while waiting for memory).

---

## 3. Memory Hierarchy

### 3.1 The Speed-Size Tradeoff

```
                    Speed            Size        Scope
                    ─────            ────        ─────
Registers        ◀══════▶  ~8 TB/s   256 KB/SM   Per thread
    │
Shared Memory    ◀══════▶  ~19 TB/s  164 KB/SM   Per thread block
    │
L1 Cache         ◀══════▶  ~19 TB/s  (shared)    Per SM
    │
L2 Cache         ◀══════▶  ~6 TB/s   40 MB       Shared across GPU
    │
Global Memory    ◀══════▶  ~2 TB/s   80 GB       Shared across GPU
(HBM)
    │
CPU Memory       ◀══════▶  ~32 GB/s  System RAM  Via PCIe
(Host)
```

### 3.2 Memory Coalescing

Global memory is accessed in **128-byte transactions**. If 32 threads each request a consecutive 4-byte float, that's one transaction. If they request scattered addresses, it could be 32 transactions:

```
Coalesced access (GOOD):
Thread 0 → addr[0]    ┐
Thread 1 → addr[1]    │ 1 × 128-byte transaction
Thread 2 → addr[2]    │
...                    │
Thread 31 → addr[31]  ┘

Strided access (BAD):
Thread 0 → addr[0]      → 1 transaction
Thread 1 → addr[1024]   → 1 transaction
Thread 2 → addr[2048]   → 1 transaction
...
Thread 31 → addr[31744] → 32 transactions!
```

**Rule**: Access memory in contiguous patterns. This is why row-major vs column-major layout matters enormously.

### 3.3 Shared Memory and Bank Conflicts

Shared memory has 32 banks, each 4 bytes wide. Threads accessing different banks → parallel. Same bank → serialized:

```
No conflict:                    Bank conflict:
Bank 0: Thread 0               Bank 0: Thread 0, Thread 1  ← serial!
Bank 1: Thread 1               Bank 1: (empty)
Bank 2: Thread 2               Bank 2: Thread 2, Thread 3  ← serial!
...                             ...
Bank 31: Thread 31             Bank 31: (empty)
Time: 1 cycle                  Time: 2 cycles
```

---

## 4. Tensor Cores

### 4.1 Matrix Multiply-Accumulate

Tensor Cores perform small matrix multiplications in hardware:

$$D = A \times B + C$$

where $A$ is $m \times k$, $B$ is $k \times n$, and $C$ is $m \times n$.

| Generation | Shape | Input Types | Throughput |
|-----------|-------|------------|-----------|
| Volta (V100) | 4×4×4 | FP16 → FP32 | 125 TFLOPS |
| Ampere (A100) | 16×8×16 | FP16/BF16/TF32/INT8 | 312 TFLOPS |
| Hopper (H100) | 16×8×16 | FP8/FP16/BF16/INT8 | 990 TFLOPS |

### 4.2 Why Tensor Cores Matter for Compilers

Using Tensor Cores requires:
1. **Correct data types**: FP16/BF16 inputs, FP32 accumulators
2. **Correct dimensions**: Multiples of the hardware tile (16×8×16)
3. **Correct memory layout**: Data must be in specific shared memory layout

The compiler must ensure all three — this is a major job of the TIR lowering pipeline in TVM.

---

## 5. The Roofline Model

### 5.1 Arithmetic Intensity

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

For A100 at FP16:
- **Peak compute**: 312 TFLOPS
- **Peak bandwidth**: 2039 GB/s
- **Ridge point**: $312 \times 10^{12} / (2039 \times 10^9) = 153$ FLOP/byte

```
Performance
(TFLOPS)
    312 ──────────────────────────────────── Peak FLOPS
    │                         ╱
    │                       ╱  Compute-bound
    │                     ╱    (matmul, conv)
    │                   ╱
    │                 ╱
    │               ╱
    │             ╱ ← Ridge point (153 FLOP/byte)
    │           ╱
    │         ╱  Memory-bound
    │       ╱    (activation, normalization)
    │     ╱
    │   ╱
    │ ╱
    └──────────────────────────────────── Arithmetic Intensity
                    153
```

### 5.2 What This Means for Compilers

| Operation | AI (FLOP/byte) | Bound | Compiler Strategy |
|-----------|----------------|-------|------------------|
| Matmul (large) | 100-1000+ | Compute | Maximize Tensor Core utilization |
| Convolution | 10-100 | Mixed | Fuse with activations |
| LayerNorm | ~5 | Memory | Fuse with adjacent ops |
| Softmax | ~3 | Memory | Fuse with attention |
| GELU/ReLU | 0.25 | Memory | MUST fuse — never run standalone |

---

## 6. Hands-On: Profiling a Real Model

### Exercise 1: GPU Utilization Analysis

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True).cuda().half()
x = torch.randn(64, 3, 224, 224, device='cuda', dtype=torch.float16)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=3, active=5),
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        with torch.no_grad():
            model(x)
        prof.step()

# Export for Chrome trace viewer
prof.export_chrome_trace("resnet18_trace.json")

# Summary
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=15
))
```

### Exercise 2: Identify Memory-Bound Operations

From the profiler output:
1. Which operations are memory-bound (low arithmetic intensity)?
2. Which operations could benefit from fusion?
3. What percentage of GPU time is spent on non-matmul operations?

### Exercise 3: Roofline Estimation

```python
# For a matmul: C[M,N] = A[M,K] @ B[K,N]
M, K, N = 4096, 4096, 4096

flops = 2 * M * K * N  # multiply + add
bytes_accessed = (M*K + K*N + M*N) * 2  # FP16 = 2 bytes

ai = flops / bytes_accessed
print(f"Arithmetic Intensity: {ai:.1f} FLOP/byte")
print(f"Above ridge point (153)? {ai > 153}")
# → AI ≈ 1365 FLOP/byte → compute-bound ✓
```

---

## 7. Key Takeaways

1. **GPUs hide latency with parallelism** — not caches like CPUs
2. **Warps** (32 threads) are the atomic execution unit — divergence is expensive
3. **Memory hierarchy**: registers > shared > L2 > global — each 10× slower
4. **Coalescing** is mandatory — scattered memory access kills performance
5. **Tensor Cores** accelerate matmul but need specific types and dimensions
6. **Roofline model** predicts whether a kernel is memory-bound or compute-bound
7. Most activation functions are **memory-bound** → fusion is the #1 optimization

---

## Further Reading

- [CUDA C++ Programming Guide — Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [Roofline: An Insightful Visual Performance Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyworof.pdf)
- [NVIDIA A100 Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

---

## Tomorrow

Day 3 shifts from theory to practice: **CUDA programming basics** — writing your first GPU kernels from scratch.
