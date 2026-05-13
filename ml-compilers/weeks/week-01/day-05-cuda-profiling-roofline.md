# Day 5: CUDA Profiling & Roofline Model

> **Phase I — GPU Foundations & CUDA** · Week 1 · Day 5 of 70

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 4: Memory Coalescing & Shared Memory](day-04-memory-coalescing-shared-memory.md) | [Day 6: Matrix Multiply — Naive to Tiled](day-06-matrix-multiply-naive-to-tiled.md) | [Week 1](../week-01/) | [Phase I](../../#phase-i) | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

Writing a kernel is half the job — the other half is knowing *why* it's slow. Is it starved for memory bandwidth? Compute-limited? Waiting on synchronization? Profiling tools answer these questions quantitatively. The **roofline model** gives you a single plot that shows your kernel's distance from hardware limits and tells you exactly which optimization to try next. Every ML compiler (Triton, XLA, TVM) uses roofline-style analysis internally to guide code generation decisions.

---

## 1. The NVIDIA Profiling Toolchain

### 1.1 Nsight Systems — Timeline Profiler

Shows the **big picture**: CPU/GPU interleaving, kernel launch gaps, memory copies, API calls.

```bash
# Profile a complete application
nsys profile --stats=true ./my_cuda_app

# Generate a timeline report
nsys profile -o my_report ./my_cuda_app
# Open in Nsight Systems GUI: nsys-ui my_report.nsys-rep
```

What to look for:
```
┌────── CPU Timeline ──────┐  ┌────── GPU Timeline ──────┐
│ main()                   │  │                           │
│  ├─ cudaMemcpy H→D ─────│──│→ DtoH copy ███           │
│  ├─ launch kernel_1 ────│──│→ kernel_1 ████████        │
│  ├─ (CPU idle) ─────────│  │                           │
│  ├─ launch kernel_2 ────│──│→ kernel_2 ████████████    │
│  └─ cudaMemcpy D→H ─────│──│→ HtoD copy ██            │
└──────────────────────────┘  └───────────────────────────┘

Red flags: gaps between kernels, long memcpy blocking compute
```

### 1.2 Nsight Compute — Kernel Profiler

Deep-dives into a **single kernel**: SM utilization, memory throughput, stall reasons, occupancy.

```bash
# Profile all kernels
ncu ./my_cuda_app

# Profile a specific kernel with full metrics
ncu --set full -k "my_kernel" ./my_cuda_app

# Save report for GUI
ncu --set full -o my_kernel_report ./my_cuda_app
```

### 1.3 Key Nsight Compute Sections

```
┌─────────────────────────────────────────────────────────────┐
│  GPU Speed of Light Throughput                              │
│  ├─ SM Throughput:        45.2%     ← compute utilization   │
│  └─ Memory Throughput:    87.3%     ← memory utilization    │
│                                                             │
│  Memory Workload Analysis                                   │
│  ├─ Global Load:          1450 GB/s                         │
│  ├─ Global Store:          380 GB/s                         │
│  └─ Shared Memory:        2100 GB/s                         │
│                                                             │
│  Occupancy                                                  │
│  ├─ Theoretical:          100%  (2048/2048 threads)         │
│  ├─ Achieved:              78%  (1597/2048 threads)         │
│  └─ Limiter:              registers (64 regs/thread)        │
│                                                             │
│  Warp Stall Reasons                                         │
│  ├─ Memory Dependency:     42.3%                            │
│  ├─ Short Scoreboard:      18.1%                            │
│  ├─ Not Selected:          31.2%                            │
│  └─ Barrier:                8.4%                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Metrics to Understand

### 2.1 SM Occupancy

Occupancy = ratio of **active warps** to maximum warps per SM.

$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Max Warps per SM}}$$

For A100: max = 64 warps (2048 threads) per SM.

Occupancy limiters:
```
Resource          Per SM Limit (A100)   Example
──────────────────────────────────────────────────
Registers         65536                 64 regs/thread × 256 threads = 16384
                                       → 4 blocks → 1024 threads → 50%
Shared Memory     164 KB               48 KB/block → 3 blocks → depends
Block size        max 1024 threads      1024 → 1 block → 32 warps → 50%
```

> **Key insight**: High occupancy helps hide latency but isn't always necessary. A kernel at 50% occupancy can still saturate memory bandwidth if each thread does enough independent loads.

### 2.2 Achieved FLOPS and Memory Throughput

```
Achieved FLOPS    = total floating-point ops / kernel time
Achieved BW       = total bytes loaded+stored / kernel time

A100 peaks:
  FP32:   19.5 TFLOPS
  FP16:   312  TFLOPS (with Tensor Cores)
  BW:     2039 GB/s (HBM2e)
```

### 2.3 Warp Stall Reasons

| Stall Reason | Meaning | Fix |
|-------------|---------|-----|
| Memory Dependency | Waiting for global memory load | Increase occupancy, prefetch, use shared memory |
| Short Scoreboard | Waiting for shared/L1/constant memory | Reduce bank conflicts, pipeline loads |
| Not Selected | Warp is eligible but scheduler picked another | Generally OK — means latency is hidden |
| Barrier | Waiting at `__syncthreads()` | Reduce sync points, balance work across threads |
| Math Pipe Throttle | Compute units are saturated | You're compute-bound — reduce ops or use Tensor Cores |

---

## 3. The Roofline Model

### 3.1 Core Idea

Every kernel has an **arithmetic intensity** (AI):

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes accessed from DRAM}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$

The roofline model says achievable performance is:

$$P = \min\left(\text{Peak FLOPS},\quad \text{Peak BW} \times \text{AI}\right)$$

```
Performance
(GFLOPS)
    │
    │              ╱ Peak FLOPS = 19500 GFLOPS (flat ceiling)
    │─────────────╱──────────────────────────────
    │           ╱│
    │         ╱  │
    │       ╱    │    ← Compute-bound region
    │     ╱      │
    │   ╱  Memory│
    │ ╱   -bound │
    │╱   region  │
    ┼────────────┼──────────────────────────────
    0          Ridge                    AI (FLOP/Byte)
              Point

Ridge Point = Peak FLOPS / Peak BW
A100: 19500 / 2039 ≈ 9.6 FLOP/Byte
```

### 3.2 Where Common ML Kernels Fall

```
Kernel                    AI (FLOP/Byte)   Regime
───────────────────────────────────────────────────
Vector addition           0.25             Memory-bound
Softmax                   ~1               Memory-bound
Layer normalization       ~2               Memory-bound
Attention (long seq)      ~4-8             Mixed
GEMM (large M,N,K)       ~64-256          Compute-bound
Convolution (3×3)         ~10-40           Compute-bound
Elementwise (GELU)        0.5              Memory-bound
```

> **Critical insight**: Most ML kernels are **memory-bound**. This is why kernel fusion (combining elementwise ops) is so valuable — it eliminates intermediate memory traffic without changing the compute.

### 3.3 Calculating AI for Your Kernel

Example: Matrix multiply $C_{M \times N} = A_{M \times K} \times B_{K \times N}$

$$\text{FLOPs} = 2 \times M \times N \times K$$

$$\text{Bytes} = (M \times K + K \times N + M \times N) \times 4 \quad (\text{FP32})$$

$$\text{AI} = \frac{2MNK}{4(MK + KN + MN)}$$

For square matrices $M = N = K$:

$$\text{AI} = \frac{2N^3}{4 \cdot 3N^2} = \frac{N}{6}$$

At $N = 4096$: AI ≈ 683 FLOP/Byte → deeply **compute-bound**.

At $N = 32$: AI ≈ 5.3 FLOP/Byte → **memory-bound**!

---

## 4. Profiling Workflow

### 4.1 Step-by-Step Process

```
1. PROFILE WITH NSIGHT SYSTEMS (big picture)
   └─ Are there CPU↔GPU sync stalls? Transfer gaps?

2. IDENTIFY HOTSPOT KERNELS
   └─ Which kernel takes the most time?

3. PROFILE WITH NSIGHT COMPUTE (deep dive)
   └─ What is SM utilization? Memory throughput?

4. CALCULATE ARITHMETIC INTENSITY
   └─ Count FLOPs and bytes in your kernel

5. PLACE ON ROOFLINE
   └─ Memory-bound? → optimize memory access
   └─ Compute-bound? → reduce FLOPs or use Tensor Cores

6. CHECK STALL REASONS
   └─ Memory dependency → more ILP, prefetch
   └─ Bank conflicts → pad shared memory
   └─ Low occupancy → reduce register pressure

7. OPTIMIZE AND RE-PROFILE
   └─ Always measure, never assume
```

### 4.2 Python Script for Roofline Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# A100 specs
peak_flops = 19500    # GFLOPS (FP32)
peak_bw    = 2039     # GB/s
ridge_point = peak_flops / peak_bw  # ~9.6 FLOP/Byte

# Roofline
ai = np.logspace(-2, 3, 1000)
perf = np.minimum(peak_flops, peak_bw * ai)

plt.figure(figsize=(10, 6))
plt.loglog(ai, perf, 'b-', linewidth=2, label='Roofline')
plt.axvline(ridge_point, color='gray', linestyle='--', alpha=0.5,
            label=f'Ridge Point ({ridge_point:.1f})')

# Plot your kernels
kernels = {
    'vec_add':       (0.25, 480),
    'softmax':       (1.0,  1200),
    'naive GEMM':    (85,   2400),
    'tiled GEMM':    (85,   8500),
    'cuBLAS GEMM':   (85,   17800),
}

for name, (k_ai, k_perf) in kernels.items():
    plt.plot(k_ai, k_perf, 'ro', markersize=8)
    plt.annotate(name, (k_ai, k_perf), textcoords="offset points",
                 xytext=(10, 5), fontsize=9)

plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
plt.ylabel('Performance (GFLOPS)')
plt.title('Roofline Model — NVIDIA A100')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('roofline.png', dpi=150)
plt.show()
```

---

## 5. Practical ncu Examples

### 5.1 Quick Occupancy Check

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    -k "my_kernel" ./my_app
```

### 5.2 Memory Throughput

```bash
ncu --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -k "my_kernel" ./my_app
```

### 5.3 Compute Throughput

```bash
ncu --metrics \
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
    -k "my_kernel" ./my_app
```

### 5.4 Warp Stall Breakdown

```bash
ncu --metrics \
    smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
    smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
    smsp__warps_issue_stalled_wait_per_issue_active.ratio,\
    smsp__warps_issue_stalled_barrier_per_issue_active.ratio \
    -k "my_kernel" ./my_app
```

---

## 6. Latency Hiding & Instruction-Level Parallelism

### 6.1 How GPUs Hide Memory Latency

Global memory latency: ~400 cycles. The GPU hides this by switching to another warp:

```
Warp 0:  LOAD ──── wait 400 cycles ──── COMPUTE
Warp 1:       LOAD ──── wait 400 cycles ──── COMPUTE
Warp 2:            LOAD ──── wait 400 cycles ──── COMPUTE
...
         ↑ scheduler rotates through ready warps
```

With enough warps, the pipeline stays full. This is why occupancy matters for memory-bound kernels.

### 6.2 ILP Within a Thread

Even within a single thread, issuing **independent** instructions lets the hardware pipeline them:

```cuda
// LOW ILP — each load depends on the previous
float a = input[i];
float b = a * 2.0f;
float c = input[i + 1];  // must wait for b? No, but a blocks pipeline

// HIGH ILP — independent loads issued together
float a = input[i];
float c = input[i + 1];     // independent of 'a'
float e = input[i + 2];     // independent of 'a' and 'c'
float b = a * 2.0f;
float d = c * 2.0f;
float f = e * 2.0f;
```

---

## Hands-On Exercises

1. **Profile vec_add**: Run `ncu --set full` on the Day 3 vector addition. Record SM%, memory throughput, and occupancy. Calculate AI manually and verify it's memory-bound.

2. **Profile the transpose**: Compare the naive and shared-memory transpose kernels from Day 4 using `ncu`. Check the memory throughput improvement and bank conflict metrics.

3. **Build a roofline plot**: Use the Python script above. Add your own kernels from Days 3-4 as data points. Which ones are close to the roofline? Which have room to improve?

4. **Stall analysis**: Profile the naive reduction from Day 3. What is the dominant stall reason? Propose an optimization based on the stall breakdown.

---

## Key Takeaways

1. **Nsight Systems** for the big picture (timeline), **Nsight Compute** for kernel-level deep dives.
2. Arithmetic intensity $\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}$ determines whether a kernel is memory-bound or compute-bound.
3. The **roofline model** gives you a single plot showing how far a kernel is from hardware limits.
4. Most ML kernels are **memory-bound** — fusing elementwise ops is almost always a win.
5. The A100 ridge point is ~9.6 FLOP/Byte (FP32) — kernels below this are memory-bound.
6. **Warp stall reasons** tell you exactly what to optimize: memory dependency → more ILP/occupancy; bank conflicts → padding; barrier → fewer syncs.

---

## Further Reading

- [Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance Model" (CACM 2009)
- [NVIDIA Kernel Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
- Volkov, "Understanding Latency Hiding on GPUs" (UC Berkeley, 2016)

---

> **Tomorrow**: [Day 6 — Matrix Multiply: Naive to Tiled](day-06-matrix-multiply-naive-to-tiled.md). You'll implement the single most important kernel in all of ML — GEMM — and watch it go from 2% to 60% of cuBLAS performance through tiling and register blocking.
