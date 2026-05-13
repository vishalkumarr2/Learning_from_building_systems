# Day 1: Why ML Needs Compilers

> **Phase I · Week 1 · Day 1 of 70 · 2.5 hours**

*"Between your Python model definition and the GPU transistors switching, there's a compiler-shaped gap. Today we learn why that gap exists."*

---

## Navigation

| | |
|---|---|
| **Previous** | — Start of curriculum |
| **Next** | [Day 2: GPU Architecture Deep Dive →](day-02-gpu-architecture.md) |
| **Week** | Week 1: GPU Architecture & CUDA |
| **Phase** | Phase I: Hardware & Compute Foundations |
| **Curriculum** | [Full Curriculum](../../curriculum.html) |

---

## Why This Matters

You write `model(x)` in Python and expect it to run fast on a GPU. But between that Python call and actual hardware execution, there's an enormous translation problem. The hardware landscape is fragmenting (NVIDIA, AMD, Apple Silicon, TPUs, custom ASICs), while model architectures are exploding in complexity. ML compilers exist to bridge this gap automatically.

By the end of this curriculum, you'll understand every layer of that translation — and know how to optimize each one.

---

## 1. The Software-Hardware Gap

### 1.1 What Happens When You Call `model(x)`

```python
import torch

model = torch.nn.Linear(768, 768)
x = torch.randn(32, 768, device='cuda')
y = model(x)   # ← What actually happens here?
```

The journey from Python to silicon:

```
Python call: model(x)
    │
    ▼
PyTorch dispatcher → selects ATen operator
    │
    ▼
ATen C++ kernel → addmm(bias, input, weight.T)
    │
    ▼
cuBLAS / cuDNN → vendor library call
    │
    ▼
CUDA PTX assembly → virtual GPU instructions
    │
    ▼
SASS → actual hardware instructions (SM-specific)
    │
    ▼
GPU Streaming Multiprocessors → transistors switch
```

Each arrow represents a **compilation or dispatch decision** that affects performance.

### 1.2 Why Not Just Use Vendor Libraries?

Vendor libraries like cuBLAS and cuDNN are heavily optimized for **standard operations**:
- Matrix multiplication → cuBLAS SGEMM
- Convolution → cuDNN's implicit GEMM or Winograd
- Batch normalization → cuDNN fused kernels

**The problem**: Real models don't run one operation at a time. They run sequences:

```python
# Common transformer pattern
x = layer_norm(x)           # cuDNN kernel, reads from global memory
x = linear(x)               # cuBLAS GEMM, reads from global memory
x = gelu(x)                 # custom kernel, reads from global memory
x = dropout(x)              # custom kernel, reads from global memory
```

Each kernel launch:
1. **Reads** input from global GPU memory (slow: ~900 GB/s on A100)
2. **Computes** the result (fast: 312 TFLOPS on A100)
3. **Writes** output back to global GPU memory (slow again)

**Memory bandwidth is the bottleneck**, not compute. For elementwise ops like GELU and dropout, you're spending 95%+ of time on memory transfers.

### 1.3 The Fusion Opportunity

A compiler can **fuse** these into a single kernel:

```
Without fusion:                        With fusion:
┌────────────┐                        ┌────────────────────────┐
│ LayerNorm  │ ← read x, write tmp1   │                        │
└────────────┘                        │  LayerNorm + Linear    │
┌────────────┐                        │  + GELU + Dropout      │
│   Linear   │ ← read tmp1, write tmp2│                        │
└────────────┘                        │  ← read x once         │
┌────────────┐                        │  → write output once   │
│    GELU    │ ← read tmp2, write tmp3│                        │
└────────────┘                        └────────────────────────┘
┌────────────┐
│  Dropout   │ ← read tmp3, write y
└────────────┘
4 kernel launches                     1 kernel launch
8 global memory round-trips           2 global memory round-trips
```

**Result**: 2-4× speedup for memory-bound sequences. This is what ML compilers do automatically.

---

## 2. The Hardware Fragmentation Problem

### 2.1 Target Diversity

A modern ML system might need to deploy to:

| Target | ISA | Memory Model | Peak FLOPS (FP16) |
|--------|-----|-------------|-------------------|
| NVIDIA A100 | CUDA/PTX | Unified (HBM2e) | 312 TFLOPS |
| NVIDIA H100 | CUDA/PTX | Unified (HBM3) | 990 TFLOPS |
| AMD MI300X | ROCm/GCN | Unified (HBM3) | 1307 TFLOPS |
| Apple M4 Max | Metal | Unified (LPDDR5) | ~28 TFLOPS |
| Google TPU v5 | XLA/HLO | HBM2e | ~459 TFLOPS |
| Intel Gaudi 3 | Synapse | HBM2e | ~1835 TFLOPS |
| ARM Cortex-A78 | NEON | Separate (LPDDR5) | ~0.05 TFLOPS |

Writing optimized kernels by hand for each target is impractical. A compiler provides **hardware abstraction** — write the model once, compile to any target.

### 2.2 The Optimization Space

For a single matmul on GPU, the optimization choices include:
- **Tile sizes**: How to partition the problem across thread blocks
- **Loop order**: Which dimension to iterate first
- **Vectorization**: How many elements per thread load
- **Shared memory usage**: What to cache in fast on-chip SRAM
- **Pipeline depth**: How to overlap compute and memory
- **Thread block dimensions**: Grid/block shape configuration

The number of valid configurations can be $10^{10}$ or more. **Auto-tuning** (searching this space) is a core ML compiler technique.

---

## 3. Compilation vs Interpretation

### 3.1 Eager Mode (Interpretation)

PyTorch default — execute operations one at a time:

```python
# Eager mode: each line dispatches immediately
x = torch.relu(x)           # dispatch → kernel → sync
x = torch.matmul(W, x)      # dispatch → kernel → sync
x = x + bias                # dispatch → kernel → sync
```

**Pros**: Easy debugging, dynamic shapes, Python control flow  
**Cons**: No cross-operator optimization, kernel launch overhead, no fusion

### 3.2 Graph Mode (Compilation)

Capture the entire computation graph, then optimize it as a whole:

```python
# Graph capture
@torch.compile
def forward(x, W, bias):
    x = torch.relu(x)
    x = torch.matmul(W, x)
    x = x + bias
    return x

# First call: trace → optimize → codegen → cache
# Subsequent calls: run optimized kernel directly
```

**Pros**: Global optimization, fusion, hardware-specific codegen  
**Cons**: Tracing limitations, recompilation on shape change, harder debugging

### 3.3 The Middle Ground

Modern ML compilers use **selective compilation**:
- Compile hot paths (attention, FFN blocks)
- Keep control flow in eager mode
- Recompile only when shapes change

`torch.compile` (PyTorch 2.x) does exactly this via TorchDynamo.

---

## 4. The ML Compiler Landscape

### 4.1 Major Projects

```
                    ┌─────────────────────┐
                    │   ML Frameworks     │
                    │ PyTorch, TF, JAX    │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌──────────────┐ ┌───────────┐ ┌───────────┐
     │ torch.compile│ │    XLA    │ │  Apache   │
     │ (Dynamo +    │ │ (Google)  │ │   TVM     │
     │  Inductor)   │ │           │ │           │
     └──────┬───────┘ └─────┬─────┘ └─────┬─────┘
            │               │             │
            ▼               ▼             ▼
     ┌──────────────┐ ┌───────────┐ ┌───────────┐
     │   Triton     │ │  LLVM /   │ │  TIR +    │
     │  (OpenAI)    │ │  MLIR     │ │  Codegen  │
     └──────┬───────┘ └─────┬─────┘ └─────┬─────┘
            │               │             │
            └───────────────┼─────────────┘
                            ▼
                    ┌──────────────┐
                    │  Hardware    │
                    │  GPU/CPU/TPU │
                    └──────────────┘
```

### 4.2 Where Each Fits

| Compiler | Frontend | Backend | Strength |
|----------|----------|---------|----------|
| torch.compile | PyTorch models | Triton (GPU), C++ (CPU) | Seamless PyTorch integration |
| XLA | JAX, TensorFlow | LLVM | TPU support, whole-program opt |
| Apache TVM | ONNX, PyTorch, TF | LLVM, CUDA, Metal, Vulkan | Universal deployment, auto-tuning |
| Triton | Python DSL | LLVM → PTX | Easy custom GPU kernels |
| MLIR | Multi-framework | Multi-target | Reusable compiler infrastructure |
| TensorRT | ONNX, TF | CUDA (NVIDIA only) | Fastest NVIDIA inference |

### 4.3 Apache TVM's Position

TVM is unique because it:
1. **Accepts multiple frontends**: ONNX, PyTorch, TensorFlow, MXNet
2. **Targets multiple backends**: NVIDIA, AMD, ARM, RISC-V, WebGPU, bare metal
3. **Auto-tunes**: Searches the optimization space per-target
4. **Open source** under Apache 2.0 with 1000+ contributors

This makes it the most **universal** ML compiler — we'll spend 3 full weeks inside it (Days 29–49).

---

## 5. Hands-On: Measuring the Gap

### Exercise 1: Unfused vs Fused Operations

```python
import torch
import time

device = 'cuda'
x = torch.randn(4096, 4096, device=device)

# Unfused: 3 separate kernels
def unfused(x):
    x = torch.relu(x)
    x = x * 0.5
    x = x + 1.0
    return x

# torch.compile fuses them
fused = torch.compile(unfused)

# Warmup
for _ in range(10):
    unfused(x)
    fused(x)
torch.cuda.synchronize()

# Benchmark
N = 100
start = time.perf_counter()
for _ in range(N):
    unfused(x)
torch.cuda.synchronize()
unfused_time = (time.perf_counter() - start) / N

start = time.perf_counter()
for _ in range(N):
    fused(x)
torch.cuda.synchronize()
fused_time = (time.perf_counter() - start) / N

print(f"Unfused: {unfused_time*1000:.2f} ms")
print(f"Fused:   {fused_time*1000:.2f} ms")
print(f"Speedup: {unfused_time/fused_time:.2f}x")
```

### Exercise 2: Profile the Difference

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        unfused(x)
        torch.cuda.synchronize()

print("=== Unfused ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        fused(x)
        torch.cuda.synchronize()

print("=== Fused ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**What to observe**:
- Unfused: 3 kernel launches, 3 memory round-trips
- Fused: 1 kernel launch (Triton-generated), 1 memory round-trip
- The fused version should be 2-3× faster for this memory-bound pattern

---

## 6. Key Takeaways

1. **The gap**: Between Python `model(x)` and GPU hardware, there are 6+ translation layers
2. **Memory bandwidth** is the bottleneck for most ML workloads, not compute
3. **Operator fusion** eliminates redundant memory transfers — this is the #1 compiler optimization
4. **Hardware fragmentation** makes hand-tuned kernels impractical — compilers provide portability
5. **The landscape**: torch.compile, XLA, TVM, Triton, TensorRT each solve different parts of the problem
6. **TVM** is the most universal — we'll go deep on it in Phase III

---

## Further Reading

- [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794) — Foundational survey paper
- [torch.compile: the missing manual](https://docs.google.com/document/d/1y811KlMlUbNEGxMerO9wpAKJ4BjqtVRGFsy0CLXB0tQ/) — PyTorch team's guide
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799) — Original TVM paper (OSDI 2018)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Triton's design paper

---

## Tomorrow

Day 2 dives into **GPU architecture** — streaming multiprocessors, warp scheduling, and the memory hierarchy that makes fusion so important.
