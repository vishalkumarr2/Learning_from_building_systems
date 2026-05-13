# Day 13: Operator Fusion Fundamentals

> **Phase I · Week 2 · Day 13 of 70 · 2.5 hours**

*"The fastest memory access is the one that never happens."*

---

| ← Previous | Next → | 📅 Week | �phase Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 12: Eager vs Graph Mode](day-12-eager-vs-graph-mode.md) | [Day 14: Stop & Reflect #1](day-14-stop-and-reflect-1.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

On modern GPUs, **arithmetic is essentially free** — the bottleneck is moving data between global memory (HBM) and compute cores (SMs). A single A100 can perform 312 TFLOPS of FP16 math but only transfer 2 TB/s of memory. That means for element-wise operations like `relu(x * 2 + bias)`, we spend more time *loading and storing* tensors than *computing* them. **Operator fusion** eliminates redundant memory traffic by combining multiple operations into a single kernel, often delivering 2–5× speedups for memory-bound workloads.

---

## 1. The Memory Wall: Why Fusion Matters

### Arithmetic Intensity

The ratio of compute to memory access determines whether an operation is **compute-bound** or **memory-bound**:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

| Operation | FLOPs/element | Bytes/element (FP32) | Arith. Intensity | Bound |
|:--|:--|:--|:--|:--|
| ReLU | 1 | 8 (read + write) | 0.125 | Memory |
| Element-wise add | 1 | 12 (2 reads + 1 write) | 0.083 | Memory |
| GELU | ~10 | 8 | 1.25 | Memory |
| MatMul (large) | $2N$ per element | $O(1/N)$ amortized | $O(N)$ | Compute |
| LayerNorm | ~5 | 8 + reduction | ~0.5 | Memory |

For an A100 (312 TFLOPS FP16, 2 TB/s bandwidth):

$$\text{Ridge point} = \frac{312 \times 10^{12}}{2 \times 10^{12}} = 156 \text{ FLOPs/byte}$$

Any operation below 156 FLOPs/byte is **memory-bound**. That includes nearly every activation function, normalization, and element-wise operation.

### Unfused vs Fused: Memory Traffic Analysis

Consider `y = relu(x * 2 + bias)` on a tensor of $N$ FP32 elements:

```
UNFUSED (3 separate kernels):

Kernel 1: mul         Kernel 2: add         Kernel 3: relu
─────────────────     ─────────────────     ─────────────────
Read  x    (4N B)     Read  t1   (4N B)     Read  t2   (4N B)
Read  2    (4  B)     Read  bias (4N B)     Write y    (4N B)
Write t1   (4N B)     Write t2   (4N B)
                                             
Total: 4N + 4N        Total: 4N + 4N + 4N   Total: 4N + 4N
     = 8N bytes            = 12N bytes            = 8N bytes

Grand total: 28N bytes transferred
             3 kernel launches (~15-45 μs overhead)
```

```
FUSED (1 kernel):

Kernel: fused_mul_add_relu
─────────────────────────
Read  x    (4N B)
Read  bias (4N B)
Write y    (4N B)
                    
Total: 12N bytes transferred
       1 kernel launch (~5-15 μs overhead)
```

$$\text{Memory savings} = 1 - \frac{12N}{28N} = 57\%$$

For $N = 1M$ elements: unfused reads/writes **112 MB**, fused reads/writes **48 MB**.

---

## 2. Fusion Categories

### 2.1 Vertical Fusion (Producer-Consumer)

Fuses operations that form a **chain** — the output of one feeds directly into the next:

```
BEFORE (vertical chain):          AFTER (fused):

┌──────┐                          ┌──────────────────┐
│ Add  │ → write to HBM           │                  │
└──┬───┘                          │  Fused kernel:   │
   │ read from HBM                │  add → mul → relu│
┌──┴───┐                          │                  │
│ Mul  │ → write to HBM           │  (intermediates  │
└──┬───┘                          │   stay in regs)  │
   │ read from HBM                │                  │
┌──┴───┐                          └──────────────────┘
│ ReLU │ → write to HBM
└──────┘

6 HBM accesses                    2 HBM accesses
                                  (1 read input, 1 write output)
```

**Rule:** Fusable when the downstream op reads exactly what the upstream op wrote, element-by-element.

### 2.2 Horizontal Fusion (Independent Operations)

Fuses operations that have **no data dependency** but operate on the same input:

```
BEFORE:                           AFTER:

     ┌───┐                             ┌───┐
     │ x │                             │ x │
     └─┬─┘                             └─┬─┘
    ┌──┴──┐                              │
    │     │                        ┌─────┴─────┐
 ┌──┴──┐ ┌┴───┐                   │  Fused:   │
 │sin  │ │cos │     ──────►        │ sin + cos │
 └──┬──┘ └┬───┘                   │(1 read,   │
    │     │                        │ 2 writes) │
    ▼     ▼                        └─────┬─────┘
   y1    y2                           ┌──┴──┐
                                      ▼     ▼
 2 reads of x                       y1    y2
                                  
                                   1 read of x
```

**Rule:** Fusable when independent ops share input tensors or have compatible iteration domains.

### 2.3 Element-wise Fusion (Pointwise)

The most common and easiest fusion — all operations are element-wise with identical shapes:

```python
# All of these can be fused into a single kernel:
def pointwise_chain(x, w, b):
    y = x * w        # element-wise mul
    y = y + b        # element-wise add
    y = y.relu()     # element-wise relu
    y = y * 0.5      # element-wise scale
    return y
```

### 2.4 Reduction Fusion

Fusing element-wise ops with a final reduction (sum, mean, max):

```python
# Fuse: element-wise chain + reduction
def fused_loss(logits, targets):
    diff = logits - targets    # element-wise
    sq = diff * diff           # element-wise
    return sq.mean()           # reduction
    
# Without fusion: 3 kernels + 3 HBM round-trips
# With fusion: element-wise part fused, reduction may be separate
```

---

## 3. Fusion Rules and Constraints

Not all operations can be fused. Key constraints:

### What CAN Be Fused

| Pattern | Example | Why It Works |
|:--|:--|:--|
| Pointwise → Pointwise | `relu(x + y)` | Same iteration domain, no deps |
| Pointwise → Reduction | `(x * y).sum()` | Element-wise feeds reduction |
| Matmul → Pointwise | `relu(A @ B)` | Epilogue fusion in CUTLASS/cuBLAS |
| Conv → BN → ReLU | Common in CNNs | Classic inference fusion |

### What CANNOT Be Fused (Easily)

| Pattern | Example | Why It Fails |
|:--|:--|:--|
| Reduction → Pointwise | `x / x.sum()` | Reduction must complete before division |
| Different shapes | `x[::2] + y` | Mismatched iteration domains |
| Data-dependent indexing | `x[indices] + y` | Irregular memory access |
| Cross-device ops | CPU tensor + CUDA tensor | Different memory spaces |

### The Fusion Barrier: Reductions

```
Can fuse INTO a reduction:      Cannot fuse ACROSS a reduction:

x ──► mul ──► add ──► sum       x ──► sum ──► mul ──► add
      │        │        │             │ BARRIER │        │
      └── fuse ┘   separate           └───────┘   separate
                                 (sum must complete before mul)
```

---

## 4. Fusion in `torch.compile` (TorchInductor)

Let's see fusion in action with `torch.compile`:

```python
import torch

def unfused(x, y):
    a = x + y
    b = a * 2.0
    c = b.relu()
    d = c - 1.0
    return d.sigmoid()

x = torch.randn(4096, 4096, device='cuda')
y = torch.randn(4096, 4096, device='cuda')

# Compile with Inductor backend
compiled = torch.compile(unfused)
result = compiled(x, y)

# To see the generated Triton code:
# TORCH_LOGS="output_code" python script.py
```

### Generated Triton Code (simplified)

When you run with `TORCH_LOGS="output_code"`, TorchInductor produces something like:

```python
@triton.jit
def fused_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    """All 5 ops fused into a single Triton kernel."""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    # Load inputs ONCE from HBM
    x = tl.load(in_ptr0 + xindex, xmask)
    y = tl.load(in_ptr1 + xindex, xmask)
    
    # All computation in registers — no HBM round-trips
    a = x + y           # add
    b = a * 2.0          # mul
    c = tl.maximum(b, 0) # relu
    d = c - 1.0          # sub
    e = tl.sigmoid(d)    # sigmoid
    
    # Store result ONCE to HBM
    tl.store(out_ptr0 + xindex, e, xmask)
```

**Result:** 5 ops → 1 kernel. Memory traffic reduced from ~10 reads/writes to 2 (one read per input, one write for output).

---

## 5. Fusion Across Compiler Stacks

Different ML compilers approach fusion with different strategies:

### 5.1 TorchInductor (PyTorch)

```
FX Graph → Lowering to ATen → Fusion passes → Triton/C++ codegen

Strategy: "Pointwise fusion + scheduling"
- Fuses all element-wise ops greedily
- Separate Triton kernel per fusion group
- Matmul epilogue fusion via extern kernels
```

### 5.2 XLA (JAX / TensorFlow)

```
HLO Graph → Fusion passes → LLVM IR → PTX

Strategy: "kFusion / kLoop / kInput fusion"
- kLoop: fuse ops with compatible loop nesting
- kInput: fuse ops reading from same input
- kFusion: general producer-consumer fusion
- More aggressive than Inductor for reductions
```

### 5.3 TVM (Apache TVM)

```
Relay IR → TE (Tensor Expressions) → Schedule → CUDA/LLVM

Strategy: "Schedule-based fusion with compute_at"
- Manual or auto-scheduled fusion via compute_at primitive
- Finer control over tiling and fusion boundaries
- AutoTVM / Ansor for automated schedule search
```

### Comparison Table

```
┌───────────────┬──────────────┬──────────────┬──────────────┐
│               │ TorchInductor│     XLA      │     TVM      │
├───────────────┼──────────────┼──────────────┼──────────────┤
│ Fusion scope  │ Pointwise +  │ Broad        │ Schedule-    │
│               │ epilogue     │ HLO fusion   │ driven       │
│ Codegen       │ Triton/C++   │ LLVM→PTX     │ LLVM/CUDA    │
│ Reduction     │ Separate     │ Fused with   │ Manual via   │
│ fusion        │ kernel       │ kInput       │ compute_at   │
│ Dynamic shape │ ✓ Guards     │ ✗ Recompile  │ ✗ Recompile  │
│ User control  │ Low          │ Low          │ High         │
│ Ease of use   │ ★★★★★        │ ★★★★☆        │ ★★☆☆☆        │
└───────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 6. Measuring Fusion Impact

### The Roofline Model

Fusion shifts an operation's position on the **Roofline plot** by increasing arithmetic intensity:

```
                    Roofline Model (A100)
Performance         ┌────────────────────────────
(TFLOPS)       312 │             ─────────────── compute ceiling
                   │            /
                   │           /
                   │          /   ← memory bandwidth ceiling
                   │         /       (2 TB/s)
                   │        /
                   │  ★    /    ★★  
                   │ (un- /   (fused)
                   │ fused)
                   │    /
                   └──/──────────────────────────
                   0    Arithmetic Intensity (FLOPs/byte)
                        │         │
                     0.125      0.625
                     (relu)    (fused 5-op chain)
```

Fusing 5 element-wise ops:
- **Unfused:** Each op reads/writes HBM → AI ≈ 0.125 each
- **Fused:** One read, one write for 5 ops → AI ≈ 0.625 → closer to roofline

---

## Hands-On Exercises

### Exercise 1: Quantify Fusion Savings (20 min)

```python
import torch
import torch.utils.benchmark as benchmark

def unfused_chain(x):
    """5 separate kernels."""
    x = x + 1.0
    x = x * 2.0
    x = x.relu()
    x = x - 0.5
    x = x.sigmoid()
    return x

compiled_chain = torch.compile(unfused_chain)

sizes = [1024, 4096, 8192, 16384]
for N in sizes:
    x = torch.randn(N, N, device='cuda')
    
    t_eager = benchmark.Timer(
        stmt='fn(x)', globals={'fn': unfused_chain, 'x': x}
    ).blocked_autorange()
    
    t_compiled = benchmark.Timer(
        stmt='fn(x)', globals={'fn': compiled_chain, 'x': x}
    ).blocked_autorange()
    
    print(f"N={N:>5}: eager={t_eager.median*1e3:.2f}ms  "
          f"compiled={t_compiled.median*1e3:.2f}ms  "
          f"speedup={t_eager.median/t_compiled.median:.2f}×")
```

### Exercise 2: Inspect Generated Code (15 min)

```bash
# See what Triton code TorchInductor generates
TORCH_LOGS="output_code" python -c "
import torch

@torch.compile
def fused(x, y):
    return ((x + y) * 2).relu().sigmoid()

fused(torch.randn(1024, 1024, device='cuda'),
      torch.randn(1024, 1024, device='cuda'))
" 2>&1 | grep -A 30 '@triton.jit'
```

### Exercise 3: Find the Fusion Barrier (10 min)

```python
# Which of these CAN the compiler fuse into one kernel?
# Think first, then verify with TORCH_LOGS="output_code"

def case_a(x): return (x + 1).relu() * 2           # ???
def case_b(x): return x / x.sum()                   # ???
def case_c(x): return (x * x).sum().sqrt()           # ???
def case_d(x, y): return x.relu() + y.sigmoid()      # ???
```

---

## Key Takeaways

1. **Memory bandwidth** — not compute — is the bottleneck for element-wise operations
2. **Fusion eliminates intermediate memory traffic** — intermediates stay in registers/SRAM instead of round-tripping to HBM
3. **Vertical fusion** (producer-consumer chains) is the most common and impactful
4. **Reductions are fusion barriers** — a reduction must complete before downstream ops can proceed
5. **`torch.compile` fuses aggressively** — generating Triton kernels that combine many PyTorch ops into one
6. **The Roofline model** quantifies why fusion works: it increases arithmetic intensity, moving ops toward the compute ceiling

---

## Further Reading

- Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation" (PLDI 2013)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [TorchInductor: A PyTorch Native Compiler](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)
- [Understanding GPU Memory Hierarchy for Deep Learning](https://horace.io/brrr_intro.html)

---

## Tomorrow's Teaser

**Day 14** is our first **Stop & Reflect** checkpoint. We'll build a concept map connecting everything from GPU architecture through profiling to fusion, test your understanding with a 10-question self-assessment, and verify you have the mental models needed for Phase II: Compiler Fundamentals.
