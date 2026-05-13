# Day 32: Tensor Expression (TE) Language

> **Phase III · Week 5 · Day 32 of 70 · 2.5 hours**

*"Halide's great insight: separate what you compute from how you compute it. TVM's great insight: let a machine learn the 'how'."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 31: Relay Optimization Passes](day-31-relay-optimization-passes.md) | [Day 33: TIR & Schedules](day-33-tir-schedules.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Relay handles the *graph level* — which operators to run and in what order. But **how** each operator executes — the loop nests, memory access patterns, vectorization, parallelism — is defined one layer down in the **Tensor Expression (TE)** language. TE is a domain-specific language for declaring tensor computations declaratively, then separately specifying an optimization **schedule**. This compute/schedule separation is TVM's core design principle, inherited from Halide, and it's what makes automatic tuning (AutoTVM, MetaSchedule) possible. Every operator in TVM — matmul, conv2d, softmax — is defined in TE.

---

## 1. TE Fundamentals

### The Three Primitives

```python
import tvm
from tvm import te
import numpy as np

# 1. PLACEHOLDER: declares an input tensor (like a function parameter)
A = te.placeholder((M, N), name="A", dtype="float32")

# 2. COMPUTE: declares a new tensor by specifying how each element is computed
#    lambda indices → expression
B = te.compute((M, N), lambda i, j: A[i, j] * 2.0, name="B")

# 3. REDUCE_AXIS: declares a reduction dimension
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
```

### How TE Relates to Math

TE lets you write mathematical specifications directly:

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

```python
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
```

$$Y_{i} = \text{ReLU}\left(\sum_{j=0}^{N-1} W_{i,j} \cdot X_j + b_i\right)$$

```python
j = te.reduce_axis((0, N), name="j")
linear = te.compute((M,), lambda i: te.sum(W[i, j] * X[j], axis=j) + b[i])
Y = te.compute((M,), lambda i: te.max(linear[i], 0.0))
```

### TE Is Declarative, Not Imperative

```
TE (declarative — WHAT):           Imperative (HOW):
──────────────────────────         ──────────────────────────
C = te.compute((M, N),             for i in range(M):
    lambda i, j:                       for j in range(N):
        te.sum(A[i,k] * B[k,j],           acc = 0
               axis=k))                   for k in range(K):
                                               acc += A[i,k] * B[k,j]
                                           C[i,j] = acc

The TE version says nothing about loop order, tiling,
vectorization, or parallelism — that's the schedule's job.
```

---

## 2. Writing Matmul in TE

### Naive Matmul

```python
import tvm
from tvm import te
import numpy as np

M, K, N = 1024, 1024, 1024

# Declare inputs
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")

# Declare computation
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

# Create default schedule (naive triple loop)
s = te.create_schedule(C.op)

# Lower to see the generated loops
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

Output (naive TIR):

```c
// Generated naive loop nest
for (i, 0, 1024) {
  for (j, 0, 1024) {
    C[i*1024 + j] = 0f
    for (k, 0, 1024) {
      C[i*1024 + j] = C[i*1024 + j] + A[i*1024 + k] * B[k*1024 + j]
    }
  }
}
```

### Build and Run

```python
# Build for CPU
target = tvm.target.Target("llvm -mcpu=native")
func = tvm.build(s, [A, B, C], target=target, name="matmul_naive")

# Create input data
dev = tvm.cpu(0)
a_np = np.random.randn(M, K).astype("float32")
b_np = np.random.randn(K, N).astype("float32")
c_np = np.zeros((M, N), dtype="float32")

a_tvm = tvm.nd.array(a_np, dev)
b_tvm = tvm.nd.array(b_np, dev)
c_tvm = tvm.nd.array(c_np, dev)

# Execute
func(a_tvm, b_tvm, c_tvm)

# Verify
np.testing.assert_allclose(c_tvm.numpy(), a_np @ b_np, rtol=1e-4)
print("✓ Matmul result matches numpy")

# Benchmark
evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
result = evaluator(a_tvm, b_tvm, c_tvm)
gflops = 2.0 * M * K * N / 1e9 / result.mean
print(f"Naive matmul: {result.mean*1000:.2f} ms ({gflops:.1f} GFLOPS)")
# Typical: ~500 ms, ~4 GFLOPS (very slow!)
```

---

## 3. Writing Conv2D in TE

### Direct Convolution

$$\text{Out}[n, c_o, h, w] = \sum_{c_i=0}^{C_{in}-1} \sum_{kh=0}^{K_H-1} \sum_{kw=0}^{K_W-1} \text{Data}[n, c_i, h+kh, w+kw] \cdot \text{Weight}[c_o, c_i, kh, kw]$$

```python
# Parameters
batch, in_channel, in_h, in_w = 1, 3, 224, 224
out_channel, kernel_h, kernel_w = 64, 3, 3
pad_h, pad_w = 1, 1
out_h = in_h - kernel_h + 2 * pad_h + 1  # 224
out_w = in_w - kernel_w + 2 * pad_w + 1  # 224

# Inputs
Data = te.placeholder((batch, in_channel, in_h, in_w), name="Data")
Weight = te.placeholder((out_channel, in_channel, kernel_h, kernel_w), name="Weight")

# Padded input
PaddedData = te.compute(
    (batch, in_channel, in_h + 2*pad_h, in_w + 2*pad_w),
    lambda n, c, h, w: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h < in_h + pad_h, w >= pad_w, w < in_w + pad_w),
        Data[n, c, h - pad_h, w - pad_w],
        tvm.tir.const(0.0, "float32")
    ),
    name="PaddedData"
)

# Reduction axes
rc = te.reduce_axis((0, in_channel), name="rc")
rh = te.reduce_axis((0, kernel_h), name="rh")
rw = te.reduce_axis((0, kernel_w), name="rw")

# Conv2d computation
Conv = te.compute(
    (batch, out_channel, out_h, out_w),
    lambda n, co, h, w: te.sum(
        PaddedData[n, rc, h + rh, w + rw] * Weight[co, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="Conv"
)
```

---

## 4. Schedule Primitives

The schedule defines **how** to execute the computation. Same `te.compute`, radically different performance.

### Overview of Schedule Primitives

```
Primitive       What It Does                          Analogy
───────────────────────────────────────────────────────────────────
split           Divide a loop into outer+inner        Loop tiling (one dim)
reorder         Change loop nesting order             Loop interchange
tile            Split two axes + reorder              2D tiling (combines split+reorder)
fuse            Merge two loops into one              Collapse nested loops
vectorize       Use SIMD for innermost loop           SSE/AVX intrinsics
parallel        Parallelize a loop over threads       OpenMP #pragma parallel
unroll          Unroll a loop (copy body N times)     -funroll-loops
bind            Map a loop to GPU thread/block axis   CUDA blockIdx/threadIdx
compute_at      Nest one computation inside another   Producer-consumer fusion
compute_inline  Inline a computation (no buffer)      Macro expansion
```

### split — Loop Tiling

```python
s = te.create_schedule(C.op)
i, j = C.op.axis           # the M and N loops
k = C.op.reduce_axis[0]    # the K reduction loop

# Split i into blocks of 32
io, ii = s[C].split(i, factor=32)
# Before: for i in range(1024)
# After:  for io in range(32):     ← outer: 1024/32 = 32 iterations
#             for ii in range(32): ← inner: 32 iterations

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

```c
// After split:
for (io, 0, 32) {
  for (ii, 0, 32) {
    for (j, 0, 1024) {
      C[(io*32 + ii)*1024 + j] = 0f
      for (k, 0, 1024) {
        C[(io*32 + ii)*1024 + j] += A[(io*32 + ii)*1024 + k] * B[k*1024 + j]
      }
    }
  }
}
```

### reorder — Loop Interchange

```python
s = te.create_schedule(C.op)
i, j = C.op.axis
k = C.op.reduce_axis[0]

io, ii = s[C].split(i, factor=32)
jo, ji = s[C].split(j, factor=32)
ko, ki = s[C].split(k, factor=4)

# Reorder for better cache locality
s[C].reorder(io, jo, ko, ii, ji, ki)

# Loop nest is now:
#   for io (M tiles)        ← outer tile over M
#     for jo (N tiles)      ← outer tile over N  
#       for ko (K tiles)    ← outer tile over K
#         for ii (32)       ← inner M
#           for ji (32)     ← inner N
#             for ki (4)    ← inner K (accumulate)
```

### tile — 2D Tiling (Shortcut)

```python
s = te.create_schedule(C.op)
i, j = C.op.axis

# tile = split both axes + reorder into [io, jo, ii, ji]
io, jo, ii, ji = s[C].tile(i, j, x_factor=32, y_factor=32)
```

### Why Tiling Helps: Cache Behavior

```
Without tiling (row-by-row access to B):
┌─────────────────────────┐
│ A row i  × entire B     │  ← B doesn't fit in L1 cache
│ A row i+1 × entire B    │  ← B reloaded from main memory
│ ...                     │
└─────────────────────────┘
L1 cache miss rate: HIGH (B is 4 MB, L1 is 32 KB)

With 32×32 tiling:
┌─────────────────────────┐
│ A tile  × B tile        │  ← 32×32×4 = 4 KB tiles fit in L1
│ reuse B tile for next   │  ← B tile stays in cache
│ A tile in same column   │
└─────────────────────────┘
L1 cache miss rate: LOW (working set = 3 × 4 KB = 12 KB)
```

### vectorize — SIMD Execution

```python
s = te.create_schedule(C.op)
i, j = C.op.axis
k = C.op.reduce_axis[0]

io, ii = s[C].split(i, factor=32)
jo, ji = s[C].split(j, factor=8)  # 8 = AVX2 vector width

s[C].reorder(io, jo, ii, ji)
s[C].vectorize(ji)  # innermost loop uses SIMD

# Generated code (conceptually):
#   for io, jo, ii:
#       C[..., ji:ji+8] = _mm256_fmadd_ps(A[...], B[..., ji:ji+8])
```

### parallel — Multi-Threading

```python
s[C].parallel(io)  # parallelize outermost loop over CPU cores

# Generated code:
#   #pragma omp parallel for
#   for io in range(32):  ← 32 chunks across cores
#       for jo, ii, ji, ki:
#           ...
```

---

## 5. Putting It All Together: Optimized Matmul

```python
import tvm
from tvm import te
import numpy as np

M, K, N = 1024, 1024, 1024

A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# ── Optimized schedule ──
s = te.create_schedule(C.op)

# Step 1: Tile the output dimensions
i, j = C.op.axis
k_ax = C.op.reduce_axis[0]
io, ii = s[C].split(i, factor=32)
jo, ji = s[C].split(j, factor=32)

# Step 2: Split reduction for better reuse
ko, ki = s[C].split(k_ax, factor=4)

# Step 3: Reorder for locality
s[C].reorder(io, jo, ko, ii, ji, ki)

# Step 4: Vectorize innermost spatial loop
jii, jiv = s[C].split(ji, factor=8)
s[C].vectorize(jiv)

# Step 5: Parallelize outermost loop
s[C].parallel(io)

# Step 6: Unroll the reduction inner loop
s[C].unroll(ki)

# Build and benchmark
target = tvm.target.Target("llvm -mcpu=native")
func = tvm.build(s, [A, B, C], target=target, name="matmul_optimized")

dev = tvm.cpu(0)
a = tvm.nd.array(np.random.randn(M, K).astype("float32"), dev)
b = tvm.nd.array(np.random.randn(K, N).astype("float32"), dev)
c = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
result = evaluator(a, b, c)
gflops = 2.0 * M * K * N / 1e9 / result.mean
print(f"Optimized matmul: {result.mean*1000:.2f} ms ({gflops:.1f} GFLOPS)")

# Verify correctness
np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-4)
```

### Performance Comparison

```
Schedule                  Time (ms)    GFLOPS    Speedup
──────────────────────────────────────────────────────────
Naive (triple loop)         ~500        ~4.0       1.0×
+ Tiling (32×32)            ~150       ~14.3       3.3×
+ Reorder (locality)         ~90       ~23.9       5.6×
+ Vectorize (AVX2)           ~40       ~53.7      12.5×
+ Parallel (8 cores)          ~8      ~268.4      62.5×
+ Unroll (ki)                 ~6      ~357.9      83.3×
──────────────────────────────────────────────────────────
NumPy (OpenBLAS):            ~3       ~716         —
Peak (8 cores × AVX2):       —       ~922         —
```

---

## 6. Lowering TE to TIR

When you call `tvm.lower()` or `tvm.build()`, TE compute + schedule are converted to TIR:

```python
# See the TIR output
tir_mod = tvm.lower(s, [A, B, C], simple_mode=True)
print(tir_mod)
```

### The Lowering Process

```
TE Compute + Schedule
         │
         │  tvm.lower()
         ▼
┌─────────────────────────────────────┐
│  TIR (TensorIR)                     │
│  • Explicit loops (for, while)      │
│  • Buffer declarations              │
│  • Index arithmetic                 │
│  • SIMD intrinsics (vectorize)      │
│  • Thread annotations (parallel)    │
└──────────────┬──────────────────────┘
               │  tvm.build()
               ▼
┌─────────────────────────────────────┐
│  Target Code                        │
│  • LLVM IR → x86/ARM assembly       │
│  • CUDA C → PTX → SASS             │
│  • OpenCL C → SPIR-V               │
│  • C source (for microTVM)          │
└─────────────────────────────────────┘
```

### Inspecting Generated Code

```python
# For LLVM targets, get the assembly
func = tvm.build(s, [A, B, C], target="llvm -mcpu=native")
print(func.get_source("asm"))  # x86 assembly

# For CUDA targets, get the PTX
func_gpu = tvm.build(s_gpu, [A, B, C], target="cuda")
print(func_gpu.imported_modules[0].get_source())  # CUDA C source
```

---

## Hands-On Exercises

### Exercise 1: Implement Element-wise Add + ReLU in TE

```python
import tvm
from tvm import te
import numpy as np

N = 1024 * 1024

A = te.placeholder((N,), name="A")
B = te.placeholder((N,), name="B")

# TODO: Define C = relu(A + B) using te.compute
# Hint: use tvm.te.max(expr, 0.0) for relu
C = te.compute((N,), lambda i: te.max(A[i] + B[i], tvm.tir.const(0.0, "float32")), name="C")

# Create schedule, vectorize, and parallelize
s = te.create_schedule(C.op)
io, ii = s[C].split(C.op.axis[0], factor=8)
s[C].vectorize(ii)
s[C].parallel(io)

# Build, run, verify against numpy
func = tvm.build(s, [A, B, C], target="llvm -mcpu=native")
dev = tvm.cpu(0)
a = np.random.randn(N).astype("float32")
b = np.random.randn(N).astype("float32")
expected = np.maximum(a + b, 0.0)

a_tvm = tvm.nd.array(a, dev)
b_tvm = tvm.nd.array(b, dev)
c_tvm = tvm.nd.array(np.zeros(N, dtype="float32"), dev)
func(a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), expected, rtol=1e-5)
print("✓ Fused add+relu correct")
```

### Exercise 2: Explore Schedule Impact on Matmul

```python
import tvm
from tvm import te
import numpy as np

M, K, N = 512, 512, 512
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

target = tvm.target.Target("llvm -mcpu=native")
dev = tvm.cpu(0)

configs = [
    ("Naive",            lambda s, C: None),
    ("Tile 16",          lambda s, C: s[C].tile(*C.op.axis, 16, 16)),
    ("Tile 32",          lambda s, C: s[C].tile(*C.op.axis, 32, 32)),
    ("Tile 64",          lambda s, C: s[C].tile(*C.op.axis, 64, 64)),
    ("Tile 32 + vec",    lambda s, C: (
        s[C].tile(*C.op.axis, 32, 32),
        s[C].vectorize(s[C].op.axis[-1])  # won't work directly after tile
    )),
]

# Benchmark each configuration and compare
for name, apply_schedule in configs:
    s = te.create_schedule(C.op)
    try:
        apply_schedule(s, C)
    except Exception:
        pass
    func = tvm.build(s, [A, B, C], target=target)
    a = tvm.nd.array(np.random.randn(M, K).astype("float32"), dev)
    b = tvm.nd.array(np.random.randn(K, N).astype("float32"), dev)
    c = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
    
    ev = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
    t = ev(a, b, c).mean
    gflops = 2.0 * M * K * N / 1e9 / t
    print(f"  {name:20s}: {t*1000:8.2f} ms  ({gflops:6.1f} GFLOPS)")
```

### Exercise 3: Visualize Loop Transformations

```python
M, K, N = 8, 8, 8  # small for readable output

A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# Print TIR at each stage
s1 = te.create_schedule(C.op)
print("=== Naive ===")
print(tvm.lower(s1, [A, B, C], simple_mode=True))

s2 = te.create_schedule(C.op)
io, ii = s2[C].split(C.op.axis[0], factor=4)
print("\n=== After split(i, 4) ===")
print(tvm.lower(s2, [A, B, C], simple_mode=True))

s3 = te.create_schedule(C.op)
io, ii = s3[C].split(C.op.axis[0], factor=4)
jo, ji = s3[C].split(C.op.axis[1], factor=4)
s3[C].reorder(io, jo, ii, ji)
print("\n=== After split + reorder (tiled) ===")
print(tvm.lower(s3, [A, B, C], simple_mode=True))

# Compare the loop structures visually
```

---

## Key Takeaways

1. **TE is a declarative DSL** — `te.compute()`, `te.placeholder()`, and `te.reduce_axis()` specify *what* to compute, not *how*
2. **Compute/schedule separation** means the same mathematical definition can be optimized completely differently for different hardware
3. **Schedule primitives** (`split`, `reorder`, `tile`, `vectorize`, `parallel`, `unroll`, `bind`) transform loop nests without changing correctness
4. **Tiling** is the single most important optimization — it converts cache-thrashing access patterns into cache-friendly block computations
5. **TE lowers to TIR** — the explicit loop-based IR that maps directly to target code (LLVM IR, CUDA C, etc.)
6. The schedule space is **exponentially large** — that's why TVM uses ML-driven search (AutoTVM, MetaSchedule) instead of hand-tuning, which we'll explore in Week 6

---

## Further Reading

- [Halide: A Language for Image Processing](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) — the paper that inspired TE's compute/schedule design (Ragan-Kelley et al., 2013)
- [TVM TE Tutorial](https://tvm.apache.org/docs/how_to/work_with_schedules/index.html) — official schedule primitives guide
- [TVM Tensor Expression API](https://tvm.apache.org/docs/reference/api/python/te.html) — Python API reference
- [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166) — AutoTVM paper showing how ML searches the schedule space

---

## Tomorrow

**Day 33: TIR & Schedules** — We'll go one layer deeper into TensorIR, TVM's low-level loop-based representation. You'll see how TIR represents the loop nests that TE + schedule produce, learn the `tvm.script` syntax for writing TIR directly, and understand how TIR maps to final target code.
