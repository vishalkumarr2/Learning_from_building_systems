# Day 33: TIR & Schedule Primitives

> **Phase III · Week 5 · Day 33 of 70 · 2.5 hours**

*"A schedule is a series of promises to the compiler: 'I guarantee this transformation preserves correctness — now make it fast.'"*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 32: Tensor Expression (TE)](day-32-tensor-expression-te.md) | [Day 34: TVM Runtime & Deployment](day-34-tvm-runtime-deployment.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Yesterday you wrote TE declarations — the *what*. Today you learn TIR (Tensor IR) — the *how*. TIR is TVM's low-level, loop-explicit intermediate representation. When you call `tvm.lower()`, your TE compute + schedule transforms into TIR: explicit loop nests, buffer allocations, and memory scope annotations. Understanding TIR is essential for three reasons: (1) it's what you debug when performance is wrong, (2) advanced schedule primitives like `compute_at`, `cache_read`, and `tensorize` are best understood by reading their TIR output, and (3) TVM's new `tvm.script` lets you write TIR directly, bypassing TE entirely for maximum control.

---

## 1. TIR as TVM's Low-Level IR

### Where TIR Sits in the Stack

```
  TE (te.compute, te.placeholder)     ← declarative, what
         │
         │  tvm.lower()
         ▼
  TIR (PrimFunc)                       ← imperative, how
         │
         │  tvm.build()
         ▼
  Target Code (LLVM IR / CUDA C / ...)
```

TIR is an **imperative** IR — it has explicit loops, conditionals, buffer loads/stores, and memory allocation. Compare this with Relay (graph-level, functional) and TE (declarative, mathematical).

### TIR Node Types

| Node | Purpose | Example |
|:--|:--|:--|
| `BufferStore` | Write to a buffer | `C[i, j] = value` |
| `BufferLoad` | Read from a buffer | `A[i, k]` |
| `For` | Loop construct | `for i in range(0, M)` |
| `Block` | Scope with bindings | Wraps computation + assertions |
| `Allocate` | Buffer allocation | Stack/shared/global memory |
| `IfThenElse` | Conditional | Guard for boundary checks |
| `SeqStmt` | Sequence of stmts | Multiple ops in order |

---

## 2. PrimFunc — The TIR Unit of Compilation

A `PrimFunc` is TIR's fundamental unit — one kernel that can be compiled to a target. It declares buffers, their shapes, dtypes, and memory scopes, then contains the loop body.

```python
import tvm
from tvm import te, tir
import numpy as np

M, N, K = 128, 128, 128
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# Create schedule and lower to TIR
s = te.create_schedule(C.op)
mod = tvm.lower(s, [A, B, C], simple_mode=True)
print(mod)
```

Output (simplified):

```
@T.prim_func
def main(A: T.Buffer((128, 128), "float32"),
         B: T.Buffer((128, 128), "float32"),
         C: T.Buffer((128, 128), "float32")):
    for i in range(128):
        for j in range(128):
            C[i, j] = T.float32(0)
            for k in range(128):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
```

### Anatomy of a PrimFunc

```
PrimFunc
├── params: [A, B, C]             # Buffer parameters
├── buffer_map: {var → Buffer}    # Shape, dtype, strides, scope
├── body: SeqStmt                 # The computation
│   └── For(i, 0, 128)
│       └── For(j, 0, 128)
│           ├── BufferStore(C, 0.0, [i, j])    # init
│           └── For(k, 0, 128)
│               └── BufferStore(C, ..., [i, j]) # accumulate
└── attrs: {"target": ..., "global_symbol": "main"}
```

---

## 3. Schedule Primitives in Detail

### 3.1 `compute_at` — Fusing Producer into Consumer

`compute_at` is the most powerful primitive. It moves a producer's computation *inside* the consumer's loop, improving locality.

```python
# Without compute_at: compute B fully, then compute C
A = te.placeholder((M, N), name="A")
B = te.compute((M, N), lambda i, j: A[i, j] + 1, name="B")
C = te.compute((M, N), lambda i, j: B[i, j] * 2, name="C")

s = te.create_schedule(C.op)
print("=== Before compute_at ===")
print(tvm.lower(s, [A, C], simple_mode=True))
# for i: for j: B[i,j] = A[i,j] + 1
# for i: for j: C[i,j] = B[i,j] * 2     ← two separate passes

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[1])  # fuse B into C's j-loop
print("=== After compute_at(C, j) ===")
print(tvm.lower(s, [A, C], simple_mode=True))
# for i: for j: B[0] = A[i,j] + 1; C[i,j] = B[0] * 2  ← fused!
```

**Memory impact**: Without fusion, `B` requires $M \times N$ storage. After `compute_at`, `B` shrinks to a single scalar — computed and consumed immediately.

```
Before compute_at:           After compute_at(C, j):
┌──────────────────┐         ┌──────────────────┐
│ for i in M:      │         │ for i in M:      │
│   for j in N:    │         │   for j in N:    │
│     B[i,j] = ... │         │     b = A[i,j]+1 │
│                  │         │     C[i,j] = b*2  │
│ for i in M:      │         └──────────────────┘
│   for j in N:    │              1 pass, scalar B
│     C[i,j] = ... │
└──────────────────┘
    2 passes, full B
```

### 3.2 `cache_read` / `cache_write` — Explicit Staging Buffers

These insert an explicit copy into a faster memory scope (shared memory on GPU, L1 scratchpad on accelerators).

```python
s = te.create_schedule(C.op)
# Stage A into shared memory before the matmul inner loop
AA = s.cache_read(A, "shared", [C])
# Stage C writes through local registers
CC = s.cache_write(C, "local")

# Typical GPU schedule pattern:
bx, tx = s[C].split(C.op.axis[0], factor=32)
by, ty = s[C].split(C.op.axis[1], factor=32)
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))
s[CC].compute_at(s[C], tx)
s[AA].compute_at(s[C], bx)
```

### 3.3 `storage_align` — Avoiding Bank Conflicts

GPU shared memory is banked (typically 32 banks, 4 bytes each). When threads in a warp access the same bank, they serialize. `storage_align` pads the allocation:

```python
# Pad shared memory by 1 element per row to avoid bank conflicts
s[AA].storage_align(AA.op.axis[0], factor=32, offset=1)
```

$$\text{Effective stride} = \lceil N / \text{factor} \rceil \times \text{factor} + \text{offset}$$

Without padding: threads 0 and 8 hit bank 0. With `offset=1`: stride becomes 33, all bank indices shift.

### 3.4 `tensorize` — Mapping to Hardware Intrinsics

`tensorize` replaces a loop nest with a hardware-specific intrinsic (e.g., NVIDIA Tensor Core `wmma`, ARM dot product, Intel VNNI).

```python
# Define a 16×16×16 matrix multiply intrinsic (Tensor Core)
def intrin_wmma_gemm():
    a = te.placeholder((16, 16), name="a", dtype="float16")
    b = te.placeholder((16, 16), name="b", dtype="float16")
    k = te.reduce_axis((0, 16), name="k")
    c = te.compute((16, 16),
                   lambda i, j: te.sum(a[i, k].astype("float32") *
                                       b[k, j].astype("float32"), axis=k))
    # ... buffer/intrinsic definition omitted for brevity
    return te.decl_tensor_intrin(c.op, ...)

# Use it in a schedule
s = te.create_schedule(C.op)
io, ii = s[C].split(C.op.axis[0], factor=16)
jo, ji = s[C].split(C.op.axis[1], factor=16)
ko, ki = s[C].split(k, factor=16)
s[C].tensorize(ii, intrin_wmma_gemm())  # replace 16×16×16 block with wmma
```

---

## 4. Reading Lowered TIR — Debugging with `tvm.lower()`

`tvm.lower()` is your primary debugging tool. It shows the TIR *after* schedule transformations but *before* target code generation.

```python
M, N = 256, 256
A = te.placeholder((M, N), name="A")
B = te.compute((M, N), lambda i, j: A[i, j] * 2.0, name="B")

s = te.create_schedule(B.op)

# Step 1: tile
io, ii = s[B].split(B.op.axis[0], factor=32)
jo, ji = s[B].split(B.op.axis[1], factor=8)
s[B].reorder(io, jo, ii, ji)

# Step 2: vectorize innermost, parallelize outermost
s[B].vectorize(ji)
s[B].parallel(io)

print(tvm.lower(s, [A, B], simple_mode=True))
```

Expected TIR output:

```
@T.prim_func
def main(A: T.Buffer((256, 256), "float32"),
         B: T.Buffer((256, 256), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    for i_outer in T.parallel(8):         # parallel over 256/32=8
        for j_outer in range(32):         # 256/8=32
            for i_inner in range(32):
                B[i_outer*32 + i_inner,
                  j_outer*8 : j_outer*8 + 8] = (
                    A[i_outer*32 + i_inner,
                      j_outer*8 : j_outer*8 + 8] * T.float32(2)
                )                          # vectorized (8-wide)
```

### Common TIR Debugging Patterns

| Symptom | TIR Check | Fix |
|:--|:--|:--|
| Slow kernel | Inner loop not vectorized | Add `s[X].vectorize(inner)` |
| GPU occupancy low | Missing `bind()` calls | Bind to `blockIdx` / `threadIdx` |
| Memory blowup | Large temporary buffers | Use `compute_at` to shrink lifetimes |
| Bank conflicts | Shared memory stride = power of 2 | `storage_align(offset=1)` |
| Correctness bug | Reduction init missing | Check `BufferStore` for init + accumulate |

---

## 5. Comparison with Halide Schedules

TVM's schedule system descends from Halide. Understanding the lineage clarifies design choices:

| Concept | Halide | TVM TE |
|:--|:--|:--|
| Compute declaration | `Func f; f(x,y) = ...` | `te.compute((M,N), lambda i,j: ...)` |
| Split | `f.split(x, xo, xi, 32)` | `xo, xi = s[f].split(x, factor=32)` |
| Reorder | `f.reorder(yo, xo, yi, xi)` | `s[f].reorder(yo, xo, yi, xi)` |
| Vectorize | `f.vectorize(xi)` | `s[f].vectorize(xi)` |
| Parallelize | `f.parallel(xo)` | `s[f].parallel(xo)` |
| Fuse into consumer | `g.compute_at(f, yo)` | `s[g].compute_at(s[f], yo)` |
| GPU binding | `f.gpu_blocks(xo).gpu_threads(xi)` | `s[f].bind(xo, thread_axis("blockIdx.x"))` |
| Hardware intrinsics | *Not built-in* | `s[f].tensorize(xi, intrin)` |
| Auto-tuning | Halide auto-scheduler | AutoTVM / MetaSchedule |

**Key difference**: Halide is primarily for image processing pipelines (stencils, multi-stage filters). TVM extends the model for neural network operators — reductions, tensor contractions, and hardware-specific intrinsics like Tensor Cores.

---

## 6. Writing TIR Directly with `tvm.script`

For advanced cases, bypass TE and write TIR directly:

```python
from tvm.script import tir as T

@T.prim_func
def matmul_tiled(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.match_buffer(b, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")

    for io, jo in T.grid(4, 4):           # outer tiles: 128/32 = 4
        for ii, ji in T.grid(32, 32):     # inner tile: 32×32
            with T.block("C"):
                vi = T.axis.spatial(128, io * 32 + ii)
                vj = T.axis.spatial(128, jo * 32 + ji)
                T.reads(A[vi, 0:128], B[0:128, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = T.float32(0)
                for k in range(128):
                    C[vi, vj] += A[vi, k] * B[k, vj]

# This is valid TIR — can be built directly
mod = tvm.IRModule({"main": matmul_tiled})
print(mod.script())
```

**When to use tvm.script over TE**:
- Irregular access patterns (sparse, ragged)
- Custom reduction strategies
- When you need explicit memory management that TE can't express
- Contributing new operators to TVM itself

---

## Hands-On Exercises

### Exercise 1: Trace the TIR (30 min)

Write a 2D convolution in TE, apply three different schedules, and compare the `tvm.lower()` output:

```python
H, W, KH, KW = 64, 64, 3, 3
data = te.placeholder((H, W), name="data")
kernel = te.placeholder((KH, KW), name="kernel")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")
conv = te.compute((H - KH + 1, W - KW + 1),
    lambda i, j: te.sum(data[i + rh, j + rw] * kernel[rh, rw],
                        axis=[rh, rw]),
    name="conv")

# Schedule A: naive (no transforms)
# Schedule B: tile 8×8, vectorize inner
# Schedule C: tile 8×8, vectorize inner, unroll reduction
# Compare TIR output for each — how do the loop nests differ?
```

### Exercise 2: compute_at Fusion Chain (30 min)

Create a three-stage pipeline: `ReLU → BatchNorm → Scale`. Fuse all stages using `compute_at` so the entire pipeline runs in a single pass with no intermediate buffers.

### Exercise 3: GPU Schedule with cache_read (30 min)

Write a GPU-targeted matmul schedule that:
1. Tiles to 32×32 thread blocks
2. Uses `cache_read` to stage A tiles into shared memory
3. Uses `cache_write` to accumulate in local registers
4. Applies `storage_align` to avoid bank conflicts
5. Lowers with `tvm.lower()` and inspect the memory scopes in TIR

---

## Key Takeaways

1. **TIR is TVM's imperative, loop-explicit IR** — the output of `tvm.lower()` and the input to target code generation
2. **PrimFunc** is the unit of compilation: buffer declarations + loop body + metadata
3. **`compute_at`** fuses producer-consumer stages, eliminating intermediate buffers and improving locality
4. **`cache_read`/`cache_write`** explicitly manage data movement between memory hierarchies (global → shared → local)
5. **`storage_align`** prevents GPU shared memory bank conflicts by adding padding
6. **`tensorize`** maps loop nests to hardware intrinsics (Tensor Cores, VNNI, NEON dot)
7. **`tvm.lower(simple_mode=True)`** is your primary debugging tool — always inspect TIR before building

---

## Further Reading

- [TIR Introduction (TVM docs)](https://tvm.apache.org/docs/arch/index.html#tir) — architectural overview of TIR
- [TVMScript Guide](https://tvm.apache.org/docs/tutorial/tvmscript.html) — writing TIR directly with `tvm.script`
- [Schedule Primitives Tutorial](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html) — official guide with examples
- [Halide: Decoupling Algorithms from Schedules](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) — the compute/schedule paper that started it all
- [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296) — the TIR paper (Feng et al., 2022)

---

## Tomorrow

**Day 34: TVM Runtime & Deployment** — You've lowered TIR into optimized loops. Now you'll learn how those loops become executable code: the TVM runtime architecture (Module, PackedFunc, NDArray, DLPack), compilation targets (LLVM, CUDA, ARM), cross-compilation for edge devices, and saving/loading compiled modules for production deployment.
