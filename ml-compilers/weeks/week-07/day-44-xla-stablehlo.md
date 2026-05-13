# Day 44: XLA & StableHLO

> **Phase III · Week 7 · Day 44 of 70 · 2.5 hours**

*"The fastest code is the code that doesn't exist — XLA's job is to fuse away everything unnecessary."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 43: MLIR for ML](day-43-mlir-for-ml.md) | [Day 45: ONNX Runtime Deep Dive](day-45-onnx-runtime-deep-dive.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

**XLA** (Accelerated Linear Algebra) is the production ML compiler that powers JAX, TensorFlow, and PyTorch/XLA at Google scale. When you `jit` a JAX function, XLA is what turns your Python into fused GPU kernels. XLA introduced many ideas that other compilers adopted: operation fusion, buffer assignment, layout optimization, and algebraic simplification. **StableHLO** extracts XLA's IR (HLO) into a portable MLIR dialect, enabling models compiled for XLA to also target IREE, Torch-MLIR, and other backends. Understanding XLA teaches you how a battle-tested production compiler works and where it excels compared to TVM.

---

## 1. XLA Architecture Overview

### The XLA Compilation Pipeline

```
XLA Compilation Pipeline
═════════════════════════

  JAX / TensorFlow / PyTorch-XLA
         │
    ┌────▼──────────────────┐
    │  HLO (High-Level Ops) │  ← Framework-independent IR
    │  • MatMul, Conv, Reduce│     ~150 operations
    │  • Shape inference     │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  HLO Optimization      │  ← Platform-independent passes
    │  • Algebraic simp.     │     (~100 passes)
    │  • Op fusion           │
    │  • Common subexpr.     │
    │  • Layout assignment   │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  Buffer Assignment     │  ← Memory planning
    │  • Live range analysis │
    │  • Buffer reuse        │
    │  • Alias analysis      │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  Code Generation       │  ← Platform-specific
    │  ┌────────┬──────────┐ │
    │  │GPU     │CPU       │ │
    │  │(LLVM→  │(LLVM→    │ │
    │  │ PTX)   │ native)  │ │
    │  └────────┴──────────┘ │
    └────────────────────────┘
```

### HLO: The Core IR

HLO (High-Level Operations) is XLA's main IR. It's a dataflow graph of ~150 operations:

```
// HLO text format for: y = relu(matmul(A, B) + bias)

HloModule matmul_relu

ENTRY main {
  %A    = f32[64,128] parameter(0)
  %B    = f32[128,256] parameter(1)
  %bias = f32[256] parameter(2)

  // Matrix multiply
  %dot  = f32[64,256] dot(%A, %B),
            lhs_contracting_dims={1},
            rhs_contracting_dims={0}

  // Broadcast bias to match dot output shape
  %bcast = f32[64,256] broadcast(%bias), dimensions={1}

  // Add bias
  %add  = f32[64,256] add(%dot, %bcast)

  // ReLU = max(0, x)
  %zero = f32[] constant(0)
  %zero_bcast = f32[64,256] broadcast(%zero)
  ROOT %relu = f32[64,256] maximum(%add, %zero_bcast)
}
```

### HLO Operation Categories

| Category | Examples | Count |
|:--|:--|:--|
| Element-wise | `add`, `multiply`, `exp`, `maximum` | ~40 |
| Reduction | `reduce`, `reduce-window` | ~5 |
| Data movement | `broadcast`, `transpose`, `reshape`, `gather`, `scatter` | ~15 |
| Linear algebra | `dot`, `convolution`, `triangular-solve` | ~10 |
| Control flow | `conditional`, `while`, `call` | ~5 |
| Communication | `all-reduce`, `all-gather`, `collective-permute` | ~10 |
| Custom | `custom-call` (escape hatch to libraries) | 1 |

---

## 2. XLA's Key Optimization Passes

### Operator Fusion — XLA's Superpower

XLA's fusion is aggressive and rule-based. It classifies ops and fuses them to eliminate memory round-trips:

```
Fusion Example: matmul + bias + relu → one kernel
══════════════════════════════════════════════════

  Before fusion (3 kernels, 2 intermediate buffers):

  ┌──────┐    buf1    ┌──────┐    buf2    ┌──────┐
  │ dot  │ ────────→ │ add  │ ────────→ │ max  │
  └──────┘           └──────┘           └──────┘
      3 kernel launches
      2 × 64 × 256 × 4 = 128 KB wasted memory traffic

  After fusion (1 kernel, 0 intermediate buffers):

  ┌─────────────────────────────┐
  │  fused_computation {        │
  │    %dot  = dot(A, B)        │
  │    %add  = add(%dot, bias)  │  ← All in registers
  │    %relu = max(%add, 0)     │
  │  }                          │
  └─────────────────────────────┘
      1 kernel launch, data stays in registers/shared memory
```

### Fusion Categories in XLA

```python
# XLA classifies ops for fusion decisions:

# 1. kLoop: element-wise ops that iterate over output shape
#    add, multiply, exp, tanh, broadcast, reshape
#    → Can be fused freely with other kLoop ops

# 2. kInput: ops that iterate over an input shape (reductions)
#    reduce, reduce-window
#    → Consumer of reduction can fuse in

# 3. kOutput: ops that define the output (like scatter)
#    → Producers can fuse into kOutput

# Fusion rule: merge ops when it reduces memory traffic
# without creating too-large kernels
```

### Algebraic Simplification

```
XLA Algebraic Rewrites (selected examples):
════════════════════════════════════════════

  x + 0         → x
  x * 1         → x
  x * 0         → broadcast(0)
  x - x         → broadcast(0)
  max(relu(x))  → relu(x)
  transpose(transpose(x, p), p⁻¹) → x
  reshape(reshape(x, s1), s2) → reshape(x, s2)
  broadcast(scalar) + broadcast(scalar) → broadcast(scalar + scalar)
  dot(A, identity) → A
  slice(concat(a,b), [0:len_a]) → a
```

### Buffer Assignment

XLA solves the **memory allocation problem** for the entire computation graph:

$$\text{minimize } \sum_{i} \text{size}(buffer_i) \quad \text{s.t. } \text{live}(b_i) \cap \text{live}(b_j) = \emptyset \implies b_i, b_j \text{ can share}$$

```
Buffer Reuse Example
════════════════════

  Timeline:  ────t1────t2────t3────t4────t5────

  buf_A:     [████████████]                    (live t1-t2)
  buf_B:     [████████████████████]            (live t1-t3)
  buf_C:              [████████████]           (live t2-t3)
  buf_D:                       [████████████]  (live t3-t5)

  Allocation plan (3 buffers → 2 physical):
  physical_0:  buf_A ──────── buf_D           (reuse after A dies)
  physical_1:  buf_B ──────── (freed)
  physical_2:  buf_C ──────── (freed)

  Result: peak memory reduced by ~33%
```

---

## 3. JAX + XLA: The Compilation Flow

### Tracing a JAX Computation Through XLA

```python
import jax
import jax.numpy as jnp

@jax.jit
def layer(x, w, b):
    """Linear + ReLU layer."""
    return jax.nn.relu(x @ w + b)

# Create inputs
x = jnp.ones((4, 8))
w = jnp.ones((8, 16))
b = jnp.ones((16,))

# First call: traces, compiles via XLA, caches
result = layer(x, w, b)

# Inspect the HLO
hlo = jax.xla_computation(layer)(x, w, b)
print(hlo.as_hlo_text())  # Human-readable HLO

# Get optimized HLO (after all passes)
compiled = jax.jit(layer).lower(x, w, b).compile()
print(compiled.as_text())  # Optimized + fused HLO
```

### Examining XLA's Decisions

```python
# See which ops XLA fused together
import jax

def matmul_gelu(x, w):
    h = x @ w
    return h * 0.5 * (1.0 + jax.lax.erf(h / jnp.sqrt(2.0)))

lowered = jax.jit(matmul_gelu).lower(
    jnp.ones((32, 64)), jnp.ones((64, 128))
)

# Print HLO with fusion boundaries marked
compiled = lowered.compile()

# Cost analysis: see memory and FLOP estimates
analysis = compiled.cost_analysis()
print(f"FLOPs: {analysis[0].get('flops', 'N/A')}")
print(f"Bytes accessed: {analysis[0].get('bytes accessed', 'N/A')}")
```

### XLA Compilation Stages Diagram

```
jax.jit(f)(x)
     │
     │  1. Abstract evaluation (shape/dtype only)
     ▼
  Jaxpr (JAX's internal trace IR)
     │
     │  2. jaxpr → HLO conversion
     ▼
  Unoptimized HLO
     │
     │  3. ~100 optimization passes
     │     • Algebraic simplification
     │     • Fusion
     │     • Layout assignment (NHWC → NCHW, etc.)
     │     • Batch normalization decomposition
     ▼
  Optimized HLO
     │
     │  4. Buffer assignment
     ▼
  HLO + memory plan
     │
     │  5. Code generation (GPU: LLVM → PTX → cubin)
     ▼
  Executable (cached for same shapes/dtypes)
```

---

## 4. StableHLO: The Portable ML Dialect

### What Is StableHLO?

StableHLO extracts XLA's HLO operations into a **versioned, portable MLIR dialect** with backward compatibility guarantees:

```
StableHLO's Position in the Ecosystem
══════════════════════════════════════

  PyTorch ──→ Torch-MLIR ──→ StableHLO ──→ XLA (GPU/TPU)
                                 │
  JAX ────→ jax2stablehlo ──→──→├──→ IREE (CPU/GPU/Vulkan)
                                 │
  TF ─────→ tf2stablehlo ──→──→└──→ Torch-MLIR (back to PyTorch)

  Key property: StableHLO is the INTERCHANGE FORMAT
  • Versioned: programs from v1.0 still work in v2.0
  • ~120 operations covering all ML needs
  • Serializable: save compiled programs to disk
```

### StableHLO Example

```mlir
// StableHLO for: relu(matmul(A, B) + bias)

func.func @matmul_relu_bias(
    %A: tensor<64x128xf32>,
    %B: tensor<128x256xf32>,
    %bias: tensor<256xf32>) -> tensor<64x256xf32> {

  // Matrix multiply
  %dot = stablehlo.dot_general %A, %B,
    contracting_dims = [1] x [0]
    : (tensor<64x128xf32>, tensor<128x256xf32>)
    -> tensor<64x256xf32>

  // Broadcast bias
  %bias_bcast = stablehlo.broadcast_in_dim %bias,
    dims = [1] : (tensor<256xf32>) -> tensor<64x256xf32>

  // Add
  %add = stablehlo.add %dot, %bias_bcast
    : tensor<64x256xf32>

  // ReLU
  %zero = stablehlo.constant dense<0.0> : tensor<64x256xf32>
  %relu = stablehlo.maximum %add, %zero
    : tensor<64x256xf32>

  return %relu : tensor<64x256xf32>
}
```

### The OpenXLA Project

OpenXLA is the umbrella project unifying XLA's components as open-source:

| Component | Purpose |
|:--|:--|
| **XLA** | The compiler (HLO optimization + codegen) |
| **StableHLO** | Portable MLIR dialect for ML |
| **PJRT** | Plugin interface for hardware backends |
| **Shardy** | Automatic SPMD partitioning |
| **IREE** | Lightweight deployment runtime |

---

## 5. XLA vs TVM: When to Choose Which

### Decision Framework

```
                    ┌──────────────────────────┐
                    │  What's your framework?  │
                    └────────┬─────────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
            ┌────▼────┐           ┌──────▼──────┐
            │ JAX/TF  │           │PyTorch/ONNX │
            └────┬────┘           └──────┬──────┘
                 │                       │
          ┌──────▼──────┐        ┌───────▼───────┐
          │ XLA (native)│        │ Target HW?    │
          │ Best choice │        └──┬──────┬─────┘
          └─────────────┘           │      │
                              ┌─────▼──┐ ┌─▼──────┐
                              │Standard│ │Custom / │
                              │GPU/CPU │ │Edge/DSA │
                              └───┬────┘ └───┬─────┘
                                  │          │
                            ┌─────▼───┐ ┌───▼──────┐
                            │torch.   │ │  TVM     │
                            │compile  │ │(broadest │
                            │or ORT   │ │ HW)      │
                            └─────────┘ └──────────┘
```

### Head-to-Head Comparison

| Dimension | XLA | TVM |
|:--|:--|:--|
| **Primary use** | JAX/TF production | Cross-framework, cross-HW |
| **Fusion** | Rule-based, very aggressive | Schedule-guided |
| **Auto-tuning** | Limited (layout, tile hints) | Extensive (MetaSchedule) |
| **Hardware** | CPU, GPU, TPU | CPU, GPU, FPGA, MCU, DSA |
| **Dynamic shapes** | Bounded, recompiles per shape | Symbolic shapes (Relax) |
| **Deployment** | Tied to XLA runtime | Lightweight TVM runtime |
| **Maturity** | Production at Google scale | Production at diverse sites |
| **SPMD/sharding** | Native (GSPMD/Shardy) | Community efforts |
| **Debugging** | HLO dumps, profiler | TVMScript, debugger |

---

## 6. Hands-On Exercises

### Exercise 1: Trace XLA Compilation (30 min)

```python
import jax
import jax.numpy as jnp
import os

# Enable XLA debug output
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text'

@jax.jit
def transformer_block(x, wq, wk, wv, wo):
    """Simplified self-attention."""
    q = x @ wq  # [B, S, D]
    k = x @ wk
    v = x @ wv

    # Scaled dot-product attention
    d_k = q.shape[-1]
    scores = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    attn = weights @ v

    return attn @ wo

# Run to trigger compilation
B, S, D = 2, 16, 64
x = jnp.ones((B, S, D))
wq = wk = wv = wo = jnp.ones((D, D))
result = transformer_block(x, wq, wk, wv, wo)

# Now examine /tmp/xla_dump/ for:
# 1. module_XXXX.before_optimizations.txt  (original HLO)
# 2. module_XXXX.after_optimizations.txt   (fused HLO)
# Compare the two — count how many fused computations XLA created
```

### Exercise 2: StableHLO Round-Trip (20 min)

```python
import jax
import jax.numpy as jnp

def gelu(x):
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))

# Export to StableHLO
lowered = jax.jit(gelu).lower(jnp.ones((4, 8)))
stablehlo_text = lowered.as_text("stablehlo")
print(stablehlo_text)

# Questions to answer:
# 1. How many stablehlo ops are in the unoptimized program?
# 2. What does the sqrt(2) become? (constant folded?)
# 3. How does stablehlo.erf compare to the JAX op?
```

### Exercise 3: XLA vs Eager Benchmark (20 min)

```python
import jax
import jax.numpy as jnp
import time

def mlp(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2

# Inputs
x  = jnp.ones((256, 512))
w1 = jnp.ones((512, 1024))
b1 = jnp.ones((1024,))
w2 = jnp.ones((1024, 256))
b2 = jnp.ones((256,))

# Eager (no JIT)
jax.config.update("jax_disable_jit", True)
_ = mlp(x, w1, b1, w2, b2).block_until_ready()
t0 = time.perf_counter()
for _ in range(100):
    _ = mlp(x, w1, b1, w2, b2).block_until_ready()
eager_time = (time.perf_counter() - t0) / 100
jax.config.update("jax_disable_jit", False)

# JIT (XLA compiled)
jit_mlp = jax.jit(mlp)
_ = jit_mlp(x, w1, b1, w2, b2).block_until_ready()  # warmup
t0 = time.perf_counter()
for _ in range(100):
    _ = jit_mlp(x, w1, b1, w2, b2).block_until_ready()
jit_time = (time.perf_counter() - t0) / 100

print(f"Eager: {eager_time*1000:.2f} ms")
print(f"XLA JIT: {jit_time*1000:.2f} ms")
print(f"Speedup: {eager_time/jit_time:.1f}x")
```

---

## Key Takeaways

1. **XLA** is a production ML compiler powering JAX and TensorFlow — it compiles HLO dataflow graphs into fused GPU/TPU/CPU kernels
2. **Operator fusion** is XLA's core strength — it classifies ops (kLoop, kInput, kOutput) and aggressively merges them to eliminate memory traffic
3. **Buffer assignment** solves a global memory allocation problem, reusing buffers across the computation graph based on liveness analysis
4. **StableHLO** is the portable MLIR dialect that extracts HLO semantics with versioning guarantees — it's the interchange format for the OpenXLA ecosystem
5. **JAX ↔ XLA** is seamless: `jax.jit` traces to Jaxpr, converts to HLO, optimizes through ~100 passes, and generates target code
6. **XLA vs TVM**: XLA excels at JAX/TF with aggressive fusion on GPU/TPU; TVM excels at cross-framework, cross-hardware deployment with auto-tuning

---

## Further Reading

- [XLA Architecture (OpenXLA)](https://openxla.org/xla/architecture) — official docs
- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) — op semantics
- [JAX Internals: How JIT Works](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html) — tracing model
- [XLA Operation Semantics](https://openxla.org/xla/operation_semantics) — all HLO ops
- [PJRT Plugin Interface](https://openxla.org/xla/pjrt_integration) — hardware backend API
- [XLA GPU Fusion Heuristics (source)](https://github.com/openxla/xla/tree/main/xla/service/gpu) — implementation

---

## Tomorrow: ONNX Runtime Deep Dive

Day 45 dives into **ONNX Runtime** — the inference engine that powers production deployment at Microsoft and beyond. You'll explore its graph partitioning strategy, execution providers (CUDA, TensorRT, OpenVINO), operator fusion passes, and understand when ORT beats TVM for deployment.
