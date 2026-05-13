# Day 15: Compiler 101 for ML Engineers

> **Phase II · Week 3 · Day 15 of 70 · 2.5 hours**

*"A compiler is a program that translates one language into another — and in ML, the source language is a computation graph and the target is a specific piece of silicon."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 14: Stop & Reflect 1](../week-02/day-14-stop-and-reflect-1.md) | [Day 16: Computation Graphs as IR](day-16-computation-graphs-ir.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Every time you call `model(x)` in PyTorch, a hidden pipeline transforms your Python-level math into GPU machine code. Understanding this pipeline — **frontend → IR → optimization → code generation** — is the difference between blindly hoping for speed and systematically engineering it. ML compilers are the reason a 70B-parameter model can run on a phone, and why the same model code can target NVIDIA, AMD, Apple, and custom TPU silicon without rewrites.

---

## 1. What Is a Compiler?

A compiler is a **structured translator** that converts a program from a source representation to a target representation, applying correctness-preserving transformations along the way.

```
Source Program → [Frontend] → IR → [Optimizer] → IR' → [Backend] → Target Code
```

The key insight: by splitting into phases with a **common intermediate representation (IR)**, you decouple the $M$ source languages from the $N$ target architectures, reducing the problem from $O(M \times N)$ to $O(M + N)$.

```
                    Traditional Compiler Architecture
  ┌──────────┐
  │    C      │──┐
  ├──────────┤   │     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  C++     │───┼────▶│          │────▶│          │────▶│  x86     │
  ├──────────┤   │     │    IR    │     │ Optimizer │     ├──────────┤
  │  Rust    │───┤     │ (LLVM)  │     │  Passes   │     │  ARM     │
  ├──────────┤   │     │          │     │          │     ├──────────┤
  │ Fortran  │───┘     └──────────┘     └──────────┘     │  RISC-V  │
  └──────────┘                                           └──────────┘
   M frontends              1 IR                         N backends
```

### The Three Phases

| Phase | Input | Output | Key Tasks |
|:------|:------|:-------|:----------|
| **Frontend** | Source code | IR | Lexing, parsing, type checking, semantic analysis |
| **Middle-end** | IR | Optimized IR | Dead code elimination, constant folding, loop optimization |
| **Backend** | Optimized IR | Machine code | Register allocation, instruction selection, scheduling |

---

## 2. Why ML Needs Compilers

### The Hardware Explosion

In 2020, you targeted one GPU family. Today:

| Hardware | Vendor | Compute Units | Memory Model |
|:---------|:-------|:-------------|:-------------|
| CUDA Cores / Tensor Cores | NVIDIA | SMs with warp scheduling | Unified HBM + shared mem |
| CDNA / RDNA | AMD | CUs with wavefronts | HBM + LDS |
| TPU v5e | Google | MXU systolic arrays | HBM with interleaved banks |
| Apple ANE | Apple | Neural engine tiles | Unified memory |
| Gaudi 3 | Intel | TPC + MME | HBM2E |

Writing optimized kernels by hand for each is **unsustainable**. A compiler that ingests one model and targets all of these is the only scalable answer.

### The Optimization Opportunity

A naive matrix multiply has arithmetic intensity far below what hardware can deliver:

$$\text{Operational Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}} = \frac{2N^3}{3 \times 4N^2} = \frac{N}{6}$$

For $N = 4096$, this is ~683 FLOPs/byte — well into the compute-bound regime. But only if we **tile loops**, **fuse operators**, and **manage memory hierarchy**. These are classic compiler transformations.

---

## 3. Traditional Compilers vs ML Compilers

| Aspect | Traditional (GCC/LLVM) | ML Compiler (TVM/XLA) |
|:-------|:----------------------|:---------------------|
| **Input** | Text source code | Computation graph (DAG) |
| **IR granularity** | Instructions (SSA) | Tensor operations |
| **Key optimizations** | Loop unrolling, inlining, register alloc | Operator fusion, layout transform, tiling |
| **Target** | General CPU code | Specialized accelerator kernels |
| **Correctness** | Type safety, memory safety | Numerical equivalence (within $\epsilon$) |
| **Search** | Heuristic-driven | Often autotuning (cost-model + search) |

The fundamental difference: traditional compilers optimize **scalar control flow**; ML compilers optimize **tensor data flow**.

```
   Traditional Compiler                    ML Compiler
   ──────────────────                    ──────────────
   int sum = 0;                          y = Conv2d(x, w)
   for(i=0; i<N; i++)                    z = BatchNorm(y)
     sum += a[i]*b[i];                   out = ReLU(z)
          │                                    │
          ▼                                    ▼
   Loop vectorization               Fuse Conv+BN+ReLU into
   SIMD instruction                 one tiled GPU kernel
   selection                        with shared memory staging
```

---

## 4. LLVM as Reference Architecture

LLVM is the gold standard for compiler infrastructure. Understanding it illuminates every ML compiler.

### LLVM IR Basics

LLVM IR is in **Static Single Assignment (SSA)** form — every variable is assigned exactly once:

```llvm
define float @dot(float* %a, float* %b, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %loop ]
  %pa = getelementptr float, float* %a, i32 %i
  %pb = getelementptr float, float* %b, i32 %i
  %va = load float, float* %pa
  %vb = load float, float* %pb
  %prod = fmul float %va, %vb
  %sum.next = fadd float %sum, %prod
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret float %sum.next
}
```

Key properties:
- **SSA form**: `%sum` and `%sum.next` are distinct values
- **phi nodes**: merge values from different control flow paths
- **Typed**: every value has an explicit type
- **Target-independent**: same IR for x86, ARM, RISC-V

### LLVM Pass Pipeline

```
Input IR → [Canonicalize] → [SimplifyCFG] → [InstCombine] → [GVN] →
           [LoopRotate] → [LICM] → [LoopVectorize] → [SLP Vectorize] →
           [InstSelect] → [RegAlloc] → [AsmPrinter] → Machine Code
```

ML compilers borrow this **pass pipeline** architecture directly.

---

## 5. The ML Compiler Landscape

### Where Each Framework Fits

```
                          ML Compiler Stack
  ┌──────────────────────────────────────────────────┐
  │              User-Facing Frameworks               │
  │         PyTorch    JAX    TensorFlow              │
  └─────────┬──────────┬──────────┬──────────────────┘
            │          │          │
  ┌─────────▼──┐  ┌────▼────┐  ┌─▼───────────┐
  │ torch.compile│  │  jit   │  │ tf.function │  Graph Capture
  │ (Dynamo)    │  │        │  │ (tf2xla)    │
  └─────────┬──┘  └────┬────┘  └─┬───────────┘
            │          │          │
  ┌─────────▼──┐  ┌────▼────┐  ┌─▼───────────┐
  │  FX Graph  │  │ StableHLO│ │  XLA HLO    │  High-Level IR
  │  / Inductor│  │ / MHLO  │  │             │
  └─────────┬──┘  └────┬────┘  └─┬───────────┘
            │          │          │
            ▼          ▼          ▼
  ┌──────────────────────────────────────────┐
  │        Low-Level IR / Codegen            │
  │   Triton IR   │   LLVM IR   │  SPIR-V   │
  └──────────┬────────────┬─────────┬────────┘
             ▼            ▼         ▼
  ┌──────────────────────────────────────────┐
  │              Hardware                     │
  │   NVIDIA GPU  │  CPU  │  AMD  │  TPU     │
  └──────────────────────────────────────────┘
```

### Framework Comparison

| Compiler | Owner | Source IR | Target | Key Innovation |
|:---------|:------|:---------|:-------|:--------------|
| **XLA** | Google | HLO | TPU, GPU, CPU | Algebraic simplification + fusion |
| **TVM** | Apache | Relay/TIR | Any via BYOC | Autotuning search over schedules |
| **Triton** | OpenAI | Triton-IR | NVIDIA GPU | Block-level programming model |
| **torch.compile** | Meta | FX Graph | GPU via Inductor | Python-native, dynamic shapes |
| **IREE** | Google | MLIR | CPU/GPU/mobile | Whole-program compilation |

---

## 6. A Concrete Example: How `torch.compile` Works

```python
import torch

@torch.compile
def fused_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(
        0.7978845608 * (x + 0.044715 * x ** 3)
    ))

x = torch.randn(1024, 1024, device='cuda')
y = fused_gelu(x)
```

What happens behind the scenes:

```
Step 1: Dynamo traces Python bytecode → FX Graph
Step 2: AOTAutograd captures forward + backward graph
Step 3: FX passes: constant folding, pattern matching
Step 4: Inductor lowers to Triton IR
Step 5: Triton compiles to PTX (NVIDIA assembly)
Step 6: PTX assembled to CUBIN (GPU binary)

Result: One fused kernel instead of 8 separate CUDA launches
```

### Inspecting the Generated Code

```python
import torch._inductor.config as config
config.debug = True

# After running, check /tmp/torchinductor_*/
# You'll find Triton kernel source like:
"""
@triton.jit
def triton_fused_gelu(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x0 = tl.load(in_ptr0 + xindex)
    tmp1 = 0.044715 * x0 * x0 * x0
    tmp2 = x0 + tmp1
    tmp3 = 0.7978845608 * tmp2
    tmp4 = tl.math.tanh(tmp3)
    tmp5 = 1.0 + tmp4
    tmp6 = 0.5 * x0 * tmp5
    tl.store(out_ptr0 + xindex, tmp6)
"""
```

---

## Hands-On Exercises

### Exercise 1: Trace the Compilation Pipeline (20 min)

```python
import torch
from torch.fx import symbolic_trace

class SimpleNet(torch.nn.Module):
    def forward(self, x):
        a = x * 2
        b = a + 1
        c = b * 3
        d = c - a  # `a` is reused — CSE opportunity
        return d

net = SimpleNet()
traced = symbolic_trace(net)

# 1. Print the FX graph
print(traced.graph)

# 2. Count operations
print(f"Number of nodes: {len(list(traced.graph.nodes))}")

# 3. Identify which optimizations could apply:
#    - Constant folding?
#    - Dead code elimination?
#    - Common subexpression elimination?
```

### Exercise 2: Compare Eager vs Compiled (20 min)

```python
import torch
import time

def benchmark(fn, x, warmup=10, iters=100):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms

def chain_ops(x):
    x = torch.relu(x)
    x = x * 2.0
    x = x + 1.0
    x = torch.sigmoid(x)
    return x

x = torch.randn(4096, 4096, device='cuda')

eager_ms = benchmark(chain_ops, x)
compiled = torch.compile(chain_ops)
compiled_ms = benchmark(compiled, x)

print(f"Eager:    {eager_ms:.3f} ms")
print(f"Compiled: {compiled_ms:.3f} ms")
print(f"Speedup:  {eager_ms / compiled_ms:.2f}x")
```

### Exercise 3: Map the LLVM Analogy (15 min)

Fill in this table — for each LLVM concept, identify the ML compiler equivalent:

| LLVM Concept | ML Compiler Equivalent | Example |
|:-------------|:----------------------|:--------|
| C source code | ? | `model.forward()` |
| Clang frontend | ? | Dynamo / `jit.trace` |
| LLVM IR | ? | FX Graph / HLO |
| `-O2` passes | ? | Fusion, layout transform |
| x86 backend | ? | Inductor / Triton codegen |

---

## Key Takeaways

1. **Compilers are structured translators** — the frontend/IR/backend split turns an $O(M \times N)$ problem into $O(M + N)$
2. **ML compilers operate on tensor dataflow graphs**, not scalar control flow — different IR, different optimizations
3. **The hardware explosion** makes hand-written kernels unsustainable — compilers are the scalability answer
4. **`torch.compile`** is a full compiler pipeline hiding behind a decorator: Dynamo → FX → Inductor → Triton → PTX
5. **LLVM's architecture** (SSA IR, pass pipeline, target-independent optimization) is the template every ML compiler follows

---

## Further Reading

- Lattner, C. & Adve, V. — *LLVM: A Compilation Framework for Lifelong Program Analysis and Transformation* (2004)
- Chen, T. et al. — *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (OSDI 2018)
- Ansel, J. et al. — *OpenAI Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019)
- PyTorch docs — [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- LLVM docs — [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html)

---

## Tomorrow's Preview

**Day 16** dives into the **computation graph as IR** — how neural networks become directed acyclic graphs, what different IR formats (ONNX, FX, Relay, StableHLO) look like in practice, and how to construct, inspect, and transform them. We'll dump real IR from PyTorch models and compare representations side by side.
