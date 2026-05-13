# Day 43: MLIR for ML

> **Phase III · Week 7 · Day 43 of 70 · 2.5 hours**

*"The best infrastructure doesn't solve one problem — it gives you the tools to solve every problem in the same way."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 42: Stop & Reflect #3](../week-06/day-42-stop-and-reflect-3.md) | [Day 44: XLA & StableHLO](day-44-xla-stablehlo.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Every major ML compiler — XLA, TVM, Torch-MLIR, IREE, Triton — is converging on **MLIR** (Multi-Level Intermediate Representation) as its foundational infrastructure. MLIR was created at Google (originally for TensorFlow) and donated to the LLVM project. Unlike LLVM IR, which is a single fixed abstraction level, MLIR lets you define **multiple IRs (dialects) that coexist in the same program** and progressively lower through well-defined transformations. Understanding MLIR's design philosophy and key dialects is essential because it's becoming the common language for the entire ML compiler ecosystem.

---

## 1. The Problem MLIR Solves

### The IR Proliferation Problem

Before MLIR, every ML framework built its own IR stack from scratch:

```
Before MLIR — Every Project Reinvents the Wheel
════════════════════════════════════════════════

  TensorFlow        PyTorch          TVM           ONNX Runtime
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ TF Graph │     │ TorchIR │     │  Relay  │     │ ORT Graph│
  └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
       │               │               │               │
  ┌────▼─────┐     ┌────▼─────┐     ┌────▼─────┐     ┌────▼─────┐
  │  XLA HLO │     │ Triton IR│     │   TIR   │     │  Custom  │
  └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
       │               │               │               │
  ┌────▼─────┐     ┌────▼─────┐     ┌────▼─────┐     ┌────▼─────┘
  │ LLVM IR  │     │ LLVM IR  │     │ LLVM IR  │     │ Provider │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘

  Problems:
  ✗ Each project writes its own passes (CSE, DCE, inlining...)
  ✗ No shared tooling for testing, debugging, verification
  ✗ Combining compilers = rewriting everything
  ✗ Optimization passes don't compose across projects
```

### MLIR's Solution: A Compiler Infrastructure for IRs

MLIR doesn't define one IR — it provides **infrastructure for building IRs**:

$$\text{MLIR} = \text{Dialect System} + \text{Pass Infrastructure} + \text{Rewrite Patterns} + \text{Verification}$$

```
MLIR — One Infrastructure, Many Dialects
═════════════════════════════════════════

  ┌──────────────────────────────────────────────────┐
  │                   MLIR Framework                  │
  │                                                   │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
  │  │ linalg  │ │ tensor  │ │  stablehlo │ │ tosa  │ │  ← ML dialects
  │  └────┬────┘ └────┬────┘ └────┬────┘ └───┬────┘ │
  │       │           │           │           │       │
  │  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐            │
  │  │  affine │ │   scf   │ │  memref │            │  ← Mid-level
  │  └────┬────┘ └────┬────┘ └────┬────┘            │
  │       │           │           │                   │
  │  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐            │
  │  │  arith  │ │   gpu   │ │  llvm   │            │  ← Low-level
  │  └─────────┘ └─────────┘ └─────────┘            │
  │                                                   │
  │  Shared: Pass Infra, Pattern Rewriter, Verifier   │
  └──────────────────────────────────────────────────┘
```

---

## 2. Core MLIR Concepts

### Dialects: Self-Contained IR Vocabularies

A **dialect** is a namespace of operations, types, and attributes. Each dialect captures one abstraction level:

```mlir
// Example: Three different dialects in ONE module

// --- tensor dialect (value semantics, no memory) ---
%result = tensor.extract_slice %input[0, 0][4, 8][1, 1]
    : tensor<16x32xf32> to tensor<4x8xf32>

// --- linalg dialect (structured computation) ---
%matmul = linalg.matmul
    ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
    outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>

// --- arith dialect (scalar arithmetic) ---
%sum = arith.addf %x, %y : f32
```

### Progressive Lowering

The key idea: programs start at a **high level** and are progressively lowered through **well-defined dialect conversions**:

```
Progressive Lowering Pipeline for a MatMul
═══════════════════════════════════════════

  linalg.matmul                          ← Structured op
      │  ──── linalg-to-loops ────
      ▼
  scf.for i = 0..M                      ← Loop nest
    scf.for j = 0..N
      scf.for k = 0..K
        %a = memref.load %A[i,k]
        %b = memref.load %B[k,j]
        %c = arith.mulf %a, %b
        %d = arith.addf %c, %acc
      │  ──── scf-to-cf ────
      ▼
  cf.br ^bb1                             ← Control flow graph
  ^bb1(%i : index):
    cf.cond_br %cmp, ^bb2, ^bb3
      │  ──── cf-to-llvm ────
      ▼
  llvm.br ^bb1(%i : i64)                 ← LLVM dialect → LLVM IR
```

### Operations, Regions, and Blocks

Every construct in MLIR is an **operation** with a uniform structure:

```
%result:2 = "dialect.op_name"(%arg0, %arg1) ({
   ^bb0(%barg0: f32):    // Block with arguments
     "dialect.yield"(%barg0) : (f32) -> ()
}) { attr = "value" } : (f32, f32) -> (f32, i1)
 ↑          ↑              ↑           ↑        ↑
 results    op name     region       attrs    types
```

---

## 3. Key Dialects for ML Compilation

### The ML Compilation Dialect Stack

| Dialect | Abstraction Level | What It Captures |
|:--|:--|:--|
| `stablehlo` / `mhlo` | Highest | ML ops: conv, dot, reduce, gather |
| `tosa` | High | Target-independent ML ops (mobile-focused) |
| `linalg` | High–Mid | Structured ops with access patterns |
| `tensor` | High | Value-semantic multi-dim arrays |
| `memref` | Mid | Memory-backed arrays with layouts |
| `affine` | Mid | Polyhedral loop nests |
| `scf` | Mid | Structured control flow (for, if, while) |
| `arith` | Low | Integer/float arithmetic |
| `math` | Low | Transcendental functions (exp, sin, log) |
| `gpu` | Low | GPU kernel launch and thread mapping |
| `vector` | Low | SIMD/vector operations |
| `llvm` | Lowest | 1:1 mapping to LLVM IR |

### The `linalg` Dialect — Heart of ML Compilation

`linalg` (Linear Algebra) captures computations as **iteration domains + access maps**:

```mlir
// linalg.generic — the universal structured op
//
// Semantics:  C[i,j] += A[i,k] * B[k,j]   (matrix multiply)

#map_A = affine_map<(i, j, k) -> (i, k)>    // A indexed by [i,k]
#map_B = affine_map<(i, j, k) -> (k, j)>    // B indexed by [k,j]
#map_C = affine_map<(i, j, k) -> (i, j)>    // C indexed by [i,j]

%result = linalg.generic {
    indexing_maps = [#map_A, #map_B, #map_C],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
    outs(%C : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %prod = arith.mulf %a, %b : f32
    %sum  = arith.addf %c, %prod : f32
    linalg.yield %sum : f32
  } -> tensor<4x16xf32>
```

The power: tiling, fusion, and parallelization are all **mechanical transformations on the indexing maps**, not ad-hoc rewrites.

### `tensor` vs `memref` — Value vs Memory Semantics

```
tensor<4x8xf32>          memref<4x8xf32, strided<[8,1]>>
     │                          │
     │  Value semantics          │  Memory semantics
     │  • Immutable              │  • Mutable (in-place)
     │  • SSA-friendly           │  • Has layout / strides
     │  • No aliasing            │  • Aliasing possible
     │  • Compiler can reason    │  • Maps to real buffers
     │    freely                 │
     │                          │
     └────── bufferization ─────┘
              (tensor → memref)
```

Bufferization is a key MLIR pass: it decides where to allocate memory, when to reuse buffers, and converts value-semantic `tensor` programs into `memref`-based programs that can run on hardware.

---

## 4. How ML Compilers Use MLIR

### IREE: End-to-End MLIR Pipeline

IREE (Intermediate Representation Execution Environment) is the most complete MLIR-based ML compiler:

```
IREE Compilation Pipeline (fully MLIR-based)
═════════════════════════════════════════════

  PyTorch / JAX / TF model
         │
    ┌────▼────────────────┐
    │  Import (StableHLO) │  ← Framework → MLIR
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Flow dialect        │  ← Workload partitioning
    │  • Dispatch regions  │     (what runs on device)
    │  • Data dependencies │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Stream dialect      │  ← Async execution scheduling
    │  • Command buffers   │
    │  • Synchronization   │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  HAL dialect         │  ← Hardware abstraction
    │  • Vulkan / CUDA /   │
    │    Metal / CPU        │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Executable binary   │  ← Deployable artifact
    └─────────────────────┘
```

### Torch-MLIR: PyTorch into the MLIR Ecosystem

```python
# Torch-MLIR bridges PyTorch → MLIR dialects
import torch
import torch_mlir

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y) + 1.0

model = MyModel()
example_inputs = (torch.randn(4, 8), torch.randn(8, 16))

# Export to linalg-on-tensors (MLIR)
module = torch_mlir.compile(
    model, example_inputs,
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

print(module.operation.get_asm())
# Produces MLIR with linalg.matmul, arith.addf, tensor ops
```

### Available Output Types

```python
# Torch-MLIR can target different MLIR dialect levels:
torch_mlir.OutputType.TORCH      # torch dialect (highest level)
torch_mlir.OutputType.LINALG_ON_TENSORS  # linalg + tensor
torch_mlir.OutputType.STABLEHLO  # for XLA/IREE consumption
torch_mlir.OutputType.TOSA       # for mobile/edge targets
```

---

## 5. MLIR vs TVM's IR Stack

### Architectural Comparison

```
TVM IR Stack                    MLIR Ecosystem
════════════                    ══════════════

 ┌──────────┐                  ┌──────────────┐
 │  Relax   │  ← Graph-level   │  StableHLO   │  ← ML ops
 │  (Relay) │                  │  / TOSA       │
 └────┬─────┘                  └──────┬───────┘
      │                               │
 ┌────▼─────┐                  ┌──────▼───────┐
 │   TIR    │  ← Loop-level    │   linalg     │  ← Structured
 │          │                  │   + affine    │
 └────┬─────┘                  └──────┬───────┘
      │                               │
 ┌────▼─────┐                  ┌──────▼───────┐
 │ LLVM IR  │  ← Code gen      │  scf → llvm  │  ← Lowering
 └──────────┘                  └──────────────┘

Key Differences:
┌──────────────────┬────────────────────┬────────────────────┐
│ Aspect           │ TVM                │ MLIR               │
├──────────────────┼────────────────────┼────────────────────┤
│ Philosophy       │ ML-specific        │ General compiler   │
│ Extensibility    │ Add to fixed IRs   │ Create new dialect │
│ Auto-tuning      │ Built-in           │ Not built-in       │
│ Community        │ ML-focused         │ Broader (HPC, HW)  │
│ Deployment       │ TVM runtime        │ Various (IREE,etc) │
│ Learning curve   │ Moderate           │ Steep              │
└──────────────────┴────────────────────┴────────────────────┘
```

### What TVM Could Gain from MLIR

TVM's Relax/TIR are powerful but isolated. MLIR provides:

1. **Shared pass infrastructure** — CSE, DCE, canonicalization work across all dialects
2. **Pattern rewriting engine** — Declarative rewrite rules (DRR) for transformations
3. **Verification** — Every op has a verifier; malformed IR is caught early
4. **Growing ecosystem** — StableHLO, TOSA, Triton-MLIR, etc. all interoperate

In fact, TVM Unity's design was partly inspired by MLIR's approach to composable IRs.

---

## 6. Hands-On: Exploring MLIR

### Exercise 1: Build and Run `mlir-opt` (30 min)

```bash
# Option A: Use a pre-built MLIR (if available via pip)
pip install mlir-python-bindings

# Option B: Build from LLVM source (takes ~30 min)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release
ninja mlir-opt

# Try progressive lowering of a simple function:
cat > matmul.mlir << 'EOF'
func.func @matmul(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>,
                   %C: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %result = linalg.matmul
    ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
    outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %result : tensor<4x16xf32>
}
EOF

# Step 1: Tile the matmul
mlir-opt matmul.mlir \
  --transform-interpreter \
  --linalg-tile="tile-sizes=2,4,8"

# Step 2: Lower linalg to loops
mlir-opt matmul.mlir --convert-linalg-to-loops

# Step 3: Full pipeline to LLVM
mlir-opt matmul.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

### Exercise 2: MLIR Python Bindings (20 min)

```python
# Using MLIR's Python bindings to build IR programmatically
from mlir.ir import Context, Module, InsertionPoint, Location
from mlir.dialects import func, arith, linalg, tensor
from mlir import ir

with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = ir.F32Type.get()
    tensor_type = ir.RankedTensorType.get([4, 8], f32)

    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(tensor_type, tensor_type)
        def add_tensors(a, b):
            return arith.AddFOp(a, b)

    print(module)
    # Verify the module is well-formed
    assert module.operation.verify()
```

### Exercise 3: Compare Lowering Stages (20 min)

```bash
# Create a pipeline that dumps IR at each stage

cat > pipeline.sh << 'EOF'
#!/bin/bash
INPUT=$1
echo "=== Stage 1: Original ==="
cat $INPUT
echo ""

echo "=== Stage 2: After Linalg Tiling ==="
mlir-opt $INPUT --linalg-tile="tile-sizes=2,4" 2>/dev/null
echo ""

echo "=== Stage 3: After Lowering to Loops ==="
mlir-opt $INPUT --convert-linalg-to-loops 2>/dev/null
echo ""

echo "=== Stage 4: After Buffer Allocation ==="
mlir-opt $INPUT \
  --one-shot-bufferize="bufferize-function-boundaries" \
  2>/dev/null
echo ""

echo "=== Stage 5: After Lowering to LLVM ==="
mlir-opt $INPUT \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-func-to-llvm 2>/dev/null
EOF

chmod +x pipeline.sh
./pipeline.sh matmul.mlir
```

---

## Key Takeaways

1. **MLIR is compiler infrastructure for building IRs** — not a single IR, but a framework of composable dialects with shared tooling
2. **Dialects** are self-contained vocabularies (ops, types, attrs) that capture one abstraction level; multiple dialects coexist in one module
3. **Progressive lowering** transforms programs through dialect conversions: `linalg → scf → cf → llvm`, each step well-defined and testable
4. **`linalg` is the key ML dialect** — it encodes computations as iteration domains + affine access maps, enabling mechanical tiling, fusion, and parallelization
5. **`tensor` vs `memref`** separates value semantics (what to compute) from memory semantics (where to store), connected by bufferization
6. **The ML compiler ecosystem is converging on MLIR** — XLA (StableHLO), Torch-MLIR, IREE, Triton, and even TVM's design philosophy borrow from MLIR

---

## Further Reading

- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) — original paper
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) — syntax and semantics
- [Linalg Dialect Rationale](https://mlir.llvm.org/docs/Dialects/Linalg/) — design decisions
- [Torch-MLIR Project](https://github.com/llvm/torch-mlir) — PyTorch to MLIR
- [IREE Documentation](https://iree.dev/) — end-to-end MLIR compiler
- [MLIR Tutorial (LLVM)](https://mlir.llvm.org/docs/Tutorials/Toy/) — the Toy language tutorial

---

## Tomorrow: XLA & StableHLO

Day 44 explores **XLA** — Google's production ML compiler that powers JAX and TensorFlow — and **StableHLO**, the portable MLIR dialect that lets you move models between XLA, IREE, and other backends. You'll trace a JAX program through XLA's HLO IR, see how fusion passes work, and understand when to choose XLA vs TVM.
