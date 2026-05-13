# Day 29: TVM Architecture Overview

> **Phase III · Week 5 · Day 29 of 70 · 2.5 hours**

*"TVM's thesis: the best compiler is one that can learn. By combining classic compiler techniques with machine-learning-driven search, TVM finds optimizations no human would write by hand."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 28: Stop & Reflect #2](../week-04/day-28-stop-and-reflect-2.md) | [Day 30: Relay IR](day-30-relay-ir.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

In Phase II you learned how PyTorch's compiler stack works — TorchDynamo captures graphs, AOTAutograd differentiates them, and Inductor lowers to Triton kernels. That stack is tightly coupled to PyTorch and NVIDIA GPUs. **Apache TVM** takes a fundamentally different approach: it's a *framework-agnostic*, *hardware-agnostic* compiler that can import models from PyTorch, TensorFlow, ONNX, and more, then compile them for CPUs, GPUs, mobile, browsers, and bare-metal MCUs. TVM pioneered the idea of **learning-based compiler optimization** — using ML to search the space of possible program transformations. Understanding TVM gives you the mental model for every modern ML compiler that followed.

---

## 1. Where TVM Sits in the Ecosystem

### The ML Compiler Landscape (2024)

```
Framework-Specific                  Framework-Agnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  torch.compile / Inductor              Apache TVM ◄── today
  XLA (JAX/TF)                          ONNX Runtime
  TF-Lite                               TensorRT (NVIDIA only)
  Core ML (Apple)                       MLIR (infrastructure)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key difference:
  torch.compile: "How do I make THIS PyTorch model fast on THIS GPU?"
  TVM:           "How do I make ANY model fast on ANY hardware?"
```

### TVM's Unique Position

| Feature | torch.compile | XLA | TVM |
|:--|:--|:--|:--|
| Input frameworks | PyTorch only | JAX, TF | PyTorch, TF, ONNX, MXNet, … |
| Target hardware | NVIDIA GPU, CPU | TPU, GPU, CPU | CPU, GPU, ARM, RISC-V, FPGA, MCU |
| Optimization | Heuristic + Triton | HLO passes | ML-driven search (AutoTVM, MetaSchedule) |
| Deployment | Python runtime | XLA runtime | Lightweight C runtime (~100 KB) |
| Open-source | Meta | Google | Apache Foundation |

---

## 2. The TVM Full-Stack Architecture

This is the central diagram for all of Phase III. Every lesson in Weeks 5–7 maps to a layer here.

```
                    ┌─────────────────────────────────┐
                    │        ML Frameworks            │
                    │  PyTorch  TF  ONNX  MXNet  ...  │
                    └──────────────┬──────────────────┘
                                   │  Frontend Importers
                                   ▼
              ┌────────────────────────────────────────────┐
              │               Relay IR                     │
              │  • Functional graph-level IR               │
              │  • Type system (Tensor, Tuple, Function)   │
              │  • High-level optimizations                │
              │    (FuseOps, FoldConstant, AlterLayout)    │
              └──────────────────┬─────────────────────────┘
                                 │  Lowering
                                 ▼
              ┌────────────────────────────────────────────┐
              │         Tensor Expression (TE)             │
              │  • DSL for defining computations           │
              │  • compute(), placeholder(), reduce()      │
              │  • Schedule primitives (split, tile, …)    │
              └──────────────────┬─────────────────────────┘
                                 │  Lowering
                                 ▼
              ┌────────────────────────────────────────────┐
              │            TensorIR (TIR)                  │
              │  • Low-level loop-based IR                 │
              │  • Explicit buffers, indices, loops        │
              │  • Target-specific transformations         │
              │  • Directly maps to hardware instructions  │
              └──────────────────┬─────────────────────────┘
                                 │  Code generation
                                 ▼
              ┌────────────────────────────────────────────┐
              │            Target Backends                 │
              │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐  │
              │  │ LLVM │ │ CUDA │ │ Metal│ │ microTVM │  │
              │  │(x86, │ │ PTX  │ │OpenCL│ │ (C, bare │  │
              │  │ ARM) │ │      │ │      │ │  metal)  │  │
              │  └──────┘ └──────┘ └──────┘ └──────────┘  │
              └──────────────────┬─────────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────────┐
              │           TVM Runtime                      │
              │  • Lightweight (~100 KB for edge)          │
              │  • Graph executor or VM executor           │
              │  • RPC for remote execution/profiling      │
              │  • Runs on Linux, Android, WASM, MCU       │
              └────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | What It Does | Analogous to |
|:--|:--|:--|
| **Frontend** | Import model → Relay graph | TorchDynamo |
| **Relay** | Graph-level optimization (fusion, layout, constant folding) | FX Graph + AOTAutograd |
| **TE** | Define tensor computations declaratively | Halide compute |
| **TIR** | Low-level loop nests, buffer management | LLVM IR / Triton IR |
| **Codegen** | Emit target-specific code (CUDA, LLVM, C) | Inductor codegen |
| **Runtime** | Execute compiled model on device | `torch.cuda` runtime |

---

## 3. Key Design Decisions

### 3.1 Separation of Compute and Schedule

TVM inherits Halide's core idea: **what** to compute is separate from **how** to compute it.

```python
# WHAT: a matrix multiply (compute definition)
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# HOW: optimization schedule (can be changed independently!)
s = te.create_schedule(C.op)
xo, xi = s[C].split(C.op.axis[0], factor=32)
yo, yi = s[C].split(C.op.axis[1], factor=32)
s[C].reorder(xo, yo, xi, yi)
s[C].vectorize(yi)
```

This separation means:
- **Same algorithm, different schedules** → different hardware targets
- **Schedules can be auto-tuned** by search algorithms
- The compute definition is a mathematical specification — always correct

### 3.2 Learning-Based Optimization

Traditional compilers use hand-written heuristics. TVM uses **ML models** to predict the best schedule:

```
Traditional compiler:              TVM:
┌──────────┐                      ┌──────────┐
│ Program  │                      │ Program  │
└────┬─────┘                      └────┬─────┘
     │                                 │
     ▼                                 ▼
┌──────────────┐               ┌─────────────────┐
│ Hand-written │               │ Generate 1000s  │
│ heuristics   │               │ of candidate    │
│ (if/else)    │               │ schedules       │
└────┬─────────┘               └────┬────────────┘
     │                              │
     ▼                              ▼
┌────────┐                  ┌──────────────────┐
│ Output │                  │ Measure on real   │
└────────┘                  │ hardware (or cost │
                            │ model prediction) │
                            └────┬─────────────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ Best schedule│
                          └──────────────┘
```

### 3.3 Lightweight Runtime

TVM compiles models into shared libraries that can run with a minimal C runtime:

- No Python dependency at inference time
- Runtime is ~100 KB (compare: PyTorch runtime is ~1 GB)
- Supports RPC for remote execution on phones, boards, etc.

---

## 4. Installation

### Option A: pip install (recommended for learning)

```bash
# CPU-only build (fastest install, sufficient for this curriculum)
pip install apache-tvm

# With CUDA support
pip install apache-tvm-cu12   # CUDA 12.x

# Verify installation
python -c "import tvm; print(tvm.__version__)"
```

### Option B: Build from Source (for development)

```bash
git clone --recursive https://github.com/apache/tvm.git
cd tvm
mkdir build && cd build

# Configure
cp ../cmake/config.cmake .
# Edit config.cmake: set USE_LLVM, USE_CUDA as needed
cmake ..
make -j$(nproc)

# Python package
cd ../python
pip install -e .
```

### Option C: Docker

```bash
docker pull tlcpack/ci-gpu:latest
docker run -it --gpus all tlcpack/ci-gpu:latest bash
```

### Verify Full Stack

```python
import tvm
from tvm import relay, te
import numpy as np

print(f"TVM version: {tvm.__version__}")
print(f"LLVM enabled: {tvm.runtime.enabled('llvm')}")
print(f"CUDA enabled: {tvm.runtime.enabled('cuda')}")
print(f"Available targets: {tvm.target.Target.list_kinds()}")
```

---

## 5. First Model Compile: ResNet-18 from ONNX

This exercise walks through the complete TVM pipeline for a real model.

### Step 1: Export ResNet-18 to ONNX

```python
import torch
import torchvision

# Get pretrained ResNet-18
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}})
```

### Step 2: Import into TVM via Relay

```python
import onnx
import tvm
from tvm import relay

# Load ONNX model
onnx_model = onnx.load("resnet18.onnx")

# Convert to Relay IR
shape_dict = {"input": (1, 3, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Inspect the Relay module
print(type(mod))          # tvm.IRModule
print(mod["main"])        # The main function in Relay IR

# You'll see something like:
# fn (%input: Tensor[(1, 3, 224, 224), float32],
#     %v193: Tensor[(64, 3, 7, 7), float32],  ← conv weights
#     ...) {
#   %0 = nn.conv2d(%input, %v193, strides=[2, 2], padding=[3, 3, 3, 3], ...);
#   %1 = nn.batch_norm(%0, ...);
#   ...
# }
```

### Step 3: Compile for CPU

```python
# Set target (CPU with LLVM)
target = tvm.target.Target("llvm -mcpu=native")

# Apply standard optimization passes and compile
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# What just happened:
# 1. Relay passes ran (FuseOps, FoldConstant, AlterOpLayout, ...)
# 2. Each fused op was lowered through TE → TIR
# 3. TIR was compiled via LLVM to native x86 code
# 4. Result: a shared library with the compiled model
```

### Step 4: Run Inference

```python
from tvm.contrib import graph_executor
import numpy as np

# Create runtime executor
dev = tvm.cpu(0)
executor = graph_executor.GraphModule(lib["default"](dev))

# Prepare input (ImageNet normalization)
input_data = np.random.randn(1, 3, 224, 224).astype("float32")
executor.set_input("input", tvm.nd.array(input_data))

# Run inference
executor.run()

# Get output
output = executor.get_output(0).numpy()
print(f"Output shape: {output.shape}")       # (1, 1000)
print(f"Top-5 classes: {output[0].argsort()[-5:][::-1]}")
```

### Step 5: Benchmark

```python
import timeit

# TVM benchmark
def tvm_inference():
    executor.set_input("input", tvm.nd.array(input_data))
    executor.run()

tvm_time = timeit.timeit(tvm_inference, number=100) / 100
print(f"TVM CPU:     {tvm_time*1000:.2f} ms")

# PyTorch baseline for comparison
with torch.no_grad():
    pt_input = torch.from_numpy(input_data)
    # Warm up
    for _ in range(10):
        model(pt_input)
    
    pt_time = timeit.timeit(lambda: model(pt_input), number=100) / 100
    print(f"PyTorch CPU: {pt_time*1000:.2f} ms")
    print(f"Speedup:     {pt_time/tvm_time:.2f}x")

# Typical result (varies by CPU):
#   TVM CPU:     8.3 ms   (with opt_level=3)
#   PyTorch CPU: 14.1 ms  (eager mode)
#   Speedup:     1.70x
```

---

## 6. Understanding What Happened Under the Hood

### The Compilation Pipeline in Detail

```python
# We can inspect each stage:

# 1. After Relay passes (before lowering)
with tvm.transform.PassContext(opt_level=3):
    mod_optimized = relay.transform.InferType()(mod)
    mod_optimized = relay.transform.FuseOps(fuse_opt_level=2)(mod_optimized)
    print("After FuseOps:")
    print(mod_optimized["main"])

# 2. Inspect the TIR for a specific fused function
# (requires building with debug info)
```

### Pass Pipeline at opt_level=3

TVM applies these Relay passes in sequence:

```
opt_level=3 pass pipeline:
────────────────────────────────────────────────────────
  1. SimplifyInference        ← Fold BatchNorm into Conv
  2. FoldConstant             ← Evaluate constant expressions
  3. FoldScaleAxis            ← Propagate scale factors
  4. CanonicalizeOps          ← Normalize op representations
  5. AlterOpLayout            ← NCHW → NCHW8c (for AVX)
  6. FuseOps                  ← Fuse element-wise + reduce chains
  7. InferType                ← Re-infer types after transforms
────────────────────────────────────────────────────────
```

### Operator Fusion Example

```
Before FuseOps:                  After FuseOps:
┌──────┐                         ┌─────────────────────┐
│Conv2D│                         │    Fused Function    │
└──┬───┘                         │  Conv2D + BN + ReLU  │
   │                             │  (single kernel)     │
┌──▼────────┐                    └─────────────────────┘
│BatchNorm  │
└──┬────────┘          Memory traffic: 3 reads → 1 read
   │                   Kernel launches: 3 → 1
┌──▼───┐
│ ReLU │
└──────┘
```

---

## Hands-On Exercises

### Exercise 1: Compile and Benchmark Different opt_levels

```python
import tvm
from tvm import relay
import onnx
import timeit
import numpy as np

onnx_model = onnx.load("resnet18.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, {"input": (1, 3, 224, 224)})
target = tvm.target.Target("llvm -mcpu=native")

for opt_level in [0, 1, 2, 3]:
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)
    
    dev = tvm.cpu(0)
    executor = graph_executor.GraphModule(lib["default"](dev))
    input_data = np.random.randn(1, 3, 224, 224).astype("float32")
    
    def run():
        executor.set_input("input", tvm.nd.array(input_data))
        executor.run()
    
    # Warm up
    for _ in range(10):
        run()
    
    time_ms = timeit.timeit(run, number=50) / 50 * 1000
    print(f"opt_level={opt_level}: {time_ms:.2f} ms")

# Expected output:
#   opt_level=0: ~25 ms  (no optimization)
#   opt_level=1: ~18 ms  (basic fusion)
#   opt_level=2: ~12 ms  (layout transforms)
#   opt_level=3: ~8 ms   (all optimizations)
```

### Exercise 2: Inspect the Relay IR

```python
# Print the full Relay IR
print(mod["main"])

# Count the number of operations
def count_ops(expr):
    ops = {}
    def visit(node):
        if isinstance(node, tvm.relay.Call):
            op_name = str(node.op)
            ops[op_name] = ops.get(op_name, 0) + 1
    tvm.relay.analysis.post_order_visit(mod["main"], visit)
    return ops

op_counts = count_ops(mod["main"])
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")
# Expected: nn.conv2d: 20, nn.batch_norm: 20, nn.relu: 17, ...
```

### Exercise 3: Export and Load Compiled Model

```python
# Save compiled model
lib.export_library("resnet18_tvm.so")

# Later, load without Python model definition:
loaded_lib = tvm.runtime.load_module("resnet18_tvm.so")
executor = graph_executor.GraphModule(loaded_lib["default"](tvm.cpu(0)))

# This .so file can run on any machine with TVM runtime
# No PyTorch, no ONNX, no Python model code needed!
```

---

## Key Takeaways

1. **TVM is framework-agnostic and hardware-agnostic** — import from any framework, deploy to any target
2. **The stack has clear layers**: Frontend → Relay (graph) → TE (compute) → TIR (loops) → Codegen → Runtime
3. **Compute/schedule separation** (from Halide) is the key design decision — the same algorithm can be optimized differently for different hardware
4. **Learning-based optimization** distinguishes TVM from traditional compilers — ML models guide the search for optimal schedules
5. **Lightweight runtime** (~100 KB) enables deployment on edge devices, phones, browsers, and MCUs
6. **At opt_level=3**, TVM applies operator fusion, constant folding, layout transformation, and batch norm folding — yielding 1.5–3× speedup over eager PyTorch on CPU

---

## Further Reading

- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.00751) — the original paper (Chen et al., 2018)
- [Apache TVM Documentation](https://tvm.apache.org/docs/) — official docs
- [TVM Design and Architecture](https://tvm.apache.org/docs/arch/index.html) — architectural overview
- [MLC.ai Course](https://mlc.ai/) — Tianqi Chen's Machine Learning Compilation course (uses TVM)
- [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166) — the AutoTVM paper

---

## Tomorrow

**Day 30: Relay IR** — We'll dive deep into TVM's graph-level intermediate representation. You'll learn Relay's type system, how to construct programs with let-bindings and pattern matching, and how frontend importers translate framework-specific ops into Relay's functional IR.
