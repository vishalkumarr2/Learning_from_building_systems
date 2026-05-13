# Day 34: TVM Runtime & Deployment

> **Phase III · Week 5 · Day 34 of 70 · 2.5 hours**

*"The fastest compiler is useless if you can't ship the result. TVM's runtime is 100 KB of C — it runs anywhere a C compiler runs."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 33: TIR & Schedules](day-33-tir-schedules.md) | [Day 35: Mini-Project — End-to-End TVM Compilation](day-35-mini-project-tvm-compile.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

You've spent four days learning how TVM *transforms* computations — Relay for graphs, TE for operator declarations, TIR for loop nests. But a compiled kernel is just an artifact on disk until the **runtime** loads it, feeds it data, and orchestrates execution. TVM's runtime is deliberately tiny (~100 KB compiled C), runs without Python, and supports cross-compilation for devices you don't have locally. This lesson covers the complete path from `tvm.build()` to inference on an edge device — the "last mile" that determines whether your optimized model actually ships.

---

## 1. Runtime Architecture Overview

```
┌────────────────────────────────────────────────────┐
│                  Python API Layer                   │
│  relay.build()  →  Module  →  module.run()         │
└───────────────────────┬────────────────────────────┘
                        │ FFI (PackedFunc)
┌───────────────────────┴────────────────────────────┐
│               TVM Runtime (C++)                     │
│                                                     │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────┐ │
│  │  Module   │  │ PackedFunc│  │    NDArray       │ │
│  │ (compiled │  │ (type-    │  │  (DLPack-based   │ │
│  │  kernels) │  │  erased   │  │   tensor)        │ │
│  │          │  │  callable) │  │                  │ │
│  └──────────┘  └───────────┘  └─────────────────┘ │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │        Device API (LLVM / CUDA / OpenCL)    │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### The Four Core Abstractions

| Abstraction | Role | C++ Type |
|:--|:--|:--|
| **Module** | Container for compiled functions | `tvm::runtime::Module` |
| **PackedFunc** | Type-erased callable (any signature) | `tvm::runtime::PackedFunc` |
| **NDArray** | N-dimensional tensor with device placement | `tvm::runtime::NDArray` |
| **DLPack** | Zero-copy tensor interchange format | `DLManagedTensor` |

---

## 2. Module — The Compilation Unit

A `Module` is a collection of compiled functions. After `tvm.build()` or `relay.build()`, you get a Module containing all the kernels your model needs.

```python
import tvm
from tvm import te
import numpy as np

# Simple vector add
n = 1024
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)
s[C].vectorize(C.op.axis[0])

# Build for CPU (LLVM backend)
mod = tvm.build(s, [A, B, C], target="llvm", name="vecadd")

# Module API
print(type(mod))           # <class 'tvm.driver.build_module.OperatorModule'>
print(mod.get_source())    # Shows LLVM IR or source code

# Call the compiled function
dev = tvm.cpu(0)
a = tvm.nd.array(np.random.randn(n).astype("float32"), dev)
b = tvm.nd.array(np.random.randn(n).astype("float32"), dev)
c = tvm.nd.array(np.zeros(n).astype("float32"), dev)
mod(a, b, c)  # execute!

np.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy(), atol=1e-5)
```

### Module Hierarchy

For complex models, modules form a tree:

```
Host Module (LLVM)
├── device_kernel_0 (CUDA)     ← GPU kernels
├── device_kernel_1 (CUDA)
├── __tvm_main__               ← entry point / graph executor
└── metadata                   ← shapes, dtypes, param names
```

---

## 3. PackedFunc — TVM's Universal Calling Convention

`PackedFunc` is TVM's secret weapon for FFI. It's a type-erased function that can be called from Python, C++, Rust, or JavaScript with any combination of argument types.

```python
# Every module function is a PackedFunc
f = mod.get_function("vecadd")
f(a, b, c)  # same as mod(a, b, c) for single-function modules

# Register your own PackedFunc from Python
@tvm.register_func("my_custom_op")
def my_custom_op(x, y):
    return x + y

# Call it from anywhere in TVM
f = tvm.get_global_func("my_custom_op")
result = f(3, 4)  # returns 7
```

**Why type-erased?** TVM needs to call functions across language boundaries (Python ↔ C++) and across devices. PackedFunc serializes arguments into a uniform `TVMArgs` array, dispatches to the implementation, and deserializes the return value. No templates, no code generation — just a universal ABI.

```
Python call: mod(a, b, c)
    │
    ▼
TVMArgs: [NDArray*, NDArray*, NDArray*]
    │
    ▼
C++ dispatch: PackedFunc::operator()(TVMArgs args, TVMRetValue* rv)
    │
    ▼
Device execution: CUDA kernel launch / LLVM JIT call
```

---

## 4. NDArray & DLPack — Zero-Copy Tensor Exchange

### NDArray

`tvm.nd.NDArray` is TVM's tensor type. It wraps a contiguous memory buffer with shape, dtype, and device metadata.

```python
import tvm
import numpy as np

# Create on CPU
x_cpu = tvm.nd.array(np.random.randn(3, 4).astype("float32"), tvm.cpu())

# Create on GPU (if available)
x_gpu = tvm.nd.array(np.random.randn(3, 4).astype("float32"), tvm.cuda(0))

# Copy between devices
x_gpu_copy = tvm.nd.empty((3, 4), "float32", tvm.cuda(0))
x_gpu_copy.copyfrom(x_cpu)

# Convert to/from NumPy (CPU only, zero-copy when possible)
np_array = x_cpu.numpy()
```

### DLPack — The Universal Tensor Exchange

DLPack is a header-only C standard for sharing tensors between frameworks **without copying data**.

```python
# PyTorch → TVM (zero-copy on same device)
import torch
pt_tensor = torch.randn(3, 4, device="cuda")
tvm_array = tvm.nd.from_dlpack(pt_tensor)

# TVM → PyTorch
pt_back = torch.from_dlpack(tvm_array)

# Works with JAX, CuPy, TensorFlow too
```

```
┌──────────────┐    DLPack     ┌──────────────┐
│   PyTorch    │ ──────────▶  │     TVM      │
│  Tensor      │  zero-copy    │   NDArray    │
│  .data_ptr() │ ◀──────────  │  .data_ptr() │
└──────────────┘               └──────────────┘
       Same GPU memory — no allocation, no memcpy
```

---

## 5. Compilation Targets

### Target Specification

TVM uses target strings to control code generation:

```python
# CPU targets
target_x86   = tvm.target.Target("llvm -mcpu=skylake-avx512")
target_arm    = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
target_riscv  = tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mattr=+v")

# GPU targets
target_cuda   = tvm.target.Target("cuda -arch=sm_80")      # A100
target_opencl  = tvm.target.Target("opencl")
target_metal   = tvm.target.Target("metal")                 # Apple GPU
target_vulkan  = tvm.target.Target("vulkan -max_threads_per_block=256")

# Specialized
target_hexagon = tvm.target.Target("hexagon")               # Qualcomm DSP
target_wasm    = tvm.target.Target("llvm -mtriple=wasm32")   # Browser
```

### Target Selection Guide

| Deployment Scenario | Target String | Notes |
|:--|:--|:--|
| Server GPU (NVIDIA) | `cuda -arch=sm_80` | Tensor Cores via `tensorize` |
| Server CPU (x86) | `llvm -mcpu=skylake-avx512` | AVX-512 vectorization |
| Edge GPU (Jetson) | `cuda -arch=sm_72` | Xavier / Orin |
| Mobile CPU (Android) | `llvm -mtriple=aarch64-linux-android` | ARM NEON auto-vec |
| Apple Silicon | `llvm -mtriple=arm64-apple-macos` | or `metal` for GPU |
| Microcontroller | `c -mcpu=cortex-m4` | C codegen, no LLVM needed |
| Browser | `llvm -mtriple=wasm32` | Via Emscripten + WASM |

---

## 6. Cross-Compilation for Edge Devices

The key insight: you **compile on your workstation** but **run on the target device**. TVM handles this via cross-compilation + RPC.

### Step 1: Cross-Compile

```python
# On your x86 workstation, compile FOR ARM
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")

# Import and compile a model
from tvm import relay
import onnx

onnx_model = onnx.load("mobilenetv2.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": (1,3,224,224)})

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save artifacts
lib.export_library("mobilenet_arm.tar")  # contains .so + metadata
```

### Step 2: Deploy on Device

```python
# On the ARM device (Raspberry Pi, Jetson, phone, etc.)
import tvm
from tvm.contrib import graph_executor

lib = tvm.runtime.load_module("mobilenet_arm.tar")
dev = tvm.cpu(0)
m = graph_executor.GraphModule(lib["default"](dev))

# Run inference
m.set_input("input", tvm.nd.array(input_data))
m.run()
output = m.get_output(0).numpy()
```

### Step 3: RPC for Remote Testing

```
┌─────────────────┐           ┌──────────────────┐
│   Workstation    │   RPC     │   Target Device   │
│  (x86, compile)  │ ────────▶│  (ARM, execute)   │
│                  │ ◀────────│                    │
│  tvm.rpc.connect │  results  │  tvm.rpc.server   │
└─────────────────┘           └──────────────────┘
```

```python
# On target device: start RPC server
# $ python -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090

# On workstation: connect and test
from tvm import rpc

remote = rpc.connect("192.168.1.42", 9090)
remote.upload("mobilenet_arm.tar")
rlib = remote.load_module("mobilenet_arm.tar")

dev = remote.cpu(0)
m = graph_executor.GraphModule(rlib["default"](dev))
m.set_input("input", tvm.nd.array(input_data))
m.run()
# Results are transferred back automatically
```

---

## 7. Saving & Loading Compiled Modules

### Export Formats

| Format | Method | Contains | Use Case |
|:--|:--|:--|:--|
| `.tar` | `export_library()` | .so + graph JSON + params | Full model deployment |
| `.so` / `.dylib` | `export_library()` | Compiled shared library | System integration |
| `.o` | `save()` | Object file | Linking into C++ apps |
| JSON + params | `get_graph_json()` + `get_params()` | Graph + weights | Separate packaging |

```python
# Full export (recommended)
lib.export_library("model.tar")

# Separate components
with open("graph.json", "w") as f:
    f.write(graph_json)
with open("params.bin", "wb") as f:
    f.write(relay.save_param_dict(params))
lib_path = "model.so"
lib.export_library(lib_path)

# Load separately
loaded_lib = tvm.runtime.load_module("model.so")
loaded_graph = open("graph.json").read()
loaded_params = bytearray(open("params.bin", "rb").read())
```

### C++ Deployment (No Python Required)

```cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

// Load compiled module
tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("model.so");
tvm::runtime::PackedFunc run = mod.GetFunction("default");

// Create input tensor
DLDevice dev{kDLCPU, 0};
auto input = tvm::runtime::NDArray::Empty({1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);

// Execute
run(input);
```

---

## Hands-On Exercises

### Exercise 1: Build → Export → Load Round-Trip (30 min)

Compile a simple matmul with `tvm.build()`, export as `.tar`, load it back in a fresh Python session, verify correctness:

```python
# 1. Build a matmul kernel
# 2. Export with export_library("matmul.tar")
# 3. In a new script: load_module("matmul.tar")
# 4. Run and verify against NumPy
```

### Exercise 2: DLPack Zero-Copy Pipeline (20 min)

Create a pipeline: PyTorch generates a tensor → pass to TVM via DLPack → TVM runs a custom kernel → return to PyTorch via DLPack. Verify no data copies occurred by checking `.data_ptr()` addresses.

### Exercise 3: Cross-Compilation Targets (30 min)

Compile the same vector-add kernel for three different targets and compare the generated code:

```python
targets = ["llvm", "llvm -mcpu=skylake-avx512", "cuda -arch=sm_80"]
for t in targets:
    mod = tvm.build(s, [A, B, C], target=t)
    print(f"=== {t} ===")
    print(mod.get_source()[:500])  # inspect generated code
```

---

## Key Takeaways

1. **Module** wraps compiled kernels; **PackedFunc** provides a universal calling convention; **NDArray** handles device-placed tensors
2. **DLPack** enables zero-copy tensor exchange between TVM, PyTorch, JAX, and CuPy — no serialization needed
3. TVM supports **15+ compilation targets** from server GPUs to microcontrollers — same frontend, different backends
4. **Cross-compilation** lets you compile on x86 for ARM/RISC-V; **RPC** lets you test on remote devices without leaving your workstation
5. **`export_library()`** bundles everything (kernels + graph + params) into a single deployable artifact
6. The C++ runtime is **~100 KB** and requires no Python — production deployment is just loading a `.so` and calling PackedFuncs

---

## Further Reading

- [TVM Runtime System](https://tvm.apache.org/docs/arch/runtime.html) — official runtime architecture docs
- [DLPack Specification](https://dmlc.github.io/dlpack/latest/) — the tensor interchange standard
- [Deploy TVM Module using C++ API](https://tvm.apache.org/docs/how_to/deploy/cpp_deploy.html) — C++ deployment tutorial
- [Cross Compilation and RPC](https://tvm.apache.org/docs/how_to/deploy/index.html) — remote device testing guide
- [PackedFunc Design](https://tvm.apache.org/docs/arch/runtime.html#packedfunc) — deep dive into type-erased calling

---

## Tomorrow

**Day 35: Mini-Project — End-to-End TVM Compilation** — You'll put together everything from this week: import MobileNetV2 from PyTorch, optimize with Relay passes, apply schedules, compile for GPU, and benchmark against PyTorch eager and `torch.compile`. The capstone for Week 5.
