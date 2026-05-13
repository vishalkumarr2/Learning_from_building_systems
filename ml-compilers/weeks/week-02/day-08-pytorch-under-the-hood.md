# Day 8: PyTorch Under the Hood

> **Phase I · Week 2 · Day 8 of 70 · 2.5 hours**

*"Any sufficiently advanced framework is indistinguishable from magic — until you read the dispatcher source."*

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 7: Mini-Project GEMM](../week-01/day-07-mini-project-gemm.md) | [Day 9: Memory Management](day-09-memory-management-pytorch.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

Every `torch.nn.Linear(768, 3072)` call triggers a cascade that spans four languages and three
hardware backends. Understanding this path — from Python syntactic sugar through the C++ dispatcher
to backend-specific kernels — is the difference between *using* PyTorch and *optimizing* it.
Compiler engineers who extend PyTorch (custom ops, torch.compile backends, quantization) must
know exactly where their code plugs into this stack.

---

## 1. The PyTorch Layer Cake

PyTorch is not a monolithic library. It is a layered system where each layer has a
well-defined responsibility:

```
┌─────────────────────────────────────────────────┐
│              Python Frontend (torch)            │  ← User-facing API
│   torch.nn, torch.Tensor, torch.autograd        │
├─────────────────────────────────────────────────┤
│           Python Bindings (torch._C)            │  ← pybind11 / nanobind
├─────────────────────────────────────────────────┤
│            C++ Dispatcher (c10)                 │  ← Routing & dispatch keys
├─────────────────────────────────────────────────┤
│         ATen Tensor Library (at::)              │  ← ~2000 operators
│   at::native::*, structured kernels             │
├─────────────────────────────────────────────────┤
│          Backend Kernels                        │
│  ┌──────────┬──────────┬───────────────────┐    │
│  │   CPU    │   CUDA   │  MPS / XLA / ...  │    │
│  │ (MKL,   │ (cuBLAS, │                   │    │
│  │  OpenMP) │  cuDNN)  │                   │    │
│  └──────────┴──────────┴───────────────────┘    │
└─────────────────────────────────────────────────┘
```

**Key insight:** The dispatcher sits between the operator schema and the implementation.
This enables features like autograd, batching (vmap), tracing (torch.compile), and
quantization — all without modifying the 2000+ operator implementations.

---

## 2. The Dispatcher Mechanism

The dispatcher is the central routing mechanism in PyTorch. Every operator call goes
through it. Think of it as a **virtual function table indexed by dispatch keys** rather
than types.

### 2.1 Dispatch Keys

A dispatch key is a tag that identifies a "functionality" to be applied. They form an
ordered stack:

```
Priority (higher = checked first):
─────────────────────────────────
  FuncTorchBatched      ← vmap transforms
  Autograd{CPU,CUDA}    ← gradient tracking
  AutocastCUDA          ← mixed precision
  FuncTorchVmapMode     ← vmap dispatch
  Functionalize         ← mutation removal
  Python                ← Python overrides (__torch_function__)
  ...
  CPU / CUDA / MPS      ← actual compute backends
  ...
  BackendSelect         ← factory functions (empty, ones, ...)
```

When you call `torch.add(a, b)`, the dispatcher:

1. Computes the **dispatch key set** from the input tensors
2. Walks the key set from highest priority to lowest
3. Calls the first registered kernel it finds

### 2.2 Operator Registration

Operators are registered with the dispatcher using the `TORCH_LIBRARY` and
`TORCH_LIBRARY_IMPL` macros:

```cpp
// Step 1: Define the schema (operator signature)
TORCH_LIBRARY(myops, m) {
    m.def("my_add(Tensor a, Tensor b) -> Tensor");
}

// Step 2: Register CPU implementation
TORCH_LIBRARY_IMPL(myops, CPU, m) {
    m.impl("my_add", my_add_cpu);
}

// Step 3: Register CUDA implementation
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
    m.impl("my_add", my_add_cuda);
}

// Step 4: Register Autograd wrapper
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
    m.impl("my_add", my_add_autograd);
}
```

The dispatcher resolves the correct kernel at runtime based on the tensor's device
and which dispatch keys are active (e.g., autograd is active if any input requires grad).

---

## 3. ATen — The Tensor Engine

ATen (A TENsor library) is the C++ core that provides:

- **`at::Tensor`** — the C++ tensor type (Python `torch.Tensor` wraps this)
- **~2000 operators** — `at::add`, `at::mm`, `at::conv2d`, etc.
- **Structured kernels** — a code-generation framework that factors out boilerplate

### 3.1 Tensor Storage Model

```
at::Tensor
├── TensorImpl*
│   ├── Storage (shared data buffer)
│   │   └── DataPtr → raw memory (CPU/CUDA/...)
│   ├── sizes_    [2, 3]        ← logical shape
│   ├── strides_  [3, 1]        ← memory layout
│   ├── storage_offset_  0      ← offset into storage
│   ├── dtype_    float32
│   ├── device_   cuda:0
│   └── autograd_meta_          ← grad_fn, requires_grad
└── (Python ref-count wrapper)
```

**Critical detail:** Multiple tensors can share the same `Storage`. A slice like
`x[0:2, 1:3]` creates a new `TensorImpl` with different sizes/strides/offset but
**the same underlying memory**. This is why in-place ops on views affect the original.

### 3.2 Native Functions

ATen operators are declared in `native_functions.yaml`:

```yaml
- func: addmm(Tensor self, Tensor mat1, Tensor mat2, *,
               Scalar beta=1, Scalar alpha=1) -> Tensor
  structured_delegate: addmm.out
  variants: function, method
  dispatch:
    CPU: addmm_cpu
    CUDA: addmm_cuda
    SparseCPU: addmm_sparse_cpu
    SparseCUDA: addmm_sparse_cuda
```

Code generation (`torchgen`) reads this YAML and produces:
- Python bindings (`torch._C._VariableFunctions`)
- Dispatcher registrations
- Structured kernel base classes

---

## 4. The Autograd Engine

Autograd intercepts operator calls *before* they reach the backend. For each
differentiable op, it:

1. Records the operation in a **computation graph** (a DAG of `Node` objects)
2. Saves tensors needed for the backward pass
3. Forwards the call to the next dispatch key (e.g., CUDA)

```
Forward pass builds the graph:
─────────────────────────────

  z = torch.addmm(bias, x, w)

  Creates: AddmmBackward0
    saved: x, w, beta=1, alpha=1
    next_functions: [x.grad_fn, w.grad_fn]

Backward pass walks it:
───────────────────────

  z.backward()
  → AddmmBackward0.apply(grad_z)
    → grad_bias = grad_z * beta
    → grad_x    = grad_z @ w.T * alpha
    → grad_w    = x.T @ grad_z * alpha
```

The autograd engine is **multi-threaded**: it uses a thread pool to execute
independent backward nodes in parallel across the DAG.

### 4.1 Autograd Formula Registration

```cpp
// derivatives.yaml (auto-generates backward nodes)
- name: addmm(Tensor self, Tensor mat1, Tensor mat2, *,
               Scalar beta, Scalar alpha)
  self: maybe_multiply(grad, beta)
  mat1: mm_mat1_backward(grad, mat2, mat1.sizes(),
                          mat1.strides(), alpha)
  mat2: mm_mat2_backward(grad, mat1, mat2.sizes(),
                          mat2.strides(), alpha)
```

---

## 5. Tracing a Real Call: `nn.Linear` → CUDA

Let's trace exactly what happens when you run:

```python
layer = torch.nn.Linear(768, 3072, bias=True).cuda()
x = torch.randn(32, 768, device='cuda')
y = layer(x)
```

### Full Call Stack

```
Python:
  1. nn.Linear.forward(x)
  2.   → F.linear(x, self.weight, self.bias)
  3.     → torch.addmm(bias, x, weight.T)
  4.       → torch._C._VariableFunctions.addmm(...)

C++ Dispatcher:
  5. c10::Dispatcher::call(addmm, args...)
  6.   dispatch_key_set = {AutogradCUDA, CUDA}
  7.   highest key = AutogradCUDA
  8.     → AddmmBackward0 node created, tensors saved
  9.   re-dispatch to CUDA key
  10.    → at::native::addmm_cuda(...)

CUDA Backend:
  11. addmm_cuda checks sizes, dtype
  12.   β=1, α=1, no transpose needed
  13.   → calls cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               N=3072, M=32, K=768,
               alpha=1.0, weight, x,
               beta=1.0, bias_expanded,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT)
  14. Returns output tensor on CUDA
```

### Verification Code

```python
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class DispatchLogger(TorchDispatchMode):
    """Log every dispatched operator."""
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        print(f"  → {func.__module__}.{func.__name__}")
        return func(*args, **(kwargs or {}))

layer = torch.nn.Linear(768, 3072).cuda()
x = torch.randn(32, 768, device='cuda')

with DispatchLogger():
    y = layer(x)

# Output (simplified):
#   → aten.t.default          (weight transpose)
#   → aten.addmm.default      (bias + x @ w.T)
```

---

## 6. Dispatch Keys in Practice

You can inspect dispatch keys at runtime:

```python
import torch

x = torch.randn(3, 3, requires_grad=True, device='cuda')

# View the dispatch key set
print(x._dispatch_key_set())
# DispatchKeySet(CUDA, AutogradCUDA)

# After detaching:
y = x.detach()
print(y._dispatch_key_set())
# DispatchKeySet(CUDA)

# The dispatcher's operator table:
print(torch._C._dispatch_dump('aten::addmm'))
# Shows all registered kernels per dispatch key
```

### Functorch Dispatch Keys

When you use `torch.vmap`, additional keys are injected:

```python
import torch
from torch.func import vmap

def f(x):
    return x @ x.T

batched_f = vmap(f)
# During execution, dispatch keys include:
# FuncTorchBatched → rewrites ops to handle batch dim
```

---

## Hands-On Exercises

### Exercise 1: Trace the Dispatcher (30 min)

```python
# Use TorchDispatchMode to log all ops during a forward pass
# of a 2-layer MLP. Count distinct operators.

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

class OpCounter(TorchDispatchMode):
    def __init__(self):
        self.ops = {}

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        name = f"{func.__module__}.{func.__name__}"
        self.ops[name] = self.ops.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).cuda()
x = torch.randn(16, 256, device='cuda')

counter = OpCounter()
with counter:
    y = model(x)

for op, count in sorted(counter.ops.items()):
    print(f"  {op}: {count}")

# QUESTION: Why does addmm appear but not mm + add separately?
```

### Exercise 2: Inspect Autograd Graph (20 min)

```python
# Visualize the autograd graph for a simple computation.

x = torch.randn(4, 4, requires_grad=True)
w = torch.randn(4, 4, requires_grad=True)
b = torch.randn(4, requires_grad=True)

y = torch.addmm(b, x, w)
z = y.sum()

# Walk the graph
node = z.grad_fn
while node:
    print(f"{type(node).__name__}")
    if node.next_functions:
        node = node.next_functions[0][0]
    else:
        break

# EXPECTED OUTPUT:
# SumBackward0
# AddmmBackward0
# AccumulateGrad
```

### Exercise 3: Custom __torch_function__ (30 min)

```python
# Create a tensor subclass that logs shape info at every op.

class ShapeTracker(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def get_shape(t):
            return tuple(t.shape) if isinstance(t, torch.Tensor) else None

        input_shapes = [get_shape(a) for a in args if isinstance(a, torch.Tensor)]
        result = super().__torch_function__(func, types, args, kwargs)
        out_shape = get_shape(result) if isinstance(result, torch.Tensor) else None
        print(f"{func.__name__}: {input_shapes} → {out_shape}")
        return result

# Test:
a = torch.randn(3, 4).as_subclass(ShapeTracker)
b = torch.randn(4, 5).as_subclass(ShapeTracker)
c = a @ b  # Should print: mm: [(3, 4), (4, 5)] → (3, 5)
```

---

## Key Takeaways

1. **PyTorch = 5 layers**: Python → bindings → dispatcher → ATen → backends
2. **The dispatcher** routes operators by dispatch key priority, enabling composable
   transforms (autograd, vmap, compile) without touching kernel code
3. **ATen operators** are declared in YAML and code-generated into C++ with
   per-backend dispatch entries
4. **Autograd** is just another dispatch key — it intercepts calls, records the
   graph, and re-dispatches to the compute backend
5. **`nn.Linear` → `addmm`** — the fused bias+matmul is a deliberate optimization
   that avoids a separate addition kernel launch

---

## Further Reading

- [PyTorch Dispatcher Deep Dive (Edward Yang)](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [PyTorch C++ Frontend docs](https://pytorch.org/cppdocs/)
- [`native_functions.yaml` source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
- [torch.library tutorial](https://pytorch.org/docs/stable/library.html)

---

## Tomorrow's Preview

**Day 9: Memory Management in PyTorch** — We'll explore the CUDA caching allocator,
understand why `torch.cuda.memory_allocated()` differs from `nvidia-smi`, learn to
debug OOM errors with memory snapshots, and see how gradient checkpointing trades
compute for memory.
