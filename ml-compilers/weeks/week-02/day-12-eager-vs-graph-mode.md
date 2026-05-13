# Day 12: Eager vs Graph Mode

> **Phase I · Week 2 · Day 12 of 70 · 2.5 hours**

*"The tension between developer experience and compiler optimization is the central drama of modern ML frameworks."*

---

| ← Previous | Next → | 📅 Week | �phase Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 11: Torch Profiler & Trace](day-11-torch-profiler-trace.md) | [Day 13: Operator Fusion](day-13-operator-fusion-fundamentals.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

Every ML framework faces the same fundamental tradeoff: **eager execution** gives you a debugger, print statements, and Python control flow — but the runtime sees only one operation at a time. **Graph mode** surrenders that flexibility so a compiler can see the full computation, enabling operator fusion, memory planning, and kernel selection that can deliver 2–5× speedups. Understanding how PyTorch navigates this tension — from TorchScript to `torch.compile` — is prerequisite knowledge for every topic in this curriculum.

---

## 1. Eager Execution: How PyTorch Dispatches Operations

In eager mode, every PyTorch operation immediately executes through the **dispatcher** — a multi-level dispatch table that routes based on device, dtype, autograd state, and more.

```python
import torch

x = torch.randn(4, 4, device='cuda')
y = torch.randn(4, 4, device='cuda')

# Each line dispatches independently — the runtime has no lookahead
z = x + y          # dispatch → CUDA kernel: elementwise_add
w = z * 2.0        # dispatch → CUDA kernel: elementwise_mul
out = w.relu()     # dispatch → CUDA kernel: relu
```

**What happens per operation:**

```
Python call: x + y
       │
       ▼
┌─────────────────────────┐
│   PyTorch Dispatcher    │
│  ┌───────────────────┐  │
│  │ Autograd dispatch  │──── record grad_fn if requires_grad
│  │ Device dispatch    │──── route to CUDA/CPU/MPS backend
│  │ Dtype dispatch     │──── select float32/float16 kernel
│  │ Layout dispatch    │──── dense vs sparse vs strided
│  └───────────────────┘  │
└─────────────────────────┘
       │
       ▼
  CUDA kernel launch
  (cudaLaunchKernel)
       │
       ▼
  Result tensor returned to Python
```

### Eager Mode Properties

| Property | Behavior |
|:--|:--|
| **Execution** | Op-by-op, immediate |
| **Shapes** | Fully dynamic — can change every call |
| **Control flow** | Full Python `if/else/for/while` |
| **Debugging** | `pdb`, `print()`, breakpoints work |
| **Performance** | No cross-op optimization, kernel launch overhead |
| **Memory** | No planning — allocator reacts to each request |

### The Cost of Ignorance

Because the runtime sees only one op at a time, it **cannot**:

1. **Fuse kernels** — `x + y` and `* 2.0` launch separate kernels, each reading/writing global memory
2. **Plan memory** — intermediate tensors are allocated on-demand via `caching_allocator`
3. **Reorder operations** — even when reordering would improve data locality
4. **Specialize shapes** — kernel selection must handle arbitrary dimensions

The overhead per op is typically 5–15 μs of Python + dispatcher cost, which dominates for small tensors.

---

## 2. Graph Mode: Seeing the Whole Picture

A **computation graph** captures the full sequence of operations before executing any of them. This gives the compiler a global view:

```
Eager view (one op at a time):     Graph view (full program):
                                   
  x + y  → launch kernel          ┌───┐  ┌───┐
  * 2.0  → launch kernel          │ x │  │ y │
  relu() → launch kernel          └─┬─┘  └─┬─┘
                                     │      │
  3 kernels, 3 memory round-trips    └──┬───┘
                                        │ add
                                        ▼
                                     ┌─────┐
                                     │  z  │
                                     └──┬──┘
                                        │ mul(2.0)
                                        ▼
                                     ┌─────┐
                                     │  w  │
                                     └──┬──┘
                                        │ relu
                                        ▼
                                     ┌─────┐
                                     │ out │
                                     └─────┘
                                   
                                   Compiler can fuse into 1 kernel!
```

### What Graph Capture Enables

| Optimization | Description | Typical Speedup |
|:--|:--|:--|
| **Operator fusion** | Merge element-wise ops into one kernel | 2–4× for chains |
| **Memory planning** | Pre-allocate and reuse buffers | 10–30% memory reduction |
| **Shape specialization** | Generate kernels for exact shapes | 5–15% compute savings |
| **Layout optimization** | Choose channels-last when beneficial | 10–20% for convolutions |
| **Dead code elimination** | Remove unused computations | Varies |
| **Constant folding** | Evaluate constant expressions at compile time | Varies |

---

## 3. TorchScript: The First Graph Mode (`torch.jit`)

TorchScript was PyTorch's first production graph mode, offering two capture mechanisms:

### 3.1 `torch.jit.trace` — Record Execution

Runs the model once with example inputs and records every operation:

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 256)
    
    def forward(self, x):
        return self.linear(x).relu()

model = SimpleModel().eval()
example_input = torch.randn(1, 784)

# Trace: run once, record the graph
traced = torch.jit.trace(model, example_input)
print(traced.graph)  # IR representation
```

**Limitations of tracing:**
- ❌ Data-dependent control flow is baked in (only one branch recorded)
- ❌ Dynamic shapes become fixed constants
- ❌ In-place dictionary/list mutations may be lost

```python
# DANGER: control flow is NOT captured correctly by trace
class ConditionalModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:       # This branch is baked at trace time!
            return x * 2
        else:
            return x * -1

traced = torch.jit.trace(ConditionalModel(), torch.ones(4))
# traced ALWAYS returns x * 2, even for negative inputs!
```

### 3.2 `torch.jit.script` — Parse Python AST

Parses the Python source code and compiles it to TorchScript IR:

```python
@torch.jit.script
def conditional_fn(x: torch.Tensor) -> torch.Tensor:
    if x.sum() > 0:       # Control flow IS captured
        return x * 2
    else:
        return x * -1

# Works correctly for all inputs
print(conditional_fn(torch.ones(4)))    # tensor([2., 2., 2., 2.])
print(conditional_fn(-torch.ones(4)))   # tensor([1., 1., 1., 1.])
```

**Limitations of scripting:**
- ❌ Only a subset of Python is supported (no `**kwargs`, limited containers)
- ❌ Type annotations required everywhere
- ❌ Many third-party libraries cannot be scripted
- ❌ Error messages are often cryptic

---

## 4. FX Symbolic Tracing

`torch.fx` (introduced PyTorch 1.8) takes a different approach: **symbolic tracing** through Python. Instead of running real tensors, it feeds `Proxy` objects that record the call graph:

```python
import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
        self.bn = torch.nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x.relu()

# FX symbolic trace
traced: torch.fx.GraphModule = torch.fx.symbolic_trace(MyModule())
print(traced.graph)
```

Output (FX IR):
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %linear : [num_users=1] = call_module[target=linear](args = (%x,), kwargs = {})
    %bn : [num_users=1] = call_module[target=bn](args = (%linear,), kwargs = {})
    %relu : [num_users=1] = call_method[target=relu](args = (%bn,), kwargs = {})
    return relu
```

### FX Graph Transformations

The real power of FX is **programmatic graph transformation**:

```python
# Example: replace all ReLU with GELU
def replace_relu_with_gelu(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == 'call_method' and node.target == 'relu':
            node.target = 'gelu'  # Swap activation
    gm.graph.lint()
    gm.recompile()
    return gm

transformed = replace_relu_with_gelu(traced)
```

---

## 5. `torch.compile` and `torch.export` (PyTorch 2.x)

PyTorch 2.0 introduced **TorchDynamo** — a CPython bytecode interceptor that captures graphs with minimal user friction.

### 5.1 `torch.compile` — JIT Compilation

```python
import torch

def my_function(x, y):
    z = x + y
    w = z * 2.0
    return w.relu()

# One line — that's it
compiled_fn = torch.compile(my_function)

# First call: TorchDynamo captures graph → TorchInductor generates code
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
result = compiled_fn(x, y)  # Triggers compilation
```

**How TorchDynamo works:**

```
Python bytecode
       │
       ▼
┌──────────────────┐
│   TorchDynamo    │ ← Intercepts at CPython frame level
│ (bytecode trans- │
│  formation)      │
└────────┬─────────┘
         │ FX Graph
         ▼
┌──────────────────┐
│   AOTAutograd    │ ← Generates forward + backward graphs
└────────┬─────────┘
         │ Aten IR
         ▼
┌──────────────────┐
│  TorchInductor   │ ← Backend compiler
│  (or other       │
│   backend)       │
└────────┬─────────┘
         │ Triton / C++ code
         ▼
   Optimized kernel
```

### 5.2 Graph Breaks

When Dynamo encounters something it can't trace, it inserts a **graph break** — splitting the graph and falling back to eager:

```python
@torch.compile
def has_graph_break(x):
    y = x + 1                     # ← Graph 1
    print(f"Shape: {y.shape}")    # ← Graph break! (print is side effect)
    z = y * 2                     # ← Graph 2
    return z

# Use TORCH_LOGS to see graph breaks:
# TORCH_LOGS="graph_breaks" python script.py
```

### 5.3 `torch.export` — Ahead-of-Time Graph Capture

```python
# torch.export produces a clean, serializable graph
from torch.export import export

class Model(torch.nn.Module):
    def forward(self, x):
        return x.relu() + 1.0

m = Model()
# Specify dynamic dimensions explicitly
batch = torch.export.Dim("batch", min=1, max=256)
exported = export(m, (torch.randn(4, 8),), dynamic_shapes={"x": {0: batch}})

print(exported.graph_module.graph)
```

---

## 6. Comparison: All Approaches at a Glance

```
┌────────────────┬──────────┬────────────┬───────────┬──────────────┬─────────────┐
│                │  Eager   │ jit.trace  │jit.script │  torch.fx    │torch.compile│
├────────────────┼──────────┼────────────┼───────────┼──────────────┼─────────────┤
│ Capture method │   None   │  Execute   │ Parse AST │Symbolic trace│ Bytecode    │
│ Control flow   │   Full   │  ✗ Baked   │ ✓ Subset  │ ✗ Limited    │ ✓ Guards    │
│ Dynamic shapes │   Full   │  ✗ Fixed   │ ✓ Limited │ ✗ Fixed      │ ✓ Guards    │
│ Python compat  │   100%   │    ~80%    │   ~60%    │    ~75%      │   ~95%      │
│ Debug ease     │   ★★★★★  │   ★★☆☆☆   │  ★★☆☆☆   │   ★★★☆☆     │  ★★★★☆     │
│ Optimization   │   None   │  Medium    │  Medium   │   High       │  Highest    │
│ Serializable   │    No    │   Yes      │   Yes     │   Yes        │  via export │
│ Status (2025)  │ Default  │ Legacy     │  Legacy   │  Used in PT2 │  Preferred  │
└────────────────┴──────────┴────────────┴───────────┴──────────────┴─────────────┘
```

---

## Hands-On Exercises

### Exercise 1: Measure Eager vs Compiled (15 min)

```python
import torch
import time

def benchmark(fn, x, warmup=10, iters=100):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms

def chain_ops(x):
    x = x + 1
    x = x * 2
    x = x.relu()
    x = x - 0.5
    x = x.sigmoid()
    return x

x = torch.randn(2048, 2048, device='cuda')
eager_ms = benchmark(chain_ops, x)
compiled_fn = torch.compile(chain_ops)
compiled_ms = benchmark(compiled_fn, x)
print(f"Eager: {eager_ms:.2f} ms | Compiled: {compiled_ms:.2f} ms | "
      f"Speedup: {eager_ms/compiled_ms:.1f}×")
```

### Exercise 2: Detect Graph Breaks (10 min)

```bash
# Run with graph break logging enabled
TORCH_LOGS="graph_breaks" python -c "
import torch

@torch.compile
def leaky(x):
    y = x.sin()
    print('debug')   # Graph break!
    return y.cos()

leaky(torch.randn(4))
"
```

### Exercise 3: Compare FX and TorchScript IR (10 min)

Trace the same model with `torch.fx.symbolic_trace()` and `torch.jit.trace()`. Compare the IR outputs. Which is more readable? Which preserves module structure?

---

## Key Takeaways

1. **Eager mode** = max flexibility, zero optimization across ops
2. **Graph mode** = restricted Python, but enables fusion, memory planning, shape specialization
3. **TorchScript** (`jit.trace`/`jit.script`) is legacy — avoid for new projects
4. **`torch.compile`** (TorchDynamo + TorchInductor) is the modern answer: ~95% Python compatibility with graph-level optimization
5. **Graph breaks** are the enemy — each break prevents optimization across the boundary
6. **`torch.export`** is for ahead-of-time capture when you need a serializable, deployable graph

---

## Further Reading

- [PyTorch 2.0: torch.compile deep dive](https://pytorch.org/get-started/pytorch-2.0/)
- [TorchDynamo internals (PyTorch dev blog)](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)
- [FX Graph Mode Quantization tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
- Ansel et al., "TorchDynamo: Run-time Python Bytecode Transformation" (2024)

---

## Tomorrow's Teaser

Now that we can capture computation graphs, what do we *do* with them? **Day 13** dives into **operator fusion** — the single most impactful optimization a compiler can perform. We'll measure memory bandwidth savings, see Triton code generated by `torch.compile`, and understand why a chain of 5 element-wise ops can run faster than a single matmul.
