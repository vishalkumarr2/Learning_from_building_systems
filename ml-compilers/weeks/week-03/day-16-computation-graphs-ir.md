# Day 16: Computation Graphs as IR

> **Phase II · Week 3 · Day 16 of 70 · 2.5 hours**

*"A neural network is just a directed acyclic graph with tensors on the edges and math on the nodes — and every ML compiler begins by building exactly that graph."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 15: Compiler 101 for ML](day-15-compiler-101-for-ml.md) | [Day 17: Graph-Level Optimizations](day-17-graph-level-optimizations.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Before a compiler can optimize anything, it needs a **representation** of the program. In ML compilers, that representation is a **computation graph** — a DAG where nodes are operations and edges carry tensors. The quality of this IR determines what optimizations are possible, how easy they are to implement, and whether the compiler can reason about the program at all. Every major ML framework has its own IR, and understanding the design tradeoffs lets you pick the right tool and debug compilation failures.

---

## 1. From Neural Network to DAG

A neural network is naturally a directed acyclic graph (DAG):

```python
# PyTorch code
def forward(x, w1, w2):
    h = torch.matmul(x, w1)       # Node 1: matmul
    h = torch.relu(h)             # Node 2: relu
    h = torch.matmul(h, w2)       # Node 3: matmul
    h = torch.softmax(h, dim=-1)  # Node 4: softmax
    return h
```

```
               Computation Graph (DAG)

     x ──────────┐
                  ▼
     w1 ───▶ [matmul] ──▶ [relu] ──┐
                                     ▼
                          w2 ───▶ [matmul] ──▶ [softmax] ──▶ output
```

Properties of this graph:
- **Nodes**: tensor operations (matmul, relu, softmax)
- **Edges**: tensors flowing between operations, with shape and dtype metadata
- **Acyclic**: no loops (recurrence is unrolled or uses scan primitives)
- **Dataflow**: execution order is determined by data dependencies, not textual order

---

## 2. Node Types in ML IRs

ML computation graphs contain more than just arithmetic:

```
  ┌─────────────────────────────────────────────────────────┐
  │                    Node Taxonomy                         │
  ├──────────────┬──────────────────────────────────────────┤
  │  Compute     │  matmul, conv2d, relu, softmax, add     │
  │              │  → actual arithmetic on tensors          │
  ├──────────────┼──────────────────────────────────────────┤
  │  Data        │  placeholder (input), get_attr (weight), │
  │              │  constant, output                        │
  │              │  → sources/sinks of tensor data          │
  ├──────────────┼──────────────────────────────────────────┤
  │  Shape       │  reshape, view, permute, slice, expand   │
  │              │  → no compute, just reinterpret memory   │
  ├──────────────┼──────────────────────────────────────────┤
  │  Control     │  if/else, while_loop, scan               │
  │              │  → conditional or iterative execution    │
  ├──────────────┼──────────────────────────────────────────┤
  │  Memory      │  copy, alloc, dealloc, to_device         │
  │              │  → explicit memory management            │
  └──────────────┴──────────────────────────────────────────┘
```

Different IRs expose different subsets. FX graphs are mostly compute + data + shape. XLA HLO includes explicit memory (and custom-call for vendor libs). MLIR can represent all five in different dialects.

---

## 3. Static Single Assignment (SSA) in ML

SSA form means every variable is defined exactly once. This is natural for computation graphs:

```python
# Python (NOT SSA — `h` is reassigned)
h = matmul(x, w1)
h = relu(h)          # h is overwritten
h = matmul(h, w2)

# SSA form (each value has a unique name)
h1 = matmul(x, w1)
h2 = relu(h1)
h3 = matmul(h2, w2)
```

Why SSA matters:
- **Unambiguous def-use chains**: for any value, you know exactly where it was produced and where it's consumed
- **Optimization is easier**: dead code elimination = remove nodes with no users; constant propagation = replace node with constant
- **Parallelism is visible**: independent nodes with no data dependency can execute concurrently

In a graph IR, SSA is **automatic** — each node output is a unique value. This is why graph IRs are so natural for compiler passes.

---

## 4. IR Comparison: Five Representations of the Same Model

Let's trace a simple model through five different IRs:

```python
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
    
    def forward(self, x):
        return torch.relu(self.linear(x))
```

### 4a. PyTorch FX Graph

```python
from torch.fx import symbolic_trace

model = TinyModel()
traced = symbolic_trace(model)
traced.graph.print_tabular()
```

Output:
```
opcode       name       target                  args            kwargs
-----------  ---------  ----------------------  --------------  --------
placeholder  x          x                       ()              {}
get_attr     linear_w   linear.weight           ()              {}
get_attr     linear_b   linear.bias             ()              {}
call_func    linear_1   torch.nn.functional.linear  (x, linear_w, linear_b)  {}
call_func    relu       torch.relu              (linear_1,)     {}
output       output     output                  (relu,)         {}
```

### 4b. TorchScript IR

```python
scripted = torch.jit.script(model)
print(scripted.graph)
```

Output:
```
graph(%self : __torch__.TinyModel,
      %x : Float(*, 4)):
  %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self)
  %weight : Float(3, 4) = prim::GetAttr[name="weight"](%linear)
  %bias : Float(3) = prim::GetAttr[name="bias"](%linear)
  %out : Float(*, 3) = aten::linear(%x, %weight, %bias)
  %relu : Float(*, 3) = aten::relu(%out)
  return (%relu)
```

### 4c. ONNX

```python
torch.onnx.export(model, torch.randn(1, 4), "tiny.onnx")
# Inspecting with onnx.load:
```

```
Nodes:
  MatMul(x, linear.weight_transposed) → mm_out
  Add(mm_out, linear.bias) → linear_out
  Relu(linear_out) → output
```

### 4d. XLA HLO (via JAX)

```
HloModule tiny_model
ENTRY main {
  p0 = f32[1,4] parameter(0)           // input x
  p1 = f32[3,4] parameter(1)           // weight
  p2 = f32[3] parameter(2)             // bias
  dot = f32[1,3] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bcast = f32[1,3] broadcast(p2), dimensions={1}
  add = f32[1,3] add(dot, bcast)
  ROOT relu = f32[1,3] maximum(add, f32[1,3] broadcast(f32[] constant(0)))
}
```

### 4e. Relay (TVM)

```
fn (%x: Tensor[(1, 4), float32],
    %weight: Tensor[(3, 4), float32],
    %bias: Tensor[(3), float32]) {
  %0 = nn.dense(%x, %weight, units=3);
  %1 = nn.bias_add(%0, %bias);
  nn.relu(%1)
}
```

### Comparison Summary

| IR | Typed? | SSA? | Control Flow | Abstraction Level |
|:---|:-------|:-----|:-------------|:-----------------|
| **FX Graph** | Shape-aware | Yes | Limited (via subgraphs) | High (PyTorch ops) |
| **TorchScript** | Typed | Yes | Full (`if`, `for`) | Medium (ATen ops) |
| **ONNX** | Typed | Yes | Limited | High (standardized ops) |
| **XLA HLO** | Fully typed | Yes | Structured (`while`, `cond`) | Low (primitives) |
| **Relay** | Type-inferred | Yes | Functional (`let`, `if`) | Medium (nn + math) |

---

## 5. Graph Construction with FX

FX uses **symbolic tracing** — it runs your Python code with proxy objects that record operations instead of executing them:

```python
import torch
from torch.fx import symbolic_trace, Graph, Node

class FusionCandidate(torch.nn.Module):
    def forward(self, x):
        # These three ops could be fused into one kernel
        a = x * 2.0
        b = a + 1.0
        c = torch.relu(b)
        return c

model = FusionCandidate()
traced = symbolic_trace(model)

# Introspect the graph
for node in traced.graph.nodes:
    print(f"  {node.op:12s} | {node.name:10s} | {node.target} | args={node.args}")
```

Output:
```
  placeholder  | x          | x       | args=()
  call_function| mul        | <built-in function mul> | args=(x, 2.0)
  call_function| add        | <built-in function add> | args=(mul, 1.0)
  call_function| relu       | <function relu> | args=(add,)
  output       | output     | output  | args=(relu,)
```

### Analyzing Data Dependencies

```python
def print_dependency_graph(gm):
    """Print which nodes depend on which."""
    for node in gm.graph.nodes:
        if node.op in ('call_function', 'call_method'):
            deps = [a.name for a in node.args if isinstance(a, Node)]
            print(f"  {node.name} depends on: {deps}")

print_dependency_graph(traced)
# mul depends on: ['x']
# add depends on: ['mul']
# relu depends on: ['add']
# → Linear chain: perfect fusion candidate!
```

---

## 6. Shape Propagation and Type Inference

A critical feature of ML IRs is **shape propagation** — deducing output tensor shapes from input shapes and operation semantics:

```python
from torch.fx.passes.shape_prop import ShapeProp

model = TinyModel()
traced = symbolic_trace(model)

# Run shape propagation with a sample input
sample = torch.randn(8, 4)
ShapeProp(traced).propagate(sample)

# Now each node has shape metadata
for node in traced.graph.nodes:
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        meta = node.meta['tensor_meta']
        print(f"  {node.name}: shape={meta.shape}, dtype={meta.dtype}")
```

Expected output:
```
  x:        shape=torch.Size([8, 4]), dtype=torch.float32
  linear_1: shape=torch.Size([8, 3]), dtype=torch.float32
  relu:     shape=torch.Size([8, 3]), dtype=torch.float32
```

Shape information enables:
- **Memory planning**: allocate exact buffer sizes
- **Tiling decisions**: choose tile sizes that divide tensor dimensions
- **Layout optimization**: decide NCHW vs NHWC based on actual dimensions

---

## Hands-On Exercises

### Exercise 1: Build and Inspect a Graph (25 min)

```python
import torch
from torch.fx import symbolic_trace

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        h = torch.relu(self.linear1(x))
        h = self.linear2(h)
        return x + h  # residual connection

model = ResidualBlock(64)
traced = symbolic_trace(model)

# Tasks:
# 1. Print the tabular graph
# 2. Count nodes by opcode type
# 3. Identify the residual (skip) connection in the graph
# 4. Which nodes could potentially be fused?
# 5. Draw the DAG on paper — is it a pure chain or does it branch?
```

### Exercise 2: Compare IR Representations (25 min)

```python
# Export the same model to ONNX
model = ResidualBlock(64)
x = torch.randn(1, 64)
torch.onnx.export(model, x, "/tmp/residual.onnx",
                  input_names=['input'], output_names=['output'])

# Load and inspect ONNX
import onnx
onnx_model = onnx.load("/tmp/residual.onnx")
for node in onnx_model.graph.node:
    inputs = ', '.join(node.input)
    outputs = ', '.join(node.output)
    print(f"  {node.op_type}: ({inputs}) -> ({outputs})")

# Compare:
# - How many nodes in FX vs ONNX?
# - Are the operations named the same?
# - Which representation is more explicit?
```

### Exercise 3: Graph Surgery with FX (20 min)

```python
import torch
from torch.fx import symbolic_trace

class Model(torch.nn.Module):
    def forward(self, x):
        a = torch.relu(x)
        b = torch.sigmoid(a)
        return b

traced = symbolic_trace(Model())

# Replace relu with leaky_relu by modifying the graph
for node in traced.graph.nodes:
    if node.target == torch.relu:
        node.target = torch.nn.functional.leaky_relu

traced.graph.lint()  # Verify graph integrity
traced.recompile()   # Regenerate forward() method

# Test it
x = torch.randn(4)
print(f"Original relu(-1) = 0, but leaky_relu(-1) = {traced(torch.tensor([-1.0]))}")
```

---

## Key Takeaways

1. **Neural networks are naturally DAGs** — nodes are ops, edges are tensors, and SSA form is automatic
2. **Five major IR families** exist (FX, TorchScript, ONNX, HLO, Relay) with different tradeoffs in typing, abstraction level, and control flow support
3. **SSA + graph IR** make optimizations trivial: dead code = no users, CSE = same op + same inputs
4. **Shape propagation** through the graph enables memory planning, tiling, and layout decisions
5. **FX provides Python-native** graph construction, inspection, and transformation — the lowest barrier to entry for IR manipulation

---

## Further Reading

- Reed, J. et al. — *torch.fx: Practical Program Capture and Transformation for Deep Learning in Python* (MLSys 2022)
- Roesch, J. et al. — *Relay: A High-Level Compiler for Deep Learning* (2019)
- ONNX Specification — [github.com/onnx/onnx](https://github.com/onnx/onnx/blob/main/docs/IR.md)
- XLA Architecture — [openxla.org/xla/architecture](https://openxla.org/xla/architecture)
- StableHLO Specification — [github.com/openxla/stablehlo](https://github.com/openxla/stablehlo)

---

## Tomorrow's Preview

**Day 17** takes these graphs and **optimizes them** — constant folding, dead code elimination, common subexpression elimination, and the crown jewel: **operator fusion**. We'll implement real FX transform passes using pattern matching and measure the impact on kernel count and execution time.
