# Day 30: Relay IR

> **Phase III · Week 5 · Day 30 of 70 · 2.5 hours**

*"Relay is a functional IR — every transformation is a function that takes a graph and returns a new graph. No mutation, no side effects, no debugging nightmares."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 29: TVM Architecture Overview](day-29-tvm-architecture-overview.md) | [Day 31: Relay Optimization Passes](day-31-relay-optimization-passes.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Every model that enters TVM — whether from PyTorch, TensorFlow, or ONNX — is first converted into **Relay IR**. Relay is TVM's graph-level intermediate representation, and it's where high-level optimizations like operator fusion and constant folding happen. Unlike PyTorch's FX Graph (a flat DAG of operations), Relay is a **functional language** with a proper type system, let-bindings, closures, and algebraic data types. Understanding Relay is essential because every optimization pass in TVM's graph layer operates on Relay programs.

---

## 1. What Is Relay?

### Relay as a Functional Language

Relay is not just a "graph format" — it's a full programming language with:

```
┌────────────────────────────────────────────────────┐
│                    Relay IR                        │
│                                                    │
│  • First-class functions (closures, higher-order)  │
│  • Let-bindings (SSA-like, no mutation)             │
│  • Pattern matching on ADTs                        │
│  • Rich type system (Tensor, Tuple, Function, Ref) │
│  • Call-by-value evaluation                        │
│  • Differentiable (supports AD natively)           │
└────────────────────────────────────────────────────┘
```

### Relay vs Other High-Level IRs

| Feature | FX Graph (PyTorch) | HLO (XLA) | Relay (TVM) | ONNX |
|:--|:--|:--|:--|:--|
| Paradigm | Imperative DAG | Functional ops | Functional language | Serialization format |
| Type system | Minimal | Shaped arrays | Rich (generics, ADTs) | Tensor types |
| Control flow | Graph breaks | While/Cond ops | If/Match/Recursion | If/Loop |
| Functions | None (flat graph) | HloComputation | First-class closures | Subgraphs |
| Let-bindings | No (node list) | No | Yes (SSA-like) | No |
| Differentiable | Via AOTAutograd | Via HLO AD | Built-in AD pass | No |

### Why Functional?

```python
# Imperative (FX-style): mutations are possible, order matters
graph.nodes[3].args = (new_input,)  # Mutating in place — dangerous!

# Functional (Relay-style): no mutation, always produce new IR
new_expr = relay.Call(relay.op.get("nn.relu"), [old_expr])
# old_expr is unchanged — safe for parallel passes
```

The functional approach means:
- Passes can't corrupt shared state
- Dead code is automatically collected
- Transformation correctness is easier to verify

---

## 2. Relay Type System

Relay has a static type system that catches shape mismatches at compile time, not at runtime.

### Tensor Types

```python
import tvm
from tvm import relay

# TensorType: shape + dtype
t1 = relay.TensorType((1, 3, 224, 224), "float32")
print(t1)  # Tensor[(1, 3, 224, 224), float32]

# Shape can include symbolic dimensions
n = tvm.tir.Var("n", "int32")
t_dynamic = relay.TensorType((n, 3, 224, 224), "float32")
print(t_dynamic)  # Tensor[(n, 3, 224, 224), float32]
```

### Function Types

```
FuncType(arg_types, ret_type, type_params, type_constraints)

Example: a conv2d operation
  fn(Tensor[(N, C_in, H, W), f32],     ← input
     Tensor[(C_out, C_in, kH, kW), f32] ← weight
  ) -> Tensor[(N, C_out, H', W'), f32]  ← output
```

### Tuple Types

```python
# Tuples group multiple values
tt = relay.TupleType([
    relay.TensorType((1, 64, 56, 56), "float32"),
    relay.TensorType((64,), "float32"),
    relay.TensorType((64,), "float32"),
])
# Used by batch_norm which returns (output, running_mean, running_var)
```

### Type Inference

Relay has a Hindley-Milner-style type inference engine:

```python
# Types are inferred automatically — you don't need to annotate
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
w = relay.var("w", shape=(64, 3, 7, 7), dtype="float32")
y = relay.nn.conv2d(x, w, strides=(2, 2), padding=(3, 3))

# After type inference:
mod = tvm.IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
print(mod["main"].ret_type)
# → Tensor[(1, 64, 112, 112), float32]
```

The type system enforces correctness:

$$\text{Output height} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1 = \left\lfloor \frac{224 + 6 - 7}{2} \right\rfloor + 1 = 112$$

---

## 3. Constructing Relay Programs

### From Primitives (Manual Construction)

```python
import tvm
from tvm import relay
import numpy as np

# Build a simple network: Linear → ReLU → Linear
# y = relu(x @ W1 + b1) @ W2 + b2

x = relay.var("x", shape=(1, 784), dtype="float32")
w1 = relay.var("w1", shape=(784, 256), dtype="float32")
b1 = relay.var("b1", shape=(256,), dtype="float32")
w2 = relay.var("w2", shape=(256, 10), dtype="float32")
b2 = relay.var("b2", shape=(10,), dtype="float32")

# Build computation graph
dense1 = relay.nn.dense(x, w1)        # (1, 256)
biased1 = relay.add(dense1, b1)       # (1, 256) — broadcast
act = relay.nn.relu(biased1)          # (1, 256)
dense2 = relay.nn.dense(act, w2)      # (1, 10)
out = relay.add(dense2, b2)           # (1, 10)

# Create a Relay function
func = relay.Function([x, w1, b1, w2, b2], out)
mod = tvm.IRModule.from_expr(func)

# Print the IR
print(mod)
```

Output:

```
def @main(%x: Tensor[(1, 784), float32],
          %w1: Tensor[(784, 256), float32],
          %b1: Tensor[(256), float32],
          %w2: Tensor[(256, 10), float32],
          %b2: Tensor[(10), float32]) {
  %0 = nn.dense(%x, %w1, units=256);
  %1 = add(%0, %b1);
  %2 = nn.relu(%1);
  %3 = nn.dense(%2, %w2, units=10);
  add(%3, %b2)
}
```

### Using Let-Bindings (Explicit Naming)

```python
# Let-bindings give names to intermediate values (SSA form)
x = relay.var("x", shape=(1, 784), dtype="float32")
w = relay.var("w", shape=(784, 256), dtype="float32")

# Explicit let-binding
v0 = relay.var("v0")
v1 = relay.var("v1")
body = relay.Let(v0, relay.nn.dense(x, w),
            relay.Let(v1, relay.nn.relu(v0),
                v1))

# Equivalent to:
#   let v0 = nn.dense(x, w) in
#   let v1 = nn.relu(v0) in
#   v1
```

### The Difference

```
Without let-bindings (expression tree):
  add(nn.relu(nn.dense(x, w1)), b1)
  └── if nn.dense appears in two places, it's DUPLICATED

With let-bindings (SSA-like):
  let %0 = nn.dense(x, w1) in
  let %1 = nn.relu(%0) in
  add(%1, b1)
  └── %0 is computed once, referenced by name
```

---

## 4. Frontend Importers

TVM includes importers for all major frameworks. They all produce Relay IR.

### ONNX Import

```python
import onnx
from tvm import relay

onnx_model = onnx.load("model.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": (1, 3, 224, 224)})
# mod:    tvm.IRModule containing the Relay function
# params: dict of parameter name → numpy array
```

### PyTorch Import

```python
import torch
from tvm import relay

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

# Convert TorchScript → Relay
input_infos = [("input", (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(scripted, input_infos)
```

### TensorFlow Import

```python
from tvm import relay

# From SavedModel
mod, params = relay.frontend.from_tensorflow(
    graph_def,           # tf.GraphDef protobuf
    layout="NHWC",       # TF default layout
    shape={"input": (1, 224, 224, 3)}
)
```

### What the Importers Do

```
Framework Model
     │
     ▼
┌─────────────────────────────────────────┐
│  Frontend Importer                      │
│                                         │
│  1. Parse framework-specific format     │
│  2. Map ops:  torch.nn.Conv2d           │
│             → relay.nn.conv2d           │
│  3. Handle layout:  NHWC ↔ NCHW        │
│  4. Extract weights as numpy arrays     │
│  5. Build Relay function + params dict  │
└─────────────────────────────────────────┘
     │
     ▼
  (IRModule, params: Dict[str, np.ndarray])
```

### Operator Mapping Examples

| PyTorch | ONNX | Relay |
|:--|:--|:--|
| `torch.nn.Conv2d` | `Conv` | `relay.nn.conv2d` |
| `torch.nn.BatchNorm2d` | `BatchNormalization` | `relay.nn.batch_norm` |
| `torch.nn.ReLU` | `Relu` | `relay.nn.relu` |
| `torch.matmul` | `MatMul` | `relay.nn.dense` or `relay.nn.batch_matmul` |
| `torch.nn.Softmax` | `Softmax` | `relay.nn.softmax` |
| `torch.cat` | `Concat` | `relay.concatenate` |

---

## 5. Visualization and Inspection

### Text Dump

```python
# Pretty-print the entire module
print(mod.astext(show_meta_data=False))

# Print just the main function signature
func = mod["main"]
print(f"Inputs:  {[p.name_hint for p in func.params]}")
print(f"Return:  {func.ret_type}")
```

### Graphviz Visualization

```python
from tvm.contrib import relay_viz

# Generate a graphviz DOT visualization
viz = relay_viz.RelayVisualizer(mod)
viz.render("relay_graph")  # Creates relay_graph.pdf

# Or as SVG for notebooks
viz.render("relay_graph", format="svg")
```

### Counting Operations

```python
def analyze_relay(mod):
    """Count ops, parameters, and compute cost."""
    op_counts = {}
    param_count = 0
    
    def visit(node):
        nonlocal param_count
        if isinstance(node, relay.Call) and hasattr(node.op, "name"):
            name = node.op.name
            op_counts[name] = op_counts.get(name, 0) + 1
    
    relay.analysis.post_order_visit(mod["main"], visit)
    
    # Count parameters
    for name, param in params.items():
        param_count += np.prod(param.shape)
    
    print("Operation counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
    print(f"\nTotal parameters: {param_count:,}")

analyze_relay(mod)
# nn.conv2d: 20
# nn.batch_norm: 20
# nn.relu: 17
# add: 1
# nn.dense: 1
# nn.global_avg_pool2d: 1
# nn.max_pool2d: 1
# nn.softmax: 0
# Total parameters: 11,689,512
```

---

## 6. Relay Expression Types Deep Dive

### The Expression Hierarchy

```
relay.Expr (base class)
├── relay.Constant       ← literal tensor value
├── relay.Var            ← named variable
├── relay.Call           ← function application (op invocation)
├── relay.Let            ← let-binding
├── relay.If             ← conditional
├── relay.Tuple          ← tuple construction
├── relay.TupleGetItem   ← tuple element access
├── relay.Function       ← function definition
├── relay.GlobalVar      ← reference to a global function
└── relay.RefCreate      ← mutable reference (rarely used)
    relay.RefRead
    relay.RefWrite
```

### Traversing Relay Expressions

```python
# Post-order visitor (most common pattern)
class OpCounter(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.ops = {}
    
    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            name = call.op.name
            self.ops[name] = self.ops.get(name, 0) + 1
        # MUST call super to continue traversal
        super().visit_call(call)

counter = OpCounter()
counter.visit(mod["main"])
print(counter.ops)

# Mutating visitor (creates new IR)
class ReLUReplacer(relay.ExprMutator):
    """Replace all nn.relu with nn.leaky_relu."""
    def visit_call(self, call):
        call = super().visit_call(call)  # recurse first
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.relu":
            return relay.nn.leaky_relu(call.args[0], alpha=0.01)
        return call

replacer = ReLUReplacer()
new_func = replacer.visit(mod["main"])
```

---

## Hands-On Exercises

### Exercise 1: Build a ConvNet in Relay

```python
import tvm
from tvm import relay
import numpy as np

def build_simple_cnn():
    """Build: Conv2d(3→16) → ReLU → MaxPool → Flatten → Dense(10)."""
    x = relay.var("x", shape=(1, 3, 32, 32), dtype="float32")
    
    # Conv2d: 3 input channels → 16 output channels, 3×3 kernel
    w_conv = relay.var("w_conv", shape=(16, 3, 3, 3), dtype="float32")
    conv = relay.nn.conv2d(x, w_conv, kernel_size=(3, 3), padding=(1, 1))
    act = relay.nn.relu(conv)
    pool = relay.nn.max_pool2d(act, pool_size=(2, 2), strides=(2, 2))
    
    # Flatten: (1, 16, 16, 16) → (1, 4096)
    flat = relay.nn.batch_flatten(pool)
    
    # Dense: 4096 → 10
    w_dense = relay.var("w_dense", shape=(10, 4096), dtype="float32")
    b_dense = relay.var("b_dense", shape=(10,), dtype="float32")
    dense = relay.nn.dense(flat, w_dense)
    out = relay.add(dense, b_dense)
    
    func = relay.Function([x, w_conv, w_dense, b_dense], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    
    return mod

mod = build_simple_cnn()
print(mod)
# Verify: check the inferred output shape is (1, 10)
```

### Exercise 2: Import and Compare Two Frameworks

```python
# Import the same model from ONNX and PyTorch, compare Relay IR
import torchvision, torch, onnx

model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()

# Path A: PyTorch → Relay
scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
mod_pt, _ = relay.frontend.from_pytorch(scripted, [("input", (1, 3, 224, 224))])

# Path B: PyTorch → ONNX → Relay
torch.onnx.export(model, torch.randn(1, 3, 224, 224), "/tmp/mbv2.onnx",
                  input_names=["input"])
mod_onnx, _ = relay.frontend.from_onnx(onnx.load("/tmp/mbv2.onnx"),
                                        {"input": (1, 3, 224, 224)})

# Count ops in each — are they identical?
# (They may differ slightly due to different op decompositions)
```

### Exercise 3: Write a Custom Relay Visitor

```python
class ShapeAnalyzer(relay.ExprVisitor):
    """Track tensor shapes at every Call node."""
    def __init__(self):
        super().__init__()
        self.shapes = []
    
    def visit_call(self, call):
        super().visit_call(call)
        if call.checked_type and isinstance(call.checked_type, relay.TensorType):
            shape = [int(x) for x in call.checked_type.shape]
            op_name = call.op.name if isinstance(call.op, tvm.ir.Op) else "fused"
            self.shapes.append((op_name, shape))

# Run on type-inferred module
mod_typed = relay.transform.InferType()(mod)
analyzer = ShapeAnalyzer()
analyzer.visit(mod_typed["main"])
for op, shape in analyzer.shapes:
    print(f"  {op:30s} → {shape}")
```

---

## Key Takeaways

1. **Relay is a functional IR** — no mutation, explicit let-bindings, first-class functions — making transformations safe and composable
2. **Rich type system** with Tensor types, Tuple types, and Function types catches shape errors at compile time via Hindley-Milner inference
3. **Frontend importers** translate from PyTorch (via TorchScript), TensorFlow, ONNX, and others into a unified Relay representation
4. **`ExprVisitor` and `ExprMutator`** are the building blocks for writing custom analysis and transformation passes over Relay IR
5. **Let-bindings** prevent expression duplication and establish SSA-like form for efficient optimization
6. Relay sits at the **same level as FX Graph or HLO** but with a richer type system and true functional semantics

---

## Further Reading

- [Relay: A New IR for Machine Learning Frameworks](https://arxiv.org/abs/1810.00952) — original Relay paper (Roesch et al., 2018)
- [TVM Relay Documentation](https://tvm.apache.org/docs/reference/langref/relay_expr.html) — expression language reference
- [Relay Type System](https://tvm.apache.org/docs/reference/langref/relay_type.html) — type system specification
- [TVM Frontend Tutorials](https://tvm.apache.org/docs/how_to/compile_models/index.html) — import from PyTorch, TF, ONNX

---

## Tomorrow

**Day 31: Relay Optimization Passes** — Now that you can read and construct Relay programs, we'll learn the pass infrastructure that transforms them. You'll see how FuseOps merges operators for memory efficiency, how FoldConstant pre-computes static expressions, and how to write your own custom Relay passes.
