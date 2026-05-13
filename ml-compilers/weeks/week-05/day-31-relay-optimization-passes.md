# Day 31: Relay Optimization Passes

> **Phase III · Week 5 · Day 31 of 70 · 2.5 hours**

*"A compiler pass is a function from IR to IR. Stack enough good passes in the right order, and your naive specification becomes production-quality code."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 30: Relay IR](day-30-relay-ir.md) | [Day 32: Tensor Expression (TE)](day-32-tensor-expression-te.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

A model imported into Relay is correct but unoptimized — every conv, batch norm, relu, and add is a separate operation with its own memory allocation and kernel launch. **Relay optimization passes** transform this naive IR into efficient code by fusing operations, folding constants, simplifying batch normalization, and rearranging data layouts. These passes are what make TVM competitive with hand-optimized libraries like cuDNN. Today you'll learn TVM's pass infrastructure, study the most important built-in passes, and write your own custom passes.

---

## 1. The Pass Infrastructure

### Pass Types

TVM organizes passes into three categories by scope:

```
┌─────────────────────────────────────────────────────────┐
│  ModulePass                                             │
│  • Operates on the entire IRModule                      │
│  • Can add/remove/modify global functions               │
│  • Example: InlinePrimitives, LambdaLift                │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  FunctionPass                                     │  │
│  │  • Operates on each function independently        │  │
│  │  • Most common type                               │  │
│  │  • Example: FuseOps, FoldConstant, InferType      │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Sequential Pass                            │  │  │
│  │  │  • Chains multiple passes in order          │  │  │
│  │  │  • Groups related transformations           │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### PassContext

Every pass runs inside a `PassContext` that controls optimization levels and configuration:

```python
import tvm
from tvm import relay

# opt_level controls which passes run
# 0: minimal (type inference only)
# 1: basic optimizations (fold constant, eliminate dead code)
# 2: more aggressive (fusion, layout transform)
# 3: all optimizations (everything)
# 4: experimental (may not be stable)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm")

# Fine-grained control
with tvm.transform.PassContext(
    opt_level=3,
    config={
        "relay.FuseOps.max_depth": 5,           # fusion depth limit
        "relay.backend.use_auto_scheduler": True, # use Ansor
    },
    disabled_pass=["AlterOpLayout"],             # skip specific passes
):
    lib = relay.build(mod, target="llvm")
```

### Running Passes Manually

```python
# Apply a single pass
mod = relay.transform.InferType()(mod)
mod = relay.transform.FoldConstant()(mod)
mod = relay.transform.FuseOps(fuse_opt_level=2)(mod)

# Chain passes with Sequential
seq = tvm.transform.Sequential([
    relay.transform.InferType(),
    relay.transform.SimplifyInference(),
    relay.transform.FoldConstant(),
    relay.transform.FuseOps(fuse_opt_level=2),
])

with tvm.transform.PassContext(opt_level=3):
    mod_optimized = seq(mod)
```

---

## 2. FuseOps — Operator Fusion

**FuseOps** is the most impactful Relay pass. It merges sequences of operators into single fused functions, eliminating intermediate memory allocations and kernel launches.

### Fusion Rules

TVM classifies operators into categories for fusion:

```
Category 0: kElemWise     — element-wise (relu, sigmoid, add, multiply)
Category 1: kBroadcast    — broadcasting (add with broadcast, expand)
Category 2: kInjective    — shape-preserving (reshape, transpose, cast)
Category 3: kCommReduce   — reduction (sum, mean, max along axes)
Category 4: kOutEWiseFusable — complex but fusable output (conv2d, matmul)
Category 7: kOpaque       — cannot be fused (custom ops, side effects)
```

### Fusion Strategy

```
Rule 1: ElemWise + ElemWise → FUSE
  relu(sigmoid(x))  →  fused_relu_sigmoid(x)

Rule 2: OutEWiseFusable + ElemWise → FUSE  
  relu(conv2d(x, w))  →  fused_conv2d_relu(x, w)

Rule 3: Injective + Injective → FUSE
  transpose(reshape(x))  →  fused_reshape_transpose(x)

Rule 4: Reduction + ElemWise → FUSE
  relu(reduce_sum(x))  →  fused_reduce_relu(x)

Rule 5: OutEWiseFusable + OutEWiseFusable → NO FUSE
  matmul(conv2d(x, w), v)  →  kept separate (each too complex)
```

### Fusion in Action

```python
import tvm
from tvm import relay
import numpy as np

# Build: Conv → BatchNorm → ReLU → Add (residual)
x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
w = relay.var("w", shape=(64, 64, 3, 3), dtype="float32")

conv = relay.nn.conv2d(x, w, padding=(1, 1))
bn = relay.nn.batch_norm(conv,
    relay.var("gamma", shape=(64,)),
    relay.var("beta", shape=(64,)),
    relay.var("mean", shape=(64,)),
    relay.var("var", shape=(64,)),
)
relu = relay.nn.relu(bn[0])  # batch_norm returns tuple
out = relay.add(relu, x)      # residual connection

func = relay.Function(relay.analysis.free_vars(out), out)
mod = tvm.IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)

print("=== Before Fusion ===")
print(mod["main"])

# Apply SimplifyInference (fold BN) then FuseOps
mod = relay.transform.SimplifyInference()(mod)
mod = relay.transform.InferType()(mod)
mod = relay.transform.FuseOps(fuse_opt_level=2)(mod)

print("\n=== After Fusion ===")
print(mod["main"])
```

Before fusion:

```
%0 = nn.conv2d(%x, %w, padding=[1, 1, 1, 1])
%1 = nn.batch_norm(%0, %gamma, %beta, %mean, %var)
%2 = %1.0
%3 = nn.relu(%2)
%4 = add(%3, %x)
```

After SimplifyInference + FuseOps:

```
%0 = fn(%p0, %p1, ...) {        ← fused function
  %1 = nn.conv2d(%p0, %p1, ...)
  %2 = multiply(%1, %scale)      ← BN folded into multiply+add
  %3 = add(%2, %bias)
  nn.relu(%3)
}
%4 = %0(%x, %w, %scale, %bias)
%5 = add(%4, %x)                 ← residual add stays separate
```

### Memory Impact

```
Before fusion (5 ops, 4 intermediate tensors):
  Reads:  5 × 64×56×56×4 bytes = 40.1 MB total memory traffic
  Writes: 4 × 64×56×56×4 bytes = 32.1 MB
  Kernel launches: 5

After fusion (2 ops, 1 intermediate tensor):
  Reads:  2 × 64×56×56×4 bytes = 16.0 MB
  Writes: 1 × 64×56×56×4 bytes =  8.0 MB
  Kernel launches: 2
  
  Memory traffic reduction: ~3.6×
```

---

## 3. FoldConstant — Constant Folding

**FoldConstant** evaluates expressions whose inputs are all constants at compile time.

```python
# Before FoldConstant:
%0 = add(relay.const(1.0), relay.const(2.0))   # constant expression
%1 = multiply(%input, %0)

# After FoldConstant:
%0 = multiply(%input, relay.const(3.0))         # pre-computed!
```

### Where Constant Folding Helps

```python
# Shape computations
shape = relay.shape_of(relay.const(np.zeros((1, 3, 224, 224))))
# → relay.const([1, 3, 224, 224])

# BatchNorm parameter computation (after SimplifyInference)
# scale = gamma / sqrt(var + eps)  ← all constants
# → pre-computed as a single const tensor

# Quantization scale factors
# scale = (max_val - min_val) / 255.0  ← all constants
```

```python
mod = relay.transform.FoldConstant()(mod)
```

---

## 4. SimplifyInference — Batch Normalization Folding

**SimplifyInference** eliminates inference-only operations. The biggest win: **folding BatchNorm into the preceding Conv2d**.

### The Math

During training, BatchNorm computes:

$$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

At inference, $\mu$ and $\sigma^2$ are fixed. We can precompute:

$$\text{scale} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad \text{bias} = \beta - \mu \cdot \text{scale}$$

And fold into the convolution:

$$W_{\text{new}} = W \cdot \text{scale}, \quad b_{\text{new}} = b \cdot \text{scale} + \text{bias}$$

### Before vs After

```
Before SimplifyInference:
  %0 = nn.conv2d(%x, %w)               ← conv kernel
  %1 = nn.batch_norm(%0, γ, β, μ, σ²)  ← BN with 5 params
  %2 = nn.relu(%1.0)

After SimplifyInference:
  %0 = nn.conv2d(%x, %w)               ← same conv
  %1 = multiply(%0, %scale)            ← scale = γ/√(σ²+ε)
  %2 = add(%1, %bias)                  ← bias = β - μ·scale
  %3 = nn.relu(%2)

  Then FoldConstant folds scale/bias into the conv weights!
```

Result: **BatchNorm is completely eliminated** — zero runtime cost.

---

## 5. AlterOpLayout — Data Layout Optimization

**AlterOpLayout** changes the memory layout of tensors to match hardware preferences.

### Layout Notation

```
NCHW:    standard PyTorch layout    (batch, channels, height, width)
NHWC:    standard TF layout         (batch, height, width, channels)
NCHW8c:  packed layout for AVX2     (batch, channels/8, height, width, 8)
NCHW16c: packed layout for AVX-512  (batch, channels/16, height, width, 16)
```

### Why Layout Matters

```
NCHW memory layout (Channel-first):
┌─────────────────────────────┐
│ C0: [h0w0 h0w1 h0w2 ...]   │  ← all pixels of channel 0
│ C1: [h0w0 h0w1 h0w2 ...]   │  ← all pixels of channel 1
│ ...                         │
└─────────────────────────────┘

NCHW8c packed layout:
┌─────────────────────────────────────────┐
│ C0-7:  [c0h0w0 c1h0w0 ... c7h0w0       │  ← 8 channels at same spatial pos
│         c0h0w1 c1h0w1 ... c7h0w1 ...]   │  ← next spatial position
└─────────────────────────────────────────┘

The packed layout enables SIMD: AVX2 processes 8 float32 values at once,
and with NCHW8c, those 8 values are 8 channels at the same (h,w) position
— exactly what a conv2d inner loop needs.
```

```python
# AlterOpLayout automatically picks the best layout for the target
target = tvm.target.Target("llvm -mcpu=skylake")  # has AVX2
with tvm.transform.PassContext(opt_level=3):
    # AlterOpLayout will convert NCHW → NCHW8c for conv2d ops
    mod_optimized = relay.transform.AlterOpLayout()(mod)
```

---

## 6. Writing Custom Relay Passes

### Using the Decorator API

```python
@relay.transform.function_pass(opt_level=1)
class ReluReplacer:
    """Replace nn.relu with clip(0, inf) — a custom pass example."""
    
    def transform_function(self, func, mod, ctx):
        class Replacer(relay.ExprMutator):
            def visit_call(self, call):
                call = super().visit_call(call)
                if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.relu":
                    # relu(x) = clip(x, a_min=0, a_max=inf)
                    return relay.clip(call.args[0], a_min=0.0, a_max=float("inf"))
                return call
        
        return Replacer().visit(func)

# Use it
mod_transformed = ReluReplacer()(mod)
```

### A More Practical Pass: Dead Code Elimination

```python
@relay.transform.function_pass(opt_level=1)
class CountAndReport:
    """Report statistics about each function (analysis pass)."""
    
    def transform_function(self, func, mod, ctx):
        op_counts = {}
        
        def count(node):
            if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
                name = node.op.name
                op_counts[name] = op_counts.get(name, 0) + 1
        
        relay.analysis.post_order_visit(func, count)
        
        total = sum(op_counts.values())
        print(f"  Function has {total} ops: {dict(op_counts)}")
        
        return func  # analysis only — return unchanged

# Chain with other passes
pipeline = tvm.transform.Sequential([
    relay.transform.InferType(),
    CountAndReport(),                    # ← our custom pass
    relay.transform.SimplifyInference(),
    CountAndReport(),                    # ← see the difference
    relay.transform.FuseOps(),
    CountAndReport(),                    # ← after fusion
])
```

### Registering as a Named Pass

```python
@tvm.ir.transform.module_pass(opt_level=2, name="MyCustomPass")
class MyCustomPass:
    def transform_module(self, mod, ctx):
        for gvar, func in mod.functions.items():
            if isinstance(func, relay.Function):
                # Transform each function
                new_func = self._optimize(func)
                mod.update_func(gvar, new_func)
        return mod
    
    def _optimize(self, func):
        # Your transformation logic
        return func
```

---

## Hands-On Exercises

### Exercise 1: Observe Pass-by-Pass Transformation

```python
import tvm
from tvm import relay
import onnx

onnx_model = onnx.load("resnet18.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, {"input": (1, 3, 224, 224)})

passes = [
    ("InferType",           relay.transform.InferType()),
    ("SimplifyInference",   relay.transform.SimplifyInference()),
    ("FoldConstant",        relay.transform.FoldConstant()),
    ("EliminateCommonSubexpr", relay.transform.EliminateCommonSubexpr()),
    ("FuseOps",             relay.transform.FuseOps(fuse_opt_level=2)),
]

def count_ops(mod):
    ops = {}
    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            ops[node.op.name] = ops.get(node.op.name, 0) + 1
    relay.analysis.post_order_visit(mod["main"], visit)
    return sum(ops.values()), ops

for name, p in passes:
    before_count, _ = count_ops(mod)
    mod = p(mod)
    after_count, after_ops = count_ops(mod)
    delta = after_count - before_count
    sign = "+" if delta >= 0 else ""
    print(f"  {name:30s} {before_count:3d} → {after_count:3d} ops ({sign}{delta})")
```

### Exercise 2: Write a Fusion Analysis Pass

```python
@relay.transform.function_pass(opt_level=0)
class FusionAnalyzer:
    """Identify fusable chains before FuseOps runs."""
    
    def transform_function(self, func, mod, ctx):
        chains = []
        current_chain = []
        
        def categorize(op_name):
            elemwise = {"nn.relu", "sigmoid", "tanh", "add", "multiply", "subtract"}
            injective = {"reshape", "transpose", "expand_dims", "squeeze"}
            compute = {"nn.conv2d", "nn.dense", "nn.batch_matmul"}
            if op_name in elemwise: return "elemwise"
            if op_name in injective: return "injective"
            if op_name in compute: return "compute"
            return "opaque"
        
        def visit(node):
            if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
                cat = categorize(node.op.name)
                current_chain.append((node.op.name, cat))
        
        relay.analysis.post_order_visit(func, visit)
        
        # Print fusable sequences
        print("  Potential fusion chains:")
        for i, (name, cat) in enumerate(current_chain):
            print(f"    [{i:2d}] {name:30s} ({cat})")
        
        return func  # analysis only

# Run before FuseOps to predict what will be fused
FusionAnalyzer()(mod)
```

### Exercise 3: Benchmark SimplifyInference Impact

```python
import timeit

target = tvm.target.Target("llvm -mcpu=native")

# Compile WITHOUT SimplifyInference
mod_raw, params = relay.frontend.from_onnx(onnx_model, {"input": (1, 3, 224, 224)})
with tvm.transform.PassContext(opt_level=3, disabled_pass=["SimplifyInference"]):
    lib_no_bn_fold = relay.build(mod_raw, target=target, params=params)

# Compile WITH SimplifyInference (default at opt_level=3)
mod_raw2, params2 = relay.frontend.from_onnx(onnx_model, {"input": (1, 3, 224, 224)})
with tvm.transform.PassContext(opt_level=3):
    lib_bn_fold = relay.build(mod_raw2, target=target, params=params2)

# Benchmark both
input_data = np.random.randn(1, 3, 224, 224).astype("float32")

for name, lib in [("No BN fold", lib_no_bn_fold), ("With BN fold", lib_bn_fold)]:
    from tvm.contrib import graph_executor
    dev = tvm.cpu(0)
    ex = graph_executor.GraphModule(lib["default"](dev))
    ex.set_input("input", tvm.nd.array(input_data))
    
    def run():
        ex.run()
    
    for _ in range(20): run()  # warmup
    t = timeit.timeit(run, number=100) / 100
    print(f"  {name:15s}: {t*1000:.2f} ms")

# Expected: BN folding gives ~5-15% speedup on CPU
```

---

## Key Takeaways

1. **Pass infrastructure** organizes transforms into ModulePass, FunctionPass, and Sequential — with `PassContext` controlling what runs at each `opt_level`
2. **FuseOps** is the highest-impact pass — merging element-wise and injective ops into compute ops reduces memory traffic by 2–4× and kernel launches proportionally
3. **SimplifyInference** eliminates BatchNorm at zero runtime cost by folding $\gamma/\sqrt{\sigma^2+\epsilon}$ into convolution weights
4. **FoldConstant** pre-computes compile-time-known expressions, critical after BN folding creates constant scale/bias tensors
5. **AlterOpLayout** changes data layout (e.g., NCHW → NCHW8c) to exploit SIMD hardware — invisible to the user but significant for CPU performance
6. **Custom passes** are straightforward: subclass `ExprMutator`, use the `@function_pass` decorator, and chain into the pipeline with `Sequential`

---

## Further Reading

- [TVM Pass Infrastructure](https://tvm.apache.org/docs/arch/pass_infra.html) — architectural guide to the pass system
- [Writing a Customized Pass](https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_infra.html) — official tutorial
- [Relay Operator Fusion Design](https://github.com/apache/tvm/blob/main/src/relay/transforms/fuse_ops.cc) — implementation source
- [Operator Fusion in Deep Learning](https://arxiv.org/abs/2202.10963) — survey of fusion strategies across compilers

---

## Tomorrow

**Day 32: Tensor Expression (TE) Language** — We'll drop below Relay to TVM's compute layer. You'll learn how to define tensor computations with `te.compute()` and `te.reduce_axis()`, then optimize them using schedule primitives like split, reorder, tile, and vectorize — the foundation of TVM's learning-based optimization.
