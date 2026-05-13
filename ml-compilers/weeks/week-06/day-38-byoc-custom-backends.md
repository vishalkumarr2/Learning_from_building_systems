# Day 38: BYOC — Bring Your Own Codegen

> **Phase III · Week 6 · Day 38 of 70 · 2.5 hours**

*"The best compiler is one that knows when not to compile — and delegates to the expert instead."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 37: MetaSchedule](day-37-metaschedule.md) | [Day 39: Quantization in TVM](day-39-quantization-in-tvm.md) | [Week 6: TVM Tuning & Backends](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

TVM's auto-tuning can produce excellent kernels, but some operators on some hardware are best handled by vendor libraries — cuDNN's convolutions on NVIDIA GPUs, oneDNN (DNNL) on Intel CPUs, or TensorRT's fused subgraphs. Rather than forcing an all-or-nothing choice, TVM's **BYOC (Bring Your Own Codegen)** framework lets you partition a model's graph: offload matched subgraphs to external backends while TVM compiles the rest. This hybrid approach gives you the best of both worlds — vendor-optimized hotspots plus TVM's flexibility for everything else.

---

## 1. The BYOC Architecture

### High-Level Flow

```
Input Relay IR
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Step 1: Pattern Matching & Annotation             │
│   Identify subgraphs that an external backend     │
│   can handle (e.g., conv2d+bias+relu for cuDNN)   │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Step 2: Partitioning                              │
│   Extract matched subgraphs into separate         │
│   "composite functions" with a compiler tag       │
│   (e.g., "dnnl" or "tensorrt")                    │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┬────────────────────────────┐
│ TVM-compiled region │ External backend region     │
│ (standard relay.    │ (codegen/runtime provided   │
│  build pipeline)    │  by BYOC integration)       │
└─────────────────────┴────────────────────────────┘
    │                       │
    ▼                       ▼
┌──────────────────────────────────────────────────┐
│ Step 3: Unified Runtime                           │
│   Single GraphExecutor dispatches between         │
│   TVM kernels and external library calls          │
└──────────────────────────────────────────────────┘
```

### Why Not Just Use the Vendor Library Directly?

| Approach | Pros | Cons |
|:--|:--|:--|
| Vendor library only | Maximum perf for supported ops | Can't handle unsupported ops, no graph optimization |
| TVM only | Full control, portable | May trail vendor on specific ops |
| **BYOC hybrid** | Best of both, graceful fallback | Slightly more complex setup |

---

## 2. Pattern Matching for Offloading

BYOC uses Relay's **pattern language** to identify subgraphs suitable for offloading. Patterns describe operator compositions that the external backend supports.

### Defining Patterns

```python
from tvm.relay import dataflow_pattern as dfp

def make_conv2d_bias_relu_pattern():
    """Match: conv2d → bias_add → relu (common fused op in cuDNN/DNNL)."""
    data = dfp.wildcard()
    weight = dfp.wildcard()
    bias = dfp.wildcard()

    # Conv2D
    conv = dfp.is_op("nn.conv2d")(data, weight)
    # Bias add
    biased = dfp.is_op("nn.bias_add")(conv, bias)
    # ReLU activation
    relu = dfp.is_op("nn.relu")(biased)

    return relu

def make_dense_pattern():
    """Match: dense (fully connected) layer."""
    data = dfp.wildcard()
    weight = dfp.wildcard()
    return dfp.is_op("nn.dense")(data, weight)

# Collect patterns with names
patterns = [
    ("dnnl.conv2d_bias_relu", make_conv2d_bias_relu_pattern()),
    ("dnnl.dense", make_dense_pattern()),
]
```

### Pattern Matching in Action

```
Before partitioning:
┌───────────────────────────────────────────────────┐
│                 Original Relay IR                  │
│                                                    │
│  input ──▶ conv2d ──▶ bias_add ──▶ relu ──▶ ...   │
│               ↑           ↑                        │
│            weight       bias                       │
│                                                    │
│         ... ──▶ add ──▶ dense ──▶ softmax          │
│                           ↑                        │
│                        weight                      │
└───────────────────────────────────────────────────┘

After partitioning (patterns matched):
┌───────────────────────────────────────────────────┐
│                                                    │
│         ┌──────────────────────────┐               │
│  input ─┤ dnnl.conv2d_bias_relu    ├──▶ ...        │
│  weight ─┤  (offloaded to DNNL)    │               │
│  bias  ──┤                          │               │
│         └──────────────────────────┘               │
│                                                    │
│         ... ──▶ add ──▶ ┌──────────┐ ──▶ softmax   │
│                  ↑      │dnnl.dense│      (TVM)    │
│                (TVM)    └──────────┘               │
│                         (offloaded)                │
└───────────────────────────────────────────────────┘
```

---

## 3. Graph Partitioning

After pattern matching, the `MergeComposite` and `PartitionGraph` passes restructure the IR:

```python
import tvm
import tvm.relay as relay
from tvm.relay.op.contrib import dnnl  # DNNL integration

# Load a model
mod, params = relay.frontend.from_pytorch(traced_model, input_infos)

# Step 1: Merge matched patterns into composite functions
mod = relay.transform.MergeComposite(patterns)(mod)

# Step 2: Annotate composite functions with the target backend
mod = relay.transform.AnnotateTarget(["dnnl"])(mod)

# Step 3: Merge adjacent annotated regions
mod = relay.transform.MergeCompilerRegions()(mod)

# Step 4: Partition into separate functions per backend
mod = relay.transform.PartitionGraph()(mod)

# Inspect the result
print(mod.astext())
```

### What the Partitioned IR Looks Like

```
// Functions offloaded to DNNL:
def @dnnl_0(%data, %weight, %bias) -> Tensor
    // compiler = "dnnl"
    %0 = nn.conv2d(%data, %weight, ...)
    %1 = nn.bias_add(%0, %bias)
    nn.relu(%1)

// Main function — TVM compiles this, calls into DNNL for tagged functions:
def @main(%input) {
    %0 = @dnnl_0(%input, meta[relay.Constant][0], meta[relay.Constant][1])
    %1 = nn.max_pool2d(%0, ...)       // TVM handles this
    %2 = @dnnl_1(%1, ...)             // another DNNL region
    nn.softmax(%2)                     // TVM handles this
}
```

---

## 4. The BYOC Codegen Interface

Each BYOC backend implements two components:

### 4a. Codegen — Relay to External Representation

```cpp
// Simplified C++ codegen interface
class DNNLCodegen : public CSourceModuleCodegenBase {
 public:
  // Convert a Relay function to external code (C source, binary blob, etc.)
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    auto func = Downcast<Function>(ref);
    // Walk the Relay subgraph, emit DNNL API calls
    std::string code = EmitDNNLCode(func);
    return CSourceModuleCreate(code, "c", {}, {});
  }
};
```

### 4b. Runtime — Execute the External Code

```cpp
// Simplified runtime wrapper
class DNNLRuntime : public ModuleNode {
 public:
  PackedFunc GetFunction(const std::string& name, ...) override {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
      // Extract input tensors from TVM runtime
      DLTensor* input = args[0];
      DLTensor* output = args[1];
      
      // Call DNNL primitive
      dnnl::memory src_mem(..., input->data);
      dnnl::memory dst_mem(..., output->data);
      conv_primitive.execute(stream, {{DNNL_ARG_SRC, src_mem},
                                      {DNNL_ARG_DST, dst_mem}});
      stream.wait();
    });
  }
};
```

### Registration

```python
# Python-side registration (using the JSON runtime approach)
from tvm.relay.op.contrib.register import register_pattern_table

@register_pattern_table("dnnl")
def dnnl_pattern_table():
    return [
        ("dnnl.conv2d_bias_relu", make_conv2d_bias_relu_pattern(), check_conv2d),
        ("dnnl.dense", make_dense_pattern(), check_dense),
    ]

def check_conv2d(extract):
    """Additional checks: data types, layout, padding constraints."""
    call = extract
    attrs = call.args[0].attrs  # conv2d attributes
    return attrs.data_layout == "NCHW" and attrs.kernel_layout == "OIHW"
```

---

## 5. Existing BYOC Integrations

TVM ships with several production-ready BYOC backends:

| Backend | Target Hardware | Supported Ops | Status |
|:--|:--|:--|:--|
| **DNNL (oneDNN)** | Intel CPU | Conv, Dense, BatchNorm, Pool | Stable |
| **cuDNN** | NVIDIA GPU | Conv, Pool, Softmax, BatchNorm | Stable |
| **TensorRT** | NVIDIA GPU | Full subgraph execution | Stable |
| **ACL (Arm Compute)** | Arm CPU | Conv, Dense, Pool, Activation | Stable |
| **CUTLASS** | NVIDIA GPU | GEMM, Conv2d (via templates) | Experimental |
| **Ethosn** | Arm Ethos-N NPU | Conv, Pool, Concat, Split | Stable |
| **CMSIS-NN** | Arm Cortex-M | Quantized Conv, Dense, Pool | Stable |
| **BNNS** | Apple Silicon | Conv, Dense, Activation | Experimental |

### Using TensorRT via BYOC

```python
from tvm.relay.op.contrib import tensorrt

# Partition for TensorRT
mod = tensorrt.partition_for_tensorrt(mod, params, target="cuda")

# Build — TVM compiles non-TRT parts, TRT compiles its subgraphs
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="cuda", params=params)

# At runtime, TVM calls into TRT for offloaded subgraphs
dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input", input_data)
module.run()
output = module.get_output(0)
```

---

## 6. When to Use BYOC vs Native TVM

### Decision Framework

```
Is the operator supported by a vendor library?
    │
    ├── No → Use TVM (auto-tune with MetaSchedule)
    │
    └── Yes
         │
         Is the vendor library significantly faster?
             │
             ├── No → Use TVM (more portable)
             │
             └── Yes
                  │
                  Is the operator a performance hotspot?
                      │
                      ├── No → Use TVM (simpler deployment)
                      │
                      └── Yes → Use BYOC offloading
```

### Practical Guidelines

| Scenario | Recommendation |
|:--|:--|
| Large Conv2D on NVIDIA | BYOC to cuDNN or TensorRT |
| GEMM on Intel CPU | BYOC to DNNL (oneDNN) |
| Custom/exotic operator | TVM native (no library support) |
| Quantized inference on Arm | BYOC to ACL or CMSIS-NN |
| Portable deployment | TVM native (no library dependency) |
| Maximum throughput, single target | BYOC with TensorRT |

---

## Hands-On Exercises

### Exercise 1: Partition ResNet-18 for DNNL (30 min)

Using the DNNL pattern table, partition a ResNet-18 model:

```python
from tvm.relay.op.contrib import dnnl

# Load ResNet-18
mod, params = relay.frontend.from_pytorch(traced_resnet, input_infos)

# Partition
mod = dnnl.partition_for_dnnl(mod, params)
print(mod.astext()[:3000])
```

Questions:
1. How many functions are offloaded to DNNL?
2. Which operators remain in TVM's domain?
3. Benchmark DNNL-partitioned vs pure-TVM compilation.

### Exercise 2: Write a Custom Pattern (25 min)

Define a pattern for `conv2d → batch_norm → relu` (the most common ResNet block). Register it as a custom BYOC backend called `"mybackend"`:

```python
def make_conv_bn_relu_pattern():
    data = dfp.wildcard()
    weight = dfp.wildcard()
    bn_gamma = dfp.wildcard()
    bn_beta = dfp.wildcard()
    bn_mean = dfp.wildcard()
    bn_var = dfp.wildcard()
    
    conv = dfp.is_op("nn.conv2d")(data, weight)
    bn = dfp.is_op("nn.batch_norm")(conv, bn_gamma, bn_beta, bn_mean, bn_var)
    # batch_norm returns a tuple — take element 0
    bn_out = dfp.is_tuple_get_item(bn, 0)
    relu = dfp.is_op("nn.relu")(bn_out)
    return relu
```

Verify the pattern matches on a ResNet-18 IR and count how many instances are found.

### Exercise 3: BYOC Profiling (20 min)

Compile MobileNetV2 three ways and benchmark:
1. Pure TVM with `opt_level=3` (no tuning)
2. TVM with MetaSchedule tuning (500 trials)
3. BYOC with DNNL (or cuDNN on GPU)

Which is fastest? Where does BYOC win and where does tuned TVM win?

---

## Key Takeaways

1. **BYOC enables hybrid compilation** — offload hot operators to vendor libraries, compile the rest with TVM
2. **Pattern matching** identifies subgraphs suitable for offloading (e.g., conv2d+bias+relu → cuDNN)
3. **Graph partitioning** splits the IR into backend-specific regions with a unified runtime
4. **The codegen interface** is two parts: compile (Relay → external code) and runtime (execute external code)
5. **Existing integrations** cover major backends: DNNL, cuDNN, TensorRT, ACL, CMSIS-NN
6. **Use BYOC when** a vendor library is significantly faster for a hotspot operator; use native TVM for portability

---

## Further Reading

- [BYOC Framework (TVM docs)](https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html) — official implementation guide
- [Collage: Automated Integration of Deep Learning Backends](https://arxiv.org/abs/2111.00655) — automated BYOC partitioning research
- [DNNL Integration Tutorial](https://tvm.apache.org/docs/how_to/work_with_relay/using_external_lib.html) — hands-on DNNL offloading
- [TensorRT Integration in TVM](https://tvm.apache.org/docs/how_to/deploy/tensorrt.html) — TensorRT via BYOC

---

## Tomorrow's Preview

Offloading to vendor libraries is powerful, but what about **quantized** inference — running models in INT8 instead of FP32 for 2–4× speedups? **Day 39** covers TVM's quantization pipeline: calibration, the QNN dialect, INT8 compilation, and how to balance accuracy against latency.
