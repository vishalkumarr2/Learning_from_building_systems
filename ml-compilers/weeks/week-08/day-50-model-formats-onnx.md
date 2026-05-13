# Day 50: Model Formats & ONNX

> **Phase IV · Week 8 · Day 50 of 70 · 2.5 hours**

*"A model trapped in one framework is a model half-deployed. Portable formats are the lingua franca of production ML."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 49: Stop & Reflect #4](../week-07/day-49-stop-and-reflect-4.md) | [Day 51: Weight Compression & Pruning](day-51-weight-compression.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Every model you train eventually needs to **leave its training framework**. PyTorch is great for research, but production inference runs on TensorRT, CoreML, ONNX Runtime, or custom hardware. Model interchange formats bridge that gap — they serialize the computation graph, weights, and metadata into a framework-agnostic representation. ONNX (Open Neural Network Exchange) dominates this space, backed by Microsoft, Meta, and NVIDIA. Understanding ONNX internals — its protobuf schema, operator sets, shape inference, and graph surgery tools — is essential for anyone deploying models at scale. Today you learn to export, inspect, optimize, and transform ONNX models fluently.

---

## 1. The Model Serialization Landscape

Different formats serve different stages and ecosystems:

```
Model Serialization Formats — The Full Landscape
══════════════════════════════════════════════════════════════════════

  Training Framework          Interchange           Deployment Runtime
  ┌─────────────────┐        ┌──────────────┐       ┌──────────────────┐
  │  PyTorch         │──┬───▶│    ONNX       │──┬──▶│ ONNX Runtime     │
  │  (.pt/.pth)      │  │    │  (.onnx)      │  │   │ TensorRT         │
  ├─────────────────┤  │    ├──────────────┤  │   │ OpenVINO         │
  │  TensorFlow      │──┤    │  SavedModel   │──┤   │ CoreML           │
  │  (.pb/.h5)       │  │    │  (.pb dir)    │  │   │ TFLite           │
  ├─────────────────┤  │    ├──────────────┤  │   ├──────────────────┤
  │  JAX             │──┤    │  StableHLO    │  │   │ XLA (TPU/GPU)    │
  │  (pytree)        │  │    │  (.mlir)      │  │   │ IREE             │
  ├─────────────────┤  │    ├──────────────┤  │   ├──────────────────┤
  │  State dicts     │──┘    │  SafeTensors  │──┘   │ llama.cpp        │
  │  (.safetensors)  │       │  (.safetensors)│      │ vLLM / TGI       │
  └─────────────────┘       ├──────────────┤       └──────────────────┘
                             │  GGUF         │
                             │  (.gguf)       │
                             └──────────────┘
```

| Format | Ecosystem | Key Feature | Typical Size |
|--------|-----------|-------------|--------------|
| **ONNX** | Cross-framework | Standardized opset, wide tooling | Graph + weights in one file |
| **SavedModel** | TensorFlow | Full TF graph with signatures | Directory structure |
| **TorchScript** | PyTorch | Traced/scripted Python → IR | `.pt` archive |
| **SafeTensors** | HuggingFace | Memory-mapped, no pickle exploits | Weights only |
| **GGUF** | llama.cpp | Quantized LLMs, single-file | Weights + metadata |

---

## 2. ONNX Specification Deep Dive

ONNX is defined as a Protocol Buffer schema. Understanding its structure is key to manipulating models programmatically.

### 2.1 Protobuf Schema Hierarchy

```
ONNX Protobuf Structure
════════════════════════════════════════════

  ModelProto
  ├── ir_version: int           (e.g., 9)
  ├── opset_import[]            (e.g., opset 18)
  ├── producer_name: str        (e.g., "pytorch")
  ├── graph: GraphProto
  │   ├── name: str
  │   ├── input[]: ValueInfoProto
  │   │   ├── name: str
  │   │   └── type: TypeProto
  │   │       └── tensor_type
  │   │           ├── elem_type: int  (1=float32)
  │   │           └── shape: TensorShapeProto
  │   ├── output[]: ValueInfoProto
  │   ├── node[]: NodeProto
  │   │   ├── op_type: str      ("Conv", "Relu", ...)
  │   │   ├── input[]: str      (tensor names)
  │   │   ├── output[]: str
  │   │   └── attribute[]: AttributeProto
  │   ├── initializer[]: TensorProto    ← weights!
  │   └── value_info[]: ValueInfoProto  ← intermediate shapes
  └── metadata_props[]
```

### 2.2 Operator Sets (Opsets)

ONNX defines **versioned operator sets** — each opset version adds, modifies, or deprecates operators:

| Opset | Key Additions | ONNX Version |
|-------|---------------|--------------|
| 13 | Squeeze/Unsqueeze axis changes | 1.8 |
| 14 | Reshape allows 0-dim, Add/Mul broadcasting | 1.9 |
| 15 | Shape op with start/end | 1.10 |
| 16 | GridSample, ScatterND updates | 1.11 |
| 17 | LayerNorm, GroupNorm as single ops | 1.12 |
| 18 | Pad edge-case fixes, BitwiseAnd/Or | 1.13 |
| 19 | DeformConv, AveragePool ceil mode | 1.14 |
| 20 | GridSample 5D, DFT inverse | 1.15 |

The opset version determines **which runtime versions can execute your model**. Always export at the lowest opset that supports your operators.

### 2.3 Shape Inference

Shape inference propagates known input dimensions through every node:

$$\text{output\_shape}[i] = f(\text{input\_shapes}, \text{attributes})$$

For convolution:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(k-1) - 1}{s} + 1 \right\rfloor$$

```python
import onnx
from onnx import shape_inference

model = onnx.load("resnet50.onnx")
model = shape_inference.infer_shapes(model)

# Verify all intermediate shapes are populated
for vi in model.graph.value_info:
    shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    print(f"{vi.name}: {shape}")
```

---

## 3. Exporting from PyTorch

### 3.1 Classic torch.onnx.export (TorchScript-based)

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True).eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},       # Dynamic batch
        "logits": {0: "batch_size"},
    },
)
```

**Limitations of TorchScript-based export:**
- Fails on Python control flow that depends on tensor values
- Cannot handle data-dependent shapes well
- Silently drops unsupported operations

### 3.2 Modern Dynamo Export (torch.onnx.dynamo_export)

```python
import torch

model = models.resnet50(pretrained=True).eval()
dummy_input = torch.randn(1, 3, 224, 224)

# Dynamo-based: captures the FX graph, not TorchScript
export_output = torch.onnx.dynamo_export(model, dummy_input)
export_output.save("resnet50_dynamo.onnx")
```

| Feature | TorchScript Export | Dynamo Export |
|---------|-------------------|---------------|
| Control flow | Limited | Full Python support |
| Dynamic shapes | Manual `dynamic_axes` | Automatic via guards |
| Custom ops | Requires scripting | Native FX decomposition |
| Speed | Faster | Slightly slower |
| Maturity | Stable | Evolving (PyTorch 2.x+) |

---

## 4. ONNX Graph Inspection & Validation

### 4.1 Programmatic Inspection

```python
import onnx

model = onnx.load("resnet50.onnx")

# Basic validation
onnx.checker.check_model(model)

# Graph statistics
print(f"IR version:  {model.ir_version}")
print(f"Opset:       {model.opset_import[0].version}")
print(f"Nodes:       {len(model.graph.node)}")
print(f"Inputs:      {[i.name for i in model.graph.input]}")
print(f"Outputs:     {[o.name for o in model.graph.output]}")
print(f"Initializers (weights): {len(model.graph.initializer)}")

# Operator histogram
from collections import Counter
op_counts = Counter(n.op_type for n in model.graph.node)
for op, count in op_counts.most_common(10):
    print(f"  {op:20s}: {count}")
```

### 4.2 Visual Inspection with Netron

```bash
pip install netron
netron resnet50.onnx
# Opens browser at http://localhost:8080
```

---

## 5. ONNX Graph Surgery with onnx-graphsurgeon

Graph surgery lets you **modify ONNX graphs programmatically** — add nodes, remove subgraphs, replace operators, or fuse patterns.

```python
import onnx_graphsurgeon as gs
import onnx
import numpy as np

# Load and wrap
graph = gs.import_onnx(onnx.load("model.onnx"))

# Remove all Dropout nodes (short-circuit input → output)
for node in graph.nodes:
    if node.op == "Dropout":
        node.outputs[0].inputs.clear()
        node.inputs[0].outputs = node.outputs

# Clean up and export
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model_modified.onnx")
```

---

## 6. Model Optimization with ONNX Tools

### 6.1 ONNX Optimizer (onnxoptimizer)

```python
import onnxoptimizer

model = onnx.load("model.onnx")

optimized = onnxoptimizer.optimize(model, passes=[
    "eliminate_identity",
    "eliminate_nop_dropout",
    "fuse_bn_into_conv",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_consecutive_transposes",
])

onnx.save(optimized, "model_optimized.onnx")
```

### 6.2 ONNX Runtime Graph Optimizations

```python
import onnxruntime as ort

# ORT applies its own graph optimizations at session creation
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "model_ort_optimized.onnx"

session = ort.InferenceSession("model.onnx", sess_options)
# The optimized model is saved to disk
```

ORT optimization levels: 0 (none) → 1 (basic) → 2 (extended, default) → 3 (full with custom fusions).

---

## Hands-On Exercises

### Exercise 1: Export and Inspect (20 min)
1. Export a HuggingFace BERT model to ONNX using both `torch.onnx.export` and `torch.onnx.dynamo_export`
2. Compare operator counts, opset versions, and file sizes
3. Run `onnx.checker.check_model()` on both — do they both pass?
4. Visualize one in Netron and screenshot the attention layer structure

### Exercise 2: Shape Inference and Validation (15 min)
1. Load the exported model, strip all intermediate shape info
2. Re-run `shape_inference.infer_shapes()` and verify all shapes are recovered
3. Change the input to a dynamic batch size — which intermediate shapes become symbolic?

### Exercise 3: Graph Surgery (25 min)
1. Use `onnx-graphsurgeon` to remove all Dropout and LayerNorm nodes from a BERT ONNX model
2. Verify the modified model still loads and passes `check_model()`
3. Count nodes before and after — what's the reduction ratio?
4. Try replacing `MatMul + Add` with fused `Gemm` using `gs`

### Exercise 4: SafeTensors vs Pickle (15 min)
```python
from safetensors.torch import save_file, load_file
import torch

# Save weights safely
tensors = {"weight": torch.randn(768, 768), "bias": torch.randn(768)}
save_file(tensors, "model.safetensors")

# Load — memory-mapped, no arbitrary code execution
loaded = load_file("model.safetensors")
assert torch.equal(tensors["weight"], loaded["weight"])
```
Why is SafeTensors preferred over pickle for weight distribution?

---

## Key Takeaways

1. **ONNX is the lingua franca** — it bridges PyTorch, TensorFlow, and dozens of inference runtimes via a standardized protobuf schema
2. **Opset version matters** — it determines which runtimes can execute your model; export at the lowest opset supporting your operators
3. **Dynamo export is the future** — `torch.onnx.dynamo_export` handles dynamic control flow and complex Python patterns that TorchScript cannot
4. **Graph surgery enables deployment** — tools like `onnx-graphsurgeon` let you rewrite graphs for specific hardware targets
5. **SafeTensors > pickle** — memory-mapped, zero-copy loading with no arbitrary code execution risks
6. **Always validate** — run `onnx.checker.check_model()` and `shape_inference` after every transformation

---

## Further Reading

- [ONNX Specification (GitHub)](https://github.com/onnx/onnx/blob/main/docs/IR.md)
- [ONNX Operator Schemas](https://onnx.ai/onnx/operators/)
- [PyTorch Dynamo ONNX Exporter](https://pytorch.org/docs/stable/onnx_dynamo.html)
- [onnx-graphsurgeon Documentation](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- [SafeTensors Format Spec](https://huggingface.co/docs/safetensors/)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

## Tomorrow's Preview

**Day 51: Weight Compression & Pruning** — We move from model formats to model size reduction. You'll learn unstructured pruning, structured channel pruning, N:M sparsity patterns that map directly to NVIDIA sparse tensor cores, and how to combine pruning with quantization for maximum compression without accuracy loss.
