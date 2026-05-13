# Day 45: ONNX Runtime Deep Dive

> **Phase III · Week 7 · Day 45 of 70 · 2.5 hours**

*"The best deployment framework isn't the one with the most optimizations — it's the one that runs everywhere your models need to go."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 44: XLA & StableHLO](day-44-xla-stablehlo.md) | [Day 46: MLC-LLM](day-46-mlc-llm.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

**ONNX Runtime (ORT)** is the most widely deployed ML inference engine in production. It powers inference at Microsoft (Azure ML, Office, Bing, Xbox), Hugging Face, and thousands of companies. While TVM and XLA focus on compilation to optimized kernels, ORT takes a pragmatic approach: **partition the model graph across the best available backend for each subgraph** — CUDA for some ops, TensorRT for others, CPU for the rest. This "best tool for each op" philosophy, combined with the ONNX standard format, makes ORT the go-to for deploying trained models. Understanding ORT's architecture reveals a different design philosophy from compiler-centric approaches.

---

## 1. ORT Architecture Overview

### The Execution Model

```
ONNX Runtime Architecture
══════════════════════════

  ┌─────────────────────────────────────────────┐
  │              ONNX Model (.onnx)              │
  │  (framework-agnostic graph + weights)        │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │            Graph Optimization                │
  │  Level 1: Basic (const folding, dead code)   │
  │  Level 2: Extended (fusion, layout)          │
  │  Level 99: Layout optimization               │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │          Graph Partitioning                  │
  │  Assign subgraphs to Execution Providers     │
  │                                              │
  │  ┌──────┐ ┌──────────┐ ┌────────┐ ┌──────┐ │
  │  │TensoRT│ │   CUDA   │ │OpenVINO│ │ CPU  │ │
  │  │ EP    │ │   EP     │ │  EP    │ │ EP   │ │
  │  └──┬───┘ └────┬─────┘ └───┬────┘ └──┬───┘ │
  └─────┼──────────┼───────────┼─────────┼─────┘
        │          │           │         │
  ┌─────▼──────────▼───────────▼─────────▼─────┐
  │            Execution Engine                  │
  │  • Sequential or parallel execution          │
  │  • Memory planning (arena allocator)         │
  │  • Kernel dispatch                           │
  └──────────────────────────────────────────────┘
```

### Key Design Principles

| Principle | Implementation |
|:--|:--|
| **Format-first** | ONNX is the input — no tracing or re-compilation |
| **Best backend per op** | Graph partitioning across Execution Providers |
| **Incremental optimization** | Three levels of graph transforms |
| **Zero-copy where possible** | Arena allocator with memory reuse patterns |
| **Backward compatible** | Old models run on new ORT versions |

---

## 2. Graph Optimization Pipeline

### Optimization Levels

ORT applies graph transformations in three levels, from safe to aggressive:

```python
import onnxruntime as ort

# Configure optimization level
options = ort.SessionOptions()

# Level 0: No optimization (debugging)
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# Level 1: Basic (always safe)
#   • Constant folding
#   • Dead node elimination
#   • Redundant node elimination
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

# Level 2: Extended (may change numerics slightly)
#   • Operator fusion (Conv+BN, MatMul+Add, etc.)
#   • Attention fusion
#   • GELU fusion
#   • Layer normalization fusion
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Level 99: All optimizations including layout transforms
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("model.onnx", options)
```

### Key Fusion Patterns

```
ORT Fusion Patterns (selected)
══════════════════════════════

1. Conv + BatchNorm → FusedConv
   ┌──────┐   ┌────┐           ┌──────────┐
   │ Conv │──→│ BN │    ═══→   │ FusedConv│  (BN folded into Conv)
   └──────┘   └────┘           └──────────┘

2. MatMul + Add → FusedMatMul (Gemm)
   ┌────────┐   ┌─────┐        ┌──────────────┐
   │ MatMul │──→│ Add │  ═══→  │ FusedGemm    │
   └────────┘   └─────┘        │ (α·A·B + β·C)│
                                └──────────────┘

3. Multi-Head Attention Fusion
   ┌────┐ ┌────┐ ┌────┐
   │ Wq │ │ Wk │ │ Wv │
   └──┬─┘ └──┬─┘ └──┬─┘        ┌──────────────────┐
      │      │      │    ═══→  │ MultiHeadAttention│
   ┌──▼──────▼──────▼──┐       │ (single fused op) │
   │  QKV + softmax +  │       └──────────────────┘
   │  weighted sum      │
   └────────────────────┘

4. LayerNorm Fusion
   ┌──────┐ ┌──────┐ ┌──────┐
   │ReduceMean │→│ Sub │→│ Pow │→...   ═══→  ┌───────────┐
   │      │→│ Div │→│ Mul │→│ Add │          │ LayerNorm │
   └──────┘ └─────┘ └─────┘ └─────┘          └───────────┘

5. GELU Approximation Fusion
   x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
   ═══→  BiasGelu(x, bias)
```

---

## 3. Execution Providers (EPs)

### The EP Architecture

Execution Providers are pluggable backends. ORT queries them in priority order — each EP claims the subgraphs it can handle:

```
Graph Partitioning with Execution Providers
════════════════════════════════════════════

  Full Model Graph:
  ┌────┐   ┌────┐   ┌────────┐   ┌──────┐   ┌────┐
  │Conv│──→│BN  │──→│Attention│──→│Custom│──→│Soft│
  └────┘   └────┘   └────────┘   │  Op  │   │max │
                                  └──────┘   └────┘

  EP Priority: TensorRT > CUDA > CPU

  TensorRT claims:  Conv+BN (as fused engine)
  CUDA claims:      Attention, Softmax
  CPU fallback:     Custom Op (not supported by GPU EPs)

  Result:
  ┌─── TensorRT EP ───┐  ┌──── CUDA EP ────┐  ┌─ CPU EP ─┐
  │ Conv+BN (engine)   │→│ Attention, Soft  │→│ CustomOp │
  └────────────────────┘  └─────────────────┘  └──────────┘
       GPU memory              GPU memory         CPU mem
                     ↕ data transfer ↕
```

### Available Execution Providers

| EP | Target Hardware | Key Advantage |
|:--|:--|:--|
| `CPUExecutionProvider` | Any CPU | Always available, baseline |
| `CUDAExecutionProvider` | NVIDIA GPU | cuDNN/cuBLAS integration |
| `TensorrtExecutionProvider` | NVIDIA GPU | INT8/FP16, layer fusion engine |
| `OpenVINOExecutionProvider` | Intel CPU/GPU/VPU | Intel-optimized, edge devices |
| `DirectMLExecutionProvider` | Windows GPU | Any DirectX 12 GPU |
| `CoreMLExecutionProvider` | Apple Silicon | Neural Engine access |
| `QNNExecutionProvider` | Qualcomm NPU | Mobile (Snapdragon) |
| `ROCmExecutionProvider` | AMD GPU | MIOpen/rocBLAS |
| `AzureExecutionProvider` | Azure cloud | Remote inference |

### Configuring EPs

```python
import onnxruntime as ort

# Priority list: try TensorRT first, fall back to CUDA, then CPU
providers = [
    ('TensorrtExecutionProvider', {
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache/',
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession("model.onnx", providers=providers)

# Check which EP each node is assigned to
for node in session.get_providers():
    print(node)
```

---

## 4. Memory Management

### Arena Allocator

ORT uses an arena-based allocator to minimize memory allocation overhead:

```
ORT Memory Arena
════════════════

  Pool Structure (pre-allocated chunks):
  ┌──────────────────────────────────────┐
  │            Arena (e.g., 256 MB)       │
  │  ┌────┐┌────────┐┌──┐┌──────┐┌────┐ │
  │  │used││  free  ││us││ free ││used│ │
  │  │ 4MB││  12MB  ││2M││  8MB ││ 4MB│ │
  │  └────┘└────────┘└──┘└──────┘└────┘ │
  └──────────────────────────────────────┘

  Allocation strategy:
  1. Best-fit from existing free blocks
  2. If no fit: extend arena (kNextPowerOfTwo)
  3. Freed blocks are returned to pool, not to OS

  Memory Reuse Patterns:
  ┌──────────────────────────────────────────┐
  │  Execution order:  A → B → C → D → E    │
  │                                          │
  │  A's buffer: [████]                      │
  │  B's buffer:       [████████]            │
  │  C reuses A:  [████]      ← same addr   │
  │  D's buffer:               [████]        │
  │  E reuses B:       [████████]← reused    │
  └──────────────────────────────────────────┘
```

### Memory Optimization Patterns

```python
options = ort.SessionOptions()

# Enable memory pattern optimization
options.enable_mem_pattern = True

# Enable memory arena shrinkage (reduce peak memory)
options.enable_mem_reuse = True

# Set execution mode
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # or ORT_PARALLEL

# Limit intra-op threads (e.g., for server deployment)
options.intra_op_num_threads = 4
options.inter_op_num_threads = 2
```

---

## 5. ORT vs TVM: Deployment Trade-offs

### Architectural Differences

```
ORT Approach                    TVM Approach
════════════                    ════════════

  Model.onnx                    Model (any framework)
      │                              │
  ┌───▼────────┐               ┌─────▼──────┐
  │ Graph Opt  │               │ Import to   │
  │ (rewrites) │               │ Relay/Relax │
  └───┬────────┘               └─────┬──────┘
      │                              │
  ┌───▼────────┐               ┌─────▼──────┐
  │ Partition  │               │ Compile +   │
  │ to EPs     │  vs.          │ Auto-Tune   │ 
  └───┬────────┘               └─────┬──────┘
      │                              │
  ┌───▼────────┐               ┌─────▼──────┐
  │ Call vendor │               │ Generate   │
  │ libraries  │               │ custom     │
  │ (cuDNN,    │               │ kernels    │
  │  TensorRT) │               │            │
  └────────────┘               └────────────┘

  ORT: "Best library for each op"
  TVM: "Best generated kernel for each op"
```

### When to Choose Each

| Scenario | Choose ORT | Choose TVM |
|:--|:--|:--|
| **Quick deployment** | ✅ Drop in ONNX, done | Needs tuning time |
| **NVIDIA GPU** | ✅ cuDNN/TRT tuned | Good, less mature |
| **Custom hardware** | ❌ Needs EP impl | ✅ Write schedule |
| **Edge / MCU** | Limited (mobile EPs) | ✅ µTVM, bare metal |
| **Transformer models** | ✅ Built-in attention fusion | ✅ Auto-tuned |
| **Model variety** | ✅ 300+ ONNX ops | Some ops need impl |
| **Latency-critical** | ✅ TRT EP very fast | ✅ After tuning |
| **Cross-platform** | ✅ Windows/Linux/Mac | ✅ Linux-focused |

---

## 6. Hands-On Exercises

### Exercise 1: Export and Optimize a Model (30 min)

```python
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np

# 1. Define a model
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

model = TransformerBlock().eval()
dummy = torch.randn(1, 16, 256)

# 2. Export to ONNX
torch.onnx.export(
    model, dummy, "transformer_block.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "seq_len"}},
    opset_version=17,
)

# 3. Load with different optimization levels and compare
for level_name, level in [
    ("DISABLED", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ("EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ("ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
]:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level
    # Save optimized model to inspect changes
    opts.optimized_model_filepath = f"optimized_{level_name}.onnx"
    sess = ort.InferenceSession("transformer_block.onnx", opts,
                                providers=["CPUExecutionProvider"])
    print(f"{level_name}: loaded successfully")

# 4. Compare: count nodes in original vs optimized
import onnx
for name in ["transformer_block", "optimized_ALL"]:
    m = onnx.load(f"{name}.onnx")
    print(f"{name}: {len(m.graph.node)} nodes")
```

### Exercise 2: Profile ORT Inference (20 min)

```python
import onnxruntime as ort
import numpy as np
import json

# Enable profiling
options = ort.SessionOptions()
options.enable_profiling = True
options.profile_file_prefix = "ort_profile"

session = ort.InferenceSession(
    "transformer_block.onnx", options,
    providers=["CPUExecutionProvider"]
)

# Run inference
input_data = np.random.randn(4, 32, 256).astype(np.float32)
for _ in range(10):
    session.run(None, {"input": input_data})

# Get profiling results
profile_file = session.end_profiling()
print(f"Profile saved to: {profile_file}")

# Parse and display top-10 slowest ops
with open(profile_file) as f:
    events = json.load(f)

kernel_events = [e for e in events if e.get("cat") == "Node"]
kernel_events.sort(key=lambda e: e.get("dur", 0), reverse=True)

print("\nTop 10 slowest operations:")
print(f"{'Op Name':<40} {'Duration (µs)':>12}")
print("-" * 54)
for e in kernel_events[:10]:
    print(f"{e['name']:<40} {e['dur']:>12}")
```

### Exercise 3: Multi-EP Deployment (20 min)

```python
import onnxruntime as ort
import numpy as np

# Check available execution providers
print("Available EPs:", ort.get_available_providers())

# Compare performance across EPs
model_path = "transformer_block.onnx"
input_data = np.random.randn(8, 64, 256).astype(np.float32)

for ep in ort.get_available_providers():
    try:
        session = ort.InferenceSession(model_path, providers=[ep])
        # Warmup
        for _ in range(5):
            session.run(None, {"input": input_data})

        # Benchmark
        import time
        t0 = time.perf_counter()
        for _ in range(100):
            session.run(None, {"input": input_data})
        elapsed = (time.perf_counter() - t0) / 100

        print(f"{ep:<35} {elapsed*1000:.2f} ms/inference")
    except Exception as e:
        print(f"{ep:<35} FAILED: {e}")
```

---

## Key Takeaways

1. **ORT's philosophy is "best backend per op"** — it partitions the graph across Execution Providers rather than generating all kernels from scratch
2. **Three optimization levels** (Basic, Extended, All) apply increasingly aggressive fusion patterns including attention fusion, GELU fusion, and LayerNorm fusion
3. **Execution Providers** are pluggable backends — TensorRT, CUDA, OpenVINO, CoreML, DirectML — each claiming subgraphs they can optimize best
4. **Graph partitioning** means ORT handles heterogeneous hardware naturally — different parts of the model run on different accelerators
5. **Arena-based memory management** with memory reuse patterns minimizes allocation overhead in production serving
6. **ORT vs TVM**: ORT excels at quick deployment with vendor-optimized libraries; TVM excels at generating custom kernels for novel hardware or maximum per-op performance

---

## Further Reading

- [ONNX Runtime Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html) — official design docs
- [Execution Providers Guide](https://onnxruntime.ai/docs/execution-providers/) — all EPs with configs
- [Graph Optimizations in ORT](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) — optimization details
- [ONNX Runtime for LLMs (GenAI)](https://onnxruntime.ai/docs/genai/) — ORT for generative AI
- [ONNX Spec (opset 21)](https://onnx.ai/onnx/operators/) — all supported operations
- [ORT Performance Tuning Guide](https://onnxruntime.ai/docs/performance/tune-performance.html) — production tips

---

## Tomorrow: MLC-LLM

Day 46 explores **MLC-LLM** — the TVM team's project for compiling large language models to run everywhere, from NVIDIA GPUs to iPhones to web browsers. You'll see how TVM's Relax IR, auto-tuning, and quantization combine to make Llama and other LLMs portable across any hardware.
