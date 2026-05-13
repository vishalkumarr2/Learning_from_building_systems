# Day 35: Mini-Project — End-to-End TVM Compilation

> **Phase III · Week 5 · Day 35 of 70 · 2.5 hours**

*"The measure of a compiler isn't its IR — it's the wall-clock time on real hardware. Today you measure."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 34: TVM Runtime & Deployment](day-34-tvm-runtime-deployment.md) | [Day 36: AutoTVM & AutoScheduler](../week-06/day-36-autotvm-autoscheduler.md) | [Week 5: TVM Foundations](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

This is the capstone for Week 5. You've learned the full TVM stack layer by layer: Relay (graph IR), TE (compute declarations), TIR (low-level loops), schedules (transformations), and the runtime (deployment). Today you wire them together end-to-end on a real model — MobileNetV2 — and answer the question every engineer asks: **"Is it actually faster?"** You'll import from PyTorch, apply Relay optimizations, compile for GPU, benchmark against PyTorch eager and `torch.compile`, and profile to understand where time is spent.

---

## Project Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  End-to-End Pipeline                          │
│                                                              │
│  PyTorch Model     Relay IR        Optimized       Target    │
│  (MobileNetV2) ──▶ (graph) ──▶   Relay IR    ──▶  Module    │
│                    import     optimization passes   build    │
│                                                              │
│  Benchmark:  PyTorch Eager  vs  torch.compile  vs  TVM      │
└──────────────────────────────────────────────────────────────┘
```

### What You'll Build

| Step | Tool | Output |
|:--|:--|:--|
| 1. Export model | PyTorch + `torch.jit.trace` | TorchScript / ONNX |
| 2. Import to Relay | `relay.frontend.from_pytorch` | Relay IR module |
| 3. Optimize | `relay.build_module.bind_params_by_name` + passes | Optimized Relay |
| 4. Compile | `relay.build()` | Runtime Module |
| 5. Benchmark | `time_evaluator` + manual timing | Latency numbers |
| 6. Profile | `debug_executor` | Per-operator breakdown |

---

## Step 1: Export MobileNetV2 from PyTorch

```python
import torch
import torchvision.models as models
import numpy as np

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

# Create dummy input
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Trace the model (captures the computation graph)
traced_model = torch.jit.trace(model, dummy_input)

# Verify correctness
with torch.no_grad():
    pt_output = model(dummy_input).numpy()
    traced_output = traced_model(dummy_input).numpy()
    np.testing.assert_allclose(pt_output, traced_output, atol=1e-5)
    print(f"Model traced successfully. Output shape: {pt_output.shape}")
```

### Why MobileNetV2?

- Representative of production mobile/edge models
- Uses depthwise separable convolutions (interesting for scheduling)
- Small enough to iterate quickly (~3.4M params, ~300M FLOPs)
- Well-studied — easy to find reference benchmarks

---

## Step 2: Import into Relay

```python
import tvm
from tvm import relay

# Convert TorchScript → Relay
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

# Inspect the Relay IR (first 50 lines)
print(mod.astext()[:2000])
```

### What the Importer Does

```
TorchScript IR                         Relay IR
━━━━━━━━━━━━━━                         ━━━━━━━━
%x = aten::conv2d(%input, %w, %b)  →   %0 = nn.conv2d(%input, %w, ...)
%y = aten::batch_norm(%x, ...)     →   %1 = nn.batch_norm(%0, ...)
%z = aten::relu_(%y)               →   %2 = nn.relu(%1)
%out = aten::adaptive_avg_pool2d   →   %3 = nn.adaptive_avg_pool2d(...)
```

Each PyTorch op maps to a Relay operator. The importer preserves the graph structure, constant-folds weights, and infers tensor shapes.

---

## Step 3: Apply Relay Optimization Passes

```python
# Standard optimization pipeline
target = tvm.target.Target("cuda -arch=sm_80")  # adjust for your GPU
# For CPU: target = tvm.target.Target("llvm -mcpu=native")

with tvm.transform.PassContext(opt_level=3):
    # opt_level=3 enables:
    #   - Operator fusion (conv2d + bn + relu → single kernel)
    #   - Constant folding (precompute batch norm parameters)
    #   - Layout optimization (NCHW → NCHW4c for vectorization)
    #   - Dead code elimination
    #   - Common subexpression elimination
    lib = relay.build(mod, target=target, params=params)

print(f"Compiled successfully for {target}")
```

### Understanding opt_level

| Level | Passes Enabled | Effect |
|:--|:--|:--|
| 0 | Type inference only | Baseline, no optimization |
| 1 | + Constant folding, dead code elimination | Minor speedup |
| 2 | + Operator fusion | Major speedup (fewer kernel launches) |
| 3 | + Layout optimization, all passes | Full optimization |
| 4 | + Experimental passes | Bleeding edge |

### Fusion Visualization

Before fusion (many small kernels):

```
conv2d ──▶ batch_norm ──▶ relu ──▶ conv2d ──▶ batch_norm ──▶ relu
  │            │           │          │            │           │
  ▼            ▼           ▼          ▼            ▼           ▼
kernel 1    kernel 2    kernel 3   kernel 4    kernel 5    kernel 6
```

After fusion (fewer, larger kernels):

```
conv2d_bn_relu ──────────────────▶ conv2d_bn_relu
       │                                  │
       ▼                                  ▼
    kernel 1                           kernel 2
```

$$\text{Speedup}_{\text{fusion}} \propto \frac{\text{kernel launches (before)}}{\text{kernel launches (after)}} \times \frac{\text{memory traffic (before)}}{\text{memory traffic (after)}}$$

---

## Step 4: Run Inference & Verify Correctness

```python
from tvm.contrib import graph_executor

# Create runtime executor
dev = tvm.cuda(0)  # or tvm.cpu(0) for CPU target
m = graph_executor.GraphModule(lib["default"](dev))

# Prepare input
input_data = dummy_input.numpy()
m.set_input(input_name, tvm.nd.array(input_data, dev))

# Run inference
m.run()

# Get output and verify against PyTorch
tvm_output = m.get_output(0).numpy()
np.testing.assert_allclose(pt_output, tvm_output, rtol=1e-4, atol=1e-4)
print("✓ TVM output matches PyTorch (rtol=1e-4)")
print(f"  Top-5 classes: {np.argsort(tvm_output[0])[-5:][::-1]}")
```

---

## Step 5: Benchmark — TVM vs PyTorch

### PyTorch Eager Baseline

```python
import time

# Warm up
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input.cuda())  # or .cpu()
torch.cuda.synchronize()

# Benchmark
n_runs = 100
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(n_runs):
        _ = model(dummy_input.cuda())
torch.cuda.synchronize()
pt_eager_ms = (time.perf_counter() - start) / n_runs * 1000
print(f"PyTorch Eager:   {pt_eager_ms:.2f} ms")
```

### torch.compile Baseline

```python
compiled_model = torch.compile(model, mode="max-autotune")

# Warm up (includes compilation)
with torch.no_grad():
    for _ in range(10):
        _ = compiled_model(dummy_input.cuda())
torch.cuda.synchronize()

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(n_runs):
        _ = compiled_model(dummy_input.cuda())
torch.cuda.synchronize()
pt_compile_ms = (time.perf_counter() - start) / n_runs * 1000
print(f"torch.compile:   {pt_compile_ms:.2f} ms")
```

### TVM Benchmark

```python
# TVM's built-in benchmarking (handles warmup + statistics)
ftimer = m.module.time_evaluator("run", dev, number=100, repeat=5)
prof_result = ftimer()
tvm_ms = np.mean(prof_result.results) * 1000
tvm_std = np.std(prof_result.results) * 1000
print(f"TVM (opt_level=3): {tvm_ms:.2f} ± {tvm_std:.2f} ms")
```

### Results Summary

```python
print("\n" + "="*55)
print(f"{'Backend':<25} {'Latency (ms)':>12} {'Speedup':>10}")
print("="*55)
print(f"{'PyTorch Eager':<25} {pt_eager_ms:>12.2f} {'1.00x':>10}")
print(f"{'torch.compile':<25} {pt_compile_ms:>12.2f} {pt_eager_ms/pt_compile_ms:>9.2f}x")
print(f"{'TVM (opt_level=3)':<25} {tvm_ms:>12.2f} {pt_eager_ms/tvm_ms:>9.2f}x")
print("="*55)
```

Expected results (A100 GPU, MobileNetV2, batch=1):

```
═══════════════════════════════════════════════════════
Backend                   Latency (ms)    Speedup
═══════════════════════════════════════════════════════
PyTorch Eager                     2.10      1.00x
torch.compile                     1.40      1.50x
TVM (opt_level=3)                 1.25      1.68x
═══════════════════════════════════════════════════════
```

> **Note**: Your numbers will vary by GPU, driver, and TVM version. The relative ranking is more important than absolute numbers. TVM without auto-tuning (Week 6) often trails torch.compile on NVIDIA GPUs — auto-tuning closes the gap significantly.

---

## Step 6: Profile with TVM's Debug Executor

```python
from tvm.contrib.debugger import debug_executor

# Build with debug info
m_debug = debug_executor.create(
    lib.get_graph_json(), lib, dev
)
m_debug.set_input(input_name, tvm.nd.array(input_data, dev))

# Profile all operators
m_debug.run()

# Get per-node timing (sorted by time)
node_times = m_debug.profile()
print(node_times)
```

### Interpreting the Profile

```
Node Name                    Time (ms)    %Total    Calls
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fused_nn_conv2d_add_relu_0      0.18      14.4%       1
fused_nn_conv2d_add_relu_1      0.15      12.0%       1
fused_nn_conv2d_add_2           0.12       9.6%       1
fused_nn_contrib_depthwise_0    0.09       7.2%       1
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total                           1.25     100.0%      52
```

**What to look for**:
- Are the top operators fused? (good: `fused_nn_conv2d_add_relu`)
- Any surprisingly slow operators? (candidate for custom schedule)
- Many tiny operators? (fusion opportunities missed)

---

## Step 7: Export for Deployment

```python
# Save the compiled model for production
lib.export_library("mobilenetv2_cuda.tar")
print("Exported: mobilenetv2_cuda.tar")

# Verify the exported model works
loaded_lib = tvm.runtime.load_module("mobilenetv2_cuda.tar")
m_loaded = graph_executor.GraphModule(loaded_lib["default"](dev))
m_loaded.set_input(input_name, tvm.nd.array(input_data, dev))
m_loaded.run()
loaded_output = m_loaded.get_output(0).numpy()
np.testing.assert_allclose(tvm_output, loaded_output, atol=1e-6)
print("✓ Exported model produces identical output")
```

---

## Bonus: CPU Compilation with Layout Optimization

```python
# Compile the same model for CPU with AVX-512
target_cpu = tvm.target.Target("llvm -mcpu=skylake-avx512")

with tvm.transform.PassContext(opt_level=3):
    lib_cpu = relay.build(mod, target=target_cpu, params=params)

dev_cpu = tvm.cpu(0)
m_cpu = graph_executor.GraphModule(lib_cpu["default"](dev_cpu))
m_cpu.set_input(input_name, tvm.nd.array(input_data, dev_cpu))

ftimer_cpu = m_cpu.module.time_evaluator("run", dev_cpu, number=50, repeat=3)
cpu_ms = np.mean(ftimer_cpu().results) * 1000
print(f"TVM CPU (AVX-512): {cpu_ms:.2f} ms")
```

### Layout Optimization on CPU

TVM's `AlterOpLayout` pass transforms data layouts for better vectorization:

```
NCHW (default)              NCHWc (TVM optimized, c=16)
━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━━━━━━━━━
Shape: (1, 64, 56, 56)      Shape: (1, 4, 56, 56, 16)
                             └──── 64/16 = 4 outer groups
                                   16 channels packed for
                                   AVX-512 (16 × float32)
```

$$\text{NCHW} \rightarrow \text{NCHW}_{c} : \quad C_{\text{outer}} = \lfloor C / c \rfloor, \quad C_{\text{inner}} = c$$

---

## Hands-On Challenges

### Challenge 1: Batch Size Sweep (20 min)

Benchmark TVM vs PyTorch at batch sizes [1, 4, 16, 64]. Plot latency and throughput (images/sec). At what batch size does TVM's advantage peak?

### Challenge 2: Model Comparison (30 min)

Repeat the pipeline for ResNet-50 and EfficientNet-B0. Which model benefits most from TVM compilation? Why?

### Challenge 3: Target Comparison (20 min)

If you have both CPU and GPU, compile MobileNetV2 for both and compare. What's the CPU-GPU crossover batch size (where GPU becomes faster)?

### Challenge 4: Inspect Fusion Decisions (20 min)

```python
# Dump the optimized Relay IR to see fusion decisions
with tvm.transform.PassContext(opt_level=3):
    mod_opt, _ = relay.optimize(mod, target=target, params=params)
print(mod_opt.astext()[:5000])
# Count fused functions — how many separate kernels does TVM produce?
```

---

## Key Takeaways

1. **The full TVM pipeline**: PyTorch → TorchScript → Relay import → optimization passes → `relay.build()` → runtime execution
2. **`opt_level=3`** enables operator fusion, constant folding, and layout optimization — the three biggest wins
3. **Operator fusion** reduces kernel launches and memory traffic — often the single largest optimization
4. **TVM without auto-tuning** is competitive but may trail `torch.compile` on NVIDIA GPUs; auto-tuning (Week 6) closes the gap
5. **`time_evaluator()`** handles proper benchmarking (warmup, statistics); never use raw `time.time()` for GPU code
6. **`debug_executor`** gives per-operator profiling to identify bottlenecks and verify fusion

---

## Further Reading

- [Compile PyTorch Models (TVM tutorial)](https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html) — official import guide
- [Relay Optimization Passes](https://tvm.apache.org/docs/how_to/work_with_relay/index.html) — complete pass reference
- [TVM Benchmarking Best Practices](https://tvm.apache.org/docs/reference/api/python/runtime.html#tvm.runtime.Module.time_evaluator) — `time_evaluator` API
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — the model architecture paper

---

## Week 5 Complete!

You now understand the full TVM stack: Relay for graph-level IR and passes, TE for declarative compute definitions, TIR for low-level loop-explicit code, schedules for performance transformations, and the runtime for deployment. **Next week** (Week 6) you'll learn how to **automate** the schedule search with AutoTVM and MetaSchedule — letting ML optimize the compiler itself.
