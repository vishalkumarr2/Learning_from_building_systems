# Day 54: Inference on CPU

> **Phase IV · Week 8 · Day 54 of 70 · 2.5 hours**

*"GPUs get the headlines, but CPUs serve the majority of production inference — and squeezing every cycle out of them is an art form."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 53: TensorRT Optimization](day-53-tensorrt-optimization.md) | [Day 55: Inference on Edge Devices](day-55-inference-on-edge.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Not every deployment has a GPU. In practice, **60–70% of production ML inference** runs on CPUs: web services behind load balancers, laptop applications, on-prem servers without accelerators, and latency-sensitive microservices where GPU scheduling overhead exceeds the compute time. Modern CPUs are not the slow, serial processors of the past — Intel's AMX (Advanced Matrix Extensions) can perform a 16×64 × 64×16 INT8 matrix multiply in a single instruction, Sapphire Rapids achieves >100 INT8 TOPS, and ARM's SVE2 brings scalable vector processing to server and mobile chips alike. The key is choosing the right runtime, precision, and threading strategy for your target CPU.

---

## 1. CPU Inference Landscape

```
CPU Inference Stack
══════════════════════════════════════════════════════════════════════

  Application Layer
  ┌──────────────────────────────────────────────────────────────┐
  │  Your ML Application (Python / C++ / Java / Rust)           │
  └──────────────┬──────────────┬──────────────┬────────────────┘
                 │              │              │
  Runtime Layer  ▼              ▼              ▼
  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────┐
  │  ONNX Runtime    │ │  OpenVINO    │ │  TFLite (x86/ARM)   │
  │  (CPU EP)        │ │  (Intel IR)  │ │                      │
  └────────┬─────────┘ └──────┬───────┘ └──────────┬───────────┘
           │                  │                     │
  Kernel   ▼                  ▼                     ▼
  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────┐
  │  oneDNN (DNNL)   │ │  oneDNN      │ │  XNNPACK / ruy      │
  │  + MLAS          │ │  + TBB       │ │                      │
  └────────┬─────────┘ └──────┬───────┘ └──────────┬───────────┘
           │                  │                     │
  ISA      ▼                  ▼                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  CPU ISA Extensions                                          │
  │  x86: SSE4.2 → AVX2 → AVX-512 → AVX-512_VNNI → AMX        │
  │  ARM: NEON → SVE → SVE2 → SME                               │
  └──────────────────────────────────────────────────────────────┘
```

---

## 2. ONNX Runtime CPU Execution Provider

ONNX Runtime's CPU EP is the most portable high-performance option — it works across x86 and ARM, uses MLAS (Microsoft Linear Algebra Subroutines) and oneDNN for kernel dispatch.

### 2.1 Basic Setup and Optimization

```python
import onnxruntime as ort
import numpy as np

# ── Session options for maximum CPU throughput ──
opts = ort.SessionOptions()

# Graph-level optimizations: constant folding, shape inference, fusion
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Threading strategy
opts.intra_op_num_threads = 4   # Threads within a single operator (e.g., GEMM)
opts.inter_op_num_threads = 2   # Threads across independent operators

# Enable operator-level profiling
opts.enable_profiling = True

# Create session with CPU EP
session = ort.InferenceSession(
    "resnet50.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)

# Warm up + benchmark
input_name = session.get_inputs()[0].name
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {input_name: dummy})

# Benchmark
import time
start = time.perf_counter()
N = 100
for _ in range(N):
    session.run(None, {input_name: dummy})
elapsed = time.perf_counter() - start
print(f"Avg latency: {elapsed / N * 1000:.2f} ms")
print(f"Throughput:  {N / elapsed:.1f} img/s")
```

### 2.2 Graph Optimizations Performed by ORT

```
ONNX Runtime Graph Optimization Levels
══════════════════════════════════════════════════════════

  Level 1 (Basic):
  ┌─────────────────────────────────────────────────┐
  │ • Constant folding (precompute static subgraphs)│
  │ • Redundant node elimination                    │
  │ • Semantic-preserving rewrites                  │
  └─────────────────────────────────────────────────┘

  Level 2 (Extended):
  ┌─────────────────────────────────────────────────┐
  │ • Conv + BN fusion                              │
  │ • Conv + Add + ReLU fusion                      │
  │ • GEMM + activation fusion                      │
  │ • Attention fusion (BERT-style)                 │
  │ • SkipLayerNormalization fusion                  │
  └─────────────────────────────────────────────────┘

  Level 3 (Layout):
  ┌─────────────────────────────────────────────────┐
  │ • NCHW → blocked layout (NCHWxc) for oneDNN    │
  │ • Weight prepacking for MLAS                    │
  │ • Memory planning (in-place reuse)              │
  └─────────────────────────────────────────────────┘
```

---

## 3. Intel OpenVINO

OpenVINO (Open Visual Inference and Neural Network Optimization) is Intel's toolkit optimized for Intel CPUs, iGPUs, and VPUs. It excels on Intel hardware with automatic precision calibration and hardware-specific kernel selection.

### 3.1 OpenVINO Pipeline

```python
from openvino import Core
import numpy as np

# Initialize OpenVINO runtime
core = Core()

# Check available devices and CPU features
devices = core.available_devices
cpu_props = core.get_property("CPU", "FULL_DEVICE_NAME")
print(f"CPU: {cpu_props}")

# Read ONNX model directly (or use .xml/.bin IR format)
model = core.read_model("resnet50.onnx")

# Apply optimizations: enable dynamic shapes, set precision hints
compiled = core.compile_model(
    model,
    device_name="CPU",
    config={
        "PERFORMANCE_HINT": "THROUGHPUT",       # or "LATENCY"
        "INFERENCE_NUM_THREADS": "8",
        "INFERENCE_PRECISION_HINT": "bf16",     # Use BF16 on supported CPUs
        "ENABLE_MMAP": "YES",                   # Memory-map model weights
    },
)

# Create infer request
infer_request = compiled.create_infer_request()

# Run inference
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
infer_request.infer({0: input_tensor})
output = infer_request.get_output_tensor(0).data
```

### 3.2 INT8 Quantization with NNCF

```python
import nncf  # Neural Network Compression Framework

# Post-training quantization with calibration dataset
quantized_model = nncf.quantize(
    model,
    calibration_dataset=nncf.Dataset(calib_loader),
    subset_size=300,             # Number of calibration samples
    preset=nncf.QuantizationPreset.MIXED,  # INT8 weights + activations
    target_device=nncf.TargetDevice.CPU,
)

# Accuracy-aware quantization (rolls back layers that hurt accuracy)
quantized_model = nncf.quantize_with_accuracy_control(
    model,
    calibration_dataset=nncf.Dataset(calib_loader),
    validation_dataset=nncf.Dataset(val_loader),
    validation_fn=validate,
    max_drop=0.01,  # Max 1% accuracy drop
)
```

---

## 4. Vectorization & CPU ISA Extensions

Modern CPUs achieve ML performance through wide vector instructions. Understanding the ISA hierarchy explains the performance gaps.

### 4.1 ISA Extension Timeline (x86)

```
x86 Vector ISA Evolution
══════════════════════════════════════════════════════════

  ISA             Width    INT8 TOPS*    Key ML Feature
  ─────────────   ─────    ──────────    ─────────────────────────
  SSE4.2 (2008)   128-bit   ~0.1        PMADDUBSW (8→16 dot)
  AVX2 (2013)     256-bit   ~0.4        VPMADDUBSW, FMA3
  AVX-512 (2017)  512-bit   ~1.0        512-bit FMA
  VNNI (2019)     512-bit   ~4.0        VPDPBUSD (INT8→INT32 dot)
  AMX (2023)      tiles     ~100        TDPBSSD (tile matmul)

  * Approximate single-core TOPS at 3 GHz

  VNNI Instruction: VPDPBUSD
  ─────────────────────────
  Takes 64 INT8 pairs → produces 16 INT32 accumulators
  in ONE instruction (vs 4 instructions with AVX-512)

  AMX Instruction: TDPBSSD
  ─────────────────────────
  Multiplies 16×64 tile × 64×16 tile → 16×16 INT32 result
  One instruction replaces ~32K scalar multiplies
```

### 4.2 How VNNI Accelerates INT8 Inference

The key instruction `VPDPBUSD` computes a dot product of unsigned/signed byte pairs:

$$\text{dst}[i] \mathrel{+}= \sum_{j=0}^{3} \text{src1}[4i+j]_{\text{u8}} \times \text{src2}[4i+j]_{\text{s8}}$$

For a single AVX-512 register (512 bits):
- 64 INT8 values × 64 INT8 values → 16 INT32 partial sums
- **Per cycle per core**: 64 multiply-accumulate operations
- **Per core at 3 GHz**: ~192 GOPS (INT8)

### 4.3 Intel AMX for Matrix Workloads

```
AMX Tile Architecture (Sapphire Rapids+)
══════════════════════════════════════════════════════════

  8 tile registers, each up to 16 rows × 64 bytes:

  ┌──────────────────────────────────────────────────────┐
  │ Tile T0  (16 × 64 bytes = 1 KiB)                    │
  │ ┌────────────────────────────────────────────────┐   │
  │ │ Row 0:  64 × INT8  or  32 × BF16              │   │
  │ │ Row 1:  ...                                     │   │
  │ │ ...                                             │   │
  │ │ Row 15: ...                                     │   │
  │ └────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────┘

  TDPBSSD (Tile Dot Product Bytes Signed-Signed Dword):
  ┌──────────┐   ┌──────────┐     ┌──────────┐
  │ Tile A   │ × │ Tile B   │  =  │ Tile C   │
  │ 16 × 64  │   │ 64 × 16  │     │ 16 × 16  │
  │ (INT8)   │   │ (INT8)   │     │ (INT32)  │
  └──────────┘   └──────────┘     └──────────┘

  Result: 16 × 16 × 64 = 16,384 INT8 MACs per instruction
  At ~1 GHz effective: ~16 INT8 TOPS per core
```

---

## 5. Threading: Intra-Op vs Inter-Op Parallelism

CPU inference performance depends heavily on how threads are assigned across and within operators.

### 5.1 Threading Models

```
Threading Strategies for CPU Inference
══════════════════════════════════════════════════════════

  Intra-op parallelism (threads within one operator):
  ┌─────────────────────────────────────────────────┐
  │  Conv2d (big GEMM)                              │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
  │  │ T0   │ │ T1   │ │ T2   │ │ T3   │          │
  │  │ rows │ │ rows │ │ rows │ │ rows │          │
  │  │ 0-63 │ │64-127│ │128-  │ │192-  │          │
  │  └──────┘ └──────┘ └──────┘ └──────┘          │
  │  → Each thread processes a tile of the output   │
  └─────────────────────────────────────────────────┘

  Inter-op parallelism (independent operators run in parallel):
  ┌────────────────────────────────────────────┐
  │  Branch A              Branch B            │
  │  ┌──────┐              ┌──────┐           │
  │  │ Conv │  (parallel)  │ Pool │           │
  │  │  T0  │              │  T1  │           │
  │  └──┬───┘              └──┬───┘           │
  │     └───────┬──────────┘                  │
  │             ▼                              │
  │        ┌──────┐                            │
  │        │ Add  │                            │
  │        └──────┘                            │
  └────────────────────────────────────────────┘
```

### 5.2 Threading Guidelines

```python
import os

# Rule of thumb for threading configuration
num_physical_cores = os.cpu_count() // 2  # Exclude hyperthreads

# Latency-optimized (single request, minimize time)
opts_latency = ort.SessionOptions()
opts_latency.intra_op_num_threads = num_physical_cores
opts_latency.inter_op_num_threads = 1

# Throughput-optimized (multiple requests, maximize utilization)
opts_throughput = ort.SessionOptions()
opts_throughput.intra_op_num_threads = 4
opts_throughput.inter_op_num_threads = num_physical_cores // 4

# NUMA-aware: bind to one socket
os.environ["OMP_NUM_THREADS"] = str(num_physical_cores // 2)  # One socket
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
```

**Key insight**: Hyperthreads share execution units and caches. For compute-bound ML inference, using only physical cores often gives **better** throughput than using all logical cores.

---

## 6. Benchmarking CPU Inference Pipelines

### 6.1 Fair Benchmarking Protocol

```python
import time
import numpy as np
import onnxruntime as ort

def benchmark_cpu_inference(model_path, input_shape, num_warmup=50, num_runs=200):
    """Benchmark ONNX model on CPU with proper methodology."""

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup (JIT compilation, cache warming)
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    # Timed runs — collect individual latencies
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    print(f"Model: {model_path}")
    print(f"  Mean latency:  {latencies.mean():.2f} ms")
    print(f"  P50 latency:   {np.percentile(latencies, 50):.2f} ms")
    print(f"  P99 latency:   {np.percentile(latencies, 99):.2f} ms")
    print(f"  Std dev:       {latencies.std():.2f} ms")
    print(f"  Throughput:    {1000 / latencies.mean():.1f} infer/s")
    return latencies


# Compare FP32 vs INT8
benchmark_cpu_inference("resnet50_fp32.onnx", (1, 3, 224, 224))
benchmark_cpu_inference("resnet50_int8.onnx", (1, 3, 224, 224))
```

### 6.2 System-Level Monitoring During Benchmark

```bash
#!/bin/bash
# Monitor CPU during inference benchmark

# Pin to physical cores 0-7 (first socket)
export OMP_NUM_THREADS=8
export GOMP_CPU_AFFINITY="0-7"

# Disable frequency scaling for consistent results
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Run benchmark with perf stat
perf stat -e cycles,instructions,cache-misses,LLC-load-misses \
    python benchmark.py 2>&1 | tee bench_results.txt

# Check which ISA extensions were used
perf record -e cpu/event=0xc7,umask=0x01/ python benchmark.py
# → look for AVX-512 or AMX usage in perf report
```

---

## Hands-On Exercises

### Exercise 1: Threading Sweep
Benchmark a ResNet-50 ONNX model varying `intra_op_num_threads` from 1 to `os.cpu_count()`. Plot latency vs thread count. Find the "knee" where adding threads no longer helps.

### Exercise 2: ORT vs OpenVINO Comparison
Take the same INT8-quantized model and benchmark it with ONNX Runtime CPU EP and OpenVINO on the same machine. Compare P50, P99 latencies, and throughput.

### Exercise 3: VNNI Detection
Write a script that checks for VNNI support (`lscpu | grep vnni`), then benchmarks INT8 vs FP32 inference. Report the speedup ratio — on VNNI hardware, INT8 should be 2–4× faster.

---

## Key Takeaways

1. **CPUs serve the majority of production inference** — understanding CPU optimization is not optional, even if GPUs get the attention
2. **ISA extensions are the multiplier** — AVX-512 VNNI gives ~4× INT8 throughput over plain AVX-512; AMX adds another ~10× on top
3. **Threading strategy matters** — use physical cores only, bind to NUMA nodes, and balance intra-op vs inter-op threads for your workload pattern
4. **ONNX Runtime vs OpenVINO** — ORT is more portable; OpenVINO extracts more performance on Intel hardware. Benchmark both for your target
5. **Graph optimizations are free performance** — enable `ORT_ENABLE_ALL`; fusion + layout transforms can give 20–40% speedup before touching precision
6. **Benchmark methodology is critical** — pin frequency, warm up caches, collect percentile latencies, and disable turbo boost for reproducible results

---

## Further Reading

- [ONNX Runtime Performance Tuning Guide](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [Intel OpenVINO Documentation](https://docs.openvino.ai/)
- [oneDNN Developer Guide](https://oneapi-src.github.io/oneDNN/)
- [Intel AMX Programming Reference](https://www.intel.com/content/www/us/en/develop/documentation/intrinsics-guide/top/amx.html)
- [NNCF (Neural Network Compression Framework)](https://github.com/openvinotoolkit/nncf)
- [MLAS Internals (ONNX Runtime)](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/mlas)

---

## Tomorrow's Preview

**Day 55: Inference on Edge Devices** — When your target is a phone, Raspberry Pi, or microcontroller, everything changes. You'll explore TFLite, Core ML, NNAPI delegates, and TVM's µTVM, learning to navigate the latency-power-accuracy tradeoff triangle that defines edge deployment.
