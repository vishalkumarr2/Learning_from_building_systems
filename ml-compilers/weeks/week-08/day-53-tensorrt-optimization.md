# Day 53: TensorRT Optimization

> **Phase IV · Week 8 · Day 53 of 70 · 2.5 hours**

*"TensorRT doesn't run your model — it rebuilds it from scratch for the GPU you have right now."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 52: Knowledge Distillation](day-52-knowledge-distillation.md) | [Day 54: Inference on CPU](day-54-inference-on-cpu.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

You've exported to ONNX, pruned weights, distilled knowledge. Now what? The model still needs to **execute** on real hardware, and generic frameworks leave enormous performance on the table. TensorRT is NVIDIA's production inference optimizer — it takes your model, fuses layers, selects optimal kernels, calibrates precision, and plans memory layout specifically for **your GPU architecture**. A ResNet-50 that runs at 200 img/s in PyTorch can reach 5000+ img/s in TensorRT. For LLMs, TensorRT-LLM achieves 2–5× the throughput of vanilla HuggingFace inference. Understanding TensorRT's architecture — parser, builder, optimizer, runtime — is essential for anyone deploying NVIDIA GPU inference at scale.

---

## 1. TensorRT Architecture

```
TensorRT Pipeline — From Model to Engine
══════════════════════════════════════════════════════════════════════

  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
  │  ONNX Model │     │   Parser    │     │      Network        │
  │  (.onnx)    │────▶│  (parses    │────▶│   Definition        │
  │             │     │   to IR)    │     │  (INetworkDefinition)│
  └─────────────┘     └─────────────┘     └──────────┬──────────┘
                                                      │
                                                      ▼
  ┌─────────────┐     ┌─────────────┐     ┌──────────────────────┐
  │  Serialized │     │   Runtime   │     │      Builder         │
  │  Engine     │◀────│  (loads &   │◀────│   (optimizes &       │
  │  (.engine)  │     │   executes) │     │    builds engine)    │
  └─────────────┘     └──────┬──────┘     └──────────────────────┘
                             │
                             ▼                 Builder optimizations:
                    ┌────────────────┐         ┌─────────────────────┐
                    │  Inference     │         │ • Layer fusion       │
                    │  Context       │         │ • Kernel auto-tuning │
                    │  (per-request) │         │ • Precision selection │
                    └────────────────┘         │ • Memory planning    │
                                               │ • Tensor reformatting│
                                               └─────────────────────┘
```

---

## 2. Layer Fusion

TensorRT fuses multiple operations into single, optimized kernels:

### 2.1 Common Fusion Patterns

```
Layer Fusion Examples
══════════════════════════════════════════════════════════

  Before Fusion:                    After Fusion:
  ┌──────┐ ┌──────┐ ┌──────┐      ┌──────────────────┐
  │ Conv │→│  BN  │→│ ReLU │  ──▶ │ ConvBNReLU       │  1 kernel
  └──────┘ └──────┘ └──────┘      │ (fused)          │
                                   └──────────────────┘

  ┌──────┐ ┌──────┐ ┌──────┐      ┌──────────────────┐
  │MatMul│→│ Add  │→│ ReLU │  ──▶ │ GemmReLU         │  1 kernel
  └──────┘ └──────┘ └──────┘      │ (fused)          │
                                   └──────────────────┘

  ┌──────┐   ┌──────┐             ┌──────────────────┐
  │  QKV │ → │Softmax│→ V mul ──▶ │ FlashAttention   │  1 kernel
  └──────┘   └──────┘             │ (fused MHA)      │
                                   └──────────────────┘

  ┌──────┐ ┌──────┐               ┌──────────────────┐
  │ Conv │→│ Conv │   (1×1→3×3)──▶│ Fused Conv       │  1 kernel
  └──────┘ └──────┘               │ (depthwise sep)  │
                                   └──────────────────┘
```

### 2.2 Why Fusion Matters

Each kernel launch has overhead: ~5–10μs for launch + memory read/write for intermediate tensors:

$$T_{\text{unfused}} = \sum_{i=1}^{N} (T_{\text{launch}_i} + T_{\text{compute}_i} + T_{\text{memory}_i})$$

$$T_{\text{fused}} = T_{\text{launch}} + T_{\text{compute}} + T_{\text{memory}}$$

For a Conv-BN-ReLU sequence:
- **Unfused**: 3 kernel launches, 3 memory round-trips = ~30μs overhead + 3× DRAM bandwidth
- **Fused**: 1 kernel launch, intermediates stay in registers/shared memory = ~10μs overhead

---

## 3. Precision Calibration

TensorRT supports FP32, FP16, and INT8 precision. INT8 requires **calibration** to find optimal quantization ranges.

### 3.1 Precision Hierarchy

```
Precision Modes & Performance
══════════════════════════════════════════════════════════

  Precision    Bits    Tensor Core    Relative Speed    Accuracy
  ─────────    ────    ───────────    ──────────────    ────────
  FP32          32     ✗              1×                Baseline
  TF32          19     ✓ (Ampere+)    2×                ~FP32
  FP16          16     ✓              4×                ~FP32 (mixed)
  BF16          16     ✓ (Ampere+)    4×                ~FP32
  FP8           8      ✓ (Hopper+)    8×                ~FP16
  INT8           8     ✓              8×                Requires calibration
  INT4           4     ✓ (Ada+)       16×               LLM weights only
```

### 3.2 INT8 Calibration

Calibration finds the optimal clipping range $[\alpha, \beta]$ for each tensor:

$$x_{\text{quantized}} = \text{round}\left(\frac{x - \alpha}{\beta - \alpha} \times 255\right)$$

TensorRT offers multiple calibration algorithms:

| Calibrator | Method | Best For |
|------------|--------|----------|
| `IInt8EntropyCalibrator2` | KL divergence minimization | CNNs, general models |
| `IInt8MinMaxCalibrator` | Min/max range | When outliers are rare |
| `IInt8Percentile` | Percentile clipping | Models with outlier activations |

```python
import tensorrt as trt

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file="calibration.cache"):
        super().__init__()
        self.data_loader = iter(data_loader)
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(32 * 3 * 224 * 224 * 4)

    def get_batch_size(self): return 32

    def get_batch(self, names):
        try:
            batch = next(self.data_loader)
            cuda.memcpy_htod(self.device_input, batch.numpy().ravel())
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        try: return open(self.cache_file, "rb").read()
        except FileNotFoundError: return None

    def write_calibration_cache(self, cache):
        open(self.cache_file, "wb").write(cache)
```

---

## 4. Building and Running a TensorRT Engine

### 4.1 Full Pipeline: ONNX → Engine → Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, precision="fp16", max_batch=8):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8Calibrator(calibration_loader)

    # Dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
        min=(1, 3, 224, 224), opt=(4, 3, 224, 224), max=(max_batch, 3, 224, 224))
    config.add_optimization_profile(profile)

    # Build & serialize
    engine_bytes = builder.build_serialized_network(network, config)
    with open("model.engine", "wb") as f:
        f.write(engine_bytes)
    return trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_bytes)


Inference allocates device memory, copies input HtoD, executes asynchronously, copies output DtoH, then synchronizes. TensorRT engines are **GPU-specific** — rebuild for each target GPU and TensorRT version (use timing caches to speed rebuilds).

---

## 5. Profiling with trtexec

`trtexec` is TensorRT's CLI tool for benchmarking and profiling:

```bash
# Basic benchmark: ONNX → engine → benchmark
trtexec --onnx=resnet50.onnx \
        --fp16 \
        --workspace=4096 \
        --warmUp=500 \
        --iterations=1000 \
        --verbose

# Output:
# [07/15/2025-10:30:45] [I] === Performance summary ===
# [I] Throughput: 4892.31 qps
# [I] Latency: min = 0.189 ms, max = 0.312 ms, mean = 0.204 ms
# [I] GPU Compute Time: min = 0.181 ms, max = 0.295 ms, mean = 0.196 ms

# INT8 calibration with data
trtexec --onnx=resnet50.onnx \
        --int8 \
        --calib=/path/to/calibration_data \
        --saveEngine=resnet50_int8.engine

# Dynamic batch shapes
trtexec --onnx=model.onnx \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:32x3x224x224 \
        --fp16

# Layer-level profiling
trtexec --loadEngine=model.engine \
        --dumpProfile \
        --exportProfile=profile.json
```

The profile output shows per-layer time, percentage, and precision. Look for **compute-bound** layers (benefit from INT8) vs **memory-bound** layers (benefit from smaller weights, not lower precision).

---

## 6. TensorRT-LLM for Transformer Inference

TensorRT-LLM extends TensorRT with **LLM-specific optimizations**:

TensorRT-LLM pipeline: **Model Definition** (Python API with quantization annotations) → **trtllm-build** (compiles with LLM fusions) → **C++ Executor** (paged KV cache, in-flight batching, tensor/pipeline parallelism, beam search, speculative decoding, FP8/INT4).

```bash
# Convert checkpoint → build engine with LLM optimizations
python convert_checkpoint.py --model_dir /models/llama-2-7b \
    --output_dir /engines/llama-7b-ckpt --dtype float16

trtllm-build --checkpoint_dir /engines/llama-7b-ckpt \
    --output_dir /engines/llama-7b-engine --gemm_plugin float16 \
    --max_batch_size 64 --max_seq_len 4096 --paged_kv_cache enable
```

### 6.2 torch_tensorrt — PyTorch Integration

For simpler models, use TensorRT directly within PyTorch:

```python
import torch
import torch_tensorrt

model = torchvision.models.resnet50(pretrained=True).eval().cuda()
example_input = torch.randn(1, 3, 224, 224).cuda()

# Compile with torch_tensorrt
optimized = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=[1, 3, 224, 224],
            opt_shape=[8, 3, 224, 224],
            max_shape=[32, 3, 224, 224],
            dtype=torch.float16,
        )
    ],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30,
)

# Use like a normal PyTorch module
output = optimized(example_input.half())

# Save as TorchScript for deployment
torch.jit.save(optimized, "resnet50_trt.ts")
```

---

## Hands-On Exercises

### Exercise 1: End-to-End TensorRT Pipeline (30 min)
1. Export ResNet-50 from PyTorch to ONNX (opset 17, dynamic batch)
2. Build TensorRT engines at FP32, FP16, and INT8
3. Benchmark all three with `trtexec` — record throughput and latency
4. Compare accuracy on 1000 ImageNet validation images

### Exercise 2: Layer Profiling (20 min)
1. Run `trtexec --dumpProfile` on your FP16 engine
2. Identify the top-3 slowest layers
3. For each: is it compute-bound or memory-bound? (Check arithmetic intensity)
4. Would INT8 help the memory-bound layers? Why or why not?

### Exercise 3: Dynamic Batching (15 min)
1. Build an engine with `minShapes=1`, `optShapes=16`, `maxShapes=64`
2. Measure latency at batch sizes 1, 4, 16, 32, 64
3. Plot latency vs batch size — where does the GPU saturate?
4. Calculate throughput (images/second) at each batch size

### Exercise 4: torch_tensorrt Integration (15 min)
```python
# Compare inference speed:
# 1. Vanilla PyTorch (FP32)
# 2. PyTorch with torch.compile (inductor)
# 3. torch_tensorrt (FP16)
# Measure with torch.cuda.Event for accurate GPU timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    output = model(input_tensor)
end.record()
torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end) / 100:.2f} ms")
```

---

## Key Takeaways

1. **TensorRT rebuilds your model** — it's not just running your graph; it fuses layers, selects kernels, and plans memory for your specific GPU
2. **Layer fusion eliminates overhead** — Conv+BN+ReLU becomes one kernel, reducing launch overhead and DRAM round-trips by 3×
3. **INT8 needs calibration** — entropy-based calibration finds optimal quantization ranges; 100–500 representative samples is usually sufficient
4. **Engines are GPU-specific** — rebuild for each target GPU and TensorRT version; use timing caches to speed up rebuilds
5. **trtexec is your profiling tool** — layer-level profiling identifies bottlenecks; `--dumpProfile` shows where time goes
6. **TensorRT-LLM is the LLM path** — paged KV cache, in-flight batching, tensor parallelism, and FP8 quantization for transformer inference at scale

---

## Further Reading

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [TensorRT Best Practices Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#performance-best-practices)
- [TensorRT-LLM (GitHub)](https://github.com/NVIDIA/TensorRT-LLM)
- [torch_tensorrt Documentation](https://pytorch.org/TensorRT/)
- [trtexec Reference](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)

---

## Tomorrow's Preview

**Day 54: Inference on CPU** — Not every deployment has a GPU. You'll learn ONNX Runtime's CPU execution providers, Intel OpenVINO, ARM compute library, XNNPACK for mobile, and how to squeeze maximum throughput from x86 and ARM CPUs using vectorization, thread scheduling, and operator-level optimizations.
