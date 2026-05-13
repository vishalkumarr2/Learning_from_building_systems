# Day 55: Inference on Edge Devices

> **Phase IV · Week 8 · Day 55 of 70 · 2.5 hours**

*"The best model is the one that runs where the data lives — and on edge devices, every millijoule and millisecond is a budget item."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 54: Inference on CPU](day-54-inference-on-cpu.md) | [Day 56: Mini-Project — Full Optimization Pipeline](day-56-mini-project-optimization-pipeline.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Edge inference is where ML meets physics. Your model runs on a phone with a 4000 mAh battery, a drone with 100 ms latency budget, or a microcontroller with 256 KB of RAM. There is no cloud round-trip, no unlimited DRAM, no 300W power envelope. The global edge AI market is projected to reach $40B by 2027, driven by autonomous vehicles, AR glasses, smart cameras, and medical wearables. Mastering edge deployment means understanding an entirely different optimization landscape: **delegates** that route operators to DSPs and NPUs, **quantization-aware training** that maintains accuracy at INT8/INT4, and tools like TVM's µTVM that compile models for bare-metal microcontrollers. The tradeoff triangle — latency, power, accuracy — defines every decision.

---

## 1. Edge Deployment Landscape

```
Edge Inference Ecosystem
══════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                     Edge Device Spectrum                        │
  ├────────────┬────────────┬────────────┬────────────┬────────────┤
  │ MCU        │ Mobile     │ SBC        │ Edge GPU   │ Edge TPU   │
  │ Cortex-M7  │ Snapdragon │ RPi 5     │ Jetson Orin│ Coral      │
  │ 256KB-2MB  │ 8GB+ RAM   │ 8GB RAM   │ 8-64GB     │ 8 TOPS     │
  │ ~0.5 TOPS  │ ~15 TOPS   │ ~2 TOPS   │ ~275 TOPS  │ INT8 only  │
  │ <1W        │ 5-10W      │ 5-15W     │ 15-60W     │ 2W         │
  └─────┬──────┴──────┬─────┴─────┬─────┴──────┬─────┴──────┬─────┘
        │             │           │            │            │
  ┌─────▼──────┐ ┌────▼────┐ ┌───▼────┐ ┌─────▼─────┐ ┌────▼────┐
  │ TFLite     │ │ TFLite  │ │ ORT    │ │ TensorRT  │ │ TFLite  │
  │ Micro      │ │ NNAPI   │ │ TFLite │ │ (JetPack) │ │ EdgeTPU │
  │ µTVM       │ │ Core ML │ │ ncnn   │ │ DeepStream│ │ Delegate│
  └────────────┘ │ QNN     │ └────────┘ └───────────┘ └─────────┘
                 └─────────┘
```

---

## 2. TensorFlow Lite (TFLite)

TFLite is the most widely deployed edge inference framework — it runs on Android, iOS, Linux, and microcontrollers with a ~1MB runtime binary.

### 2.1 Model Conversion and Quantization

```python
import tensorflow as tf
import numpy as np

# ── Convert a SavedModel to TFLite ──
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_v2")

# Enable standard optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for full integer quantization
def representative_dataset():
    for _ in range(100):
        data = np.random.randn(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# Full INT8 quantization (inputs and outputs too)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save — check the size difference
with open("mobilenet_v2_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024 / 1024:.1f} MB")
```

### 2.2 Delegate Architecture

```
TFLite Delegate System
══════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────┐
  │  TFLite Interpreter                                  │
  │                                                      │
  │  Operator graph: [Conv] → [BN] → [ReLU] → [Pool]   │
  │                    │        │       │        │       │
  │  Delegate check:  GPU?    GPU?    GPU?     CPU      │
  │                    ✓        ✓       ✓        ✗       │
  │                    │        │       │        │       │
  │                    ▼        ▼       ▼        ▼       │
  │  ┌─────────────────────────────┐  ┌──────────┐      │
  │  │ GPU Delegate (fused subgraph)│  │ CPU      │      │
  │  │  Conv → BN → ReLU           │  │ Pool     │      │
  │  │  (OpenCL / Metal / Vulkan)   │  │ (XNNPACK)│      │
  │  └─────────────────────────────┘  └──────────┘      │
  └──────────────────────────────────────────────────────┘

  Available Delegates:
  ┌──────────────┬────────────────┬────────────────────┐
  │ Delegate     │ Hardware       │ Platform           │
  ├──────────────┼────────────────┼────────────────────┤
  │ XNNPACK      │ CPU (SIMD)     │ Android/iOS/Linux  │
  │ GPU          │ Mobile GPU     │ Android (OpenCL)   │
  │              │                │ iOS (Metal)        │
  │ NNAPI        │ Android NPU    │ Android 8.1+       │
  │ Hexagon      │ Qualcomm DSP   │ Snapdragon         │
  │ QNN          │ Qualcomm NPU   │ Snapdragon 8 Gen2+ │
  │ CoreML       │ Apple ANE      │ iOS 12+            │
  │ Edge TPU     │ Google Coral    │ USB/PCIe/SoM       │
  └──────────────┴────────────────┴────────────────────┘
```

### 2.3 Running with Delegates

```python
import tflite_runtime.interpreter as tflite

# ── NNAPI Delegate (Android NPU) ──
interpreter = tflite.Interpreter(
    model_path="mobilenet_v2_int8.tflite",
    experimental_delegates=[tflite.load_delegate("libnnapi_delegate.so")],
)
interpreter.allocate_tensors()

# ── Edge TPU Delegate (Coral) ──
interpreter = tflite.Interpreter(
    model_path="mobilenet_v2_edgetpu.tflite",
    experimental_delegates=[
        tflite.load_delegate("libedgetpu.so.1.0", {"device": "usb"})
    ],
)

# ── XNNPACK Delegate (optimized CPU) ──
interpreter = tflite.Interpreter(
    model_path="mobilenet_v2_fp32.tflite",
    num_threads=4,
    experimental_delegates=[
        tflite.load_delegate("libXNNPACK_delegate.so")
    ],
)
```

---

## 3. Apple Core ML

Core ML is the inference framework for Apple devices — it automatically dispatches operators to the CPU, GPU, or Apple Neural Engine (ANE) for optimal performance.

### 3.1 Converting to Core ML

```python
import coremltools as ct
import torch

# Export PyTorch model to Core ML
model = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=True)
model.eval()

traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 224, 224),
                         scale=1/255.0, bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225])],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
    compute_precision=ct.precision.FLOAT16,
)

# Quantize to INT8 with calibration
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

op_config = OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
config = OptimizationConfig(global_config=op_config)
mlmodel_int8 = linear_quantize_weights(mlmodel, config=config)

mlmodel_int8.save("mobilenet_v2_int8.mlpackage")
```

### 3.2 ANE Compatibility Constraints

```
Apple Neural Engine (ANE) — What It Supports
══════════════════════════════════════════════════════════

  Supported (runs on ANE at full speed):
  ✓ Conv2d (up to 8192 channels)
  ✓ Depthwise Conv2d
  ✓ Linear / MatMul
  ✓ Elementwise (Add, Mul, ReLU, Sigmoid, Tanh)
  ✓ Pooling (Average, Max)
  ✓ Softmax, LayerNorm, InstanceNorm, BatchNorm
  ✓ Reshape, Transpose, Concat, Split, Slice

  Falls back to GPU/CPU:
  ✗ Dynamic shapes (variable batch/seq length)
  ✗ Gather with non-constant indices
  ✗ Complex control flow (if/while)
  ✗ Custom operators
  ✗ INT8 activations (ANE uses FP16 internally)

  Performance tip: Keep the entire graph ANE-compatible
  to avoid CPU↔ANE memory copies (~0.5ms per transfer)
```

---

## 4. TVM's µTVM for Microcontrollers

µTVM (micro TVM) compiles ML models for bare-metal devices with no OS, no dynamic memory allocation, and kilobytes of RAM.

### 4.1 µTVM Compilation Flow

```
µTVM Pipeline — From Model to Bare-Metal Binary
══════════════════════════════════════════════════════════

  ┌────────────┐    ┌──────────────┐    ┌───────────────┐
  │ TFLite     │    │ TVM Relay    │    │ TVM Schedule  │
  │ Model      │───▶│ IR           │───▶│ Optimization  │
  │ (quantized)│    │ (graph-level)│    │ (tile, unroll) │
  └────────────┘    └──────────────┘    └───────┬───────┘
                                                │
                                                ▼
  ┌────────────┐    ┌──────────────┐    ┌───────────────┐
  │ .bin flash │    │ C Code       │    │ TIR (Tensor   │
  │ to MCU     │◀───│ Generation   │◀───│  IR) Lowering │
  │            │    │ (no malloc!) │    │               │
  └────────────┘    └──────────────┘    └───────────────┘

  Memory constraints:
  ┌──────────────────────────────────────────────────────┐
  │  Target: ARM Cortex-M7 (STM32F7)                    │
  │  Flash: 2 MB  (model weights + code)                │
  │  SRAM:  512 KB (activations + workspace)            │
  │                                                      │
  │  µTVM scheduler:                                     │
  │  • Statically allocates all buffers at compile time  │
  │  • No heap allocation, no malloc/free                │
  │  • Operator fusion to minimize peak activation memory│
  │  • Generates plain C — compiles with arm-none-eabi-gcc│
  └──────────────────────────────────────────────────────┘
```

### 4.2 µTVM in Practice

```python
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import tvm.micro as micro

# Load a TFLite model
tflite_model_buf = open("person_detect_int8.tflite", "rb").read()
tflite_model = tvm.relay.frontend.from_tflite(
    tflite_model_buf,
    shape_dict={"input": [1, 96, 96, 1]},
    dtype_dict={"input": "int8"},
)

# Target: ARM Cortex-M7
target = tvm.target.Target("c -mcpu=cortex-m7 -runtime=c --executor=aot --link-params")

# Compile with AOT (Ahead-of-Time) executor
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = relay.build(tflite_model["main"], target=target, params=tflite_model["params"])

# Export C source files for embedding
module.export_library("model.tar")
# → Contains: model.c, model.h, params.bin
# → Compile with: arm-none-eabi-gcc -mcpu=cortex-m7 -mfpu=fpv5-sp-d16
```

---

## 5. Quantization-Aware Training (QAT) for Edge

Post-training quantization (PTQ) works for large models but often fails on small edge models. QAT inserts fake quantization nodes during training so the model learns to be robust to quantization noise.

### 5.1 QAT vs PTQ Accuracy Comparison

```
Accuracy Comparison: PTQ vs QAT on Edge Models
══════════════════════════════════════════════════════════

  Model              FP32    PTQ INT8   QAT INT8   QAT Gain
  ─────────────────  ─────   ────────   ────────   ────────
  MobileNetV2        71.8%   69.1%      71.5%      +2.4%
  MobileNetV3-Small  67.4%   62.8%      66.9%      +4.1%
  EfficientNet-B0    77.1%   75.3%      76.8%      +1.5%
  MNASNet-1.0        73.5%   70.2%      73.1%      +2.9%

  Observation: Smaller models degrade more with PTQ
  because fewer parameters means less redundancy to
  absorb quantization noise.
```

### 5.2 QAT with PyTorch

```python
import torch
import torch.quantization as quant

# Prepare model for QAT
model.train()
model.qconfig = quant.get_default_qat_qconfig("x86")  # or "qnnpack" for ARM

# Insert fake quantize modules
model_prepared = quant.prepare_qat(model)

# Train for a few epochs with fake quantization
optimizer = torch.optim.SGD(model_prepared.parameters(), lr=1e-4)
for epoch in range(5):
    for images, labels in train_loader:
        output = model_prepared(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Freeze BN statistics after epoch 3 for stability
    if epoch >= 3:
        model_prepared.apply(quant.disable_observer)
        model_prepared.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# Convert to quantized model
model_quantized = quant.convert(model_prepared.eval())
```

---

## 6. Latency–Power–Accuracy Tradeoff

### 6.1 The Edge Tradeoff Triangle

$$\text{Edge Score} = \frac{\text{Accuracy}}{\text{Latency} \times \text{Power}}$$

This is an informal metric, but it captures the core tension:

```
Edge Tradeoff Triangle
══════════════════════════════════════════════════════════

                    Accuracy
                       △
                      / \
                     /   \
                    /     \
                   / Sweet \
                  /  Spot   \
                 /     ★     \
                /             \
               ───────────────
           Latency            Power

  Moving toward one vertex hurts the other two:
  • Higher accuracy → larger model → higher latency + power
  • Lower latency  → aggressive quantization → lower accuracy
  • Lower power    → clock gating / simpler ops → higher latency
```

### 6.2 Benchmarking Across Edge Runtimes

```python
"""
Benchmark the same MobileNetV2 model across edge runtimes.
Run this on a Raspberry Pi 4 or similar ARM device.
"""
import subprocess
import json

models = {
    "fp32_tflite": "mobilenet_v2_fp32.tflite",
    "int8_tflite": "mobilenet_v2_int8.tflite",
    "int8_xnnpack": "mobilenet_v2_int8.tflite",  # with XNNPACK delegate
    "int8_ort": "mobilenet_v2_int8.onnx",
}

# TFLite benchmark tool
for name, path in models.items():
    cmd = [
        "benchmark_model",
        f"--graph={path}",
        "--num_runs=100",
        "--warmup_runs=20",
        "--num_threads=4",
    ]
    if "xnnpack" in name:
        cmd.append("--use_xnnpack=true")

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse: "Inference (avg): 12.3ms"
    print(f"{name}: {result.stdout}")
```

---

## Hands-On Exercises

### Exercise 1: TFLite Conversion Pipeline
Convert a pretrained MobileNetV3-Small from PyTorch → ONNX → TFLite (INT8). Compare the model file sizes and top-1 accuracy on ImageNet validation set (100 images is sufficient).

### Exercise 2: Delegate Benchmarking
On an Android phone (or emulator), benchmark the same TFLite model with: (a) CPU only, (b) XNNPACK delegate, (c) GPU delegate, (d) NNAPI delegate. Report latency for each.

### Exercise 3: Memory Budget Analysis
For a Cortex-M7 target with 512 KB SRAM, calculate the peak activation memory needed for a [96×96×1] → 3×Conv2d → GlobalPool → Dense → [2] person detection model. Can it fit?

---

## Key Takeaways

1. **The edge is fragmented** — each platform has its own runtime and accelerator; TFLite is the closest to universal, but delegates are hardware-specific
2. **Delegates are essential** — routing operators to the NPU/DSP via NNAPI, QNN, or Core ML delegates can give 5–20× speedup over CPU-only
3. **QAT outperforms PTQ on small models** — when model capacity is limited, training with fake quantization recovers 2–4% accuracy lost by PTQ
4. **µTVM enables bare-metal ML** — no OS, no heap, no dynamic shapes; the entire inference graph is compiled to static C code
5. **ANE compatibility matters** — one unsupported operator forces a CPU fallback and a costly memory transfer, negating the ANE advantage
6. **Measure the tradeoff triangle** — optimizing for latency alone is insufficient; power consumption determines battery life and thermal constraints on edge devices

---

## Further Reading

- [TFLite Model Optimization Guide](https://www.tensorflow.org/lite/performance/best_practices)
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Apache TVM µTVM Tutorial](https://tvm.apache.org/docs/topic/microtvm/)
- [Qualcomm Neural Processing SDK (QNN)](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- [MLPerf Tiny Benchmark](https://mlcommons.org/en/inference-tiny/)
- [MCUNet: Tiny Deep Learning on IoT Devices (MIT)](https://mcunet.mit.edu/)

---

## Tomorrow's Preview

**Day 56: Mini-Project — Full Optimization Pipeline** — Week 8 capstone. You'll take a model from raw PyTorch through the complete optimization pipeline: ONNX export → quantization (PTQ vs QAT) → pruning → TVM/TensorRT compilation → multi-target benchmarking. The goal: map the accuracy-latency Pareto frontier across GPU, CPU, and edge.
