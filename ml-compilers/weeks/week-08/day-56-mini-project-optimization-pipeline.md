# Day 56: Mini-Project — Full Optimization Pipeline

> **Phase IV · Week 8 · Day 56 of 70 · 2.5 hours**

*"Optimization without measurement is just guessing. Today you measure everything."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 55: Inference on Edge Devices](day-55-inference-on-edge.md) | [Day 57: LLM Inference Challenges](../week-09/day-57-llm-inference-challenges.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

This week you've learned individual optimization techniques in isolation — ONNX export, quantization, pruning, distillation, TensorRT, CPU inference, edge deployment. But in production, these techniques combine and interact. Quantizing a pruned model is different from pruning a quantized model. TVM's autotuning finds different schedules for INT8 vs FP32. The "best" configuration depends on your deployment target, latency budget, and accuracy floor. This capstone project builds the complete pipeline end-to-end and produces the one artifact that matters: an **accuracy-latency Pareto frontier** showing exactly what you trade for every millisecond you save.

---

## 1. Project Architecture

```
Full Optimization Pipeline — End-to-End
══════════════════════════════════════════════════════════════════════

  ┌────────────────┐
  │ PyTorch Model   │  (ResNet-18 or MobileNetV2, pretrained)
  │ FP32, 11.7M     │
  │ params          │
  └───────┬────────┘
          │
  ┌───────▼────────┐   ┌────────────────────────────────────────┐
  │ 1. ONNX Export │   │  Validate: onnx.checker.check_model()  │
  │    opset=17    │   │  Compare:  torch vs onnx max-diff < 1e-5│
  └───────┬────────┘   └────────────────────────────────────────┘
          │
          ├────────────────────┬──────────────────────┐
          ▼                    ▼                      ▼
  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │ 2a. PTQ      │   │ 2b. QAT          │   │ 2c. Prune + PTQ  │
  │ (INT8 static)│   │ (3 epoch retrain)│   │ (50% unstructured)│
  └──────┬───────┘   └────────┬─────────┘   └────────┬─────────┘
         │                    │                       │
         ├────────────────────┼───────────────────────┤
         ▼                    ▼                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 3. Compile to Target                                         │
  │    ┌──────────┐  ┌──────────────┐  ┌───────────────────┐    │
  │    │ TensorRT │  │ ORT CPU      │  │ TVM (LLVM target) │    │
  │    │ (FP16,   │  │ (oneDNN,     │  │ (AutoTVM tuned)   │    │
  │    │  INT8)   │  │  VNNI)       │  │                   │    │
  │    └──────────┘  └──────────────┘  └───────────────────┘    │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 4. Benchmark                                                 │
  │    • Latency (P50, P99) per configuration                   │
  │    • Top-1 accuracy on validation set                       │
  │    • Throughput (images/sec)                                 │
  │    • Model size (MB)                                        │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 5. Pareto Frontier                                           │
  │    Plot accuracy vs latency, identify optimal configurations │
  └──────────────────────────────────────────────────────────────┘
```

---

## 2. Step 1 — ONNX Export with Validation

```python
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# ── Load pretrained model ──
model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
model.eval()

# ── Export to ONNX ──
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18_fp32.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

# ── Validate ONNX model ──
onnx_model = onnx.load("resnet18_fp32.onnx")
onnx.checker.check_model(onnx_model)

# ── Numerical comparison: PyTorch vs ONNX Runtime ──
session = ort.InferenceSession("resnet18_fp32.onnx", providers=["CPUExecutionProvider"])

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
torch_output = model(torch.from_numpy(test_input)).detach().numpy()
ort_output = session.run(None, {"input": test_input})[0]

max_diff = np.max(np.abs(torch_output - ort_output))
print(f"Max difference PyTorch vs ORT: {max_diff:.2e}")
assert max_diff < 1e-5, f"Numerical mismatch: {max_diff}"
print("✓ ONNX export validated")
```

---

## 3. Step 2 — Quantization Variants

### 3.1 Post-Training Quantization (PTQ)

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class ImageNetCalibrator(CalibrationDataReader):
    """Feeds calibration images to the quantizer."""

    def __init__(self, calib_dir, num_samples=200):
        self.data = self._load_images(calib_dir, num_samples)
        self.iter = iter(self.data)

    def get_next(self):
        return next(self.iter, None)

    def _load_images(self, path, n):
        # Load and preprocess n images from calibration set
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ds = datasets.ImageFolder(path, transform=transform)
        items = []
        for i in range(min(n, len(ds))):
            img, _ = ds[i]
            items.append({"input": img.unsqueeze(0).numpy()})
        return items


# PTQ: static quantization with entropy calibration
quantize_static(
    model_input="resnet18_fp32.onnx",
    model_output="resnet18_ptq_int8.onnx",
    calibration_data_reader=ImageNetCalibrator("./calib_images"),
    quant_format=ort.quantization.QuantFormat.QDQ,
    activation_type=ort.quantization.QuantType.QInt8,
    weight_type=ort.quantization.QuantType.QInt8,
    calibrate_method=ort.quantization.CalibrationMethod.Entropy,
)
```

### 3.2 Quantization-Aware Training (QAT)

```python
import torch.quantization as quant

# QAT: retrain with fake quantization
model_qat = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
model_qat.train()
model_qat.qconfig = quant.get_default_qat_qconfig("x86")
model_qat = quant.prepare_qat(model_qat)

# Fine-tune for 3 epochs
optimizer = torch.optim.SGD(model_qat.parameters(), lr=1e-4, momentum=0.9)
for epoch in range(3):
    for images, labels in train_loader:
        output = model_qat(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"QAT epoch {epoch}: loss={loss.item():.4f}")

# Convert and export
model_qat_converted = quant.convert(model_qat.eval())
torch.onnx.export(model_qat_converted, dummy_input, "resnet18_qat_int8.onnx", opset_version=17)
```

### 3.3 Pruning + Quantization

```python
import torch.nn.utils.prune as prune

# Unstructured pruning: remove 50% of weights by magnitude
model_pruned = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")

for name, module in model_pruned.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.5)
        prune.remove(module, "weight")  # Make pruning permanent

# Check sparsity
total, zeros = 0, 0
for p in model_pruned.parameters():
    total += p.numel()
    zeros += (p == 0).sum().item()
print(f"Sparsity: {zeros / total * 100:.1f}%")

# Export pruned model → then apply PTQ
torch.onnx.export(model_pruned.eval(), dummy_input, "resnet18_pruned_50.onnx", opset_version=17)
# Follow with quantize_static() as in 3.1
```

---

## 4. Step 3 — Compilation to Targets

### 4.1 TensorRT (GPU)

```python
import tensorrt as trt

def build_trt_engine(onnx_path, precision="fp16"):
    """Build TensorRT engine from ONNX model."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # Set calibrator for INT8
        config.int8_calibrator = EntropyCalibrator(calib_data)

    engine = builder.build_serialized_network(network, config)
    with open(f"resnet18_{precision}.engine", "wb") as f:
        f.write(engine)
    return engine

build_trt_engine("resnet18_fp32.onnx", "fp16")
build_trt_engine("resnet18_fp32.onnx", "int8")
```

### 4.2 TVM with AutoTVM

```python
import tvm
from tvm import relay, auto_scheduler
import onnx

# Load ONNX model into TVM Relay
onnx_model = onnx.load("resnet18_fp32.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": (1, 3, 224, 224)})

# Target: LLVM for CPU
target = tvm.target.Target("llvm -mcpu=skylake-avx512")

# AutoScheduler: search for optimal schedules
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=2000,
    measure_callbacks=[auto_scheduler.RecordToFile("tuning_log.json")],
    verbose=1,
)
tuner.tune(tune_option)

# Compile with tuned schedules
with auto_scheduler.ApplyHistoryBest("tuning_log.json"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

lib.export_library("resnet18_tvm_tuned.so")
```

---

## 5. Step 4 — Unified Benchmarking

```python
"""
Unified benchmarking harness — same protocol for all backends.
"""
import time
import numpy as np
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    name: str
    accuracy_top1: float
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_ips: float
    model_size_mb: float

def benchmark_ort(model_path, val_loader, num_runs=200):
    """Benchmark ONNX Runtime CPU inference."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # ── Accuracy ──
    correct, total = 0, 0
    for images, labels in val_loader:
        out = session.run(None, {input_name: images.numpy()})[0]
        preds = np.argmax(out, axis=1)
        correct += (preds == labels.numpy()).sum()
        total += len(labels)
        if total >= 1000:
            break
    accuracy = correct / total

    # ── Latency ──
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(50):  # warmup
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    import os
    size_mb = os.path.getsize(model_path) / 1024 / 1024

    return BenchmarkResult(
        name=model_path,
        accuracy_top1=accuracy,
        latency_p50_ms=np.percentile(latencies, 50),
        latency_p99_ms=np.percentile(latencies, 99),
        throughput_ips=1000 / np.mean(latencies),
        model_size_mb=size_mb,
    )


# ── Run all configurations ──
configs = [
    "resnet18_fp32.onnx",
    "resnet18_ptq_int8.onnx",
    "resnet18_qat_int8.onnx",
    "resnet18_pruned_50_int8.onnx",
]

results = []
for cfg in configs:
    r = benchmark_ort(cfg, val_loader)
    results.append(r)
    print(f"{r.name:40s}  acc={r.accuracy_top1:.3f}  "
          f"p50={r.latency_p50_ms:.2f}ms  size={r.model_size_mb:.1f}MB")
```

---

## 6. Step 5 — Pareto Frontier Analysis

### 6.1 What Is a Pareto Frontier?

A configuration is **Pareto-optimal** if no other configuration is simultaneously better in both accuracy and latency:

$$\text{Config } A \text{ dominates } B \iff \text{acc}(A) \geq \text{acc}(B) \wedge \text{lat}(A) \leq \text{lat}(B)$$

$$\text{Pareto set} = \{A \mid \nexists B \text{ that dominates } A\}$$

### 6.2 Plotting the Pareto Frontier

```python
import matplotlib.pyplot as plt

def plot_pareto_frontier(results):
    """Plot accuracy vs latency and highlight Pareto-optimal points."""
    accs = [r.accuracy_top1 for r in results]
    lats = [r.latency_p50_ms for r in results]
    names = [r.name.replace(".onnx", "") for r in results]
    sizes = [r.model_size_mb * 5 for r in results]  # Bubble size = model size

    # Find Pareto-optimal points
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, s in enumerate(results):
            if i != j and s.accuracy_top1 >= r.accuracy_top1 and s.latency_p50_ms <= r.latency_p50_ms:
                if s.accuracy_top1 > r.accuracy_top1 or s.latency_p50_ms < r.latency_p50_ms:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Non-Pareto points
    for i in range(len(results)):
        color = "green" if i in pareto else "gray"
        marker = "★" if i in pareto else "o"
        ax.scatter(lats[i], accs[i], s=sizes[i] * 10, c=color, alpha=0.7, edgecolors="black")
        ax.annotate(names[i], (lats[i], accs[i]), fontsize=8, ha="left", va="bottom")

    # Pareto frontier line
    pareto_pts = sorted([(lats[i], accs[i]) for i in pareto])
    ax.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts],
            "g--", linewidth=2, label="Pareto frontier")

    ax.set_xlabel("Latency P50 (ms)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy vs Latency — Optimization Pipeline Results")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pareto_frontier.png", dpi=150)
    plt.show()

plot_pareto_frontier(results)
```

### 6.3 Expected Results (ResNet-18 on a Skylake CPU)

```
Expected Pareto Frontier — ResNet-18 Optimization
══════════════════════════════════════════════════════════

  Configuration          Accuracy   P50 Latency   Size     Pareto?
  ─────────────────────  ────────   ───────────   ──────   ───────
  FP32 (baseline)        69.76%     12.3 ms       44.7 MB    ✗
  PTQ INT8               69.51%      4.8 ms       11.4 MB    ★
  QAT INT8               69.72%      4.7 ms       11.4 MB    ★
  Pruned 50% FP32        69.20%      9.1 ms       44.7 MB    ✗
  Pruned 50% + INT8      68.90%      4.2 ms       11.4 MB    ★
  TensorRT FP16 (GPU)    69.76%      0.8 ms       22.4 MB    ★
  TensorRT INT8 (GPU)    69.55%      0.4 ms       11.5 MB    ★
  TVM autotuned FP32     69.76%     10.1 ms       44.7 MB    ✗

  Accuracy
  70.0% ┤ ★ QAT-INT8        ★ TRT-FP16
        │   ★ PTQ-INT8
  69.5% ┤                        ★ TRT-INT8
        │
  69.0% ┤     ★ Pruned+INT8
        │
        └──────┬───────┬───────┬──────────
             0.4      4.7    10.1    12.3  → Latency (ms)
```

---

## Hands-On Exercises

### Exercise 1: Build Your Pipeline
Implement the full pipeline from Sections 2–5 for MobileNetV2 instead of ResNet-18. You should have at least 6 configurations (FP32, PTQ, QAT, pruned, TRT FP16, TRT INT8). Generate the Pareto plot.

### Exercise 2: Structured vs Unstructured Pruning
Compare 50% unstructured pruning vs 30% structured (channel) pruning on ResNet-18. Which gives better actual latency reduction? Why does unstructured pruning often fail to improve latency despite removing weights?

### Exercise 3: Cross-Target Comparison
Take the same model and benchmark it on: (a) your x86 CPU with ORT, (b) GPU with TensorRT, (c) TFLite on CPU. Create a 3-column table showing the accuracy-latency-power tradeoff.

### Exercise 4 (Stretch): Automated Pipeline
Write a single `optimize.py` script that takes a PyTorch model path and produces a Pareto chart. Use argparse for model path, calibration data path, and target devices.

---

## Key Takeaways

1. **Pipeline order matters** — prune first, then quantize; the reverse can eliminate weights that quantization needs for range estimation
2. **PTQ is good enough for large models** — ResNet-18 loses only ~0.25% accuracy with INT8 PTQ; save QAT for smaller or accuracy-critical models
3. **Pruning without structure doesn't help latency** — 50% unstructured sparsity looks good on paper but doesn't reduce compute on dense hardware; structured pruning actually removes channels
4. **TensorRT dominates on GPU** — FP16 matches FP32 accuracy with 15× lower latency; INT8 adds another 2× on top
5. **The Pareto frontier is the deliverable** — production teams don't want one model; they want the accuracy-latency tradeoff curve so they can pick the operating point for their constraints
6. **Automate the pipeline** — manual optimization doesn't scale; build scripts that produce Pareto curves for every model release

---

## Further Reading

- [ONNX Runtime Quantization Tutorial](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [TVM AutoScheduler Guide](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/)
- [MLPerf Inference Benchmark](https://mlcommons.org/en/inference-datacenter/)
- [Efficient Deep Learning Book (MIT)](https://efficientdlbook.com/)
- [Neural Architecture Search + Compression (Han et al.)](https://arxiv.org/abs/1802.03494)

---

## Week 8 Complete!

You've covered the entire model optimization stack: formats (ONNX), compression (quantization, pruning, distillation), hardware-specific compilation (TensorRT, OpenVINO, TVM), and deployment targets (GPU, CPU, edge, microcontrollers). Next week dives into **LLM-specific inference** — where everything changes because the models are 100× larger, autoregressive, and memory-bound.

**Next: [Day 57: LLM Inference Challenges](../week-09/day-57-llm-inference-challenges.md)** — KV cache, prefill vs decode, memory bandwidth walls, and why LLM inference is fundamentally different from CNN inference.
