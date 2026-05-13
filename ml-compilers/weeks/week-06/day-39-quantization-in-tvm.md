# Day 39: Quantization in TVM

> **Phase III · Week 6 · Day 39 of 70 · 2.5 hours**

*"Quantization is the art of throwing away precision you don't need — and the science of knowing exactly how much you can afford to lose."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 38: BYOC — Bring Your Own Codegen](day-38-byoc-custom-backends.md) | [Day 40: TVM for Edge Devices](day-40-tvm-for-edge-devices.md) | [Week 6: TVM Tuning & Backends](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Modern neural networks are trained in FP32 (32-bit floating point), but inference rarely needs that precision. **Quantization** maps weights and activations to lower-precision types — typically INT8 (8-bit integers) — for 2–4× speedup and 4× memory reduction with minimal accuracy loss. Every major deployment target (NVIDIA GPUs via INT8 Tensor Cores, Intel CPUs via VNNI, Arm via NEON/SVE, mobile NPUs) has dedicated hardware for integer arithmetic. TVM provides a complete quantization pipeline: calibration, the QNN dialect for quantized operators, and compilation to target-specific INT8 instructions. Understanding this pipeline is essential for deploying efficient models at the edge.

---

## 1. Quantization Fundamentals

### The Core Mapping

Quantization maps floating-point values to integers using a **scale** ($s$) and **zero point** ($z$):

$$q = \text{round}\left(\frac{x}{s}\right) + z$$

$$x = s \cdot (q - z)$$

where:
- $x$ is the real (FP32) value
- $q$ is the quantized (INT8) value
- $s$ is the scale factor (FP32)
- $z$ is the zero point (INT8)

### Symmetric vs Asymmetric

```
Symmetric (z = 0):
  FP32 range: [-α, +α]  →  INT8 range: [-127, +127]
  Scale: s = α / 127
  
  ◄──────────────────────────────────────────────►
  -α        -α/2         0         +α/2        +α     FP32
  -127       -64         0          +64        +127    INT8

Asymmetric (z ≠ 0):
  FP32 range: [x_min, x_max]  →  INT8 range: [0, 255]
  Scale: s = (x_max - x_min) / 255
  Zero point: z = round(-x_min / s)
  
  ◄──────────────────────────────────────────────►
  x_min                                   x_max       FP32
  0                                       255         UINT8
       ↑ z (maps to 0.0 in FP32)
```

### Per-Tensor vs Per-Channel

| Granularity | Scale/ZP | Accuracy | Speed |
|:--|:--|:--|:--|
| **Per-tensor** | One $(s, z)$ per entire tensor | Lower | Faster (simpler compute) |
| **Per-channel** | One $(s, z)$ per output channel | Higher | Slightly slower |

Per-channel is preferred for weights (especially convolutions) because different channels can have very different value ranges:

$$s_c = \frac{\max(|W_{c,:}|)}{127}, \quad z_c = 0 \quad \text{(symmetric per-channel)}$$

---

## 2. Quantized Arithmetic

### INT8 Matrix Multiplication

The key insight: multiplying two INT8 values produces an INT32 result, which is then rescaled:

$$C_{fp} = s_A \cdot s_B \cdot \left(\sum_k (q_A[i,k] - z_A) \cdot (q_B[k,j] - z_B)\right)$$

Expanding and optimizing:

```
INT8 × INT8 → INT32 accumulation (done in integer hardware)
    │
    ▼
INT32 result + precomputed bias corrections
    │
    ▼
Rescale: multiply by (s_A · s_B / s_C), add z_C
    │
    ▼
INT8 output (clamp to [0, 255] or [-128, 127])
```

The entire inner loop stays in integer arithmetic — only the final rescaling uses floating point (or fixed-point on devices without FPU).

### Fused Quantized Operations

```
FP32 pipeline:           conv2d → bias_add → relu → pool
                          (FP32)   (FP32)   (FP32)  (FP32)
                          
INT8 pipeline:            qconv2d → requantize → relu → pool
                          (INT8→INT32) (INT32→INT8) (INT8) (INT8)
```

The `requantize` step rescales the INT32 accumulator back to INT8 between operations.

---

## 3. TVM's Quantization Pipeline

### Three-Phase Workflow

```
Phase 1: Calibration           Phase 2: Quantization        Phase 3: Compilation
┌───────────────────┐          ┌─────────────────┐          ┌──────────────┐
│ Run FP32 model on │          │ Insert quantize/ │          │ Lower QNN ops│
│ representative    │  ──▶     │ dequantize nodes │  ──▶     │ to target    │
│ data, collect     │          │ using calibrated │          │ hardware     │
│ activation ranges │          │ scale/zero_point │          │ instructions │
└───────────────────┘          └─────────────────┘          └──────────────┘
```

### Phase 1: Calibration

```python
import tvm
import tvm.relay as relay
from tvm.relay.quantize import quantize as qtz
import numpy as np

# Load an FP32 model
mod, params = relay.frontend.from_pytorch(traced_model, input_infos)

# Calibration: collect activation statistics
with qtz.qconfig(
    calibrate_mode="kl_divergence",       # or "global_scale", "percentile"
    weight_scale="max",                    # symmetric for weights
    calibrate_chunk_by=16,                 # batch calibration data
    skip_conv_layers=[],                   # quantize all conv layers
    skip_dense_layer=False,
):
    # Provide calibration data via a dataset generator
    def calibrate_dataset():
        for i in range(100):
            # Representative input data (100 samples)
            data = np.random.uniform(size=(1, 3, 224, 224)).astype("float32")
            yield {"input": data}
    
    # Run quantization (calibrate + transform)
    qmod = qtz.quantize(mod, params, dataset=calibrate_dataset())

print(qmod.astext()[:2000])
```

### Calibration Methods

| Method | How It Works | Best For |
|:--|:--|:--|
| **max** | $s = \max(|x|) / 127$ | Weights, small ranges |
| **percentile** | $s = \text{percentile}(|x|, 99.99) / 127$ | Activations with outliers |
| **KL divergence** | Minimize KL divergence between FP32 and INT8 distributions | Best accuracy, slower |

### KL Divergence Calibration

KL divergence finds the optimal clipping threshold $\alpha$ by minimizing the information loss:

$$\alpha^* = \arg\min_\alpha D_{KL}(P_{\text{FP32}} \| Q_{\alpha,\text{INT8}})$$

where $P$ is the FP32 activation histogram and $Q_\alpha$ is the INT8 approximation clipped at $\alpha$.

```
Activation distribution:
                    ┌─┐
                  ┌─┤ ├─┐
                ┌─┤ │ │ ├─┐
              ┌─┤ │ │ │ │ ├─┐
            ┌─┤ │ │ │ │ │ │ ├─┐
          ──┤ │ │ │ │ │ │ │ │ ├──
  ◄─────────┼─┴─┴─┴─┴─┴─┴─┴─┼──────►
          -α*    clipping    +α*
           threshold         threshold
           
  Values outside [-α*, +α*] are clipped.
  KL divergence finds the α* that minimizes accuracy loss.
```

---

## 4. The QNN Dialect

TVM's **QNN (Quantized Neural Network)** dialect provides Relay-level operators for quantized computation:

### QNN Operators

```python
# QNN operators are Relay ops with explicit quantization parameters
from tvm.relay import qnn

# Quantized convolution
qnn.op.conv2d(
    data,                        # INT8 input
    weight,                      # INT8 weights
    input_zero_point=relay.const(128, "int32"),
    kernel_zero_point=relay.const(0, "int32"),
    input_scale=relay.const(0.0078, "float32"),
    kernel_scale=relay.const(0.0039, "float32"),
    kernel_size=(3, 3),
    channels=64,
    out_dtype="int32",           # accumulate in INT32
)

# Requantize: INT32 → INT8 (rescale between operations)
qnn.op.requantize(
    accumulated,                 # INT32 input
    input_scale=relay.const(0.0078 * 0.0039, "float32"),
    input_zero_point=relay.const(0, "int32"),
    output_scale=relay.const(0.0312, "float32"),
    output_zero_point=relay.const(128, "int32"),
    out_dtype="uint8",
)

# Quantize: FP32 → INT8 (at model input)
qnn.op.quantize(
    fp32_data,
    output_scale=relay.const(0.0078, "float32"),
    output_zero_point=relay.const(128, "int32"),
    out_dtype="uint8",
)

# Dequantize: INT8 → FP32 (at model output)
qnn.op.dequantize(
    int8_data,
    input_scale=relay.const(0.0312, "float32"),
    input_zero_point=relay.const(128, "int32"),
)
```

### QNN Lowering

During compilation, QNN ops are lowered to target-specific implementations:

```
qnn.conv2d (Relay level)
    │
    ├──▶ LLVM: VNNI instructions (Intel)
    │     vpdpbusd ymm0, ymm1, ymm2  (INT8 dot product)
    │
    ├──▶ CUDA: INT8 Tensor Core (NVIDIA)
    │     wmma::mma_sync (INT8 × INT8 → INT32)
    │
    └──▶ ARM: NEON sdot/udot instructions
          vsdot.s8 q0, q1, q2  (4-element dot product)
```

---

## 5. End-to-End INT8 Inference Pipeline

### Complete Example

```python
import tvm
import tvm.relay as relay
from tvm.relay.quantize import quantize as qtz
from tvm.contrib import graph_executor
import numpy as np

# 1. Load FP32 model
import torchvision.models as models
import torch
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
input_shape = (1, 3, 224, 224)
dummy = torch.randn(input_shape)
traced = torch.jit.trace(model, dummy)
mod, params = relay.frontend.from_pytorch(traced, [("input", input_shape)])

target = tvm.target.Target("llvm -mcpu=cascadelake")  # supports VNNI

# 2. Calibrate and quantize
with qtz.qconfig(
    calibrate_mode="kl_divergence",
    weight_scale="power2",          # power-of-2 scales for bitshift
    round_for_shift=True,
    skip_conv_layers=[0],           # keep first conv in FP32 (common practice)
):
    calib_data = [np.random.randn(*input_shape).astype("float32") for _ in range(50)]
    
    def dataset():
        for d in calib_data:
            yield {"input": d}
    
    qmod = qtz.quantize(mod, params, dataset=dataset())

# 3. Compile quantized model
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(qmod, target=target)

# 4. Run inference
dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))

input_data = np.random.randn(*input_shape).astype("float32")
module.set_input("input", input_data)
module.run()
output = module.get_output(0).numpy()

# 5. Benchmark
ftimer = module.module.time_evaluator("run", dev, number=100, repeat=3)
prof = ftimer()
print(f"INT8 latency: {prof.mean * 1000:.2f} ms")
```

### Accuracy Validation

```python
# Compare FP32 vs INT8 outputs
def validate_accuracy(fp32_module, int8_module, test_loader, top_k=5):
    fp32_correct = 0
    int8_correct = 0
    total = 0
    
    for images, labels in test_loader:
        # FP32 inference
        fp32_module.set_input("input", images.numpy())
        fp32_module.run()
        fp32_pred = fp32_module.get_output(0).numpy()
        
        # INT8 inference
        int8_module.set_input("input", images.numpy())
        int8_module.run()
        int8_pred = int8_module.get_output(0).numpy()
        
        fp32_correct += (fp32_pred.argmax(1) == labels.numpy()).sum()
        int8_correct += (int8_pred.argmax(1) == labels.numpy()).sum()
        total += len(labels)
    
    print(f"FP32 Top-1: {fp32_correct/total:.4f}")
    print(f"INT8 Top-1: {int8_correct/total:.4f}")
    print(f"Accuracy drop: {(fp32_correct - int8_correct)/total:.4f}")
```

---

## 6. Comparison with PyTorch Quantization

| Aspect | TVM Quantization | PyTorch Quantization |
|:--|:--|:--|
| **Type** | Post-training (PTQ) | PTQ + Quantization-Aware Training (QAT) |
| **Calibration** | KL divergence, percentile, max | MinMax, MovingAverage, Histogram |
| **Backend** | Any TVM target (CPU, GPU, NPU) | FBGEMM (x86), QNNPACK (Arm) |
| **Graph optimizations** | Full Relay pass pipeline | `torch.fx` graph mode |
| **Quantization-aware training** | Not built-in (import QAT model) | Native `torch.ao.quantization` |
| **Custom operators** | QNN dialect extensible | Limited to supported ops |
| **Deployment** | TVM runtime (portable) | PyTorch runtime or ONNX export |

### When to Use Which

```
Need QAT (highest accuracy)?
    │
    └── Yes → Train with PyTorch QAT → import quantized model into TVM
    
Deploying to non-standard hardware (NPU, FPGA)?
    │
    └── Yes → TVM quantization (target-aware lowering)
    
Standard x86/Arm deployment?
    │
    ├── Want portability → TVM
    └── Want simplicity  → PyTorch + FBGEMM/QNNPACK
```

### Importing PyTorch QAT Models

```python
# Train with PyTorch QAT, then import into TVM
import torch.ao.quantization as quant

# PyTorch QAT training (simplified)
model_fp32 = models.resnet18(pretrained=True)
model_fp32.qconfig = quant.get_default_qat_qconfig("fbgemm")
model_prepared = quant.prepare_qat(model_fp32.train())
# ... train for a few epochs ...
model_int8 = quant.convert(model_prepared.eval())

# Export and import into TVM
traced_int8 = torch.jit.trace(model_int8, dummy)
mod, params = relay.frontend.from_pytorch(traced_int8, [("input", input_shape)])
# TVM recognizes quantized ops and maps them to QNN dialect
```

---

## Hands-On Exercises

### Exercise 1: Quantize MobileNetV2 (30 min)

Quantize MobileNetV2 using TVM's pipeline with KL divergence calibration. Use 200 calibration images (random or from ImageNet validation). Measure:
1. FP32 latency
2. INT8 latency
3. Speedup ratio
4. Top-1 accuracy change (if you have ImageNet validation data)

### Exercise 2: Calibration Method Comparison (25 min)

Quantize the same model three times with different calibration methods: `"max"`, `"percentile"`, `"kl_divergence"`. Compare:
1. Output MSE vs FP32 reference (lower is better)
2. Compilation time (calibration overhead)
3. Any per-layer accuracy differences

### Exercise 3: Per-Channel vs Per-Tensor (20 min)

Modify the quantization config to compare per-tensor vs per-channel weight quantization:

```python
# Per-tensor
with qtz.qconfig(weight_scale="max", ...):
    qmod_per_tensor = qtz.quantize(mod, params, dataset=dataset())

# Per-channel  
with qtz.qconfig(weight_scale="channel_max", ...):
    qmod_per_channel = qtz.quantize(mod, params, dataset=dataset())
```

Which has lower output MSE? On which layers is the difference largest?

---

## Key Takeaways

1. **Quantization maps FP32 → INT8** using scale and zero point: $q = \text{round}(x/s) + z$
2. **Symmetric** quantization ($z=0$) is simpler; **asymmetric** handles one-sided distributions (e.g., ReLU outputs)
3. **Per-channel** weight quantization preserves accuracy better than per-tensor
4. **TVM's pipeline**: calibrate (collect ranges) → quantize (insert QNN ops) → compile (lower to hardware)
5. **KL divergence** calibration finds optimal clipping thresholds by minimizing information loss
6. **QNN dialect** provides Relay-level quantized operators that lower to VNNI, INT8 Tensor Cores, or NEON dot products

---

## Further Reading

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Google's landmark quantization paper (Jacob et al., 2018)
- [TVM Quantization Tutorial](https://tvm.apache.org/docs/how_to/deploy_models/deploy_quantized.html) — official deployment guide
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) — comprehensive survey (Gholami et al., 2021)
- [QNN Dialect RFC](https://discuss.tvm.apache.org/t/rfc-qnn-dialect-for-relay/3operand/3539) — design rationale for TVM's quantized operators

---

## Tomorrow's Preview

Quantization shines brightest on **edge devices** — microcontrollers, mobile SoCs, and NPUs where every byte of memory and every milliwatt matters. **Day 40** covers TVM's edge deployment story: compiling for Arm Cortex-M via µTVM, Android/iOS deployment, and the challenges of running models on devices with kilobytes, not gigabytes, of memory.
