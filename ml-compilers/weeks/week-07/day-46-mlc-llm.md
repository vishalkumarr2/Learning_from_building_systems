# Day 46: MLC-LLM

> **Phase III · Week 7 · Day 46 of 70 · 2.5 hours**

*"The real test of a compiler isn't how fast it makes things on one device — it's whether it can make them fast on every device."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 45: ONNX Runtime Deep Dive](day-45-onnx-runtime-deep-dive.md) | [Day 47: Contributing to TVM](day-47-contributing-to-tvm.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Large language models are transforming software, but deploying them is a nightmare. A 7B-parameter model in FP16 needs 14 GB of memory — that won't fit on most phones, laptops, or edge devices. Even when it fits, naive inference is slow. **MLC-LLM** (Machine Learning Compilation for Large Language Models) solves this by applying TVM's compilation stack to LLMs: it takes a model like Llama, quantizes it, compiles it with Relax IR, auto-tunes kernels for the target device, and produces a self-contained deployment package. The result: Llama running on iPhones, Android phones, web browsers (via WebGPU), and every major GPU — all from a single compilation pipeline. MLC-LLM is the culmination of everything you've learned about TVM, demonstrating how IR design, scheduling, and auto-tuning combine for real-world impact.

---

## 1. The LLM Deployment Problem

### Why LLMs Are Hard to Deploy

```
LLM Deployment Challenges
══════════════════════════

  Model Size vs Device Memory:
  ┌────────────────────────────────────────────────┐
  │ Model        │ Params │ FP16    │ INT4 (Q4)    │
  ├────────────────────────────────────────────────┤
  │ Llama-3-8B   │  8.0B  │ 16 GB  │  ~4.5 GB     │
  │ Llama-3-70B  │ 70.0B  │ 140 GB │  ~38 GB      │
  │ Mixtral-8x7B │ 46.7B  │  93 GB │  ~26 GB      │
  │ Phi-3-mini   │  3.8B  │  7.6 GB│  ~2.2 GB     │
  └────────────────────────────────────────────────┘

  Target Devices:
  ┌────────────────────────────────────────────────┐
  │ Device          │ Memory  │ Compute           │
  ├────────────────────────────────────────────────┤
  │ NVIDIA A100     │  80 GB  │ CUDA, Tensor Cores│
  │ Apple M2 Max    │  32 GB  │ Metal, ANE        │
  │ iPhone 15 Pro   │   6 GB  │ Metal, ANE        │
  │ Pixel 8         │   8 GB  │ OpenCL/Vulkan     │
  │ Web Browser     │  ~4 GB  │ WebGPU            │
  │ AMD RX 7900     │  24 GB  │ Vulkan/ROCm       │
  └────────────────────────────────────────────────┘

  Key bottleneck: LLM inference is MEMORY-BANDWIDTH bound
  (decoding reads the entire model for each token generated)

  Arithmetic Intensity ≈ 1 FLOP / 2 bytes (for autoregressive decoding)
  → Must minimize memory footprint AND optimize memory access
```

### The MLC-LLM Solution

MLC-LLM combines three techniques:

$$\text{MLC-LLM} = \underbrace{\text{Quantization}}_{\text{shrink model}} + \underbrace{\text{Compilation}}_{\text{optimize kernels}} + \underbrace{\text{Universal Runtime}}_{\text{run anywhere}}$$

---

## 2. MLC-LLM Architecture

### End-to-End Pipeline

```
MLC-LLM Compilation Pipeline
═════════════════════════════

  Hugging Face Model (e.g., meta-llama/Llama-3-8B)
         │
    ┌────▼──────────────────┐
    │  1. Model Import       │  Weight loading + architecture
    │     (HF → Relax IR)   │  detection (Llama, GPT, Mistral...)
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  2. Quantization       │  Group quantization (q4f16_1, etc.)
    │     (weights → INT4)  │  Dequantize fused into matmul
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  3. Relax Optimization │  Graph-level transforms:
    │     • Attention fusion │  • KV cache management
    │     • Flash attention  │  • Rotary embedding
    │     • Operator fusion  │  • Prefill / decode split
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  4. TIR Scheduling     │  Low-level kernel optimization:
    │     • Tiling            │  • Memory coalescing
    │     • Vectorization     │  • Shared memory usage
    │     • Auto-tuning       │  • Target-specific intrinsics
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  5. Code Generation    │  Target-specific output:
    │     ├─ CUDA (.cu)      │
    │     ├─ Metal (.metal)  │
    │     ├─ Vulkan (.spv)   │
    │     ├─ OpenCL (.cl)    │
    │     └─ WebGPU (.wgsl)  │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  6. Packaging           │  Self-contained deployment:
    │     • Compiled library  │  • Model weights (quantized)
    │     • Tokenizer         │  • Chat config
    │     • Runtime           │
    └────────────────────────┘
```

### Key Architectural Decisions

```
Why Relax (not Relay) for LLMs
══════════════════════════════

  LLM Requirement          Relay           Relax
  ─────────────────        ──────          ──────
  Dynamic seq length       ✗ (Any hack)    ✓ (symbolic shapes)
  KV cache mutation        ✗ (pure FP)     ✓ (effects outside dataflow)
  Variable batch size      ✗ (recompile)   ✓ (symbolic "batch" dim)
  PagedAttention           ✗ (can't)       ✓ (explicit memory mgmt)
  Speculative decoding     ✗ (can't)       ✓ (control flow + effects)

  Example: KV cache in Relax

  @R.function
  def prefill(input_ids: R.Tensor(("batch", "seq_len"), "int32"),
              kv_cache: R.Object):
      # input_ids shape is symbolic — no recompilation needed
      # kv_cache is an opaque object that can be mutated
      ...
```

---

## 3. Quantization for Deployment

### Group Quantization

MLC-LLM uses **group quantization** to compress model weights while maintaining accuracy:

```
Group Quantization (q4f16_1 — 4-bit weights, FP16 compute)
══════════════════════════════════════════════════════════

  Original FP16 weight row (1024 elements):
  [0.23, -0.15, 0.87, -0.42, 0.11, ... ]  ← 2048 bytes

  Group into blocks of 32 (group_size=32):
  ┌──── Group 0 ────┐ ┌──── Group 1 ────┐
  │ 32 FP16 values   │ │ 32 FP16 values   │ ...
  └──────────────────┘ └──────────────────┘

  Per group: compute scale and zero_point
    scale = (max - min) / 15          (4-bit range: 0-15)
    zero_point = round(-min / scale)

  Quantize each value:
    q_val = round(val / scale) + zero_point   ← 4-bit integer
    q_val = clamp(q_val, 0, 15)

  Storage: 32 × 4 bits = 16 bytes + 2 bytes (scale) + 2 bytes (zp)
         = 20 bytes vs original 64 bytes ≈ 3.2× compression

  Dequantize (fused into matmul at compute time):
    val ≈ scale × (q_val - zero_point)
```

### Quantization Modes in MLC-LLM

| Mode | Weight Bits | Activation | Group Size | Compression |
|:--|:--|:--|:--|:--|
| `q0f16` | 16 (no quant) | FP16 | — | 1× |
| `q0f32` | 32 (no quant) | FP32 | — | 0.5× |
| `q4f16_1` | 4 | FP16 | 32 | ~3.5× |
| `q4f32_1` | 4 | FP32 | 32 | ~3.5× |
| `q4f16_0` | 4 | FP16 | 128 | ~3.8× |
| `q3f16_1` | 3 | FP16 | 32 | ~4.5× |
| `q8f16_1` | 8 | FP16 | 32 | ~1.9× |

### Fused Dequantize-MatMul

The key optimization: **dequantization is fused into the matrix multiplication kernel**, avoiding materialization of full-precision weights:

```python
# Pseudocode: fused dequant-matmul TIR kernel

@T.prim_func
def fused_dequant_matmul(
    x: T.Buffer((M, K), "float16"),         # activations
    w_q: T.Buffer((N, K // 8), "uint32"),    # packed 4-bit weights
    scales: T.Buffer((N, K // 32), "float16"), # per-group scales
    output: T.Buffer((M, N), "float16"),     # result
):
    for i, j in T.grid(M, N):
        acc = T.float16(0)
        for k in range(K):
            # Extract 4-bit weight from packed uint32
            group_idx = k // 32
            w_fp16 = dequantize(w_q[j, k // 8], k % 8, 
                                scales[j, group_idx])
            acc += x[i, k] * w_fp16
        output[i, j] = acc

# The compiler tiles, vectorizes, and uses shared memory
# to make this efficient on each target
```

---

## 4. Universal Deployment

### WebLLM: LLMs in the Browser

**WebLLM** runs compiled LLMs directly in the browser via WebGPU:

```
WebLLM Architecture
═══════════════════

  Browser (Chrome/Edge/Firefox with WebGPU)
  ┌────────────────────────────────────────┐
  │  JavaScript Application                │
  │  ┌──────────────────────────────────┐  │
  │  │  WebLLM Engine                    │  │
  │  │  • Tokenizer (in JS/WASM)        │  │
  │  │  • Chat template handler         │  │
  │  │  • Streaming response manager    │  │
  │  └──────────┬───────────────────────┘  │
  │             │                          │
  │  ┌──────────▼───────────────────────┐  │
  │  │  TVM Web Runtime                  │  │
  │  │  • WASM for control logic         │  │
  │  │  • WebGPU for compute kernels     │  │
  │  │  • Cached model weights (IndexDB) │  │
  │  └──────────────────────────────────┘  │
  └────────────────────────────────────────┘

  First load: downloads ~2-4 GB (quantized weights)
  Subsequent: loads from IndexedDB cache
  Inference: GPU-accelerated via WebGPU shaders
```

```javascript
// WebLLM usage in JavaScript
import * as webllm from "@mlc-ai/web-llm";

const engine = new webllm.MLCEngine();
await engine.reload("Llama-3-8B-Instruct-q4f16_1-MLC");

const reply = await engine.chat.completions.create({
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is machine learning compilation?" },
    ],
    temperature: 0.7,
    max_tokens: 256,
    stream: true,  // Streaming token generation
});

for await (const chunk of reply) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

### Mobile Deployment (iOS / Android)

```
MLC-LLM on Mobile
══════════════════

  iOS (Swift)                    Android (Kotlin)
  ┌────────────────┐             ┌────────────────┐
  │ MLCSwift       │             │ MLCAndroid      │
  │ Framework      │             │ Library         │
  │                │             │                 │
  │ Metal kernels  │             │ Vulkan/OpenCL   │
  │ (compiled .lib)│             │ kernels (.so)   │
  │                │             │                 │
  │ Quantized      │             │ Quantized       │
  │ weights (4-bit)│             │ weights (4-bit) │
  │ ~2-4 GB        │             │ ~2-4 GB         │
  └────────────────┘             └────────────────┘

  Performance (Llama-3-8B-q4f16_1):
  ┌────────────────────────────────────────┐
  │ Device           │ Prefill │ Decode   │
  ├────────────────────────────────────────┤
  │ iPhone 15 Pro    │ ~35 t/s │ ~15 t/s  │
  │ Pixel 8 Pro      │ ~25 t/s │ ~10 t/s  │
  │ M2 MacBook Air   │ ~80 t/s │ ~35 t/s  │
  │ RTX 4090         │ ~250 t/s│ ~90 t/s  │
  │ Chrome (WebGPU)  │ ~20 t/s │ ~8 t/s   │
  └────────────────────────────────────────┘
  (Approximate; varies by prompt length and device load)
```

---

## 5. Compilation Deep Dive

### Attention Kernel Optimization

MLC-LLM compiles optimized attention kernels for each backend. The TIR representation enables target-specific tuning:

```
FlashAttention via TVM Compilation
══════════════════════════════════

  Standard Attention:
    S = Q · Kᵀ / √d_k           ← O(n²d) memory for S
    P = softmax(S)               ← O(n²) memory for P
    O = P · V                    ← O(n²d) total

  Flash Attention (tiled, online softmax):
    for each Q tile (block_q):
      for each K,V tile (block_kv):
        S_tile = Q_tile · K_tileᵀ / √d_k
        m_new = max(m_old, rowmax(S_tile))
        P_tile = exp(S_tile - m_new)
        l_new = exp(m_old - m_new) * l_old + rowsum(P_tile)
        O = diag(exp(m_old - m_new)) * O_old + P_tile · V_tile
      O = diag(1/l_new) * O

  Memory: O(n) instead of O(n²)
  
  TVM advantage: the SAME algorithm compiles to:
  ┌─────────────────────────────────────────┐
  │ Target   │ Implementation              │
  ├─────────────────────────────────────────┤
  │ CUDA     │ Shared mem tiles, warp ops   │
  │ Metal    │ Threadgroup mem, SIMD groups │
  │ Vulkan   │ Workgroup shared memory      │
  │ WebGPU   │ Workgroup variables          │
  └─────────────────────────────────────────┘
```

### Prefill vs Decode Kernels

MLC-LLM generates different kernels for the two phases:

```
Prefill (process prompt):           Decode (generate tokens):
  • Batch matmul: [1, S, D] × [D, D]   • Single-token: [1, 1, D] × [D, D]
  • Compute-bound (high arithmetic     • Memory-bound (low arithmetic
    intensity, S can be large)            intensity, reads entire KV cache)
  • Use large tiles, maximize FLOPs     • Optimize for memory bandwidth
  • Parallelize across sequence         • Parallelize across batch

  ┌──────────────────────┐     ┌──────────────────────┐
  │  Prefill Kernel       │     │  Decode Kernel        │
  │  ┌─────────────────┐ │     │  ┌─────────────────┐ │
  │  │ Large tiles      │ │     │  │ Small tiles      │ │
  │  │ (128×128)        │ │     │  │ (1×128)          │ │
  │  │ Compute-heavy    │ │     │  │ Memory-access    │ │
  │  │ loop body        │ │     │  │ optimized        │ │
  │  └─────────────────┘ │     │  └─────────────────┘ │
  └──────────────────────┘     └──────────────────────┘
```

---

## 6. Hands-On Exercises

### Exercise 1: Compile a Model with MLC-LLM (40 min)

```bash
# Install MLC-LLM
pip install mlc-llm mlc-ai-nightly

# Option 1: Use pre-compiled model (quick start)
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

# Option 2: Compile from Hugging Face (full pipeline)
# Step 1: Convert model weights
mlc_llm convert_weight \
    --model-type llama \
    --quantization q4f16_1 \
    --source "meta-llama/Llama-3-8B-Instruct" \
    --output ./dist/Llama-3-8B-q4f16_1/

# Step 2: Generate model config
mlc_llm gen_config \
    --model-type llama \
    --quantization q4f16_1 \
    --source "meta-llama/Llama-3-8B-Instruct" \
    --output ./dist/Llama-3-8B-q4f16_1/

# Step 3: Compile model library for target
mlc_llm compile \
    --model ./dist/Llama-3-8B-q4f16_1/ \
    --quantization q4f16_1 \
    --target cuda \           # or: metal, vulkan, webgpu
    --output ./dist/libs/Llama-3-8B-q4f16_1-cuda.so
```

### Exercise 2: MLC-LLM Python API (20 min)

```python
from mlc_llm import MLCEngine

# Create engine with compiled model
engine = MLCEngine(
    model="./dist/Llama-3-8B-q4f16_1/",
    model_lib="./dist/libs/Llama-3-8B-q4f16_1-cuda.so",
)

# OpenAI-compatible API
response = engine.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a compiler expert."},
        {"role": "user", "content": "Explain operator fusion in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=128,
)
print(response.choices[0].message.content)

# Streaming
for chunk in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is TVM?"}],
    stream=True,
):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

# Benchmark
import time
prompt = "Explain the transformer architecture step by step."
t0 = time.perf_counter()
resp = engine.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
)
elapsed = time.perf_counter() - t0
n_tokens = resp.usage.completion_tokens
print(f"\n\nTokens: {n_tokens}, Time: {elapsed:.2f}s")
print(f"Throughput: {n_tokens/elapsed:.1f} tokens/sec")
```

### Exercise 3: Inspect the Compiled IR (20 min)

```python
# Examine what MLC-LLM produces at the Relax level
import tvm
from tvm import relax

# Load the compiled library
lib = tvm.runtime.load_module("./dist/libs/Llama-3-8B-q4f16_1-cuda.so")

# If you have the pre-compilation Relax IR:
# (saved during mlc_llm compile with --debug-dump)
# mod = tvm.ir.load_json(open("model_relax.json").read())

# Key things to look for:
# 1. Fused functions: prefill_fused_* and decode_fused_*
# 2. Quantized matmul: fused_dequantize_matmul
# 3. Attention: fused_attention_kv_cache_*
# 4. Symbolic shapes: R.Tensor(("batch", "seq_len", 4096), "float16")

# Compare model sizes
import os
original_size_gb = 8e9 * 2 / 1e9  # 8B params × 2 bytes (FP16)
quant_size_gb = os.path.getsize(
    "./dist/Llama-3-8B-q4f16_1/params_shard_0.bin"
) / 1e9
print(f"Original (FP16): ~{original_size_gb:.1f} GB")
print(f"Quantized (Q4):  ~{quant_size_gb:.1f} GB")
print(f"Compression:     ~{original_size_gb/quant_size_gb:.1f}×")
```

---

## Key Takeaways

1. **MLC-LLM** applies TVM's full compilation stack (Relax IR, TIR scheduling, auto-tuning) to make LLMs run on any hardware — from NVIDIA GPUs to web browsers
2. **Group quantization** (e.g., q4f16_1) compresses 16-bit weights to 4 bits with per-group scale factors, achieving ~3.5× compression with minimal accuracy loss
3. **Fused dequantize-matmul** is the critical kernel — dequantization happens inside the matmul loop, never materializing full-precision weights in memory
4. **Relax IR** (not Relay) is essential for LLMs because it supports symbolic shapes, mutable KV caches, and the prefill/decode split natively
5. **WebLLM** demonstrates universal deployment — the same compilation pipeline targets CUDA, Metal, Vulkan, and WebGPU, enabling LLMs to run entirely in-browser
6. **Prefill vs decode** require different kernel strategies: prefill is compute-bound (large matrix ops), decode is memory-bound (single-token with full KV cache reads)

---

## Further Reading

- [MLC-LLM Documentation](https://llm.mlc.ai/docs/) — official guide
- [WebLLM Project](https://webllm.mlc.ai/) — browser-based LLM demo
- [Machine Learning Compilation (MLC) Course](https://mlc.ai/) — full course by Tianqi Chen
- [MLC-LLM GitHub](https://github.com/mlc-ai/mlc-llm) — source code
- [Universal LLM Deployment (blog)](https://blog.mlc.ai/2023/05/01/bringing-hardware-accelerated-language-models-to-consumer-devices) — design rationale
- [WebGPU Standard](https://www.w3.org/TR/webgpu/) — the API enabling browser deployment

---

## Tomorrow: Contributing to TVM

Day 47 shifts from using TVM to **contributing to it**. You'll learn TVM's development workflow, how to navigate the codebase, write a simple pass, and submit a pull request. Open-source contribution is how you solidify your understanding and join the community.
