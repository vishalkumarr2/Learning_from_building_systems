# Day 37: Quantization & Inference

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"The best model is the one you can actually run." — Practical ML wisdom*

## Navigation
- **Previous:** [Day 36: LLM Evaluation](day-36-llm-evaluation.md)
- **Next:** [Day 38: In-Context Learning](day-38-in-context-learning.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Engineering](../../study-notes/07-llm-engineering.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 37.1 The Inference Cost Problem

LLM inference is bottlenecked by **memory bandwidth**, not compute:

```
Autoregressive generation: 1 token at a time
Each token requires loading ALL model weights from GPU memory

7B model (FP16): 14 GB of weights
GPU memory bandwidth: ~2 TB/s (A100)
→ Max throughput: 2000 / 14 ≈ 143 forward passes/sec
→ ~143 tokens/sec (memory-bound, not compute-bound)

Solution: Make the model smaller → fewer bytes to load → faster
```

### 37.2 Number Formats

```
Format     Bits   Range                    Example
──────     ────   ─────                    ───────
FP32       32     ±3.4 × 10³⁸             Standard training
FP16       16     ±65,504                  Mixed-precision training
BF16       16     ±3.4 × 10³⁸             Same range as FP32, less precision
INT8       8      [-128, 127]              Post-training quantization
INT4       4      [-8, 7]                  Aggressive compression
NF4        4      Normal-float 4-bit       QLoRA's innovation

Memory per parameter:
  FP32:  4 bytes
  FP16:  2 bytes  → 2× compression
  INT8:  1 byte   → 4× compression
  INT4:  0.5 byte → 8× compression
```

**BFloat16 vs Float16:**

```
FP16:   1 sign | 5 exponent  | 10 mantissa  → high precision, limited range
BF16:   1 sign | 8 exponent  | 7 mantissa   → lower precision, FP32 range

BF16 is preferred for training because gradient values can be very large,
and the wider exponent range prevents overflow.
```

### 37.3 Post-Training Quantization (PTQ)

Quantize a pretrained model without retraining:

**Absmax quantization (symmetric):**

$$
x_q = \text{round}\left(\frac{x}{\alpha} \cdot (2^{b-1} - 1)\right), \quad \alpha = \max(|x|)
$$

**Zero-point quantization (asymmetric):**

$$
x_q = \text{round}\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^b - 1)\right)
$$

**Per-channel vs per-tensor:** Quantizing each output channel separately preserves more information than a single scale for the whole tensor.

### 37.4 Advanced Quantization: GPTQ & AWQ

**GPTQ (GPT Quantization):**
- Uses second-order information (Hessian) to minimize quantization error
- Quantizes weights column by column, adjusting remaining columns to compensate
- One-shot: only needs a small calibration dataset (~128 samples)

$$
\min_{\hat{W}} \| W X - \hat{W} X \|_2^2
$$

**AWQ (Activation-Aware Weight Quantization):**
- Key insight: 1% of weights are disproportionately important (they correspond to large activations)
- Scale these important weights up before quantization, scale activations down
- Better quality than GPTQ at the same bit width

```
GPTQ: Minimize reconstruction error directly
AWQ:  Protect important weights → better overall quality

Both → 4-bit quantization with <1% quality loss on most benchmarks
```

**GGUF (GPT-Generated Unified Format):**
- CPU-friendly format used by llama.cpp
- Supports mixed quantization (different bits per layer)
- Enables running 7B models on laptops without GPU

### 37.5 Speculative Decoding

Generate tokens faster by using a small "draft" model:

```
Standard decoding (slow):
  Big model generates tokens one at a time
  Each token: load 14 GB of weights → 1 token

Speculative decoding (fast):
  1. Draft model (small, fast) generates K tokens quickly
  2. Big model verifies all K tokens in ONE forward pass
  3. Accept correct tokens, reject wrong ones, revert to that point
  
  If draft model is good: accept most tokens → K× speedup
  If draft model is bad: reject many → fall back to standard speed
  Guarantee: output distribution is identical to big model alone!
```

**Medusa** takes this further — adds multiple prediction heads to the model itself, generating multiple candidate continuations in parallel.

### 37.6 Inference Engines: vLLM & Friends

**vLLM** introduces **PagedAttention** — managing KV cache like virtual memory:

```
Standard KV cache:
  Each request gets contiguous GPU memory for max_length
  → 50%+ memory wasted on incomplete sequences

PagedAttention:
  KV cache split into fixed-size "pages"
  Pages allocated on demand, can be non-contiguous
  → Near-zero waste, 2-4× more concurrent requests
```

| Engine | Key Feature | Best For |
|--------|------------|----------|
| vLLM | PagedAttention, continuous batching | High-throughput serving |
| TGI | HuggingFace integration | Easy deployment |
| llama.cpp | CPU inference, GGUF format | Edge/laptop deployment |
| TensorRT-LLM | NVIDIA optimization | Maximum GPU performance |
| Ollama | Simple local deployment | Developer convenience |

---

## Implementation (60 min)

### Quantize a 7B Model and Benchmark

```python
"""
Day 37 Implementation: Quantization comparison.
Quantize a model to different precisions and benchmark.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def measure_model(model, tokenizer, label: str, prompt: str) -> dict:
    """Measure memory, latency, and output quality."""
    # Memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_gb = torch.cuda.memory_allocated() / 1e9
    else:
        mem_gb = sum(p.numel() * p.element_size()
                     for p in model.parameters()) / 1e9

    # Latency (generate 50 tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tokens = 50

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=n_tokens, do_sample=False,
        )
    elapsed = time.perf_counter() - start
    tokens_per_sec = n_tokens / elapsed

    # Output text
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "label": label,
        "memory_gb": mem_gb,
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": elapsed * 1000,
        "output_preview": text[:200],
    }


def load_fp16():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )

def load_int8():
    config = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=config, device_map="auto",
    )

def load_int4():
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=config, device_map="auto",
    )


# --- Manual quantization demo ---
def demonstrate_quantization():
    """Show quantization math on a small tensor."""
    # Original FP32 weights
    W = torch.randn(4, 4) * 0.5
    print("Original (FP32):")
    print(W)

    # INT8 absmax quantization
    scale = W.abs().max() / 127
    W_int8 = torch.round(W / scale).clamp(-128, 127).to(torch.int8)
    W_dequant_8 = W_int8.float() * scale

    # INT4 absmax quantization (simulated)
    scale_4 = W.abs().max() / 7
    W_int4 = torch.round(W / scale_4).clamp(-8, 7)
    W_dequant_4 = W_int4 * scale_4

    # Quantization error
    err_8 = (W - W_dequant_8).abs().mean().item()
    err_4 = (W - W_dequant_4).abs().mean().item()
    print(f"\nINT8 mean abs error: {err_8:.6f}")
    print(f"INT4 mean abs error: {err_4:.6f}")
    print(f"INT4/INT8 error ratio: {err_4/err_8:.1f}×")


if __name__ == "__main__":
    print("=" * 60)
    print("Quantization Math Demo")
    print("=" * 60)
    demonstrate_quantization()

    if torch.cuda.is_available():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt = "Explain how a warehouse robot navigates safely."

        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)

        configs = [
            ("FP16", load_fp16),
            ("INT8", load_int8),
            ("NF4", load_int4),
        ]

        results = []
        for label, loader in configs:
            torch.cuda.empty_cache()
            model = loader()
            r = measure_model(model, tokenizer, label, prompt)
            results.append(r)
            del model

        print(f"\n{'Method':<8} {'Memory':>8} {'Tok/s':>8} {'Latency':>10}")
        print("-" * 40)
        for r in results:
            print(f"{r['label']:<8} {r['memory_gb']:>7.2f}G "
                  f"{r['tokens_per_sec']:>7.1f} {r['latency_ms']:>9.0f}ms")
```

---

## Exercise (45 min)

### E37.1 — Per-Channel vs Per-Tensor Quantization (20 min)
1. Create a weight matrix with varying magnitudes across rows
2. Quantize with per-tensor scale vs per-row (per-channel) scale
3. Compare reconstruction error — when does per-channel matter most?

### E37.2 — Speculative Decoding Simulator (25 min)
1. Simulate speculative decoding with a "draft model" acceptance rate $p$
2. For $K \in \{2, 4, 8, 16\}$ draft tokens and $p \in \{0.5, 0.7, 0.9\}$:
   - Calculate expected speedup = $K \cdot p^K / (1 + K \cdot (1-p))$ (approximate)
3. Plot: at what acceptance rate does speculative decoding stop helping?

---

## Key Takeaways

1. **LLM inference is memory-bandwidth-bound** — smaller models = faster
2. **4-bit quantization** (GPTQ, AWQ, NF4) compresses 7B models to ~3.5 GB with <1% quality loss
3. **GGUF + llama.cpp** enables running models on CPU/laptops
4. **Speculative decoding** gets big-model quality at near-small-model speed
5. **vLLM's PagedAttention** is the standard for production serving — 2-4× throughput improvement

---

## Connection to the Thread

Edge deployment is everything in robotics. A warehouse robot can't call a cloud API for every decision — latency kills. Quantization (INT4/INT8) + efficient inference engines are how you run a VLA model on robot hardware (Jetson, TPU Edge). Understanding quantization math is prerequisite for deploying any model on a real robot.

---

## Further Reading

- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) — Frantar et al., 2023
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., 2023
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023
- [Speculative Decoding: Fast Inference from Transformers](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2023
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
