# Day 61: LLM Quantization — GPTQ, AWQ, GGUF

> **Phase IV · Week 9 · Day 61 of 70 · 2.5 hours**

*"The difference between a 140 GB model that needs a cluster and a 4 GB model that runs on a laptop is quantization. Same knowledge, different precision."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 60: Speculative Decoding](day-60-speculative-decoding.md) | [Day 62: Serving Frameworks Comparison](day-62-serving-frameworks-comparison.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

A 70B-parameter LLM in FP16 weighs ~140 GB — that's two A100-80GBs just for weights alone, before any KV cache. Quantizing to INT4 shrinks it to ~35 GB, fitting on a single GPU with room to spare. But LLMs aren't ordinary neural networks: they exhibit **activation outliers** — channels where values are 100× larger than average — that make naive quantization catastrophic. Specialized methods like GPTQ, AWQ, and GGUF solve this problem, each with different tradeoffs between accuracy, speed, and flexibility. Understanding these methods is essential because quantization is the single highest-impact deployment optimization: it simultaneously reduces memory, increases throughput, and lowers cost.

---

## 1. Why LLMs Break Naive Quantization

Standard per-tensor INT8 quantization works well for CNNs and small transformers. It fails for LLMs because of **outlier channels** discovered by Dettmers et al. (LLM.int8(), 2022):

```
Activation Outlier Problem in LLMs
═══════════════════════════════════════════════════════════════

  Normal activations (99.9% of channels):
  ┌─────────────────────────────────────────┐
  │ Values: [-0.5, 0.3, -0.1, 0.4, -0.2]   │  Range: ±1
  └─────────────────────────────────────────┘

  Outlier channels (0.1% of channels, but CRITICAL):
  ┌─────────────────────────────────────────┐
  │ Values: [-60.0, 80.0, -45.0, 120.0]     │  Range: ±120
  └─────────────────────────────────────────┘

  Naive per-tensor INT8 quantization:
  ┌─────────────────────────────────────────────────────────┐
  │  Scale = 120 / 127 ≈ 0.945                              │
  │                                                          │
  │  Normal:  0.3 / 0.945 = 0.32 → round to 0               │
  │  Normal: -0.5 / 0.945 = -0.53 → round to -1             │
  │  Outlier: 120.0 / 0.945 = 127 → maps correctly          │
  │                                                          │
  │  Problem: Normal values get quantized to {-1, 0, 1}!     │
  │  → 99.9% of activations lose almost all information      │
  └─────────────────────────────────────────────────────────┘

  Solution approaches:
  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐
  │  LLM.int8()    │  │    GPTQ      │  │      AWQ        │
  │  Mixed FP16/   │  │  Hessian-    │  │  Protect salient│
  │  INT8 by       │  │  guided      │  │  weights by     │
  │  channel       │  │  rounding    │  │  activation     │
  └────────────────┘  └──────────────┘  └─────────────────┘
```

The key insight: these outliers carry disproportionate information. Removing 6 outlier dimensions from a 5120-dim hidden state (0.12%) can drop accuracy by 20+ perplexity points.

---

## 2. GPTQ — Hessian-Based Weight Quantization

GPTQ (Frantar et al., 2023) quantizes weights **layer by layer**, using second-order information (the Hessian) to minimize the quantization error:

$$\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

where $X$ is a calibration dataset, $W$ is the original weight, and $\hat{W}$ is the quantized weight.

### The OBQ → GPTQ Pipeline

GPTQ builds on Optimal Brain Quantization (OBQ). For each weight column, it:

1. Quantizes one weight to the nearest grid point
2. Computes the error from that rounding
3. **Distributes the error** across remaining un-quantized weights using the inverse Hessian

$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q}$$

```python
import torch
import torch.nn as nn

def gptq_quantize_layer(
    weight: torch.Tensor,     # [out_features, in_features]
    hessian: torch.Tensor,    # [in_features, in_features]  = X @ X^T
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,    # columns processed together
):
    """Simplified GPTQ quantization for one linear layer.
    
    Args:
        weight: Original FP16 weight matrix
        hessian: H = 2 * X @ X^T from calibration data
        bits: Target bit width (typically 4)
        group_size: Number of columns sharing a scale factor
        block_size: Columns processed in one Cholesky block
    """
    W = weight.clone().float()
    n_rows, n_cols = W.shape
    
    # Quantization grid for N bits
    qmin, qmax = 0, 2**bits - 1  # asymmetric: [0, 15] for 4-bit
    
    # Cholesky decomposition of Hessian (with damping)
    damp = 0.01 * torch.diag(hessian).mean()
    H = hessian + damp * torch.eye(n_cols, device=W.device)
    H_inv = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(H_inv)
    
    quantized = torch.zeros_like(W)
    scales = torch.zeros(n_rows, n_cols // group_size, device=W.device)
    zeros = torch.zeros_like(scales)
    
    # Process columns in blocks
    for block_start in range(0, n_cols, block_size):
        block_end = min(block_start + block_size, n_cols)
        
        W_block = W[:, block_start:block_end].clone()
        H_block = H_inv[block_start:block_end, block_start:block_end]
        error = torch.zeros_like(W_block)
        
        for col in range(block_end - block_start):
            global_col = block_start + col
            
            # Compute per-group scale
            group_idx = global_col // group_size
            if global_col % group_size == 0:
                group_end = min(global_col + group_size, n_cols)
                w_group = W[:, global_col:group_end]
                s = (w_group.max(dim=1).values - w_group.min(dim=1).values) / qmax
                z = torch.round(-w_group.min(dim=1).values / s)
                scales[:, group_idx] = s
                zeros[:, group_idx] = z
            
            s = scales[:, group_idx]
            z = zeros[:, group_idx]
            
            # Quantize this column
            w_col = W_block[:, col]
            q_col = torch.clamp(torch.round(w_col / s + z), qmin, qmax)
            quantized[:, global_col] = q_col
            
            # Compute rounding error
            w_hat = (q_col - z) * s
            err = (w_col - w_hat) / H_block[col, col]
            
            # Distribute error to remaining columns in this block
            W_block[:, col+1:] -= err.unsqueeze(1) * H_block[col, col+1:].unsqueeze(0)
        
        # Distribute block error to remaining columns
        W[:, block_end:] -= error @ H_inv[block_start:block_end, block_end:]
    
    return quantized, scales, zeros
```

### GPTQ Performance

| Model | Bits | Perplexity (Wiki2) | Size | Speed (A100) |
|---|---|---|---|---|
| LLaMA-2 7B FP16 | 16 | 5.47 | 13.5 GB | 1.0× |
| LLaMA-2 7B GPTQ | 4 | 5.63 (+0.16) | 3.9 GB | 2.8× |
| LLaMA-2 7B GPTQ | 3 | 6.29 (+0.82) | 3.0 GB | 3.2× |
| LLaMA-2 70B FP16 | 16 | 3.32 | 137 GB | 1.0× |
| LLaMA-2 70B GPTQ | 4 | 3.48 (+0.16) | 37 GB | 3.5× |

---

## 3. AWQ — Activation-Aware Weight Quantization

AWQ (Lin et al., 2024) observes that **not all weights are equally important**. Weights connected to large-magnitude activation channels carry more information and should be quantized more carefully:

$$\text{Quantization error} \propto |w \cdot s_x|$$

where $s_x$ is the average activation magnitude for that channel.

```
AWQ Core Insight
═══════════════════════════════════════════════════════════════

  Weight matrix W: [out_features × in_features]
  Activation magnitudes per channel: s = mean(|X|, dim=batch)

  Channel importance = |weight| × |activation|

  ┌──────────────────────────────────────────────────────────┐
  │  Channel 42:   w = 0.02,  s_x = 0.3   → impact: 0.006  │  Low
  │  Channel 107:  w = 0.05,  s_x = 80.0  → impact: 4.0    │  HIGH!
  │  Channel 512:  w = 3.00,  s_x = 0.1   → impact: 0.3    │  Medium
  └──────────────────────────────────────────────────────────┘

  AWQ strategy: Scale up salient channels BEFORE quantizing
  ─────────────────────────────────────────────────────────
  For salient channel i:
    w_i' = w_i × α       (scale weight UP → more quantization levels)
    x_i' = x_i / α       (scale activation DOWN → preserves W·X)

  Net effect: W' · X' = W · X  (mathematically equivalent)
  But salient weights now use more of the INT4 range!
```

The scale factor $\alpha$ per channel is found by grid search:

$$\alpha^* = \arg\min_\alpha \|Q(W \cdot \text{diag}(\alpha)) \cdot \text{diag}(\alpha)^{-1} \cdot X - W \cdot X\|$$

```python
def awq_compute_scales(
    weight: torch.Tensor,    # [out, in]
    activations: torch.Tensor,  # [n_samples, in]
    bits: int = 4,
    grid_size: int = 20,
):
    """Find optimal per-channel scaling factors for AWQ.
    
    Key insight: scale up important channels before quantizing,
    scale down activations to compensate → same output, better precision.
    """
    n_samples = activations.shape[0]
    in_features = weight.shape[1]
    
    # Channel importance = average activation magnitude
    act_scales = activations.abs().mean(dim=0)  # [in_features]
    
    # Original output for reference
    original_out = weight @ activations.T  # [out, n_samples]
    
    best_scales = torch.ones(in_features, device=weight.device)
    best_error = float('inf')
    
    # Grid search over scale exponents
    for ratio in torch.linspace(0, 1, grid_size):
        # Scale = act_scales^ratio (only scales important channels)
        scales = act_scales.pow(ratio).clamp(min=1e-4)
        
        # Scale weights up, quantize, scale back
        w_scaled = weight * scales.unsqueeze(0)
        w_quant = pseudo_quantize(w_scaled, bits=bits)
        w_dequant = w_quant / scales.unsqueeze(0)
        
        # Measure error on calibration data
        quant_out = w_dequant @ activations.T
        error = (original_out - quant_out).pow(2).mean()
        
        if error < best_error:
            best_error = error
            best_scales = scales.clone()
    
    return best_scales

def pseudo_quantize(w: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Simulate quantization (quantize then dequantize)."""
    qmax = 2**bits - 1
    # Per-group quantization
    orig_shape = w.shape
    w = w.reshape(-1, group_size)
    w_min = w.min(dim=1, keepdim=True).values
    w_max = w.max(dim=1, keepdim=True).values
    scale = (w_max - w_min) / qmax
    w_q = torch.round((w - w_min) / scale).clamp(0, qmax)
    w_deq = w_q * scale + w_min
    return w_deq.reshape(orig_shape)
```

### AWQ vs GPTQ

| Aspect | GPTQ | AWQ |
|---|---|---|
| Approach | Hessian-guided rounding + error compensation | Activation-aware channel scaling |
| Calibration | ~128 samples, ~5 min for 7B | ~128 samples, ~2 min for 7B |
| Accuracy (4-bit) | Very good | Slightly better on most benchmarks |
| Speed (inference) | Fast (optimized kernels) | Fast (AutoAWQ kernels) |
| Robustness | Can be sensitive to calibration data | More robust across domains |
| Key advantage | Error redistribution across weights | Protects salient channels |

---

## 4. GGUF Format and llama.cpp Quantization

GGUF (GPT-Generated Unified Format) is the native format for llama.cpp, designed for **CPU and edge inference**. It supports a rich menu of quantization types:

```
GGUF Quantization Types — The Precision Ladder
═══════════════════════════════════════════════════════════════

  Type      Bits   Size/7B    Perplexity   Use Case
  ─────────────────────────────────────────────────────────
  F16       16.0   13.5 GB    5.47         Reference baseline
  Q8_0       8.5    7.2 GB    5.48         Near-lossless
  Q6_K       6.6    5.5 GB    5.50         Best quality below 8-bit
  Q5_K_M     5.7    4.8 GB    5.52         ★ Best accuracy/size ratio
  Q5_K_S     5.5    4.6 GB    5.54         Slightly smaller
  Q4_K_M     4.8    4.1 GB    5.63         ★ Most popular choice
  Q4_K_S     4.6    3.9 GB    5.68         Good for smaller RAM
  Q4_0       4.5    3.8 GB    5.72         Basic 4-bit (legacy)
  Q3_K_M     3.9    3.3 GB    6.15         Noticeable degradation
  Q3_K_S     3.5    3.0 GB    6.42         Aggressive compression
  Q2_K       3.3    2.8 GB    8.50         Research only
  IQ4_XS     4.3    3.7 GB    5.60         ★ imatrix, best <4GB
  IQ3_XXS    3.1    2.6 GB    6.80         imatrix, extreme
  IQ2_XXS    2.1    1.8 GB   12.00         Pushing limits
  ─────────────────────────────────────────────────────────

  K-quants: "K" = k-means clustering for optimal quantization levels
  S/M/L:    Small/Medium/Large — more bits for attention layers
  IQ:       Importance-matrix guided quantization (imatrix)

  ┌────────────────────────────────────────────────────────┐
  │  Rule of thumb:                                        │
  │  • Plenty of RAM → Q5_K_M (best quality per GB)        │
  │  • Tight RAM     → Q4_K_M (sweet spot)                 │
  │  • Extreme edge  → IQ4_XS with imatrix                 │
  └────────────────────────────────────────────────────────┘
```

### K-Quant Architecture

K-quants don't use uniform bit width — they allocate more bits to sensitive layers:

```
K-Quant Mixed Precision Strategy
═══════════════════════════════════════════════════════════════

  Q4_K_M allocation for a 32-layer transformer:

  Layer Type          Bits   Why
  ──────────────────────────────────────────────────
  Embedding           6      First/last layers are sensitive
  Attention Q,K       4      Can tolerate some noise
  Attention V         5      Values carry content information
  Attention Output    5      Aggregation layer, moderate
  FFN Gate/Up         4      Largest matrices, most savings
  FFN Down            5      Output projection, moderate
  Final LayerNorm     32     Tiny, keep full precision
  LM Head             6      Classification layer, sensitive
  ──────────────────────────────────────────────────

  Result: "4-bit" model actually uses 4.8 bits average
  → Q4_K_M vs Q4_0: same name, 0.1 better perplexity
```

### Quantizing with llama.cpp

```bash
# Step 1: Convert HuggingFace model to GGUF
python convert_hf_to_gguf.py \
    /models/llama-2-7b-hf \
    --outfile llama-2-7b-f16.gguf \
    --outtype f16

# Step 2: Quantize (basic)
./llama-quantize llama-2-7b-f16.gguf llama-2-7b-Q4_K_M.gguf Q4_K_M

# Step 3: Quantize with importance matrix (better quality!)
# First, generate importance matrix from calibration data
./llama-imatrix \
    -m llama-2-7b-f16.gguf \
    -f wiki.train.raw \
    -o imatrix.dat \
    --chunks 200

# Then quantize using importance data
./llama-quantize \
    --imatrix imatrix.dat \
    llama-2-7b-f16.gguf \
    llama-2-7b-IQ4_XS.gguf \
    IQ4_XS

# Step 4: Test perplexity
./llama-perplexity \
    -m llama-2-7b-Q4_K_M.gguf \
    -f wiki.test.raw
```

---

## 5. SqueezeLLM and QuIP# — Beyond Standard Quantization

### SqueezeLLM — Non-Uniform Quantization

SqueezeLLM (Kim et al., 2024) observes that weight distributions are non-uniform and uses **k-means clustering** to find optimal quantization centroids instead of uniform grid points:

$$\min_{\{c_1, \ldots, c_{2^b}\}} \sum_i s_i \cdot (w_i - c_{k_i})^2$$

where $s_i$ is the sensitivity (Hessian diagonal) and $c_{k_i}$ is the nearest centroid.

```
Uniform vs Non-Uniform Quantization
═══════════════════════════════════════════════════════════════

  Weight distribution (typical LLM layer):
                    ▄▄
                   ▄██▄
                  ▄████▄
                 ▄██████▄
                ▄████████▄
           ▄▄▄▄██████████▄▄▄▄
  ─────────────────────────────────────── value
         -0.1        0        +0.1

  Uniform 4-bit (16 levels):
  ──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──
    Evenly spaced → wastes levels on sparse tails

  Non-uniform (k-means, 16 centroids):
  ─────────┼┼┼┼┼┼┼┼┼┼┼┼──────┼──────┼──
           More levels where data is dense
           → 0.1-0.3 perplexity improvement
```

### QuIP# — Incoherence Processing

QuIP# (Tseng et al., 2024) randomizes the weight matrix with orthogonal transformations to spread outliers, enabling near-optimal 2-bit quantization:

$$W_{\text{quantized}} = Q_{\text{lattice}}(U^T W V)$$

where $U, V$ are random orthogonal (Hadamard) matrices and $Q_{\text{lattice}}$ is E8 lattice quantization.

| Method | 2-bit PPL | 3-bit PPL | 4-bit PPL | Speed |
|---|---|---|---|---|
| GPTQ | 107.3 | 6.29 | 5.63 | Fast |
| AWQ | — | 6.15 | 5.60 | Fast |
| SqueezeLLM | — | 5.95 | 5.54 | Moderate |
| QuIP# | 8.33 | 5.72 | 5.49 | Slow |

---

## 6. Benchmarking: Accuracy vs Speed vs Memory

```python
"""Quantization benchmark framework for LLMs."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def measure_perplexity(model, tokenizer, dataset_name="wikitext", split="test"):
    """Measure perplexity on standard benchmark."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin:end].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100  # mask context
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.item())
    
    ppl = np.exp(np.mean(nlls))
    return ppl

def measure_throughput(model, tokenizer, prompt="Hello", n_tokens=128, n_runs=5):
    """Measure tokens/second generation speed."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=10, do_sample=False)
    
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    tokens_per_sec = n_tokens / avg_time
    return tokens_per_sec, avg_time

def benchmark_report(results: dict):
    """Print comparison table."""
    print(f"{'Method':<15} {'PPL':>8} {'Tok/s':>8} {'Size GB':>8} {'VRAM GB':>8}")
    print("─" * 50)
    for name, r in results.items():
        print(f"{name:<15} {r['ppl']:>8.2f} {r['tps']:>8.1f} "
              f"{r['size_gb']:>8.1f} {r['vram_gb']:>8.1f}")
```

---

## Hands-On Exercises

### Exercise 1: GPTQ Quantization with AutoGPTQ

```python
"""Quantize a model with GPTQ and measure quality."""
# pip install auto-gptq transformers

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,       # Use activation order (slower but better)
    damp_percent=0.01,   # Hessian damping
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
model.quantize(
    examples=[tokenizer("Example calibration text", return_tensors="pt")],
    batch_size=1,
)
model.save_quantized("llama-7b-gptq-4bit")
```

### Exercise 2: AWQ Quantization

```python
"""Quantize with AWQ and compare to GPTQ."""
# pip install autoawq

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"

model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",   # GEMM or GEMV kernel
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama-7b-awq-4bit")
```

### Exercise Tasks

1. **Head-to-head comparison**: Quantize the same model with GPTQ and AWQ at 4-bit. Compare perplexity on WikiText-2, MMLU accuracy, and generation speed. Which wins for which metric?
2. **GGUF exploration**: Convert a model to GGUF, create Q4_K_M and Q5_K_M variants. Measure perplexity and tokens/second on CPU with llama.cpp. When is Q5_K_M worth the extra memory?
3. **Calibration sensitivity**: Run GPTQ with 32, 128, and 512 calibration samples. Does more data always help? Try calibration from different domains (code vs Wikipedia vs chat).

---

## Key Takeaways

1. **LLMs have activation outliers** — 0.1% of channels carry 100× larger values, making naive quantization catastrophic
2. **GPTQ uses Hessian-guided error redistribution** — quantization error at one weight is compensated by adjusting remaining weights
3. **AWQ protects salient channels by scaling** — mathematically equivalent transformation that gives important weights more quantization levels
4. **GGUF K-quants allocate bits non-uniformly** — attention value projections and embeddings get more bits than FFN gate matrices
5. **4-bit quantization typically costs 0.1-0.3 perplexity** on a 7B model — a 3.5× size reduction for minimal quality loss
6. **QuIP# enables viable 2-bit quantization** through incoherence processing, but at significant speed cost

---

## Further Reading

- **Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers"** (ICLR 2023)
- **Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression"** (MLSys 2024)
- **Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** (NeurIPS 2022)
- **Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization"** (ICML 2024)
- **Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence"** (2024)
- **llama.cpp GGUF documentation**: github.com/ggerganov/llama.cpp

---

## Tomorrow's Preview

You now know how to make models smaller. But which framework should you actually deploy them with? **Day 62: Serving Frameworks Comparison** puts vLLM, TensorRT-LLM, TGI, SGLang, llama.cpp, and Triton side-by-side — architecture, performance, GPU strategies, and when each one wins.
