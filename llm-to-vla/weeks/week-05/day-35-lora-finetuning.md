# Day 35: LoRA & Efficient Fine-Tuning

> **Phase III — LLMs: Training & Alignment | Week 5 | 2.5 hours**
> *"Why update 7 billion parameters when 4 million will do?" — Edward Hu*

## Navigation
- **Previous:** [Day 34: DPO & Modern Alignment](day-34-dpo-alignment.md)
- **Next:** [Day 36: LLM Evaluation](../week-06/day-36-llm-evaluation.md)
- **Week:** [Week 5 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 35.1 The Fine-Tuning Cost Problem

Full fine-tuning updates every parameter in the model:

```
Model Size    Parameters     FP16 Memory    Optimizer Memory    Total VRAM
──────────    ──────────     ───────────    ────────────────    ──────────
7B            7 × 10⁹       14 GB          42 GB (AdamW)       ~56 GB
13B           13 × 10⁹      26 GB          78 GB               ~104 GB
70B           70 × 10⁹      140 GB         420 GB              ~560 GB

Problem: Most researchers have 1-2 GPUs (24-80 GB)
→ Full fine-tuning of 7B+ models is impractical
```

### 35.2 LoRA: Low-Rank Adaptation

**Key insight:** The weight updates during fine-tuning have low intrinsic rank. Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, decompose the update:

$$
W' = W_0 + \Delta W = W_0 + BA
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

```
Full fine-tuning:                    LoRA:
┌────────────────────┐              ┌────────────────────┐
│    W  (d × k)      │              │   W₀ (d × k)       │ ← frozen!
│    trainable        │              │   frozen            │
│    d × k params     │              │                     │
└────────────────────┘              │   + B(d×r) · A(r×k) │ ← trainable
                                    │     (r << d, k)      │
                                    └────────────────────┘
                                    
d=4096, k=4096:                     d=4096, k=4096, r=16:
Full: 16.8M params                  LoRA: 2 × 4096 × 16 = 131K params
                                    → 128× fewer parameters!
```

**Forward pass with LoRA:**

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

The $\frac{\alpha}{r}$ scaling factor controls the magnitude of the LoRA update. Typically $\alpha = 2r$.

### 35.3 Where to Apply LoRA

```
Transformer Block:
┌─────────────────────────────────────────┐
│  Multi-Head Attention                   │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌─────┐ │
│  │ W_q   │ │ W_k   │ │ W_v   │ │ W_o │ │
│  │ ✅LoRA│ │ ✅LoRA│ │ ✅LoRA│ │✅   │ │
│  └───────┘ └───────┘ └───────┘ └─────┘ │
│                                         │
│  Feed-Forward Network                   │
│  ┌──────────┐ ┌──────────┐              │
│  │ W_gate   │ │ W_up     │              │
│  │ ⚠️maybe  │ │ ⚠️maybe  │              │
│  └──────────┘ └──────────┘              │
│  ┌──────────┐                           │
│  │ W_down   │                           │
│  │ ⚠️maybe  │                           │
│  └──────────┘                           │
└─────────────────────────────────────────┘

Common strategy: Apply LoRA to Q, K, V, O projections
Aggressive strategy: Also apply to FFN weights
```

### 35.4 QLoRA: Quantized LoRA

**QLoRA** combines two ideas:
1. **Quantize** the base model to 4-bit (NF4 format) — saves memory
2. **Add LoRA adapters** in full precision — maintains quality

$$
h = \underbrace{W_0^{\text{NF4}}}_{\text{4-bit frozen}} x + \underbrace{\frac{\alpha}{r} B A}_{\text{16-bit trainable}} x
$$

```
Memory comparison for 7B model:
                    Base Model    LoRA Params    Optimizer    Total
Full FT (FP16):     14 GB         —              42 GB       ~56 GB
LoRA (FP16):        14 GB         ~8 MB          ~24 MB      ~14.1 GB
QLoRA (NF4):        3.5 GB        ~8 MB          ~24 MB      ~3.6 GB

QLoRA: Fine-tune a 7B model on a single 4GB GPU!
```

**NF4 (NormalFloat 4-bit):** Quantization format optimized for normally-distributed weights (which neural network weights are). Better than uniform INT4.

### 35.5 Other PEFT Methods

```
Method               Trainable Params    Where           How
──────               ────────────────    ─────           ───
Full fine-tuning     100%                All weights     Standard backprop
LoRA                 0.1-1%              Attention       Low-rank matrices
QLoRA                0.1-1%              Attention       LoRA + 4-bit base
Adapters             1-5%                After layers    Small bottleneck MLPs
Prefix Tuning        <0.1%               Input prefix    Learnable prefix tokens
Prompt Tuning        <0.01%              Input prefix    Soft prompt embeddings
IA3                  <0.01%              Activations     Learned scaling vectors
```

**Prefix Tuning** prepends learnable "virtual tokens" to the key and value in each attention layer:

$$
\text{Attention}(Q, [P_K; K], [P_V; V])
$$

where $P_K, P_V \in \mathbb{R}^{l \times d}$ are the learnable prefix matrices with $l$ virtual tokens.

**Prompt Tuning** is even simpler — only prepend to the input embeddings:

$$
\tilde{X} = [P; X] \quad \text{where } P \in \mathbb{R}^{l \times d}
$$

---

## Implementation (60 min)

### Compare Full Fine-Tune vs LoRA vs QLoRA

```python
"""
Day 35 Implementation: Compare full, LoRA, and QLoRA fine-tuning.
Uses TinyLlama for feasibility on consumer hardware.
"""
import torch
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Shared dataset ---
def get_dataset() -> Dataset:
    examples = [
        {"text": f"<|im_start|>user\nTask {i}<|im_end|>\n"
                 f"<|im_start|>assistant\nResponse {i}<|im_end|>"}
        for i in range(100)
    ]
    return Dataset.from_list(examples)


# ============================================================
# Method 1: Full Fine-Tuning
# ============================================================
def setup_full_finetune():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "method": "Full Fine-Tune",
        "trainable_params": trainable,
        "total_params": total,
        "pct_trainable": 100.0,
        "gpu_memory_gb": mem,
    }


# ============================================================
# Method 2: LoRA (FP16 base)
# ============================================================
def setup_lora():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "method": "LoRA (r=16, FP16)",
        "trainable_params": trainable,
        "total_params": total,
        "pct_trainable": pct,
        "gpu_memory_gb": mem,
    }


# ============================================================
# Method 3: QLoRA (4-bit base + LoRA)
# ============================================================
def setup_qlora():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "method": "QLoRA (r=16, NF4)",
        "trainable_params": trainable,
        "total_params": total,
        "pct_trainable": pct,
        "gpu_memory_gb": mem,
    }


# ============================================================
# LoRA Math Verification
# ============================================================
def verify_lora_math():
    """Verify LoRA decomposition W' = W0 + BA."""
    d, k, r = 512, 512, 8
    W0 = torch.randn(d, k)        # frozen base weight
    B = torch.randn(d, r) * 0.01  # LoRA down-projection
    A = torch.randn(r, k) * 0.01  # LoRA up-projection
    alpha = 16
    scaling = alpha / r

    x = torch.randn(1, k)

    # Standard computation
    h_full = x @ (W0 + scaling * B @ A).T

    # Decomposed (what happens at inference)
    h_base = x @ W0.T
    h_lora = scaling * (x @ A.T @ B.T)
    h_decomposed = h_base + h_lora

    diff = (h_full - h_decomposed).abs().max().item()
    print(f"LoRA decomposition error: {diff:.2e}")
    assert diff < 1e-5, "LoRA math verification failed!"

    # Parameter savings
    full_params = d * k
    lora_params = d * r + r * k
    savings = 100 * (1 - lora_params / full_params)
    print(f"Full params: {full_params:,}")
    print(f"LoRA params: {lora_params:,} ({savings:.1f}% reduction)")


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Math Verification")
    print("=" * 60)
    verify_lora_math()

    print("\n" + "=" * 60)
    print("Method Comparison (TinyLlama 1.1B)")
    print("=" * 60)

    # Run comparison (only if GPU available)
    if torch.cuda.is_available():
        for setup_fn in [setup_full_finetune, setup_lora, setup_qlora]:
            torch.cuda.empty_cache()
            info = setup_fn()
            print(f"\n{info['method']}:")
            print(f"  Trainable: {info['trainable_params']:>12,} "
                  f"({info['pct_trainable']:.2f}%)")
            print(f"  Total:     {info['total_params']:>12,}")
            print(f"  GPU Mem:   {info['gpu_memory_gb']:.2f} GB")
    else:
        print("No GPU — run LoRA math verification only.")
```

---

## Exercise (45 min)

### E35.1 — Rank Ablation (25 min)
Investigate the effect of LoRA rank $r$ on performance:
1. Train LoRA with $r \in \{2, 4, 8, 16, 32, 64\}$ on the same task
2. For each, record: trainable params, training loss, inference quality
3. Plot the Pareto frontier: quality vs. parameter count
4. What rank gives the best quality/cost trade-off?

### E35.2 — Target Module Comparison (20 min)
Compare applying LoRA to different weight matrices:
1. Q+V only (original paper recommendation)
2. Q+K+V+O (all attention)
3. All attention + FFN
4. Which configuration gives the best quality per trainable parameter?

---

## Key Takeaways

1. **LoRA: $W' = W_0 + BA$** — factorize updates into low-rank matrices, train $<1\%$ of parameters
2. **QLoRA** adds 4-bit quantization of the base → fine-tune 7B on a single consumer GPU
3. **Rank $r$** controls the expressiveness/efficiency trade-off — $r=16$ is a sweet spot
4. **$\alpha/r$ scaling** ensures stable training regardless of rank choice
5. **LoRA adapters are composable** — swap different task adapters at inference time without reloading the base model

---

## Connection to the Thread

LoRA's low-rank assumption directly connects to the manifold hypothesis from Phase I: neural network weight updates lie on a low-dimensional manifold within the high-dimensional parameter space. For robotics VLAs, LoRA enables **task-specific adaptation** — a single base model with different LoRA adapters for different warehouse layouts, robot morphologies, or task types. This is the practical bridge between "one foundation model" and "many specialized deployments."

---

## Further Reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190) — Li & Liang, 2021
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) — Lester et al., 2021
- [LoRA+: Efficient Low Rank Adaptation](https://arxiv.org/abs/2402.12354) — Improved LoRA, 2024
