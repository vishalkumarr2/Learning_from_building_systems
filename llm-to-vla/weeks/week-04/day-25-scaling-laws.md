# Day 25: Scaling Laws & Emergence

> **Phase II — Attention, Transformers & Scaling | Week 4 | 2.5 hours**
> *"There is a smooth, predictable relationship between compute and capability. This is either the most exciting or the most terrifying fact in AI." — Ilya Sutskever*

## Navigation
- **Previous:** [Day 24: nanoGPT Ablation Experiments](day-24-nanogpt-ablations.md)
- **Next:** [Day 26: Stop & Reflect #2](day-26-stop-reflect-2.md)
- **Week:** [Week 4 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/05-gpt-scaling.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 25.1 Kaplan Scaling Laws (2020)

The Kaplan et al. paper ("Scaling Laws for Neural Language Models") showed that language model loss follows **power laws** in three independent variables:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

where:
- $N$ = number of parameters (excluding embeddings)
- $D$ = number of data tokens
- $C$ = compute budget in FLOPs ($C \approx 6ND$ for a forward+backward pass)
- $N_c, D_c, C_c$ are constants
- $\alpha_N, \alpha_D, \alpha_C$ are the scaling exponents

```
Log-Log Plot of Loss vs Compute:

  Loss (log)
    │
  2 ┤  ●
    │    ●
    │      ●
  1 ┤        ●
    │          ●
    │            ●          ← straight line on log-log = power law!
 0.5┤              ●
    │                ●
    │                  ●
    └──┬──┬──┬──┬──┬──┬──→ Compute (log FLOPs)
      10¹⁷  10¹⁹  10²¹  10²³
```

**Key insight:** The loss doesn't plateau — it keeps improving as a smooth power law. No architecture change needed. Just add more compute.

### 25.2 The Kaplan Allocation Rule (and Why It Was Wrong)

Kaplan's original recommendation: when given more compute, scale **parameters faster than data**. Specifically:

$$N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}$$

This led to GPT-3: 175B parameters trained on "only" 300B tokens. Massive model, relatively little data.

### 25.3 Chinchilla Scaling Laws (2022)

Hoffmann et al. at DeepMind showed Kaplan was wrong about the optimal allocation. Their finding:

$$N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}$$

Parameters and data should scale **equally**. The rule of thumb:

> **Chinchilla-optimal: train on ~20 tokens per parameter.**

| Model | Params | Training Tokens | Tokens/Param | Status |
|-------|--------|----------------|--------------|--------|
| GPT-3 | 175B | 300B | 1.7 | Severely undertrained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| LLaMA 2 70B | 70B | 2T | 29 | Over-trained (for inference efficiency) |
| LLaMA 3 8B | 8B | 15T | 1875 | Massively over-trained (deliberate) |

**Why over-train?** For deployment: a smaller, over-trained model is cheaper to serve than a larger, optimally-trained model with the same quality. LLaMA 3 pushed this to the extreme.

### 25.4 The Compression Interpretation (Sutskever)

Ilya Sutskever's elegant reframing:

> *"Prediction is compression. A model that predicts the next token well is one that has found a compact representation of the data's structure. Scaling laws are compression efficiency curves."*

Cross-entropy loss is literally the number of bits needed to encode the data under the model's predictions:

$$L = -\frac{1}{T}\sum_{t=1}^{T} \log_2 p(x_t | x_{<t})$$

As compute increases:
- The model finds better compressions of the training data
- Loss decreases along a power law
- Each bit of improvement means the model captures more structure

```
Compression View:

  Bits per token
       │
  10   ┤  Raw text (no compression)
       │
   5   ┤  ●  Small model (memorizes common patterns)
       │     ●
   3   ┤        ●  Medium model (grammar, basic semantics)
       │           ●
   2   ┤              ●  Large model (world knowledge)
       │                 ●
  1.5  ┤                    ●  Very large (reasoning, analogy)
       │                       ●
       └──────────────────────────→ log(Compute)
```

**The profound implication:** If compression = understanding, and scaling laws say compression improves smoothly with compute, then understanding improves smoothly with compute. There's no wall (yet).

### 25.5 Emergence: Real or Artifact?

**The original claim (Wei et al., 2022):** Some abilities appear *suddenly* at a certain scale:
- Chain-of-thought reasoning: absent in small models, present in large ones
- In-context learning: few-shot performance jumps discontinuously
- Multi-step arithmetic: impossible below ~100B parameters

**The counter-argument (Schaeffer et al., 2023):** "Emergent abilities" may be a measurement artifact:
- If you measure with accuracy (0 or 1), performance looks discontinuous
- If you measure with log-probability (continuous), improvement is smooth
- The "emergence" is in the metric, not the model

```
Same model, different metrics:

  Accuracy                         Log-probability
     │                                │
  1  ┤          ●●●●●                 │         ●●●●●
     │         /                      │       ●
     │        /                       │     ●
     │       /                        │   ●
  0  ┤●●●●●/  ← looks sudden!     -5 ┤ ●  ← looks smooth!
     └──────────→ scale              └──────────→ scale
```

**Current consensus:** The truth is nuanced. Some capabilities genuinely have phase transitions (likely related to the model's ability to compose learned primitives), but many reported emergences are metric artifacts.

### 25.6 In-Context Learning as Bayesian Inference

One of the most remarkable abilities of large language models: **in-context learning** (ICL). Given a few examples in the prompt, the model generalizes to new instances — without any gradient updates.

Xie et al. (2022) showed that ICL can be understood as implicit Bayesian inference:

$$p(y | x, \text{examples}) \approx \sum_{\theta} p(y | x, \theta) \cdot p(\theta | \text{examples})$$

The model maintains an implicit posterior over "concepts" $\theta$ and updates it as it processes each example. Larger models maintain better posteriors — they become better *learners*, not just better *memorizers*.

This is directly relevant to robotics: if a VLA can do in-context learning, you could show it a few demonstrations and it would generalize to new situations without fine-tuning.

---

## Implementation (60 min)

### Reproducing Scaling Curves from Published Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Published data points from Kaplan et al. (approximate)
# (Parameters, Cross-entropy loss)
kaplan_data = {
    "params": np.array([7.7e5, 1.5e6, 3.0e6, 6.1e6, 1.2e7, 2.4e7,
                         4.9e7, 9.8e7, 2.0e8, 3.5e8, 7.7e8, 1.5e9]),
    "loss":   np.array([4.20, 3.95, 3.72, 3.52, 3.35, 3.20,
                         3.07, 2.95, 2.85, 2.77, 2.68, 2.60]),
}

# Fit power law: L(N) = a * N^(-alpha)
log_N = np.log(kaplan_data["params"])
log_L = np.log(kaplan_data["loss"])
slope, intercept = np.polyfit(log_N, log_L, 1)
alpha_N = -slope

print(f"Fitted scaling exponent: α_N = {alpha_N:.4f}")
print(f"(Kaplan reported: α_N ≈ 0.076)")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(kaplan_data["params"], kaplan_data["loss"],
           s=60, zorder=5, label="Published data")

N_fit = np.logspace(np.log10(5e5), np.log10(3e9), 100)
L_fit = np.exp(intercept) * N_fit ** slope
ax.plot(N_fit, L_fit, '--', color='red',
        label=f"Power law fit: α = {alpha_N:.4f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Parameters (N)", fontsize=12)
ax.set_ylabel("Cross-entropy Loss", fontsize=12)
ax.set_title("Scaling Law: Loss vs Parameters", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("kaplan_scaling_law.png", dpi=150)
plt.show()
```

### Chinchilla Optimal Compute Calculator

```python
def chinchilla_optimal(compute_flops: float):
    """Calculate Chinchilla-optimal params and data for a given compute budget.
    
    Rule: C ≈ 6 * N * D, with N_opt ∝ C^0.5 and D_opt ∝ C^0.5
    Chinchilla ratio: D ≈ 20 * N
    
    Returns:
        (optimal_params, optimal_tokens)
    """
    # From C = 6ND and D = 20N:
    # C = 6N(20N) = 120N²
    # N = sqrt(C / 120)
    N_opt = np.sqrt(compute_flops / 120)
    D_opt = 20 * N_opt
    
    return N_opt, D_opt


# Example: How big should a model be for different compute budgets?
print("Chinchilla-Optimal Configurations:")
print(f"{'Compute (FLOPs)':>20} | {'Params':>12} | {'Tokens':>12} | {'Tokens/Param':>12}")
print("-" * 65)

for log_c in [18, 19, 20, 21, 22, 23, 24]:
    C = 10 ** log_c
    N, D = chinchilla_optimal(C)
    ratio = D / N
    
    # Human-readable
    def fmt(x):
        if x >= 1e12: return f"{x/1e12:.1f}T"
        if x >= 1e9: return f"{x/1e9:.1f}B"
        if x >= 1e6: return f"{x/1e6:.1f}M"
        return f"{x:.0f}"
    
    print(f"  10^{log_c:>2} FLOPs       | {fmt(N):>12} | {fmt(D):>12} | {ratio:>12.0f}")
```

### Compute Estimation for Training a 7B Model

```python
def estimate_training_compute(
    n_params: float,
    n_tokens: float,
    flops_per_token_per_param: float = 6.0,
    gpu_tflops: float = 312.0,  # A100 peak TF32
    gpu_utilization: float = 0.4,  # typical MFU
    n_gpus: int = 1,
):
    """Estimate wall-clock training time."""
    # Total FLOPs
    total_flops = flops_per_token_per_param * n_params * n_tokens
    
    # Effective throughput
    effective_tflops = gpu_tflops * gpu_utilization * n_gpus
    effective_flops_per_sec = effective_tflops * 1e12
    
    # Time
    seconds = total_flops / effective_flops_per_sec
    hours = seconds / 3600
    days = hours / 24
    
    return {
        "total_flops": total_flops,
        "gpu_hours": hours * n_gpus,
        "wall_days": days,
        "cost_at_2_per_gpu_hr": hours * n_gpus * 2.0,
    }


# 7B Chinchilla-optimal
result = estimate_training_compute(
    n_params=7e9,
    n_tokens=140e9,  # 20 * 7B = 140B tokens
    n_gpus=64,
)

print(f"\n7B Chinchilla-optimal training estimate:")
print(f"  Total FLOPs: {result['total_flops']:.2e}")
print(f"  GPU-hours (A100): {result['gpu_hours']:,.0f}")
print(f"  Wall time (64 GPUs): {result['wall_days']:.1f} days")
print(f"  Est. cost @ $2/GPU-hr: ${result['cost_at_2_per_gpu_hr']:,.0f}")
```

### How Much Data for a 70B Model?

```python
# Question: how much data for 70B Chinchilla-optimal?
N_70B = 70e9
D_optimal = 20 * N_70B  # Chinchilla rule
D_tokens = D_optimal

# How many bytes of text is that?
# Typical: ~4 characters per token, ~1 byte per character
bytes_per_token = 4
total_bytes = D_tokens * bytes_per_token
total_tb = total_bytes / 1e12

print(f"\n70B model, Chinchilla-optimal:")
print(f"  Tokens needed: {D_tokens/1e12:.1f}T")
print(f"  Raw text size: ~{total_tb:.1f} TB")
print(f"  (For reference, Common Crawl is ~250 TB compressed)")

# What about LLaMA 3's approach (over-train for smaller model)?
print(f"\n  LLaMA 3 8B trained on 15T tokens:")
print(f"  That's {15e12 / 8e9:.0f}x tokens per parameter")
print(f"  Chinchilla says 20x is optimal, so LLaMA 3 is ~{15e12/8e9/20:.0f}x over-trained")
```

---

## Exercise (45 min)

### E25.1 — Scaling Law Extrapolation

Using the power law $L(N) = aN^{-\alpha}$ with $\alpha \approx 0.076$:

1. If a 1B param model achieves loss 2.5, predict the loss for 10B, 100B, 1T params
2. How many parameters to reach loss 1.5? Is this realistic?
3. Plot your predictions alongside published data points

### E25.2 — Chinchilla Calculator

Build a calculator that takes a GPU budget (in GPU-hours) and outputs:
1. The Chinchilla-optimal model size and data requirement
2. The expected final loss (using Kaplan scaling law)
3. Whether the data exists (compare against known dataset sizes)

### E25.3 — The Robotics Question

Answer in 200 words: **Do scaling laws hold for robot learning?**

Consider:
- OpenVLA trained on 970K robot episodes — is this Chinchilla-optimal for its 7B params?
- Robot data costs ~$100/hour to collect. How much would Chinchilla-optimal data cost for a 7B VLA?
- Are there fundamental differences between language tokens and robot actions that might change the scaling exponent?

---

## Key Takeaways

1. **Loss = power law in compute** — $L(C) \propto C^{-\alpha}$, smooth and predictable
2. **Chinchilla changed everything** — most models before 2022 were undertrained; 20 tokens per parameter is the rule
3. **Over-training is deliberate** — smaller models trained longer are cheaper to deploy
4. **Compression = understanding** (Sutskever) — scaling laws are compression efficiency curves
5. **Emergence is nuanced** — some abilities are genuinely discontinuous, others are metric artifacts
6. **In-context learning = implicit Bayesian inference** — models become better learners at scale

## Connection to the Thread

Scaling laws are the empirical foundation of the entire LLM revolution. They tell us *why* spending billions on training works. For VLAs, the critical question is whether the same laws hold when the "tokens" are robot actions. Early evidence (RT-2, OpenVLA) suggests yes — but robot data is 1000× more expensive than text. This tension between scaling potential and data scarcity is the central challenge of robot intelligence.

## Further Reading

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) — Chinchilla
- Schaeffer et al., "Are Emergent Abilities of LLMs a Mirage?" (2023)
- Xie et al., "An Explanation of In-context Learning as Implicit Bayesian Inference" (2022)
- Sutton, "The Bitter Lesson" (2019) — http://www.incompleteideas.net/IncsightIdea/BitterLesson.html
- Sardana & Frankle, "Beyond Chinchilla-Optimal: Accounting for Inference" (2023)
