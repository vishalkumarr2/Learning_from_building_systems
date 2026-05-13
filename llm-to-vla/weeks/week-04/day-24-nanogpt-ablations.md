# Day 24: nanoGPT Ablation Experiments (Day 2)

> **Phase II — Attention, Transformers & Scaling | Week 4 | 2.5 hours**
> *"Science is systematic elimination of variables. An ablation study is science applied to architecture."*

## Navigation
- **Previous:** [Day 23: GPT & nanoGPT — Ablation Lab](day-23-nanogpt-ablation-lab.md)
- **Next:** [Day 25: Scaling Laws & Emergence](day-25-scaling-laws.md)
- **Week:** [Week 4 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/04-transformer-variants.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 24.1 Systematic Ablation Methodology

An ablation study answers: *"How much does each component contribute to performance?"*

The golden rule: **change one thing at a time, measure impact, hold everything else constant.**

```
Ablation Design:
                                                    
  Baseline Model ──→ Change ONE variable ──→ Retrain ──→ Compare
       │                                                    │
       │         ┌─ n_layers: 2, 4, 6, 8, 12              │
       │         ├─ n_heads: 1, 2, 4, 8                    │
       │         ├─ d_model: 64, 128, 256, 512             │
       │         ├─ activation: ReLU, GELU, SwiGLU         │
       │         └─ norm: LayerNorm, RMSNorm, Pre/Post-LN  │
       │                                                    │
       └──── Same data, same optimizer, same seed ──────────┘
```

### 24.2 Why Ablations Matter

Without ablations, you can't distinguish between:
- **A component that helps** (removing it hurts performance)
- **A component that's neutral** (removing it has no effect)
- **A component that hurts** (removing it improves performance)

Many famous architecture innovations were never properly ablated. When later work did ablations, results were surprising — e.g., Pre-LN transformers are much easier to train than Post-LN, but Post-LN often achieves slightly better final loss when you can get it to converge.

### 24.3 What to Measure

For each ablation run, track:

| Metric | Why |
|--------|-----|
| Final validation loss | Primary quality metric |
| Training loss curve shape | Convergence speed and stability |
| Wall-clock time per step | Computational efficiency |
| Peak GPU memory | Practical constraint |
| Generated text quality | Qualitative sanity check |

### 24.4 Statistical Considerations

A single run per configuration is often noisy. For rigorous results:

1. **Run with 3 different random seeds** (minimum)
2. **Report mean ± standard deviation**
3. **Use the same data splits** across all runs
4. **Train for the same number of steps** (not epochs — batch size may differ)

In practice, for this exercise we'll do 1 run per config (time constraint), but note where variance might matter.

### 24.5 Interaction Effects

Ablating one variable at a time misses **interaction effects**:
- Increasing `n_heads` may only help if `d_model` is large enough (each head needs sufficient dimension $d_k = d_{\text{model}} / n_{\text{heads}}$)
- SwiGLU activation works best with adjusted FFN dimensions ($\frac{8}{3}d_{\text{model}}$ instead of $4d_{\text{model}}$)
- RMSNorm is most impactful in Pre-LN position

Full factorial design ($5 \times 4 \times 4 \times 3 \times 4 = 960$ runs) is impractical. We ablate one axis at a time from a reasonable baseline.

---

## Implementation (60 min)

### Setting Up the Ablation Framework

First, create a configuration system that makes ablations easy to run:

```python
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Literal
import json
import time

@dataclass
class AblationConfig:
    """One configuration for an ablation run."""
    # Architecture
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024  # feedforward dimension
    activation: Literal["relu", "gelu", "swiglu"] = "gelu"
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_position: Literal["pre", "post"] = "pre"
    
    # Training (held constant across ablations)
    batch_size: int = 64
    block_size: int = 256
    learning_rate: float = 3e-4
    max_steps: int = 5000
    eval_interval: int = 250
    seed: int = 42
    
    # Metadata
    run_name: str = "baseline"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(xW) * (xV)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # SwiGLU uses 2/3 of the FF dimension for each gate
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
```

### Running the Depth Ablation

```python
def run_depth_ablation():
    """Ablation 1: Vary number of layers."""
    results = []
    
    for n_layers in [2, 4, 6, 8, 12]:
        config = AblationConfig(
            n_layers=n_layers,
            run_name=f"depth_{n_layers}L"
        )
        
        torch.manual_seed(config.seed)
        model = build_model(config)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        
        # Train and collect metrics
        t0 = time.time()
        train_losses, val_losses = train(model, config)
        wall_time = time.time() - t0
        
        results.append({
            "n_layers": n_layers,
            "n_params": n_params,
            "final_val_loss": val_losses[-1],
            "min_val_loss": min(val_losses),
            "wall_time_s": wall_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
        })
        
        print(f"  {n_layers}L | {n_params/1e6:.1f}M params | "
              f"val_loss={val_losses[-1]:.4f} | {wall_time:.0f}s")
    
    return results
```

### Running the Attention Head Ablation

```python
def run_head_ablation():
    """Ablation 2: Vary number of attention heads.
    
    Key constraint: d_model must be divisible by n_heads.
    With d_model=256: valid n_heads = 1, 2, 4, 8, 16, 32, ...
    """
    results = []
    d_model = 256
    
    for n_heads in [1, 2, 4, 8]:
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        d_k = d_model // n_heads
        
        config = AblationConfig(
            n_heads=n_heads,
            run_name=f"heads_{n_heads}H"
        )
        
        print(f"\n  n_heads={n_heads}, d_k={d_k}")
        
        torch.manual_seed(config.seed)
        model = build_model(config)
        train_losses, val_losses = train(model, config)
        
        results.append({
            "n_heads": n_heads,
            "d_k": d_k,
            "final_val_loss": val_losses[-1],
            "val_losses": val_losses,
        })
    
    return results
```

### Running the Model Width Ablation

```python
def run_width_ablation():
    """Ablation 3: Vary d_model (and proportionally d_ff = 4 * d_model)."""
    results = []
    
    for d_model in [64, 128, 256, 512]:
        config = AblationConfig(
            d_model=d_model,
            d_ff=4 * d_model,
            n_heads=max(1, d_model // 64),  # keep d_k = 64
            run_name=f"width_{d_model}d"
        )
        
        torch.manual_seed(config.seed)
        model = build_model(config)
        n_params = sum(p.numel() for p in model.parameters())
        train_losses, val_losses = train(model, config)
        
        results.append({
            "d_model": d_model,
            "n_params": n_params,
            "final_val_loss": val_losses[-1],
            "val_losses": val_losses,
        })
    
    return results
```

### Activation and Normalization Ablations

```python
def run_activation_ablation():
    """Ablation 4: Swap activation function."""
    results = []
    
    for act in ["relu", "gelu", "swiglu"]:
        config = AblationConfig(
            activation=act,
            # SwiGLU conventionally uses 8/3 * d_model for FF dim
            d_ff=int(8/3 * 256) if act == "swiglu" else 1024,
            run_name=f"act_{act}"
        )
        
        torch.manual_seed(config.seed)
        model = build_model(config)
        train_losses, val_losses = train(model, config)
        
        results.append({
            "activation": act,
            "final_val_loss": val_losses[-1],
            "val_losses": val_losses,
        })
    
    return results


def run_norm_ablation():
    """Ablation 5: Swap normalization type and position."""
    configs = [
        ("layernorm", "pre"),
        ("layernorm", "post"),
        ("rmsnorm", "pre"),
        ("rmsnorm", "post"),
    ]
    results = []
    
    for norm_type, norm_pos in configs:
        config = AblationConfig(
            norm_type=norm_type,
            norm_position=norm_pos,
            run_name=f"norm_{norm_type}_{norm_pos}"
        )
        
        torch.manual_seed(config.seed)
        model = build_model(config)
        train_losses, val_losses = train(model, config)
        
        results.append({
            "norm": f"{norm_type}-{norm_pos}",
            "final_val_loss": val_losses[-1],
            "val_losses": val_losses,
        })
    
    return results
```

### Plotting Results

```python
import matplotlib.pyplot as plt

def plot_ablation_curves(results, variable_name, title):
    """Plot validation loss curves for an ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: training curves
    for r in results:
        label = str(r[variable_name])
        axes[0].plot(r["val_losses"], label=label)
    axes[0].set_xlabel("Evaluation step")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title(f"{title} — Training Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: final loss bar chart
    names = [str(r[variable_name]) for r in results]
    losses = [r["final_val_loss"] for r in results]
    axes[1].bar(names, losses, color="steelblue")
    axes[1].set_xlabel(variable_name)
    axes[1].set_ylabel("Final val loss")
    axes[1].set_title(f"{title} — Final Performance")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(f"ablation_{variable_name}.png", dpi=150)
    plt.show()

# Plot all ablations
# plot_ablation_curves(depth_results, "n_layers", "Depth Ablation")
# plot_ablation_curves(head_results, "n_heads", "Head Count Ablation")
# plot_ablation_curves(width_results, "d_model", "Width Ablation")
# plot_ablation_curves(act_results, "activation", "Activation Ablation")
# plot_ablation_curves(norm_results, "norm", "Normalization Ablation")
```

### Plotting Your Own Scaling Curves

```python
def plot_scaling_curves(all_results):
    """Plot loss vs compute (params × steps) — your own scaling law!"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    params_list = []
    losses_list = []
    
    for r in all_results:
        n_params = r["n_params"]
        final_loss = r["final_val_loss"]
        params_list.append(n_params)
        losses_list.append(final_loss)
    
    ax.scatter(params_list, losses_list, s=80, zorder=5)
    
    # Fit power law: L = a * N^(-alpha)
    import numpy as np
    log_params = np.log(params_list)
    log_losses = np.log(losses_list)
    slope, intercept = np.polyfit(log_params, log_losses, 1)
    
    x_fit = np.linspace(min(log_params), max(log_params), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(np.exp(x_fit), np.exp(y_fit), '--', color='red',
            label=f"Power law: α = {-slope:.3f}")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Your Scaling Curve: Loss vs Parameters")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("my_scaling_curve.png", dpi=150)
    plt.show()
    
    print(f"Estimated scaling exponent: α = {-slope:.3f}")
    print(f"(Kaplan et al. found α ≈ 0.076 for params)")
```

---

## Exercise (45 min)

### E24.1 — Ablation Report

Create a structured ablation report with the following sections:

1. **Baseline specification:** Describe your baseline model configuration exactly
2. **Depth ablation:** Which depth gave best loss? At what cost?
3. **Head count ablation:** Did more heads help? What happened with 1 head vs 8?
4. **Width ablation:** How does d_model affect quality vs compute?
5. **Activation ablation:** Rank: ReLU vs GELU vs SwiGLU
6. **Normalization ablation:** Pre-LN vs Post-LN — which trains more stably?
7. **Scaling curve:** What's your estimated scaling exponent?

### E24.2 — Surprising Results

Answer these reflection questions:
1. Was any result surprising? Which component mattered more than you expected?
2. Are there interaction effects you suspect but didn't test?
3. If you could only run 3 more experiments, what would they be?

### E24.3 — Optimal Configuration

Based on your ablation results:
1. What's the best configuration you found?
2. How many parameters does it have?
3. How does it compare to the baseline?

---

## Key Takeaways

1. **Ablation = science:** Change one variable, measure impact, hold the rest constant
2. **Depth matters, but diminishing returns:** Going from 2→6 layers helps a lot; 6→12 helps less
3. **Width vs depth tradeoff:** For a fixed parameter budget, there's an optimal balance
4. **Pre-LN >> Post-LN** for training stability (the transformer community learned this the hard way)
5. **GELU ≥ SwiGLU > ReLU** in most settings, but SwiGLU can win with proper FFN sizing
6. **Your own scaling curves** follow the same power-law pattern as published results — the laws are real

## Connection to the Thread

Day 23 gave you a working nanoGPT. Today you turned it into a **scientific instrument**. The ablation methodology you practiced here is exactly what VLA papers use:

- RT-2 ablated vision encoder choice (ViT-B vs ViT-L)
- Octo ablated action head design (MLP vs diffusion)
- π₀ ablated training data mix ratios

The ability to run systematic experiments and interpret results is as important as the ability to build models.

## Further Reading

- Karpathy, "Let's build GPT from scratch" (YouTube, 2023)
- Merity et al., "Regularizing and Optimizing LSTM Language Models" (2018) — excellent ablation methodology
- Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond" (2020) — RAdam, ablation-driven
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) — the RMSNorm paper
- Shazeer, "GLU Variants Improve Transformer" (2020) — the SwiGLU paper
