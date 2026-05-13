# Day 75: Diffusion Day 2 — DDIM + Classifier-Free Guidance

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"Why take 1000 random steps when 50 deterministic ones will do?" — Song et al., 2021*

## Navigation
- **Previous:** [Day 74: Diffusion Models — DDPM](day-74-ddpm.md)
- **Next:** [Day 76: Diffusion Day 3 — Latent Diffusion](day-76-latent-diffusion.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/12-diffusion-flow.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 75.1 DDPM's Speed Problem

DDPM requires $T = 1000$ sequential denoising steps. For a robot running at 10 Hz, that's 100 seconds per action — completely impractical.

**DDIM** (Denoising Diffusion Implicit Models) solves this with deterministic sampling that can skip steps.

### 75.2 DDIM: Deterministic Sampling

DDIM reformulates the reverse process as a **non-Markovian** process:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \, z_t$$

When $\sigma_t = 0$: **deterministic** (same noise → same output, can skip steps).  
When $\sigma_t = \sqrt{\beta_t}$: reduces to DDPM (stochastic).

**Step skipping:** Instead of $t = 1000, 999, \ldots, 1, 0$, use a subsequence:

$$\tau = [1000, 800, 600, 400, 200, 0] \quad \text{(only 5 steps!)}$$

### 75.3 Speed vs Quality Trade-off

| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 1000 | Best | 100s | Offline image generation |
| 100 | Very good | 10s | High-quality generation |
| 50 | Good | 5s | Interactive generation |
| 10 | Acceptable | 1s | Real-time robot control |
| 1 | Poor | 0.1s | Distillation needed |

### 75.4 Classifier-Free Guidance (CFG)

**Goal:** generate data conditioned on some signal $c$ (text prompt, goal image, task description).

**Training:** randomly drop the condition with probability $p_\text{uncond}$ (typically 10%):

$$\epsilon_\theta(x_t, t, c) \quad \text{and} \quad \epsilon_\theta(x_t, t, \varnothing)$$

**Sampling:** interpolate between conditional and unconditional predictions:

$$\tilde{\epsilon}_\theta(x_t, t, c) = (1 + w) \cdot \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \varnothing)$$

where $w$ is the **guidance scale**:
- $w = 0$: no guidance (unconditional)
- $w = 1$: standard conditional
- $w = 3-7$: strong guidance (sharper, more aligned to condition)
- $w > 10$: over-saturated, artifacts

**For robot actions:** $c$ = task instruction + observation. CFG lets us control how strongly the policy follows the instruction vs exploring freely.

### 75.5 Conditional Diffusion for Actions (Preview)

```
Condition c = (image observation, language instruction)
                    │
            ┌───────▼────────┐
            │ Cross-attention │
            │ or FiLM layers  │
            └───────┬────────┘
                    │
    x_t (noisy actions) → ε_θ(x_t, t, c) → predicted noise
```

This is exactly how **Diffusion Policy** (Day 81) works — but for robot action sequences instead of images.

---

## Implementation (60 min)

### DDIM Sampler

```python
import torch
import numpy as np

def ddim_sample(model, shape, alpha_bars, steps=50, eta=0.0):
    """DDIM sampling with configurable step count and stochasticity.

    Args:
        model: trained noise predictor ε_θ(x_t, t)
        shape: output shape (batch, dim)
        alpha_bars: cumulative product of (1-β)
        steps: number of denoising steps (<<T)
        eta: stochasticity (0=deterministic, 1=DDPM)
    """
    T = len(alpha_bars)
    # Create step subsequence
    timesteps = torch.linspace(T-1, 0, steps+1, dtype=torch.long)

    x = torch.randn(shape)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]

        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict noise
        t_batch = torch.full((shape[0],), t, dtype=torch.long)
        eps_pred = model(x, t_batch)

        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)

        # DDIM update
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
        dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * eps_pred
        x = torch.sqrt(ab_prev) * x0_pred + dir_xt

        if sigma > 0 and t_prev > 0:
            x = x + sigma * torch.randn_like(x)

    return x

# Compare DDPM (1000 steps) vs DDIM (50 steps)
# Use the model trained in Day 74
samples_ddpm = sample(model, 1000)       # 1000 steps
samples_ddim = ddim_sample(model, (1000, 2), alpha_bars, steps=50)
```

### Classifier-Free Guidance Training

```python
class ConditionalNoisePredictor(nn.Module):
    def __init__(self, data_dim=2, cond_dim=1, hidden=128, time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + 2 * time_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )
        self.null_cond = nn.Parameter(torch.zeros(time_emb_dim))

    def forward(self, x, t, cond=None, p_uncond=0.1):
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / 1000)

        if cond is not None and self.training:
            # Randomly drop condition for CFG training
            mask = torch.rand(x.shape[0]) > p_uncond
            c_emb = self.cond_mlp(cond)
            c_emb[~mask] = self.null_cond
        elif cond is not None:
            c_emb = self.cond_mlp(cond)
        else:
            c_emb = self.null_cond.expand(x.shape[0], -1)

        return self.net(torch.cat([x, t_emb, c_emb], dim=-1))

def cfg_sample(model, shape, alpha_bars, cond, guidance_scale=3.0, steps=50):
    """Sample with classifier-free guidance."""
    x = torch.randn(shape)
    timesteps = torch.linspace(999, 0, steps+1, dtype=torch.long)

    for i in range(len(timesteps)-1):
        t = timesteps[i]
        t_batch = torch.full((shape[0],), t, dtype=torch.long)

        # Conditional and unconditional predictions
        eps_cond = model(x, t_batch, cond)
        eps_uncond = model(x, t_batch, None)

        # CFG interpolation
        eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

        # DDIM step with the guided noise prediction
        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[timesteps[i+1]] if timesteps[i+1] >= 0 else 1.0
        x0_pred = (x - torch.sqrt(1-ab_t) * eps) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1-ab_prev) * eps

    return x
```

---

## Exercise (45 min)

1. **Step count sweep:** Generate samples with DDIM using 5, 10, 25, 50, 100, 500, 1000 steps. Plot sample quality (visual or FID-like metric) vs inference time.

2. **Eta ablation:** Compare $\eta = 0$ (deterministic) vs $\eta = 0.5$ vs $\eta = 1.0$ (DDPM). For which settings are samples more diverse?

3. **Guidance scale sweep:** Train a conditional model where condition = quadrant label. Sample with $w \in \{0, 1, 3, 7, 15\}$. Visualize how guidance sharpens the distribution.

4. **Latency budget:** A robot arm runs at 10 Hz. Given your GPU, how many DDIM steps can you afford in 100ms? What quality trade-offs does that force?

---

## Key Takeaways

1. **DDIM enables fast sampling** by making denoising deterministic and skippable
2. **50 DDIM steps ≈ 1000 DDPM steps** in quality for many tasks
3. **CFG** = train with random condition dropout, sample with weighted interpolation
4. **Guidance scale $w$** trades diversity for condition alignment
5. **For robots:** DDIM's speed is essential — 10-50 steps fits within control loop latency

---

## Connection to the Thread

> *DDIM gives us speed, CFG gives us controllability. Tomorrow we move diffusion from pixel space to latent space (Stable Diffusion), cutting compute by another 10×. Day 77 introduces flow matching — a cleaner, ODE-based alternative to diffusion that π₀ uses for robot actions. The progression: DDPM (theory) → DDIM (speed) → Latent (efficiency) → Flow Matching (simplicity).*

---

## Further Reading

- Song et al. (2021), "Denoising Diffusion Implicit Models" (DDIM)
- Ho & Salimans (2021), "Classifier-Free Diffusion Guidance"
- Nichol & Dhariwal (2021), "Improved Denoising Diffusion Probabilistic Models"
