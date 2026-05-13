# Day 74: Diffusion Models Day 1 — DDPM

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"What if, to generate data, we simply learned to reverse the process of destroying it?" — Ho et al., 2020*

## Navigation
- **Previous:** [Day 73: PPO & RLHF Connection](day-73-ppo-rlhf.md)
- **Next:** [Day 75: Diffusion Day 2 — DDIM + CFG](day-75-ddim-cfg.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/12-diffusion-flow.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 74.1 The Core Idea

Diffusion models learn to generate data by learning to **denoise**:

```
Forward process (fixed):    x₀ → x₁ → x₂ → ··· → x_T ≈ N(0, I)
                           data   slightly   more        pure
                                  noisy     noisy       noise

Reverse process (learned):  x_T → x_{T-1} → ··· → x₁ → x₀
                           noise   slightly          nearly   clean
                                   cleaner          clean    data
```

**Key insight:** adding noise is trivial. Learning to remove noise is where the intelligence lives — and it turns out to be tractable as a sequence of small denoising steps.

### 74.2 Forward Process (Adding Noise)

Given data $x_0 \sim q(x)$, add Gaussian noise over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

where $\beta_t$ is a noise schedule ($\beta_1 < \beta_2 < \cdots < \beta_T$).

**Closed-form** — jump directly to any timestep $t$:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I)$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

**Reparameterization trick:**

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 74.3 Reverse Process (Learning to Denoise)

The reverse process is also Gaussian (when $\beta_t$ is small):

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

We train a neural network $\epsilon_\theta(x_t, t)$ to predict the noise that was added:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

### 74.4 Training Objective

The simplified DDPM loss is remarkably elegant:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**Training algorithm:**
1. Sample $x_0$ from data
2. Sample $t \sim \text{Uniform}(1, T)$
3. Sample $\epsilon \sim \mathcal{N}(0, I)$
4. Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. Train $\epsilon_\theta$ to predict $\epsilon$ from $(x_t, t)$

### 74.5 Noise Schedule

The schedule $\{\beta_t\}_{t=1}^T$ controls how fast noise is added:

| Schedule | Formula | Behavior |
|----------|---------|----------|
| Linear | $\beta_t = \beta_\min + \frac{t}{T}(\beta_\max - \beta_\min)$ | Original DDPM |
| Cosine | $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ | Better for images |
| Sigmoid | $\beta_t = \sigma(\text{linspace}(-6, 6, T))$ | Smoother transitions |

### 74.6 Architecture: U-Net with Time Conditioning

```
Input: x_t (noisy data) + t (timestep)
                │
        ┌───────▼───────┐
        │  Time Embed    │ ← sinusoidal embedding of t
        │  t → MLP → h_t│
        └───────┬───────┘
                │ (added/concatenated at each layer)
        ┌───────▼───────┐
        │   U-Net        │
        │   Encoder      │ → skip connections → Decoder
        │   (downsample) │                      (upsample)
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ Predicted ε    │ ← same shape as x_t
        └───────────────┘
```

---

## Implementation (60 min)

### Minimal DDPM on 2D Data

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Noise schedule ---
def linear_schedule(T=1000, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

T = 1000
betas, alphas, alpha_bars = linear_schedule(T)

# --- Forward process ---
def q_sample(x0, t, noise=None):
    """Add noise to x0 at timestep t."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bars[t].unsqueeze(-1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

# --- Denoising network (simple MLP for 2D) ---
class NoisePredictor(nn.Module):
    def __init__(self, data_dim=2, hidden=128, time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / T)
        return self.net(torch.cat([x, t_emb], dim=-1))

# --- Training ---
def make_swiss_roll(n=2000):
    t = torch.linspace(0, 4 * np.pi, n)
    x = t * torch.cos(t) / (4 * np.pi)
    y = t * torch.sin(t) / (4 * np.pi)
    return torch.stack([x, y], dim=-1) + 0.02 * torch.randn(n, 2)

model = NoisePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = make_swiss_roll(5000)

for epoch in range(2000):
    idx = torch.randint(0, len(data), (256,))
    x0 = data[idx]
    t = torch.randint(0, T, (256,))
    xt, noise = q_sample(x0, t)

    pred_noise = model(xt, t)
    loss = ((pred_noise - noise) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# --- Sampling (reverse process) ---
@torch.no_grad()
def p_sample(model, x, t_idx):
    t = torch.full((x.shape[0],), t_idx, dtype=torch.long)
    pred_noise = model(x, t)
    alpha = alphas[t_idx]
    alpha_bar = alpha_bars[t_idx]
    mean = (1 / torch.sqrt(alpha)) * (x - (betas[t_idx] / torch.sqrt(1 - alpha_bar)) * pred_noise)
    if t_idx > 0:
        noise = torch.randn_like(x)
        return mean + torch.sqrt(betas[t_idx]) * noise
    return mean

@torch.no_grad()
def sample(model, n_samples=1000):
    x = torch.randn(n_samples, 2)
    for t in reversed(range(T)):
        x = p_sample(model, x, t)
    return x

samples = sample(model)
```

---

## Exercise (45 min)

1. **Visualize the forward process:** Plot $x_t$ at $t \in \{0, 100, 300, 500, 800, 1000\}$. Show data dissolving into noise.

2. **Schedule comparison:** Implement cosine schedule. Compare $\bar{\alpha}_t$ curves. Train on the same data — which produces better samples?

3. **Loss landscape:** Log training loss per timestep bucket. Which timesteps are hardest to denoise?

4. **Connection to Day 5 (Information Theory):** Diffusion destroys information (adds entropy). Denoising recovers it. Write a paragraph connecting this to compression = prediction.

---

## Key Takeaways

1. **Forward process adds noise** — closed-form, no learning needed
2. **Reverse process removes noise** — learned by predicting $\epsilon$
3. **Loss = MSE on predicted noise** — simple but powerful
4. **U-Net + time conditioning** is the standard architecture
5. **$T = 1000$ steps** — many small denoising steps for quality

---

## Connection to the Thread

> *DDPM gives us the generative framework. But 1000 denoising steps is slow. Tomorrow: DDIM speeds this up with deterministic sampling, and classifier-free guidance lets us condition generation on text or goals. Day 76 moves diffusion to latent space (Stable Diffusion). Day 77 introduces flow matching — a simpler, faster alternative used by π₀. Then Week 12 applies all of this to robot actions.*

---

## Further Reading

- Ho et al. (2020), "Denoising Diffusion Probabilistic Models" — the foundational paper
- Sohl-Dickstein et al. (2015), "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
- Lilian Weng, "What are Diffusion Models?" — excellent illustrated tutorial
