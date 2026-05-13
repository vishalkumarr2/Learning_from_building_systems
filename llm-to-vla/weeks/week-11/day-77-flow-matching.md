# Day 77: Flow Matching — From Diffusion to Straight Paths

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"Why curve when you can go straight? Flow matching replaces diffusion's wandering path with a highway." — Lipman et al., 2023*

## Navigation
- **Previous:** [Day 76: Diffusion Day 3 — Latent Diffusion](day-76-latent-diffusion.md)
- **Next:** [Day 78: Imitation Learning](../week-12/day-78-imitation-learning.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/12-diffusion-flow.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 77.1 Continuous Normalizing Flows (CNFs)

Instead of discrete noise steps, define a **continuous** transformation via an ODE:

$$\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]$$

- At $t = 0$: noise $x_0 \sim \mathcal{N}(0, I)$
- At $t = 1$: data $x_1 \sim p_\text{data}$
- The vector field $v_\theta$ transports noise to data along smooth paths

### 77.2 Flow Matching vs Diffusion

| Aspect | Diffusion (DDPM) | Flow Matching |
|--------|-----------------|---------------|
| Forward | Add noise via schedule | Linear interpolation |
| Paths | Curved, many steps | Straight lines |
| Training target | Predict noise $\epsilon$ | Predict velocity $v$ |
| ODE/SDE | SDE with noise schedule | ODE, cleaner math |
| Sampling | 50-1000 steps | 10-100 steps (Euler) |
| Noise schedule | Required (β₁...βₜ) | Not needed |

### 77.3 Conditional Flow Matching (CFM)

The **conditional flow matching** objective is beautifully simple:

Given data $x_1$ and noise $x_0 \sim \mathcal{N}(0, I)$, define the interpolation:

$$x_t = (1 - t) x_0 + t x_1$$

The target velocity field is just the difference:

$$u_t(x_t | x_1) = x_1 - x_0$$

**Training loss:**

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

That's it. No noise schedules. No $\bar{\alpha}_t$. No ELBO derivation. Just predict the straight-line velocity.

### 77.4 Optimal Transport Path

The straight-line interpolation is actually the **optimal transport** path between noise and data:

```
Diffusion path (curved):        Flow matching path (straight):

  x₁ (data) ·                    x₁ (data) ·
             · ·                            |
                · ·                         |
                   ·                        |
                  · ·                       |
                 ·                          |
  x₀ (noise) ·                   x₀ (noise)·
```

Straight paths = faster convergence, fewer integration steps, simpler optimization landscape.

### 77.5 Sampling via ODE Integration

At inference, integrate the learned velocity field:

$$x_1 = x_0 + \int_0^1 v_\theta(x_t, t) \, dt$$

Using Euler integration with $N$ steps:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t), \quad \Delta t = 1/N$$

### 77.6 Why π₀ Uses Flow Matching

Physical Intelligence's π₀ (Day 96) chose flow matching for robot actions because:
1. **Faster sampling**: 10-step Euler vs 50-step DDIM
2. **Simpler training**: no noise schedule tuning
3. **Better for continuous actions**: velocity fields naturally model smooth trajectories
4. **Composability**: easy to condition on VLM features via cross-attention

```
π₀ Architecture (preview):

  Image + Text → VLM Backbone → Features
                                    │
                                    ▼ (cross-attention)
  Noise x₀ → Flow Matching Head → Actions x₁
                   │
            v_θ(x_t, t, features)
```

---

## Implementation (60 min)

### Flow Matching on 2D Data

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class VelocityField(nn.Module):
    """Predict velocity v(x_t, t) that transports noise to data."""
    def __init__(self, dim=2, hidden=128, time_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x, t_emb], dim=-1))

def make_moons(n=2000):
    from sklearn.datasets import make_moons
    data, _ = make_moons(n, noise=0.05)
    return torch.tensor(data, dtype=torch.float32)

# Training
model = VelocityField()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = make_moons(5000)

for epoch in range(3000):
    idx = torch.randint(0, len(data), (256,))
    x1 = data[idx]                          # Data samples
    x0 = torch.randn_like(x1)              # Noise samples
    t = torch.rand(256)                      # Random time in [0, 1]

    # Interpolation
    xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1

    # Target velocity = straight-line direction
    target_v = x1 - x0

    # Predict velocity
    pred_v = model(xt, t)
    loss = ((pred_v - target_v) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# Sampling via Euler integration
@torch.no_grad()
def flow_sample(model, n_samples=1000, n_steps=100):
    x = torch.randn(n_samples, 2)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt)
        v = model(x, t)
        x = x + dt * v
    return x

samples = flow_sample(model, n_steps=50)
```

### Compare: Diffusion vs Flow Matching

```python
# Train both on same data, compare:
# 1. Training convergence speed
# 2. Sample quality at various step counts
# 3. Code simplicity

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data
axes[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
axes[0].set_title("Original Data")

# Flow matching samples (50 steps)
fm_samples = flow_sample(model, n_steps=50)
axes[1].scatter(fm_samples[:, 0], fm_samples[:, 1], s=1, alpha=0.5)
axes[1].set_title("Flow Matching (50 steps)")

# Flow matching samples (10 steps)
fm_10 = flow_sample(model, n_steps=10)
axes[2].scatter(fm_10[:, 0], fm_10[:, 1], s=1, alpha=0.5)
axes[2].set_title("Flow Matching (10 steps)")

for ax in axes:
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

---

## Exercise (45 min)

1. **Step count sweep:** Compare flow matching at 5, 10, 25, 50, 100, 500 steps. At how few steps does quality remain acceptable? Compare with DDIM at the same step counts.

2. **Conditional flow matching:** Add a class label condition. Train to generate only one class at a time. How does this compare to CFG in diffusion?

3. **Higher-order integrators:** Replace Euler with midpoint or RK4 integration. Does quality improve at low step counts?

4. **Action-space preview:** Instead of 2D points, generate 7-dimensional "actions" (e.g., 7-DOF joint positions). Does flow matching scale cleanly to higher dimensions?

---

## Key Takeaways

1. **Flow matching trains a velocity field** $v_\theta(x_t, t)$ instead of a noise predictor
2. **Linear interpolation** $x_t = (1-t)x_0 + tx_1$ replaces the noise schedule
3. **Target velocity = $x_1 - x_0$** — the simplest possible training signal
4. **10-50 Euler steps** suffice for good samples (vs 50-1000 for diffusion)
5. **π₀ uses flow matching** because it's faster and cleaner for continuous robot actions

---

## Connection to the Thread

> *Flow matching completes our generative model toolkit: DDPM (rigorous theory), DDIM (fast sampling), Latent Diffusion (efficient space), Flow Matching (elegant simplicity). Next week, we apply these frameworks to robot actions: imitation learning (Day 78), ACT (Day 79), Decision Transformer (Day 80), and Diffusion Policy (Day 81). The math stays the same — only the data changes from pixels to actions.*

---

## Further Reading

- Lipman et al. (2023), "Flow Matching for Generative Modeling"
- Liu et al. (2023), "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- Tong et al. (2024), "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"
- Black et al. (2024), "π₀: A Vision-Language-Action Flow Model for General Robot Control"
