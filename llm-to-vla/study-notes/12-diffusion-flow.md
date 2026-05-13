# 12 — Diffusion Models & Flow Matching

> Phase VI · Days 74–77 · ~10 hours
> Prerequisites: 01-dl-foundations
> Learning Objectives: Understand diffusion process, DDPM, diffusion for action generation, flow matching

---

## Navigation

| Previous | [11-rl-foundations](11-rl-foundations.md) | Next | [13-imitation-learning](13-imitation-learning.md) |
|----------|------------------------------------------|------|--------------------------------------------------|

---

## Overview

Diffusion models are the most important generative modeling breakthrough since GANs.
They learn to reverse a noise-adding process: starting from structured data and gradually
adding noise until it becomes pure Gaussian, then learning to reverse each step. For
robotics, diffusion models offer something transformative: the ability to model
**multimodal action distributions** — when there are multiple valid ways to perform a task,
diffusion can represent all of them, unlike simple regression which averages them out.

**Why diffusion for robotics?**
1. **Multimodality**: a robot reaching around an obstacle has multiple valid paths
2. **Expressiveness**: can model complex, high-dimensional distributions over action sequences
3. **Composability**: can condition on language, images, robot state simultaneously
4. **Action chunking**: naturally generates sequences of actions, not just single steps

---

## Day 74 — The Diffusion Process

### Forward Process: Destroying Data with Noise

Given a data point $x_0$ from our data distribution $q(x_0)$, the forward process
progressively adds Gaussian noise over $T$ timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \, \beta_t I)$$

where $\beta_t \in (0, 1)$ is the noise schedule at timestep $t$.

**Key property** — we can jump directly to any timestep $t$ without iterating:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, \, (1 - \bar{\alpha}_t) I)$$

where:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (cumulative product)

This is the **reparameterization trick**: sample $\epsilon \sim \mathcal{N}(0, I)$ and compute:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

At $t = T$ (with large $T$), $\bar{\alpha}_T \approx 0$, so $x_T \approx \epsilon \sim \mathcal{N}(0, I)$ — pure noise.

### Noise Schedules

The noise schedule $\{\beta_t\}_{t=1}^{T}$ controls how quickly data is destroyed:

| Schedule | Formula | Properties |
|----------|---------|-----------|
| Linear | $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$ | Original DDPM, simple |
| Cosine | $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ | Better for images, slower degradation |
| Sigmoid | $\beta_t = \sigma(a + b \cdot t/T)$ | Smooth, tunable |

For action diffusion in robotics, the schedule matters less than for image generation
because action spaces are lower-dimensional.

### The Reverse Process: What We Want to Learn

If we could reverse each forward step, we could start from noise $x_T \sim \mathcal{N}(0,I)$
and progressively denoise to get a sample from $q(x_0)$.

The true reverse process is:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

where:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

**Problem**: this requires knowing $x_0$, which is what we're trying to generate!

**Solution**: learn a neural network to estimate $x_0$ (or equivalently, the noise $\epsilon$)
from $x_t$ and $t$.

### Connection to Score Matching

The score function is the gradient of the log density: $\nabla_x \log q(x_t)$.

**Score matching equivalence**: predicting the noise $\epsilon$ is equivalent to predicting
the score:

$$\nabla_{x_t} \log q(x_t) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

This connects diffusion models to the rich theory of score-based generative models
(Song & Ermon, 2019).

### Key Concepts Checklist — Day 74

- [ ] Can write the forward process equation and the closed-form jump to timestep $t$
- [ ] Understand why $x_T$ is approximately Gaussian for large $T$
- [ ] Know the reparameterization trick for sampling $x_t$ given $x_0$
- [ ] Understand the relationship between noise prediction and score matching
- [ ] Can explain why the reverse process needs a learned model

---

## Day 75 — DDPM & Sampling

### Training: Predicting the Noise

**Denoising Diffusion Probabilistic Models** (Ho et al., 2020) train a neural network
$\epsilon_\theta(x_t, t)$ to predict the noise that was added:

**Training objective** (simplified):

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

where:
- $x_0 \sim q(x_0)$ — sample from data
- $\epsilon \sim \mathcal{N}(0, I)$ — sample noise
- $t \sim \text{Uniform}(1, T)$ — sample timestep
- $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$ — noisy version

**Training algorithm**:
```
repeat:
    sample x₀ from dataset
    sample t ~ Uniform(1, T)
    sample ε ~ N(0, I)
    compute x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
    take gradient step on ‖ε - ε_θ(x_t, t)‖²
until converged
```

### The Network Architecture

For images: U-Net with time embedding (sinusoidal positional encoding of $t$).

For actions (robotics): typically a simpler architecture:
- **1D temporal U-Net** for action sequences
- **MLP with time conditioning** for single-step actions
- **Transformer** for variable-length action sequences

Time conditioning is crucial: the network must know which noise level to denoise from.

### Sampling: Iterative Denoising

**DDPM sampling** (ancestral sampling):
```
x_T ~ N(0, I)                          # start from pure noise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    ε̂ = ε_θ(x_t, t)                    # predict noise
    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε̂) + σ_t · z
```

This requires $T$ forward passes (typically $T = 1000$), which is slow.

### DDIM: Faster Sampling

**Denoising Diffusion Implicit Models** (Song et al., 2021) enable non-Markovian sampling
that skips steps:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } x_0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t,t) + \sigma_t\,\epsilon_t$$

When $\sigma_t = 0$ (deterministic): DDIM is a deterministic ODE solver.
When $\sigma_t = \tilde{\beta}_t^{1/2}$: recovers DDPM.

**DDIM can use fewer steps** (e.g., 50 instead of 1000) by subsampling the timestep
sequence, making it much faster for deployment on robots.

### Classifier-Free Guidance (CFG)

To generate samples conditioned on a label/condition $c$ (e.g., a language instruction):

$$\tilde{\epsilon}_\theta(x_t, t, c) = (1 + w) \cdot \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \varnothing)$$

where:
- $w$ is the guidance weight (higher = more adherence to condition)
- $\varnothing$ is the null/unconditional input (dropout the condition during training)

**For robotics**: CFG can strengthen adherence to language commands.

### Comparison: Prediction Targets

| Target | Predict | Loss | Properties |
|--------|---------|------|-----------|
| $\epsilon$-prediction | Noise $\epsilon$ | $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ | Standard DDPM, stable training |
| $x_0$-prediction | Clean data $x_0$ | $\|x_0 - x_{0,\theta}(x_t,t)\|^2$ | Direct, simpler to interpret |
| $v$-prediction | Velocity $v = \sqrt{\bar\alpha}\epsilon - \sqrt{1-\bar\alpha}x_0$ | $\|v - v_\theta(x_t,t)\|^2$ | Better for high-resolution |

### Key Concepts Checklist — Day 75

- [ ] Can write the DDPM training loss
- [ ] Understand the sampling procedure (iterative denoising)
- [ ] Know why DDIM is faster than DDPM
- [ ] Can explain classifier-free guidance and when to use it
- [ ] Understand the different prediction targets (ε, x₀, v)

---

## Day 76 — Diffusion for Actions

### The Core Insight: Actions as Distributions

Traditional approach: predict a single action $a = f_\theta(\text{obs})$ (regression).

**Problem**: when multiple actions are valid (reaching around an obstacle from the left or
right), regression averages them → invalid action (going through the obstacle).

**Diffusion approach**: model the full distribution $p(a | \text{obs})$ by learning to
denoise random noise into valid actions.

### Diffusion Policy (Chi et al., 2023)

**Architecture**: condition the denoising network on observations:

$$\epsilon_\theta(a_t^k, k, O_t) \quad \text{where } O_t = \{o_{t-H_o}, \ldots, o_t\}$$

- $a_t^k$: noisy action at diffusion step $k$
- $k$: diffusion timestep (noise level)
- $O_t$: observation history (images, robot state)

**Key design choices**:

1. **Action chunking**: predict a sequence of $T_a$ future actions $[a_t, a_{t+1}, \ldots, a_{t+T_a-1}]$
   - Provides temporal consistency
   - Reduces compounding errors
   - The diffusion process denoises the entire chunk at once

2. **Observation conditioning**: encode observations with a vision encoder (ResNet, ViT),
   then condition the denoising network via cross-attention or concatenation.

3. **Receding horizon control**: predict chunk of $T_a$ actions, execute first $T_e < T_a$
   actions, re-plan with new observations.

### Training Diffusion Policy

```
for each (observation, action_chunk) pair from demonstrations:
    O = encode_observations(obs_history)
    a₀ = action_chunk                    # clean actions
    k ~ Uniform(1, K)                    # sample diffusion step
    ε ~ N(0, I)                          # sample noise
    aₖ = √ᾱₖ · a₀ + √(1-ᾱₖ) · ε       # noisy actions
    loss = ‖ε - ε_θ(aₖ, k, O)‖²        # predict noise
    update θ
```

### Inference: Action Generation

```
O = encode_observations(current_obs_history)
a_K ~ N(0, I)                            # start from random actions
for k = K, K-1, ..., 1:                  # denoise iteratively
    ε̂ = ε_θ(aₖ, k, O)
    a_{k-1} = denoise_step(aₖ, ε̂, k)    # DDPM or DDIM step
return a₀                                # denoised action chunk
```

**With DDIM**: can use ~10-20 steps instead of 100, making real-time control feasible.

### Why Diffusion is Great for Multimodal Actions

| Property | Regression | Gaussian Mixture | Diffusion |
|----------|-----------|-----------------|-----------|
| Single mode | ✓ | ✓ | ✓ |
| Multiple modes | ✗ (averages) | ✓ (fixed K) | ✓ (unlimited) |
| Continuous modes | ✗ | ✗ | ✓ |
| Complex correlations | Limited | Limited | **Rich** |
| Action sequences | Possible | Hard | **Natural** |

**Example**: pick up a mug.
- It can be grasped from the handle, the rim, or the body
- Each grasp requires a different approach trajectory
- Regression → hand goes to the "average" position (middle of mug, bad grasp)
- Diffusion → each sample produces a valid, complete grasp trajectory

### Architecture Variants

**1D Temporal U-Net** (Chi et al., 2023):
- Treats action sequence as a 1D signal
- Downsampling → bottleneck → upsampling with skip connections
- Observation conditioning via FiLM (feature-wise linear modulation)

**Transformer-based** (Reuss et al., 2024):
- Self-attention over action tokens
- Cross-attention to observation tokens
- Better for variable-length sequences

### Key Concepts Checklist — Day 76

- [ ] Can explain why regression fails for multimodal action distributions
- [ ] Understand the Diffusion Policy architecture and training loop
- [ ] Know what action chunking is and why it helps
- [ ] Can explain receding horizon control for real-time deployment
- [ ] Understand the inference procedure: noise → valid actions

---

## Day 77 — Flow Matching

### Motivation: Simplifying Diffusion

Diffusion models work but have complications:
- Complex noise schedules
- Many denoising steps
- The reverse process is approximate
- Connection to ODEs/SDEs is indirect

**Flow matching** (Lipman et al., 2023; Liu et al., 2023) takes a cleaner approach:
define a direct velocity field that transports noise to data.

### Continuous Normalizing Flows (Background)

A continuous normalizing flow defines a time-dependent vector field $v_t(x)$ that
transports probability densities:

$$\frac{dx}{dt} = v_t(x), \quad t \in [0, 1]$$

Starting from $x_0 \sim p_0$ (e.g., Gaussian noise) at $t=0$ and integrating the ODE
to $t=1$ gives samples from $p_1$ (the data distribution).

### Conditional Flow Matching

**Key idea**: instead of learning the marginal velocity field (intractable), learn
conditional velocity fields that interpolate between noise and data.

**Conditional probability path** (simplest: optimal transport):

$$x_t = (1-t)\,\epsilon + t\,x_1, \quad \epsilon \sim \mathcal{N}(0,I), \; x_1 \sim q(x_1)$$

This is a straight-line interpolation between noise ($t=0$) and data ($t=1$).

**Conditional velocity field** (the target):

$$u_t(x | x_1) = x_1 - \epsilon = x_1 - x_0$$

**Flow matching loss**:

$$L_{\text{FM}} = \mathbb{E}_{t, x_1, \epsilon}\left[\|v_\theta(x_t, t) - (x_1 - \epsilon)\|^2\right]$$

where $v_\theta$ is the learned velocity field (neural network).

### Comparison: Diffusion vs Flow Matching

| Aspect | DDPM/DDIM | Flow Matching |
|--------|-----------|---------------|
| Path | Curved (noise schedule-dependent) | **Straight line** (optimal transport) |
| Training target | Noise $\epsilon$ | Velocity $v = x_1 - x_0$ |
| Loss | $\|\epsilon - \epsilon_\theta\|^2$ | $\|v - v_\theta\|^2$ |
| Sampling | 20-1000 steps | **5-20 steps** (straighter paths) |
| Theory | Approximate reverse SDE | **Exact** ODE |
| Noise schedule | Required (sensitive) | **Not needed** |
| Implementation | Moderate | **Simpler** |

**Why fewer steps?** Straight-line paths are easier to integrate numerically than the
curved paths in diffusion. The ODE solver needs fewer steps to follow a straight line.

### Flow Matching for Actions

**Flow Matching Policy** (similar to Diffusion Policy but with flow matching):

$$L = \mathbb{E}_{t, a_1, \epsilon}\left[\|v_\theta(a_t, t, O) - (a_1 - \epsilon)\|^2\right]$$

where:
- $a_1$ is the clean action chunk from demonstrations
- $\epsilon \sim \mathcal{N}(0, I)$ is noise
- $a_t = (1-t)\epsilon + t \cdot a_1$ is the interpolated action
- $O$ is the observation conditioning
- $t \sim \text{Uniform}(0, 1)$

**Sampling** (Euler ODE solver):
```
a₀ ~ N(0, I)                           # start from noise
for i = 0, 1, ..., N-1:                # N integration steps
    t = i / N
    v = v_θ(aₜ, t, O)                  # predict velocity
    a_{t+dt} = aₜ + (1/N) · v          # Euler step
return a₁                              # final action
```

### Rectified Flows and Reflow

**Rectified flows** (Liu et al., 2023): iteratively straighten the learned flow paths
to enable even fewer sampling steps.

1. Train initial flow matching model
2. Generate $(x_0, x_1)$ pairs using the model
3. Retrain on the straightened pairs
4. Repeat → paths become straighter → fewer ODE steps needed

After 2-3 iterations, **1-step generation** becomes possible.

### Advanced: Conditional Flow Matching Variants

| Variant | Path | Properties |
|---------|------|-----------|
| Gaussian (OT) | $x_t = (1-t)\epsilon + tx_1$ | Simplest, straight lines |
| VP (variance preserving) | $x_t = \cos(\pi t/2)\epsilon + \sin(\pi t/2)x_1$ | Equivalent to diffusion VP |
| Stochastic interpolants | General affine interpolation | Most flexible |

### Connection to Diffusion Models

Flow matching and diffusion are different perspectives on the same underlying process:

- **Diffusion**: adds noise (forward SDE), removes noise (reverse SDE or ODE)
- **Flow matching**: defines a direct transport (ODE) from noise to data
- **Equivalence**: the probability flow ODE of diffusion = the flow matching ODE with specific choices

Flow matching is often simpler to implement and faster to sample from, making it
increasingly popular for robotics applications.

### Key Concepts Checklist — Day 77

- [ ] Can explain the conditional flow matching loss
- [ ] Understand why straight-line paths enable fewer sampling steps
- [ ] Know the difference between diffusion and flow matching training
- [ ] Can describe the Euler sampling procedure for flow matching
- [ ] Understand rectified flows and why they enable 1-step generation

---

## Hands-On: Minimal 2D Diffusion Model

Let's implement the core of DDPM — the forward noising and reverse denoising:

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleDenoisingNet(nn.Module):
    """Predicts noise ε given noisy input x_t and timestep t."""
    def __init__(self, data_dim=2, hidden=256, T=100):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden)
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, data_dim),
        )
    
    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))

class SimpleDDPM:
    """Minimal DDPM for 2D data."""
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.model = SimpleDenoisingNet(T=T)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def forward_process(self, x_0, t):
        """q(x_t | x_0) = N(sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)"""
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise
    
    def train_step(self, x_0):
        """One training step: predict the noise that was added."""
        t = torch.randint(0, self.T, (x_0.shape[0],))
        x_t, noise = self.forward_process(x_0, t)
        noise_pred = self.model(x_t, t)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def sample(self, n_samples=200):
        """Reverse process: start from noise, iteratively denoise."""
        x = torch.randn(n_samples, 2)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long)
            noise_pred = self.model(x, t_batch)
            
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            
            # DDPM sampling formula
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )
            if t > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)
        return x

# Generate training data: mixture of 4 Gaussians (a simple 2D distribution)
def make_moons_data(n=1000):
    """2D crescent moon dataset."""
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=n, noise=0.05)
    return torch.FloatTensor(X)

# Train
ddpm = SimpleDDPM(T=100)
data = make_moons_data(2000)

for epoch in range(200):
    idx = torch.randperm(len(data))[:256]
    loss = ddpm.train_step(data[idx])
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f}")

# Sample from the learned distribution
samples = ddpm.sample(500)
print(f"Generated {len(samples)} samples, shape: {samples.shape}")
# To visualize: plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
```

> **Key insight**: The diffusion model learns to reverse a *known* noising process.
> Training = predict the noise ε that was added. Sampling = iteratively subtract
> predicted noise. This same principle, applied to robot actions instead of 2D
> points, gives us **Diffusion Policy** (Note 13-14).

---

## Key Equations Reference

### DDPM Forward Process

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)\,I)$$

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

### DDPM Training Loss

$$L = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### Classifier-Free Guidance

$$\tilde{\epsilon}_\theta = (1+w)\,\epsilon_\theta(x_t, t, c) - w\,\epsilon_\theta(x_t, t, \varnothing)$$

### Flow Matching Loss

$$L_{\text{FM}} = \mathbb{E}_{t, x_1, \epsilon}\left[\|v_\theta(x_t, t) - (x_1 - \epsilon)\|^2\right], \quad x_t = (1-t)\epsilon + t\,x_1$$

### Diffusion Policy Action Generation

$$\epsilon_\theta(a_t^k, k, O_t): \text{noise prediction conditioned on observations}$$

---

## Comparison: Generative Models for Actions

| Method | Multimodal | Steps | Training | Architecture | Use in VLAs |
|--------|-----------|-------|----------|-------------|-------------|
| Regression (MSE) | ✗ | 1 | Simplest | MLP/Transformer | RT-1, RT-2 |
| Gaussian Mixture | Limited | 1 | Simple | MLP + mixture head | Rare |
| VAE | Moderate | 1 | ELBO | Encoder + Decoder | ACT |
| Autoregressive | ✓ | $T_a$ | Cross-entropy | Transformer | RT-2, π₀ |
| **Diffusion** | **✓** | 10-100 | MSE | U-Net/Transformer | **Diffusion Policy, π₀** |
| **Flow Matching** | **✓** | **5-20** | MSE | MLP/Transformer | **π₀, emerging** |
| Consistency | ✓ | **1-2** | Distillation | U-Net/Transformer | Emerging |

---

## Key Takeaways

1. **Diffusion models** learn to reverse a noise-adding process. Training is simple
   (predict the noise), but sampling requires many iterative steps.

2. **The key insight for robotics**: diffusion can model multimodal action distributions —
   when there are multiple valid ways to do a task, diffusion naturally represents all of them.

3. **Action chunking** with diffusion = denoise entire action sequences at once, providing
   temporal consistency and reducing compounding errors.

4. **Flow matching** is a simpler, faster alternative: straight-line interpolation between
   noise and data enables fewer sampling steps and simpler training.

5. **The trend in VLAs**: moving from autoregressive action prediction (RT-2) to diffusion/flow
   matching (π₀), because multimodal action distributions matter for dexterous manipulation.

---

## Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Ho et al., "Denoising Diffusion Probabilistic Models" | 2020 | DDPM — foundational diffusion paper |
| Song et al., "Denoising Diffusion Implicit Models" | 2021 | DDIM — faster sampling |
| Song & Ermon, "Score-Based Generative Modeling" | 2019 | Score matching perspective |
| Chi et al., "Diffusion Policy" | 2023 | **Diffusion for robot action generation** |
| Ho & Salimans, "Classifier-Free Diffusion Guidance" | 2022 | CFG for conditional generation |
| Lipman et al., "Flow Matching for Generative Modeling" | 2023 | Flow matching framework |
| Liu et al., "Flow Straight and Fast (Rectified Flow)" | 2023 | Rectified flows, 1-step generation |
| Black et al., "π₀: A Vision-Language-Action Flow Model" | 2024 | Flow matching for VLA |

---

## Connection to the Thread

> **Diffusion = iterative decompression from noise**: start with maximum entropy (random noise)
> and progressively add structure until a valid prediction emerges. Each denoising step adds
> a small amount of information, gradually compressing the infinite space of possibilities
> into a specific, valid action.
>
> **Flow matching = direct transport**: draw a straight line from noise to data in probability
> space. Simpler, faster, and mathematically cleaner.
>
> Both methods solve the same problem: turning noise into structured predictions. For robotics,
> this means turning random joint configurations into smooth, purposeful motion. The key
> advantage over simple regression: when multiple valid actions exist, these methods can
> represent ALL of them, not just their average.
>
> **Next**: we'll see how to combine these generative models with demonstrations to create
> imitation learning systems that can compress expert behavior into deployable policies.

---

## What's Next

In [Study Note 13](13-imitation-learning.md), we bring together RL foundations and diffusion
models to study **imitation learning** — the dominant paradigm for robot learning. We'll
implement behavioral cloning, ACT (Action Chunking with Transformers), and Diffusion Policy,
understanding why demonstrations are the most efficient way to compress expert knowledge into
robot behavior. The STOP AND REFLECT moment will connect everything: language understanding +
visual perception + action generation = VLA.
