# Day 81: Diffusion Policy — Denoising Actions Instead of Images

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 12 | 2.5 hours**
> *"Same math as Stable Diffusion. Different data. Instead of denoising pixels into images, we denoise noise into robot actions." — Chi et al., 2023*

## Navigation
- **Previous:** [Day 80: Decision Transformer](day-80-decision-transformer.md)
- **Next:** [Day 82: Action Representations](day-82-action-representations.md)
- **Week:** [Week 12 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/12-diffusion-flow.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 81.1 The Key Insight

Images and robot actions are both continuous, high-dimensional signals. If diffusion models can generate diverse, high-quality images, they can generate diverse, high-quality action sequences.

```
Image Diffusion:                    Diffusion Policy:
  Noise → denoise → image            Noise → denoise → action sequence
  Condition: text prompt              Condition: observation (image + state)
  Output: 512×512×3 pixels            Output: H×D action chunk (e.g., 16×7)
```

### 81.2 Architecture

```
Observation o_t:
  ┌────────────┐    ┌─────────────┐
  │ Image(s)   │───▶│ Visual      │──┐
  │            │    │ Encoder     │  │
  └────────────┘    └─────────────┘  │
  ┌────────────┐                     │  ┌──────────────┐
  │ Robot state│─────────────────────┼─▶│  Condition   │
  │ (joints)   │                     │  │  Features    │
  └────────────┘                     │  └──────┬───────┘
                                     │         │
                                     │    ┌────▼────────────────────┐
  Noisy actions a^(k)_t:t+H ────────┼───▶│  U-Net / Transformer   │
                                         │  ε_θ(a^(k), k, o_t)    │
                                         │  (predict noise)         │
                                         └────┬────────────────────┘
                                              │
                                    Predicted noise ε̂
                                              │
                                    DDPM/DDIM step
                                              │
                                    a^(k-1)_t:t+H (less noisy actions)
```

### 81.3 Why Diffusion for Actions?

| Challenge | BC Solution | Diffusion Policy Solution |
|-----------|-------------|--------------------------|
| Multimodal actions | Average modes (fails) | Samples from full distribution |
| Temporal consistency | Action chunking helps | Chunk + denoise = smooth trajectories |
| High-dimensional | Works but limited | Scales naturally to high-dim |
| Expressiveness | Single Gaussian | Arbitrary distribution shape |

### 81.4 Training

$$\mathcal{L} = \mathbb{E}_{k, a_0 \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \| \epsilon - \epsilon_\theta(a_0^{(k)}, k, o) \|^2 \right]$$

where:
- $a_0 = (a_t, a_{t+1}, \ldots, a_{t+H-1})$ — action chunk from demonstrations
- $k$ — diffusion timestep
- $o$ — observation conditioning
- $a_0^{(k)}$ — action chunk with noise added at level $k$

### 81.5 Inference

```python
# Start with random noise
a_K ~ N(0, I)   # shape: (H, action_dim)

# Iteratively denoise (DDIM for speed)
for k in [K, K-Δ, ..., 0]:
    ε̂ = ε_θ(a_k, k, observation)
    a_{k-Δ} = ddim_step(a_k, ε̂, k)

# Execute first few actions from the denoised chunk
execute(a_0[:n_exec])
```

### 81.6 Two Backbone Options

| Backbone | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **1D U-Net** (CNN) | Fast, good inductive bias for temporal structure | Fixed architecture | Real-time control |
| **Transformer** | Flexible, handles variable-length | Slower, needs more data | Multi-task, large models |

---

## Implementation (60 min)

### Diffusion Policy with 1D U-Net

```python
import torch
import torch.nn as nn

class ConditionalUNet1D(nn.Module):
    """1D U-Net for denoising action sequences."""
    def __init__(self, action_dim, obs_dim, hidden=256, diffusion_steps=100):
        super().__init__()
        self.diffusion_steps = diffusion_steps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(diffusion_steps, hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # Observation embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # 1D Conv blocks (simplified U-Net)
        self.down1 = nn.Sequential(
            nn.Conv1d(action_dim, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(hidden*2, hidden, 4, stride=2, padding=1), nn.SiLU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv1d(hidden*2, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, action_dim, 3, padding=1),
        )

    def forward(self, noisy_actions, timestep, obs):
        """
        noisy_actions: (B, H, action_dim) → permute to (B, action_dim, H)
        timestep: (B,)
        obs: (B, obs_dim)
        """
        x = noisy_actions.permute(0, 2, 1)  # (B, action_dim, H)

        # Conditioning
        t_emb = self.time_embed(timestep)  # (B, hidden)
        o_emb = self.obs_embed(obs)        # (B, hidden)
        cond = (t_emb + o_emb).unsqueeze(-1)  # (B, hidden, 1) for broadcasting

        # Down
        h1 = self.down1(x)        # (B, hidden, H)
        h1 = h1 + cond            # Add conditioning
        h2 = self.down2(h1)       # (B, hidden, H/2)

        # Mid
        h = self.mid(h2) + cond[:, :, :1].expand_as(h2)

        # Up (with skip connections)
        h = self.up2(torch.cat([h, h2], dim=1))
        h = self.up1(torch.cat([h, h1], dim=1))

        return h.permute(0, 2, 1)  # (B, H, action_dim)

class DiffusionPolicy:
    def __init__(self, action_dim, obs_dim, chunk_size=16, n_steps=100):
        self.model = ConditionalUNet1D(action_dim, obs_dim, diffusion_steps=n_steps)
        self.n_steps = n_steps

        # Linear noise schedule
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, 0)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def train_step(self, obs, actions, optimizer):
        """One training step."""
        B = obs.shape[0]
        k = torch.randint(0, self.n_steps, (B,))
        noise = torch.randn_like(actions)

        ab = self.alpha_bars[k].view(B, 1, 1)
        noisy = torch.sqrt(ab) * actions + torch.sqrt(1-ab) * noise

        pred_noise = self.model(noisy, k, obs)
        loss = ((pred_noise - noise)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, obs, n_steps=20):
        """DDIM-style sampling."""
        B = obs.shape[0]
        x = torch.randn(B, self.chunk_size, self.action_dim)
        step_indices = torch.linspace(self.n_steps-1, 0, n_steps, dtype=torch.long)

        for i in range(len(step_indices)-1):
            k = step_indices[i]
            k_batch = torch.full((B,), k, dtype=torch.long)
            pred_noise = self.model(x, k_batch, obs)

            ab = self.alpha_bars[k]
            ab_prev = self.alpha_bars[step_indices[i+1]]

            x0_pred = (x - torch.sqrt(1-ab) * pred_noise) / torch.sqrt(ab)
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1-ab_prev) * pred_noise

        return x
```

---

## Exercise (45 min)

1. **Compare BC vs Diffusion Policy:** On a multimodal toy task (two valid strategies), train both. Show that BC averages while diffusion captures both modes.

2. **Denoising steps at inference:** Test with 5, 10, 20, 50, 100 DDIM steps. What's the minimum for acceptable action quality?

3. **Chunk size sweep:** Try $H \in \{4, 8, 16, 32\}$. Measure trajectory smoothness and task success.

4. **Observation history:** Modify to condition on last $K$ observations instead of just current. Does temporal context help?

---

## Key Takeaways

1. **Same diffusion math** as image generation — just applied to action sequences
2. **Handles multimodality** — samples diverse strategies from the learned distribution
3. **Action chunking built-in** — denoise entire sequences for temporal consistency
4. **10-20 DDIM steps** is practical for real-time robot control (~5-10 Hz)
5. **Diffusion Policy is the backbone** of many modern VLA action heads

---

## Connection to the Thread

> *Diffusion Policy is the action-generation engine that will power VLAs. When we study π₀ (Day 96), it's a VLM backbone + flow matching action head — which is just a faster version of what we built today. Before VLAs though, we need to understand action representations (tomorrow) and how to tokenize actions for transformer-based VLAs like RT-2 (Day 83).*

---

## Further Reading

- Chi et al. (2023), "Diffusion Policy: Visuomotor Policy Learning via Action Score Gradients"
- Reuss et al. (2023), "Goal-Conditioned Imitation Learning using Score-based Diffusion Policies"
- [LeRobot Diffusion Policy](https://github.com/huggingface/lerobot)
