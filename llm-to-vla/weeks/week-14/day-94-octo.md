# Day 94: Octo — The Open Generalist Policy

> **Phase VII — VLAs: Architecture to Deployment | Week 14 | 2.5 hours**
> *"93M parameters. Open-source. Diffusion action head. Trains on Open X-Embodiment. Fine-tunes in hours." — Octo Team, 2024*

## Navigation
- **Previous:** [Day 93: RT-2](day-93-rt2.md)
- **Next:** [Day 95: OpenVLA](day-95-openvla.md)
- **Week:** [Week 14 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 94.1 Octo's Design Philosophy

| Design Choice | RT-2 | Octo |
|--------------|------|------|
| Size | 55B | 93M (600× smaller) |
| Action head | Autoregressive tokens | Diffusion |
| Pre-training | Web VLM + robot | Robot-only (Open X-Embodiment) |
| Open-source | No | Yes |
| Fine-tuning | Impractical | 4 GPU-hours |
| Embodiment | Single (Everyday Robots) | Multi (9 robots) |

### 94.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        OCTO                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Task tokens:     Language ──→ tokenize ──→ [t₁...tₖ]   │
│                   Goal image ──→ ViT patch ──→ [g₁...gₘ] │
│                                                          │
│  Observation:     Current image ──→ ViT patch ──→ [o₁..oₙ]│
│                   Proprioception ──→ MLP ──→ [p]          │
│                                                          │
│  Readout tokens:  [r₁, r₂, ..., rₖ] (learned)           │
│                                                          │
│  ┌───────────────────────────────────────────────┐       │
│  │  Transformer (blockwise causal attention)      │       │
│  │  Task tokens → observe self + task             │       │
│  │  Obs tokens  → observe self + task + obs       │       │
│  │  Readout tokens → observe everything           │       │
│  └───────────────────────────────────────────┬───┘       │
│                                              │            │
│  ┌───────────────────────────────────────────▼───┐       │
│  │  Diffusion Action Head                        │       │
│  │  Input: readout tokens + noisy actions        │       │
│  │  Output: denoised action chunk                │       │
│  └───────────────────────────────────────────────┘       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 94.3 Blockwise Causal Attention

Octo uses a specific attention pattern:

```
              Task  Observation  Readout
Task          ✓         ✗          ✗
Observation   ✓         ✓          ✗
Readout       ✓         ✓          ✓

✓ = can attend to
✗ = cannot attend to
```

**Why:** Task tokens encode the instruction (language or goal image). Observation tokens can see the task but not the readout. Readout tokens aggregate everything for the action head.

### 94.4 Diffusion Action Head

Instead of predicting discrete tokens like RT-2, Octo uses a continuous diffusion head:

$$\hat{a} = \text{DiffusionHead}(\text{readout\_tokens}, a^{(k)}, k)$$

- **Input:** readout tokens + noisy action chunk + diffusion timestep
- **Architecture:** small MLP with FiLM conditioning
- **Inference:** 10-20 DDIM denoising steps
- **Output:** continuous action chunk $(a_t, a_{t+1}, \ldots, a_{t+H})$

**Advantages over tokenized actions:**
1. Continuous output → no discretization loss
2. Multimodal distribution → handles ambiguous actions
3. Parallel decoding → faster than autoregressive

### 94.5 Multi-Embodiment Training

Octo handles different robots with per-embodiment action heads:

```
Shared backbone (transformer):
  Same weights for all robots
  Learns general visuomotor features
  
Per-embodiment heads:
  Robot A (7-DOF arm): diffusion head with 7D output
  Robot B (6-DOF + gripper): diffusion head with 7D output
  Robot C (mobile base): diffusion head with 3D output
```

### 94.6 Fine-Tuning Protocol

```
Pre-trained Octo (800K episodes, 9 robots)
    │
    ▼ Fine-tune on target robot (50-1000 demos)
    │
    ├── Option 1: Full fine-tune (all weights)
    │   Best performance, most compute
    │
    ├── Option 2: Head-only (freeze backbone)
    │   Fast, works with <100 demos
    │
    └── Option 3: LoRA (low-rank adaptation)
        Balance of performance and efficiency
```

---

## Implementation (60 min)

### Octo-Style Architecture

```python
import torch
import torch.nn as nn

class BlockwiseCausalTransformer(nn.Module):
    """Transformer with blockwise causal attention mask."""

    def __init__(self, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4, batch_first=True,
            )
            for _ in range(n_layers)
        ])

    def build_mask(self, n_task, n_obs, n_readout, device):
        """Build blockwise causal attention mask."""
        total = n_task + n_obs + n_readout
        mask = torch.ones(total, total, device=device, dtype=torch.bool)

        # Task tokens: attend to task only
        mask[:n_task, :n_task] = False

        # Obs tokens: attend to task + obs
        t_o = n_task + n_obs
        mask[n_task:t_o, :t_o] = False

        # Readout tokens: attend to everything
        mask[t_o:, :] = False

        return mask  # True = masked out

    def forward(self, task_tokens, obs_tokens, readout_tokens):
        n_t, n_o, n_r = task_tokens.shape[1], obs_tokens.shape[1], readout_tokens.shape[1]
        x = torch.cat([task_tokens, obs_tokens, readout_tokens], dim=1)
        mask = self.build_mask(n_t, n_o, n_r, x.device)

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        # Return readout tokens only
        return x[:, n_t + n_o:]

class DiffusionActionHead(nn.Module):
    """Diffusion head conditioned on readout tokens."""

    def __init__(self, readout_dim, action_dim, chunk_size=4,
                 hidden=256, n_steps=100):
        super().__init__()
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # FiLM conditioning from readout + timestep
        self.cond_net = nn.Sequential(
            nn.Linear(readout_dim + 128, hidden), nn.SiLU(),
        )
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, 128),
            nn.Linear(128, 128), nn.SiLU(),
        )
        # Noise prediction network
        self.noise_pred = nn.Sequential(
            nn.Linear(action_dim * chunk_size + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim * chunk_size),
        )

        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1 - betas
        self.register_buffer('alpha_bars', torch.cumprod(alphas, 0))

    def forward(self, readout, noisy_actions, timestep):
        t_emb = self.time_embed(timestep)
        # Pool readout tokens
        r = readout.mean(dim=1)
        cond = self.cond_net(torch.cat([r, t_emb], dim=-1))
        flat_actions = noisy_actions.flatten(1)
        pred = self.noise_pred(torch.cat([flat_actions, cond], dim=-1))
        return pred.view(-1, self.chunk_size, self.action_dim)

class OctoStyle(nn.Module):
    """Simplified Octo-style model."""

    def __init__(self, obs_dim=512, lang_dim=256, action_dim=7,
                 d_model=256, n_readout=4, chunk_size=4):
        super().__init__()
        self.n_readout = n_readout

        # Encoders
        self.obs_encoder = nn.Linear(obs_dim, d_model)
        self.lang_encoder = nn.Linear(lang_dim, d_model)
        self.readout_tokens = nn.Parameter(torch.randn(n_readout, d_model))

        # Transformer
        self.transformer = BlockwiseCausalTransformer(d_model)

        # Diffusion action head
        self.action_head = DiffusionActionHead(
            d_model, action_dim, chunk_size,
        )

    def forward(self, obs_features, lang_features, actions):
        """Training: predict noise in action chunk."""
        B = obs_features.shape[0]

        task = self.lang_encoder(lang_features).unsqueeze(1)
        obs = self.obs_encoder(obs_features).unsqueeze(1)
        readout = self.readout_tokens.unsqueeze(0).expand(B, -1, -1)

        readout_out = self.transformer(task, obs, readout)

        # Diffusion training
        k = torch.randint(0, self.action_head.n_steps, (B,), device=obs_features.device)
        noise = torch.randn_like(actions)
        ab = self.action_head.alpha_bars[k].view(B, 1, 1)
        noisy = torch.sqrt(ab) * actions + torch.sqrt(1-ab) * noise

        pred_noise = self.action_head(readout_out, noisy, k)
        return ((pred_noise - noise)**2).mean()

# Demo
model = OctoStyle()
obs = torch.randn(8, 512)
lang = torch.randn(8, 256)
actions = torch.randn(8, 4, 7)
loss = model(obs, lang, actions)
print(f"Loss: {loss.item():.4f}")
```

---

## Exercise (45 min)

1. **Attention mask visualization:** Draw the blockwise causal mask for 3 task tokens, 5 obs tokens, 4 readout tokens. Explain why each block is masked or unmasked.

2. **Diffusion head vs token head:** On the same task, compare Octo's diffusion action head with RT-2's autoregressive tokens. Which handles multimodal actions better? Measure with a bimodal task.

3. **Fine-tuning efficiency:** Fine-tune the pre-trained Octo model with 50, 100, 500 demos. Compare full fine-tune vs head-only vs LoRA. Plot data efficiency curves.

4. **Multi-embodiment:** Add a second "robot" with different action dimensions. Train with shared backbone. Does transfer help?

---

## Key Takeaways

1. **Octo = 93M params, open-source, diffusion head** — practical alternative to RT-2
2. **Blockwise causal attention** controls information flow between task, observation, and readout
3. **Diffusion action head** handles multimodal actions without tokenization
4. **Multi-embodiment training** with per-robot action heads enables cross-robot transfer
5. **Fine-tuning in hours** makes Octo practical for new robots and tasks

---

## Connection to the Thread

> *Octo proved open-source robot foundation models work. Tomorrow, OpenVLA takes a different approach: start from a pre-trained VLM (like RT-2) but make it small enough to run on one GPU (7B parameters). It's the first open-source VLA that actually ships with weights, training code, and fine-tuning recipes.*

---

## Further Reading

- Octo Team (2024), "Octo: An Open-Source Generalist Robot Policy"
- [Octo GitHub](https://github.com/octo-models/octo)
- [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/)
