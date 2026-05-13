# Day 96: π₀ — Flow Matching Meets VLAs

> **Phase VII — VLAs: Architecture to Deployment | Week 14 | 2.5 hours**
> *"Pre-trained VLM backbone + flow matching action expert. π₀ doesn't tokenize actions — it generates them as continuous trajectories." — Black et al., 2024*

## Navigation
- **Previous:** [Day 95: OpenVLA](day-95-openvla.md)
- **Next:** [Day 97: π₀.5 — Hybrid VLA](day-97-pi0-5.md)
- **Week:** [Week 14 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 96.1 Why π₀ Matters

The action representation debate:

```
Tokenized actions (RT-2, OpenVLA):
  + Compatible with LM head
  + Simple training (cross-entropy)
  - Discretization artifacts
  - Autoregressive → slow

Diffusion actions (Octo):
  + Continuous output
  + Handles multimodality
  - No pre-trained VLM reasoning
  - Small model (93M)

π₀ = Pre-trained VLM + Flow Matching action head
  + VLM reasoning (3B params)
  + Continuous, multimodal actions
  + Fast inference (flow matching)
  + Dexterous manipulation SOTA
```

### 96.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                          π₀                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Image ──→ SigLIP ──→ visual tokens                      │
│  Language ──→ Gemma tokenizer ──→ text tokens             │
│  Proprioception ──→ MLP ──→ proprio tokens                │
│                                                          │
│  ┌───────────────────────────────────────────────┐       │
│  │          PaliGemma 2B (frozen VLM)            │       │
│  │          + trainable expert layers             │       │
│  └───────────────────────────────────────────┬───┘       │
│                                              │            │
│  ┌───────────────────────────────────────────▼───┐       │
│  │      Flow Matching Action Expert              │       │
│  │                                                │       │
│  │  Input:  VLM features + noisy action chunk     │       │
│  │  Process: predict velocity field v_θ(x_t, t)   │       │
│  │  Output:  action chunk (H=50 steps @ 50Hz)     │       │
│  └───────────────────────────────────────────────┘       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 96.3 Flow Matching Action Expert

Recall flow matching from Day 77. π₀ applies it to action generation:

$$v_\theta(x_t, t) = \text{ActionExpert}(\text{VLM\_features}, x_t, t)$$

**Training:**
$$\mathcal{L} = \mathbb{E}_{t, a_0, a_1}\left[\|v_\theta(a_t, t) - (a_1 - a_0)\|^2\right]$$

where $a_t = (1-t)a_0 + t \cdot a_1$ is the interpolated action between noise $a_0$ and data $a_1$.

**Inference (ODE):**
$$a_1 = a_0 + \int_0^1 v_\theta(a_t, t) \, dt$$

In practice, 10 Euler steps:
$$a_{t+\Delta t} = a_t + \Delta t \cdot v_\theta(a_t, t), \quad \Delta t = 0.1$$

### 96.4 Expert Layers

π₀ doesn't fine-tune the full VLM. Instead, it adds **expert layers** that interleave with VLM layers:

```
VLM Layer 1 (frozen) → Expert Layer 1 (trainable)
VLM Layer 2 (frozen) → Expert Layer 2 (trainable)
...
VLM Layer N (frozen) → Expert Layer N (trainable)
```

Expert layers see both VLM features and action-specific information. This preserves VLM capabilities while learning robot-specific features.

### 96.5 Action Chunk

π₀ predicts long action chunks for dexterous manipulation:

- **Chunk size:** 50 timesteps at 50 Hz (1 second horizon)
- **Action dims:** 24 per step (bimanual: 2 × 6 DoF arm + 2 × 6 DoF hand)
- **Total output:** $50 \times 24 = 1200$ continuous values per prediction

This is far too large for tokenization (would need 1200 × 256 = 307K tokens per step).

### 96.6 Results

| Task | π₀ | OpenVLA | Octo |
|------|-----|---------|------|
| Laundry folding | **80%** | N/A | N/A |
| Table clearing | **95%** | 72% | 65% |
| Box assembly | **70%** | N/A | N/A |
| Dexterous manipulation | **SOTA** | Can't (no hand control) | Limited |

---

## Implementation (60 min)

### π₀-Style Flow Matching VLA

```python
import torch
import torch.nn as nn

class FlowMatchingActionExpert(nn.Module):
    """Action expert using flow matching."""

    def __init__(self, context_dim=512, action_dim=7, chunk_size=16,
                 hidden=512, n_layers=4):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        flat_action = action_dim * chunk_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 128),
        )

        # Context projection
        self.context_proj = nn.Linear(context_dim, hidden)

        # Velocity prediction network
        layers = []
        in_dim = flat_action + hidden + 128  # action + context + time
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, flat_action))
        self.velocity_net = nn.Sequential(*layers)

    def forward(self, context, noisy_action, t):
        """Predict velocity field."""
        B = context.shape[0]
        t_emb = self.time_mlp(t.view(B, 1))
        ctx = self.context_proj(context)
        flat_action = noisy_action.flatten(1)
        inp = torch.cat([flat_action, ctx, t_emb], dim=-1)
        velocity = self.velocity_net(inp)
        return velocity.view(B, self.chunk_size, self.action_dim)

    def training_loss(self, context, target_actions):
        """Flow matching loss."""
        B = target_actions.shape[0]
        # Sample time
        t = torch.rand(B, device=context.device)
        # Sample noise (source distribution)
        noise = torch.randn_like(target_actions)
        # Interpolate
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * noise + t_expand * target_actions
        # Target velocity: data - noise
        target_v = target_actions - noise
        # Predict
        pred_v = self.forward(context, x_t, t)
        return ((pred_v - target_v)**2).mean()

    @torch.no_grad()
    def sample(self, context, n_steps=10):
        """Generate actions via ODE integration."""
        B = context.shape[0]
        device = context.device
        # Start from noise
        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(context, x, t)
            x = x + dt * v
        return x

class Pi0Style(nn.Module):
    """Simplified π₀ architecture."""

    def __init__(self, obs_dim=512, lang_dim=256, action_dim=7,
                 d_model=512, chunk_size=16):
        super().__init__()
        # Simplified VLM backbone
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, d_model),
        )
        self.lang_encoder = nn.Sequential(
            nn.Embedding(10000, 256),
        )
        self.proprio_encoder = nn.Linear(action_dim, d_model)

        # VLM layers (frozen in real π₀)
        vlm_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True,
        )
        self.vlm = nn.TransformerEncoder(vlm_layer, num_layers=4)

        # Expert layers (trainable)
        expert_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True,
        )
        self.expert = nn.TransformerEncoder(expert_layer, num_layers=4)

        # Flow matching action head
        self.action_head = FlowMatchingActionExpert(
            context_dim=d_model, action_dim=action_dim,
            chunk_size=chunk_size,
        )

    def encode(self, image, text_tokens, proprio):
        vis = self.vision_encoder(image).unsqueeze(1)
        lang = self.lang_encoder(text_tokens).mean(1, keepdim=True)
        prop = self.proprio_encoder(proprio).unsqueeze(1)
        tokens = torch.cat([vis, lang, prop], dim=1)

        # VLM (frozen) + Expert (trainable)
        vlm_out = self.vlm(tokens)
        expert_out = self.expert(vlm_out)
        return expert_out.mean(dim=1)  # Pool to (B, D)

    def forward(self, image, text_tokens, proprio, actions):
        context = self.encode(image, text_tokens, proprio)
        return self.action_head.training_loss(context, actions)

    @torch.no_grad()
    def predict(self, image, text_tokens, proprio, n_steps=10):
        context = self.encode(image, text_tokens, proprio)
        return self.action_head.sample(context, n_steps)

# Demo
model = Pi0Style()
img = torch.randn(4, 3, 224, 224)
txt = torch.randint(0, 1000, (4, 10))
prop = torch.randn(4, 7)
actions = torch.randn(4, 16, 7)

loss = model(img, txt, prop, actions)
print(f"Loss: {loss.item():.4f}")

pred = model.predict(img, txt, prop)
print(f"Predicted actions: {pred.shape}")  # (4, 16, 7)
```

---

## Exercise (45 min)

1. **Flow vs diffusion speed:** Compare flow matching (10 Euler steps) with DDIM (10 steps) for action generation. Measure quality (MSE to ground truth) and speed.

2. **Expert layer analysis:** Freeze the VLM and train only expert layers. Then compare with full fine-tuning. What fraction of VLM knowledge transfers to actions?

3. **Chunk size analysis:** Train π₀ with chunk sizes {4, 16, 50}. How does chunk size affect dexterous task performance? At what point do long horizons help vs hurt?

4. **Bimodal demonstration:** Create a dataset where the optimal action is bimodal (reach left or right). Show that flow matching correctly samples both modes while MSE regression averages.

---

## Key Takeaways

1. **π₀ = PaliGemma + flow matching action expert** — best of both worlds
2. **Flow matching** generates continuous action chunks without tokenization
3. **Expert layers** preserve VLM reasoning while adding robot-specific features
4. **Long action chunks** (50 steps) enable dexterous manipulation
5. **SOTA on bimanual tasks** — laundry folding, assembly, dexterous manipulation

---

## Connection to the Thread

> *π₀ showed that flow matching + VLMs is the new frontier. Tomorrow, π₀.5 takes this further: it adds **language-based high-level planning** on top of π₀'s flow matching action generation. The VLM reasons about what to do, then the flow matching head figures out how to do it. This hybrid approach is where the field is heading.*

---

## Further Reading

- Black et al. (2024), "π₀: A Vision-Language-Action Flow Model for General Robot Control"
- [Physical Intelligence](https://www.physicalintelligence.company/)
- Lipman et al. (2023), "Flow Matching for Generative Modeling" (background)
