# Day 79: ACT — Action Chunking with Transformers

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 12 | 2.5 hours**
> *"Predict not one action, but a whole chunk of future actions — and use a CVAE to capture the multimodality." — Tony Zhao et al.*

## Navigation
- **Previous:** [Day 78: Imitation Learning](day-78-imitation-learning.md)
- **Next:** [Day 80: Decision Transformer](day-80-decision-transformer.md)
- **Week:** [Week 12 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/13-imitation-learning.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 79.1 ACT Architecture Overview

ACT (Action Chunking with Transformers) addresses two key challenges:
1. **Temporal consistency** → action chunking (predict $H$ actions at once)
2. **Multimodal actions** → CVAE (sample from a learned latent distribution)

```
Training:
  Observation o_t ─────────────────────────────────┐
  Future actions [a_t...a_{t+H}] → CVAE Encoder → z ← latent code
                                                    │
                                     ┌──────────────▼──────────────┐
                                     │  Transformer Decoder        │
                                     │  Input: o_t features + z    │
                                     │  Output: [â_t...â_{t+H}]   │
                                     └─────────────────────────────┘

Inference:
  Observation o_t ──────────────────────────────────┐
  z ~ N(0, I) ← sample from prior                   │
                                     ┌──────────────▼──────────────┐
                                     │  Transformer Decoder        │
                                     │  Output: [â_t...â_{t+H}]   │
                                     └─────────────────────────────┘
```

### 79.2 CVAE: Conditional Variational Autoencoder

The CVAE encoder maps observations + actions to a latent distribution:

$$q_\phi(z | o, a_{t:t+H}) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$$

Training loss combines reconstruction and KL:

$$\mathcal{L} = \underbrace{\|a_{t:t+H} - \hat{a}_{t:t+H}\|^2}_{\text{action prediction}} + \beta \cdot \underbrace{D_\text{KL}(q_\phi(z|o,a) \| \mathcal{N}(0,I))}_{\text{regularize latent}}$$

At inference, sample $z \sim \mathcal{N}(0, I)$ — different $z$ → different valid action sequences.

### 79.3 Temporal Ensembling

When executing chunked actions, consecutive predictions overlap:

```
t=0: predict [a₀, a₁, a₂, a₃]
t=1: predict [a₁', a₂', a₃', a₄']
t=2: predict [a₂'', a₃'', a₄'', a₅'']

For action at t=2: average a₂, a₂', a₂'' with exponential weighting
```

Temporal ensembling smooths the executed trajectory:

$$a_t^\text{exec} = \sum_{k} w_k \cdot a_t^{(k)}, \quad w_k \propto e^{-\lambda k}$$

### 79.4 Observation Processing

ACT processes multiple observation modalities:

| Input | Encoding | Dimension |
|-------|----------|-----------|
| Robot joint positions | MLP | $d_\text{joint}$ |
| Wrist camera image | ResNet-18 → spatial features | $h \times w \times d$ |
| External camera image | ResNet-18 → spatial features | $h \times w \times d$ |

All modalities are projected to the same dimension and fed as tokens to the transformer.

### 79.5 Key Design Choices

| Choice | ACT Decision | Rationale |
|--------|-------------|-----------|
| Chunk size $H$ | 100 steps | Cover full manipulation trajectories |
| Latent dim | 32 | Enough for action variety |
| KL weight $\beta$ | 10 | Prevent posterior collapse |
| Backbone | ResNet-18 | Fast, sufficient for tabletop |
| Action space | Joint positions (absolute) | More precise than velocities |

---

## Implementation (60 min)

### Simplified ACT

```python
import torch
import torch.nn as nn

class CVAEEncoder(nn.Module):
    """Encode (observation, action_chunk) → latent z."""
    def __init__(self, obs_dim, act_dim, chunk_size, latent_dim=32, hidden=256):
        super().__init__()
        input_dim = obs_dim + act_dim * chunk_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.log_var = nn.Linear(hidden, latent_dim)

    def forward(self, obs, actions):
        # actions: (B, H, act_dim)
        x = torch.cat([obs, actions.flatten(1)], dim=-1)
        h = self.net(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

class ACTDecoder(nn.Module):
    """Transformer decoder: (observation features, z) → action chunk."""
    def __init__(self, obs_dim, act_dim, chunk_size, latent_dim=32,
                 d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.act_dim = act_dim

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.z_proj = nn.Linear(latent_dim, d_model)

        # Learnable action queries (one per timestep)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(self, obs, z):
        B = obs.shape[0]
        # Context: observation + latent
        obs_token = self.obs_proj(obs).unsqueeze(1)  # (B, 1, d)
        z_token = self.z_proj(z).unsqueeze(1)        # (B, 1, d)
        memory = torch.cat([obs_token, z_token], dim=1)  # (B, 2, d)

        # Action queries
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)

        # Decode
        decoded = self.transformer(queries, memory)
        actions = self.action_head(decoded)  # (B, H, act_dim)
        return actions

class ACT(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_size=16, latent_dim=32):
        super().__init__()
        self.encoder = CVAEEncoder(obs_dim, act_dim, chunk_size, latent_dim)
        self.decoder = ACTDecoder(obs_dim, act_dim, chunk_size, latent_dim)
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

    def forward(self, obs, actions):
        """Training forward pass."""
        mu, log_var = self.encoder(obs, actions)
        z = self.encoder.reparameterize(mu, log_var)
        pred_actions = self.decoder(obs, z)

        # Losses
        recon_loss = ((pred_actions - actions) ** 2).mean()
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        return recon_loss, kl_loss

    @torch.no_grad()
    def predict(self, obs):
        """Inference: sample z from prior."""
        z = torch.randn(obs.shape[0], self.latent_dim, device=obs.device)
        return self.decoder(obs, z)

# Training loop
model = ACT(obs_dim=7, act_dim=7, chunk_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    # Dummy data: replace with real robot demonstrations
    obs = torch.randn(32, 7)
    actions = torch.randn(32, 16, 7)

    recon_loss, kl_loss = model(obs, actions)
    loss = recon_loss + 10.0 * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: recon={recon_loss:.4f}, kl={kl_loss:.4f}")
```

---

## Exercise (45 min)

1. **KL weight sweep:** Train with $\beta \in \{0.1, 1, 10, 100\}$. At low $\beta$, the CVAE ignores $z$. At high $\beta$, predictions become blurry. Find the sweet spot.

2. **Temporal ensembling:** Implement the exponential averaging scheme. Compare trajectory smoothness with and without ensembling.

3. **Chunk size ablation:** Train with $H \in \{1, 4, 16, 64\}$. Measure prediction accuracy vs temporal consistency.

4. **Multimodality test:** Create a toy dataset where the same observation leads to two different action sequences. Show that ACT (with CVAE) captures both modes while vanilla BC averages them.

---

## Key Takeaways

1. **ACT = CVAE + Transformer + Action Chunking** — three ideas that synergize
2. **CVAE handles multimodality** — different $z$ samples → different valid strategies
3. **Chunking reduces compounding error** — predict $H$ actions, not one
4. **Temporal ensembling** smooths transitions between consecutive chunks
5. **ACT works with low-cost hardware** — bimanual manipulation with $\sim$\$5K setup

---

## Connection to the Thread

> *ACT uses a CVAE for multimodal actions. Decision Transformer (tomorrow) takes a completely different approach: frame IL as autoregressive sequence prediction, just like GPT. Diffusion Policy (Day 81) replaces the CVAE with a diffusion model for richer action distributions. Three architectures, one goal: generate robot actions from demonstrations.*

---

## Further Reading

- Zhao et al. (2023), "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT)
- Sohn et al. (2015), "Learning Structured Output Representation using Deep Conditional Generative Models" (CVAE theory)
- [LeRobot ACT implementation](https://github.com/huggingface/lerobot)
