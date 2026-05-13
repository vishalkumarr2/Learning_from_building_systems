# Day 92: RT-1 — The First Robotics Transformer

> **Phase VII — VLAs: Architecture to Deployment | Week 14 | 2.5 hours**
> *"130K real robot episodes. 35M parameters. One transformer. The era of scaling robot learning begins." — Brohan et al., 2022*

## Navigation
- **Previous:** [Day 91: Phase VI Capstone Day 3](../week-13/day-91-phase6-capstone-3.md)
- **Next:** [Day 93: RT-2 — VLM Meets Robot Actions](day-93-rt2.md)
- **Week:** [Week 14 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 92.1 RT-1: Architecture

```
Image (300×300×3) ──→ ImageNet-pretrained EfficientNet-B3
                           │
                           ▼
                    Visual tokens (9×9×512)
                           │
                    FiLM-conditioned on language
                           │
Language ──→ USE encoder ──┤
                           │
                           ▼
              TokenLearner (8 learned tokens)
                           │
                           ▼
              Transformer (8 layers, 6 heads)
                           │
                           ▼
              Action tokens (11 dimensions × 256 bins)
```

### 92.2 Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Vision backbone | EfficientNet-B3 | ImageNet pre-trained, efficient |
| Language encoder | Universal Sentence Encoder | Frozen, good enough |
| Token reduction | TokenLearner | 81 → 8 tokens (10× compression) |
| Action space | 7-DOF arm + base + gripper | Delta EE + terminate |
| Discretization | 256 bins per dimension | 0.4mm resolution |
| Training data | 130K real episodes, 700+ tasks | 17 months of collection |

### 92.3 TokenLearner

The critical bottleneck that makes RT-1 practical:

$$\text{TokenLearner: } \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{S \times C}$$

where $S = 8$ learned tokens (down from $H \times W = 81$ spatial tokens).

```
Input: 9×9×512 feature map
  │
  ├──→ S attention maps (each 9×9, learned via 1×1 conv)
  │         │
  │         ▼
  │    spatial_attn_i(x) = sigmoid(conv(x))  # per-token attention
  │         │
  │         ▼
  └──→ token_i = Σ(spatial_attn_i ⊙ x)  # weighted spatial pooling
       │
       ▼
Output: 8 tokens × 512 dims
```

### 92.4 FiLM Conditioning

Language conditions vision via **Feature-wise Linear Modulation**:

$$\text{FiLM}(x, \gamma, \beta) = \gamma \odot x + \beta$$

where $\gamma, \beta$ are predicted from the language embedding.

### 92.5 Action Head

RT-1 predicts actions as discrete tokens (Day 83's uniform binning):

| Dimension | Range | Description |
|-----------|-------|-------------|
| $\Delta x$ | [-0.05, 0.05]m | End-effector x delta |
| $\Delta y$ | [-0.05, 0.05]m | End-effector y delta |
| $\Delta z$ | [-0.05, 0.05]m | End-effector z delta |
| $\Delta\text{roll}$ | [-π/4, π/4] | Rotation |
| $\Delta\text{pitch}$ | [-π/4, π/4] | Rotation |
| $\Delta\text{yaw}$ | [-π/4, π/4] | Rotation |
| gripper | {open, close} | Binary |
| base $\Delta x$ | [-0.05, 0.05]m | Mobile base |
| base $\Delta y$ | [-0.05, 0.05]m | Mobile base |
| base $\Delta\theta$ | [-π/4, π/4] | Base rotation |
| terminate | {continue, stop} | Episode termination |

Each predicted independently via separate softmax heads (parallel decoding).

### 92.6 Results

| Metric | RT-1 | BC-Z (prior SOTA) | Gato |
|--------|------|-------------------|------|
| Seen tasks | **97%** | 89% | 57% |
| Unseen tasks | **76%** | 53% | 33% |
| Unseen objects | **56%** | 25% | - |
| Parameters | 35M | 120M | 1.2B |

**Key insight:** RT-1 shows that a relatively small transformer with lots of real data can generalize to novel situations.

---

## Implementation (60 min)

### Simplified RT-1

```python
import torch
import torch.nn as nn
import torchvision.models as models

class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        # x: (B, H, W, C) or (B, C)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        if x.dim() == 4:
            gamma = gamma.unsqueeze(1).unsqueeze(1)
            beta = beta.unsqueeze(1).unsqueeze(1)
        return gamma * x + beta

class TokenLearner(nn.Module):
    """Learn S tokens from spatial features."""
    def __init__(self, in_channels, n_tokens=8):
        super().__init__()
        self.n_tokens = n_tokens
        self.attention_maps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_tokens)
        ])

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = []
        for attn in self.attention_maps:
            mask = attn(x)  # (B, 1, H, W)
            token = (x * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
            tokens.append(token)
        return torch.stack(tokens, dim=1)  # (B, S, C)

class RT1(nn.Module):
    def __init__(self, n_actions=11, n_bins=256, lang_dim=512,
                 n_tokens=8, d_model=512, n_heads=8, n_layers=8):
        super().__init__()
        # Vision backbone (frozen EfficientNet → simplified to ResNet-18)
        resnet = models.resnet18(pretrained=True)
        self.vision = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool + fc
        vis_dim = 512

        # Language conditioning
        self.film = FiLM(lang_dim, vis_dim)

        # Token reduction
        self.token_learner = TokenLearner(vis_dim, n_tokens)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Action heads (parallel, one per dimension)
        self.action_heads = nn.ModuleList([
            nn.Linear(d_model, n_bins) for _ in range(n_actions)
        ])

    def forward(self, image, lang_embed):
        """
        image: (B, 3, 224, 224)
        lang_embed: (B, lang_dim)
        Returns: list of (B, n_bins) logits per action dimension
        """
        # Visual features
        vis = self.vision(image)  # (B, 512, 7, 7)

        # Language conditioning via FiLM
        vis = self.film(vis, lang_embed)

        # TokenLearner
        tokens = self.token_learner(vis)  # (B, 8, 512)

        # Transformer
        out = self.transformer(tokens)  # (B, 8, 512)

        # Pool and predict actions
        pooled = out.mean(dim=1)  # (B, 512)
        action_logits = [head(pooled) for head in self.action_heads]

        return action_logits

    def predict(self, image, lang_embed):
        """Greedy action prediction."""
        logits = self.forward(image, lang_embed)
        actions = [l.argmax(dim=-1) for l in logits]
        return torch.stack(actions, dim=-1)  # (B, n_actions)

# Demo
model = RT1(n_actions=11, n_bins=256)
image = torch.randn(2, 3, 224, 224)
lang = torch.randn(2, 512)
logits = model(image, lang)
print(f"Action logits: {len(logits)} heads, each {logits[0].shape}")
actions = model.predict(image, lang)
print(f"Predicted actions: {actions.shape}")  # (2, 11)
```

---

## Exercise (45 min)

1. **TokenLearner analysis:** Visualize the 8 attention maps on a sample image. What spatial regions does each token attend to?

2. **Scaling analysis:** Train RT-1 with {100, 1K, 10K, 100K} simulated episodes. Plot success rate vs data size. Does it follow a power law?

3. **Parallel vs autoregressive:** Compare RT-1's parallel action decoding (independent heads) with autoregressive (predict $a_1$, then $a_2|a_1$, etc.). Measure accuracy and speed.

4. **FiLM ablation:** Remove FiLM conditioning and instead concatenate language to tokens. Compare performance on multi-task benchmark.

---

## Key Takeaways

1. **RT-1 = EfficientNet + TokenLearner + Transformer + discrete actions** — elegant simplicity
2. **TokenLearner** reduces 81 spatial tokens to 8, making real-time inference feasible
3. **FiLM conditioning** fuses language and vision efficiently
4. **130K real demos** enable generalization to unseen tasks and objects
5. **35M parameters** outperforms 1.2B parameter Gato — data and architecture matter more than scale

---

## Connection to the Thread

> *RT-1 shows that transformers work for robotics. But it doesn't use a pre-trained language model — the language encoder is frozen USE, not a VLM. Tomorrow, RT-2 asks: what if we take a 55B parameter VLM (PaLI-X) and fine-tune it to output action tokens? That's where VLAs truly begin.*

---

## Further Reading

- Brohan et al. (2022), "RT-1: Robotics Transformer for Real-World Control at Scale"
- Ryoo et al. (2021), "TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?"
- Perez et al. (2018), "FiLM: Visual Reasoning with a General Conditioning Layer"
