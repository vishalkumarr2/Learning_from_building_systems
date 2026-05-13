# Day 99: GR-2 Deep Dive — Video Generation for Robot Control

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2.5 hours**
> *"Pre-train on 38M video clips, fine-tune on robot data. The video model understands physics before it ever sees a robot." — GR-2*

## Navigation
- **Previous:** [Day 98: GR-2 & GROOT N1 Overview](../week-14/day-98-groot-palme.md)
- **Next:** [Day 100: VLA Training Recipes](day-100-training-recipes.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 99.1 GR-2 Training Pipeline

```
Phase 1: Video pre-training (38M clips, Internet)
  ├── Learns object permanence, gravity, contact
  ├── Learns visual dynamics (how scenes change)
  └── No action labels needed

Phase 2: Robot video fine-tuning (100K+ episodes)
  ├── Same video prediction objective
  ├── Robot-specific visual dynamics
  └── Still no action labels

Phase 3: Action head training
  ├── Add action prediction alongside video
  ├── Joint training: predict next frame + action
  └── Action supervision from demonstrations
```

### 99.2 Video Diffusion Transformer (DiT)

GR-2 uses a Diffusion Transformer for video generation:

$$p_\theta(v_{1:T} | v_0, l) = \prod_{k=K}^{1} p_\theta(v^{(k-1)} | v^{(k)}, v_0, l)$$

where $v^{(k)}$ are noisy video frames at diffusion step $k$, $v_0$ is the conditioning frame, and $l$ is the language instruction.

```
Architecture:
  Spatial patches → temporal patches → DiT blocks
  
  Frame (256×256×3) → 16×16 patches → 256 spatial tokens
  T frames → T×256 = T*256 spatiotemporal tokens
  
  DiT Block:
    LayerNorm → Self-Attention → LayerNorm → FFN
    + AdaLN conditioning on timestep k and language l
```

### 99.3 Action Conditioning

GR-2 predicts actions alongside video:

$$\mathcal{L} = \underbrace{\mathcal{L}_\text{video}}_{\text{reconstruct frames}} + \lambda \underbrace{\mathcal{L}_\text{action}}_{\text{predict actions}}$$

The action head reads from DiT hidden states:

```
DiT hidden states ──→ temporal pooling ──→ MLP ──→ action
     (shared)              (per step)
```

### 99.4 What Video Pre-training Teaches

```python
# Emergent capabilities from video pre-training:

capability_1 = "Object tracking"
# Model learns to predict where objects move
# → Robot tracks target object through occlusion

capability_2 = "Contact prediction"
# Model learns that contact changes motion
# → Robot predicts interaction outcomes

capability_3 = "Gravity and support"
# Objects fall when unsupported
# → Robot plans stable placements

capability_4 = "Deformable object dynamics"
# Cloth, rope, liquid behavior
# → Robot handles non-rigid manipulation

capability_5 = "Tool use patterns"
# Human tool interactions transfer
# → Robot learns affordances from human videos
```

---

## Implementation (60 min)

### DiT-Based Video Predictor

```python
import torch
import torch.nn as nn
import math

class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on timestep and language."""
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )

    def forward(self, x, cond):
        params = self.proj(cond)
        if params.dim() == 2:
            params = params.unsqueeze(1)
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift

class DiTBlock(nn.Module):
    """Diffusion Transformer block."""
    def __init__(self, d_model, n_heads, cond_dim):
        super().__init__()
        self.norm1 = AdaLayerNorm(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, cond):
        # Self-attention with AdaLN
        normed = self.norm1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with AdaLN
        x = x + self.ffn(self.norm2(x, cond))
        return x

class VideoActionDiT(nn.Module):
    """Simplified GR-2-style video + action DiT."""

    def __init__(self, img_size=64, patch_size=8, n_future=4,
                 d_model=256, n_heads=8, n_layers=6,
                 lang_dim=128, action_dim=7):
        super().__init__()
        self.n_future = n_future
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches * (n_future + 1), d_model) * 0.02)

        # Conditioning: timestep + language
        cond_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, cond_dim),
        )
        self.lang_embed = nn.Linear(lang_dim, cond_dim)
        self.cond_merge = nn.Linear(cond_dim * 2, cond_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, cond_dim)
            for _ in range(n_layers)
        ])

        # Video reconstruction head
        self.video_head = nn.Linear(d_model, patch_dim)

        # Action head (reads from DiT features)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.n_patches = n_patches
        self.img_size = img_size

    def patchify(self, imgs):
        """Convert images to patches."""
        B, T, C, H, W = imgs.shape
        p = self.patch_size
        patches = imgs.reshape(B, T, C, H//p, p, W//p, p)
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6)
        patches = patches.reshape(B, T * (H//p) * (W//p), C*p*p)
        return patches

    def unpatchify(self, patches, T):
        """Convert patches back to images."""
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        patches = patches.reshape(B, T, h, w, 3, p, p)
        imgs = patches.permute(0, 1, 4, 2, 5, 3, 6)
        imgs = imgs.reshape(B, T, 3, self.img_size, self.img_size)
        return imgs

    def forward(self, cond_frame, noisy_futures, timestep, lang,
                target_actions=None):
        """
        cond_frame: (B, 3, H, W) — conditioning frame
        noisy_futures: (B, n_future, 3, H, W) — noisy future frames
        timestep: (B,) — diffusion timestep
        lang: (B, lang_dim) — language embedding
        """
        B = cond_frame.shape[0]

        # Combine frames
        all_frames = torch.cat([cond_frame.unsqueeze(1), noisy_futures], dim=1)
        patches = self.patchify(all_frames)
        x = self.patch_embed(patches) + self.pos_embed[:, :patches.shape[1]]

        # Conditioning
        t_emb = self.time_embed(timestep.float().unsqueeze(1))
        l_emb = self.lang_embed(lang)
        cond = self.cond_merge(torch.cat([t_emb, l_emb], dim=-1))

        # DiT blocks
        for block in self.blocks:
            x = block(x, cond)

        # Video prediction: noise for future patches only
        future_patches = x[:, self.n_patches:]  # Skip conditioning frame
        noise_pred = self.video_head(future_patches)

        # Action prediction: pool temporal features per frame
        actions_pred = None
        if target_actions is not None:
            frame_features = []
            for t in range(self.n_future):
                start = self.n_patches * (t + 1)
                end = start + self.n_patches
                frame_feat = x[:, start:end].mean(dim=1)
                frame_features.append(frame_feat)
            frame_features = torch.stack(frame_features, dim=1)
            actions_pred = self.action_head(frame_features)

        return noise_pred, actions_pred

# Demo
model = VideoActionDiT(img_size=64, n_future=4, action_dim=7)
cond = torch.randn(2, 3, 64, 64)
noisy = torch.randn(2, 4, 3, 64, 64)
t = torch.randint(0, 100, (2,))
lang = torch.randn(2, 128)
gt_actions = torch.randn(2, 4, 7)

noise_pred, actions_pred = model(cond, noisy, t, lang, gt_actions)
print(f"Noise prediction: {noise_pred.shape}")
print(f"Action prediction: {actions_pred.shape}")
```

---

## Exercise (45 min)

1. **Video quality vs action quality:** Train the model for 500 epochs. Plot video reconstruction loss and action prediction loss on the same graph. Do they correlate?

2. **Pre-training transfer:** Pre-train on non-robot video (e.g., synthetic bouncing balls). Fine-tune on robot data with 50, 200, 1000 demos. Does video pre-training improve sample efficiency?

3. **Temporal attention:** Visualize attention patterns in the DiT blocks. Do tokens attend more within their frame (spatial) or across frames (temporal)?

4. **Action head comparison:** Compare extracting actions from DiT hidden states vs inverse dynamics on generated frames. Which is more accurate?

---

## Key Takeaways

1. **GR-2 pre-trains on 38M video clips** — Internet videos teach physics implicitly
2. **DiT architecture** handles spatiotemporal video generation effectively
3. **Joint video + action training** shares representations for both objectives
4. **Video pre-training transfers** to robot manipulation through visual dynamics
5. **Trade-off: computational cost** — generating full video frames is expensive

---

## Connection to the Thread

> *Tomorrow: VLA Training Recipes. We've seen 7 different VLA architectures. Now we systematize: what data recipes, training schedules, and hyperparameters actually produce the best VLAs? We distill practical training wisdom from RT-2, OpenVLA, Octo, and π₀.*

---

## Further Reading

- Cheang et al. (2024), "GR-2: A Generative Video-Language-Action Model"
- Peebles & Xie (2023), "Scalable Diffusion Models with Transformers" (DiT)
- Ho et al. (2022), "Video Diffusion Models"
