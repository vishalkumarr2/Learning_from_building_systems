# Day 53: Video Understanding Day 1

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"Video is just images with a time axis. The transformer handles it the same way — more tokens, same mechanism." — Bertasius et al., 2021*

## Navigation
- **Previous:** [Day 52: Point Clouds & 3D Scenes](day-52-point-clouds-3d.md)
- **Next:** [Day 54: Video Understanding Day 2](day-54-video-understanding-2.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/09-3d-video-detection.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### Video as a Sequence of Frame-Patch Tokens

A video clip with $T$ frames and $N$ patches per frame produces $T \times N$ tokens:

```
Frame 1:  [p₁¹, p₂¹, ..., pₙ¹]
Frame 2:  [p₁², p₂², ..., pₙ²]
  ...
Frame T:  [p₁ᵀ, p₂ᵀ, ..., pₙᵀ]

Total: T × N tokens → self-attention on ALL of them?
       8 frames × 196 patches = 1,568 tokens
       32 frames × 196 patches = 6,272 tokens  ← expensive!
```

### Attention Strategies for Video

Full space-time attention ($O((TN)^2)$) is impractical. Solutions:

```
Strategy 1: Divided Space-Time (TimeSformer)
┌─────────────────────────────────────────┐
│ Block 1: Temporal attention             │
│   Each patch attends to same patch      │
│   across all frames                     │
│   Cost: O(T² × N)                       │
│                                         │
│ Block 2: Spatial attention              │
│   Each patch attends to all patches     │
│   within same frame                     │
│   Cost: O(N² × T)                       │
│                                         │
│ Total: O(T²N + N²T) vs O(T²N²)        │
└─────────────────────────────────────────┘

Strategy 2: Joint Space-Time (ViViT)
┌─────────────────────────────────────────┐
│ Tubelet embedding: 3D patches           │
│   (t×h×w) → tokens from space-time      │
│                                         │
│ Factorized encoder:                     │
│   1. Spatial transformer per frame      │
│   2. Temporal transformer across frames │
└─────────────────────────────────────────┘
```

### TimeSformer: Divided Space-Time Attention

TimeSformer (2021) alternates between temporal and spatial attention in each block:

$$\text{Block}(x) = \text{SpatialAttn}(\text{TemporalAttn}(x))$$

**Temporal attention:** For patch at position $(t, i)$, attend to $\{(1, i), (2, i), \ldots, (T, i)\}$ — same spatial position, all frames.

**Spatial attention:** For patch at position $(t, i)$, attend to $\{(t, 1), (t, 2), \ldots, (t, N)\}$ — same frame, all positions.

### Positional Embeddings for Video

Video needs both spatial and temporal position information:

$$\mathbf{E}_{\text{pos}}^{(t,i)} = \mathbf{E}_{\text{spatial}}^{(i)} + \mathbf{E}_{\text{temporal}}^{(t)}$$

Separable positional embeddings work as well as joint ones and are more parameter-efficient.

### Why Video Matters for VLAs

Robot actions unfold over time. Understanding video means understanding:
- **Temporal dynamics:** How do objects move?
- **Causality:** What action caused what effect?
- **Anticipation:** What will happen next?

---

## Implementation (60 min)

### Divided Space-Time Attention

```python
import torch
import torch.nn as nn
from einops import rearrange


class TemporalAttention(nn.Module):
    """Attention across time for each spatial position."""
    
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, T, N):
        """
        Args:
            x: (B, T*N, D) flattened video tokens
            T: number of frames
            N: patches per frame
        """
        B, _, D = x.shape
        residual = x
        x = self.norm(x)
        
        # Reshape: (B, T, N, D) → (B*N, T, D) — group by spatial position
        x = rearrange(x, 'b (t n) d -> (b n) t d', t=T, n=N)
        
        BN, T_len, _ = x.shape
        qkv = self.qkv(x).reshape(BN, T_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(BN, T_len, D)
        out = self.proj(out)
        
        # Reshape back: (B*N, T, D) → (B, T*N, D)
        out = rearrange(out, '(b n) t d -> b (t n) d', b=B, n=N)
        
        return residual + out


class SpatialAttention(nn.Module):
    """Attention within each frame."""
    
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, T, N):
        B, _, D = x.shape
        residual = x
        x = self.norm(x)
        
        # Reshape: (B, T, N, D) → (B*T, N, D) — group by frame
        x = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N)
        
        BT, N_len, _ = x.shape
        qkv = self.qkv(x).reshape(BT, N_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(BT, N_len, D)
        out = self.proj(out)
        
        out = rearrange(out, '(b t) n d -> b (t n) d', b=B, t=T)
        
        return residual + out


class TimeSformerBlock(nn.Module):
    """Divided space-time attention block."""
    
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.temporal_attn = TemporalAttention(dim, n_heads)
        self.spatial_attn = SpatialAttention(dim, n_heads)
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x, T, N):
        x = self.temporal_attn(x, T, N)
        x = self.spatial_attn(x, T, N)
        x = x + self.mlp(self.norm(x))
        return x


class TimeSformer(nn.Module):
    """Simplified TimeSformer for video classification."""
    
    def __init__(self, img_size=224, patch_size=16, n_frames=8,
                 n_classes=400, embed_dim=768, depth=12, n_heads=12):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.n_frames = n_frames
        self.n_patches = n_patches
        
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, n_frames, embed_dim))
        
        self.blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, n_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W)
        """
        B, T, C, H, W = video.shape
        P = int((H * W / self.n_patches) ** 0.5)
        
        # Patchify each frame
        x = rearrange(video, 'b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)',
                       p1=P, p2=P)
        x = self.patch_embed(x)  # (B, T*N, D)
        
        # Add separable positional embeddings
        spatial = self.spatial_pos.repeat(1, T, 1)
        temporal = self.temporal_pos.repeat_interleave(self.n_patches, dim=1)
        x = x + spatial + temporal
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, T, self.n_patches)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pool
        return self.head(x)
```

---

## Exercise (45 min)

1. **Complexity comparison:** Calculate FLOPs for a video with T=8 frames, N=196 patches, D=768:
   - Full space-time attention: $O((TN)^2 D)$
   - Divided space-time: $O((T^2 N + N^2 T) D)$
   - What's the speedup factor?

2. **Temporal attention visualization:** Run TimeSformer on a short video clip. Extract temporal attention maps — which frames attend most strongly to each other? Do temporally adjacent frames have higher attention?

3. **Frame sampling strategy:** Compare uniform sampling (every $k$-th frame) vs random sampling of $T$ frames from a video. Does sampling strategy affect classification accuracy?

---

## Key Takeaways

1. **Video = spatial + temporal tokens.** Same transformer, more tokens, factored attention
2. **Divided attention saves compute.** Separate temporal and spatial attention reduces $O((TN)^2)$ to $O(T^2N + N^2T)$
3. **Separable positions.** Spatial + temporal position embeddings work as well as joint
4. **Same architecture.** TimeSformer reuses the ViT backbone — just adds temporal attention
5. **Robotics relevance.** Understanding video dynamics is key for action prediction

## Connection to the Thread

> *Video extends the transformer's domain from static images to temporal sequences. Tomorrow: self-supervised video learning with VideoMAE and video-text pretraining — the video counterparts of DINO/MAE and CLIP.*

---

## Further Reading

- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095)
- [ViViT (Arnab et al., 2021)](https://arxiv.org/abs/2103.15691)
- [Video Vision Transformer survey (Selva et al., 2023)](https://arxiv.org/abs/2305.06575)
