# Day 47: Swin Transformer

> **Phase IV — Vision: ViT, 3D, Video | Week 7 | 2.5 hours**
> *"Shifted windows: the trick that made vision transformers practical for dense prediction." — Liu et al., 2021*

## Navigation
- **Previous:** [Day 46: Training ViT + DeiT](day-46-training-vit-deit.md)
- **Next:** [Day 48: DINO & Self-Supervised Vision](day-48-dino-self-supervised.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Problem with ViT

ViT computes global self-attention over all patches — $O(n^2)$ where $n$ is the number of patches. For a 224×224 image with 16×16 patches, $n = 196$. Manageable.

But for dense prediction tasks (object detection, segmentation) at higher resolutions:
- 512×512, patch_size=4 → $n = 16{,}384$ patches → attention matrix is 16K × 16K
- This is **computationally infeasible**

### Swin's Solution: Windowed Attention

Swin Transformer computes attention **within local windows**, then shifts the windows to enable cross-window communication:

```
Stage 1: Window Attention         Stage 2: Shifted Window Attention
┌───┬───┬───┬───┐               ┌──┬────┬────┬──┐
│ A │ A │ B │ B │               │  │    │    │  │
│   │   │   │   │               │──┼────┼────┼──│
│ A │ A │ B │ B │    shift by   │  │ E  │ F  │  │
├───┼───┼───┼───┤  ──────────►  │──┼────┼────┼──│
│ C │ C │ D │ D │    (M/2, M/2) │  │ G  │ H  │  │
│   │   │   │   │               │──┼────┼────┼──│
│ C │ C │ D │ D │               │  │    │    │  │
└───┴───┴───┴───┘               └──┴────┴────┴──┘

Attention in A,B,C,D: local      Shifted: new windows E,F,G,H
                                  cross previous boundaries!
```

### Complexity Comparison

For an image with $h \times w$ patches and window size $M$:

| Approach | Complexity |
|----------|-----------|
| Global attention (ViT) | $O(h^2 w^2 \cdot d)$ — quadratic in image size |
| Window attention (Swin) | $O(h w \cdot M^2 \cdot d)$ — **linear** in image size |

With $M = 7$ (typical), attention is computed on $7 \times 7 = 49$ tokens per window — tiny and fixed!

### Hierarchical Feature Maps via Patch Merging

Unlike ViT (single-scale), Swin produces multi-scale features like a CNN backbone:

```
Stage 1: 56×56, C=96      ← high resolution, fine features
    │ Patch Merging (2×2 → 1, double channels)
    ▼
Stage 2: 28×28, C=192     ← medium resolution
    │ Patch Merging
    ▼
Stage 3: 14×14, C=384     ← low resolution, semantic features
    │ Patch Merging
    ▼
Stage 4: 7×7, C=768       ← coarsest features
```

**Patch Merging:** Concatenate 2×2 neighboring patches (4C channels) → linear projection to 2C.

This hierarchical design makes Swin a drop-in replacement for CNN backbones in detection (FPN) and segmentation (UPerNet).

### Relative Position Bias

Instead of absolute position embeddings, Swin uses **relative position bias** $B \in \mathbb{R}^{M^2 \times M^2}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V$$

This handles variable window positions and improves performance.

---

## Implementation (60 min)

### Window Attention

```python
import torch
import torch.nn as nn
from einops import rearrange


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size: int (M)
    Returns:
        windows: (B * num_windows, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window_partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    
    def __init__(self, dim, window_size, n_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias table
        # (2*M-1) * (2*M-1) possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index for each token pair
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, M, M)
        coords_flat = coords.view(2, -1)  # (2, M*M)
        
        # Relative coordinates
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, M*M, M*M)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (M*M, M*M)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape  # B_ = batch * num_windows
        head_dim = C // self.n_heads
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        ).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        
        if mask is not None:
            attn = attn + mask.unsqueeze(1)
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)
```

### Swin Transformer Block

```python
class SwinBlock(nn.Module):
    """Two consecutive Swin blocks: W-MSA then SW-MSA."""
    
    def __init__(self, dim, n_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        x_reshaped = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted = torch.roll(x_reshaped, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_reshaped
        
        # Partition into windows
        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_out = self.attn(self.norm1(windows).view(-1, self.window_size * self.window_size, C))
        
        # Reverse windows
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        shifted = window_reverse(attn_out, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            out = shifted
        
        out = out.view(B, H * W, C)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## Exercise (45 min)

1. **Complexity comparison:** For a 512×512 image with patch_size=4 and window_size=7, compute:
   - Number of patches (tokens) for ViT
   - Number of windows for Swin
   - FLOPs for global attention vs window attention

2. **Shifted window visualization:** Create a 8×8 grid and manually trace which tokens can attend to each other after one regular + one shifted window attention layer with window_size=4.

3. **Use pretrained Swin:** Load `swin_tiny_patch4_window7_224` from timm and extract multi-scale features. Print the shapes at each stage.

---

## Key Takeaways

1. **Linear complexity.** Window attention makes transformers practical for high-resolution images
2. **Shifted windows.** Alternating between regular and shifted windows enables cross-window communication
3. **Hierarchical features.** Patch merging creates multi-scale feature maps like CNN backbones
4. **Relative position bias.** Better than absolute position embeddings for windowed attention
5. **Drop-in backbone.** Swin replaces ResNet in FPN, UPerNet, Mask R-CNN with better performance

## Connection to the Thread

> *Swin solved ViT's efficiency bottleneck. But both ViT and Swin need labeled data. Tomorrow: DINO learns visual features without any labels at all — through self-distillation.*

---

## Further Reading

- [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030)
- [Swin Transformer V2 (Liu et al., 2022)](https://arxiv.org/abs/2111.09883)
- Official implementation: github.com/microsoft/Swin-Transformer
