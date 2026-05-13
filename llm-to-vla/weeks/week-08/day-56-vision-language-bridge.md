# Day 56: Vision-Language Bridge

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"The hardest problem in multi-modal AI: how do you make a language model understand images? Three answers: projection, cross-attention, and Q-Former." — The bridge patterns*

## Navigation
- **Previous:** [Day 55: DETR + Florence-2 + SAM 2](day-55-detr-florence-sam.md)
- **Next:** [Day 57: Phase IV Capstone Day 1](../week-09/day-57-phase4-capstone-1.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/09-3d-video-detection.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Bridge Problem

Vision encoders output spatial features: $(N, D_v)$ patch tokens with $D_v = 768\text{–}1024$.
Language models consume text tokens: $(L, D_l)$ with $D_l = 2048\text{–}4096$.

How do you feed visual features into a language model? Three architectural patterns have emerged:

```
Pattern 1: Linear/MLP Projection (LLaVA)
┌──────────┐     ┌──────────┐     ┌──────────────┐
│ Vision   │ ──► │ MLP      │ ──► │ LLM          │
│ Encoder  │     │ (2-layer)│     │ (frozen/tuned)│
│ (frozen) │     └──────────┘     └──────────────┘
└──────────┘
  N visual tokens projected to LLM dimension
  Simple, effective, fast

Pattern 2: Cross-Attention (Flamingo)
┌──────────┐     ┌──────────────────────────────┐
│ Vision   │ ──► │ LLM with cross-attention     │
│ Encoder  │     │ layers interleaved            │
│ (frozen) │     │                               │
└──────────┘     │ text ─── cross-attn ─── text │
                 │            ↑                  │
                 │         vision                │
                 └──────────────────────────────┘
  Visual features attend to text through gated cross-attention
  Powerful but adds parameters to LLM

Pattern 3: Q-Former (BLIP-2)
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Vision   │ ──► │ Q-Former     │ ──► │ LLM          │
│ Encoder  │     │ (32 queries) │     │ (frozen)     │
│ (frozen) │     │ cross-attn   │     └──────────────┘
└──────────┘     └──────────────┘
  Learned queries extract fixed-size visual tokens
  Reduces N visual tokens → 32 query tokens
  Compression: 257 → 32 tokens
```

### Pattern 1: Linear Projection (LLaVA)

The simplest bridge. A 2-layer MLP projects each visual token from $D_v$ to $D_l$:

$$h_i^v = W_2 \cdot \text{GELU}(W_1 \cdot z_i^v + b_1) + b_2$$

where $z_i^v$ is the $i$-th vision encoder output and $h_i^v$ becomes a "visual word" for the LLM.

The visual tokens are simply **concatenated** with text tokens:

$$\text{LLM input} = [h_1^v, h_2^v, \ldots, h_N^v, \; t_1, t_2, \ldots, t_L]$$

### Pattern 2: Cross-Attention (Flamingo)

Flamingo inserts **gated cross-attention** layers between existing LLM layers:

$$\text{output} = \text{FFN}(\text{self-attn}(x) + \tanh(\alpha) \cdot \text{cross-attn}(x, v))$$

where:
- $\alpha$ is a learnable gate initialized to 0 (so the LLM starts unchanged)
- $v$ are the visual features
- Cross-attention queries from text, keys/values from vision

The **Perceiver Resampler** first reduces visual tokens to a fixed number:

$$\text{queries} = \text{learned}(64 \text{ tokens})$$
$$\text{resampled} = \text{CrossAttn}(\text{queries}, \text{visual\_tokens})$$

### Pattern 3: Q-Former (BLIP-2)

Q-Former uses 32 learnable query tokens that cross-attend to vision features:

```
┌─────────────────────────────────────────────┐
│               Q-Former                       │
│                                              │
│  32 learned queries                          │
│       │                                      │
│  Self-attention (queries attend to queries)  │
│       │                                      │
│  Cross-attention (queries attend to vision)  │
│       │                                      │
│  FFN                                         │
│       │                                      │
│  Output: 32 tokens → project to LLM dim     │
└─────────────────────────────────────────────┘
```

Q-Former is pretrained with three objectives:
1. **Image-text contrastive** (like CLIP)
2. **Image-grounded text generation**
3. **Image-text matching**

### Comparison

| Pattern | Visual tokens to LLM | Training cost | Quality | Simplicity |
|---------|---------------------|---------------|---------|------------|
| MLP (LLaVA) | N (196–576) | Low | Good | ★★★ |
| Cross-Attn (Flamingo) | Resampled (64) | Medium | Very good | ★★ |
| Q-Former (BLIP-2) | 32 | High (3-stage) | Very good | ★ |

Modern trend: **simpler is better**. LLaVA-1.5 with just an MLP matches BLIP-2/Flamingo while being much simpler.

---

## Implementation (60 min)

### MLP Projection Bridge (LLaVA-style)

```python
import torch
import torch.nn as nn


class VisionLanguageBridge(nn.Module):
    """MLP bridge from vision encoder to LLM (LLaVA-style)."""
    
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: (B, N_patches, vision_dim) from ViT
        Returns:
            visual_tokens: (B, N_patches, llm_dim) — ready for LLM
        """
        return self.projector(vision_features)


class QFormerBridge(nn.Module):
    """Simplified Q-Former bridge (BLIP-2-style)."""
    
    def __init__(self, vision_dim=1024, llm_dim=4096, n_queries=32, n_layers=6, n_heads=8):
        super().__init__()
        hidden_dim = 768
        
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'norm3': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
            }))
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, llm_dim)
    
    def forward(self, vision_features):
        B = vision_features.shape[0]
        queries = self.queries.expand(B, -1, -1)
        vision_kv = self.vision_proj(vision_features)
        
        x = queries
        for layer in self.layers:
            # Self-attention among queries
            residual = x
            x = layer['norm1'](x)
            x = residual + layer['self_attn'](x, x, x)[0]
            
            # Cross-attention: queries attend to vision features
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['cross_attn'](x, vision_kv, vision_kv)[0]
            
            # FFN
            residual = x
            x = residual + layer['ffn'](layer['norm3'](x))
        
        return self.output_proj(x)  # (B, n_queries, llm_dim)


# Compare the two approaches
vision_features = torch.randn(2, 196, 1024)  # ViT output

mlp_bridge = VisionLanguageBridge(1024, 4096)
qformer_bridge = QFormerBridge(1024, 4096, n_queries=32)

mlp_out = mlp_bridge(vision_features)
qf_out = qformer_bridge(vision_features)

print(f"MLP bridge: {vision_features.shape} → {mlp_out.shape}")  # (2, 196, 4096)
print(f"Q-Former:   {vision_features.shape} → {qf_out.shape}")   # (2, 32, 4096)
print(f"Token reduction: {196} → {32} ({196/32:.1f}× compression)")
```

---

## Exercise (45 min)

1. **Token budget analysis:** For an LLM with 8K context window, calculate how many text tokens remain after inserting visual tokens with each bridge method (MLP: 196, Flamingo: 64, Q-Former: 32). Which matters most for multi-turn conversation?

2. **Bridge ablation:** Implement a single linear layer bridge (no hidden layer). Compare with the 2-layer MLP on a simple image captioning task using a small LLM. How much does the nonlinearity matter?

3. **Attention visualization:** In the Q-Former, extract the cross-attention weights. Which image patches do the 32 queries attend to? Do different queries specialize in different image regions?

---

## Key Takeaways

1. **Three bridge patterns.** MLP projection, cross-attention, and Q-Former each trade off simplicity vs compression
2. **Simple works.** LLaVA's MLP projection matches more complex approaches
3. **Token compression matters.** Q-Former's 32 tokens save LLM context vs MLP's 196+
4. **Frozen components.** Both vision encoder and LLM are typically frozen — only the bridge trains
5. **This enables VLMs.** The bridge is what makes language models multimodal

## Connection to the Thread

> *You've now seen every component: vision encoders (ViT, Swin, DINO), dense prediction (depth, detection, segmentation), and the bridge that connects vision to language. Next week: the Phase IV capstone, then Phase V where these pieces come together as full Vision-Language Models.*

---

## Further Reading

- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — MLP projection
- [Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198) — gated cross-attention
- [BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597) — Q-Former
- [LLaVA-1.5 (Liu et al., 2023)](https://arxiv.org/abs/2310.03744) — MLP wins
