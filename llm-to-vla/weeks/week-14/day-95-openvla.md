# Day 95: OpenVLA — Open-Source Vision-Language-Action Model

> **Phase VII — VLAs: Architecture to Deployment | Week 14 | 2.5 hours**
> *"7B parameters. Open weights. Fine-tune on 1 GPU. The first VLA the community can actually use." — Kim et al., 2024*

## Navigation
- **Previous:** [Day 94: Octo](day-94-octo.md)
- **Next:** [Day 96: π₀ — Physical Intelligence](day-96-pi0.md)
- **Week:** [Week 14 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 95.1 OpenVLA Positioning

```
Closed/Large:     RT-2 (55B, closed)
                     │
Open/Large:       PaLM-E (562B, partial)
                     │
Open/Medium:      OpenVLA (7B, fully open)  ← sweet spot
                     │
Open/Small:       Octo (93M, open)
```

### 95.2 Architecture

OpenVLA = **Prismatic VLM** + **action tokenization**

```
Image (224×224) ──→ SigLIP + DINOv2 (fused vision)
                         │
                    visual tokens (256)
                         │
Language ──→ Llama 2 (7B) tokenizer + embedding
                         │
                ┌────────▼────────────────┐
                │   Llama 2 7B backbone    │
                │   (autoregressive LM)    │
                └────────┬────────────────┘
                         │
                    [action tokens]
                    256 bins × 7 dims
```

### 95.3 Key Innovations

| Component | OpenVLA Choice | Why |
|-----------|---------------|-----|
| **Vision** | SigLIP + DINOv2 (fused) | Combines semantic + spatial features |
| **Language** | Llama 2 7B | Open-source, efficient |
| **Action** | 256-bin tokenization | Compatible with LM head |
| **Training** | Open X-Embodiment subset | 970K episodes |
| **Fine-tuning** | LoRA (rank 32) | Single GPU, 1-4 hours |

### 95.4 Prismatic Vision Encoder

OpenVLA fuses two vision encoders for complementary features:

```
Image ─┬──→ SigLIP-SO400M (semantic features)
       │        │
       │        ▼
       │   [384 tokens × 1152 dims]
       │        │
       └──→ DINOv2-L (spatial features)
                │
                ▼
           [256 tokens × 1024 dims]
                │
     MLP fusion → [256 tokens × 4096 dims]
                │
                ▼
           Llama 2 input
```

**SigLIP:** trained on image-text pairs → knows what objects are  
**DINOv2:** self-supervised → knows where objects are and their geometry

### 95.5 Training Recipe

```
Phase 1: VLM pre-training (from Prismatic)
  - SigLIP + DINOv2 + Llama 2
  - Standard VLM training on image-text data
  - Produces a strong visual reasoner

Phase 2: Action fine-tuning
  - Add action tokens to vocabulary
  - Fine-tune on Open X-Embodiment (970K episodes)
  - Mixed batch: 50% robot, 50% VLM data
  - 210K gradient steps, 8× A100s, 14 days
```

### 95.6 Fine-Tuning for New Robots

```python
# Fine-tuning recipe (conceptual)
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,                    # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(openvla_model, config)
# Only ~2% of parameters are trainable
# Fine-tune on 50-1000 demos in 1-4 GPU-hours
```

---

## Implementation (60 min)

### OpenVLA-Style Architecture

```python
import torch
import torch.nn as nn

class PrismaticVisionEncoder(nn.Module):
    """Fused dual-encoder vision backbone."""
    def __init__(self, out_dim=4096):
        super().__init__()
        # Simplified: in practice these are SigLIP + DINOv2
        self.semantic_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Flatten(2),  # (B, 128, 256)
        )
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Flatten(2),
        )
        self.fusion = nn.Linear(256, out_dim)  # 128 + 128

    def forward(self, image):
        sem = self.semantic_encoder(image).permute(0, 2, 1)  # (B, 256, 128)
        spa = self.spatial_encoder(image).permute(0, 2, 1)   # (B, 256, 128)
        fused = torch.cat([sem, spa], dim=-1)                 # (B, 256, 256)
        return self.fusion(fused)                              # (B, 256, 4096)

class OpenVLAStyle(nn.Module):
    """Simplified OpenVLA architecture."""

    def __init__(self, d_model=512, n_heads=8, n_layers=8,
                 text_vocab=32000, action_bins=256, action_dims=7):
        super().__init__()
        self.action_bins = action_bins
        self.action_dims = action_dims
        self.text_vocab = text_vocab
        total_vocab = text_vocab + action_bins

        # Vision
        self.vision = PrismaticVisionEncoder(out_dim=d_model)

        # Text embedding
        self.text_embed = nn.Embedding(total_vocab, d_model)

        # Transformer (simplified Llama)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # LM head
        self.lm_head = nn.Linear(d_model, total_vocab)

    def forward(self, image, text_tokens, action_tokens):
        """
        image: (B, 3, 224, 224)
        text_tokens: (B, T_text) — text token ids
        action_tokens: (B, action_dims) — ground truth action bin ids
        """
        B = image.shape[0]

        # Encode vision
        vis_tokens = self.vision(image)  # (B, 256, D)

        # Encode text
        text_emb = self.text_embed(text_tokens)  # (B, T, D)

        # Encode action tokens (teacher forcing, shift right)
        action_input = action_tokens[:, :-1] + self.text_vocab  # Offset to action vocab
        action_emb = self.text_embed(action_input)  # (B, action_dims-1, D)

        # Concatenate: [vision | text | actions]
        sequence = torch.cat([vis_tokens, text_emb, action_emb], dim=1)

        # Causal mask
        T = sequence.shape[1]
        mask = torch.triu(torch.ones(T, T, device=image.device), diagonal=1).bool()

        # Transform
        out = self.transformer(sequence, mask=mask)

        # Get predictions at action positions
        n_vis = vis_tokens.shape[1]
        n_text = text_tokens.shape[1]
        action_start = n_vis + n_text - 1  # One before first action
        action_end = action_start + self.action_dims

        action_logits = self.lm_head(out[:, action_start:action_end])
        action_logits = action_logits[:, :, self.text_vocab:]  # Action bins only

        loss = nn.functional.cross_entropy(
            action_logits.reshape(-1, self.action_bins),
            action_tokens.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def predict(self, image, text_tokens):
        """Autoregressive action prediction."""
        vis = self.vision(image)
        text_emb = self.text_embed(text_tokens)
        sequence = torch.cat([vis, text_emb], dim=1)

        actions = []
        for _ in range(self.action_dims):
            T = sequence.shape[1]
            mask = torch.triu(torch.ones(T, T, device=image.device), diagonal=1).bool()
            out = self.transformer(sequence, mask=mask)
            logits = self.lm_head(out[:, -1:])
            action_logits = logits[:, :, self.text_vocab:]
            bin_idx = action_logits.argmax(dim=-1)  # (B, 1)
            actions.append(bin_idx.squeeze(-1))

            token_emb = self.text_embed(bin_idx.squeeze(-1) + self.text_vocab)
            sequence = torch.cat([sequence, token_emb.unsqueeze(1)], dim=1)

        return torch.stack(actions, dim=-1)

# Demo
model = OpenVLAStyle()
img = torch.randn(2, 3, 224, 224)
txt = torch.randint(0, 1000, (2, 20))
act = torch.randint(0, 256, (2, 7))

loss = model(img, txt, act)
print(f"Loss: {loss.item():.4f}")
pred = model.predict(img, txt)
print(f"Predicted: {pred.shape}")
```

---

## Exercise (45 min)

1. **Dual encoder analysis:** Compare SigLIP-only vs DINOv2-only vs fused. Which features help more for grasping (spatial) vs object identification (semantic)?

2. **LoRA fine-tuning:** Implement LoRA for the transformer layers. Compare trainable parameter count vs full fine-tuning. Measure performance with 100 demos.

3. **Vocabulary efficiency:** The action vocabulary (256 tokens) is <1% of the text vocabulary (32K). Analyze: does this imbalance affect training? What if you use 1024 action bins?

4. **Scaling comparison:** Plot parameter count vs success rate for RT-1 (35M), Octo (93M), OpenVLA (7B), RT-2 (55B). Is bigger always better?

---

## Key Takeaways

1. **OpenVLA = Prismatic VLM + action tokens** — the first fully open-source VLA
2. **Dual vision encoder** (SigLIP + DINOv2) combines semantic and spatial understanding
3. **LoRA fine-tuning** enables adaptation on a single GPU in hours
4. **7B is the sweet spot** — large enough for reasoning, small enough for deployment
5. **Open-source** enables the community to build on and improve VLA research

---

## Connection to the Thread

> *RT-2 showed VLMs can be VLAs. Octo showed open-source works. OpenVLA made it practical with LoRA fine-tuning. Tomorrow: π₀ from Physical Intelligence takes a radically different approach — a VLM backbone with a flow matching action head, achieving state-of-the-art on dexterous manipulation. The question: tokenize actions or generate them continuously?*

---

## Further Reading

- Kim et al. (2024), "OpenVLA: An Open-Source Vision-Language-Action Model"
- Karamcheti et al. (2024), "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models"
- [OpenVLA GitHub](https://github.com/openvla/openvla)
