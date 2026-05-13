# Day 63: PaLI & CoCa

> **Phase V — Vision-Language Models | Week 9 | 2.5 hours**
> *"Contrastive loss for alignment. Captioning loss for generation. Together, they build the strongest vision-language representations." — Yu et al., 2022*

## Navigation
- **Previous:** [Day 62: LLaVA — Visual Instruction Tuning](day-62-llava.md)
- **Next:** [Day 64: Open VLM Landscape](../week-10/day-64-open-vlm-landscape.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### CoCa: Contrastive Captioners

CoCa (Contrastive Captioner, 2022) combines the best of CLIP and generative models:

```
┌────────────────────────────────────────────────────────────┐
│                    CoCa Architecture                        │
│                                                             │
│                  Image Encoder (ViT)                        │
│                        │                                    │
│              ┌─────────┴──────────┐                        │
│              │                     │                        │
│         [CLS] token          Patch tokens                   │
│              │                     │                        │
│              ▼                     ▼                        │
│    Contrastive Loss        Text Decoder                     │
│    (like CLIP)             (autoregressive)                 │
│                                    │                        │
│    image [CLS] ←sim→ text [CLS]  Cross-attention           │
│                           to patch tokens                   │
│                                    │                        │
│                           Captioning Loss                   │
│                           (generate text)                   │
│                                                             │
│    $\mathcal{L} = \mathcal{L}_{con} + \mathcal{L}_{cap}$  │
└────────────────────────────────────────────────────────────┘
```

**Key insight:** Split the text decoder into two halves:
- **Bottom half (unimodal):** Self-attention only on text → produces text [CLS] for contrastive
- **Top half (multimodal):** Cross-attention to image patches → generates captions

This dual-purpose design gives CoCa:
- CLIP's zero-shot transfer capability (contrastive)
- Captioning/generation capability (autoregressive)

### PaLI: Pathways Language and Image Model

PaLI (2022) scales VLMs to **17B parameters** with a simpler approach:

```
┌────────────────────────────────────────────────────────────┐
│                      PaLI Architecture                      │
│                                                             │
│  Image ──► ViT-e (4B params, frozen) ──► Visual tokens     │
│                                              │              │
│                                         [concatenate]       │
│                                              │              │
│  Text  ──► Tokenizer ──► Text tokens ────────┘              │
│                                              │              │
│                                         mT5 Encoder-Decoder │
│                                         (13B params)        │
│                                              │              │
│                                         Generated output    │
│                                                             │
│  Task examples (all as text generation):                    │
│    "caption en" → "A dog playing in the park"              │
│    "answer en: What color is the car?" → "red"             │
│    "detect car" → "<loc_x1><loc_y1><loc_x2><loc_y2>"      │
│    "ocr" → "STOP"                                          │
└────────────────────────────────────────────────────────────┘
```

PaLI's key principles:
1. **Scale the vision encoder.** ViT-e (4B params) — largest ViT ever at the time
2. **Reuse a strong LLM.** mT5 (multilingual T5) as the language backbone
3. **Unified text interface.** ALL tasks reformulated as text generation
4. **Multilingual.** Train on data in 100+ languages

### PaLI-2 and PaLI-3

| Model | Vision | Language | Params | Key Innovation |
|-------|--------|----------|--------|---------------|
| PaLI | ViT-e | mT5-XXL | 17B | Scale + unified interface |
| PaLI-2 | ViT-G | UL2 | 55B | Mixture of denoisers |
| PaLI-3 | SigLIP ViT-G | UL2 | 5B | Smaller but efficient, SigLIP |

PaLI-3 is notable: it achieves PaLI-level performance at 5B params by using SigLIP instead of ViT-e.

### Contrastive vs Captioning: Complementary Objectives

| Objective | Learns | Good at | Weak at |
|-----------|--------|---------|---------|
| **Contrastive** | Image↔text alignment | Zero-shot classification, retrieval | Text generation |
| **Captioning** | Image → text generation | Description, VQA | Discriminative tasks |
| **Both (CoCa)** | Alignment + generation | Everything | Nothing (best of both) |

---

## Implementation (60 min)

### CoCa-style Dual Objective

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoCaModel(nn.Module):
    """Simplified CoCa: contrastive + captioning in one model."""
    
    def __init__(self, vision_dim=768, text_dim=768, n_heads=12,
                 n_unimodal_layers=6, n_multimodal_layers=6):
        super().__init__()
        
        # Vision encoder (simplified — use ViT in practice)
        self.vision_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(vision_dim, n_heads, dim_feedforward=vision_dim*4, batch_first=True),
            num_layers=12
        )
        self.vision_cls = nn.Parameter(torch.randn(1, 1, vision_dim) * 0.02)
        
        # Text: unimodal layers (self-attention only)
        self.text_embed = nn.Embedding(32000, text_dim)
        self.text_pos = nn.Embedding(512, text_dim)
        
        self.unimodal_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(text_dim, n_heads, dim_feedforward=text_dim*4, batch_first=True),
            num_layers=n_unimodal_layers
        )
        
        # Text: multimodal layers (self-attention + cross-attention to vision)
        self.multimodal_layers = nn.ModuleList()
        for _ in range(n_multimodal_layers):
            self.multimodal_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(text_dim, n_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(text_dim, n_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(text_dim, text_dim * 4), nn.GELU(), nn.Linear(text_dim * 4, text_dim)
                ),
                'norm1': nn.LayerNorm(text_dim),
                'norm2': nn.LayerNorm(text_dim),
                'norm3': nn.LayerNorm(text_dim),
            }))
        
        self.output_head = nn.Linear(text_dim, 32000)
        
        # Contrastive projections
        self.vision_proj = nn.Linear(vision_dim, 512)
        self.text_proj = nn.Linear(text_dim, 512)
        self.temperature = nn.Parameter(torch.tensor(0.07).log())
    
    def encode_image(self, vision_tokens):
        """Encode image, return CLS and patch tokens separately."""
        B = vision_tokens.shape[0]
        cls = self.vision_cls.expand(B, -1, -1)
        x = torch.cat([cls, vision_tokens], dim=1)
        x = self.vision_encoder(x)
        return x[:, 0], x[:, 1:]  # cls_token, patch_tokens
    
    def forward(self, vision_tokens, input_ids, labels=None):
        # Image encoding
        img_cls, img_patches = self.encode_image(vision_tokens)
        
        # Text: unimodal layers (for contrastive)
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)
        text_embeds = self.text_embed(input_ids) + self.text_pos(positions)
        
        # Causal mask for autoregressive text
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        
        text_unimodal = self.unimodal_layers(text_embeds, mask=causal_mask)
        txt_cls = text_unimodal[:, -1]  # last token as CLS
        
        # Contrastive loss
        img_emb = F.normalize(self.vision_proj(img_cls), dim=-1)
        txt_emb = F.normalize(self.text_proj(txt_cls), dim=-1)
        
        temp = self.temperature.exp().clamp(max=100)
        logits_con = (img_emb @ txt_emb.T) / temp
        con_labels = torch.arange(B, device=logits_con.device)
        loss_con = (F.cross_entropy(logits_con, con_labels) + 
                     F.cross_entropy(logits_con.T, con_labels)) / 2
        
        # Multimodal layers (for captioning)
        x = text_unimodal
        for layer in self.multimodal_layers:
            r = x
            x = layer['norm1'](x)
            x = r + layer['self_attn'](x, x, x, attn_mask=causal_mask)[0]
            
            r = x
            x = layer['norm2'](x)
            x = r + layer['cross_attn'](x, img_patches, img_patches)[0]
            
            x = x + layer['ffn'](layer['norm3'](x))
        
        # Captioning loss
        logits_cap = self.output_head(x)
        loss_cap = 0
        if labels is not None:
            loss_cap = F.cross_entropy(logits_cap.view(-1, 32000), labels.view(-1))
        
        total_loss = loss_con + loss_cap
        return total_loss, loss_con, loss_cap
```

### PaLI-style Unified Task Interface

```python
def format_pali_tasks(image, task_type, **kwargs):
    """Format different tasks as text generation for PaLI."""
    task_formats = {
        'caption': lambda: f"caption en",
        'vqa': lambda: f"answer en: {kwargs['question']}",
        'detect': lambda: f"detect {kwargs['object']}",
        'ocr': lambda: f"ocr",
        'segment': lambda: f"segment {kwargs['object']}",
        'caption_de': lambda: f"caption de",  # German caption
    }
    
    prompt = task_formats[task_type]()
    return {'image': image, 'text_input': prompt}


# All tasks become text generation:
tasks = [
    format_pali_tasks(img, 'caption'),
    format_pali_tasks(img, 'vqa', question="What color is the robot?"),
    format_pali_tasks(img, 'detect', object="screwdriver"),
    format_pali_tasks(img, 'ocr'),
]
```

---

## Exercise (45 min)

1. **Objective comparison:** Train three small models on image-text data: (a) contrastive only, (b) captioning only, (c) CoCa (both). Compare zero-shot accuracy AND caption quality.

2. **Task unification:** Take 5 different vision tasks and reformulate each as text generation (PaLI-style). What are the advantages of this unified interface?

3. **Scaling analysis:** PaLI-3 (5B) matches PaLI (17B) by using SigLIP. Hypothesize why a better vision encoder reduces the total parameter budget needed.

---

## Key Takeaways

1. **Dual objectives.** CoCa's contrastive + captioning combines alignment and generation
2. **Split decoder.** Unimodal layers for contrastive, multimodal for captioning — elegant reuse
3. **Unified text interface.** PaLI shows all vision tasks can be framed as text generation
4. **Scale helps.** PaLI scales vision encoder to 4B+ params for unprecedented quality
5. **Efficiency wins.** PaLI-3 achieves same quality at 5B params with SigLIP

## Connection to the Thread

> *CoCa and PaLI represent the high-end of VLM architectures — combining multiple objectives and massive scale. Next week: the practical landscape of open VLMs, spatial reasoning, and hands-on fine-tuning.*

---

## Further Reading

- [CoCa (Yu et al., 2022)](https://arxiv.org/abs/2205.01917)
- [PaLI (Chen et al., 2022)](https://arxiv.org/abs/2209.06794)
- [PaLI-3 (Chen et al., 2023)](https://arxiv.org/abs/2310.09199)
- [PaLI-X (Chen et al., 2023)](https://arxiv.org/abs/2305.18565)
