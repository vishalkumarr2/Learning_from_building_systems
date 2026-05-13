# Day 61: Flamingo & BLIP-2

> **Phase V — Vision-Language Models | Week 9 | 2.5 hours**
> *"Freeze the LLM. Freeze the vision encoder. Only train the bridge. And suddenly your language model can see." — Li et al., 2023*

## Navigation
- **Previous:** [Day 60: CLIP Internals + SigLIP](day-60-clip-internals-siglip.md)
- **Next:** [Day 62: LLaVA — Visual Instruction Tuning](day-62-llava.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Architecture Challenge

You have a powerful vision encoder (ViT/CLIP) and a powerful LLM (LLaMA, PaLM). Both are frozen. How do you make them work together?

```
Challenge: connect two frozen models

┌──────────────┐         ┌──────────────┐
│ Vision Encoder│   ???   │ LLM          │
│ (frozen, 1B) │ ──────► │ (frozen, 7B) │
│              │         │              │
│ Output: 257  │         │ Input: text  │
│ tokens × 1024│         │ tokens × 4096│
└──────────────┘         └──────────────┘

The bridge must:
1. Translate vision features → LLM-compatible tokens
2. Preserve visual information
3. Train with limited compute (only bridge parameters)
```

### Flamingo (DeepMind, 2022)

Flamingo uses **Perceiver Resampler + Gated Cross-Attention**:

```
┌─────────────────────────────────────────────────────────┐
│                     Flamingo                             │
│                                                          │
│  Step 1: Perceiver Resampler                            │
│    Input: variable-length vision tokens                  │
│    Learned queries (64 tokens) cross-attend to vision   │
│    Output: fixed 64 visual tokens                       │
│                                                          │
│  Step 2: Gated Cross-Attention (inserted into LLM)     │
│    Every N-th LLM layer gets a cross-attention block:   │
│                                                          │
│    x = x + tanh(α) · CrossAttn(x, visual_tokens)       │
│              ↑                                           │
│         gate initialized to 0                            │
│         (LLM starts as if no vision exists)              │
│                                                          │
│  Step 3: Interleaved image-text                         │
│    Can process sequences like:                           │
│    <img>photo</img> What is this? It's a cat.           │
│    <img>photo2</img> And this? A dog.                   │
│    (few-shot in-context learning with images!)           │
└─────────────────────────────────────────────────────────┘
```

### BLIP-2 (Salesforce, 2023)

BLIP-2 takes a different approach with **Q-Former** — a 3-stage training process:

```
Stage 1: Vision-Language Representation Learning
  Q-Former ←→ Frozen ViT
  Objectives: ITC (contrastive) + ITM (matching) + ITG (generation)

Stage 2: Vision-to-Language Generative Learning  
  Q-Former → FC layer → Frozen LLM
  Train only the FC projection
  Q-Former outputs become visual prefix for LLM

Stage 3 (optional): Instruction tuning
```

**Q-Former architecture:**
- 32 learnable query tokens (shared across images)
- Self-attention among queries
- Cross-attention from queries to ViT features
- Lightweight: ~188M parameters

### Comparison Table

| Feature | Flamingo | BLIP-2 |
|---------|----------|--------|
| Vision encoder | NFNet (frozen) | ViT-G (frozen) |
| LLM | Chinchilla (frozen) | OPT/FlanT5 (frozen) |
| Bridge | Perceiver + gated x-attn | Q-Former + FC |
| Visual tokens to LLM | 64 (resampled) | 32 (queried) |
| Bridge params | ~1.5B | ~188M |
| Few-shot capability | Yes (interleaved) | Limited |
| Training stages | 1 | 3 |

---

## Implementation (60 min)

### Perceiver Resampler (Flamingo-style)

```python
import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Reduce variable-length vision tokens to fixed-length via cross-attention."""
    
    def __init__(self, dim=1024, n_queries=64, n_layers=6, n_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, n_heads, batch_first=True),
                'cross_norm': nn.LayerNorm(dim),
                'self_attn': nn.MultiheadAttention(dim, n_heads, batch_first=True),
                'self_norm': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
                'ffn_norm': nn.LayerNorm(dim),
            }))
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: (B, N_patches, D) — variable N
        Returns:
            resampled: (B, n_queries, D) — fixed size
        """
        B = vision_features.shape[0]
        queries = self.queries.expand(B, -1, -1)
        
        for layer in self.layers:
            # Cross-attend to vision features
            q = layer['cross_norm'](queries)
            queries = queries + layer['cross_attn'](q, vision_features, vision_features)[0]
            
            # Self-attend among queries
            q = layer['self_norm'](queries)
            queries = queries + layer['self_attn'](q, q, q)[0]
            
            # FFN
            queries = queries + layer['ffn'](layer['ffn_norm'](queries))
        
        return queries


class GatedCrossAttention(nn.Module):
    """Gated cross-attention layer inserted between LLM layers."""
    
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))  # initialized to 0!
    
    def forward(self, text_hidden, visual_tokens):
        """
        Args:
            text_hidden: (B, L, D) — LLM hidden states
            visual_tokens: (B, N_vis, D) — resampled visual tokens
        """
        residual = text_hidden
        text_normed = self.norm(text_hidden)
        
        cross_out = self.cross_attn(text_normed, visual_tokens, visual_tokens)[0]
        
        # Gating: starts at 0, gradually learns to incorporate vision
        return residual + torch.tanh(self.gate) * cross_out
```

### Using BLIP-2 with HuggingFace

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image


def blip2_caption(image_path, prompt=""):
    """Generate caption or answer question with BLIP-2."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
    )
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    
    if prompt:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            dtype=torch.float16
        )
    else:
        inputs = processor(images=image, return_tensors="pt").to(dtype=torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"BLIP-2: {text}")
    return text


def blip2_vqa(image_path, question):
    """Visual question answering with BLIP-2."""
    return blip2_caption(image_path, prompt=f"Question: {question} Answer:")
```

---

## Exercise (45 min)

1. **Gate analysis:** Initialize a GatedCrossAttention and train it on a simple image-captioning task. Track the gate value over training. How quickly does the gate open (move from 0)?

2. **Q-Former probe:** Load BLIP-2 and extract the 32 Q-Former output tokens for different images. Compute similarity between query tokens — do some queries specialize?

3. **Few-shot comparison:** Test BLIP-2 on visual question answering with 0-shot vs providing 2-3 example image-answer pairs in the prompt. Does few-shot help?

---

## Key Takeaways

1. **Frozen models + trained bridge.** Both Flamingo and BLIP-2 keep vision/LLM frozen
2. **Perceiver compresses.** Variable-length vision tokens → fixed-length through cross-attention
3. **Gating preserves LLM.** Zero-initialized gates let the LLM start unchanged and gradually incorporate vision
4. **Q-Former is efficient.** 188M parameters bridge a 1B vision encoder and 7B LLM
5. **Three-stage training.** BLIP-2's staged approach builds alignment progressively

## Connection to the Thread

> *Flamingo and BLIP-2 proved that frozen LLMs can be made visual with lightweight bridges. Tomorrow: LLaVA takes a simpler approach — just an MLP — and shows that instruction tuning matters more than bridge complexity.*

---

## Further Reading

- [Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198)
- [BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597)
- [InstructBLIP (Dai et al., 2023)](https://arxiv.org/abs/2305.06500)
- [Perceiver IO (Jaegle et al., 2021)](https://arxiv.org/abs/2107.14795)
