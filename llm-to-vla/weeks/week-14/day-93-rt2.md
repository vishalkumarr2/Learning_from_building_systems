# Day 93: RT-2 — When a VLM Becomes a VLA

> **Phase VII — VLAs: Architecture to Deployment | Week 14 | 2.5 hours**
> *"Take PaLI-X (55B parameters, trained on web-scale vision-language data). Fine-tune it to output action tokens. That's it. That's the paper." — Brohan et al., 2023*

## Navigation
- **Previous:** [Day 92: RT-1](day-92-rt1.md)
- **Next:** [Day 94: Octo — The Generalist Policy](day-94-octo.md)
- **Week:** [Week 14 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 93.1 The RT-2 Insight

RT-1 used a custom architecture for robotics. RT-2 asks: what if we just **fine-tune an existing VLM** to output robot actions as text tokens?

```
RT-1:                               RT-2:
  Custom vision encoder               PaLI-X or PaLM-E (pre-trained VLM)
  Custom transformer                   Same architecture
  Custom action heads                  Actions = special text tokens
  35M params, robot data only          55B params, web + robot data
```

### 93.2 Architecture

```
Input:
  Image ──→ ViT-22B encoder ──→ visual tokens
                                    │
  Text: "pick up the red block" ──→ text tokens
                                    │
                              ┌─────▼──────────────┐
                              │    PaLI-X / PaLM-E  │
                              │    (55B parameters)  │
                              │    Autoregressive    │
                              └─────┬──────────────┘
                                    │
Output tokens:                      ▼
  "1 128 91 241 5 101 127 128 255"
   │  │   │  │  │  │   │   │   │
   │  Δx  Δy Δz Δr Δp  Δyw grip term
   └── mode token
```

### 93.3 Action Token Integration

RT-2 adds action tokens to the VLM's text vocabulary:

```python
# Original VLM vocabulary: 0-31999 (text tokens)
# RT-2 extension: 32000-32255 (256 action bin tokens)

# Example output sequence:
# "pick up the apple" → [text...] [32128] [32091] [32241] [32005] [32101] [32127] [32255]
#                                   Δx=0.0  Δy=-0.2  Δz=0.9  ...        grip=open
```

**The key trick:** actions are just more tokens in the sequence. The VLM doesn't need architectural changes — it already knows how to predict the next token.

### 93.4 Co-Fine-Tuning

RT-2 fine-tunes on robot data while keeping web data:

$$\mathcal{L} = \lambda_\text{robot} \cdot \mathcal{L}_\text{robot} + \lambda_\text{web} \cdot \mathcal{L}_\text{web}$$

| Data Source | Purpose | Proportion |
|------------|---------|------------|
| Robot demonstrations | Learn to act | ~50% |
| Web vision-language | Retain VLM capabilities | ~50% |

This prevents **catastrophic forgetting** of visual reasoning.

### 93.5 Emergent Capabilities

Because RT-2 retains VLM reasoning, it can:

```
Instruction: "Move the banana to the plate with the same color"
  → VLM reasons: banana is yellow → yellow plate
  → Outputs: action tokens to move banana to yellow plate

Instruction: "Pick up the object that Taylor Swift might like"
  → VLM reasons: pop star → heart-shaped object? music note?
  → Outputs: action tokens to pick up relevant object
```

These **chain-of-thought robot actions** were impossible with RT-1.

### 93.6 RT-1 vs RT-2 Comparison

| Aspect | RT-1 | RT-2 |
|--------|------|------|
| Parameters | 35M | 55B |
| Pre-training | ImageNet only | Web-scale VL |
| Language understanding | Shallow (USE) | Deep (VLM) |
| Semantic generalization | Limited | Strong |
| Inference speed | 5 Hz | 1-3 Hz |
| Novel object reasoning | No | Yes |
| Action representation | Parallel softmax | Autoregressive tokens |

---

## Implementation (60 min)

### RT-2 Style Action Token Training

```python
import torch
import torch.nn as nn

class SimplifiedRT2(nn.Module):
    """Simplified RT-2: VLM backbone + action token prediction."""

    def __init__(self, vlm_dim=768, n_action_dims=7, n_bins=256,
                 text_vocab_size=32000):
        super().__init__()
        self.n_bins = n_bins
        self.n_action_dims = n_action_dims
        self.action_vocab_offset = text_vocab_size
        self.total_vocab = text_vocab_size + n_bins

        # Simplified VLM backbone (replace with real VLM)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, vlm_dim),
        )
        self.text_encoder = nn.Embedding(text_vocab_size, vlm_dim)
        self.action_token_embed = nn.Embedding(n_bins, vlm_dim)

        # Autoregressive transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vlm_dim, nhead=8, dim_feedforward=vlm_dim*4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output head (shared for text and action tokens)
        self.output_head = nn.Linear(vlm_dim, self.total_vocab)

        # Learnable start token for action sequence
        self.action_start = nn.Parameter(torch.randn(vlm_dim))

    def encode_context(self, image, text_tokens):
        """Encode image + text into context tokens."""
        img_emb = self.image_encoder(image).unsqueeze(1)  # (B, 1, D)
        txt_emb = self.text_encoder(text_tokens)            # (B, T, D)
        return torch.cat([img_emb, txt_emb], dim=1)         # (B, 1+T, D)

    def forward(self, image, text_tokens, action_bins):
        """
        Training forward pass.
        action_bins: (B, n_action_dims) — ground truth action bin indices
        """
        B = image.shape[0]
        context = self.encode_context(image, text_tokens)

        # Build action input sequence: [start, a₁, a₂, ..., a₆]
        start = self.action_start.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        action_emb = self.action_token_embed(action_bins[:, :-1])  # Teacher forcing
        decoder_input = torch.cat([start, action_emb], dim=1)  # (B, n_dims, D)

        # Causal mask
        T = decoder_input.shape[1]
        mask = torch.triu(torch.ones(T, T, device=image.device), diagonal=1).bool()

        # Decode
        out = self.decoder(decoder_input, context, tgt_mask=mask)
        logits = self.output_head(out)  # (B, n_dims, total_vocab)

        # Only care about action token predictions
        action_logits = logits[:, :, self.action_vocab_offset:]  # (B, n_dims, n_bins)

        # Loss: cross-entropy per action dimension
        loss = nn.functional.cross_entropy(
            action_logits.reshape(-1, self.n_bins),
            action_bins.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def predict(self, image, text_tokens):
        """Autoregressive action generation."""
        B = image.shape[0]
        context = self.encode_context(image, text_tokens)

        # Start with start token
        current = self.action_start.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        actions = []

        for dim in range(self.n_action_dims):
            out = self.decoder(current, context)
            logits = self.output_head(out[:, -1:])  # Last position
            action_logits = logits[:, :, self.action_vocab_offset:]
            bin_idx = action_logits.argmax(dim=-1).squeeze(1)  # (B,)
            actions.append(bin_idx)

            # Add predicted token to sequence
            new_token = self.action_token_embed(bin_idx).unsqueeze(1)
            current = torch.cat([current, new_token], dim=1)

        return torch.stack(actions, dim=-1)  # (B, n_action_dims)

# Demo
model = SimplifiedRT2()
image = torch.randn(4, 3, 224, 224)
text = torch.randint(0, 1000, (4, 10))
actions_gt = torch.randint(0, 256, (4, 7))

loss = model(image, text, actions_gt)
print(f"Training loss: {loss.item():.4f}")

pred = model.predict(image, text)
print(f"Predicted actions: {pred.shape}")  # (4, 7)
```

---

## Exercise (45 min)

1. **Vocabulary design:** Calculate the total vocabulary size for RT-2 with PaLI-X (32K text tokens) + 256 action bins. What fraction of the vocabulary is actions? Why might this ratio matter for training?

2. **Autoregressive speed:** Measure the time for RT-2 autoregressive decoding (7 forward passes) vs RT-1 parallel decoding (1 forward pass). What's the practical speed difference?

3. **Emergent reasoning:** Design 5 instructions that require VLM reasoning to execute (e.g., "pick up the fruit that's not an apple"). Would RT-1 handle these? Why not?

4. **Co-fine-tuning ratio:** Train your simplified RT-2 with robot-only data vs 50/50 robot+text data. Does keeping text data during fine-tuning help action prediction?

---

## Key Takeaways

1. **RT-2 = pre-trained VLM + action tokens** — no architectural changes needed
2. **VLM reasoning transfers to robot actions** — semantic generalization emerges
3. **Co-fine-tuning** prevents catastrophic forgetting of visual reasoning
4. **Autoregressive action generation** is 7× slower but captures inter-dimension dependencies
5. **55B parameters** is too large for edge deployment — motivating Octo and OpenVLA

---

## Connection to the Thread

> *RT-2 proves that VLMs can be VLAs. But 55B parameters is impractical. Tomorrow, Octo (2024) takes a different approach: a 93M parameter generalist policy that uses a diffusion action head instead of autoregressive tokens. Day 95, OpenVLA makes VLAs open-source with a 7B parameter model anyone can fine-tune.*

---

## Further Reading

- Brohan et al. (2023), "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
- Chen et al. (2023), "PaLI-X: On Scaling up a Multilingual Vision and Language Model"
- Driess et al. (2023), "PaLM-E: An Embodied Multimodal Language Model"
