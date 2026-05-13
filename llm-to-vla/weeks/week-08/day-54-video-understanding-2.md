# Day 54: Video Understanding Day 2

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"Mask 90% of video and still learn powerful representations — temporal redundancy is even higher than spatial." — Tong et al., 2022*

## Navigation
- **Previous:** [Day 53: Video Understanding Day 1](day-53-video-understanding-1.md)
- **Next:** [Day 55: DETR + Florence-2 + SAM 2](day-55-detr-florence-sam.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/09-3d-video-detection.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### VideoMAE: Masked Autoencoding for Video

VideoMAE extends MAE from images to video with two key modifications:
1. **Tube masking:** Mask the same spatial position across ALL frames (not random per-frame)
2. **90% masking ratio:** Video has even more redundancy than images

```
Tube masking (same position masked in all frames):

Frame 1:  [■ □ ■ □]     ■ = visible
Frame 2:  [■ □ ■ □]     □ = masked
Frame 3:  [■ □ ■ □]
Frame 4:  [■ □ ■ □]

vs Random masking (independent per frame):

Frame 1:  [■ □ □ ■]     Different positions
Frame 2:  [□ ■ ■ □]     masked per frame
Frame 3:  [■ ■ □ □]     → too easy (temporal copy)
Frame 4:  [□ □ ■ ■]
```

**Why tube masking?** Random per-frame masking is too easy — the model just copies from adjacent frames. Tube masking forces understanding of motion and change.

### Video-Text Pretraining

Like CLIP aligns images and text, video-text models align video clips and text descriptions:

```
┌──────────────────────────────────────────────────┐
│              Video-Text Pretraining               │
│                                                   │
│  Video: [frame₁, ..., frame_T] → Video Encoder   │
│                                     │              │
│                                 similarity        │
│                                     │              │
│  Text: "a person riding a bike" → Text Encoder   │
│                                                   │
│  Loss: contrastive (like CLIP) or matching        │
│                                                   │
│  Models: VideoCLIP, InternVideo, LanguageBind     │
└──────────────────────────────────────────────────┘
```

### Temporal Coherence

Videos have a fundamental property that images lack: **temporal coherence**. Adjacent frames are strongly correlated. Models must learn:

- **Short-term dynamics:** Object motion between frames (optical flow)
- **Long-term structure:** Activity progression (beginning → middle → end)
- **Causal relationships:** Action → effect sequences

### InternVideo2: State of the Art

InternVideo2 (2024) combines multiple pretraining objectives:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{mask}} + \lambda_2 \mathcal{L}_{\text{contrastive}} + \lambda_3 \mathcal{L}_{\text{next-token}}$$

- **Masked reconstruction** (VideoMAE-style)
- **Video-text contrastive** (CLIP-style)
- **Next-token prediction** (GPT-style on video tokens)

---

## Implementation (60 min)

### VideoMAE Training

```python
import torch
import torch.nn as nn
from einops import rearrange


class TubeMasking(nn.Module):
    """Tube masking: mask same spatial positions across all frames."""
    
    def __init__(self, n_patches_per_frame, mask_ratio=0.9):
        super().__init__()
        self.n_patches = n_patches_per_frame
        self.mask_ratio = mask_ratio
    
    def forward(self, x, n_frames):
        """
        Args:
            x: (B, T*N, D) video tokens
            n_frames: T
        Returns:
            visible: (B, T*N_vis, D)
            mask: (B, T*N) binary mask
            ids_restore: for reconstruction
        """
        B, TN, D = x.shape
        N = self.n_patches
        
        # Decide which spatial positions to keep (same for all frames)
        n_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_keep_spatial = ids_shuffle[:, :n_keep]  # (B, n_keep)
        
        # Expand to all frames
        ids_keep = []
        for t in range(n_frames):
            ids_keep.append(ids_keep_spatial + t * N)
        ids_keep = torch.cat(ids_keep, dim=1)  # (B, T*n_keep)
        
        # Gather visible tokens
        visible = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Create mask (1 = masked, 0 = visible)
        mask = torch.ones(B, TN, device=x.device)
        mask.scatter_(1, ids_keep, 0)
        
        # Restore indices for reconstruction
        ids_restore = torch.argsort(
            torch.cat([ids_keep, 
                       torch.arange(TN, device=x.device).unsqueeze(0).expand(B, -1)
                       .gather(1, mask.nonzero(as_tuple=False)[:, 1].view(B, -1))], dim=1),
            dim=1
        )
        
        return visible, mask, ids_keep


class VideoMAE(nn.Module):
    """Simplified VideoMAE for self-supervised video pretraining."""
    
    def __init__(self, img_size=224, patch_size=16, n_frames=16,
                 embed_dim=768, encoder_depth=12, decoder_depth=4,
                 n_heads=12, mask_ratio=0.9):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.n_frames = n_frames
        self.mask_ratio = mask_ratio
        patch_dim = 3 * patch_size ** 2
        
        # Encoder
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.temporal_embed = nn.Parameter(torch.zeros(1, n_frames, embed_dim))
        self.spatial_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, n_heads)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder (lightweight)
        decoder_dim = embed_dim // 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TimeSformerBlock(decoder_dim, n_heads // 2)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)
        
        self.tube_mask = TubeMasking(self.n_patches, mask_ratio)
    
    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W)
        Returns:
            loss: reconstruction loss on masked tubes
        """
        B, T, C, H, W = video.shape
        P = int((H // int(self.n_patches ** 0.5)))
        
        # Patchify
        patches = rearrange(video, 'b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=P, p2=P)
        target = patches.clone()
        
        x = self.patch_embed(patches)
        
        # Add positional embeddings
        spatial = self.spatial_embed.repeat(1, T, 1)
        temporal = self.temporal_embed.repeat_interleave(self.n_patches, dim=1)
        x = x + spatial + temporal
        
        # Tube masking — 90% masked!
        visible, mask, ids_keep = self.tube_mask(x, T)
        
        # Encode visible tokens only
        n_keep = visible.shape[1] // T
        for block in self.encoder_blocks:
            visible = block(visible, T, n_keep)
        visible = self.encoder_norm(visible)
        
        # Decode: insert mask tokens, reconstruct
        # (simplified — full implementation handles restore indices)
        decoded = self.decoder_embed(visible)
        pred = self.decoder_pred(self.decoder_norm(decoded))
        
        # MSE loss on masked positions
        loss = ((pred - target.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, target.shape[-1]))) ** 2).mean()
        
        return loss
```

### Video Feature Extraction with HuggingFace

```python
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
import numpy as np


def classify_video(video_frames, model_name="MCG-NJU/videomae-base-finetuned-kinetics"):
    """Classify a video clip using pretrained VideoMAE."""
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    model.eval()
    
    # Prepare input: list of PIL images or numpy arrays
    inputs = feature_extractor(list(video_frames), return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]
    confidence = torch.softmax(logits, dim=-1).max().item()
    
    print(f"Predicted: {label} ({confidence:.2%})")
    return label, confidence
```

---

## Exercise (45 min)

1. **Tube vs random masking:** Implement random per-frame masking for VideoMAE. Compare reconstruction quality with tube masking at 75% and 90% ratios. Why does tube masking learn better representations?

2. **Video-text retrieval:** Using a pretrained video-text model (e.g., from HuggingFace), embed 5 video clips and 10 text descriptions. Compute the similarity matrix. Can the model match videos to correct descriptions?

3. **Temporal understanding probe:** Create two videos — one showing "putting a cup on a table" and the reverse "removing a cup from a table." Can a VideoMAE model distinguish temporal order?

---

## Key Takeaways

1. **90% masking works.** Video is so redundant that masking 90% of spatiotemporal patches still allows reconstruction
2. **Tube masking prevents shortcuts.** Random per-frame masking lets the model copy from adjacent frames
3. **Video-text alignment.** Contrastive learning connects video understanding to natural language
4. **Temporal coherence.** The key property distinguishing video from image bags — models must learn dynamics
5. **Multiple objectives.** Best video models combine masking + contrastive + prediction losses

## Connection to the Thread

> *Video understanding adds temporal reasoning to our vision toolkit. Tomorrow: object detection (DETR), open-vocabulary detection (Florence-2), and segment anything (SAM 2) — the perception modules robots need to interact with specific objects.*

---

## Further Reading

- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602)
- [VideoMAE V2 (Wang et al., 2023)](https://arxiv.org/abs/2302.13669)
- [InternVideo2 (Wang et al., 2024)](https://arxiv.org/abs/2403.15377)
- [LanguageBind (Zhu et al., 2024)](https://arxiv.org/abs/2310.01852)
