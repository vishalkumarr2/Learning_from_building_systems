# Week 7: Vision Transformers

> **Days 45–49 · 12.5 hours**

This week opens Phase IV by applying the transformer — the same architecture you mastered in Phase II — to images. You'll see that an image is just a sequence of patch tokens.

## Daily Lessons

| Day | Topic | Phase | Focus |
|-----|-------|-------|-------|
| 45 | [ViT — An Image Is Worth 16×16 Words](day-45-vit.md) | IV | Patch embedding, [CLS] token |
| 46 | [Training ViT + DeiT](day-46-training-vit-deit.md) | IV | Data augmentation, distillation |
| 47 | [Swin Transformer](day-47-swin-transformer.md) | IV | Shifted windows, hierarchical features |
| 48 | [DINO & Self-Supervised Vision](day-48-dino-self-supervised.md) | IV | Self-distillation, attention maps |
| 49 | [MAE — Masked Autoencoders](day-49-mae.md) | IV | 75% masking, reconstruction |

## Key Concepts

- Images as sequences: split into 16×16 patches, linearly project, add position embeddings → transformer encoder
- Efficient attention: Swin's shifted-window approach reduces $O(n^2)$ to $O(n)$ per window
- Self-supervised vision: DINO learns segmentation from self-distillation; MAE learns by reconstructing masked patches
- Connection to BERT: MAE is the visual analog of masked language modeling

## Study Notes References

- [08 — Vision Transformers](../../study-notes/08-vision-transformers.md)
- [09 — 3D, Video & Detection](../../study-notes/09-3d-video-detection.md)
