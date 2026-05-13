# Day 98: GR-2 & GROOT N1 — Video-to-Action Models

> **Phase VII — VLAs: Architecture to Deployment | Week 14 | 2.5 hours**
> *"What if predicting future video frames is all you need for robot control?" — Video prediction as a world model for action generation.*

## Navigation
- **Previous:** [Day 97: π₀.5](day-97-pi0-5.md)
- **Next:** [Day 99: GR-2 Details](../week-15/day-99-gr2.md)
- **Week:** [Week 14 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 98.1 The Video Prediction Approach

Traditional VLAs: observation → action  
Video-prediction VLAs: observation → future video → action

```
Traditional pipeline:
  image + language ──→ VLA ──→ actions

Video prediction pipeline:
  image + language ──→ Video Model ──→ future frames
                                            │
                                            ▼
                                      Inverse dynamics
                                            │
                                            ▼
                                         actions
```

### 98.2 GR-2: Generative Robot 2

```
┌─────────────────────────────────────────────────────────┐
│                         GR-2                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Pre-training: video generation on Internet videos       │
│  Fine-tuning: video + action prediction on robot data    │
│                                                          │
│  Architecture:                                           │
│    Image (t) ──→ Video DiT ──→ Image (t+1..t+K)         │
│    Language  ──┘                    │                     │
│                            Action prediction head        │
│                                    │                     │
│                               actions (t)                │
│                                                          │
│  Key insight:                                            │
│    Internet videos teach physics understanding           │
│    Robot videos teach action correspondence              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 98.3 Why Video Pre-training Helps

Internet videos contain implicit physics knowledge:

```
Video of someone pouring water:
  → Model learns: liquid flows down, containers have openings
  → Transfer: robot learns to pour without spilling

Video of folding clothes:
  → Model learns: fabric deforms, fold lines, symmetry
  → Transfer: robot learns folding strategies

Video of cooking:
  → Model learns: object interactions, tool use, sequences
  → Transfer: robot learns manipulation sequences
```

### 98.4 GROOT N1

NVIDIA's approach to video-conditioned action generation:

```
┌─────────────────────────────────────────────────────────┐
│                      GROOT N1                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Dual-system architecture:                               │
│                                                          │
│  System 1: Vision-Language Planner                       │
│    Input:  current image + instruction                   │
│    Output: target image (what the scene should look like)│
│                                                          │
│  System 2: Action Diffusion Policy                       │
│    Input:  current image + target image                  │
│    Output: action trajectory to reach target             │
│                                                          │
│  ┌──────────┐     target     ┌──────────────┐           │
│  │  Planner │ ────image────→ │  Actor       │           │
│  │  (VLM)   │                │  (Diffusion) │           │
│  └──────────┘                └──────────────┘           │
│       ↑                            ↑                     │
│    language                    current                   │
│  + current img                  image                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 98.5 Comparison: Video-to-Action Models

| Model | Video Role | Action Head | Pre-training |
|-------|-----------|-------------|--------------|
| GR-2 | Direct prediction | Learned alongside | Internet video |
| GROOT N1 | Target image | Diffusion policy | Mixed |
| UniPi | Future frame plan | Inverse dynamics | Video + robot |
| SuSIE | Sub-goal image | Low-level policy | Internet video |

### 98.6 Trade-offs

```
Advantages of video prediction:
  + Massive pre-training data (Internet)
  + Physics understanding emerges
  + Visual planning (interpretable)
  + Sim-to-real transfer via visual similarity

Disadvantages:
  - Slow (generate full video frames)
  - Compounding errors in long horizons
  - Action extraction adds complexity
  - Generated frames may hallucinate
```

---

## Implementation (60 min)

### Simplified Video-to-Action Pipeline

```python
import torch
import torch.nn as nn

class VideoPredictor(nn.Module):
    """Simplified video frame predictor."""

    def __init__(self, img_channels=3, hidden_dim=256, n_future=4):
        super().__init__()
        self.n_future = n_future

        # Encode current frame
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 4, stride=2, padding=1), nn.ReLU(),
        )

        # Language conditioning
        self.lang_proj = nn.Linear(256, hidden_dim)

        # Predict future (simple ConvTranspose decoder per frame)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, 128, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1), nn.Sigmoid(),
            )
            for _ in range(n_future)
        ])

    def forward(self, current_frame, lang_embed):
        """Predict n_future frames."""
        B = current_frame.shape[0]
        z = self.encoder(current_frame)  # (B, H, h, w)

        # Condition on language (global add)
        lang = self.lang_proj(lang_embed).view(B, -1, 1, 1)
        z = z + lang

        future_frames = [decoder(z) for decoder in self.decoders]
        return torch.stack(future_frames, dim=1)  # (B, n_future, C, H, W)

class InverseDynamics(nn.Module):
    """Predict action from (current_frame, next_frame)."""

    def __init__(self, img_channels=3, action_dim=7, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels * 2, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, current, next_frame):
        """Predict action that transitions current → next."""
        paired = torch.cat([current, next_frame], dim=1)
        return self.encoder(paired)

class VideoToActionPipeline(nn.Module):
    """Complete video prediction → action pipeline."""

    def __init__(self, action_dim=7, n_future=4):
        super().__init__()
        self.video_pred = VideoPredictor(n_future=n_future)
        self.inv_dynamics = InverseDynamics(action_dim=action_dim)

    def forward(self, current_frame, lang_embed, gt_futures=None, gt_actions=None):
        """
        Training: predict future frames + inverse dynamics.
        """
        # Video prediction loss
        pred_futures = self.video_pred(current_frame, lang_embed)

        losses = {}
        if gt_futures is not None:
            losses["video"] = ((pred_futures - gt_futures)**2).mean()

        # Inverse dynamics loss (use ground truth frames for training)
        if gt_futures is not None and gt_actions is not None:
            pred_actions = []
            for t in range(gt_futures.shape[1]):
                if t == 0:
                    prev = current_frame
                else:
                    prev = gt_futures[:, t-1]
                action = self.inv_dynamics(prev, gt_futures[:, t])
                pred_actions.append(action)
            pred_actions = torch.stack(pred_actions, dim=1)
            losses["action"] = ((pred_actions - gt_actions)**2).mean()

        return losses

    @torch.no_grad()
    def predict(self, current_frame, lang_embed):
        """Inference: predict video → extract actions."""
        pred_futures = self.video_pred(current_frame, lang_embed)

        actions = []
        prev = current_frame
        for t in range(pred_futures.shape[1]):
            action = self.inv_dynamics(prev, pred_futures[:, t])
            actions.append(action)
            prev = pred_futures[:, t]

        return torch.stack(actions, dim=1)  # (B, n_future, action_dim)

# Demo
pipeline = VideoToActionPipeline(action_dim=7, n_future=4)
frame = torch.randn(2, 3, 64, 64)
lang = torch.randn(2, 256)
gt_future = torch.randn(2, 4, 3, 64, 64).sigmoid()
gt_act = torch.randn(2, 4, 7)

losses = pipeline(frame, lang, gt_future, gt_act)
print(f"Video loss: {losses['video'].item():.4f}")
print(f"Action loss: {losses['action'].item():.4f}")

pred = pipeline.predict(frame, lang)
print(f"Predicted actions: {pred.shape}")  # (2, 4, 7)
```

---

## Exercise (45 min)

1. **Video prediction quality:** Train the video predictor on a simple dataset (e.g., moving MNIST). Visualize predicted vs actual future frames. After how many steps do predictions diverge?

2. **Inverse dynamics accuracy:** Compare inverse dynamics (image pair → action) vs direct action prediction (image → action). Which is more accurate? Why?

3. **Target image vs trajectory:** Compare GROOT N1's approach (predict one target image) vs GR-2's approach (predict full video sequence). When does each work better?

4. **Internet pre-training simulation:** Pre-train video predictor on non-robot videos (e.g., bouncing balls). Then fine-tune on robot data. Does pre-training help? Measure in terms of sample efficiency.

---

## Key Takeaways

1. **Video prediction models** leverage Internet-scale data for physics understanding
2. **GR-2** directly predicts future video + actions simultaneously
3. **GROOT N1** uses a planner (target image) + actor (diffusion policy) separation
4. **Inverse dynamics** extracts actions from predicted video sequences
5. **Trade-off:** rich visual representation vs computational cost and error compounding

---

## Connection to the Thread

> *Week 14 complete. We've surveyed the VLA landscape: RT-1/RT-2 (tokenized actions), Octo (diffusion head), OpenVLA (open-source VLM), π₀/π₀.5 (flow matching + planning), and video-to-action models (GR-2, GROOT). Week 15 dives deeper: training recipes, sim-to-real, and hybrid architectures that combine the best of each approach.*

---

## Further Reading

- Cheang et al. (2024), "GR-2: A Generative Video-Language-Action Model for Robot Manipulation"
- [NVIDIA GROOT N1](https://developer.nvidia.com/groot)
- Du et al. (2024), "UniPi: Learning Universal Policies via Text-Guided Video Generation"
- Black et al. (2023), "SuSIE: Sub-Goal Synthesis via Image Editing"
