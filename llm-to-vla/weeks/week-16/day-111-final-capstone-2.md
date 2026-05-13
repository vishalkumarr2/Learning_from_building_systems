# Day 111: Final Capstone — Day 2: Implementation

> **Phase VII — VLAs: Architecture to Deployment | Week 16 | 3 hours**
> *"Turn your Day 110 design into working code. Every component, wired together."*

## Navigation
- **Previous:** [Day 110: Final Capstone — Day 1](day-110-final-capstone-1.md)
- **Next:** [Day 112: Final Capstone — Day 3](day-112-final-capstone-3.md)
- **Week:** [Week 16 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Day 2: Implementation (3 hours)

Implement each component from your Day 110 design. Use the skeleton below and fill in based on your architecture choices.

### Part 1: VLA Model (60 min)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KitchenVLA(nn.Module):
    """Complete VLA for kitchen manipulation.
    
    Architecture from Day 110 design:
    - Vision: Pre-trained encoder (frozen or fine-tuned)
    - Language: Pre-trained text encoder
    - Fusion: Cross-attention
    - Action: [Your chosen representation]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # === Vision Encoder ===
        # TODO: Choose based on your Day 110 spec
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 16, config.get("d_model", 256)),
        )

        # === Language Encoder ===
        self.lang_encoder = nn.Sequential(
            nn.Linear(config.get("lang_dim", 128), config.get("d_model", 256)),
            nn.ReLU(),
            nn.Linear(config.get("d_model", 256), config.get("d_model", 256)),
        )

        # === Proprioception Encoder ===
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.get("proprio_dim", 7), 128),
            nn.ReLU(),
            nn.Linear(128, config.get("d_model", 256)),
        )

        # === Fusion (cross-attention) ===
        d_model = config.get("d_model", 256)
        self.fusion = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # === Action Head ===
        # TODO: Implement your chosen action representation
        action_dim = config.get("action_dim", 7)
        chunk_size = config.get("chunk_size", 8)

        self.action_head = self._build_action_head(
            d_model, action_dim, chunk_size,
            config.get("action_type", "flow")
        )

    def _build_action_head(self, d_model, action_dim, chunk_size, action_type):
        """Build action head based on design choice."""
        if action_type == "token":
            n_bins = 256
            return nn.ModuleDict({
                "heads": nn.ModuleList([
                    nn.Linear(d_model, n_bins * chunk_size)
                    for _ in range(action_dim)
                ]),
                "type": nn.Identity(),  # Placeholder
            })
        elif action_type == "flow":
            return nn.ModuleDict({
                "time_embed": nn.Sequential(
                    nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64)),
                "velocity": nn.Sequential(
                    nn.Linear(d_model + action_dim * chunk_size + 64, 512),
                    nn.SiLU(),
                    nn.Linear(512, 512), nn.SiLU(),
                    nn.Linear(512, action_dim * chunk_size)),
            })
        else:
            return nn.Sequential(
                nn.Linear(d_model, 256), nn.ReLU(),
                nn.Linear(256, action_dim * chunk_size),
            )

    def encode(self, image, language, proprio):
        """Encode all modalities."""
        vis_feat = self.vision_encoder(image).unsqueeze(1)  # (B, 1, D)
        lang_feat = self.lang_encoder(language).unsqueeze(1)  # (B, 1, D)
        prop_feat = self.proprio_encoder(proprio).unsqueeze(1)  # (B, 1, D)

        # Concatenate modalities as sequence
        tokens = torch.cat([vis_feat, lang_feat, prop_feat], dim=1)

        # Self-attention fusion
        fused, _ = self.fusion(tokens, tokens, tokens)
        fused = self.fusion_norm(fused + tokens)

        # Pool to single feature vector
        return fused.mean(dim=1)  # (B, D)

    def predict_actions_flow(self, features, n_steps=10):
        """Flow matching action prediction."""
        B = features.shape[0]
        action_dim = self.config.get("action_dim", 7)
        chunk_size = self.config.get("chunk_size", 8)
        device = features.device

        # Start from noise
        x = torch.randn(B, action_dim * chunk_size, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=device)
            t_emb = self.action_head["time_embed"](t)
            inp = torch.cat([features, x, t_emb], dim=-1)
            velocity = self.action_head["velocity"](inp)
            x = x + dt * velocity

        return x.view(B, chunk_size, action_dim)

    def forward(self, image, language, proprio, target_actions=None):
        """Full forward pass."""
        features = self.encode(image, language, proprio)

        action_type = self.config.get("action_type", "flow")
        if action_type == "flow":
            actions = self.predict_actions_flow(features)
        else:
            actions = self.action_head(features)
            actions = actions.view(
                -1, self.config.get("chunk_size", 8),
                self.config.get("action_dim", 7))

        if target_actions is not None:
            loss = F.mse_loss(actions, target_actions)
            return loss, actions
        return actions
```

### Part 2: Training Loop (45 min)

```python
class KitchenVLATrainer:
    """Training pipeline for the kitchen VLA."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train_stage(self, train_data, val_data, stage, n_epochs):
        """Train one stage of the multi-stage recipe."""
        # Configure optimizer based on stage
        if stage == 1:
            # Freeze vision, train projection only
            for name, param in self.model.named_parameters():
                param.requires_grad = "fusion" in name or "action" in name
            lr = 1e-3
        elif stage == 2:
            # Unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
            lr = 2e-5
        else:
            # Action head focus
            for name, param in self.model.named_parameters():
                param.requires_grad = "action" in name
            lr = 1e-5

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=0.01
        )

        best_val_loss = float("inf")
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_losses = []
            for batch in train_data:
                loss, _ = self.model(
                    batch["image"], batch["language"],
                    batch["proprio"], batch["actions"]
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_data:
                    loss, _ = self.model(
                        batch["image"], batch["language"],
                        batch["proprio"], batch["actions"]
                    )
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"best_stage{stage}.pt")

            if epoch % 5 == 0:
                print(f"  Stage {stage} Epoch {epoch}: "
                      f"train={train_loss:.4f}, val={val_loss:.4f}")

        return best_val_loss

    def train_full(self, train_data, val_data):
        """Complete 3-stage training."""
        print("Stage 1: Projection Alignment")
        self.train_stage(train_data, val_data, stage=1, n_epochs=10)

        print("\nStage 2: Co-Fine-Tuning")
        self.train_stage(train_data, val_data, stage=2, n_epochs=50)

        print("\nStage 3: Task Specialization")
        self.train_stage(train_data, val_data, stage=3, n_epochs=20)
```

### Part 3: Deployment Wrapper (45 min)

```python
class KitchenVLADeployment:
    """Production deployment wrapper."""

    def __init__(self, model, safety_config):
        self.model = model
        self.model.eval()

        # Safety
        self.workspace = safety_config.get("workspace", [(-0.5, 0.5)] * 7)
        self.max_vel = safety_config.get("max_velocity", 0.5)
        self.prev_action = None

        # Monitoring
        self.step_count = 0
        self.successes = 0

        # Action chunk buffer
        self.chunk_buffer = None
        self.chunk_idx = 0

    @torch.no_grad()
    def step(self, image, language, proprio):
        """One control step."""
        # Check if we need a new chunk
        if self.chunk_buffer is None or self.chunk_idx >= len(self.chunk_buffer):
            # Full VLA inference
            image_t = torch.FloatTensor(image).unsqueeze(0)
            lang_t = torch.FloatTensor(language).unsqueeze(0)
            prop_t = torch.FloatTensor(proprio).unsqueeze(0)
            self.chunk_buffer = self.model(image_t, lang_t, prop_t)[0]
            self.chunk_idx = 0

        # Get action from chunk
        action = self.chunk_buffer[self.chunk_idx].numpy()
        self.chunk_idx += 1

        # Safety filter
        action = self._safety_filter(action)
        self.step_count += 1

        return action

    def _safety_filter(self, action):
        """Apply safety constraints."""
        safe = action.copy()
        # Workspace clipping
        for i, (lo, hi) in enumerate(self.workspace):
            safe[i] = np.clip(safe[i], lo, hi)
        # Velocity limiting
        if self.prev_action is not None:
            vel = safe - self.prev_action
            speed = np.linalg.norm(vel)
            if speed > self.max_vel * 0.02:  # dt=0.02
                safe = self.prev_action + vel / speed * self.max_vel * 0.02
        self.prev_action = safe.copy()
        return safe

# Integration test
config = {
    "d_model": 256, "lang_dim": 128, "proprio_dim": 7,
    "action_dim": 7, "chunk_size": 8, "action_type": "flow",
}
model = KitchenVLA(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
img = torch.randn(2, 3, 64, 64)
lang = torch.randn(2, 128)
prop = torch.randn(2, 7)
target = torch.randn(2, 8, 7)

loss, actions = model(img, lang, prop, target)
print(f"Loss: {loss.item():.4f}")
print(f"Actions shape: {actions.shape}")

# Test deployment
deploy = KitchenVLADeployment(model, {"max_velocity": 0.5})
for i in range(20):
    action = deploy.step(
        np.random.randn(3, 64, 64).astype(np.float32),
        np.random.randn(128).astype(np.float32),
        np.random.randn(7).astype(np.float32),
    )
    if i % 5 == 0:
        print(f"Step {i}: action={action[:3]}")
```

---

## Deliverables for Day 2

By end of session, you should have:
- [ ] Working VLA model with chosen architecture
- [ ] Training pipeline with 3-stage schedule
- [ ] Deployment wrapper with safety and chunk recycling
- [ ] Verified forward pass (shapes, gradients, no NaN)
- [ ] Code runs end-to-end (even if with dummy data)

---

## Connection to the Thread

> *Model built. Tomorrow (Day 112): final evaluation, comprehensive testing, and the curriculum-closing reflection. You'll run your VLA through a gauntlet of tests and write your synthesis of the entire 16-week journey.*
