# Day 104: Hybrid VLA Architectures — Day 2: Deployment & Adaptation

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2.5 hours**
> *"A VLA that works in the lab but fails in deployment is worth nothing. The last mile is multi-embodiment, online adaptation, and robustness."*

## Navigation
- **Previous:** [Day 103: Hybrid Architectures Day 1](day-103-hybrid-1.md)
- **Next:** [Day 105: Stop & Reflect #6](day-105-stop-reflect-6.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 104.1 Multi-Embodiment Hybrid

A single VLA serving multiple robot form factors:

```
┌─────────────────────────────────────────────────────────┐
│             MULTI-EMBODIMENT VLA                         │
│                                                          │
│  Shared VLM Backbone (frozen, web-pretrained)            │
│  "Understands objects, language, spatial relationships"   │
│                                                          │
│  ┌───────────┬───────────┬───────────┬───────────┐      │
│  │ 7-DOF Arm │ Bimanual  │ Mobile    │ Dexterous │      │
│  │ (Franka)  │ (ALOHA)   │ (Hello)   │ (LEAP)    │      │
│  │           │           │           │           │      │
│  │ Token     │ Flow      │ Token     │ Flow      │      │
│  │ head      │ head      │ head      │ head      │      │
│  │ 7 dims    │ 14 dims   │ 3 dims    │ 24 dims   │      │
│  └───────────┴───────────┴───────────┴───────────┘      │
│                                                          │
│  Embodiment token: [EMB_FRANKA] [EMB_ALOHA] etc.        │
│  Prepended to input sequence, selects right head         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 104.2 Online Adaptation

The robot learns from its deployment experience:

```
Deployment Loop:
  1. Execute action from VLA
  2. Human corrects if needed (teleoperation override)
  3. Store (observation, correction) pair
  4. Every N corrections: fine-tune action head
  5. Performance improves over deployment lifetime

Key constraints:
  - Fine-tuning must be fast (<1 min for batch)
  - Must not degrade on already-learned tasks
  - Must handle distributional shift gracefully
```

```python
# Online adaptation with experience replay
class OnlineAdapter:
    def __init__(self, model, buffer_size=1000, adapt_every=50):
        self.model = model
        self.buffer = []
        self.buffer_size = buffer_size
        self.adapt_every = adapt_every
        self.step_count = 0
        self.optimizer = torch.optim.Adam(
            model.action_head.parameters(), lr=1e-5
        )

    def add_correction(self, obs, corrected_action):
        """Human provides correction."""
        self.buffer.append((obs, corrected_action))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)  # FIFO
        self.step_count += 1

        if self.step_count % self.adapt_every == 0:
            self.adapt()

    def adapt(self, n_steps=10):
        """Quick fine-tuning on correction buffer."""
        if len(self.buffer) < 10:
            return

        self.model.train()
        for _ in range(n_steps):
            # Sample mini-batch from buffer
            indices = np.random.choice(len(self.buffer), min(32, len(self.buffer)))
            obs_batch = torch.stack([self.buffer[i][0] for i in indices])
            act_batch = torch.stack([self.buffer[i][1] for i in indices])

            loss = self.model.action_loss(obs_batch, act_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
```

### 104.3 Robustness Patterns

```
Production VLA Robustness Checklist:
  
  ☐ Action safety bounds
    - Clip actions to workspace limits
    - Velocity limiting (max joint speed)
    - Force/torque thresholds → emergency stop
  
  ☐ Confidence estimation
    - Entropy of action distribution
    - Ensemble disagreement
    - Out-of-distribution detection
  
  ☐ Graceful degradation
    - If confidence < threshold: pause + ask human
    - If consecutive failures > N: escalate
    - If novel object detected: use conservative policy
  
  ☐ Monitoring
    - Success rate over time (should not drop)
    - Action distribution shift detection
    - Latency tracking (must stay < budget)
```

### 104.4 Inference Optimization

Making VLAs run fast enough for real-time control:

| Technique | Speedup | Quality Impact |
|-----------|---------|---------------|
| Quantization (INT8) | 2-3× | Minimal (<1%) |
| KV cache | 2× (autoregressive) | None |
| TensorRT / ONNX | 2-5× | None |
| Vision encoder caching | 3× | None (if scene static) |
| Action chunk recycling | 4-8× | Depends on horizon |
| LoRA merge (remove adapters) | 1.5× | None |

```python
# Action chunk recycling: don't recompute every step
class ChunkRecycler:
    def __init__(self, model, chunk_size=16, recompute_every=4):
        self.model = model
        self.chunk_size = chunk_size
        self.recompute_every = recompute_every
        self.current_chunk = None
        self.chunk_index = 0

    def get_action(self, observation):
        if self.current_chunk is None or self.chunk_index >= self.recompute_every:
            # Recompute: expensive VLA forward pass
            self.current_chunk = self.model.predict(observation)
            self.chunk_index = 0

        action = self.current_chunk[self.chunk_index]
        self.chunk_index += 1
        return action

    # Effective frequency: if model runs at 5 Hz and chunk_size=16,
    # recycling at recompute_every=4 gives 20 Hz effective control
```

### 104.5 Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│              PRODUCTION VLA DEPLOYMENT                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Camera (30 Hz) ──→ Image buffer ──→ VLA Server          │
│  Proprio (100 Hz) ─┘                    │                │
│                                     ┌───▼────────┐      │
│                                     │ VLM (GPU)  │      │
│                                     │ Batch=1    │      │
│                                     │ FP16/INT8  │      │
│                                     └───┬────────┘      │
│                                         │                │
│                                  ┌──────▼────────┐      │
│                                  │ Action Head   │      │
│                                  │ (chunk=16)    │      │
│                                  └──────┬────────┘      │
│                                         │                │
│  ┌──────────────┐               ┌───────▼────────┐      │
│  │ Safety Layer │ ←─────────── │ Action Buffer  │      │
│  │ - Workspace  │               │ (recycling)    │      │
│  │ - Velocity   │               └────────────────┘      │
│  │ - Force      │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  Robot Controller (1000 Hz)                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation (60 min)

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class SafetyLayer:
    """Enforce safety constraints on predicted actions."""

    def __init__(self, workspace_bounds, max_velocity, max_force):
        self.workspace_bounds = workspace_bounds  # (low, high) per dim
        self.max_velocity = max_velocity
        self.max_force = max_force
        self.prev_action = None

    def __call__(self, action, dt=0.02):
        """Apply safety constraints."""
        safe_action = action.copy()

        # Workspace clipping
        for i, (low, high) in enumerate(self.workspace_bounds):
            safe_action[i] = np.clip(safe_action[i], low, high)

        # Velocity limiting
        if self.prev_action is not None:
            velocity = (safe_action - self.prev_action) / dt
            speed = np.linalg.norm(velocity)
            if speed > self.max_velocity:
                safe_action = self.prev_action + \
                    velocity / speed * self.max_velocity * dt

        self.prev_action = safe_action.copy()
        return safe_action

class ConfidenceEstimator:
    """Estimate policy confidence for human handoff."""

    def __init__(self, model, n_samples=5, entropy_threshold=2.0):
        self.model = model
        self.n_samples = n_samples
        self.threshold = entropy_threshold

    def estimate(self, observation):
        """Return confidence score and whether to proceed."""
        actions = []
        for _ in range(self.n_samples):
            action = self.model.predict(observation)
            actions.append(action)

        actions = torch.stack(actions)
        # Disagreement = std across samples
        disagreement = actions.std(dim=0).mean().item()
        # Confidence = inverse of disagreement
        confidence = 1.0 / (1.0 + disagreement)

        return {
            "confidence": confidence,
            "proceed": confidence > 0.5,
            "mean_action": actions.mean(dim=0),
            "disagreement": disagreement,
        }

class ProductionVLA:
    """Complete deployment wrapper."""

    def __init__(self, model, safety_config, chunk_size=16, recompute_every=4):
        self.model = model
        self.model.eval()

        self.safety = SafetyLayer(**safety_config)
        self.confidence = ConfidenceEstimator(model)
        self.recycler = ChunkRecycler(model, chunk_size, recompute_every)
        self.adapter = OnlineAdapter(model)

        # Monitoring
        self.action_history = deque(maxlen=1000)
        self.success_count = 0
        self.total_count = 0

    @torch.no_grad()
    def step(self, observation):
        """One control step."""
        # Confidence check
        conf = self.confidence.estimate(observation)
        if not conf["proceed"]:
            return None, "LOW_CONFIDENCE"

        # Get action (with chunk recycling)
        action = self.recycler.get_action(observation)

        # Safety filter
        safe_action = self.safety(action.numpy())

        # Monitor
        self.action_history.append(safe_action)

        return safe_action, "OK"

    def report_outcome(self, success):
        """Track deployment performance."""
        self.total_count += 1
        if success:
            self.success_count += 1

    @property
    def success_rate(self):
        if self.total_count == 0:
            return 0
        return self.success_count / self.total_count

# Demo
print("Production VLA Deployment Stack")
print("Components: Safety + Confidence + Recycling + Adaptation + Monitoring")

safety_config = {
    "workspace_bounds": [(-0.5, 0.5)] * 7,
    "max_velocity": 1.0,
    "max_force": 20.0,
}
safety = SafetyLayer(**safety_config)
action = np.array([0.3, -0.8, 0.1, 0.0, 0.0, 0.0, 0.5])
safe = safety(action)
print(f"Raw action:  {action}")
print(f"Safe action: {safe}")
```

---

## Exercise (45 min)

1. **Multi-embodiment transfer:** Train a shared backbone on two simulated robots (3-DOF and 7-DOF). Add embodiment tokens. Does cross-robot training help?

2. **Online adaptation:** Deploy a VLA, inject 20 corrections, fine-tune. Compare pre- and post-adaptation success rates. Plot the learning curve.

3. **Safety layer stress test:** Generate 1000 random actions. Apply the safety layer. What percentage are modified? Visualize the clipping distribution.

4. **Inference optimization:** Measure latency with FP32, FP16, and INT8 quantization. Plot accuracy vs speed trade-off.

---

## Key Takeaways

1. **Multi-embodiment** shares a backbone across robots with per-embodiment action heads
2. **Online adaptation** lets the VLA improve from deployment corrections
3. **Safety layers** are non-negotiable for production deployment
4. **Confidence estimation** enables human-in-the-loop fallback
5. **Chunk recycling + quantization** makes real-time VLA inference practical

---

## Connection to the Thread

> *We've covered the complete VLA lifecycle: architecture → training → sim-to-real → hybrid design → deployment. Tomorrow: Stop & Reflect #6, consolidating everything from Phase VII Weeks 14-15 before the final capstone.*

---

## Further Reading

- Open X-Embodiment Collaboration (2024), "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
- Brohan et al. (2023), "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (deployment details)
- Physical Intelligence blog posts on π₀ deployment
