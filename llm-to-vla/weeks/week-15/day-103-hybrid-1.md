# Day 103: Hybrid VLA Architectures — Day 1

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2.5 hours**
> *"The next generation of VLAs won't be purely one design. They'll combine tokenized actions with diffusion, planning with reactive control, large VLMs with small action experts."*

## Navigation
- **Previous:** [Day 102: Sim-to-Real Day 2](day-102-sim-to-real-2.md)
- **Next:** [Day 104: Hybrid VLA Architectures — Day 2](day-104-hybrid-2.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 103.1 The Convergence Pattern

Every VLA design has trade-offs. The field is converging on hybrids:

```
Design Dimension        Options                 Hybrid Resolution
──────────────────────────────────────────────────────────────────
Action representation   Tokens vs Continuous    → Tokens for coarse +
                                                   flow for fine control

Planning depth          Reactive vs Planning    → VLM plans sub-goals +
                                                   action head executes

Model size              Large vs Small          → Large VLM (frozen) +
                                                   small action expert

Training data           Web vs Robot            → Pre-train on web +
                                                   fine-tune on robot

Inference speed         Fast vs Expressive      → Parallel decode coarse +
                                                   iterative refine
```

### 103.2 Hybrid Pattern 1: Coarse-to-Fine Actions

```
Step 1: Tokenized coarse action (fast, 1 forward pass)
  "move right, down, close gripper"
  Resolution: 256 bins → ~0.4mm per bin

Step 2: Continuous refinement (flow matching, 10 steps)
  Refine the coarse action to sub-millimeter precision
  Resolution: continuous → <0.1mm

┌──────────────────┐     ┌──────────────────┐
│  Coarse Head     │ ──→ │  Refinement Head  │
│  (tokenized,     │     │  (flow matching,  │
│   parallel)      │     │   conditioned on  │
│                  │     │   coarse action)  │
│  Speed: ~10ms    │     │  Speed: ~50ms     │
└──────────────────┘     └──────────────────┘

Total: ~60ms = 16 Hz (sufficient for manipulation)
```

### 103.3 Hybrid Pattern 2: Hierarchical Plan + Act

Building on π₀.5's insight, with more structure:

```
Level 3: Strategic (VLM, ~1 Hz)
  "To make coffee: grind beans → boil water → brew → pour"
  Input: instruction + scene
  Output: ordered sub-goal list

Level 2: Tactical (small transformer, ~5 Hz)
  "Grind beans: open grinder lid → add beans → close → press button"
  Input: sub-goal + current observation
  Output: waypoints or target poses

Level 1: Motor (diffusion/flow, ~20 Hz)
  Generate smooth joint trajectories between waypoints
  Input: target pose + current state
  Output: action chunk (100ms horizon)
```

### 103.4 Hybrid Pattern 3: Mixture of Action Experts

```
┌─────────────────────────────────────────────────────────┐
│  Shared VLM Backbone (frozen)                           │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Expert 1 │  │  Expert 2 │  │  Expert 3 │             │
│  │  Token    │  │  Diffusion│  │  Flow     │             │
│  │  Head     │  │  Head     │  │  Head     │             │
│  │           │  │           │  │           │             │
│  │  Fast     │  │  Multimod │  │  Smooth   │             │
│  │  Coarse   │  │  Actions  │  │  Dexterous│             │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘              │
│        │              │              │                    │
│        └──────────────┴──────────────┘                    │
│                       │                                   │
│                   Router (learned)                        │
│                 "Which expert for this task?"             │
└─────────────────────────────────────────────────────────┘
```

### 103.5 Design Decision Framework

When to use which component:

| Requirement | Best Approach |
|------------|---------------|
| Speed (>20 Hz) | Tokenized, parallel decode |
| Precision (<1mm) | Continuous, flow/diffusion |
| Multimodal actions | Diffusion/flow head |
| Language reasoning | Large VLM backbone |
| Novel objects | Web-pretrained vision |
| Dexterous tasks | Long action chunks, flow |
| Simple pick-and-place | Tokenized, small model |

---

## Implementation (60 min)

### Coarse-to-Fine Hybrid

```python
import torch
import torch.nn as nn

class CoarseToFineVLA(nn.Module):
    """Hybrid: tokenized coarse + flow matching refinement."""

    def __init__(self, d_model=256, n_bins=64, action_dim=7,
                 chunk_size=8, refine_steps=5):
        super().__init__()
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.refine_steps = refine_steps

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(512, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Coarse head: tokenized (fast, parallel)
        self.coarse_heads = nn.ModuleList([
            nn.Linear(d_model, n_bins * chunk_size)
            for _ in range(action_dim)
        ])

        # Refinement head: flow matching (precise)
        self.refine_time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
        )
        self.refine_net = nn.Sequential(
            nn.Linear(action_dim * chunk_size + d_model + 64, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, action_dim * chunk_size),
        )

    def coarse_predict(self, features):
        """Fast parallel prediction of discretized actions."""
        B = features.shape[0]
        coarse_actions = []
        for dim, head in enumerate(self.coarse_heads):
            logits = head(features).view(B, self.chunk_size, self.n_bins)
            bin_idx = logits.argmax(dim=-1)  # (B, chunk_size)
            # Convert bins to continuous values ([-1, 1])
            continuous = (bin_idx.float() / (self.n_bins - 1)) * 2 - 1
            coarse_actions.append(continuous)
        return torch.stack(coarse_actions, dim=-1)  # (B, chunk, act_dim)

    def refine(self, features, coarse_actions):
        """Flow matching refinement of coarse actions."""
        B = coarse_actions.shape[0]
        device = coarse_actions.device

        # Start from coarse + small noise
        x = coarse_actions + torch.randn_like(coarse_actions) * 0.1
        dt = 1.0 / self.refine_steps

        for i in range(self.refine_steps):
            t = torch.full((B, 1), i * dt, device=device)
            t_emb = self.refine_time_embed(t)
            inp = torch.cat([x.flatten(1), features, t_emb], dim=-1)
            velocity = self.refine_net(inp).view(B, self.chunk_size, self.action_dim)
            x = x + dt * velocity

        return x

    def forward(self, obs_features, target_actions=None):
        """Full coarse-to-fine prediction."""
        features = self.encoder(obs_features)

        # Coarse prediction (parallel, fast)
        coarse = self.coarse_predict(features)

        # Refinement (flow matching, precise)
        refined = self.refine(features, coarse)

        if target_actions is not None:
            # Training losses
            coarse_loss = ((coarse - target_actions)**2).mean()
            refine_loss = ((refined - target_actions)**2).mean()
            return coarse_loss + refine_loss, coarse, refined
        return refined

# Hierarchical Plan-Act Architecture
class HierarchicalVLA(nn.Module):
    """Three-level: strategic → tactical → motor."""

    def __init__(self, d_model=256, n_subgoals=5, n_waypoints=4,
                 action_dim=7, chunk_size=8):
        super().__init__()
        # Level 3: Strategic planner (VLM)
        self.strategic = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, n_subgoals * d_model),
        )

        # Level 2: Tactical planner
        self.tactical = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Linear(d_model, n_waypoints * action_dim),
        )

        # Level 1: Motor controller (generates smooth trajectories)
        self.motor = nn.Sequential(
            nn.Linear(action_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, chunk_size * action_dim),
        )

        self.n_subgoals = n_subgoals
        self.n_waypoints = n_waypoints
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(self, scene_features, current_state):
        B = scene_features.shape[0]
        d = scene_features.shape[-1]

        # Level 3: What to do (sub-goals)
        subgoals = self.strategic(scene_features).view(B, self.n_subgoals, d)

        # Level 2: How to do it (waypoints for first sub-goal)
        first_subgoal = subgoals[:, 0]
        tactical_input = torch.cat([scene_features, first_subgoal], dim=-1)
        waypoints = self.tactical(tactical_input).view(
            B, self.n_waypoints, self.action_dim)

        # Level 3: Execute (smooth trajectory to first waypoint)
        first_waypoint = waypoints[:, 0]
        motor_input = torch.cat([current_state, first_waypoint], dim=-1)
        trajectory = self.motor(motor_input).view(
            B, self.chunk_size, self.action_dim)

        return {
            "subgoals": subgoals,
            "waypoints": waypoints,
            "trajectory": trajectory,
        }

# Demo
print("=== Coarse-to-Fine ===")
c2f = CoarseToFineVLA()
obs = torch.randn(4, 512)
target = torch.randn(4, 8, 7)
loss, coarse, refined = c2f(obs, target)
print(f"Coarse error: {((coarse - target)**2).mean():.4f}")
print(f"Refined error: {((refined - target)**2).mean():.4f}")

print("\n=== Hierarchical ===")
hier = HierarchicalVLA()
scene = torch.randn(4, 256)
state = torch.randn(4, 7)
result = hier(scene, state)
print(f"Subgoals: {result['subgoals'].shape}")
print(f"Waypoints: {result['waypoints'].shape}")
print(f"Trajectory: {result['trajectory'].shape}")
```

---

## Exercise (45 min)

1. **Coarse-to-fine benefit:** Compare coarse-only (64 bins) vs coarse+refine on a precision placement task (target tolerance: 1mm). Measure final position error.

2. **Hierarchical depth:** Compare 2-level (plan+act) vs 3-level (strategic+tactical+motor). At what task complexity does the third level help?

3. **Expert routing:** Implement a simple router that selects between token head and diffusion head based on task difficulty. Does routing outperform a single head?

4. **Latency budget:** Profile each component. Given a 50ms budget (20 Hz), what's the optimal allocation between coarse and refinement?

---

## Key Takeaways

1. **Hybrid VLAs combine strengths** — tokens for speed, flow for precision, VLMs for reasoning
2. **Coarse-to-fine** gives both speed and precision within a latency budget
3. **Hierarchical control** decomposes complex tasks into manageable levels
4. **Mixture of experts** routes tasks to the best action representation
5. **The field is converging** on these hybrid patterns as the next generation of VLAs

---

## Connection to the Thread

> *Tomorrow: Part 2 of hybrid architectures — multi-embodiment hybrids, online adaptation (learning from failures at deployment), and how to combine all these ideas into a practical deployment-ready VLA system.*

---

## Further Reading

- Review: RT-1 (tokens), Octo (diffusion), π₀ (flow), π₀.5 (planning)
- Shridhar et al. (2023), "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation" (coarse-to-fine)
- Zhao et al. (2024), "Hierarchical Diffusion Policy"
