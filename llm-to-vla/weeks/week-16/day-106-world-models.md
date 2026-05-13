# Day 106: World Models for Robot Control

> **Phase VII — VLAs: Architecture to Deployment | Week 16 | 2.5 hours**
> *"A robot that can imagine the consequences of its actions before executing them is fundamentally more capable." — World Models for Robotics*

## Navigation
- **Previous:** [Day 105: Stop & Reflect #6](../week-15/day-105-stop-reflect-6.md)
- **Next:** [Day 107: Deployment — Latency & Compute](day-107-deployment-1.md)
- **Week:** [Week 16 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 106.1 What Is a World Model?

A world model predicts what happens next given current state and action:

$$\hat{s}_{t+1} = f_\theta(s_t, a_t)$$

For vision-based robots, this becomes:

$$\hat{o}_{t+1} = f_\theta(o_{1:t}, a_t)$$

where $o$ are observations (images) and $a$ are actions.

```
Classical control:        World model control:
  Observe → Act            Observe → Imagine → Evaluate → Act
  
  No lookahead             Simulate multiple futures
  Reactive only            Proactive planning
  One step at a time       Multi-step optimization
```

### 106.2 World Model Architectures

```
Type 1: Latent dynamics (RSSM, Dreamer)
  o_t → encoder → z_t ─┐
                        ├→ transition → ẑ_{t+1} → decoder → ô_{t+1}
  a_t ─────────────────┘
  
  + Fast (operates in latent space)
  + Can unroll many steps
  - Latent space may miss details

Type 2: Video prediction (GR-2, UniSim)
  o_{1:t} + a_t → video diffusion → ô_{t+1:t+H}
  
  + Rich visual detail
  + Can evaluate plans visually
  - Slow (full frame generation)
  - Expensive to unroll

Type 3: Tokenized prediction (GAIA-1, Genie)
  Tokenize observation → predict next tokens → detokenize
  
  + Compatible with LLM training
  + Discrete, easy to model
  - Quantization loses precision
```

### 106.3 Planning with World Models

**Model Predictive Control (MPC) with learned dynamics:**

$$a^*_{0:H} = \arg\min_{a_{0:H}} \sum_{t=0}^{H} c(f_\theta^{(t)}(s_0, a_{0:t}), g)$$

where $c$ is cost function, $g$ is goal, $H$ is planning horizon.

```python
# Cross-Entropy Method (CEM) planning
def cem_plan(world_model, current_obs, goal, n_candidates=64,
             n_elite=8, horizon=10, n_iters=5, action_dim=7):
    """Plan actions by imagining futures and selecting the best."""
    # Initialize action distribution
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * 0.5

    for iteration in range(n_iters):
        # Sample candidate action sequences
        candidates = np.random.normal(
            mean, std, size=(n_candidates, horizon, action_dim)
        )
        candidates = np.clip(candidates, -1, 1)

        # Evaluate each candidate by imagining the future
        costs = []
        for actions in candidates:
            obs = current_obs
            total_cost = 0
            for t in range(horizon):
                obs = world_model.predict(obs, actions[t])
                total_cost += cost_fn(obs, goal)
            costs.append(total_cost)

        # Select elite candidates
        elite_idx = np.argsort(costs)[:n_elite]
        elite_actions = candidates[elite_idx]

        # Update distribution
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0) + 1e-6

    return mean[0]  # Execute first action
```

### 106.4 VLAs as Implicit World Models

Key insight: VLAs trained on video prediction are implicit world models.

```
GR-2:    Predicts future frames → actions are a byproduct of
         understanding dynamics

π₀:      Flow matching generates trajectories → implicitly models
         how actions change the scene

RT-2:    VLM backbone encodes "common-sense physics" from web
         pre-training → world knowledge without explicit dynamics
```

### 106.5 World Model + VLA Integration

```
Option A: World model for planning, VLA for execution
  World model plans sub-goals → VLA executes each sub-goal
  
  + VLA is reactive, handles perturbations
  + World model provides long-horizon foresight
  
Option B: World model for data augmentation
  Train world model → generate synthetic experience → train VLA
  
  + More training data (free)
  + Can generate edge cases
  - Quality depends on world model accuracy
  
Option C: World model inside VLA
  VLA's backbone IS the world model (GR-2 approach)
  
  + Single model, simple deployment
  + Video pre-training = world model pre-training
  - Computationally expensive
```

---

## Implementation (60 min)

### Latent World Model + Planning

```python
import torch
import torch.nn as nn
import numpy as np

class LatentWorldModel(nn.Module):
    """RSSM-inspired latent dynamics model."""

    def __init__(self, obs_dim=512, action_dim=7, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: observation → latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean, logvar
        )

        # Transition: latent + action → next latent
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean, logvar
        )

        # Reward predictor: latent → reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Decoder: latent → observation (for visualization)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode(self, obs):
        """Encode observation to latent."""
        params = self.encoder(obs)
        mean, logvar = params.chunk(2, dim=-1)
        # Reparameterization
        std = (0.5 * logvar).exp()
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def predict_next(self, z, action):
        """Predict next latent state."""
        inp = torch.cat([z, action], dim=-1)
        params = self.transition(inp)
        mean, logvar = params.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        z_next = mean + std * torch.randn_like(std)
        return z_next, mean, logvar

    def predict_reward(self, z):
        """Predict reward from latent state."""
        return self.reward_head(z)

    def imagine(self, z_start, action_sequence):
        """Unroll dynamics in latent space."""
        z = z_start
        trajectory = [z]
        rewards = []

        for action in action_sequence:
            z, _, _ = self.predict_next(z, action)
            r = self.predict_reward(z)
            trajectory.append(z)
            rewards.append(r)

        return torch.stack(trajectory), torch.cat(rewards)

class WorldModelPlanner:
    """MPC planning with learned world model."""

    def __init__(self, world_model, action_dim=7, horizon=10,
                 n_candidates=256, n_elite=32, n_iters=5):
        self.wm = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iters = n_iters

    @torch.no_grad()
    def plan(self, observation, goal_latent=None):
        """CEM planning in latent space."""
        self.wm.eval()

        # Encode current observation
        z_start, _, _ = self.wm.encode(observation.unsqueeze(0))

        # Initialize action distribution
        mean = torch.zeros(self.horizon, self.action_dim)
        std = torch.ones(self.horizon, self.action_dim) * 0.3

        for _ in range(self.n_iters):
            # Sample candidates
            noise = torch.randn(self.n_candidates, self.horizon, self.action_dim)
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates = torch.clamp(candidates, -1, 1)

            # Evaluate each candidate
            returns = []
            for i in range(self.n_candidates):
                z = z_start.clone()
                total_reward = 0
                for t in range(self.horizon):
                    action = candidates[i, t].unsqueeze(0)
                    z, _, _ = self.wm.predict_next(z, action)
                    reward = self.wm.predict_reward(z).item()

                    # Optionally add goal distance cost
                    if goal_latent is not None:
                        dist = ((z - goal_latent)**2).sum().item()
                        reward -= 0.1 * dist

                    total_reward += reward * (0.99 ** t)  # Discount
                returns.append(total_reward)

            returns = torch.tensor(returns)

            # Select elites
            elite_idx = returns.topk(self.n_elite).indices
            elites = candidates[elite_idx]

            # Update distribution
            mean = elites.mean(dim=0)
            std = elites.std(dim=0) + 0.05

        return mean[0]  # First action of best plan

# Demo
wm = LatentWorldModel(obs_dim=512, action_dim=7)
planner = WorldModelPlanner(wm, action_dim=7, horizon=10)

# Imagine a trajectory
obs = torch.randn(1, 512)
z, _, _ = wm.encode(obs)
actions = [torch.randn(1, 7) for _ in range(10)]
traj, rewards = wm.imagine(z, actions)

print(f"Imagined trajectory: {traj.shape} ({traj.shape[0]} states)")
print(f"Predicted rewards: {rewards.shape}")
print(f"Total return: {rewards.sum().item():.3f}")

# Plan
best_action = planner.plan(torch.randn(512))
print(f"\nPlanned action: {best_action.shape}")
```

---

## Exercise (45 min)

1. **Imagination quality:** Train the world model on simulated trajectories. Compare imagined trajectories (5, 10, 20 steps) with ground truth. At what horizon does prediction quality degrade?

2. **Planning vs reactive:** Compare (A) MPC with the world model vs (B) direct BC policy. On which task types does planning help more? (Hint: long-horizon, precise placement)

3. **Data augmentation:** Use the world model to generate 10× synthetic data. Train a BC policy on real-only vs real+synthetic. Does the world model data help?

4. **Latent space visualization:** Encode 100 observations, project to 2D with t-SNE. Do similar scenes cluster? Does the latent space capture task-relevant structure?

---

## Key Takeaways

1. **World models predict consequences** — enabling planning before acting
2. **Latent dynamics models** (RSSM) enable fast multi-step imagination
3. **CEM planning** in latent space finds good action sequences without gradients
4. **VLAs trained on video are implicit world models** — GR-2 embodies this
5. **World models complement VLAs** for planning, augmentation, or as the backbone itself

---

## Connection to the Thread

> *World models enable planning. But deployment demands more: latency budgets, compute constraints, edge hardware, fleet management. Days 107-109: three sessions on real-world deployment challenges — the engineering that makes VLAs actually work.*

---

## Further Reading

- Ha & Schmidhuber (2018), "World Models"
- Hafner et al. (2023), "Mastering Diverse Domains through World Models" (DreamerV3)
- Yang et al. (2024), "Learning Interactive Real-World Simulators" (UniSim)
