# Day 78: Imitation Learning — BC, DAgger & Action Chunking

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 12 | 2.5 hours**
> *"Don't define rewards. Just show the robot what to do." — Chelsea Finn*

## Navigation
- **Previous:** [Day 77: Flow Matching](../week-11/day-77-flow-matching.md)
- **Next:** [Day 79: ACT — Action Chunking Transformers](day-79-act.md)
- **Week:** [Week 12 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/13-imitation-learning.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 78.1 Why Imitation Learning?

RL requires reward functions. For robots, rewards are hard to specify:
- "Pick up the mug" — +1 at success? What about grip quality, speed, safety?
- "Navigate to the shelf" — distance reward? What about obstacle avoidance elegance?

**Imitation Learning (IL):** learn a policy $\pi_\theta(a|s)$ from expert demonstrations $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^N$.

### 78.2 Behavioral Cloning (BC)

The simplest IL approach — supervised learning on state-action pairs:

$$\mathcal{L}_\text{BC} = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \| \pi_\theta(s) - a \|^2 \right]$$

```
Expert demos: (s₁,a₁), (s₂,a₂), ..., (sₙ,aₙ)
    │
    ▼
Train: π_θ(s) ≈ a   (supervised regression)
    │
    ▼
Deploy: robot follows learned π_θ
```

**The compounding error problem:**

Small errors accumulate because the policy never sees recovery demonstrations:
1. Policy makes a tiny error at step 1
2. Enters a state slightly different from training data
3. Makes a bigger error at step 2
4. Cascading failures → the robot falls off the demonstrated trajectory

$$\text{Error} \propto T^2 \quad \text{(quadratic in horizon length!)}$$

### 78.3 DAgger: Dataset Aggregation

DAgger fixes compounding errors by **iteratively collecting corrections**:

```
Algorithm DAgger:
  1. Train π₁ on D₀ (initial expert demos)
  2. For i = 1, 2, ...:
     a. Execute πᵢ in the environment
     b. At each visited state s, query expert for optimal action a*
     c. Add (s, a*) to dataset: Dᵢ = Dᵢ₋₁ ∪ {(s, a*)}
     d. Train πᵢ₊₁ on Dᵢ
```

**Key insight:** DAgger ensures the policy sees recovery states in training data.

| Method | Pros | Cons |
|--------|------|------|
| BC | Simple, no env needed | Compounding errors |
| DAgger | Handles distribution shift | Needs online expert |
| BC + data augmentation | No expert needed | Limited diversity |

### 78.4 Action Chunking

Instead of predicting **one action** per step, predict a **sequence of actions**:

$$\pi_\theta(s) = (a_t, a_{t+1}, \ldots, a_{t+H-1})$$

where $H$ is the **chunk size** (horizon).

**Why chunking helps:**
1. **Temporal consistency:** no jittering between conflicting single-step predictions
2. **Multimodality:** easier to capture diverse strategies over a sequence
3. **Compounding error:** fewer decision points = less error accumulation

```
Without chunking (H=1):
  s₁ → a₁, s₂ → a₂, s₃ → a₃, ... (N decisions)

With chunking (H=4):
  s₁ → [a₁,a₂,a₃,a₄], s₅ → [a₅,a₆,a₇,a₈], ... (N/4 decisions)
```

### 78.5 Multimodal Action Distributions

A critical challenge: given the same observation, there may be **multiple valid actions**:

```
Observation: mug on table
Valid actions: grab from left OR grab from right

MSE loss → averages the two → grabs air in the middle!
```

**Solutions:**
- **Gaussian Mixture Models** (GMM)
- **Diffusion Policy** (Day 81) — models full distribution
- **CVAE** — ACT uses this (Day 79)
- **Action tokenization** — discretize and use cross-entropy (Day 83)

---

## Implementation (60 min)

### Behavioral Cloning with LeRobot

```python
# Using HuggingFace LeRobot framework
# pip install lerobot

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np

# --- Simple BC Policy ---
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)

# --- Action Chunking BC ---
class ChunkedBCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_size=4, hidden=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim * chunk_size),
        )
        self.act_dim = act_dim

    def forward(self, obs):
        out = self.net(obs)
        return out.view(-1, self.chunk_size, self.act_dim)

# --- Collect expert demonstrations ---
def collect_expert_demos(env_name="Pendulum-v1", n_episodes=100):
    """Collect demonstrations using a simple PD controller as 'expert'."""
    env = gym.make(env_name)
    demos = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode = {"observations": [], "actions": []}
        for _ in range(200):
            # Simple expert: PD control toward upright
            theta = np.arctan2(obs[1], obs[0])
            action = np.clip([-5.0 * theta - 0.5 * obs[2]], -2.0, 2.0)
            episode["observations"].append(obs)
            episode["actions"].append(action)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        demos.append(episode)
    return demos

# --- Train BC ---
def train_bc(demos, epochs=100, lr=1e-3, chunk_size=1):
    obs = np.concatenate([d["observations"] for d in demos])
    acts = np.concatenate([d["actions"] for d in demos])
    obs_t = torch.FloatTensor(obs)
    acts_t = torch.FloatTensor(acts)

    obs_dim, act_dim = obs.shape[1], acts.shape[1]

    if chunk_size > 1:
        policy = ChunkedBCPolicy(obs_dim, act_dim, chunk_size)
        # Reshape actions into chunks
        n = (len(acts_t) // chunk_size) * chunk_size
        acts_chunked = acts_t[:n].view(-1, chunk_size, act_dim)
        obs_chunked = obs_t[:n:chunk_size]
        dataset = torch.utils.data.TensorDataset(obs_chunked, acts_chunked)
    else:
        policy = BCPolicy(obs_dim, act_dim)
        dataset = torch.utils.data.TensorDataset(obs_t, acts_t)

    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_acts in loader:
            pred = policy(batch_obs)
            loss = ((pred - batch_acts) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss = {total_loss/len(loader):.4f}")

    return policy

demos = collect_expert_demos()
policy_h1 = train_bc(demos, chunk_size=1)
policy_h4 = train_bc(demos, chunk_size=4)
```

---

## Exercise (45 min)

1. **Compounding error demo:** Train BC with few demos (10 episodes). Roll out the policy for 200 steps. Measure how error grows with time horizon. Plot error vs timestep.

2. **Chunking ablation:** Compare chunk sizes $H \in \{1, 2, 4, 8, 16\}$. Measure task success rate. What's the sweet spot?

3. **DAgger implementation:** Implement DAgger for Pendulum. Run 5 rounds of data aggregation. Compare final policy to BC-only.

4. **Multimodality failure:** Create a toy dataset where the same observation has two valid actions. Train BC with MSE loss. Show that the policy predicts the average (neither valid action).

---

## Key Takeaways

1. **BC is supervised learning** on state-action pairs — simple but fragile
2. **Compounding errors** grow quadratically with horizon length
3. **DAgger** fixes distribution shift by collecting on-policy corrections
4. **Action chunking** reduces decision points and improves temporal consistency
5. **Multimodal actions** require distribution-aware losses (not MSE)

---

## Connection to the Thread

> *BC gives us the baseline. Tomorrow, ACT (Day 79) combines action chunking with a CVAE to handle multimodal actions. Decision Transformer (Day 80) frames the problem as sequence modeling — connecting back to GPT. Diffusion Policy (Day 81) uses the diffusion framework from Days 74-76 to model the full action distribution. The question driving this week: what's the best way to represent and generate robot actions?*

---

## Further Reading

- Pomerleau (1988), "ALVINN: An Autonomous Land Vehicle in a Neural Network" — first BC
- Ross et al. (2011), "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger)
- Zhao et al. (2023), "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT)
- Mandlekar et al. (2021), "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation" (robomimic)
