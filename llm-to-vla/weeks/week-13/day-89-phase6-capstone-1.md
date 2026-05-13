# Day 89: Phase VI Capstone — Day 1: Pipeline Design & Data

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 13 | 3 hours**
> *"Build the full pipeline: data → model → evaluation → debug → iterate."*

## Navigation
- **Previous:** [Day 88: Policy Debugging](day-88-policy-debugging.md)
- **Next:** [Day 90: Phase VI Capstone Day 2](day-90-phase6-capstone-2.md)
- **Week:** [Week 13 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/14-robot-data-eval.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Capstone Overview

The Phase VI capstone spans three days:
- **Day 89:** Pipeline design, data collection, baseline BC policy
- **Day 90:** Advanced policy (diffusion or ACT), evaluation, debugging
- **Day 91:** Final integration, reflection, checkpoint quiz

## The Challenge

Build a complete robot learning pipeline for a manipulation task in simulation:

```
Task options (choose one):
  A) Push-T: push a T-shaped block to a target pose
  B) Pick-and-place: pick up a cube and place it in a bin
  C) Reach: move end-effector to a target position
```

---

## Part 1: Pipeline Design (60 min)

### 1.1 System Architecture

Design your pipeline before writing code:

```
┌─────────────────────────────────────────────────────────┐
│                    ROBOT LEARNING PIPELINE               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Data Collection                                      │
│     ├── Environment setup (gymnasium)                    │
│     ├── Expert policy (scripted or PD controller)        │
│     ├── Demonstration recording                          │
│     └── Data validation                                  │
│                                                          │
│  2. Data Processing                                      │
│     ├── Normalization (actions to [-1, 1])               │
│     ├── Chunking (group into sequences of length H)      │
│     ├── Augmentation (observation noise)                 │
│     └── Train/val split (80/20)                          │
│                                                          │
│  3. Model Training                                       │
│     ├── Architecture: BC → Diffusion/ACT                 │
│     ├── Loss: MSE → Diffusion loss / CVAE loss           │
│     ├── Optimizer: Adam, lr=1e-4                         │
│     └── Checkpointing every 50 epochs                    │
│                                                          │
│  4. Evaluation                                           │
│     ├── Offline: val loss, action distribution           │
│     ├── Online: success rate (50 trials, Wilson CI)       │
│     └── Ablation: chunk size, data amount, augmentation  │
│                                                          │
│  5. Debugging                                            │
│     ├── Failure classification                           │
│     ├── Action trajectory visualization                  │
│     └── Hypothesis → test → fix cycle                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Design Decisions

Document your choices:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task | ? | |
| Action space | absolute / delta | |
| Action representation | joint / EE | |
| Observation | state / image / both | |
| Chunk size | ? | |
| Training epochs | ? | |
| Number of demos | ? | |

---

## Part 2: Data Collection (60 min)

```python
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

# --- Environment Setup ---
# Using Pendulum as a proxy manipulation task
env = gym.make("Pendulum-v1")

# --- Expert Policy ---
def expert_policy(obs):
    """PD controller to swing up and balance the pendulum."""
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)
    # PD control
    torque = -8.0 * theta - 2.0 * theta_dot
    return np.clip([torque], -2.0, 2.0)

# --- Collect Demonstrations ---
def collect_demonstrations(env, expert_fn, n_episodes=100, max_steps=200):
    dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "episode_starts": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_obs, ep_acts, ep_rews = [], [], []

        for step in range(max_steps):
            action = expert_fn(obs)
            ep_obs.append(obs)
            ep_acts.append(action)

            obs, reward, term, trunc, _ = env.step(action)
            ep_rews.append(reward)
            if term or trunc:
                break

        dataset["observations"].append(np.array(ep_obs))
        dataset["actions"].append(np.array(ep_acts))
        dataset["rewards"].append(np.array(ep_rews))
        dataset["episode_starts"].append(len(dataset["observations"]) - 1)

    return dataset

# --- Data Processing ---
def process_dataset(dataset, chunk_size=8):
    """Normalize, chunk, and prepare for training."""
    all_obs = np.concatenate(dataset["observations"])
    all_acts = np.concatenate(dataset["actions"])

    # Compute normalization statistics
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0) + 1e-6
    act_mean, act_std = all_acts.mean(0), all_acts.std(0) + 1e-6

    # Normalize
    all_obs_norm = (all_obs - obs_mean) / obs_std
    all_acts_norm = (all_acts - act_mean) / act_std

    # Create chunks
    n_chunks = len(all_obs_norm) - chunk_size + 1
    obs_chunks = np.array([all_obs_norm[i] for i in range(n_chunks)])
    act_chunks = np.array([all_acts_norm[i:i+chunk_size] for i in range(n_chunks)])

    # Train/val split
    n_train = int(0.8 * n_chunks)
    indices = np.random.permutation(n_chunks)

    train_data = {
        "obs": torch.FloatTensor(obs_chunks[indices[:n_train]]),
        "acts": torch.FloatTensor(act_chunks[indices[:n_train]]),
    }
    val_data = {
        "obs": torch.FloatTensor(obs_chunks[indices[n_train:]]),
        "acts": torch.FloatTensor(act_chunks[indices[n_train:]]),
    }

    stats = {"obs_mean": obs_mean, "obs_std": obs_std,
             "act_mean": act_mean, "act_std": act_std}

    return train_data, val_data, stats

print("Collecting demonstrations...")
dataset = collect_demonstrations(env, expert_policy, n_episodes=100)
print(f"Collected {len(dataset['observations'])} episodes")
print(f"Total transitions: {sum(len(ep) for ep in dataset['observations'])}")

train_data, val_data, stats = process_dataset(dataset, chunk_size=8)
print(f"Train chunks: {len(train_data['obs'])}")
print(f"Val chunks: {len(val_data['obs'])}")
```

---

## Part 3: Baseline BC Policy (60 min)

```python
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class BaselineBC(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim * chunk_size),
        )
        self.chunk_size = chunk_size
        self.act_dim = act_dim

    def forward(self, obs):
        return self.net(obs).view(-1, self.chunk_size, self.act_dim)

    @torch.no_grad()
    def predict(self, obs_raw, stats):
        obs_norm = (obs_raw - stats["obs_mean"]) / stats["obs_std"]
        obs_t = torch.FloatTensor(obs_norm).unsqueeze(0)
        act_norm = self.forward(obs_t)[0, 0].numpy()  # First action in chunk
        return act_norm * stats["act_std"] + stats["act_mean"]

# Train
obs_dim = train_data["obs"].shape[1]
act_dim = train_data["acts"].shape[2]
chunk_size = train_data["acts"].shape[1]

model = BaselineBC(obs_dim, act_dim, chunk_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loader = DataLoader(
    TensorDataset(train_data["obs"], train_data["acts"]),
    batch_size=256, shuffle=True,
)

for epoch in range(200):
    total_loss = 0
    for obs_batch, act_batch in loader:
        pred = model(obs_batch)
        loss = ((pred - act_batch) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        # Validation
        with torch.no_grad():
            val_pred = model(val_data["obs"])
            val_loss = ((val_pred - val_data["acts"]) ** 2).mean()
        print(f"Epoch {epoch}: train={total_loss/len(loader):.4f}, val={val_loss:.4f}")

print("\nBaseline BC trained. Continue to Day 90 for advanced policy.")
```

---

## Deliverables for Today

- [ ] Pipeline design document (architecture diagram + decision table)
- [ ] Collected dataset with validation report
- [ ] Trained baseline BC policy
- [ ] Baseline evaluation results (success rate + CI)

---

## Connection to the Thread

> *Day 90: upgrade from BC to Diffusion Policy or ACT. Run systematic evaluation. Debug failures. Day 91: final reflection and Phase VI checkpoint quiz.*
