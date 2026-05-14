# Exercise 10 — Imitation Learning

> Phase VI · Days 78–89
> Covers: Study Notes [13](../study-notes/13-imitation-learning.md) and [14](../study-notes/14-robot-data-eval.md)
> Time: ~10 hours hands-on

---

## Setup

### Environment

```bash
# Core dependencies
pip install torch torchvision matplotlib numpy tqdm

# LeRobot (HuggingFace robot learning library)
pip install lerobot

# For simulation environments
pip install gymnasium
pip install gymnasium-robotics  # optional, for MuJoCo tasks

# For visualization
pip install imageio[ffmpeg]
```

### Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
```

---

## Exercise 1: Behavioral Cloning

**Goal**: Implement basic behavioral cloning and observe the covariate shift problem.

### 1.1 — Generate Expert Demonstrations

```python
def expert_policy(obs: np.ndarray) -> np.ndarray:
    """Scripted expert for a 2D reaching task.
    obs: [ee_x, ee_y, target_x, target_y]
    action: [dx, dy] — end-effector displacement
    """
    ee = obs[:2]
    target = obs[2:4]
    direction = target - ee
    dist = np.linalg.norm(direction)
    if dist > 0.01:
        return 0.1 * direction / max(dist, 0.1)
    return np.zeros(2, dtype=np.float32)


class SimpleReachEnv:
    """Toy 2D reaching environment."""

    def __init__(self):
        self.ee_pos = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.max_steps = 50
        self.step_count = 0

    def reset(self):
        self.ee_pos = np.random.uniform(-1, 1, size=2).astype(np.float32)
        self.target = np.random.uniform(-1, 1, size=2).astype(np.float32)
        self.step_count = 0
        return self._obs()

    def step(self, action):
        self.ee_pos = np.clip(self.ee_pos + action, -2, 2)
        self.step_count += 1
        dist = np.linalg.norm(self.ee_pos - self.target)
        success = dist < 0.05
        done = success or self.step_count >= self.max_steps
        return self._obs(), success, done

    def _obs(self):
        return np.concatenate([self.ee_pos, self.target])


def collect_demonstrations(n_episodes: int = 200) -> dict:
    """Collect expert demonstrations."""
    env = SimpleReachEnv()
    observations, actions = [], []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = expert_policy(obs)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, success, done = env.step(action)

    return {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
    }

# Collect data
demos = collect_demonstrations(200)
print(f"Collected {len(demos['observations'])} observation-action pairs")
```

### 1.2 — Train BC Policy

```python
class BCPolicy(nn.Module):
    """Simple MLP policy for behavioral cloning."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def train_bc(demos: dict, epochs: int = 100, batch_size: int = 64, lr: float = 1e-3):
    """Train behavioral cloning policy."""
    obs = torch.FloatTensor(demos["observations"])
    act = torch.FloatTensor(demos["actions"])

    # TODO: Normalize observations and actions
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-8
    act_mean, act_std = act.mean(0), act.std(0) + 1e-8
    obs_norm = (obs - obs_mean) / obs_std
    act_norm = (act - act_mean) / act_std

    policy = BCPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    losses = []

    for epoch in trange(epochs, desc="BC Training"):
        perm = torch.randperm(len(obs_norm))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(obs_norm) - batch_size, batch_size):
            batch_obs = obs_norm[perm[i:i+batch_size]]
            batch_act = act_norm[perm[i:i+batch_size]]

            pred_act = policy(batch_obs)
            loss = F.mse_loss(pred_act, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

    return policy, {"obs_mean": obs_mean, "obs_std": obs_std,
                     "act_mean": act_mean, "act_std": act_std}, losses
```

### 1.3 — Evaluate: Success Rate and Distribution Shift

```python
def evaluate_policy(policy, stats, n_episodes: int = 100) -> dict:
    """Evaluate BC policy, measuring success rate and trajectory quality."""
    env = SimpleReachEnv()
    successes = 0
    trajectories = []

    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            traj = [obs[:2].copy()]
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                obs_norm = (obs_t - stats["obs_mean"]) / stats["obs_std"]
                act_norm = policy(obs_norm)
                action = (act_norm * stats["act_std"] + stats["act_mean"]).squeeze(0).numpy()
                obs, success, done = env.step(action)
                traj.append(obs[:2].copy())
            successes += int(success)
            trajectories.append(np.array(traj))

    return {
        "success_rate": successes / n_episodes,
        "trajectories": trajectories,
    }

# TODO: Train and evaluate
policy, stats, losses = train_bc(demos)
results = evaluate_policy(policy, stats)
print(f"BC Success Rate: {results['success_rate']:.1%}")

# TODO: Visualize a few trajectories
# Compare expert trajectories (straight lines to target) vs BC (may drift)
```

**Questions**:
- What success rate does BC achieve?
- Do you observe distribution shift (trajectory drifting from expert distribution)?
- What happens with fewer demonstrations (try 20 vs 200)?

---

## Exercise 2: Action Chunking

**Goal**: Modify BC to predict multi-step action chunks and measure improvement.

### 2.1 — Create Chunked Dataset

```python
def create_chunked_dataset(demos: dict, chunk_size: int = 4) -> dict:
    """Convert single-step demos to chunked format.
    Each observation maps to chunk_size future actions.

    IMPORTANT: Only chunks within episode boundaries — never crosses episodes.
    """
    obs_list, chunk_list = [], []

    observations = demos["observations"]
    actions = demos["actions"]

    # Determine episode boundaries from the data.
    # If demos has an "episode_ends" key (list of end indices), use it.
    # Otherwise, assume a single episode (the flat array is one trajectory).
    if "episode_ends" in demos:
        episode_ends = demos["episode_ends"]
    else:
        # Fallback: treat entire dataset as one episode
        episode_ends = [len(observations)]

    start = 0
    for end in episode_ends:
        ep_obs = observations[start:end]
        ep_act = actions[start:end]
        # Only create chunks that stay within this episode
        for i in range(len(ep_obs) - chunk_size):
            obs_list.append(ep_obs[i])
            chunk_list.append(ep_act[i:i+chunk_size].flatten())  # H actions concatenated
        start = end

    return {
        "observations": np.array(obs_list, dtype=np.float32),
        "action_chunks": np.array(chunk_list, dtype=np.float32),
    }
```

### 2.2 — Chunked BC Policy

```python
class ChunkedBCPolicy(nn.Module):
    """BC policy that predicts H future actions."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2,
                 chunk_size: int = 4, hidden: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim * chunk_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        flat = self.net(obs)
        return flat.view(-1, self.chunk_size, self.act_dim)
```

### 2.3 — Compare Chunk Sizes

```python
# TODO: Train and evaluate with different chunk sizes
chunk_sizes = [1, 4, 8, 16]
results_by_chunk = {}

for H in chunk_sizes:
    # Create chunked dataset
    chunked_demos = create_chunked_dataset(demos, chunk_size=H)

    # TODO: Train chunked BC policy
    # TODO: Evaluate with receding horizon execution:
    #   - Predict H actions
    #   - Execute first action only (or first H_e actions)
    #   - Re-observe and re-predict

    # Store results
    # results_by_chunk[H] = success_rate
    pass

# TODO: Plot: chunk_size vs success_rate
# Expected: moderate chunk sizes (4-8) outperform H=1 and very large H
```

**Questions**:
- Which chunk size works best?
- Why does very large chunk size (H=16) not necessarily improve further?
- How does execution strategy matter (execute 1 vs execute H/2)?

---

## Exercise 3: ACT Implementation (Simplified)

**Goal**: Implement a simplified ACT model with CVAE for multimodal action prediction.

### 3.1 — CVAE Architecture

```python
class SimplifiedACT(nn.Module):
    """Simplified ACT: CVAE with MLP encoder/decoder."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2,
                 chunk_size: int = 8, z_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        flat_act = act_dim * chunk_size

        # Encoder: (obs, actions) → z distribution
        # Only used during training
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + flat_act, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # Decoder: (obs, z) → action chunk
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, flat_act),
        )

    def encode(self, obs, actions_flat):
        """Encode (obs, actions) → (mu, logvar)."""
        h = self.encoder(torch.cat([obs, actions_flat], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, obs, z):
        """Decode (obs, z) → action chunk."""
        flat = self.decoder(torch.cat([obs, z], dim=-1))
        return flat.view(-1, self.chunk_size, self.act_dim)

    def forward(self, obs, actions_flat):
        """Training forward pass."""
        mu, logvar = self.encode(obs, actions_flat)
        z = self.reparameterize(mu, logvar)
        pred_actions = self.decode(obs, z)
        return pred_actions, mu, logvar

    def predict(self, obs):
        """Inference: sample z from prior, decode."""
        batch_size = obs.shape[0]
        z = torch.randn(batch_size, self.z_dim, device=obs.device)
        return self.decode(obs, z)
```

### 3.2 — ELBO Training Loss

```python
def act_loss(pred_actions, true_actions, mu, logvar, beta: float = 0.01):
    """ACT training loss = reconstruction + β * KL divergence."""
    # Reconstruction loss
    recon = F.mse_loss(pred_actions, true_actions)

    # KL divergence: D_KL(N(mu, sigma) || N(0, I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + beta * kl, {"recon": recon.item(), "kl": kl.item()}
```

### 3.3 — Train ACT on PushT (LeRobot)

```python
# Option A: Use LeRobot dataset
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# dataset = LeRobotDataset("lerobot/pusht")

# Option B: Use the toy environment from Exercise 1
# This is simpler for learning purposes

def train_act(demos: dict, chunk_size: int = 8, epochs: int = 200,
              beta: float = 0.01, lr: float = 1e-3):
    """Train simplified ACT model."""
    chunked = create_chunked_dataset(demos, chunk_size=chunk_size)
    obs_t = torch.FloatTensor(chunked["observations"])
    act_t = torch.FloatTensor(chunked["action_chunks"])

    model = SimplifiedACT(chunk_size=chunk_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs, desc="ACT Training"):
        perm = torch.randperm(len(obs_t))
        for i in range(0, len(obs_t) - 64, 64):
            batch_obs = obs_t[perm[i:i+64]]
            batch_act = act_t[perm[i:i+64]]
            batch_act_flat = batch_act  # already flat from create_chunked_dataset

            pred, mu, logvar = model(batch_obs, batch_act_flat)
            true_actions = batch_act.view(-1, chunk_size, 2)
            loss, metrics = act_loss(pred, true_actions, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

### 3.4 — Visualize Multimodality

```python
# TODO: For a single observation, sample z multiple times
# and visualize the different action trajectories generated

def visualize_multimodality(model, obs: np.ndarray, n_samples: int = 20):
    """Sample multiple trajectories from ACT for the same observation."""
    model.eval()
    obs_t = torch.FloatTensor(obs).unsqueeze(0).repeat(n_samples, 1)

    with torch.no_grad():
        action_chunks = model.predict(obs_t)  # (n_samples, chunk_size, 2)

    plt.figure(figsize=(6, 6))
    ee = obs[:2]
    target = obs[2:4]

    for i in range(n_samples):
        traj = action_chunks[i].numpy().cumsum(axis=0) + ee  # cumulative for positions
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1)

    plt.scatter(*ee, c="blue", s=100, marker="o", label="Start", zorder=5)
    plt.scatter(*target, c="red", s=100, marker="*", label="Target", zorder=5)
    plt.legend()
    plt.title("ACT: Multiple Sampled Trajectories (Different z)")
    plt.axis("equal")
    plt.savefig("act_multimodality.png", dpi=150)
    plt.show()

# TODO: Run visualization
# obs_test = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)
# visualize_multimodality(model, obs_test)
```

---

## Exercise 4: Diffusion Policy Training

**Goal**: Train a Diffusion Policy and compare with BC and ACT.

### 4.1 — Conditional Diffusion Policy

```python
class SimpleDiffusionPolicy(nn.Module):
    """Simplified Diffusion Policy for 2D actions."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2,
                 chunk_size: int = 8, hidden: int = 256, T: int = 50):
        super().__init__()
        self.T = T
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        flat_act = act_dim * chunk_size

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

        # Noise predictor conditioned on obs and timestep
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.net = nn.Sequential(
            nn.Linear(flat_act + 64 + 128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_act),
        )

    def predict_noise(self, noisy_actions_flat, t, obs):
        t_emb = self.time_embed(t.float().unsqueeze(-1) / self.T)
        o_emb = self.obs_embed(obs)
        inp = torch.cat([noisy_actions_flat, t_emb, o_emb], dim=-1)
        return self.net(inp)

    def add_noise(self, actions_flat, t):
        noise = torch.randn_like(actions_flat)
        ab = self.alpha_bars[t].unsqueeze(-1)
        noisy = torch.sqrt(ab) * actions_flat + torch.sqrt(1 - ab) * noise
        return noisy, noise

    @torch.no_grad()
    def sample(self, obs, n_samples: int = 1):
        """DDPM sampling: denoise from random noise to action chunk."""
        obs_batch = obs.unsqueeze(0).repeat(n_samples, 1) if obs.dim() == 1 else obs
        flat_dim = self.chunk_size * self.act_dim
        x = torch.randn(n_samples, flat_dim, device=obs.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, device=obs.device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t_batch, obs_batch)

            alpha = 1.0 - (self.alpha_bars[t] if t == 0 else
                           self.alpha_bars[t] - self.alpha_bars[t-1]) / (
                           1.0 if t == 0 else 1.0 - self.alpha_bars[t-1])
            alpha_bar = self.alpha_bars[t]

            # Simplified DDPM step
            coef1 = 1.0 / torch.sqrt(torch.tensor(alpha))
            coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
            mean = coef1 * (x - coef2 * pred_noise)

            if t > 0:
                beta = 1.0 - alpha
                x = mean + torch.sqrt(torch.tensor(beta)) * torch.randn_like(x)
            else:
                x = mean

        return x.view(n_samples, self.chunk_size, self.act_dim)
```

### 4.2 — Training Loop

```python
def train_diffusion_policy(demos: dict, chunk_size: int = 8,
                            epochs: int = 300, lr: float = 1e-3):
    """Train Diffusion Policy."""
    chunked = create_chunked_dataset(demos, chunk_size=chunk_size)
    obs_t = torch.FloatTensor(chunked["observations"])
    act_t = torch.FloatTensor(chunked["action_chunks"])  # flat

    model = SimpleDiffusionPolicy(chunk_size=chunk_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs, desc="Diffusion Policy"):
        perm = torch.randperm(len(obs_t))
        for i in range(0, len(obs_t) - 64, 64):
            batch_obs = obs_t[perm[i:i+64]]
            batch_act = act_t[perm[i:i+64]]

            # Sample random timestep
            t = torch.randint(0, model.T, (len(batch_obs),))

            # Add noise to actions
            noisy_act, noise = model.add_noise(batch_act, t)

            # Predict noise
            pred_noise = model.predict_noise(noisy_act, t, batch_obs)

            # Loss
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

### 4.3 — Comparison: BC vs ACT vs Diffusion Policy

```python
# TODO: Train all three on the same data
# 1. BC (single step)
# 2. ACT (CVAE + chunking)
# 3. Diffusion Policy (denoising + chunking)

# TODO: Evaluate each on 100 episodes
# Metrics:
# - Success rate
# - Average distance to target at episode end
# - Trajectory smoothness (sum of squared jerks)

# TODO: Create comparison table and visualization
# Expected ranking: Diffusion Policy ≥ ACT > BC (single step)
# But all should work reasonably on this simple task
```

---

## Exercise 5: Temporal Ensembling for Action Chunking

**Goal**: Implement temporal ensembling — the crucial technique that makes action
chunking smooth at chunk boundaries. Used by both ACT and Diffusion Policy.

### 5.1 — The Chunk Boundary Problem

```python
# Without temporal ensembling, naively using action chunks creates jerky motion:
#
# Time:    0  1  2  3  4  5  6  7  8  9  10 11 ...
# Chunk 1: [a0 a1 a2 a3]
# Chunk 2:             [a4 a5 a6 a7]     ← discontinuity at boundary!
# Chunk 3:                         [a8 a9 a10 a11]
#
# With temporal ensembling (query every step, overlap predictions):
# Chunk at t=0: [a0|0  a1|0  a2|0  a3|0]
# Chunk at t=1:  [a1|1  a2|1  a3|1  a4|1]
# Chunk at t=2:   [a2|2  a3|2  a4|2  a5|2]
# Chunk at t=3:    [a3|3  a4|3  a5|3  a6|3]
#
# At time t=2, we have predictions: a2|0, a2|1, a2|2 → blend them!
```

### 5.2 — Exponential Weighting Implementation

```python
class TemporalEnsemble:
    """Temporal ensembling for overlapping action chunks.
    
    Key idea: newer predictions are more trustworthy because they
    see more recent observations. Weight by recency.
    """
    
    def __init__(self, chunk_size: int, action_dim: int, decay: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay  # λ in exp(-λ * age)
        # Buffer: stores (prediction_time, action_chunk)
        self.buffer: list[tuple[int, np.ndarray]] = []
        self.current_time = 0
    
    def add_chunk(self, action_chunk: np.ndarray):
        """Add a new predicted chunk (shape: [chunk_size, action_dim])."""
        assert action_chunk.shape == (self.chunk_size, self.action_dim)
        self.buffer.append((self.current_time, action_chunk.copy()))
        # Remove chunks that no longer overlap with current time
        self.buffer = [
            (t, chunk) for (t, chunk) in self.buffer
            if t + self.chunk_size > self.current_time
        ]
    
    def get_action(self) -> np.ndarray:
        """Get the ensembled action for the current timestep.
        
        TODO: Implement temporal ensembling:
        1. For each chunk in buffer, find its prediction for current time
        2. Compute weight = exp(-decay * age) where age = current_time - chunk_time
        3. Return weighted average of all predictions
        """
        predictions = []
        weights = []
        
        for chunk_time, chunk in self.buffer:
            # Index within this chunk for current time
            idx = self.current_time - chunk_time
            if 0 <= idx < self.chunk_size:
                predictions.append(chunk[idx])
                age = self.current_time - chunk_time
                weights.append(np.exp(-self.decay * age))
        
        if not predictions:
            raise ValueError("No predictions available for current time")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        predictions = np.array(predictions)
        ensembled = np.sum(predictions * weights[:, None], axis=0)
        return ensembled
    
    def step(self):
        """Advance time by one step."""
        self.current_time += 1


def evaluate_smoothness(actions: np.ndarray) -> dict:
    """Compute smoothness metrics for an action trajectory.
    
    Args:
        actions: (T, action_dim) array of executed actions
    
    Returns dict with:
        - velocity: mean absolute first derivative
        - acceleration: mean absolute second derivative  
        - jerk: mean absolute third derivative (lower = smoother)
    """
    vel = np.diff(actions, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    
    return {
        "velocity": np.abs(vel).mean(),
        "acceleration": np.abs(acc).mean(),
        "jerk": np.abs(jerk).mean(),
    }
```

### 5.3 — Comparison Experiment

```python
# TODO: Compare three execution strategies on your trained policy:
# 
# Strategy A: Execute full chunk, then generate new (no overlap)
# Strategy B: Query every step, use only first action (no ensembling)
# Strategy C: Query every step, temporal ensembling (lambda=0.01)
# Strategy D: Query every step, temporal ensembling (lambda=0.1)
#
# For each, run 50 episodes and measure:
# 1. Success rate
# 2. Smoothness (jerk metric)
# 3. Total path length
# 4. Execution time per step (with inference cost)
#
# Plot:
# - All 4 trajectories overlaid for same initial condition
# - Smoothness comparison bar chart
# - Success rate comparison

# Expected results:
# - Strategy A: jerky at boundaries, fastest inference (query every H steps)
# - Strategy B: smooth but wasteful (re-predicts whole chunk to use 1 action)
# - Strategy C: smoothest, best success, most expensive (query every step)
# - Strategy D: slightly less smooth than C, but still better than A/B

# Key insight: lambda controls the tradeoff between:
# - responsiveness (high lambda → trust newest chunk more)
# - smoothness (low lambda → blend many overlapping predictions)
```

**Exercises**:
1. Implement `TemporalEnsemble` and run on your ACT model from Exercise 3
2. Ablate λ: try 0.001, 0.01, 0.1, 1.0, 10.0 — plot jerk vs λ
3. What happens at λ→0 (uniform weighting)? At λ→∞ (use latest only)?
4. **Real-time constraint**: If inference takes 50ms and control runs at 20Hz:
   - Can you query every step? (50ms per step, 20 Hz = 50ms budget ✓ barely)
   - What if inference takes 100ms? (Must use chunk_size > 1 without ensembling)
5. **Connection to Diffusion Policy**: ACT paper uses λ=0.01. What does the Diffusion Policy paper use? (Answer: they recede-horizon replan every k steps instead of exponential ensembling.)

---

## Self-Check

After completing these exercises, you should be able to answer:

- [ ] What is the covariate shift problem in BC and how does action chunking help?
- [ ] How does the CVAE in ACT enable multimodal action generation?
- [ ] What is the training procedure for Diffusion Policy (what is the loss)?
- [ ] How does inference work for Diffusion Policy (iterative denoising)?
- [ ] What is temporal ensembling and why does it improve execution smoothness?
- [ ] How does λ in temporal ensembling trade off smoothness vs. responsiveness?
- [ ] Which method works best on your task, and why?

## Stretch Goals

1. **LeRobot integration**: Load `lerobot/pusht` dataset and train all three methods on it
   instead of toy data — compare with LeRobot's pretrained checkpoints.

2. **DDIM sampling**: Implement DDIM for the Diffusion Policy — compare quality at
   5 vs 10 vs 50 steps.

3. **Language conditioning**: Add a language embedding input to the Diffusion Policy.
   Use simple one-hot task IDs as a proxy for language embeddings. Train a multi-task
   policy that can "push left" or "push right" based on the language input.

4. **Multimodal data**: Create a dataset where the expert sometimes goes left and
   sometimes right around an obstacle. Verify that:
   - BC averages the two modes (goes through the obstacle)
   - ACT and Diffusion Policy can sample both modes
