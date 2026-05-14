# Exercise 09 — RL & Diffusion Basics

> Phase VI · Days 71–77
> Covers: Study Notes [11](../study-notes/11-rl-foundations.md) and [12](../study-notes/12-diffusion-flow.md)
> Time: ~8 hours hands-on

---

## Setup

### Environment

```bash
# Create/activate environment
conda activate vla-curriculum  # or your preferred env

# Install dependencies
pip install gymnasium stable-baselines3 torch matplotlib numpy tqdm
pip install gymnasium[mujoco]  # for FetchReach (optional, needs MuJoCo)
```

### Imports

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from collections import deque
from tqdm import trange
```

---

## Exercise 1: REINFORCE on CartPole

**Goal**: Implement the REINFORCE policy gradient algorithm from scratch and understand
variance reduction with baselines.

### 1.1 — Vanilla REINFORCE

```python
class PolicyNetwork(nn.Module):
    """Simple policy network for discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        """Sample action and return log probability."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.forward(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def reinforce_train(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    seed: int = 42,
) -> list[float]:
    """Train with vanilla REINFORCE. Return episode rewards."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []

    for ep in trange(episodes, desc="REINFORCE"):
        # TODO: Collect a full episode
        # Store: log_probs, rewards at each step
        log_probs = []
        rewards = []

        obs, _ = env.reset()
        done = False
        while not done:
            action, log_prob = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        # TODO: Compute returns G_t = Σ γ^k r_{t+k}
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # TODO: Normalize returns (helps stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # TODO: Compute policy gradient loss
        # Loss = -Σ log π(a|s) · G_t
        loss = 0.0
        for log_prob, G_t in zip(log_probs, returns):
            loss -= log_prob * G_t

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(sum(rewards))

    env.close()
    return all_rewards
```

**Task**: Run `reinforce_train()` and plot the learning curve.

### 1.2 — REINFORCE with Baseline

```python
class ValueNetwork(nn.Module):
    """Baseline value function V(s)."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def reinforce_baseline_train(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    gamma: float = 0.99,
    lr_policy: float = 1e-3,
    lr_value: float = 1e-3,
) -> list[float]:
    """Train REINFORCE with learned baseline. Return episode rewards."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim)
    value_fn = ValueNetwork(obs_dim)
    policy_opt = optim.Adam(policy.parameters(), lr=lr_policy)
    value_opt = optim.Adam(value_fn.parameters(), lr=lr_value)

    all_rewards = []

    for ep in trange(episodes, desc="REINFORCE+Baseline"):
        log_probs, rewards, states = [], [], []

        obs, _ = env.reset()
        done = False
        while not done:
            states.append(obs.copy())
            action, log_prob = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        states_t = torch.FloatTensor(np.array(states))

        # TODO: Compute advantages = returns - V(s)
        values = value_fn(states_t).detach()
        advantages = returns - values

        # TODO: Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # TODO: Update policy with advantages (not raw returns)
        policy_loss = 0.0
        for log_prob, adv in zip(log_probs, advantages):
            policy_loss -= log_prob * adv

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # TODO: Update value function
        value_pred = value_fn(states_t)
        value_loss = F.mse_loss(value_pred, returns)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        all_rewards.append(sum(rewards))

    env.close()
    return all_rewards
```

**Task**: Run both versions and compare:

```python
rewards_vanilla = reinforce_train(episodes=1000)
rewards_baseline = reinforce_baseline_train(episodes=1000)

# Plot comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
window = 50
for label, data in [("Vanilla", rewards_vanilla), ("+ Baseline", rewards_baseline)]:
    smoothed = np.convolve(data, np.ones(window)/window, mode="valid")
    ax.plot(smoothed, label=label)
ax.set_xlabel("Episode")
ax.set_ylabel("Reward (smoothed)")
ax.set_title("REINFORCE: Variance Reduction with Baseline")
ax.legend()
plt.tight_layout()
plt.savefig("reinforce_comparison.png", dpi=150)
plt.show()
```

**Questions to answer**:
- How much faster does the baseline version converge?
- What happens if you don't normalize advantages?
- What is the final average reward for each variant?

---

## Exercise 2: PPO on Manipulation Task

**Goal**: Use stable-baselines3 to train PPO and understand sample efficiency.

### 2.1 — PPO on CartPole (Warm-Up)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Vectorized environments for faster collection
env = make_vec_env("CartPole-v1", n_envs=4)
eval_env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,         # steps per rollout per env
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,       # PPO clip parameter
    verbose=1,
)

# Train
eval_callback = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=20)
model.learn(total_timesteps=100_000, callback=eval_callback)
```

### 2.2 — PPO on FetchReach (Continuous Control)

```python
# FetchReach: move end-effector to target position
# Requires: pip install gymnasium-robotics

import gymnasium as gym
# NOTE: FetchReach may require gymnasium-robotics
# Alternative: use "Pendulum-v1" for continuous control practice

env = gym.make("Pendulum-v1")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
)

model.learn(total_timesteps=200_000)
```

### 2.3 — Compare Sample Efficiency

```python
# Track: how many environment steps to reach threshold performance?

# TODO: Train both REINFORCE and PPO, tracking steps-to-threshold
# For CartPole: threshold = 475 average reward over 100 episodes

# Questions:
# - How many steps does REINFORCE need vs PPO?
# - How does parallelization (n_envs) affect PPO wall-clock time?
# - What happens if you change clip_range to 0.1 or 0.3?
```

---

## Exercise 3: DDPM from Scratch (2D)

**Goal**: Implement a simple DDPM that learns to generate 2D points from a target
distribution. This builds intuition before applying diffusion to actions.

### 3.1 — Setup: Target Distribution

```python
def sample_swiss_roll(n: int = 1000) -> np.ndarray:
    """Generate Swiss Roll 2D distribution."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data.astype(np.float32)

# Visualize
data = sample_swiss_roll(2000)
plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
plt.title("Target Distribution (Swiss Roll)")
plt.axis("equal")
plt.show()
```

### 3.2 — Implement Noise Schedule

```python
class NoiseSchedule:
    """Linear noise schedule for DDPM."""

    def __init__(self, T: int = 200, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x0 at timestep t. Return (noisy_x, noise)."""
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_omab * noise
        return x_t, noise
```

### 3.3 — Implement Noise Prediction Network

```python
class NoisePredictor(nn.Module):
    """Simple MLP that predicts noise given (x_t, t)."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, time_emb_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 200.0)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)
```

### 3.4 — Training Loop

```python
def train_ddpm(
    data: np.ndarray,
    epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    T: int = 200,
) -> tuple[NoisePredictor, NoiseSchedule]:
    """Train DDPM on 2D data."""
    schedule = NoiseSchedule(T=T)
    model = NoisePredictor(data_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.FloatTensor(data)
    losses = []

    for epoch in trange(epochs, desc="Training DDPM"):
        # Shuffle
        perm = torch.randperm(len(dataset))
        for i in range(0, len(dataset) - batch_size, batch_size):
            batch = dataset[perm[i:i+batch_size]]
            t = torch.randint(0, T, (len(batch),))

            # TODO: Add noise
            x_t, noise = schedule.add_noise(batch, t)

            # TODO: Predict noise
            pred_noise = model(x_t, t)

            # TODO: Compute loss
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return model, schedule
```

### 3.5 — Sampling (Denoising)

```python
@torch.no_grad()
def sample_ddpm(
    model: NoisePredictor,
    schedule: NoiseSchedule,
    n_samples: int = 1000,
) -> list[np.ndarray]:
    """Sample from DDPM, returning intermediate steps for visualization."""
    x = torch.randn(n_samples, 2)  # Start from noise
    trajectory = [x.numpy().copy()]

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long)
        pred_noise = model(x, t_batch)

        # DDPM reverse step
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * noise
        else:
            x = mean

        if t % 20 == 0:
            trajectory.append(x.numpy().copy())

    return trajectory

# TODO: Train, sample, and visualize the denoising process
model, schedule = train_ddpm(data, epochs=500)
trajectory = sample_ddpm(model, schedule)

# Visualize denoising steps
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for idx, ax in enumerate(axes.flat):
    step = min(idx, len(trajectory) - 1)
    ax.scatter(trajectory[step][:, 0], trajectory[step][:, 1], s=1, alpha=0.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(f"Step {step * 20}")
    ax.set_aspect("equal")
plt.suptitle("DDPM Denoising: Noise → Swiss Roll")
plt.tight_layout()
plt.savefig("ddpm_denoising.png", dpi=150)
plt.show()
```

---

## Exercise 4: Diffusion for Action Prediction

**Goal**: Apply diffusion to predict robot actions conditioned on observations.

### 4.1 — Create Toy Pushing Task Data

```python
def generate_push_data(n_demos: int = 500, steps: int = 20) -> dict:
    """Generate toy pushing task data.
    Observation: (object_x, object_y, target_x, target_y)
    Action: (dx, dy) — end-effector displacement
    """
    observations = []
    actions = []

    for _ in range(n_demos):
        obj_pos = np.random.uniform(-0.5, 0.5, size=2)
        target = np.random.uniform(-0.5, 0.5, size=2)

        obs_seq = []
        act_seq = []

        pos = obj_pos.copy()
        for t in range(steps):
            obs = np.concatenate([pos, target])
            direction = target - pos
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                action = 0.05 * direction / max(dist, 0.05) + np.random.randn(2) * 0.005
            else:
                action = np.random.randn(2) * 0.001
            pos = pos + action
            obs_seq.append(obs)
            act_seq.append(action)

        observations.append(np.array(obs_seq, dtype=np.float32))
        actions.append(np.array(act_seq, dtype=np.float32))

    return {"observations": observations, "actions": actions}
```

### 4.2 — Conditional Noise Predictor

```python
class ConditionalNoisePredictor(nn.Module):
    """Predict noise for action, conditioned on observation."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.net = nn.Sequential(
            nn.Linear(act_dim + 64 + 128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, noisy_action, t, obs):
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 100.0)
        o_emb = self.obs_embed(obs)
        inp = torch.cat([noisy_action, t_emb, o_emb], dim=-1)
        return self.net(inp)
```

### 4.3 — Train and Evaluate

```python
# TODO: Train the conditional diffusion model
# 1. Sample (obs, action) pairs from the push data
# 2. Add noise to actions at random timestep
# 3. Predict noise conditioned on observation
# 4. MSE loss between predicted and true noise

# TODO: At inference:
# 1. Given a new observation (obj_pos, target_pos)
# 2. Start from random action noise
# 3. Iteratively denoise, conditioning on the observation
# 4. The denoised action should push toward the target

# TODO: Visualize:
# - Given the same observation, sample 10 different actions
# - Do they all point toward the target? (they should!)
# - Compare with a simple regression baseline
```

---

## Exercise 5: PPO from Scratch

**Goal**: Implement Proximal Policy Optimization — the algorithm behind RLHF and many
robot learning papers. Understand the clipped surrogate objective.

### 5.1 — Actor-Critic Network

```python
class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for continuous actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Actor: outputs mean of Gaussian policy
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        # Critic: outputs state value
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        value = self.critic(features)
        return action_mean, action_std, value.squeeze(-1)

    def get_action(self, obs: np.ndarray):
        """Sample action, return (action, log_prob, value)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mean, std, value = self.forward(obs_t)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return (
            action.squeeze(0).detach().numpy(),
            log_prob.item(),
            value.item(),
        )
```

### 5.2 — GAE (Generalized Advantage Estimation)

```python
def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns.
    
    Returns:
        advantages: array of shape (T,)
        returns: array of shape (T,) — advantages + values
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    
    # TODO: Implement GAE
    # gae = 0
    # for t in reversed(range(T)):
    #     if dones[t]:
    #         next_val = 0.0
    #     else:
    #         next_val = values[t+1] if t+1 < T else next_value
    #     delta = rewards[t] + gamma * next_val - values[t]
    #     gae = delta + gamma * lam * (0 if dones[t] else gae)
    #     advantages[t] = gae
    
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns
```

### 5.3 — PPO Update with Clipping

```python
def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    obs_batch: torch.Tensor,
    act_batch: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 10,
    batch_size: int = 64,
) -> dict:
    """PPO clipped surrogate update.
    
    Key insight: we want to improve the policy but not too much per update.
    The clip prevents the ratio π_new/π_old from going beyond [1-ε, 1+ε].
    """
    N = obs_batch.shape[0]
    losses = {"policy": [], "value": [], "entropy": []}

    for _ in range(epochs):
        # Shuffle data
        indices = torch.randperm(N)
        for start in range(0, N, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            obs = obs_batch[idx]
            acts = act_batch[idx]
            old_lp = old_log_probs[idx]
            adv = advantages[idx]
            ret = returns[idx]

            # Normalize advantages (crucial for stability)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Forward pass
            mean, std, values = model(obs)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(acts).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # TODO: Compute ratio and clipped surrogate objective
            # ratio = exp(new_log_probs - old_lp)
            # surr1 = ratio * adv
            # surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            # policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped or unclipped)
            value_loss = F.mse_loss(values, ret)

            # Total loss
            # loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # optimizer.zero_grad()
            # loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # optimizer.step()
            pass

    return losses
```

### 5.4 — Training Loop

```python
def train_ppo(
    env_name: str = "Pendulum-v1",
    total_timesteps: int = 200_000,
    steps_per_update: int = 2048,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    seed: int = 42,
):
    """Full PPO training loop on continuous control.

    TODO: Implement the collection + update loop:
    1. Collect `steps_per_update` transitions using current policy
    2. Compute GAE advantages
    3. Run ppo_update for multiple epochs
    4. Log mean episode reward

    Track and plot:
    - Episode rewards over time
    - Policy loss and value loss per update
    - Mean advantage magnitude (should decrease as policy improves)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # TODO: Implement training loop
    pass


# Run and compare with REINFORCE
# PPO should solve Pendulum much faster and more stably than REINFORCE
# train_ppo()
```

**Exercises**:
1. Train PPO on Pendulum-v1. Plot reward curve.
2. Ablate clip_eps: try 0.1, 0.2, 0.3. Which is most stable?
3. Remove clipping (set clip_eps=1.0). What happens? Why?
4. Compare PPO vs your REINFORCE on CartPole — measure sample efficiency.
5. **Connection to RLHF**: If `reward = human_preference_score`, this is exactly how ChatGPT was trained!

---

## Exercise 6: DDIM Deterministic Sampling

**Goal**: Implement DDIM (Denoising Diffusion Implicit Models) — the fast, deterministic
sampler used by Diffusion Policy. Same trained model, 10× fewer steps.

### 6.1 — Understanding DDIM vs DDPM

```python
# Key insight: DDPM adds noise at each reverse step (stochastic)
# DDIM removes the noise term → deterministic, and allows skipping steps

def ddpm_reverse_step(x_t, t, noise_pred, schedule):
    """DDPM: stochastic reverse step (from Exercise 3)."""
    alpha_t = schedule.alphas[t]
    alpha_bar_t = schedule.alpha_bars[t]
    beta_t = schedule.betas[t]
    
    mean = (1 / alpha_t.sqrt()) * (
        x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred
    )
    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = beta_t.sqrt()
        return mean + sigma * noise
    return mean


def ddim_reverse_step(x_t, t, t_prev, noise_pred, schedule, eta=0.0):
    """DDIM: (optionally) deterministic reverse step.
    
    When eta=0: fully deterministic (same noise → same output)
    When eta=1: equivalent to DDPM
    
    Formula:
    x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred 
              + sqrt(1 - ᾱ_{t-1} - σ²) * noise_pred
              + σ * random_noise
    
    where x0_pred = (x_t - sqrt(1-ᾱ_t) * noise_pred) / sqrt(ᾱ_t)
    """
    alpha_bar_t = schedule.alpha_bars[t]
    alpha_bar_prev = schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)
    
    # TODO: Implement DDIM step
    # 1. Predict x_0 from x_t and noise prediction
    # x0_pred = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
    
    # 2. Compute sigma (controls stochasticity)
    # sigma = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_prev)
    
    # 3. Compute direction pointing to x_t
    # direction = sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
    
    # 4. Combine
    # x_prev = sqrt(alpha_bar_prev) * x0_pred + direction + sigma * torch.randn_like(x_t)
    
    pass
```

### 6.2 — Subsampled Timestep Schedules

```python
def make_ddim_schedule(total_steps: int, ddim_steps: int) -> list[int]:
    """Create evenly-spaced timestep schedule for DDIM.
    
    Example: total_steps=200, ddim_steps=20
    → [190, 180, 170, ..., 10, 0]
    """
    # TODO: Implement
    # step_size = total_steps // ddim_steps
    # timesteps = list(range(total_steps - 1, -1, -step_size))[:ddim_steps]
    pass


def ddim_sample(model, schedule, shape, ddim_steps=20, eta=0.0):
    """Sample using DDIM with fewer steps.
    
    Uses the same trained noise-prediction model as DDPM!
    """
    timesteps = make_ddim_schedule(schedule.T, ddim_steps)
    
    x = torch.randn(shape)
    
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        
        t_batch = torch.full((shape[0],), t, dtype=torch.long)
        noise_pred = model(x, t_batch)
        
        x = ddim_reverse_step(x, t, t_prev, noise_pred, schedule, eta=eta)
    
    return x
```

### 6.3 — Comparison Experiment

```python
# TODO: Using your DDPM model trained in Exercise 3:
# 1. Sample with DDPM (200 steps) — measure time and quality
# 2. Sample with DDIM (200 steps, eta=0) — should give same quality
# 3. Sample with DDIM (50 steps, eta=0) — slightly worse but 4x faster
# 4. Sample with DDIM (20 steps, eta=0) — much faster, some quality loss
# 5. Sample with DDIM (10 steps, eta=0) — minimal steps, visible artifacts?

# Metrics to compare:
# - Wall-clock time per sample batch
# - Visual quality (plot samples from each)
# - Wasserstein distance to target distribution (scipy.stats.wasserstein_distance)

# Key takeaway: DDIM enables Diffusion Policy to run at 10 Hz
# because you only need ~10-16 denoising steps instead of 100-1000
```

**Exercises**:
1. Implement `ddim_reverse_step` and verify it reproduces DDPM when `eta=1.0`
2. Generate the same 2D Swiss Roll distribution with 200, 50, 20, and 10 DDIM steps
3. Plot quality vs speed tradeoff curve
4. **Determinism test**: with `eta=0`, starting from the same noise, DDIM always gives the same output. Verify.
5. **Connection to Diffusion Policy**: if one denoising step takes 5ms on GPU, how many steps can you afford at 10 Hz? (Answer: 10ms per action → 2 steps! This is why DDIM matters.)

---

## Self-Check

After completing these exercises, you should be able to answer:

- [ ] Why does REINFORCE have high variance, and how does a baseline help?
- [ ] How does PPO's clip mechanism prevent destructive updates?
- [ ] What is GAE and why does λ trade off bias vs variance?
- [ ] How does PPO connect to RLHF for language models?
- [ ] What happens during DDPM training (forward process) vs sampling (reverse)?
- [ ] Why can diffusion models represent multimodal distributions while regression cannot?
- [ ] How do you condition a diffusion model on observations?
- [ ] What makes DDIM deterministic, and why does that enable fewer steps?
- [ ] Why is DDIM critical for real-time robot control (Diffusion Policy)?

## Stretch Goals

1. **Flow matching**: Implement a simple flow matching variant — compare with DDPM/DDIM
2. **Multi-modal push data**: sometimes push left, sometimes right around obstacle
   — verify diffusion captures both modes while regression averages them
3. **PPO on MuJoCo**: Try PPO on HalfCheetah-v4 (harder continuous control)
4. **Action chunking**: modify Exercise 4 to predict sequences of 5 actions at once
