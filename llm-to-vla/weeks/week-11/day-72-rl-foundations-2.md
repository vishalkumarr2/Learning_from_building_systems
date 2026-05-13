# Day 72: RL Foundations Day 2 — Actor-Critic & GAE

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"The trick is to use one network to reduce the variance of the other." — Sutton & Barto*

## Navigation
- **Previous:** [Day 71: RL Foundations Day 1](day-71-rl-foundations-1.md)
- **Next:** [Day 73: PPO & RLHF Connection](day-73-ppo-rlhf.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/11-rl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 72.1 The Variance Problem with REINFORCE

REINFORCE uses the full return $G_t$ as the learning signal. This is **unbiased** but has enormous variance:

```
Episode 1: G_t = 127.3  → big positive update
Episode 2: G_t = 42.1   → smaller positive update
Episode 3: G_t = 189.7  → huge positive update

Problem: all actions get credit/blame, even the bad ones in a good episode
```

**Solution:** subtract a baseline $b(s)$ from the return:

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (G_t - b(s_t))\right]$$

The optimal baseline is $b(s) = V^\pi(s)$. Then $(G_t - V(s_t)) \approx A(s_t, a_t)$ — the advantage!

### 72.2 Actor-Critic Architecture

Two networks working together:

```
    State s_t
       │
   ┌───┴───────────────────────────┐
   │   Shared Feature Extractor    │
   └───┬───────────────┬───────────┘
       │               │
  ┌────▼────┐    ┌─────▼─────┐
  │  Actor  │    │  Critic   │
  │ π_θ(a|s)│    │  V_φ(s)   │
  └────┬────┘    └─────┬─────┘
       │               │
    Action a_t    Value estimate
```

- **Actor** (policy): chooses actions
- **Critic** (value function): estimates how good a state is

**Advantage Actor-Critic (A2C):**
- Actor loss: $\mathcal{L}_\text{actor} = -\log \pi_\theta(a_t|s_t) \cdot \hat{A}_t$
- Critic loss: $\mathcal{L}_\text{critic} = \|V_\phi(s_t) - G_t\|^2$
- Entropy bonus: $\mathcal{L}_\text{entropy} = -H[\pi_\theta(\cdot|s_t)]$ (encourages exploration)

### 72.3 Temporal Difference (TD) Learning

Instead of waiting for the full return $G_t$, use a one-step estimate:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This **TD error** $\delta_t$ is a biased but low-variance estimate of the advantage.

| Method | Bias | Variance | Data Needed |
|--------|------|----------|-------------|
| Monte Carlo ($G_t$) | None | High | Full episode |
| TD(0) ($\delta_t$) | Some | Low | Single step |
| GAE ($\hat{A}_t^\lambda$) | Tunable | Tunable | Partial trajectory |

### 72.4 Generalized Advantage Estimation (GAE)

GAE interpolates between MC and TD using parameter $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

| $\lambda$ | Behavior | Bias | Variance |
|-----------|----------|------|----------|
| 0 | Pure TD | High | Low |
| 1 | Pure MC | None | High |
| 0.95 | Sweet spot | Low | Moderate |

**In practice:** $\lambda = 0.95$, $\gamma = 0.99$ works well for most tasks.

### 72.5 Why This Matters for Robot Learning

Robots can't run millions of episodes like Atari. We need:
- **Sample efficiency**: learn from fewer interactions
- **Low variance**: stable training with limited data
- **Continuous actions**: joint velocities, not discrete buttons

Actor-critic + GAE gives us all three. PPO (tomorrow) adds stability guarantees.

---

## Implementation (60 min)

### Actor-Critic with GAE

```python
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return torch.distributions.Categorical(logits=logits), value.squeeze(-1)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    # Append 0 for terminal value
    values = list(values) + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

def train_a2c(env_name="CartPole-v1", steps=50000, n_steps=128,
              gamma=0.99, lam=0.95, lr=3e-4):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    state, _ = env.reset()
    ep_rewards = []
    current_ep_reward = 0

    for step in range(0, steps, n_steps):
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

        for _ in range(n_steps):
            state_t = torch.FloatTensor(state)
            dist, value = model(state_t)
            action = dist.sample()

            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value.item())

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
            dones.append(float(done))
            current_ep_reward += reward

            if done:
                ep_rewards.append(current_ep_reward)
                current_ep_reward = 0
                state, _ = env.reset()

        # Compute GAE advantages
        advantages = compute_gae(rewards, values, dones, gamma, lam)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        values_t = torch.tensor(values)

        # Losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = ((values_t - returns.detach()) ** 2).mean()
        entropy = -log_probs.mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if len(ep_rewards) >= 50 and step % (n_steps * 10) == 0:
            print(f"Step {step}: avg reward = {np.mean(ep_rewards[-50:]):.1f}")

    return model, ep_rewards
```

---

## Exercise (45 min)

1. **GAE by hand:** Given trajectory $(r_0=1, r_1=0, r_2=1)$ with $V = (0.5, 0.3, 0.7)$, $\gamma=0.99$, $\lambda=0.95$, compute $\hat{A}_0^{\text{GAE}}$.

2. **Lambda ablation:** Train A2C with $\lambda \in \{0.0, 0.5, 0.95, 1.0\}$. Plot learning curves and measure variance of the gradient estimates.

3. **Shared vs separate networks:** Modify `ActorCritic` to use separate feature extractors for actor and critic. Compare training stability.

4. **Continuous actions:** Adapt to `Pendulum-v1` with a Gaussian actor head outputting mean and log-std.

---

## Key Takeaways

1. **Baselines reduce variance** without introducing bias — $V(s)$ is the optimal baseline
2. **Actor-critic** = policy network + value network working together
3. **GAE** smoothly trades bias for variance via $\lambda$
4. **TD errors** $\delta_t$ are the building blocks of advantage estimation
5. **$\lambda = 0.95$** is the standard choice across most RL implementations

---

## Connection to the Thread

> *Actor-critic + GAE gives us the advantage estimation backbone. Tomorrow's PPO adds the critical ingredient for stable training: a clipped objective that prevents destructive policy updates. This same PPO algorithm is what OpenAI uses for RLHF (Day 33) — the connection between aligning LLMs and training robot policies is deeper than it seems.*

---

## Further Reading

- Schulman et al. (2016), "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- Mnih et al. (2016), "Asynchronous Methods for Deep Reinforcement Learning" (A3C/A2C)
- Sutton & Barto Ch. 13 — Actor-Critic methods
