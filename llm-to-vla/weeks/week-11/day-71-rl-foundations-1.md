# Day 71: RL Foundations Day 1 — MDPs, Policies & Value Functions

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"Reinforcement learning is the science of decision-making under uncertainty." — Richard Sutton*

## Navigation
- **Previous:** [Day 70: VLM Fine-Tuning Day 2](../week-10/day-70-vlm-finetuning-2.md)
- **Next:** [Day 72: RL Foundations Day 2](day-72-rl-foundations-2.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/11-rl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 71.1 Markov Decision Process (MDP)

An MDP is a tuple $(S, A, P, R, \gamma)$:

| Symbol | Meaning | Robot Example |
|--------|---------|---------------|
| $S$ | State space | Joint angles, camera image, LiDAR |
| $A$ | Action space | Velocity commands, joint torques |
| $P(s' \mid s, a)$ | Transition dynamics | Physics + uncertainty |
| $R(s, a)$ | Reward function | +1 for goal, -0.01 per step |
| $\gamma$ | Discount factor | 0.99 (value future rewards) |

**Markov Property:** $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t)$

The future depends only on the current state, not the history. This is why state representation matters so much for robots.

### 71.2 Policies

A **policy** $\pi$ maps states to actions:
- **Deterministic:** $a = \pi(s)$
- **Stochastic:** $a \sim \pi(a \mid s)$

Why stochastic? Exploration. A deterministic policy can't discover better strategies.

```
Deterministic:  π(s) = argmax_a Q(s,a)     ← greedy, no exploration
Stochastic:     π(a|s) = softmax(Q(s,a)/τ)  ← temperature-controlled exploration
```

### 71.3 Value Functions

**State value function** — expected return from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s \right]$$

**Action value function** — expected return from state $s$, taking action $a$, then following $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s, a_0 = a \right]$$

**Advantage function** — how much better is action $a$ compared to the average?

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

### 71.4 Bellman Equations

The recursive structure of value functions:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s,a) + \gamma V^\pi(s') \right]$$

$$Q^\pi(s, a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')$$

**Optimal Bellman equation** (no policy — just take the best action):

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]$$

### 71.5 Policy Gradient Theorem

When the action space is continuous (like robot control), we parameterize the policy $\pi_\theta$ and optimize directly:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a) \right]$$

**Key insight:** we don't need to differentiate through the environment dynamics! Only through our policy.

**REINFORCE** — simplest policy gradient:
1. Collect a trajectory $\tau = (s_0, a_0, r_0, s_1, \ldots)$
2. Compute returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$
3. Update: $\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t$

**Problem:** high variance. We'll fix this tomorrow with baselines and actor-critic.

---

## Implementation (60 min)

### REINFORCE on CartPole

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return Categorical(logits=self.net(state))

def reinforce(env_name="CartPole-v1", episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        log_probs, rewards = [], []

        done = False
        while not done:
            state_t = torch.FloatTensor(state)
            dist = policy(state_t)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        loss = -sum(lp * G for lp, G in zip(log_probs, returns))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(sum(rewards))
        if ep % 50 == 0:
            avg = np.mean(reward_history[-50:])
            print(f"Episode {ep}: avg reward = {avg:.1f}")

    return policy, reward_history

policy, history = reinforce()
```

### Visualize Learning

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
window = 50
smoothed = [np.mean(history[max(0,i-window):i+1]) for i in range(len(history))]
plt.plot(history, alpha=0.3, label="Raw")
plt.plot(smoothed, label=f"{window}-episode average")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("REINFORCE on CartPole")
plt.legend()
plt.show()
```

---

## Exercise (45 min)

1. **Bellman by hand:** For a 3-state MDP with known transitions, compute $V^*$ by value iteration (3 iterations).

2. **Variance experiment:** Run REINFORCE with and without return normalization. Plot the learning curves — what happens?

3. **Continuous action space:** Replace CartPole with `Pendulum-v1` (continuous actions). Use a Gaussian policy:
   ```python
   class GaussianPolicy(nn.Module):
       def __init__(self, state_dim, action_dim):
           super().__init__()
           self.mean_net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
           self.log_std = nn.Parameter(torch.zeros(action_dim))

       def forward(self, state):
           mean = self.mean_net(state)
           return torch.distributions.Normal(mean, self.log_std.exp())
   ```

4. **Connect to Day 33 (RLHF):** Write down how REINFORCE relates to RLHF. What plays the role of the reward model?

---

## Key Takeaways

1. **MDPs formalize sequential decision-making** — state, action, transition, reward, discount
2. **Policy gradient theorem** lets us optimize policies without differentiating through dynamics
3. **REINFORCE is simple but high-variance** — we need baselines (tomorrow)
4. **Continuous actions need stochastic policies** — Gaussian policies for robot control
5. **The advantage function** $A(s,a)$ will become crucial for PPO (Day 73)

---

## Connection to the Thread

> *RL provides the optimization framework, but REINFORCE has too much variance for robotics. Tomorrow we add baselines and actor-critic methods. Day 73 brings PPO — the same algorithm that trains ChatGPT via RLHF. Then Days 74-77 introduce diffusion models, which will replace RL's explicit reward function with learned action distributions in Diffusion Policy (Day 81).*

---

## Further Reading

- Sutton & Barto, "Reinforcement Learning: An Introduction" — Ch. 3 (MDPs), Ch. 13 (Policy Gradient)
- Williams (1992), "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" — original REINFORCE
- Schulman et al. (2016), "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- OpenAI Spinning Up: [spinningup.openai.com](https://spinningup.openai.com) — excellent RL tutorials
