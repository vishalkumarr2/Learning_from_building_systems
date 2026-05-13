# Day 73: PPO & RLHF Connection

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"PPO is the workhorse of modern RL — simple enough to implement, stable enough to actually work." — John Schulman*

## Navigation
- **Previous:** [Day 72: RL Foundations Day 2](day-72-rl-foundations-2.md)
- **Next:** [Day 74: Diffusion Models — DDPM](day-74-ddpm.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/11-rl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 73.1 Why Not Just Use Policy Gradients?

Vanilla policy gradient (REINFORCE, A2C) has a critical flaw: **step size sensitivity**.

```
Too small step  → painfully slow learning
Too large step  → policy collapses, catastrophic forgetting
Just right      → unknown in advance, changes during training
```

**Trust Region** idea: constrain how much the policy can change per update.

### 73.2 PPO: Clipped Surrogate Objective

PPO approximates a trust region using a simple clipped objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$ — probability ratio
- $\epsilon = 0.2$ — clipping threshold
- $\hat{A}_t$ — GAE advantage from Day 72

```
           r(θ)·Â
             │
    ─────────┼───────── (1+ε)·Â
             │ ╱
             │╱  ← clipped: prevents too-large updates
    ─────────┼───────── (1-ε)·Â
            ╱│
           ╱ │
    ──────╱──┼─────────
         ╱   │
     r(θ): 0.8  1.0  1.2
```

**Why it works:** if the advantage is positive and the ratio exceeds $1+\epsilon$, the gradient is zeroed out. The policy can't move too far in one update.

### 73.3 PPO Training Loop

```
repeat:
  1. Collect N timesteps with current policy π_old
  2. Compute GAE advantages Â_t
  3. For K epochs over the collected data:
     a. Compute r(θ) = π_θ(a|s) / π_old(a|s)
     b. Compute clipped objective L^CLIP
     c. Update θ via gradient ascent
  4. Set π_old ← π_θ
```

**Key hyperparameters:**

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| $\epsilon$ (clip) | 0.2 | Trust region width |
| K (epochs) | 3–10 | Reuse of collected data |
| N (batch) | 2048–8192 | Data per update |
| $\gamma$ | 0.99 | Discount factor |
| $\lambda$ (GAE) | 0.95 | Advantage bias-variance |
| lr | 3e-4 | Learning rate |

### 73.4 The RLHF Connection

Recall Day 33: RLHF trains LLMs using PPO where:

| RL Component | RLHF Mapping |
|-------------|--------------|
| State $s$ | Prompt + partial generation |
| Action $a$ | Next token |
| Policy $\pi_\theta$ | Language model |
| Reward $R$ | Reward model score |
| $\pi_\text{ref}$ | Frozen pre-trained LM (KL penalty) |

$$\text{RLHF objective} = \mathbb{E}\left[R(y) - \beta \cdot D_\text{KL}(\pi_\theta \| \pi_\text{ref})\right]$$

**Same algorithm, different domain:** PPO clips policy updates to prevent catastrophic forgetting — whether the policy generates text or robot actions.

### 73.5 PPO for Robot Learning

For continuous-action robot control:
- Policy outputs Gaussian: $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$
- Actions: joint velocities, end-effector deltas
- Reward: task-specific (reach target, grasp object)

**Challenge:** reward engineering is hard for robots. This motivates learning from demonstrations (Day 78) and diffusion-based policies (Day 81) instead.

---

## Implementation (60 min)

### PPO Implementation

```python
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def get_action_and_value(self, state):
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(state).squeeze(-1)

    def evaluate(self, state, action):
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), self.critic(state).squeeze(-1)

def ppo_train(env_name="CartPole-v1", total_steps=100000,
              n_steps=2048, n_epochs=10, clip_eps=0.2,
              gamma=0.99, lam=0.95, lr=3e-4):

    env = gym.make(env_name)
    model = PPOActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    state, _ = env.reset()
    ep_rewards, current_reward = [], 0

    for update in range(total_steps // n_steps):
        # --- Collect rollout ---
        states, actions, rewards, dones = [], [], [], []
        old_log_probs, values = [], []

        for _ in range(n_steps):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(state_t)

            states.append(state_t)
            actions.append(action)
            old_log_probs.append(log_prob)
            values.append(value)

            state, reward, term, trunc, _ = env.step(action.item())
            rewards.append(reward)
            dones.append(float(term or trunc))
            current_reward += reward

            if term or trunc:
                ep_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()

        # --- Compute GAE ---
        vals = [v.item() for v in values] + [0]
        advantages, gae = [], 0
        for t in reversed(range(n_steps)):
            delta = rewards[t] + gamma * vals[t+1] * (1-dones[t]) - vals[t]
            gae = delta + gamma * lam * (1-dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(vals[:-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)

        # --- PPO update ---
        for epoch in range(n_epochs):
            new_log_probs, entropy, new_values = model.evaluate(states, actions)
            ratio = (new_log_probs - old_log_probs.detach()).exp()

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((new_values - returns.detach())**2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if ep_rewards:
            print(f"Update {update}: avg reward = {np.mean(ep_rewards[-20:]):.1f}")

    return model, ep_rewards
```

---

## Exercise (45 min)

1. **Clip visualization:** For fixed $\hat{A} > 0$, plot $L^\text{CLIP}$ vs $r(\theta)$. Identify the flat region where gradient = 0.

2. **Epsilon ablation:** Train PPO with $\epsilon \in \{0.05, 0.1, 0.2, 0.3, 0.5\}$. What breaks at extremes?

3. **RLHF simulation:** Implement a toy RLHF loop: use PPO to fine-tune a small character-level LM where the "reward model" is just string length. Observe mode collapse vs KL penalty.

4. **Compare REINFORCE vs A2C vs PPO:** Run all three on CartPole with same number of environment steps. Plot learning curves.

---

## Key Takeaways

1. **PPO clips the probability ratio** to prevent destructive policy updates
2. **Same data, multiple epochs** — PPO reuses collected trajectories (K epochs)
3. **PPO = the algorithm behind RLHF** — Day 33's missing implementation detail
4. **Reward engineering is the bottleneck** for robot RL → motivates imitation learning
5. **GAE + PPO + clipping** = the standard recipe for modern RL

---

## Connection to the Thread

> *PPO completes the RL toolkit. But robot RL's Achilles heel is reward design: how do you specify "pick up the mug carefully" as a scalar reward? The answer is you don't — you show demonstrations instead. Before we get to imitation learning (Week 12), we need one more tool: diffusion models. Starting tomorrow, we learn to generate by denoising — a framework that will transform how robots produce actions.*

---

## Further Reading

- Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback" (InstructGPT/RLHF)
- Andrychowicz et al. (2020), "What Matters In On-Policy Reinforcement Learning?"
