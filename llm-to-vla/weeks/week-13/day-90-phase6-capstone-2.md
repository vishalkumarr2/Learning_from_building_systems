# Day 90: Phase VI Capstone — Day 2: Advanced Policy & Evaluation

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 13 | 3 hours**
> *"Upgrade from supervised regression to generative action modeling. Then prove it works."*

## Navigation
- **Previous:** [Day 89: Phase VI Capstone Day 1](day-89-phase6-capstone-1.md)
- **Next:** [Day 91: Phase VI Capstone Day 3](day-91-phase6-capstone-3.md)
- **Week:** [Week 13 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/14-robot-data-eval.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Part 1: Advanced Policy (90 min)

### Choose Your Architecture

Pick one and implement it using the patterns from Week 12:

| Option | Complexity | Best For |
|--------|-----------|----------|
| **A) Diffusion Policy** (Day 81) | High | Multimodal tasks |
| **B) ACT** (Day 79) | High | Bimanual, long-horizon |
| **C) Chunked BC + GMM** | Medium | Good middle ground |

### Option C: Chunked BC with Gaussian Mixture Model

If time is limited, this is the fastest upgrade from baseline:

```python
import torch
import torch.nn as nn
import torch.distributions as D

class GMMPolicy(nn.Module):
    """BC with Gaussian Mixture Model output for multimodal actions."""
    def __init__(self, obs_dim, act_dim, chunk_size, n_modes=5, hidden=256):
        super().__init__()
        self.n_modes = n_modes
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        output_per_mode = act_dim * chunk_size  # mean
        total_output = n_modes * (1 + output_per_mode + output_per_mode)
        # n_modes * (log_weight + mean + log_std)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, total_output),
        )

    def forward(self, obs):
        B = obs.shape[0]
        out = self.net(obs)
        flat_dim = self.act_dim * self.chunk_size

        # Split into weights, means, log_stds
        log_weights = out[:, :self.n_modes]
        means = out[:, self.n_modes:self.n_modes + self.n_modes * flat_dim]
        log_stds = out[:, self.n_modes + self.n_modes * flat_dim:]

        means = means.view(B, self.n_modes, self.chunk_size, self.act_dim)
        log_stds = log_stds.view(B, self.n_modes, self.chunk_size, self.act_dim)
        log_stds = torch.clamp(log_stds, -5, 2)

        return log_weights, means, log_stds

    def loss(self, obs, target_actions):
        """Negative log-likelihood of GMM."""
        log_weights, means, log_stds = self.forward(obs)

        # target_actions: (B, H, act_dim) → expand for modes
        target = target_actions.unsqueeze(1).expand_as(means)

        # Per-mode log probability
        var = (2 * log_stds).exp()
        log_probs = -0.5 * (((target - means) ** 2) / var + 2 * log_stds + np.log(2 * np.pi))
        log_probs = log_probs.sum(dim=(-1, -2))  # Sum over time and action dims

        # Log-sum-exp over modes
        log_weights_norm = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        log_mixture = torch.logsumexp(log_weights_norm + log_probs, dim=1)

        return -log_mixture.mean()

    @torch.no_grad()
    def predict(self, obs):
        log_weights, means, log_stds = self.forward(obs)
        # Sample mode
        weights = torch.softmax(log_weights, dim=1)
        mode_idx = torch.multinomial(weights, 1).squeeze(1)
        # Get mean of selected mode
        batch_idx = torch.arange(obs.shape[0])
        return means[batch_idx, mode_idx]  # (B, H, act_dim)

# Training loop
import numpy as np

gmm_model = GMMPolicy(obs_dim=3, act_dim=1, chunk_size=8, n_modes=5)
gmm_optimizer = torch.optim.Adam(gmm_model.parameters(), lr=1e-4)

for epoch in range(300):
    total_loss = 0
    for obs_batch, act_batch in loader:
        loss = gmm_model.loss(obs_batch, act_batch)
        gmm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gmm_model.parameters(), 1.0)
        gmm_optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: GMM loss = {total_loss/len(loader):.4f}")
```

---

## Part 2: Systematic Evaluation (60 min)

### Head-to-Head Comparison

```python
def evaluate_policy(env, policy_fn, n_trials=50, max_steps=200):
    """Evaluate any policy function."""
    successes = 0
    total_reward_list = []

    for trial in range(n_trials):
        obs, _ = env.reset(seed=trial + 1000)
        total_reward = 0
        for step in range(max_steps):
            action = policy_fn(obs)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                break
        total_reward_list.append(total_reward)
        if total_reward > -500:  # Pendulum: higher is better
            successes += 1

    sr = successes / n_trials
    mean_reward = np.mean(total_reward_list)
    std_reward = np.std(total_reward_list)

    # Wilson CI
    z = 1.96
    denom = 1 + z**2/n_trials
    center = (sr + z**2/(2*n_trials)) / denom
    spread = z * np.sqrt((sr*(1-sr) + z**2/(4*n_trials))/n_trials) / denom

    return {
        "success_rate": sr,
        "ci_95": (max(0, center-spread), min(1, center+spread)),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

# Compare
print("Evaluating Expert...")
expert_results = evaluate_policy(env, expert_policy)
print(f"  Expert: SR={expert_results['success_rate']:.1%}, "
      f"Reward={expert_results['mean_reward']:.1f}")

print("\nEvaluating BC Baseline...")
bc_fn = lambda obs: model.predict(obs, stats)
bc_results = evaluate_policy(env, bc_fn)
print(f"  BC: SR={bc_results['success_rate']:.1%}, "
      f"CI={bc_results['ci_95']}, "
      f"Reward={bc_results['mean_reward']:.1f}")

print("\nEvaluating GMM Policy...")
def gmm_fn(obs):
    obs_norm = (obs - stats["obs_mean"]) / stats["obs_std"]
    obs_t = torch.FloatTensor(obs_norm).unsqueeze(0)
    act_norm = gmm_model.predict(obs_t)[0, 0].numpy()
    return act_norm * stats["act_std"] + stats["act_mean"]

gmm_results = evaluate_policy(env, gmm_fn)
print(f"  GMM: SR={gmm_results['success_rate']:.1%}, "
      f"CI={gmm_results['ci_95']}, "
      f"Reward={gmm_results['mean_reward']:.1f}")
```

### Ablation Study

```python
# Ablation: vary number of demonstrations
for n_demos in [10, 25, 50, 100]:
    subset = {k: [v[i] for i in range(min(n_demos, len(v)))]
              for k, v in dataset.items() if k != "episode_starts"}
    subset["episode_starts"] = list(range(len(subset["observations"])))
    train_sub, _, stats_sub = process_dataset(subset, chunk_size=8)

    # Train quick BC
    model_sub = BaselineBC(obs_dim, act_dim, chunk_size)
    opt_sub = torch.optim.Adam(model_sub.parameters(), lr=1e-4)
    loader_sub = DataLoader(
        TensorDataset(train_sub["obs"], train_sub["acts"]),
        batch_size=256, shuffle=True,
    )
    for _ in range(200):
        for ob, ac in loader_sub:
            loss = ((model_sub(ob) - ac)**2).mean()
            opt_sub.zero_grad()
            loss.backward()
            opt_sub.step()

    fn = lambda obs, m=model_sub, s=stats_sub: m.predict(obs, s)
    r = evaluate_policy(env, fn, n_trials=30)
    print(f"  {n_demos} demos: SR={r['success_rate']:.1%}, "
          f"Reward={r['mean_reward']:.1f}")
```

---

## Part 3: Debugging (30 min)

Use the debugging toolkit from Day 88:

1. **Collect 20 debug rollouts** with your best policy
2. **Classify failures** into categories
3. **Form a hypothesis** about the dominant failure mode
4. **Implement one fix** and re-evaluate

---

## Deliverables for Today

- [ ] Advanced policy (Diffusion, ACT, or GMM) trained
- [ ] Head-to-head comparison table: Expert vs BC vs Advanced
- [ ] Ablation study: data quantity impact
- [ ] Failure analysis with at least one fix applied

---

## Connection to the Thread

> *Tomorrow: final integration, Phase VI reflection, and the 8-question checkpoint quiz covering RL, diffusion, imitation learning, action representations, and evaluation. Then Phase VII begins: actual VLA architectures.*
