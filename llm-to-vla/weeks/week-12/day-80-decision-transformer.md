# Day 80: Decision Transformer — Offline RL as Sequence Modeling

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 12 | 2.5 hours**
> *"What if we just treat RL as a sequence prediction problem? No Bellman. No value functions. Just transformers." — Chen et al., 2021*

## Navigation
- **Previous:** [Day 79: ACT — Action Chunking Transformers](day-79-act.md)
- **Next:** [Day 81: Diffusion Policy](day-81-diffusion-policy.md)
- **Week:** [Week 12 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/13-imitation-learning.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 80.1 The Key Insight

Traditional RL: learn value functions, compute Bellman backups, estimate advantages.

Decision Transformer: **just predict the next token in a sequence** — where the sequence is (return, state, action, return, state, action, ...).

```
Standard RL:                      Decision Transformer:
  s → π(a|s) optimized by          s → Transformer → a
  reward signal + value func        conditioned on desired return R̂

  Requires: reward, value fn        Requires: offline dataset only
  Online interaction: yes           Online interaction: no
  Algorithm: PPO, SAC, etc.         Algorithm: supervised learning!
```

### 80.2 Architecture

The input sequence interleaves returns-to-go, states, and actions:

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T)$$

where $\hat{R}_t = \sum_{k=t}^{T} r_k$ is the **return-to-go** (desired future return).

```
Input tokens:
  [R̂₁] [s₁] [a₁] [R̂₂] [s₂] [a₂] [R̂₃] [s₃] [a₃] ...
    │    │    │     │    │    │     │    │    │
    ▼    ▼    ▼     ▼    ▼    ▼     ▼    ▼    ▼
  ┌──────────────────────────────────────────────────┐
  │         Causal Transformer (GPT-style)           │
  └──────────────────────────────────────────────────┘
    │    │    │     │    │    │     │    │    │
    ▼    ▼    ▼     ▼    ▼    ▼     ▼    ▼    ▼
  [·]  [·]  [â₁]  [·]  [·]  [â₂]  [·]  [·]  [â₃]

  Only action predictions are used (at every 3rd position)
```

### 80.3 Return-Conditioned Generation

At inference, **you specify the desired return**:

```python
# "I want a high-performing trajectory"
desired_return = 500  # max possible return

# Feed: [R̂=500, s_current, ???]
# Model predicts: a_1
# Execute a_1, observe s_2, r_1
# Update: R̂_2 = R̂_1 - r_1 = 500 - r_1
# Feed: [..., R̂_2, s_2, ???]
# Model predicts: a_2
# ...
```

**This is the connection to LLMs:** just like prompting a language model with "write excellent code" biases generation toward quality, conditioning DT on high returns biases it toward good actions.

### 80.4 Comparison to Standard Offline RL

| Aspect | Offline RL (CQL, IQL) | Decision Transformer |
|--------|----------------------|---------------------|
| Algorithm | Modified Bellman backups | Supervised sequence prediction |
| Value function | Required | Not needed |
| Stitching | Can stitch suboptimal trajectories | Struggles with stitching |
| Implementation | Complex | Simple (GPT training loop) |
| Conditioning | Fixed objective | Flexible (vary return target) |

**Stitching limitation:** DT can only reproduce behaviors seen in the dataset. It can't combine the first half of one trajectory with the second half of another to get a better result. Standard offline RL methods can.

### 80.5 Connection to Language Models

| LM Concept | DT Equivalent |
|-----------|---------------|
| Token vocabulary | (return, state, action) triplets |
| Context window | Trajectory history |
| Prompt | Desired return-to-go |
| Next-token prediction | Next-action prediction |
| Temperature | Not applicable (continuous) |

---

## Implementation (60 min)

### Decision Transformer

```python
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden=128, n_layers=3,
                 n_heads=4, max_length=20):
        super().__init__()
        self.hidden = hidden
        self.max_length = max_length

        # Embeddings for each modality
        self.state_embed = nn.Linear(state_dim, hidden)
        self.action_embed = nn.Linear(act_dim, hidden)
        self.return_embed = nn.Linear(1, hidden)

        # Learned position embedding
        self.pos_embed = nn.Embedding(3 * max_length, hidden)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Action prediction head
        self.action_head = nn.Linear(hidden, act_dim)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, returns_to_go, states, actions, timesteps):
        B, T = states.shape[0], states.shape[1]

        # Embed each modality
        r_emb = self.return_embed(returns_to_go.unsqueeze(-1))  # (B, T, H)
        s_emb = self.state_embed(states)                         # (B, T, H)
        a_emb = self.action_embed(actions)                       # (B, T, H)

        # Interleave: [R₁, s₁, a₁, R₂, s₂, a₂, ...]
        seq = torch.stack([r_emb, s_emb, a_emb], dim=2)  # (B, T, 3, H)
        seq = seq.view(B, 3*T, self.hidden)

        # Add position embeddings
        positions = torch.arange(3*T, device=seq.device)
        seq = seq + self.pos_embed(positions)

        # Causal mask
        mask = torch.triu(torch.ones(3*T, 3*T, device=seq.device), diagonal=1).bool()

        # Transform
        out = self.transformer(self.ln(seq), mask=mask)

        # Extract action predictions (at state positions: indices 1, 4, 7, ...)
        state_positions = torch.arange(1, 3*T, 3)
        action_preds = self.action_head(out[:, state_positions])

        return action_preds

# --- Training ---
def train_dt(dataset, epochs=200, lr=1e-4, context_len=20):
    state_dim = dataset["observations"].shape[-1]
    act_dim = dataset["actions"].shape[-1]
    model = DecisionTransformer(state_dim, act_dim, max_length=context_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Sample random trajectory windows
        batch_idx = np.random.randint(0, len(dataset["observations"]), 32)
        start = np.random.randint(0, dataset["observations"].shape[1] - context_len, 32)

        states = torch.FloatTensor(np.array([
            dataset["observations"][i, s:s+context_len]
            for i, s in zip(batch_idx, start)
        ]))
        actions = torch.FloatTensor(np.array([
            dataset["actions"][i, s:s+context_len]
            for i, s in zip(batch_idx, start)
        ]))
        returns = torch.FloatTensor(np.array([
            dataset["returns_to_go"][i, s:s+context_len]
            for i, s in zip(batch_idx, start)
        ]))
        timesteps = torch.arange(context_len).unsqueeze(0).expand(32, -1)

        pred_actions = model(returns, states, actions, timesteps)
        loss = ((pred_actions - actions) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return model
```

---

## Exercise (45 min)

1. **Return conditioning:** Collect a mixed-quality dataset. Train DT. At inference, try $\hat{R} \in$ {low, medium, high}. Does the policy quality scale with the requested return?

2. **Context length ablation:** Train with context lengths $K \in \{1, 5, 10, 20\}$. How does performance change? When does more context stop helping?

3. **Compare BC vs DT:** On the same dataset, train vanilla BC (ignoring returns) and DT. When does DT's return conditioning help? When doesn't it?

4. **Connection to GPT:** Write a paragraph comparing DT's return-conditioned generation to LLM prompting. What's the "system prompt" equivalent for a robot?

---

## Key Takeaways

1. **DT treats RL as sequence modeling** — no value functions, no Bellman
2. **Return-to-go conditioning** = "prompt" for desired behavior quality
3. **Same training as GPT** — just predict the next action token
4. **Limitation: no stitching** — can't combine suboptimal trajectory segments
5. **The bridge:** this approach directly led to RT-2's idea of treating robot control as language modeling

---

## Connection to the Thread

> *Decision Transformer shows that transformers can do RL without RL algorithms. Tomorrow, Diffusion Policy takes the opposite approach: use diffusion models (Days 74-76) to model the full action distribution. Where DT conditions on desired return, Diffusion Policy conditions on observations. Where DT predicts one action sequence, Diffusion Policy can sample diverse strategies. RT-2 (Day 93) will combine both ideas: a VLM that generates action tokens like a language model.*

---

## Further Reading

- Chen et al. (2021), "Decision Transformer: Reinforcement Learning via Sequence Modeling"
- Janner et al. (2021), "Offline Reinforcement Learning as One Big Sequence Modeling Problem" (Trajectory Transformer)
- Lee et al. (2022), "Multi-Game Decision Transformers"
