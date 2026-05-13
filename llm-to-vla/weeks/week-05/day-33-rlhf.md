# Day 33: RLHF — Reinforcement Learning from Human Feedback

> **Phase III — LLMs: Training & Alignment | Week 5 | 2.5 hours**
> *"RLHF is how you turn a knowledgeable autocomplete into a helpful assistant." — Jan Leike*

## Navigation
- **Previous:** [Day 32: Supervised Fine-Tuning](day-32-supervised-finetuning.md)
- **Next:** [Day 34: DPO & Modern Alignment](day-34-dpo-alignment.md)
- **Week:** [Week 5 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 33.1 The Alignment Problem

SFT teaches format but doesn't teach *values*. A model can follow instructions perfectly while being:
- Harmful (gives instructions for dangerous activities)
- Dishonest (confidently states falsehoods)
- Unhelpful (refuses benign requests out of excessive caution)

**The core challenge:** Human preferences are hard to specify as a loss function. We can't write `loss = harmfulness(output)` — but we *can* ask humans "which output do you prefer?"

### 33.2 The RLHF Pipeline

```
Step 1: Collect Comparisons          Step 2: Train Reward Model       Step 3: Optimize Policy
━━━━━━━━━━━━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━━
                                                                      
Prompt → SFT model → y₁, y₂          (prompt, y_w, y_l) pairs        LLM generates response
Human labels: y₁ > y₂                     ↓                          Reward model scores it
                                     Train classifier:               PPO updates LLM to
                                     r(prompt, y_w) > r(prompt, y_l)  maximize reward
                                                                      
  "Which is better?"                   "Learn what good means"        "Get better at good"
```

### 33.3 Reward Modeling

The reward model learns human preferences from pairwise comparisons.

**Bradley-Terry model:** Given chosen response $y_w$ and rejected response $y_l$:

$$
P(y_w \succ y_l \mid x) = \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)
$$

**Reward model loss:**

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\right]
$$

The reward model is typically initialized from the SFT model with the final unembedding layer replaced by a scalar head.

```
SFT Model (1.3B)                    Reward Model (1.3B)
┌──────────────┐                    ┌──────────────┐
│  Transformer │                    │  Transformer │  (same weights)
│  layers      │                    │  layers      │
│              │                    │              │
│  LM Head     │  ← replace →      │ Linear(d, 1) │  ← scalar reward
│  (vocab_size)│                    │              │
└──────────────┘                    └──────────────┘
```

### 33.4 PPO Optimization

PPO (Proximal Policy Optimization) updates the LLM policy to maximize the reward while staying close to the SFT model:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} A(x,y),\; \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A(x,y)\right)\right]
$$

With a **KL penalty** to prevent the model from drifting too far from the SFT model:

$$
\text{Reward}_{\text{total}} = r_\phi(x, y) - \beta \cdot D_{\text{KL}}\left[\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\right]
$$

**Why the KL penalty?** Without it, the model *reward hacks* — it finds degenerate outputs that score high on the imperfect reward model but are clearly bad to humans.

### 33.5 Reward Hacking

```
Without KL penalty:
  Model discovers: "I'm so sorry, I cannot help with that. As an AI..."
  → Reward model gives high score (looks safe!)
  → But it's useless for benign requests
  
  Model discovers: "Great question! Here are 47 reasons why..."
  → Reward model gives high score (looks helpful!)
  → But it's verbose and repetitive

With KL penalty:
  Model stays close to SFT behavior
  → Modest improvements in preferred directions
  → No mode collapse to degenerate patterns
```

### 33.6 The Full RLHF Loop

```
for each training step:
    1. Sample prompt x from dataset
    2. Generate response y ~ πθ(·|x)
    3. Compute reward r = rφ(x, y)
    4. Compute KL penalty: kl = β · log(πθ(y|x) / πref(y|x))
    5. Adjusted reward: r_adj = r - kl
    6. Compute PPO advantage estimates
    7. Update πθ with PPO objective
    8. Periodically: check reward model accuracy hasn't degraded
```

---

## Implementation (60 min)

### Simplified RLHF Loop with TRL

```python
"""
Day 33 Implementation: Simplified RLHF training loop.
Demonstrates reward modeling and PPO optimization concepts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ============================================================
# Part 1: Reward Model Training
# ============================================================

@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str

class SimpleRewardModel(nn.Module):
    """Reward model: scores (prompt, response) pairs."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        # Use last token's representation
        return self.reward_head(x[:, -1, :]).squeeze(-1)


def train_reward_model(
    model: SimpleRewardModel,
    pairs: list[PreferencePair],
    tokenizer,
    epochs: int = 10,
    lr: float = 1e-4,
) -> list[float]:
    """Train reward model on preference pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for pair in pairs:
            chosen_text = pair.prompt + " " + pair.chosen
            rejected_text = pair.prompt + " " + pair.rejected

            chosen_ids = tokenizer.encode(chosen_text, return_tensors="pt")
            rejected_ids = tokenizer.encode(rejected_text, return_tensors="pt")

            # Pad to same length
            max_len = max(chosen_ids.size(1), rejected_ids.size(1))
            chosen_ids = F.pad(chosen_ids, (0, max_len - chosen_ids.size(1)))
            rejected_ids = F.pad(rejected_ids, (0, max_len - rejected_ids.size(1)))

            r_chosen = model(chosen_ids)
            r_rejected = model(rejected_ids)

            # Bradley-Terry loss
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(pairs)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: RM loss = {avg:.4f}")

    return losses


# ============================================================
# Part 2: PPO-style Policy Update (Conceptual)
# ============================================================

def compute_kl_penalty(
    logprobs_policy: torch.Tensor,
    logprobs_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute per-token KL divergence penalty.

    KL(π_θ || π_ref) ≈ Σ_t [log π_θ(y_t|...) - log π_ref(y_t|...)]
    """
    return beta * (logprobs_policy - logprobs_ref).sum(dim=-1)


def rlhf_step(
    policy_model,
    ref_model,
    reward_model,
    prompt_ids: torch.Tensor,
    beta: float = 0.1,
):
    """Single RLHF training step (conceptual)."""
    # 1. Generate response from policy
    with torch.no_grad():
        response_ids = policy_model.generate(prompt_ids, max_new_tokens=50)

    # 2. Score with reward model
    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    reward = reward_model(full_ids)

    # 3. Compute log-probs under both models
    policy_logits = policy_model(full_ids).logits
    ref_logits = ref_model(full_ids).logits

    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1)

    # Gather log-probs for chosen tokens
    token_policy_lp = policy_logprobs.gather(-1, full_ids.unsqueeze(-1)).squeeze(-1)
    token_ref_lp = ref_logprobs.gather(-1, full_ids.unsqueeze(-1)).squeeze(-1)

    # 4. KL penalty
    kl = compute_kl_penalty(token_policy_lp, token_ref_lp, beta)

    # 5. Adjusted reward
    adjusted_reward = reward - kl

    return {
        "reward": reward.item(),
        "kl": kl.item(),
        "adjusted_reward": adjusted_reward.item(),
    }


# --- Demo ---
if __name__ == "__main__":
    # Demonstrate reward model training
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    rm = SimpleRewardModel(vocab_size=tokenizer.vocab_size)

    pairs = [
        PreferencePair(
            "Explain gravity",
            "Gravity is the force of attraction between masses, "
            "described by Newton's law F=Gm1m2/r^2.",
            "idk lol just google it"
        ),
        PreferencePair(
            "How to make a cake?",
            "Mix flour, sugar, eggs, and butter. Bake at 350°F for 30 min.",
            "I REFUSE to answer. This could be dangerous."
        ),
    ]

    losses = train_reward_model(rm, pairs, tokenizer, epochs=20)
    print(f"Final RM loss: {losses[-1]:.4f}")
```

---

## Exercise (45 min)

### E33.1 — Reward Model Analysis (25 min)
Using the `SimpleRewardModel` above:
1. Create 10 preference pairs covering different quality dimensions (helpfulness, safety, accuracy)
2. Train the reward model and visualize the loss curve
3. Test: does the model correctly rank new unseen response pairs?
4. Find a case where the reward model *disagrees* with your judgment — what does this tell you about reward hacking?

### E33.2 — KL Penalty Sweep (20 min)
Implement a sweep over $\beta \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$:
1. For each β, compute the adjusted reward for 5 example responses
2. Plot how β trades off reward vs. divergence from reference
3. At what β does the model start producing "safe but useless" outputs?

---

## Key Takeaways

1. **RLHF converts human preferences into a trainable signal** via reward modeling
2. **Bradley-Terry model** provides the mathematical framework for pairwise preferences
3. **PPO + KL penalty** optimizes the policy while preventing reward hacking
4. **Reward hacking** is the central failure mode — the model exploits reward model weaknesses
5. **RLHF is expensive:** requires human labelers, reward model training, and PPO optimization (3 models in memory)

---

## Connection to the Thread

RLHF's reward model is conceptually identical to the value function in robot RL (Phase VI). A robot learning to grasp objects needs a reward signal — either from a human ("that grasp looked good") or a learned critic. The KL penalty maps to keeping robot behavior close to a safe demonstration policy. We'll revisit PPO in detail during Phase VI.

---

## Further Reading

- [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) — The RLHF paper
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Early RLHF application
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [Secrets of RLHF in Large Language Models](https://arxiv.org/abs/2307.04964) — Practical guide
- [Open Problems in AI X-Risk: Reward Hacking](https://arxiv.org/abs/2109.13916) — Alignment research
