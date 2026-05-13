# Day 34: DPO & Modern Alignment

> **Phase III — LLMs: Training & Alignment | Week 5 | 2.5 hours**
> *"What if we could skip the reward model entirely?" — Rafael Rafailov*

## Navigation
- **Previous:** [Day 33: RLHF](day-33-rlhf.md)
- **Next:** [Day 35: LoRA & Efficient Fine-Tuning](day-35-lora-finetuning.md)
- **Week:** [Week 5 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 34.1 The Problem with RLHF

RLHF works but has significant practical downsides:

```
RLHF Pipeline (3 models in memory):
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│ Policy Model│   │Reference Model│   │ Reward Model │
│ (trainable) │   │   (frozen)   │   │  (frozen)    │
│    ~7B      │   │     ~7B      │   │    ~7B       │
└──────┬──────┘   └──────┬───────┘   └──────┬───────┘
       │                 │                   │
       └────── PPO ──────┴───── Score ───────┘
       
Total VRAM: ~3 × model size + optimizer states
Instability: PPO hyperparameters are finicky
```

**DPO's insight:** The optimal policy under RLHF has a closed-form solution. We can derive a loss that directly trains the policy on preferences — no reward model, no PPO, no RL at all.

### 34.2 Deriving DPO (The Key Math)

Starting from the RLHF objective with KL constraint:

$$
\max_{\pi_\theta} \mathbb{E}_{x,y \sim \pi_\theta}\left[r(x, y)\right] - \beta \cdot D_{\text{KL}}\left[\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\right]
$$

The **optimal policy** has a closed form:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

Solving for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

Substituting into the Bradley-Terry preference model (the $Z(x)$ terms cancel!):

$$
\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]}
$$

**What this says:** Increase the probability of chosen responses *relative to reference*, decrease the probability of rejected responses *relative to reference*.

### 34.3 DPO vs RLHF Comparison

```
                    RLHF                        DPO
                    ────                        ───
Models needed:      3 (policy + ref + reward)   2 (policy + ref)
Training:           PPO (RL loop)               Supervised (just backprop!)
Stability:          Sensitive to hyperparams    Much more stable
Memory:             ~3× model size              ~2× model size
Compute:            High (generation + scoring)  Lower (just forward passes)
Performance:        Gold standard               Comparable (sometimes better)
Reward hacking:     Possible                    Less susceptible
Hyperparameters:    Many (PPO clip, GAE λ, ...)  Few (β, learning rate)
```

### 34.4 Beyond DPO: Modern Alignment Methods

**IPO (Identity Preference Optimization):**
Fixes DPO's overfitting problem by adding a regularization term:

$$
\mathcal{L}_{\text{IPO}} = \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2
$$

**KTO (Kahneman-Tversky Optimization):**
Doesn't need paired preferences — works with individual "good" / "bad" labels:

$$
\mathcal{L}_{\text{KTO}} = \begin{cases}
1 - \sigma\bigl(\beta(r_\theta - z_{\text{ref}})\bigr) & \text{if } y \text{ is desirable} \\
1 - \sigma\bigl(\beta(z_{\text{ref}} - r_\theta)\bigr) & \text{if } y \text{ is undesirable}
\end{cases}
$$

where $r_\theta = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.

**ORPO (Odds Ratio Preference Optimization):**
Eliminates the reference model entirely by combining SFT and alignment:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

**Constitutional AI (Anthropic):**
Uses the model itself to generate and judge preferences:

```
1. Generate response to potentially harmful prompt
2. Ask model to critique its own response ("Red team")
3. Ask model to revise based on principles ("Constitutional revision")
4. Use (original, revised) as preference pair for DPO/RLHF
→ No human labelers needed for preference data!
```

### 34.5 Choosing an Alignment Method

```
Decision tree:
                    Have paired preferences?
                   /                        \
                 Yes                         No
                  │                           │
          Budget for 3 models?          Have good/bad labels?
         /              \              /              \
       Yes              No           Yes              No
        │                │             │                │
      RLHF             DPO           KTO        Constitutional AI
   (gold std)    (most popular)   (unpaired)    (self-supervised)
```

---

## Implementation (60 min)

### DPO Training with TRL

```python
"""
Day 34 Implementation: Direct Preference Optimization with TRL.
Fine-tune a model using DPO on preference data.
"""
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

# ============================================================
# Part 1: DPO Loss from Scratch
# ============================================================

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: Log-probs of chosen under policy [B]
        policy_rejected_logps: Log-probs of rejected under policy [B]
        reference_chosen_logps: Log-probs of chosen under reference [B]
        reference_rejected_logps: Log-probs of rejected under reference [B]
        beta: Temperature parameter
    """
    # Log-ratios
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Metrics for monitoring
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean()
    accuracy = (logits > 0).float().mean()

    return loss, {
        "loss": loss.item(),
        "reward_margin": reward_margin.item(),
        "accuracy": accuracy.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


def compute_sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute total log-probability of a sequence."""
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]  # shift
        targets = labels[:, 1:]              # shift

        logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        mask = (targets != 0).float()
        return (token_logprobs * mask).sum(dim=-1)


# ============================================================
# Part 2: DPO with TRL (production-ready)
# ============================================================

def create_preference_dataset() -> Dataset:
    """Create preference dataset for DPO training."""
    data = [
        {
            "prompt": "Explain how a robot navigates a warehouse.",
            "chosen": "A warehouse robot navigates using a combination of "
                      "SLAM (Simultaneous Localization and Mapping), LiDAR "
                      "for obstacle detection, and a global planner that "
                      "computes optimal paths using algorithms like A*.",
            "rejected": "The robot just drives around and tries not to hit "
                        "things. It uses sensors or something.",
        },
        {
            "prompt": "What should I do if a robot's sensorbar shows errors?",
            "chosen": "First, check the sensorbar LED indicators for fault "
                      "codes. Common issues include SPI communication "
                      "failures (check wiring), stiction from debris (clean "
                      "the sensors), or firmware version mismatch. Restart "
                      "the sensorbar controller and monitor /diagnostics.",
            "rejected": "Just restart the robot. If it doesn't work, restart "
                        "it again. Maybe call someone who knows robots.",
        },
        {
            "prompt": "How does battery exchange work for AMRs?",
            "chosen": "The AMR docks at a Battery Exchange Controller (BEC) "
                      "station. The process: 1) Robot aligns with dock using "
                      "IR sensors, 2) Mechanical latches release the depleted "
                      "battery, 3) Conveyor system extracts old battery, "
                      "4) Fresh battery is inserted, 5) Electrical connection "
                      "verified, 6) Robot undocks and resumes operations.",
            "rejected": "The robot goes to a station and swaps batteries. "
                        "It's automatic.",
        },
    ]
    return Dataset.from_list(data)


# --- Demo: DPO loss computation ---
if __name__ == "__main__":
    # Simulate log-probabilities
    B = 4
    policy_chosen_lp = torch.randn(B)
    policy_rejected_lp = torch.randn(B) - 0.5  # slightly worse
    ref_chosen_lp = torch.randn(B)
    ref_rejected_lp = torch.randn(B) - 0.3

    loss, metrics = compute_dpo_loss(
        policy_chosen_lp, policy_rejected_lp,
        ref_chosen_lp, ref_rejected_lp,
        beta=0.1,
    )
    print(f"DPO Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Reward margin: {metrics['reward_margin']:.4f}")

    # Beta sweep
    print("\nBeta sweep:")
    for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
        loss, m = compute_dpo_loss(
            policy_chosen_lp, policy_rejected_lp,
            ref_chosen_lp, ref_rejected_lp,
            beta=beta,
        )
        print(f"  β={beta:.2f}: loss={m['loss']:.4f}, "
              f"acc={m['accuracy']:.2%}, margin={m['reward_margin']:.4f}")
```

---

## Exercise (45 min)

### E34.1 — DPO Derivation Walk-Through (20 min)
Reproduce the DPO derivation on paper:
1. Start from the constrained RLHF objective
2. Write the Lagrangian and find the optimal policy $\pi^*$
3. Solve for $r(x,y)$ in terms of $\pi^*$ and $\pi_{\text{ref}}$
4. Substitute into Bradley-Terry and show the $Z(x)$ terms cancel
5. Write the final loss — verify it matches the boxed equation above

### E34.2 — Method Comparison (25 min)
Implement KTO loss alongside DPO:
1. Write `compute_kto_loss()` using individual good/bad labels
2. Convert the preference dataset to KTO format (split pairs into individual examples)
3. Compare: which loss is more stable across different β values?
4. Discuss: when would you prefer KTO over DPO?

---

## Key Takeaways

1. **DPO eliminates the reward model** by deriving a closed-form relationship between optimal policy and preferences
2. **The DPO loss** pushes chosen responses up and rejected responses down, *relative to a reference model*
3. **β controls alignment strength** — low β = strong alignment (may hurt capabilities), high β = weak alignment
4. **KTO works without paired data** — only needs "good" and "bad" labels
5. **Constitutional AI** generates its own preference data — no human labelers needed

---

## Connection to the Thread

DPO's core insight — bypassing RL by solving for the optimal policy analytically — recurs in robotics. Methods like IQL (Implicit Q-Learning) similarly avoid explicit RL by learning the value function directly from offline data. When we reach Phase VI, we'll see how robot alignment (learning human-preferred grasps, navigation styles) mirrors LLM alignment.

---

## Further Reading

- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — 2024
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Anthropic, 2022
- [IPO: A General Theoretical Paradigm](https://arxiv.org/abs/2310.12036) — Azar et al., 2023
