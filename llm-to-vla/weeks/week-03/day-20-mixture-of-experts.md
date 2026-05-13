# Day 20: Mixture of Experts (MoE) — Sparse Activation at Scale

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"Not all experts need to weigh in on every question. The art is in the routing." — Shazeer et al.*

## Navigation
- **Previous:** [Day 19: Normalization + Activations](day-19-normalization-activations.md)
- **Next:** [Day 21: BERT & Masked LM](day-21-bert-masked-lm.md)
- **Week:** [Week 3 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/03-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 20.1 The Scaling Dilemma

Dense transformers have a problem: **every parameter is active for every token**.

- LLaMA-70B: 70B parameters active per token → expensive inference
- Want: more knowledge capacity (more parameters) WITHOUT proportionally more compute

**MoE insight:** Replace the single FFN in each transformer layer with $N$ parallel FFN "experts," but only activate $k$ of them per token.

```
Dense Transformer Layer:               MoE Transformer Layer:
┌────────────┐                         ┌────────────┐
│   Attention │                         │   Attention │
├────────────┤                         ├────────────┤
│   Single   │  ← all params          │   Router    │ ← tiny network
│   FFN      │     active             ┌┤ scores each│   picks top-k
│            │                        │ │ expert     │
│            │                        │ ├────────────┤
│            │                        │ │ Expert 1   │ ← only 2 of 8
│            │                        │ │ Expert 2   │   are active!
│            │                        │ │ Expert 3 ✓ │
└────────────┘                        │ │ Expert 4   │
                                      │ │ Expert 5 ✓ │
Total params = active params          │ │ Expert 6   │
                                      │ │ Expert 7   │
                                      │ │ Expert 8   │
                                      │ └────────────┘
                                      Total params >> active params
```

### 20.2 The Router

The router (or gating network) decides which experts process each token:

$$G(x) = \text{TopK}(\text{softmax}(xW_g), k)$$

where $W_g \in \mathbb{R}^{d \times N}$ maps each token to $N$ expert scores.

```
Token embedding ──→ [W_g: d×8] ──→ softmax ──→ [0.05, 0.12, 0.41, 0.03, 0.31, 0.02, 0.04, 0.02]
                                                         ↓ top-2
                                              Expert 3 (weight 0.57) ──→ ┐
                                              Expert 5 (weight 0.43) ──→ ┤──→ weighted sum
                                                                          └──→ output

Output = 0.57 × Expert3(x) + 0.43 × Expert5(x)
```

The router weights are renormalized so the top-k weights sum to 1.

### 20.3 Load Balancing — The Critical Challenge

**Problem: expert collapse.** Without intervention, the router learns to send ALL tokens to 1-2 experts (the rich get richer):

```
Step 0:    Expert usage: [13%, 12%, 13%, 12%, 13%, 12%, 13%, 12%]  ← balanced
Step 1000: Expert usage: [45%, 30%, 5%, 5%, 5%, 5%, 3%, 2%]       ← collapsing
Step 5000: Expert usage: [92%, 8%, 0%, 0%, 0%, 0%, 0%, 0%]        ← collapsed!
```

**Solution: auxiliary load balancing loss.**

For $N$ experts and a batch of $T$ tokens:

$$\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i = \frac{\text{tokens routed to expert } i}{T}$ (fraction of tokens)
- $P_i = \frac{1}{T}\sum_{t=1}^{T} p_i(x_t)$ (average routing probability for expert $i$)

This loss is minimized when $f_i = P_i = \frac{1}{N}$ (uniform distribution).

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{balance}}$$

Typically $\alpha = 0.01$.

### 20.4 Key MoE Architectures

**Switch Transformer (Fedus et al., 2022):**
- Top-1 routing (only ONE expert per token)
- Simplest routing, fewest communication needs
- Capacity factor limits how many tokens each expert processes
- Result: 7× faster training with same compute as dense T5

**Mixtral 8×7B (Mistral AI, 2024):**
- 8 experts, top-2 routing
- Each expert is a 7B-parameter FFN (same as Mistral-7B)
- Total params: **46.7B** (8 experts × ~5.6B FFN params + shared attention)
- Active params: **12.9B** per token (attention + 2 expert FFNs)
- Matches or exceeds LLaMA-2 70B quality with 5× less inference compute

```
Mixtral Architecture:
┌──────────────────────────────────────┐
│ Shared: Attention (same for all tokens) │
├──────────────────────────────────────┤
│ Router → Top-2 selection              │
├───────┬───────┬─────┬───────┬────────┤
│ FFN-1 │ FFN-2 │ ... │ FFN-7 │ FFN-8  │
│ (7B)  │ (7B)  │     │ (7B)  │ (7B)   │  ← Only 2 active
└───────┴───────┴─────┴───────┴────────┘

Active per token:  Attention (~1.3B) + 2×FFN (~5.8B×2) = 12.9B
Total parameters:  Attention + 8×FFN = 46.7B
Memory needed:     ALL 46.7B must be in memory (even though only 12.9B active)
```

### 20.5 Trade-offs

| Aspect | Dense Model | MoE Model |
|--------|-------------|-----------|
| Parameters | All active | Only top-k active |
| Quality at same compute | Baseline | Better (more capacity) |
| Memory | = active params | = total params (>>active) |
| Inference speed | Proportional to params | Proportional to active params |
| Training complexity | Simple | Need load balancing, communication |
| Hardware utilization | High | Can be low (expert imbalance) |

**The memory catch:** Even though Mixtral only uses 12.9B params per forward pass, you need ALL 46.7B in GPU memory. MoE trades compute savings for memory overhead.

---

## Implementation (60 min)

### 20.6 MoE Layer from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Expert(nn.Module):
    """A single expert — standard FFN."""

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, d_model=256, hidden_dim=512, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: maps token → expert scores
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Create n_experts independent FFNs
        self.experts = nn.ModuleList([
            Expert(d_model, hidden_dim) for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: load balancing loss
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        # Router scores
        router_logits = self.router(x_flat)  # (B*T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs (simple loop — not optimal but clear)
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]  # which expert for each token
            weights = top_k_weights[:, i]          # weight for this expert

            for e_idx in range(self.n_experts):
                mask = (expert_indices == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        output = output.view(B, T, D)

        # Load balancing auxiliary loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss

    def _load_balancing_loss(self, router_probs, top_k_indices):
        """
        Compute auxiliary loss to encourage balanced expert usage.
        L_balance = N * sum(f_i * P_i)
        """
        n_tokens = router_probs.size(0)

        # f_i: fraction of tokens assigned to each expert
        expert_counts = torch.zeros(self.n_experts, device=router_probs.device)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                expert_counts[e] += (top_k_indices[:, k] == e).float().sum()
        f = expert_counts / (n_tokens * self.top_k)

        # P_i: average router probability for each expert
        P = router_probs.mean(dim=0)

        return self.n_experts * (f * P).sum()
```

### 20.7 Training with MoE

```python
class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN."""

    def __init__(self, d_model=256, n_heads=8, n_experts=8, top_k=2):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = MoELayer(d_model, d_model * 4, n_experts, top_k)

    def forward(self, x):
        # Pre-LN attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-LN MoE FFN
        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss


def train_moe_model(n_experts=8, top_k=2, n_steps=500):
    """Train a small MoE model and track expert usage."""
    d_model = 256
    block = MoETransformerBlock(d_model, n_experts=n_experts, top_k=top_k)
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)

    losses = []
    aux_losses = []
    expert_usage_history = []

    for step in range(n_steps):
        x = torch.randn(8, 32, d_model)
        target = torch.randn(8, 32, d_model)

        out, aux_loss = block(x)
        task_loss = F.mse_loss(out, target)
        total_loss = task_loss + 0.01 * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(task_loss.item())
        aux_losses.append(aux_loss.item())

        # Track expert usage every 50 steps
        if step % 50 == 0:
            with torch.no_grad():
                probe = torch.randn(16, 64, d_model)
                logits = block.moe.router(probe.view(-1, d_model))
                probs = F.softmax(logits, dim=-1)
                usage = probs.mean(dim=0)
                expert_usage_history.append(usage.numpy())
                print(f"Step {step}: loss={task_loss.item():.4f}, "
                      f"aux={aux_loss.item():.4f}, "
                      f"usage={usage.numpy().round(3)}")

    return losses, aux_losses, expert_usage_history

losses, aux_losses, usage_hist = train_moe_model()
```

### 20.8 Visualize Expert Specialization

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training loss
axes[0].plot(losses, alpha=0.5, label='Task Loss')
axes[0].plot(
    range(0, len(losses), 10),
    [sum(losses[i:i+10])/10 for i in range(0, len(losses), 10)],
    'r-', label='Smoothed'
)
axes[0].set_title('Task Loss')
axes[0].set_xlabel('Step')
axes[0].legend()

# Auxiliary loss
axes[1].plot(aux_losses, alpha=0.5, label='Balance Loss')
axes[1].axhline(y=1.0, color='r', linestyle='--', label='Perfect balance')
axes[1].set_title('Load Balancing Loss')
axes[1].set_xlabel('Step')
axes[1].legend()

# Expert usage over time
usage_array = torch.tensor(usage_hist).numpy()
for e in range(usage_array.shape[1]):
    axes[2].plot(
        range(0, len(losses), 50)[:len(usage_array)],
        usage_array[:, e],
        label=f'Expert {e}'
    )
axes[2].axhline(y=1/8, color='k', linestyle='--', alpha=0.5, label='Ideal (1/8)')
axes[2].set_title('Expert Usage Over Training')
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Average Routing Probability')
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig('moe_training.png', dpi=150)
plt.show()
```

---

## Exercise (45 min)

### E20.1 Mixtral Math

Mixtral 8×7B has:
- 32 transformer layers
- Each layer: shared attention + 8 FFN experts (each ~7B-equivalent FFN)
- Top-2 routing
- Total parameters: 46.7B
- Active parameters per token: 12.9B

**Questions:**
1. What is the theoretical FLOPs reduction vs a dense 46.7B model?
2. What is the memory requirement (fp16) to load the full model?
3. What is the inference speedup vs a dense 46.7B model? vs a dense 12.9B model?
4. A dense 12.9B model needs ~26 GB. Mixtral needs ~93 GB. Is MoE "cheaper"?

### E20.2 Expert Collapse Experiment

Train the MoE model **without** the auxiliary loss ($\alpha = 0$):

```python
# Modify train_moe_model to set aux_weight=0
losses_no_bal, _, usage_no_bal = train_moe_model_custom(aux_weight=0.0)
losses_with_bal, _, usage_with_bal = train_moe_model_custom(aux_weight=0.01)

# Compare expert usage distributions
# Does expert collapse occur? How quickly?
```

### E20.3 Top-k Ablation

Compare top-1 (Switch Transformer style) vs top-2 (Mixtral style) vs top-4 routing:

```python
for k in [1, 2, 4]:
    losses, _, usage = train_moe_model(n_experts=8, top_k=k)
    print(f"top-{k}: final_loss={losses[-1]:.4f}")
    # Plot and compare convergence
```

What happens to quality and compute as $k$ increases? At what point does MoE lose its advantage over a dense model?

---

## Key Takeaways

1. **MoE decouples model capacity from compute cost** — huge total parameters, small active parameters per token
2. **The router is a tiny learned gating network** that selects which experts process each token
3. **Load balancing is critical** — without it, the router collapses to using 1-2 experts and wastes the rest
4. **Mixtral 8×7B matches LLaMA-2 70B** with only 12.9B active parameters — a 5× inference speedup
5. **Memory is the trade-off** — you need ALL parameters in memory even though only a fraction are active

## Connection to the Thread

MoE is the architectural answer to the question: "How do we make models smarter without making them proportionally slower?" You've seen computational optimizations (Flash Attention, KV Cache) and architectural refinements (Pre-LN, SwiGLU). MoE is a fundamentally different approach — instead of making every neuron work on every token, specialize. This idea will recur in Phase VII when you see how VLA models handle multimodal inputs: different "experts" for vision, language, and action.

## Further Reading

- **[Switch Transformer](https://arxiv.org/abs/2101.03961)** — Fedus et al., 2022
- **[Mixtral Paper](https://arxiv.org/abs/2401.04088)** — Jiang et al., 2024
- **[GShard](https://arxiv.org/abs/2006.16668)** — Lepikhin et al., 2020 (MoE at scale)
- **[ST-MoE](https://arxiv.org/abs/2202.08906)** — Zoph et al., 2022 (stable MoE training)
- **[DeepSeek-MoE](https://arxiv.org/abs/2401.06066)** — DeepSeek AI, 2024 (fine-grained experts)
