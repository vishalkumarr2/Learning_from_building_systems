# Day 39: Long Context & Reasoning

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"Context is all you need — but how much context can you actually use?" — An LLM scaling researcher*

## Navigation
- **Previous:** [Day 38: In-Context Learning](day-38-in-context-learning.md)
- **Next:** [Day 40: RAG & Tool Use](day-40-rag-tool-use.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Engineering](../../study-notes/07-llm-engineering.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 39.1 The Context Length Problem

Transformers have a fundamental limitation: self-attention is $O(n^2)$ in sequence length.

```
Context length evolution:
  GPT-2 (2019):      1,024 tokens
  GPT-3 (2020):      2,048 tokens
  GPT-3.5 (2023):    4,096 → 16,384 tokens
  GPT-4 (2023):      8,192 → 128,000 tokens
  Claude 3 (2024):   200,000 tokens
  Gemini 1.5 (2024): 1,000,000+ tokens
  
Memory for KV cache at 128K context (7B model, FP16):
  Layers: 32, Heads: 32, Head dim: 128
  KV per layer: 2 × 128K × 128 × 2 bytes = 64 MB
  Total: 32 × 64 MB = 2 GB (just for one request's KV cache!)
```

### 39.2 RoPE: Rotary Position Embeddings

Most modern LLMs use **RoPE** (Rotary Position Encoding):

$$
\text{RoPE}(x_m, m) = \begin{pmatrix} x_m^{(1)} \cos(m\theta_1) - x_m^{(2)} \sin(m\theta_1) \\ x_m^{(1)} \sin(m\theta_1) + x_m^{(2)} \cos(m\theta_1) \\ \vdots \end{pmatrix}
$$

where $\theta_i = 10000^{-2i/d}$ and $m$ is the position index.

**Key property:** The attention score between positions $m$ and $n$ depends only on $m - n$ (relative position), but is computed by rotating the query and key vectors.

### 39.3 Context Extension: YaRN & Friends

To extend a model trained on 4K context to 128K+ without retraining:

**Position interpolation (simple):**

$$
\theta'_i = \theta_i / s, \quad s = \frac{L_{\text{target}}}{L_{\text{trained}}}
$$

Problem: compresses all frequencies equally, losing fine-grained position information.

**YaRN (Yet another RoPE extensioN):**

$$
\theta'_i = \begin{cases}
\theta_i & \text{if } \lambda_i < 1 \quad \text{(high frequency: keep)} \\
\theta_i / s & \text{if } \lambda_i > 1 \quad \text{(low frequency: interpolate)} \\
(1-\gamma)\frac{\theta_i}{s} + \gamma\theta_i & \text{otherwise (smooth blend)}
\end{cases}
$$

```
Intuition:
  High-frequency RoPE dimensions → encode nearby positions (keep intact)
  Low-frequency RoPE dimensions  → encode distant positions (interpolate)
  
  Like a rubber band: stretch only the low-frequency part
  → preserves local attention while extending global reach
```

### 39.4 Ring Attention & Distributed Context

For very long contexts (1M+ tokens), **Ring Attention** distributes the sequence across multiple devices:

```
Standard attention:         Ring attention:
┌─────────────────┐        Device 1    Device 2    Device 3
│ Full Q·K^T·V    │        ┌──────┐    ┌──────┐    ┌──────┐
│ (all in one GPU)│        │ Q₁K₁ │→→→│ Q₁K₂ │→→→│ Q₁K₃ │
│                 │        │ V₁   │    │ V₂   │    │ V₃   │
│ Memory: O(n²)   │        └──┬───┘    └──┬───┘    └──┬───┘
└─────────────────┘           │           │           │
                              ←←← Ring communication ←←←
                              
Each device holds 1/N of the sequence
K,V blocks rotate around the ring
Each device computes partial attention
→ Memory per device: O(n²/N)
→ Enables 1M+ context on multi-GPU
```

### 39.5 Chain-of-Thought & Reasoning

**Chain-of-Thought (CoT)** prompting unlocks multi-step reasoning:

```
Standard prompting:
  Q: "Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls?"
  A: "11"  ← Often wrong without reasoning

Chain-of-thought:
  Q: "Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls?"
  A: "Roger starts with 5 balls. He buys 2 cans × 3 balls = 6 balls.
      Total: 5 + 6 = 11 balls."  ← Correct with reasoning trace
```

### 39.6 o1-Style Reasoning: Test-Time Compute

OpenAI's o1 introduced **test-time compute scaling**: instead of making the model bigger, give it more time to think.

```
Traditional scaling:                  Test-time compute scaling:
  More parameters → better            More thinking tokens → better
  Fixed compute per token              Variable compute per problem
  
  Cost: training time                  Cost: inference time
  Limit: GPU memory                    Limit: context length + latency
```

**Key mechanisms:**

1. **Thinking tokens:** Model generates hidden reasoning before answering
2. **Verification:** Model checks its own work ("Wait, let me verify...")
3. **Backtracking:** Model reconsiders approaches ("Actually, a better approach is...")
4. **Planning:** Model decomposes complex problems before solving

$$
\text{Performance} \propto \log(\text{test-time compute}) \quad \text{(on reasoning tasks)}
$$

```
Standard model:     Prompt → Answer (fast, sometimes wrong)
o1-style model:     Prompt → [Think][Think][Think]... → Answer (slow, more accurate)

Trade-off:
  Easy questions: standard model is efficient (thinking wastes tokens)
  Hard questions: o1-style is better (thinking prevents errors)
  
→ Optimal strategy: route easy questions to fast model,
  hard questions to reasoning model
```

---

## Implementation (60 min)

### RoPE Extension & Reasoning Experiments

```python
"""
Day 39 Implementation: RoPE scaling and chain-of-thought reasoning.
"""
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Part 1: RoPE Mathematics
# ============================================================

def compute_rope_frequencies(
    dim: int, base: float = 10000.0, seq_len: int = 4096,
) -> torch.Tensor:
    """Compute RoPE rotation frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    # Outer product: [seq_len, dim/2]
    angles = torch.outer(positions, freqs)
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor x of shape [seq_len, dim]."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def position_interpolation(
    dim: int, base: float, trained_len: int, target_len: int,
) -> torch.Tensor:
    """Simple position interpolation for RoPE extension."""
    scale = target_len / trained_len
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    freqs_scaled = freqs / scale  # interpolate all frequencies
    positions = torch.arange(target_len).float()
    return torch.outer(positions, freqs_scaled)


def yarn_frequencies(
    dim: int,
    base: float,
    trained_len: int,
    target_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN frequency scaling for RoPE extension."""
    scale = target_len / trained_len
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Compute wavelengths
    wavelengths = 2 * math.pi / freqs

    # Ramp function: smooth interpolation between keeping and scaling
    low = max(math.floor(beta_fast * dim / (2 * math.pi * (trained_len - 1))), 1)
    high = min(math.ceil(beta_slow * dim / (2 * math.pi * (trained_len - 1))),
               dim // 2 - 1)

    yarn_freqs = torch.zeros_like(freqs)
    for i in range(len(freqs)):
        if i < low:
            # High frequency: keep original
            yarn_freqs[i] = freqs[i]
        elif i > high:
            # Low frequency: interpolate
            yarn_freqs[i] = freqs[i] / scale
        else:
            # Blend
            gamma = (i - low) / (high - low)
            yarn_freqs[i] = (1 - gamma) * freqs[i] + gamma * (freqs[i] / scale)

    positions = torch.arange(target_len).float()
    return torch.outer(positions, yarn_freqs)


# ============================================================
# Part 2: Attention Distance Analysis
# ============================================================

def analyze_attention_decay(dim: int = 64, max_dist: int = 1000):
    """Show how RoPE makes attention decay with distance."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))

    # Create query and key at position 0
    q = torch.randn(dim)
    k = torch.randn(dim)

    distances = list(range(0, max_dist, 10))
    dot_products = []

    for dist in distances:
        angles_q = torch.zeros(dim // 2)
        angles_k = torch.tensor([dist]).float() * freqs

        q_rot = apply_rope(q.unsqueeze(0), angles_q.unsqueeze(0)).squeeze()
        k_rot = apply_rope(k.unsqueeze(0), angles_k.unsqueeze(0)).squeeze()

        dot = (q_rot * k_rot).sum().item()
        dot_products.append(dot)

    return distances, dot_products


# ============================================================
# Part 3: Chain-of-Thought Prompting
# ============================================================

def create_cot_prompt(question: str, examples: list[dict]) -> str:
    """Create a chain-of-thought prompt."""
    prompt = "Solve each problem step by step.\n\n"
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: Let's think step by step.\n{ex['reasoning']}\n"
        prompt += f"The answer is {ex['answer']}.\n\n"
    prompt += f"Q: {question}\nA: Let's think step by step.\n"
    return prompt


COT_EXAMPLES = [
    {
        "question": "A robot picks 3 items per minute. It works for 5 minutes, "
                    "rests for 2, then works for 3 more. How many items?",
        "reasoning": "First work period: 3 items/min × 5 min = 15 items.\n"
                     "Rest period: 0 items.\n"
                     "Second work period: 3 items/min × 3 min = 9 items.\n"
                     "Total: 15 + 9 = 24 items.",
        "answer": "24 items",
    },
    {
        "question": "A warehouse has 4 aisles. Each aisle has 10 shelves. "
                    "Each shelf holds 6 bins. How many bins total?",
        "reasoning": "Bins per aisle: 10 shelves × 6 bins = 60 bins.\n"
                     "Total bins: 4 aisles × 60 bins = 240 bins.",
        "answer": "240 bins",
    },
]


if __name__ == "__main__":
    # RoPE visualization
    print("=" * 60)
    print("RoPE Frequency Analysis")
    print("=" * 60)

    dim = 64
    angles_original = compute_rope_frequencies(dim, seq_len=4096)
    angles_pi = position_interpolation(dim, 10000.0, 4096, 32768)
    angles_yarn = yarn_frequencies(dim, 10000.0, 4096, 32768)

    print(f"Original RoPE: {angles_original.shape} (4K context)")
    print(f"PI-extended:   {angles_pi.shape} (32K context)")
    print(f"YaRN-extended: {angles_yarn.shape} (32K context)")

    # Frequency comparison
    freqs_orig = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    print(f"\nHighest freq:  {freqs_orig[0]:.6f} (wavelength: {2*math.pi/freqs_orig[0]:.0f} positions)")
    print(f"Lowest freq:   {freqs_orig[-1]:.6f} (wavelength: {2*math.pi/freqs_orig[-1]:.0f} positions)")

    # Chain-of-thought demo
    print("\n" + "=" * 60)
    print("Chain-of-Thought Prompt")
    print("=" * 60)

    test_question = ("A robot fleet has 8 robots. Each charges for 20 minutes "
                     "every 3 hours. In a 12-hour shift, how much total "
                     "charging time across all robots?")
    cot = create_cot_prompt(test_question, COT_EXAMPLES)
    print(cot[:500])
    print(f"...\n[Prompt length: {len(cot)} chars]")
```

---

## Exercise (45 min)

### E39.1 — RoPE Extension Comparison (25 min)
1. Implement NTK-aware scaling (modify the RoPE base): $\text{base}' = \text{base} \cdot s^{d/(d-2)}$
2. Compare three methods (PI, YaRN, NTK) on a perplexity test: generate text with position indices beyond the training length
3. Which method degrades least at 4× the training context?

### E39.2 — Test-Time Compute Scaling (20 min)
Simulate o1-style reasoning:
1. For a set of math problems, generate answers with different "thinking budgets" (50, 100, 200, 500 tokens of reasoning)
2. Plot accuracy vs. thinking tokens
3. Identify the point of diminishing returns — when does more thinking stop helping?

---

## Key Takeaways

1. **RoPE** encodes position through rotation — relative position naturally emerges from the math
2. **YaRN** extends context by scaling low frequencies while preserving high frequencies
3. **Ring attention** distributes long contexts across devices for million-token sequences
4. **Chain-of-thought** unlocks reasoning by generating intermediate steps
5. **Test-time compute** (o1) trades inference time for accuracy — a new scaling dimension

---

## Connection to the Thread

Long context is critical for robotics: a robot needs to remember its entire mission (thousands of time steps) to make coherent decisions. Ring attention techniques map to distributed robot compute, and chain-of-thought reasoning is how robots will plan complex multi-step warehouse operations — decomposing "sort all packages by destination" into individual pick-place steps.

---

## Further Reading

- [RoFormer: Enhanced Transformer with RoPE](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) — Peng et al., 2023
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) — Snell et al., 2024
