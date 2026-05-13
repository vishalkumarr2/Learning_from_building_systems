# Day 27: Sampling & Generation

> **Phase II — Attention, Transformers & Scaling | Week 4 | 2.5 hours**
> *"The model learns a distribution. Sampling is the art of drawing from it well."*

## Navigation
- **Previous:** [Day 26: Stop & Reflect #2](day-26-stop-reflect-2.md)
- **Next:** [Day 28: T5 & Encoder-Decoder LMs](day-28-t5-encoder-decoder.md)
- **Week:** [Week 4 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/05-gpt-scaling.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 27.1 The Generation Problem

An autoregressive language model gives you $p(x_t | x_{<t})$ — a probability distribution over the next token at every step. **Generation = repeatedly sampling from this distribution.**

The question is *how* to sample. Different strategies produce dramatically different outputs from the same model.

```
Model output at each step:

  Token   :  "the"  "a"   "an"  "one"  "this"  ...
  p(token):  0.35   0.20  0.15  0.10   0.08   ...
                │
                ├── Greedy: always pick "the" (highest prob)
                ├── Top-k (k=3): sample from {"the", "a", "an"}
                ├── Top-p (p=0.7): sample from {"the", "a", "an"} (cumsum ≥ 0.7)
                └── Temperature: reshape the distribution, then sample
```

### 27.2 Greedy Decoding

The simplest strategy: always pick the highest-probability token.

$$x_t = \arg\max_{v} p(v | x_{<t})$$

**Pros:** Deterministic, fast, never picks a "bad" token.

**Cons:** Produces repetitive, boring, degenerate text. The globally optimal sequence is almost never composed of locally optimal choices. This is the classic greedy algorithm failure.

```
Greedy output:  "The cat sat on the mat. The cat sat on the mat. The cat..."
                ← degenerates into repetition loops
```

### 27.3 Temperature Scaling

Before sampling, divide logits by temperature $T$:

$$p_T(v | x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}$$

where $z_v$ are the raw logits from the model.

| Temperature | Effect | Distribution |
|-------------|--------|-------------|
| $T \to 0$ | Approaches greedy (argmax) | Delta function on top token |
| $T = 1$ | Model's learned distribution | As trained |
| $T > 1$ | Flatter distribution | More random, more creative |
| $T \to \infty$ | Uniform random | Every token equally likely |

**Mathematical proof that $T \to 0$ gives greedy:**

As $T \to 0$, $\exp(z_v / T)$ grows exponentially faster for the largest $z_v$. The softmax concentrates all mass on $\arg\max z_v$.

**Mathematical proof that $T \to \infty$ gives uniform:**

As $T \to \infty$, $z_v / T \to 0$ for all $v$, so $\exp(z_v / T) \to 1$ for all $v$. The softmax becomes $1/|V|$ — uniform.

### 27.4 Top-k Sampling (Fan et al., 2018)

Restrict sampling to the $k$ most probable tokens. Redistribute probability mass among them.

$$p_{\text{top-k}}(v) = \begin{cases} \frac{p(v)}{\sum_{v' \in \text{top-k}} p(v')} & \text{if } v \in \text{top-k} \\ 0 & \text{otherwise} \end{cases}$$

**Problem:** Fixed $k$ is suboptimal. Sometimes the distribution is peaked (only 1-2 good tokens) and sometimes it's flat (many reasonable tokens). A fixed $k$ either cuts good options or keeps bad ones.

### 27.5 Top-p (Nucleus) Sampling (Holtzman et al., 2020)

Dynamically select the smallest set of tokens whose cumulative probability exceeds $p$:

$$\text{top-p}(p) = \min \left\{ k : \sum_{i=1}^{k} p(v_{(i)} | x_{<t}) \geq p \right\}$$

where $v_{(i)}$ is the $i$-th most probable token.

```
Example with p = 0.9:

  Token  :  "the"  "a"   "an"  "one"  "this"  "some"  ...
  Prob   :  0.35   0.20  0.15  0.10   0.08    0.05   ...
  Cumsum :  0.35   0.55  0.70  0.80   0.88    0.93   ← stop here!
                                                ↑
                                        top-p set = first 6 tokens
```

**Why top-p > top-k:** It adapts to the shape of the distribution. For confident predictions, nucleus is small. For uncertain predictions, nucleus is large.

### 27.6 Repetition Penalty

A common pathology: models repeat themselves. Repetition penalty modifies logits for tokens that already appeared:

$$z'_v = \begin{cases} z_v / \alpha & \text{if } v \in \text{generated tokens and } z_v > 0 \\ z_v \cdot \alpha & \text{if } v \in \text{generated tokens and } z_v < 0 \end{cases}$$

where $\alpha > 1$ is the penalty factor (typical: 1.1–1.3).

### 27.7 Structured Generation (Constrained Decoding)

Sometimes you need the output to conform to a specific format — JSON, SQL, code. **Constrained decoding** masks out tokens that would violate the grammar at each step.

```
Grammar-guided decoding for JSON:

  State: expecting key after "{"
  Valid next tokens: '"' (quote to start key)
  Invalid tokens: everything else → mask to -∞
  
  State: expecting value after ":"
  Valid next tokens: '"', digit, '[', '{', 'true', 'false', 'null'
  Invalid tokens: mask to -∞
```

Libraries: `outlines`, `guidance`, `lm-format-enforcer`.

---

## Implementation (60 min)

### Complete Sampling Implementation

```python
import torch
import torch.nn.functional as F
from typing import Optional


def sample_next_token(
    logits: torch.Tensor,           # (vocab_size,)
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[torch.Tensor] = None,
) -> int:
    """Sample the next token using various strategies.
    
    Args:
        logits: Raw logits from the model for the next position
        temperature: Scaling factor (0 = greedy, 1 = normal, >1 = creative)
        top_k: If set, only sample from top-k tokens
        top_p: If set, only sample from nucleus (cumulative prob >= p)
        repetition_penalty: Penalize already-generated tokens (>1.0)
        generated_ids: Previously generated token IDs for rep penalty
    
    Returns:
        Sampled token ID
    """
    # Step 1: Apply repetition penalty
    if repetition_penalty != 1.0 and generated_ids is not None:
        for token_id in set(generated_ids.tolist()):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty
    
    # Step 2: Apply temperature
    if temperature == 0.0:
        # Greedy decoding
        return logits.argmax().item()
    
    logits = logits / temperature
    
    # Step 3: Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = float('-inf')
    
    # Step 4: Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # Step 5: Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,  # (seq_len,)
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Generate text autoregressively."""
    model.eval()
    generated = prompt_ids.clone()
    
    for _ in range(max_new_tokens):
        # Get logits for the last position
        # Crop context if needed (model has max context length)
        context = generated[-model.block_size:] if len(generated) > model.block_size else generated
        logits = model(context.unsqueeze(0))  # (1, seq_len, vocab_size)
        logits = logits[0, -1, :]  # (vocab_size,) — last position
        
        # Sample next token
        next_id = sample_next_token(
            logits.clone(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated,
        )
        
        generated = torch.cat([generated, torch.tensor([next_id])])
    
    return generated
```

### Comparing Sampling Strategies

```python
def compare_sampling_strategies(model, tokenizer, prompt: str):
    """Generate with multiple strategies and compare outputs."""
    prompt_ids = torch.tensor(tokenizer.encode(prompt))
    
    strategies = [
        {"name": "Greedy",         "temperature": 0.0},
        {"name": "T=0.5",         "temperature": 0.5},
        {"name": "T=1.0",         "temperature": 1.0},
        {"name": "T=1.5",         "temperature": 1.5},
        {"name": "Top-k=10",      "temperature": 1.0, "top_k": 10},
        {"name": "Top-k=50",      "temperature": 1.0, "top_k": 50},
        {"name": "Top-p=0.9",     "temperature": 1.0, "top_p": 0.9},
        {"name": "Top-p=0.95",    "temperature": 1.0, "top_p": 0.95},
        {"name": "Rep penalty",   "temperature": 1.0, "top_p": 0.9, 
         "repetition_penalty": 1.2},
    ]
    
    print(f"Prompt: '{prompt}'\n")
    print("=" * 80)
    
    for s in strategies:
        name = s.pop("name")
        ids = generate(model, prompt_ids, max_new_tokens=100, **s)
        text = tokenizer.decode(ids.tolist())
        print(f"\n[{name}]")
        print(text[:300])
        print("-" * 40)
```

### Constrained JSON Decoding

```python
import json
import re


class JSONConstrainedDecoder:
    """Simple grammar-guided decoder for JSON output."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.json_so_far = ""
    
    def get_valid_token_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a mask of valid next tokens given current JSON state."""
        mask = torch.zeros_like(logits, dtype=torch.bool)
        vocab_size = logits.size(-1)
        
        for token_id in range(vocab_size):
            token_str = self.tokenizer.decode([token_id])
            candidate = self.json_so_far + token_str
            
            # Check if this token could lead to valid JSON
            if self._could_be_valid_json_prefix(candidate):
                mask[token_id] = True
        
        return mask
    
    def _could_be_valid_json_prefix(self, s: str) -> bool:
        """Check if string could be a prefix of valid JSON."""
        s = s.strip()
        if not s:
            return True
        
        # Try parsing — if it works, it's valid JSON
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError as e:
            # If error is at the end, it might be a valid prefix
            # (incomplete but could become valid with more tokens)
            if e.pos is not None and e.pos >= len(s) - 1:
                return True
            # Allow if we're still inside a string or number
            if "Unterminated" in str(e) or "Expecting" in str(e):
                return True
            return False
    
    def decode_step(self, logits: torch.Tensor) -> int:
        """Apply JSON constraints and sample."""
        mask = self.get_valid_token_mask(logits)
        
        # Mask invalid tokens
        logits[~mask] = float('-inf')
        
        # Sample from valid tokens only
        token_id = sample_next_token(logits, temperature=0.8)
        
        # Update state
        self.json_so_far += self.tokenizer.decode([token_id])
        return token_id
```

---

## Exercise (45 min)

### E27.1 — Temperature Extremes

Demonstrate empirically:
1. **$T \to 0$:** Generate 5 completions with $T = 0.01$. Show they are all identical (greedy).
2. **$T \to \infty$:** Generate with $T = 100$. Show it's nearly random gibberish.
3. **Find the Goldilocks zone:** Try $T = 0.3, 0.5, 0.7, 0.9, 1.0, 1.2$. Which produces the most coherent yet creative text?

### E27.2 — Top-p vs Top-k Analysis

For the same prompt, generate 10 samples each with:
- Top-k = 5, 10, 20, 50
- Top-p = 0.5, 0.7, 0.9, 0.95

Measure:
1. **Diversity:** How different are the 10 samples from each other?
2. **Quality:** Subjective ranking of coherence
3. **Nucleus size:** For top-p, how many tokens are in the nucleus on average?

### E27.3 — Constrained Decoding Challenge

Make your model output valid JSON for these tasks:
1. Generate a JSON object with keys `"name"`, `"age"`, `"city"`
2. Generate a JSON array of 3 numbers
3. Discuss: What are the limitations of token-level grammar constraints?

---

## Key Takeaways

1. **Greedy decoding degenerates** — locally optimal ≠ globally optimal for sequences
2. **Temperature controls the exploration-exploitation tradeoff** — $T = 0$ exploits, $T \to \infty$ explores
3. **Top-p adapts to distribution shape** — better than fixed top-k for varying confidence levels
4. **Repetition penalty is essential** in practice to prevent degenerate loops
5. **Constrained decoding** enables structured outputs — critical for using LLMs as components in systems
6. **For VLAs:** action generation uses similar strategies — continuous action sampling with temperature-like noise controls exploration vs exploitation in robot behavior

## Connection to the Thread

Sampling strategies connect directly to robot action generation. In diffusion-based VLAs (like π₀), the denoising process is analogous to temperature-controlled sampling — more denoising steps = more "greedy," fewer steps = more exploration. Understanding these tradeoffs in the language domain prepares you for understanding them in action space.

## Further Reading

- Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020) — nucleus sampling paper
- Fan et al., "Hierarchical Neural Story Generation" (2018) — top-k sampling
- Willard & Louf, "Efficient Guided Generation for Large Language Models" (2023) — outlines library
- Meister et al., "Locally Typical Sampling" (2023) — information-theoretic sampling alternative
