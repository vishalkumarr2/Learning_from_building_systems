# Exercise 04 — nanoGPT Ablations & Scaling Experiments

> **Phase II · Days 23–28 · ~15 hours hands-on**
> Companion to: [05-gpt-scaling.md](../study-notes/05-gpt-scaling.md)
> Deliverables: Ablation table, scaling curve plot, BPE tokenizer, sampling comparison

---

## Prerequisites & Setup

### Hardware Requirements
- GPU with ≥4GB VRAM (training takes ~5 min/run on a single GPU)
- CPU-only works but expect ~30 min/run
- All experiments designed to fit on a single consumer GPU

### Environment Setup

```bash
# Clone nanoGPT
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib

# Prepare Shakespeare dataset
python data/shakespeare_char/prepare.py

# Verify setup — quick smoke test
python train.py config/train_shakespeare_char.py \
    --max_iters=100 \
    --eval_interval=50 \
    --log_interval=10
```

### Baseline Config

Create `config/ablation_baseline.py`:
```python
# Ablation baseline — all experiments start from here
# Change EXACTLY ONE parameter per experiment

# Model
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2
bias = False

# Training
batch_size = 64
learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 100
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Evaluation
eval_interval = 250
eval_iters = 200

# Logging
log_interval = 10
wandb_log = False  # Set True if using W&B

# Data
dataset = 'shakespeare_char'

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = True  # PyTorch 2.0+ required; set False on older versions
```

### Experiment Tracking

Create `experiments/run_ablation.sh`:
```bash
#!/bin/bash
# Usage: ./run_ablation.sh <experiment_name> <override_args>
# Example: ./run_ablation.sh depth_2 --n_layer=2

EXP_NAME="$1"
shift

mkdir -p "experiments/results/$EXP_NAME"

python train.py config/ablation_baseline.py \
    "$@" \
    --out_dir="experiments/results/$EXP_NAME" \
    2>&1 | tee "experiments/results/$EXP_NAME/train.log"

echo "Done: $EXP_NAME"
echo "Results saved to experiments/results/$EXP_NAME/"
```

---

## Exercise 1: nanoGPT Code Walkthrough

**Time**: ~3 hours
**Goal**: Read and annotate every architectural decision in nanoGPT

### Task 1.1: Annotate `model.py`

Open `model.py` and add comments explaining every line. Focus on:

```
Checklist — answer in your own words:
□ What is LayerNorm doing before each sub-layer? (Pre-LN vs Post-LN)
□ Why is the causal mask registered as a buffer, not a parameter?
□ How does the attention computation enforce causality?
□ What does weight tying between wte and lm_head accomplish?
□ Why is the MLP hidden dimension 4× the model dimension?
□ Where exactly is dropout applied and why?
□ What is the purpose of the residual connections?
□ How does the model handle variable-length sequences?
```

### Task 1.2: Annotate `train.py`

```
Checklist — answer in your own words:
□ How is the learning rate schedule implemented? (warmup + cosine decay)
□ What is gradient accumulation and when is it useful?
□ How does the evaluation loop differ from training?
□ What does `torch.compile` do and why does it help?
□ How are checkpoints saved and loaded?
□ What is the purpose of `ctx` (autocast context)?
□ How does mixed precision training (fp16/bf16) work here?
```

### Task 1.3: Config Parameter Documentation

Create a table documenting every config parameter:

```markdown
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| n_layer   | 6       | 2-96  | Number of transformer blocks. More = deeper representations. |
| n_head    | 6       | 1-64  | Number of attention heads. More = more diverse attention patterns. |
| n_embd    | 384     | 64-12288 | Model dimension. Wider = more capacity per layer. |
| ...       | ...     | ...   | ... |
```

Fill in ALL parameters from `model.py` and `train.py`.

### Task 1.4: Draw the Architecture

Hand-draw (or ASCII-draw) the full forward pass:

```
Input IDs → Token Embedding + Position Embedding → Dropout
    → [Block × n_layer]
        → LayerNorm → CausalSelfAttention → Residual
        → LayerNorm → MLP → Residual
    → Final LayerNorm → LM Head → Logits
    → Cross-Entropy Loss (if targets provided)
```

Label shapes at each point: (B, T) → (B, T, C) → ... → (B, T, V)

---

## Exercise 2: Systematic Ablation Experiments

**Time**: ~4 hours (mostly training time)
**Goal**: Run controlled experiments varying one architectural choice at a time

### Task 2.1: Depth Ablation

```bash
# Run each with ONLY n_layer changed from baseline
./run_ablation.sh depth_02 --n_layer=2
./run_ablation.sh depth_04 --n_layer=4
./run_ablation.sh depth_06 --n_layer=6   # baseline
./run_ablation.sh depth_08 --n_layer=8
./run_ablation.sh depth_12 --n_layer=12
```

Record in your ablation table:

```
┌───────────┬────────────────┬──────────────┬──────────────┐
│ n_layers  │ Total Params   │ Train Loss   │ Val Loss     │
├───────────┼────────────────┼──────────────┼──────────────┤
│     2     │                │              │              │
│     4     │                │              │              │
│     6     │                │              │              │
│     8     │                │              │              │
│    12     │                │              │              │
└───────────┴────────────────┴──────────────┴──────────────┘
```

**Questions to answer:**
- How does loss scale with depth?
- Is there a point of diminishing returns?
- Does the train/val gap change with depth? (overfitting indicator)

### Task 2.2: Head Count Ablation

```bash
# n_embd must be divisible by n_head
./run_ablation.sh heads_01 --n_head=1
./run_ablation.sh heads_02 --n_head=2
./run_ablation.sh heads_04 --n_head=4   # head_dim = 96
./run_ablation.sh heads_06 --n_head=6   # baseline, head_dim = 64
./run_ablation.sh heads_08 --n_head=8   # head_dim = 48

# Note: n_embd=384, so valid n_head values: 1, 2, 3, 4, 6, 8, 12, 16, ...
```

Record results and answer:
- Does 1 head vs 6 heads make a big difference?
- What's the sweet spot for head dimension?
- Does more heads mean more attention patterns? (Check by visualizing attention)

### Task 2.3: Width Ablation

```bash
# Adjust n_head to be compatible with n_embd
./run_ablation.sh width_064 --n_embd=64  --n_head=2
./run_ablation.sh width_128 --n_embd=128 --n_head=4
./run_ablation.sh width_256 --n_embd=256 --n_head=4
./run_ablation.sh width_384 --n_embd=384 --n_head=6   # baseline
./run_ablation.sh width_512 --n_embd=512 --n_head=8
```

**Important**: This is your most informative ablation for scaling curves. Width changes affect parameter count significantly.

### Task 2.4: Activation Function Ablation

You need to modify `model.py` for this. Add a config option:

```python
# In CausalSelfAttention or MLP class:
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, 'activation', 'gelu') == 'relu':
            self.act = nn.ReLU()
        elif config.activation == 'gelu':
            self.act = nn.GELU()
        elif config.activation == 'swiglu':
            # SwiGLU: gate linear unit
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
            self._use_swiglu = True
            return
        # ... standard init for non-SwiGLU
```

```bash
./run_ablation.sh act_relu   --activation=relu
./run_ablation.sh act_gelu   --activation=gelu    # baseline
./run_ablation.sh act_swiglu --activation=swiglu
```

⚠️ **Note**: SwiGLU has more parameters due to the extra gate projection. For a fair comparison, you'd need to adjust d_model to match total params. For this exercise, just document the difference.

### Task 2.5: Normalization Ablation

Modify `model.py` to support different normalization:

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

```bash
./run_ablation.sh norm_preln   --norm_type=pre_ln     # baseline
./run_ablation.sh norm_postln  --norm_type=post_ln
./run_ablation.sh norm_rmsnorm --norm_type=rmsnorm
```

### Task 2.6: Compile Results

Create a comprehensive ablation summary:

```markdown
# Ablation Summary

## Overall Rankings (by val loss improvement over baseline)
1. Width 512 (n_embd=512): Δ = -X.XX
2. SwiGLU activation: Δ = -X.XX
3. ...

## Key Findings
- Depth vs Width: Width matters more per-parameter
- Activation: SwiGLU > GELU > ReLU
- Normalization: RMSNorm ≈ Pre-LN > Post-LN
- Heads: Diminishing returns beyond head_dim ≈ 64
```

---

## Exercise 3: BPE Tokenizer Implementation

**Time**: ~3 hours
**Goal**: Implement Byte Pair Encoding from scratch

### Task 3.1: Implement Basic BPE

```python
"""
bpe.py — Byte Pair Encoding from scratch
Following Karpathy's minbpe approach
"""

class BasicBPE:
    """Minimal BPE tokenizer."""

    def __init__(self):
        self.merges = {}      # (pair) -> new_token_id
        self.vocab = {}       # token_id -> bytes

    def _get_pair_counts(self, ids):
        """Count consecutive pairs in a list of token ids."""
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, new_id):
        """Replace all occurrences of pair in ids with new_id."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size):
        """Train BPE on text to reach target vocab_size."""
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte-level)"
        num_merges = vocab_size - 256

        # Start with raw bytes
        ids = list(text.encode('utf-8'))

        # Initialize vocab with single bytes
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            # YOUR CODE: Find most common pair
            counts = self._get_pair_counts(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)

            # YOUR CODE: Create new token and merge
            new_id = 256 + i
            ids = self._merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: "
                      f"{self.vocab[best_pair[0]]} + {self.vocab[best_pair[1]]} "
                      f"-> {self.vocab[new_id]} (count: {counts[best_pair]})")

        print(f"Trained BPE: {len(self.vocab)} tokens, "
              f"compression ratio: {len(text.encode('utf-8'))/len(ids):.2f}x")

    def encode(self, text):
        """Encode text to token ids."""
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            counts = self._get_pair_counts(ids)
            # Find the pair with the lowest merge index (earliest merge)
            pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """Decode token ids back to text."""
        tokens = b''.join(self.vocab[i] for i in ids)
        return tokens.decode('utf-8', errors='replace')
```

### Task 3.2: Train and Inspect

```python
# Train on a text corpus
with open('data/shakespeare_char/input.txt', 'r') as f:
    text = f.read()

tokenizer = BasicBPE()
tokenizer.train(text, vocab_size=512)

# Inspect vocabulary
print("\n--- Vocabulary Sample ---")
for token_id in range(256, min(300, len(tokenizer.vocab))):
    print(f"  {token_id}: {tokenizer.vocab[token_id]}")

# Test encode/decode roundtrip
test_text = "To be, or not to be, that is the question."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"\nOriginal:  {test_text}")
print(f"Encoded:   {encoded}")
print(f"Decoded:   {decoded}")
print(f"Tokens:    {len(encoded)} (from {len(test_text)} chars)")
assert decoded == test_text, "Roundtrip failed!"
```

**Questions to answer:**
- What are the first merges? (Should be common bigrams like `th`, `he`, `in`)
- What's the compression ratio?
- What happens when you encode text NOT in the training corpus?

### Task 3.3: Compare with tiktoken

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")

test_texts = [
    "To be, or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "こんにちは世界",  # Japanese
]

for text in test_texts:
    my_tokens = tokenizer.encode(text)
    gpt2_tokens = enc.encode(text)
    print(f"\nText: {text[:50]}...")
    print(f"  My BPE:  {len(my_tokens)} tokens")
    print(f"  GPT-2:   {len(gpt2_tokens)} tokens")
    print(f"  Ratio:   {len(my_tokens)/len(gpt2_tokens):.2f}x")
```

**Questions:**
- Why does GPT-2's tokenizer produce fewer tokens?
- What happens with non-English text?
- How does vocabulary size affect compression?

---

## Exercise 4: Scaling Curve Plotting

**Time**: ~2 hours
**Goal**: Create publication-quality scaling curves from your ablation data

### Task 4.1: Collect All Results

```python
"""
plot_scaling.py — Create scaling curves from ablation experiments
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Collect results from all experiments
# Format: (total_params, best_val_loss)
# Fill these in from your actual ablation runs!

width_results = [
    # (params, val_loss) — from width ablation
    # (XXX, X.XX),  # d_model=64
    # (XXX, X.XX),  # d_model=128
    # (XXX, X.XX),  # d_model=256
    # (XXX, X.XX),  # d_model=384
    # (XXX, X.XX),  # d_model=512
]

depth_results = [
    # (params, val_loss) — from depth ablation
]

# TODO: Fill in your actual numbers!
```

### Task 4.2: Fit Power Law

```python
def fit_power_law(params, losses):
    """Fit L = a * N^(-alpha) to data.

    Returns:
        alpha: scaling exponent
        a: constant
        r_squared: goodness of fit
    """
    log_N = np.log(np.array(params))
    log_L = np.log(np.array(losses))

    # Linear regression in log space
    coeffs = np.polyfit(log_N, log_L, 1)
    alpha = -coeffs[0]
    a = np.exp(coeffs[1])

    # R² calculation
    predicted = coeffs[0] * log_N + coeffs[1]
    ss_res = np.sum((log_L - predicted) ** 2)
    ss_tot = np.sum((log_L - np.mean(log_L)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return alpha, a, r_squared


# Fit and report
alpha, a, r2 = fit_power_law(
    [p for p, _ in width_results],
    [l for _, l in width_results]
)
print(f"Scaling exponent: α = {alpha:.4f}")
print(f"Constant: a = {a:.4f}")
print(f"R² = {r2:.4f}")
print(f"Power law: L(N) = {a:.4f} · N^(-{alpha:.4f})")
print(f"\nPublished Kaplan α_N ≈ 0.076")
print(f"Your α_N = {alpha:.4f}")
print(f"Difference: {abs(alpha - 0.076):.4f}")
```

### Task 4.3: Create Publication-Quality Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Loss vs Parameters (scaling curve) ---
ax = axes[0]
params_w = [p for p, _ in width_results]
losses_w = [l for _, l in width_results]

ax.scatter(params_w, losses_w, s=80, c='blue', zorder=5, label='Width ablation')

# Plot power law fit
N_fit = np.logspace(np.log10(min(params_w)*0.5), np.log10(max(params_w)*2), 100)
ax.plot(N_fit, a * N_fit**(-alpha), 'r--',
        label=f'$L = {a:.2f} \\cdot N^{{-{alpha:.3f}}}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (N)', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Scaling Curve: Loss vs Parameters', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Plot 2: Ablation comparison bar chart ---
ax = axes[1]
# TODO: Create grouped bar chart comparing all ablations
# Categories: depth, width, activation, normalization, heads
# Y-axis: validation loss
# Color-coded by category

# --- Plot 3: Training curves overlay ---
ax = axes[2]
# TODO: Plot training loss curves for all width experiments
# X-axis: iteration
# Y-axis: training loss
# One line per d_model value

plt.tight_layout()
plt.savefig('experiments/scaling_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved to experiments/scaling_curves.png")
```

### Task 4.4: Compare with Published Results

```python
# Published Kaplan scaling law values
kaplan_alpha_N = 0.076   # parameter scaling exponent
kaplan_alpha_D = 0.095   # data scaling exponent

# Your results
your_alpha_N = alpha  # from your width ablation fit

print("=" * 50)
print("COMPARISON WITH PUBLISHED SCALING LAWS")
print("=" * 50)
print(f"{'Metric':<25} {'Kaplan':<10} {'Yours':<10} {'Match?':<10}")
print("-" * 50)
print(f"{'α_N (param scaling)':<25} {kaplan_alpha_N:<10.4f} {your_alpha_N:<10.4f} "
      f"{'✓' if abs(your_alpha_N - kaplan_alpha_N) < 0.03 else '✗'}")

# Discussion: Why might your exponent differ?
# 1. Small model regime (nanoGPT) vs large model regime (Kaplan)
# 2. Character-level vs subword tokenization
# 3. Shakespeare vs WebText dataset
# 4. Short training (5K iters) vs converged training
```

---

## Exercise 5: Sampling Strategies

**Time**: ~3 hours
**Goal**: Implement and compare all major sampling strategies

### Task 5.1: Implement Sampling Functions

```python
"""
sampling.py — All major sampling strategies from scratch
"""
import torch
import torch.nn.functional as F


def greedy_decode(logits):
    """Always pick the highest probability token."""
    return logits.argmax(dim=-1, keepdim=True)


def sample_with_temperature(logits, temperature=1.0):
    """Scale logits by temperature before sampling."""
    assert temperature > 0, "Temperature must be positive"
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_k_sample(logits, k=50, temperature=1.0):
    """Sample from the top-k most likely tokens."""
    logits = logits / temperature
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1:]
    logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_p_sample(logits, p=0.9, temperature=1.0):
    """Nucleus sampling: sample from smallest set with cumulative prob >= p."""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Create mask for tokens to remove (cumulative prob > p)
    remove_mask = cumulative_probs > p
    # Keep at least one token
    remove_mask[..., 0] = False
    # Shift right so the token that pushes past p is kept
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_logits[remove_mask] = float('-inf')
    # Unsort
    logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_with_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Reduce probability of already-generated tokens."""
    logits = logits.clone()
    if generated_ids is not None and len(generated_ids) > 0:
        unique_ids = generated_ids.unique()
        for token_id in unique_ids:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
    return logits
```

### Task 5.2: Generate and Compare

```python
def generate(model, prompt, max_tokens=200, strategy='greedy', **kwargs):
    """Generate text using specified sampling strategy."""
    model.eval()
    # Encode prompt (character-level for Shakespeare)
    idx = torch.tensor([encode(prompt)], device=device)

    generated_ids = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits for last position
            logits, _ = model(idx[:, -block_size:])
            logits = logits[:, -1, :]

            # Apply repetition penalty
            if kwargs.get('repetition_penalty', 1.0) > 1.0:
                logits = sample_with_repetition_penalty(
                    logits, torch.tensor(generated_ids),
                    kwargs['repetition_penalty']
                )

            # Apply sampling strategy
            if strategy == 'greedy':
                next_id = greedy_decode(logits)
            elif strategy == 'temperature':
                next_id = sample_with_temperature(logits, kwargs.get('temperature', 1.0))
            elif strategy == 'top_k':
                next_id = top_k_sample(logits, kwargs.get('k', 50), kwargs.get('temperature', 1.0))
            elif strategy == 'top_p':
                next_id = top_p_sample(logits, kwargs.get('p', 0.9), kwargs.get('temperature', 1.0))

            idx = torch.cat([idx, next_id], dim=-1)
            generated_ids.append(next_id.item())

    return decode(idx[0].tolist())


# Compare strategies with the same prompt
prompt = "ROMEO:\nO, she doth teach the torches"

strategies = [
    ('greedy', {}),
    ('temperature', {'temperature': 0.5}),
    ('temperature', {'temperature': 1.0}),
    ('temperature', {'temperature': 1.5}),
    ('top_k', {'k': 10, 'temperature': 0.8}),
    ('top_k', {'k': 50, 'temperature': 0.8}),
    ('top_p', {'p': 0.9, 'temperature': 0.8}),
    ('top_p', {'p': 0.95, 'temperature': 1.0}),
    ('top_p', {'p': 0.9, 'temperature': 0.8, 'repetition_penalty': 1.2}),
]

print("=" * 70)
print(f"PROMPT: {prompt}")
print("=" * 70)
for strategy, kwargs in strategies:
    label = f"{strategy}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
    output = generate(model, prompt, max_tokens=150, strategy=strategy, **kwargs)
    print(f"\n--- {label} ---")
    print(output[len(prompt):])
    print()
```

### Task 5.3: Implement Constrained JSON Decoding

```python
"""
Constrained decoding: force model to output valid JSON.
This is a simplified version — real implementations use context-free grammars.
"""
import json
import re


class SimpleJSONConstrainer:
    """Force token-by-token generation to produce valid JSON."""

    VALID_AFTER = {
        'START':    ['{'],
        '{':        ['"', '}'],
        'KEY':      [':'],
        ':':        ['"', '-', '0', '1', '2', '3', '4', '5',
                     '6', '7', '8', '9', 't', 'f', 'n', '{', '['],
        'VALUE':    [',', '}', ']'],
        ',':        ['"', '{', '[', '-', '0', '1', '2', '3',
                     '4', '5', '6', '7', '8', '9', 't', 'f', 'n'],
    }

    def __init__(self):
        self.state = 'START'
        self.depth = 0

    def get_allowed_chars(self):
        """Return characters allowed at current position."""
        return self.VALID_AFTER.get(self.state, [])

    def update_state(self, char):
        """Update FSM state based on generated character."""
        if char == '{':
            self.state = '{'
            self.depth += 1
        elif char == '}':
            self.depth -= 1
            self.state = 'VALUE' if self.depth > 0 else 'END'
        elif char == ':':
            self.state = ':'
        elif char == ',':
            self.state = ','
        elif char == '"':
            # Toggle between key/value contexts
            if self.state in ['{', ',']:
                self.state = 'KEY'
            else:
                self.state = 'VALUE'
        # ... extend for full JSON grammar


# Use this with your generate() function:
# At each step, mask logits for characters not in get_allowed_chars()
```

**Challenge**: Extend the constrainer to handle:
- Arrays (`[]`)
- Nested objects
- String values with escapes
- Numbers (integers and floats)
- Boolean values (`true`, `false`)
- Null values

---

## Self-Check Questions

Before moving on, verify you can answer:

```
□ Can you explain every line in nanoGPT's model.py?
□ Did you run all 5 ablation categories?
□ Can you describe the impact of each architectural choice quantitatively?
□ Did your scaling curve fit a power law? What was your α?
□ Can you implement BPE from scratch without looking at the code?
□ Can you explain the difference between top-k and top-p sampling?
□ Can you implement constrained decoding for JSON output?
□ Can you compute FLOPs for a transformer forward pass?
□ Can you estimate training cost in GPU-hours for a given model size?
```

---

## Exercise 6: FLOP Counting and Compute Budgets

**Goal**: Develop the skill of estimating computational cost for any model.
Essential for deciding if a VLA is feasible for real-time robot control.

### 6.1 — Theory: Counting FLOPs in Transformers

```python
"""
FLOP counting for transformer models.

Key formula (forward pass, per token):
  FLOPs_per_token ≈ 2 * N  (where N = total parameters, ignoring embeddings)

Why 2×? Each parameter participates in one multiply AND one add.

Breakdown per layer:
  Attention:
    - QKV projection: 3 * (2 * d_model * d_model) = 6 * d_model²
    - Attention scores: 2 * seq_len * d_model  (per token)
    - Attention × V:   2 * seq_len * d_model  (per token)
    - Output proj:     2 * d_model * d_model = 2 * d_model²
    Total attention = 8 * d_model² + 4 * seq_len * d_model
    
  FFN (with 4× expansion):
    - Up projection:   2 * d_model * 4*d_model = 8 * d_model²
    - Down projection: 2 * 4*d_model * d_model = 8 * d_model²
    Total FFN = 16 * d_model²
    
  Per layer total = 24 * d_model² + 4 * seq_len * d_model
  
  Full model (forward) = n_layers * (24 * d_model² + 4 * seq_len * d_model)
  
Training = 3× forward (forward + backward uses ~2× forward FLOPs)
"""


def count_transformer_flops(
    n_layers: int,
    d_model: int,
    n_heads: int,
    seq_len: int,
    vocab_size: int,
    batch_size: int = 1,
    include_backward: bool = True,
) -> dict:
    """Count FLOPs for a transformer model.
    
    Returns dict with breakdown and total.
    """
    d_k = d_model // n_heads
    
    # Per-layer FLOPs (per token)
    # QKV projections: 3 matmuls of (d_model → d_model)
    qkv_flops = 3 * 2 * d_model * d_model
    
    # Attention score computation: Q @ K^T → (seq_len, d_k) per head × n_heads
    attn_score_flops = 2 * seq_len * d_model  # across all heads
    
    # Attention @ V
    attn_v_flops = 2 * seq_len * d_model
    
    # Output projection
    out_proj_flops = 2 * d_model * d_model
    
    # FFN (SwiGLU or standard 4x)
    ffn_flops = 2 * (2 * d_model * 4 * d_model)  # up + down
    
    per_layer_per_token = qkv_flops + attn_score_flops + attn_v_flops + out_proj_flops + ffn_flops
    
    # Embedding lookup is essentially free (table lookup, not matmul)
    # But output logit projection: (d_model → vocab_size)
    logit_flops = 2 * d_model * vocab_size
    
    # Totals
    forward_per_token = n_layers * per_layer_per_token + logit_flops
    forward_total = forward_per_token * seq_len * batch_size
    
    multiplier = 3.0 if include_backward else 1.0
    total = forward_total * multiplier
    
    return {
        "per_layer_per_token": per_layer_per_token,
        "forward_per_token": forward_per_token,
        "forward_total": forward_total,
        "total_with_backward": total if include_backward else None,
        "total_flops": total,
        "tflops": total / 1e12,
    }


def estimate_training_cost(
    model_params: int,
    n_tokens: int,
    gpu_tflops: float = 312.0,  # A100 theoretical peak
    utilization: float = 0.4,    # Realistic MFU
) -> dict:
    """Estimate training cost using the Chinchilla scaling approximation.
    
    Rule of thumb: Total training FLOPs ≈ 6 * N * D
    where N = params, D = tokens
    
    Returns GPU-hours and cost estimates.
    """
    total_flops = 6 * model_params * n_tokens
    
    effective_tflops = gpu_tflops * utilization
    seconds = total_flops / (effective_tflops * 1e12)
    gpu_hours = seconds / 3600
    
    return {
        "total_flops": total_flops,
        "total_pflops": total_flops / 1e15,
        "gpu_seconds": seconds,
        "gpu_hours": gpu_hours,
        "gpu_days": gpu_hours / 24,
        "cost_a100_spot": gpu_hours * 1.5,  # ~$1.50/hr spot
        "cost_a100_ondemand": gpu_hours * 3.0,  # ~$3/hr on-demand
    }


# ── Exercises ──

# TODO 6a: Compute FLOPs for GPT-2 Small (12 layers, 768 dim, 12 heads, 1024 seq)
# gpt2_flops = count_transformer_flops(12, 768, 12, 1024, 50257)
# Print the breakdown. Does 2*N approximation hold?

# TODO 6b: Compute FLOPs for a VLA (24 layers, 4096 dim, 32 heads)
# Note: VLA sequence = vision_tokens (256) + language_tokens (128) + action_tokens (7)
# How many FLOPs per action prediction step?

# TODO 6c: Real-time budget analysis
# A robot runs at 10 Hz (100ms per action).
# GPU has 100 TFLOPS effective throughput.
# What's the maximum model size (in params) that fits in the time budget?
# max_params = time_budget * tflops / (2 * seq_len)

# TODO 6d: Compare training costs:
# | Model | Params | Tokens | GPU-hours (A100) | Cost |
# |-------|--------|--------|------------------|------|
# | GPT-2 | 124M   | 10B    | ?                | ?    |
# | LLaMA-7B | 7B  | 1.4T   | ?                | ?    |
# | OpenVLA | 7B   | 970K demos × 64 tokens | ? | ? |
```

### 6.2 — Profiling with PyTorch

```python
"""Profile actual FLOPs using PyTorch's built-in tools."""
from torch.profiler import profile, ProfilerActivity
from fvcore.nn import FlopCountAnalysis  # pip install fvcore


def profile_model_flops(model: nn.Module, input_shape: tuple):
    """Use fvcore to count actual FLOPs.
    
    Compare with our analytical estimate above.
    """
    dummy_input = torch.randn(*input_shape)
    
    # fvcore analysis
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"Total FLOPs: {flops.total():,.0f}")
    print(f"Total TFLOPs: {flops.total() / 1e12:.3f}")
    print(f"\nPer-module breakdown:")
    print(flops.by_module())
    
    return flops.total()


# TODO 6e: Build a small transformer, count FLOPs with fvcore,
# then compare with your count_transformer_flops() output.
# Are they within 5% of each other?
```

### 6.3 — Key Insight for VLA Design

```
Real-time constraint analysis:
─────────────────────────────────
Robot control rate: 10 Hz → 100ms per inference
GPU budget: ~100 TFLOPS (A100 at 30% util)

Available FLOPs per step: 100 TFLOPS × 0.1s = 10 TFLOPs

OpenVLA (7B params):
  Forward FLOPs ≈ 2 × 7B × 391 tokens = 5.5 TFLOPs → just barely fits!
  
Smaller VLA (1B params):
  Forward FLOPs ≈ 2 × 1B × 391 tokens = 0.78 TFLOPs → comfortable margin
  
This is why π₀ and other production VLAs use ~3B params or quantization.
```

---

## Stretch Goals

### Stretch 1: Attention Visualization
Visualize attention patterns from different layers and heads. Use `matplotlib.imshow()` on the attention weights. What patterns do you see? (Diagonal = positional, vertical stripes = key tokens, blocks = phrases)

### Stretch 2: Mixed-Precision Ablation
Compare training in fp32, fp16, and bf16. Measure:
- Training speed (iterations/sec)
- Memory usage
- Final validation loss
- Any numerical issues?

### Stretch 3: Custom Dataset
Train nanoGPT on a different dataset (code, Wikipedia, your own writing). How does the optimal architecture change? Do scaling curves shift?

### Stretch 4: Weight Tying Ablation
Train two versions: one with weight tying, one without. Compare:
- Parameter count
- Validation loss
- Generated text quality
- Is weight tying always beneficial?

---

*Next exercise: [05-finetune-llm.md](05-finetune-llm.md) — Fine-tuning LLMs*
