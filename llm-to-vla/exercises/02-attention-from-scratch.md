# Exercise 02 — Attention from Scratch

> Phase II · Days 10–12 · Prerequisites: 01-foundations-exercises
> Build every attention variant by hand, verify numerically, visualize what each one learns.

---

## Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We'll use these throughout
def plot_attention(attn_weights, title="Attention Weights", xlabel="Key", ylabel="Query"):
    """Visualize attention weights as a heatmap."""
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_weights, cmap='Blues', aspect='auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(attn_weights.shape[0]):
        for j in range(attn_weights.shape[1]):
            ax.text(j, i, f'{attn_weights[i, j]:.2f}',
                   ha='center', va='center', fontsize=8,
                   color='white' if attn_weights[i, j] > 0.5 else 'black')
    
    plt.tight_layout()
    plt.show()
```

---

## Exercise 1: Bahdanau (Additive) Attention

**Goal**: Implement the original additive attention mechanism and visualize alignment on a toy sequence-to-sequence problem.

### Background

Bahdanau attention computes alignment scores using a small neural network:

$$e_{ij} = v^T \tanh(W_1 h_i^{dec} + W_2 h_j^{enc})$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$
$$c_i = \sum_j \alpha_{ij} h_j^{enc}$$

### Task

```python
class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism.
    
    Args:
        hidden_dim: Dimension of encoder and decoder hidden states
        attn_dim: Dimension of the attention hidden layer
    """
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        # TODO: Define W1 (projects decoder state), W2 (projects encoder state), v (scoring)
        self.W1 = None  # nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W2 = None  # nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = None   # nn.Linear(attn_dim, 1, bias=False)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        Args:
            decoder_hidden: (batch, hidden_dim) — current decoder state
            encoder_outputs: (batch, src_len, hidden_dim) — all encoder states
        
        Returns:
            context: (batch, hidden_dim) — weighted sum of encoder states
            attn_weights: (batch, src_len) — attention distribution
        """
        # TODO: 
        # 1. Project decoder_hidden through W1: (batch, 1, attn_dim)
        # 2. Project encoder_outputs through W2: (batch, src_len, attn_dim)
        # 3. Add them, apply tanh
        # 4. Score with v: (batch, src_len, 1) → squeeze to (batch, src_len)
        # 5. Softmax to get weights
        # 6. Weighted sum of encoder_outputs
        pass


# Test with toy data
batch_size, src_len, hidden_dim, attn_dim = 2, 6, 16, 8
encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
decoder_hidden = torch.randn(batch_size, hidden_dim)

attn = BahdanauAttention(hidden_dim, attn_dim)
context, weights = attn(decoder_hidden, encoder_outputs)

assert context.shape == (batch_size, hidden_dim), f"Expected ({batch_size}, {hidden_dim}), got {context.shape}"
assert weights.shape == (batch_size, src_len), f"Expected ({batch_size}, {src_len}), got {weights.shape}"
assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size)), "Weights must sum to 1"
print("✅ Bahdanau attention shapes correct")

# Visualize
plot_attention(weights[0].unsqueeze(0), title="Bahdanau Attention (random init)")
```

### Visualization Task

Create a simple sequence reversal task and train a small RNN + Bahdanau attention model:
- Input: sequences of digits [3, 1, 4, 1, 5]
- Output: reversed [5, 1, 4, 1, 3]
- After training, visualize the attention alignment — you should see a **diagonal stripe** going from top-right to bottom-left.

```python
# TODO: Train a small seq2seq model with Bahdanau attention on sequence reversal
# Then plot the attention heatmap
# Expected: anti-diagonal pattern (attending to reverse positions)
```

---

## Exercise 2: Scaled Dot-Product Attention

**Goal**: Implement scaled dot-product attention from scratch, verify against PyTorch, and understand why scaling matters.

### Task 2a: Implement from Scratch

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.
    
    Args:
        query: (batch, ..., seq_q, d_k)
        key: (batch, ..., seq_k, d_k)
        value: (batch, ..., seq_k, d_v)
        mask: (batch, ..., seq_q, seq_k) — True where attention should be blocked
    
    Returns:
        output: (batch, ..., seq_q, d_v)
        attn_weights: (batch, ..., seq_q, seq_k)
    """
    d_k = query.size(-1)
    
    # TODO:
    # 1. Compute raw scores: Q @ K^T
    # 2. Scale by sqrt(d_k)
    # 3. Apply mask (set masked positions to -inf)
    # 4. Softmax over key dimension
    # 5. Multiply by V
    pass
```

### Task 2b: Verify Against PyTorch

```python
# Create test inputs
batch, seq_q, seq_k, d_k, d_v = 2, 4, 6, 8, 8
Q = torch.randn(batch, seq_q, d_k)
K = torch.randn(batch, seq_k, d_k)
V = torch.randn(batch, seq_k, d_v)

# Your implementation
my_output, my_weights = scaled_dot_product_attention(Q, K, V)

# PyTorch reference
ref_output = F.scaled_dot_product_attention(Q, K, V)

assert torch.allclose(my_output, ref_output, atol=1e-5), "Output doesn't match PyTorch!"
print("✅ Your attention matches PyTorch's implementation")
```

### Task 2c: Hand-Compute for a 3-Token Sequence

**This is the most important part.** Do this on paper first, then verify with code.

```python
# Setup: 3 tokens, d_k = 2
# These are GIVEN — do the math by hand first!
Q = torch.tensor([
    [1.0, 0.0],   # Query for token 0
    [0.0, 1.0],   # Query for token 1
    [1.0, 1.0],   # Query for token 2
])

K = torch.tensor([
    [1.0, 0.0],   # Key for token 0
    [0.0, 1.0],   # Key for token 1
    [1.0, 1.0],   # Key for token 2
])

V = torch.tensor([
    [1.0, 0.0],   # Value for token 0
    [0.0, 1.0],   # Value for token 1
    [0.5, 0.5],   # Value for token 2
])

# Step 1: Raw scores = Q @ K^T (3x3 matrix)
# TODO: Calculate by hand:
# scores[0,0] = Q[0] · K[0] = 1*1 + 0*0 = ?
# scores[0,1] = Q[0] · K[1] = 1*0 + 0*1 = ?
# ... fill in all 9 values

# Step 2: Scale by sqrt(d_k) = sqrt(2) ≈ 1.414
# scaled_scores = scores / 1.414

# Step 3: Softmax each row
# attn_weights = softmax(scaled_scores, dim=-1)

# Step 4: Output = attn_weights @ V
# output[0] = weights[0,0]*V[0] + weights[0,1]*V[1] + weights[0,2]*V[2]

# Now verify your hand calculation:
output, weights = scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
)
print("Attention weights:")
print(weights.squeeze())
print("\nOutput:")
print(output.squeeze())

# TODO: Do your hand-computed values match?
```

### Task 2d: Demonstrate Softmax Saturation

```python
def show_saturation():
    """Show why scaling by sqrt(d_k) matters."""
    seq_len = 8
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, d_k in enumerate([2, 64, 512]):
        Q = torch.randn(1, seq_len, d_k)
        K = torch.randn(1, seq_len, d_k)
        
        # WITHOUT scaling
        raw_scores = Q @ K.transpose(-2, -1)
        unscaled_attn = torch.softmax(raw_scores, dim=-1)
        
        # WITH scaling
        scaled_scores = raw_scores / math.sqrt(d_k)
        scaled_attn = torch.softmax(scaled_scores, dim=-1)
        
        # Plot
        axes[idx].bar(range(seq_len), unscaled_attn[0, 0].detach(), alpha=0.5, label='Unscaled')
        axes[idx].bar(range(seq_len), scaled_attn[0, 0].detach(), alpha=0.5, label='Scaled')
        axes[idx].set_title(f'd_k = {d_k}\nRaw score std ≈ {raw_scores.std():.1f}')
        axes[idx].set_xlabel('Key position')
        axes[idx].set_ylabel('Attention weight')
        axes[idx].legend()
        axes[idx].set_ylim(0, 1)
    
    plt.suptitle('Softmax Saturation Without Scaling')
    plt.tight_layout()
    plt.show()

# TODO: Run this and observe:
# - At d_k=2: unscaled and scaled look similar
# - At d_k=64: unscaled becomes peaky
# - At d_k=512: unscaled is nearly one-hot → saturated gradients!
show_saturation()
```

> 💡 **Key insight to verify**: The variance of the dot product $q \cdot k$ scales as $O(d_k)$ when entries are ~$\mathcal{N}(0, 1)$. Dividing by $\sqrt{d_k}$ normalizes the variance back to ~1, keeping softmax in its useful gradient regime.

---

## Exercise 3: Multi-Head Attention

**Goal**: Build a complete multi-head attention module from explicit weight matrices.

### Task 3a: Implementation

```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit Q, K, V, O projections.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # TODO: Define four linear layers (no bias for clarity)
        self.W_Q = None  # nn.Linear(d_model, d_model, bias=False)
        self.W_K = None
        self.W_V = None
        self.W_O = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional mask
        
        Returns:
            output: (batch, seq_q, d_model)
            attn_weights: (batch, n_heads, seq_q, seq_k)
        """
        batch_size = query.size(0)
        
        # TODO:
        # 1. Project through W_Q, W_K, W_V
        # 2. Reshape to (batch, n_heads, seq_len, d_k) — split heads
        # 3. Apply scaled_dot_product_attention (your function from Ex 2)
        # 4. Reshape back to (batch, seq_len, d_model) — concatenate heads
        # 5. Project through W_O
        pass
```

### Task 3b: Verify Parameter Count

```python
d_model, n_heads = 512, 8

mha = MultiHeadAttention(d_model, n_heads)

# Count parameters
total_params = sum(p.numel() for p in mha.parameters())
expected = 4 * d_model * d_model  # W_Q + W_K + W_V + W_O (no bias)

print(f"Total parameters: {total_params:,}")
print(f"Expected: {expected:,}")
assert total_params == expected, f"Parameter count mismatch!"
print("✅ Parameter count correct")

# Breakdown
print(f"\nBreakdown:")
print(f"  W_Q: {d_model} × {d_model} = {d_model**2:,}")
print(f"  W_K: {d_model} × {d_model} = {d_model**2:,}")
print(f"  W_V: {d_model} × {d_model} = {d_model**2:,}")
print(f"  W_O: {d_model} × {d_model} = {d_model**2:,}")
print(f"  d_k = d_v = d_model / n_heads = {d_model // n_heads}")
```

### Task 3c: Visualize Different Heads' Attention Patterns

```python
def visualize_heads(mha, input_seq, labels=None):
    """Visualize attention patterns across all heads."""
    with torch.no_grad():
        output, attn_weights = mha(input_seq, input_seq, input_seq)
    
    n_heads = attn_weights.size(1)
    fig, axes = plt.subplots(2, n_heads // 2, figsize=(4 * n_heads // 2, 8))
    axes = axes.flatten()
    
    for h in range(n_heads):
        w = attn_weights[0, h].cpu().numpy()
        axes[h].imshow(w, cmap='Blues', vmin=0, vmax=1)
        axes[h].set_title(f'Head {h}')
        if labels:
            axes[h].set_xticks(range(len(labels)))
            axes[h].set_xticklabels(labels, rotation=45)
            axes[h].set_yticks(range(len(labels)))
            axes[h].set_yticklabels(labels)
    
    plt.suptitle('Attention Patterns by Head (random init)')
    plt.tight_layout()
    plt.show()

# TODO: Create a simple input and visualize
# Even at random init, different heads will show different (random) patterns
seq_len = 8
x = torch.randn(1, seq_len, d_model)
labels = [f'tok_{i}' for i in range(seq_len)]
visualize_heads(mha, x, labels)
```

### Task 3d: Experiment — 1 vs 4 vs 8 Heads

```python
def compare_head_counts(d_model=256, seq_len=32, n_trials=100):
    """Compare representational diversity across head counts.
    
    Measure: average cosine similarity between heads' attention patterns.
    Lower = more diverse = better.
    """
    results = {}
    
    for n_heads in [1, 4, 8, 16]:
        if d_model % n_heads != 0:
            continue
        
        diversities = []
        for _ in range(n_trials):
            mha = MultiHeadAttention(d_model, n_heads)
            x = torch.randn(1, seq_len, d_model)
            
            with torch.no_grad():
                _, attn = mha(x, x, x)  # (1, n_heads, seq, seq)
            
            if n_heads > 1:
                # Flatten attention patterns and compute pairwise cosine similarity
                patterns = attn[0].reshape(n_heads, -1)  # (n_heads, seq*seq)
                cos_sim = F.cosine_similarity(
                    patterns.unsqueeze(0), patterns.unsqueeze(1), dim=-1
                )
                # Average off-diagonal similarity
                mask = ~torch.eye(n_heads, dtype=torch.bool)
                avg_sim = cos_sim[mask].mean().item()
                diversities.append(avg_sim)
            else:
                diversities.append(1.0)
        
        results[n_heads] = np.mean(diversities)
        print(f"  n_heads={n_heads:2d}: avg inter-head similarity = {results[n_heads]:.4f}")
    
    # TODO: What do you observe?
    # More heads → lower similarity → more diverse attention patterns
    # But diminishing returns past a certain point

# compare_head_counts()
```

---

## Self-Check

Before moving on, verify you can answer:

- [ ] What is the difference between additive and dot-product attention? (Additive uses a learned MLP; dot-product uses direct similarity)
- [ ] Why does the softmax saturate without scaling? (Dot product variance grows with $d_k$, pushing softmax toward one-hot)
- [ ] How do multiple heads help? (Each head can learn a different attention pattern / type of relationship)
- [ ] What is the total parameter count for MHA? ($4 \times d_{model}^2$ without biases)
- [ ] What shape is the attention weight matrix? (seq_q × seq_k per head)
- [ ] Can you hand-compute attention for a 3-token example? (Exercise 2c)

## Stretch Goals

1. **Implement cross-attention**: Modify your MHA so Q comes from one sequence and K, V from another. Verify that attention weights have shape (batch, n_heads, seq_q, seq_k) where seq_q ≠ seq_k.

2. **Causal masking**: Add a causal mask to your attention function and verify that position $i$ only attends to positions $\leq i$.

3. **Attention entropy**: Compute the entropy of each attention distribution. Compare entropy at initialization vs after training. What does low entropy mean? (Focused attention on few positions.) What does high entropy mean? (Uniform, diffuse attention.)

4. **Relative position experiment**: Create two sequences that are identical but shuffled. Show that without positional encoding, attention produces identical outputs (up to permutation). Then add sinusoidal PE and show the outputs differ.

5. **Flash Attention comparison**: Use `torch.nn.functional.scaled_dot_product_attention` with `enable_flash=True` and compare speed for seq_len=1024, 4096, 16384. Plot the speedup curve.
