# Exercise 03 — Build a Transformer

> Phase II · Days 13–16 · Prerequisites: 02-attention-from-scratch
> Build the full transformer architecture from scratch, train on translation, understand every component.

---

## Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

---

## Exercise 1: Positional Encoding

### Task 1a: Sinusoidal Positional Encoding

```python
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # TODO: Create PE matrix of shape (max_len, d_model)
        # Register as a buffer (not a parameter — no gradients)
        pe = torch.zeros(max_len, d_model)
        
        # TODO:
        # 1. Create position indices: (max_len, 1)
        # 2. Create dimension indices and compute div_term: 10000^(2i/d_model)
        # 3. Apply sin to even indices, cos to odd indices
        # 4. Register as buffer
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + PE[:seq_len] with dropout
        """
        # TODO: Add positional encoding and apply dropout
        pass


# Test
d_model = 64
pe_module = SinusoidalPE(d_model)
x = torch.zeros(1, 20, d_model)
out = pe_module(x)
assert out.shape == (1, 20, d_model)
print("✅ Sinusoidal PE shape correct")
```

### Task 1b: Implement RoPE

```python
class RoPE(nn.Module):
    """Rotary Position Embedding.
    
    Encodes position by rotating pairs of dimensions in Q and K.
    """
    def __init__(self, d_model: int, max_len: int = 5000, theta: float = 10000.0):
        super().__init__()
        
        # TODO: Precompute rotation frequencies
        # freqs = 1.0 / (theta^(2i/d_model)) for i = 0, 1, ..., d_model/2 - 1
        # Create a (max_len, d_model/2) matrix of angles: position * freq
        # Store as complex exponentials for efficient rotation
        pass
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.
        
        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)
        
        Returns:
            Rotated q and k with same shapes
        """
        # TODO:
        # 1. View q, k as complex numbers (pairs of consecutive dims)
        # 2. Multiply by precomputed complex exponentials
        # 3. Convert back to real
        pass

# Test
rope = RoPE(d_model=64)
q = torch.randn(2, 8, 16, 8)  # (batch, heads, seq, head_dim)
k = torch.randn(2, 8, 16, 8)
q_rot, k_rot = rope(q, k)
assert q_rot.shape == q.shape
assert k_rot.shape == k.shape
print("✅ RoPE shapes correct")
```

### Task 1c: Visualize Encoding Matrices

```python
def visualize_pe():
    """Visualize positional encoding patterns."""
    d_model = 128
    max_len = 100
    
    pe = SinusoidalPE(d_model, max_len, dropout=0.0)
    # Extract the PE buffer
    pe_matrix = pe.pe[:max_len].numpy() if hasattr(pe, 'pe') else None
    
    if pe_matrix is None:
        print("⚠️ Implement SinusoidalPE first!")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Full PE heatmap
    axes[0].imshow(pe_matrix, cmap='RdBu', aspect='auto')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    axes[0].set_title('Positional Encoding Matrix')
    
    # 2. Selected dimensions
    for dim in [0, 1, 10, 11, 60, 61]:
        axes[1].plot(pe_matrix[:, dim], label=f'dim {dim}')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('PE value')
    axes[1].set_title('PE values across positions')
    axes[1].legend(fontsize=8)
    
    # 3. Dot products showing relative position sensitivity
    # PE(pos) · PE(pos + offset) should depend mainly on offset
    dot_products = pe_matrix @ pe_matrix.T
    axes[2].imshow(dot_products, cmap='RdBu', aspect='auto')
    axes[2].set_xlabel('Position j')
    axes[2].set_ylabel('Position i')
    axes[2].set_title('PE Dot Products (relative position)')
    
    plt.tight_layout()
    plt.show()

# TODO: Run and observe:
# - Low dimensions oscillate slowly (global position)
# - High dimensions oscillate fast (local position)
# - Dot product matrix shows diagonal banding (relative position)
# visualize_pe()
```

### Task 1d: Relative Position via Dot Products

```python
def show_relative_position():
    """Show that sinusoidal PE enables relative position awareness."""
    d_model = 64
    pe = SinusoidalPE(d_model, 200, dropout=0.0)
    pe_matrix = pe.pe  # (200, d_model)
    
    # TODO: For positions 50, 60, 70, compute:
    # dot(PE[50], PE[60]) vs dot(PE[60], PE[70])
    # They should be approximately equal (both are offset 10)
    
    # Also compare:
    # dot(PE[50], PE[51]) vs dot(PE[100], PE[101])
    # These should also be approximately equal (both are offset 1)
    
    # This shows that the dot product captures RELATIVE position
    pass
```

---

## Exercise 2: Transformer Encoder Layer

### Task 2a: Build from Scratch

```python
class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer.
    
    Architecture (Post-LN):
        x → MultiHeadAttn → Dropout → Add(x) → LayerNorm → 
          → FFN → Dropout → Add → LayerNorm → output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Initialize components
        # - MultiHeadAttention (from exercise 02)
        # - FeedForward network (Linear → GELU → Linear)
        # - Two LayerNorms
        # - Dropout
        pass
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        # TODO: Implement Post-LN encoder layer
        # attn_out = self.self_attn(x, x, x, mask)
        # x = self.norm1(x + self.dropout(attn_out))
        # ffn_out = self.ffn(x)
        # x = self.norm2(x + self.dropout(ffn_out))
        pass


# Test
d_model, n_heads, d_ff = 256, 8, 1024
enc_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

x = torch.randn(2, 10, d_model)
out = enc_layer(x)

assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
print("✅ Encoder layer shape correct")

# Verify residual connection: output should be close to input at init
# (because sub-layers output ~0 at initialization)
print(f"Input-output distance: {(out - x).norm():.4f}")
print(f"(Should be small relative to input norm: {x.norm():.4f})")
```

### Task 2b: Pre-LN Variant

```python
class PreLNEncoderLayer(nn.Module):
    """Pre-LayerNorm encoder layer (modern standard).
    
    Architecture:
        x → LayerNorm → MultiHeadAttn → Dropout → Add(x) →
          → LayerNorm → FFN → Dropout → Add → output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Same components, different forward order
        pass
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # TODO: Pre-LN forward
        # residual = x
        # x = self.norm1(x)
        # x = residual + self.dropout(self.self_attn(x, x, x, mask))
        # residual = x
        # x = self.norm2(x)
        # x = residual + self.dropout(self.ffn(x))
        pass

# TODO: Compare gradient norms between Post-LN and Pre-LN for a deep stack
```

---

## Exercise 3: Transformer Decoder Layer

### Task 3a: Causal Mask

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal attention mask.
    
    Returns:
        Boolean mask of shape (seq_len, seq_len)
        True = position should be masked (cannot attend)
    """
    # TODO: Create upper triangular mask
    # Position i can attend to positions 0..i only
    pass


# Test
mask = create_causal_mask(5)
print("Causal mask (True = blocked):")
print(mask.int())
# Expected:
# 0 1 1 1 1
# 0 0 1 1 1
# 0 0 0 1 1
# 0 0 0 0 1
# 0 0 0 0 0

# Verify: position 2 can see positions 0, 1, 2 but not 3, 4
assert mask[2, 0] == False, "Position 2 should see position 0"
assert mask[2, 2] == False, "Position 2 should see itself"
assert mask[2, 3] == True, "Position 2 should NOT see position 3"
print("✅ Causal mask correct")
```

### Task 3b: Decoder Layer with Cross-Attention

```python
class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with three sub-layers.
    
    1. Masked self-attention (causal)
    2. Cross-attention (Q from decoder, K/V from encoder)
    3. Feed-forward network
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Initialize components
        # - Masked self-attention
        # - Cross-attention
        # - FFN
        # - Three LayerNorms (one per sub-layer)
        # - Dropout
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) — decoder input
            encoder_output: (batch, src_len, d_model) — encoder output
            src_mask: mask for cross-attention
            tgt_mask: causal mask for self-attention
        Returns:
            (batch, tgt_len, d_model)
        """
        # TODO: Three sub-layers with residuals and norms
        # 1. Masked self-attention: query=key=value=x, mask=tgt_mask
        # 2. Cross-attention: query=x, key=value=encoder_output, mask=src_mask
        # 3. FFN
        pass


# Test
d_model, n_heads, d_ff = 256, 8, 1024
dec_layer = TransformerDecoderLayer(d_model, n_heads, d_ff)

tgt = torch.randn(2, 8, d_model)   # Decoder input
memory = torch.randn(2, 12, d_model)  # Encoder output
tgt_mask = create_causal_mask(8)

out = dec_layer(tgt, memory, tgt_mask=tgt_mask)
assert out.shape == tgt.shape, f"Expected {tgt.shape}, got {out.shape}"
print("✅ Decoder layer shape correct")
```

---

## Exercise 4: Full Transformer

### Task 4a: Complete Model

```python
@dataclass
class TransformerConfig:
    src_vocab_size: int = 10000
    tgt_vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    d_ff: int = 1024
    max_len: int = 512
    dropout: float = 0.1
    pad_idx: int = 0


class Transformer(nn.Module):
    """Complete encoder-decoder transformer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # TODO: Build:
        # 1. Source and target embeddings
        # 2. Positional encoding (shared or separate)
        # 3. Encoder stack (N encoder layers)
        # 4. Decoder stack (N decoder layers)
        # 5. Output linear projection (d_model → tgt_vocab_size)
        # 6. Embedding scale factor: sqrt(d_model)
        
        pass
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Run encoder on source sequence.
        
        Args:
            src: (batch, src_len) — token indices
        Returns:
            (batch, src_len, d_model) — encoder output
        """
        # TODO: Embed + scale + PE + dropout → N encoder layers
        pass
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run decoder on target sequence with encoder memory.
        
        Args:
            tgt: (batch, tgt_len) — token indices
            memory: (batch, src_len, d_model) — encoder output
        Returns:
            (batch, tgt_len, d_model) — decoder output
        """
        # TODO: Embed + scale + PE + dropout → N decoder layers
        pass
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full forward pass.
        
        Returns:
            (batch, tgt_len, tgt_vocab_size) — logits
        """
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.output_proj(output)


# Test
config = TransformerConfig(src_vocab_size=1000, tgt_vocab_size=1000)
model = Transformer(config).to(device)

src = torch.randint(1, 1000, (2, 15)).to(device)
tgt = torch.randint(1, 1000, (2, 12)).to(device)
tgt_mask = create_causal_mask(12).to(device)

logits = model(src, tgt, tgt_mask=tgt_mask)
assert logits.shape == (2, 12, 1000), f"Expected (2, 12, 1000), got {logits.shape}"
print("✅ Full transformer shape correct")
```

### Task 4b: Count Parameters

```python
def count_parameters(model):
    """Count and categorize model parameters."""
    total = 0
    categories = {}
    
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        
        # Categorize
        if 'embed' in name:
            cat = 'embeddings'
        elif 'encoder' in name and 'attn' in name:
            cat = 'encoder_attention'
        elif 'encoder' in name and 'ffn' in name:
            cat = 'encoder_ffn'
        elif 'decoder' in name and 'attn' in name:
            cat = 'decoder_attention'
        elif 'decoder' in name and 'ffn' in name:
            cat = 'decoder_ffn'
        elif 'output' in name or 'proj' in name:
            cat = 'output_projection'
        else:
            cat = 'other'
        
        categories[cat] = categories.get(cat, 0) + n
    
    print(f"Total parameters: {total:,}")
    print(f"\nBreakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {count:>10,} ({100*count/total:.1f}%)")
    
    return total

# TODO: Run and analyze — where are most parameters?
# count_parameters(model)
```

### Task 4c: Train on Multi30k EN→DE Translation

```python
# NOTE: This is a full training loop. It may take 30-60 minutes on GPU.
# If you don't have Multi30k, use a simple synthetic task first (see below).

def create_synthetic_translation_data(n_samples=5000, src_vocab=100, tgt_vocab=100, max_len=20):
    """Create synthetic data: reverse sequence with vocabulary shift."""
    data = []
    for _ in range(n_samples):
        length = torch.randint(5, max_len, (1,)).item()
        src = torch.randint(2, src_vocab, (length,))
        # Target is reversed source with a vocabulary offset
        tgt = src.flip(0) + tgt_vocab // 2
        tgt = tgt % tgt_vocab
        tgt = torch.clamp(tgt, min=2)  # Avoid special tokens
        data.append((src, tgt))
    return data

# TODO: 
# 1. Create or load data
# 2. Build vocabulary and tokenize (or use synthetic data)
# 3. Create DataLoader with padding collation
# 4. Training loop:
#    - Teacher forcing: tgt_input = tgt[:, :-1], tgt_target = tgt[:, 1:]
#    - Causal mask on decoder
#    - Cross-entropy loss with ignore_index=pad_idx
#    - Track training loss
# 5. Simple greedy decoding for evaluation

# Skeleton:
"""
for epoch in range(num_epochs):
    model.train()
    for src, tgt in train_loader:
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        
        tgt_mask = create_causal_mask(tgt_input.size(1)).to(device)
        
        logits = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, config.tgt_vocab_size),
            tgt_target.reshape(-1),
            ignore_index=config.pad_idx,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
"""
```

---

## Exercise 5: Training with Stability

### Task 5a: Implement Warmup + Cosine Decay

```python
class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 max_lr: float, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self._step = 0
    
    def step(self):
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        # TODO:
        # If step < warmup_steps: linear ramp from 0 to max_lr
        # Else: cosine decay from max_lr to min_lr
        pass


# Test: visualize the schedule
def plot_schedule():
    dummy_model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(dummy_model.parameters())
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=1000, 
                                       total_steps=10000, max_lr=3e-4)
    
    lrs = []
    for _ in range(10000):
        scheduler.step()
        lrs.append(scheduler.get_lr())
    
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Warmup Schedule')
    plt.axvline(x=1000, color='r', linestyle='--', label='Warmup end')
    plt.legend()
    plt.show()

# plot_schedule()
```

### Task 5b: Training Monitor (reuse from Exercise 01)

```python
class TrainingMonitor:
    """Track training metrics with gradient and loss monitoring."""
    
    def __init__(self):
        self.losses = []
        self.grad_norms = []
        self.lrs = []
    
    def log(self, loss: float, model: nn.Module, lr: float):
        self.losses.append(loss)
        self.lrs.append(lr)
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
    
    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.losses)
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Step')
        
        axes[1].plot(self.grad_norms)
        axes[1].set_title('Gradient Norm')
        axes[1].set_xlabel('Step')
        
        axes[2].plot(self.lrs)
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Step')
        
        plt.tight_layout()
        plt.show()
```

### Task 5c: Compare with Seq2Seq (Exercise 01 Ex4)

```python
# TODO: Train both models on the same synthetic reversal task:
# 1. RNN seq2seq + Bahdanau attention (from exercise 01/02)
# 2. Transformer (from this exercise)
#
# Compare:
# - Training loss curves
# - Final accuracy on test set
# - Training time per epoch
# - Gradient norm stability
#
# Expected observations:
# - Transformer trains faster per epoch (parallelism)
# - Transformer may need more warmup steps
# - Transformer should achieve equal or better final accuracy
# - RNN may converge faster in terms of number of examples seen (sample efficiency)
#   but transformer wins on wall-clock time with GPU
```

---

## Exercise 6: KV-Cache for Efficient Generation

**Goal**: Implement the KV-cache optimization that makes autoregressive generation
$O(n)$ per token instead of $O(n^2)$. This is critical for real-time VLA inference.

### 6.1 — The Problem Without Caching

```python
# Naive autoregressive generation:
# To generate token at position t, you recompute attention over ALL previous tokens
# 
# Step 1: process [token_0] → compute K,V for pos 0 → output token_1
# Step 2: process [token_0, token_1] → recompute K,V for pos 0,1 → output token_2
# Step 3: process [token_0, token_1, token_2] → recompute K,V for 0,1,2 → output token_3
# ...
# Step n: process entire sequence again!
#
# Total compute: O(1 + 2 + 3 + ... + n) = O(n²) attention computations
# This is wasteful because K,V for past tokens never change!
```

### 6.2 — KV-Cache Implementation

```python
class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with KV-cache for efficient generation.
    
    During generation:
    - First call (prefill): process entire prompt, cache all K,V
    - Subsequent calls: only process the new token, append to cache
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Cache storage
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None
    
    def reset_cache(self):
        """Clear the KV cache (call at start of each new sequence)."""
        self.cache_k = None
        self.cache_v = None
    
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — full sequence or single new token
            use_cache: if True, use and update the KV cache
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, T, _ = x.shape
        
        # Compute Q, K, V for the input
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        if use_cache:
            if self.cache_k is not None:
                # Append new K, V to cache
                K = torch.cat([self.cache_k, K], dim=2)  # (B, h, cache_len + T, d_k)
                V = torch.cat([self.cache_v, V], dim=2)
            # Update cache
            self.cache_k = K.detach()
            self.cache_v = V.detach()
        
        # Attention: Q attends to ALL K (including cached)
        # Q: (B, h, T, d_k), K: (B, h, S, d_k) where S = total sequence so far
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Causal mask (only needed for prefill, not single-token generation)
        if T > 1:
            S = K.shape[2]
            # TODO: Create causal mask that allows attending to all cached tokens
            # but prevents future tokens within the current chunk
            causal_mask = torch.triu(torch.ones(T, S, device=x.device), diagonal=S-T+1)
            scores = scores.masked_fill(causal_mask.bool().unsqueeze(0).unsqueeze(0), -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, h, T, d_k)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


def benchmark_generation(model, prompt_len: int = 100, gen_len: int = 200):
    """Benchmark generation with and without KV-cache.
    
    TODO: 
    1. Generate `gen_len` tokens using naive approach (feed full sequence each step)
    2. Generate `gen_len` tokens using KV-cache (feed only new token each step)
    3. Verify outputs are identical
    4. Measure and compare wall-clock time
    5. Plot: generation time vs sequence length for both approaches
    """
    import time
    
    # TODO: Implement benchmarking
    # Expected speedup: ~5-20x for sequences of length 200+
    pass
```

### 6.3 — Exercises

1. Implement `CachedMultiHeadAttention` and verify outputs match non-cached version
2. Benchmark: plot generation time vs sequence length (50, 100, 200, 500 tokens)
3. Measure memory: KV-cache grows linearly with sequence length. For a 12-layer model with d_model=768, how much memory does the cache use for 2048 tokens?
   - Calculate: $2 \times n_{layers} \times seq\_len \times d_{model} \times sizeof(float16)$
4. **Paged attention** (bonus): Research how vLLM's PagedAttention manages KV-cache memory. Write a 1-paragraph summary.
5. **VLA connection**: If a VLA needs to generate 7 action tokens conditioned on 256 vision tokens, how much does KV-cache help? (Answer: vision tokens are cached, only 7 forward passes needed.)

---

## Exercise 7: Gradient Checkpointing

**Goal**: Implement gradient checkpointing to trade compute for memory —
essential for training large models on limited GPU memory.

### 7.1 — The Memory Problem

```python
# Normal backpropagation stores ALL intermediate activations:
# 12-layer transformer, batch=32, seq=512, d=768:
# Memory per layer ≈ 2 * batch * seq * d * sizeof(float32)
#                  ≈ 2 * 32 * 512 * 768 * 4 bytes ≈ 100 MB per layer
# Total: 12 layers × 100 MB = 1.2 GB just for activations!
#
# Gradient checkpointing: only store activations at "checkpoint" layers.
# During backward pass, recompute non-stored activations from checkpoints.
# Tradeoff: ~30% slower training, but ~70% less memory!
```

### 7.2 — Implementation

```python
import torch
from torch.utils.checkpoint import checkpoint


class TransformerWithCheckpointing(nn.Module):
    """Transformer that optionally uses gradient checkpointing."""
    
    def __init__(self, n_layers: int = 12, d_model: int = 768, 
                 n_heads: int = 12, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Recompute this layer's activations during backward
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


def measure_memory_savings(d_model=768, n_layers=12, seq_len=512, batch_size=32):
    """Compare memory usage with and without gradient checkpointing.
    
    TODO:
    1. Create model WITHOUT checkpointing, measure peak memory during training step
    2. Create model WITH checkpointing, measure peak memory during training step
    3. Also measure time per training step (checkpointing is slower)
    4. Print: memory savings (%) and time overhead (%)
    """
    # Hint: use torch.cuda.max_memory_allocated() if GPU available
    # Or estimate analytically:
    # Without: O(n_layers * batch * seq * d_model) activation memory
    # With (every-other-layer checkpoints): O(n_layers/2 * batch * seq * d_model)
    pass
```

### 7.3 — Exercises

1. Implement gradient checkpointing using `torch.utils.checkpoint`
2. Measure memory savings on your Exercise 5 transformer (if GPU available)
3. Measure the time overhead (typically 25-35%)
4. Calculate analytically: for a 24-layer model, checkpointing every 4th layer:
   - How much activation memory is saved? (75%)
   - What's the recomputation cost? (each non-checkpointed layer recomputed once)
5. **When to use**: You're fine-tuning a 7B VLA on a 24GB GPU. Without checkpointing you OOM. With checkpointing, estimate the max batch size that fits.

---

## Self-Check

Before moving on, verify:

- [ ] You can implement sinusoidal PE and explain why each frequency captures different position scales
- [ ] You understand RoPE's rotation mechanism and why it gives relative position for free
- [ ] You can build a complete encoder and decoder layer from scratch
- [ ] You understand the difference between Post-LN and Pre-LN
- [ ] You can construct a causal mask and explain what it prevents
- [ ] You can wire up a full transformer and train it
- [ ] You know the warmup + cosine schedule formula
- [ ] You can count parameters for a given config
- [ ] You can implement KV-cache and explain why it speeds up generation
- [ ] You understand the memory-compute tradeoff of gradient checkpointing

## Stretch Goals

1. **Implement greedy decoding**: Write an inference loop that generates one token at a time, feeding the output back as input. Include a `<eos>` stopping condition.

2. **Beam search**: Implement beam search with beam width $k$. Compare output quality with greedy decoding.

3. **Weight tying**: Share the target embedding matrix with the output projection matrix (a common trick that saves parameters). Does it help on your task?

4. **Compare activation functions**: Swap GELU for SwiGLU in the FFN. Does training improve? Compare loss curves.

5. **Deep vs wide**: Train two models with similar parameter counts:
   - Deep: d_model=128, 12 layers
   - Wide: d_model=512, 3 layers
   - Which performs better? Which trains more stably?

6. **Ablation study**: Remove one component at a time and measure the impact:
   - No residual connections
   - No layer norm
   - No positional encoding
   - Single head (n_heads=1)
   - No dropout
