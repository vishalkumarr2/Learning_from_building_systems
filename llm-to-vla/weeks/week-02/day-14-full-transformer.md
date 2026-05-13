# Day 14: The Full Transformer — "Attention Is All You Need"

> **Phase II — Attention, Transformers & Scaling | Week 2 | 2.5 hours**
> *"The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output."*
> — Vaswani et al., 2017

## Navigation
- **Previous:** [Day 13: Positional Encoding](day-13-positional-encoding.md)
- **Next:** [Day 15: Training a Transformer](../week-03/day-15-training-transformer.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 14.1 The Complete Architecture

The transformer consists of an **encoder** (processes the source) and a **decoder** (generates the target). Each is a stack of identical layers.

```
                    THE TRANSFORMER ARCHITECTURE

  ┌──────────── ENCODER ────────────┐    ┌──────────── DECODER ────────────┐
  │                                 │    │                                 │
  │  ┌─────────────────────────┐    │    │  ┌─────────────────────────┐    │
  │  │  Encoder Layer × N      │    │    │  │  Decoder Layer × N      │    │
  │  │  ┌───────────────────┐  │    │    │  │  ┌───────────────────┐  │    │
  │  │  │ Multi-Head        │  │    │    │  │  │ Masked Multi-Head │  │    │
  │  │  │ Self-Attention    │  │    │    │  │  │ Self-Attention    │  │    │
  │  │  └────────┬──────────┘  │    │    │  │  └────────┬──────────┘  │    │
  │  │       Add & Norm        │    │    │  │       Add & Norm        │    │
  │  │  ┌───────────────────┐  │    │    │  │  ┌───────────────────┐  │    │
  │  │  │ Feed-Forward      │  │    │    │  │  │ Cross-Attention   │  │    │
  │  │  │ Network           │  │    │    │  │  │ (Q=decoder,       │  │    │
  │  │  └────────┬──────────┘  │    │    │  │  │  K,V=encoder)     │  │    │
  │  │       Add & Norm        │    │    │  │  └────────┬──────────┘  │    │
  │  └─────────────────────────┘    │    │  │       Add & Norm        │    │
  │                                 │    │  │  ┌───────────────────┐  │    │
  │  Input Embedding                │    │  │  │ Feed-Forward      │  │    │
  │       +                         │    │  │  │ Network           │  │    │
  │  Positional Encoding            │    │  │  └────────┬──────────┘  │    │
  │       ↑                         │    │  │       Add & Norm        │    │
  │  Source Tokens                  │    │  └─────────────────────────┘    │
  └─────────────────────────────────┘    │                                 │
                                         │  Output Embedding              │
                                         │       +                        │
                                         │  Positional Encoding           │
                                         │       ↑                        │
                                         │  Target Tokens (shifted right) │
                                         └─────────────────────────────────┘
```

### 14.2 Encoder Block

Each encoder layer has two sublayers:

**Sublayer 1: Multi-Head Self-Attention**
$$\text{MHA}(X, X, X) \quad \text{// Q=K=V=X, every position attends to every position}$$

**Sublayer 2: Position-wise Feed-Forward Network (FFN)**
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$. Typically $d_{\text{ff}} = 4 \times d_{\text{model}}$.

**Each sublayer** is wrapped with a residual connection and layer normalization:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 14.3 Decoder Block

Each decoder layer has **three** sublayers:

1. **Masked self-attention** — the decoder attends to itself, but with a causal mask preventing positions from attending to future tokens
2. **Cross-attention** — the decoder queries attend to encoder key/value pairs. This is where the decoder "looks at" the source
3. **Feed-forward network** — same as encoder

```
Target tokens → Embed + PE → [Masked Self-Attn] → [Cross-Attn] → [FFN] → Output
                                    ↑                    ↑
                               causal mask          encoder output
```

### 14.4 Pre-LN vs Post-LN

The original paper uses **Post-LN** (normalize after the residual):
$$x = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Modern practice uses **Pre-LN** (normalize before the sublayer):
$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Pre-LN is much easier to train:
- Gradients flow directly through the residual path without passing through LayerNorm
- No need for careful warmup scheduling
- Enables training deeper models (100+ layers)

### 14.5 The Feed-Forward Network: Per-Position MLP

The FFN processes each position independently (no interaction between positions). It's the same MLP applied to each token:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

**Activation functions** have evolved:
- **ReLU** (original): $\sigma(x) = \max(0, x)$
- **GELU** (BERT, GPT-2): $\sigma(x) = x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF
- **SwiGLU** (LLaMA, modern): $\sigma(x) = \text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$

The FFN's hidden dimension $d_{\text{ff}} = 4 d_{\text{model}}$ creates a bottleneck:

$$d_{\text{model}} \xrightarrow{W_1} d_{\text{ff}} \xrightarrow{W_2} d_{\text{model}}$$

This is an **expand-and-compress** pattern — the inner dimension provides capacity for complex transformations before projecting back down.

### 14.6 Residual Connections: The Highway for Gradients

Every sublayer is wrapped with $x + \text{Sublayer}(x)$. This is essential for deep networks:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=l}^{L-1} \left(I + \frac{\partial f_i}{\partial x_i}\right)$$

The identity term $I$ guarantees that gradients can flow through the residual stream unchanged. Even if a sublayer's gradient vanishes, the gradient through the skip connection survives.

**The residual stream** view (Elhage et al., 2021): think of the transformer as a stream of vectors where each layer *writes additions* to the stream. Each attention head and FFN block reads from the stream and writes back a delta.

---

## Implementation (60 min)

### 14.7 Transformer Encoder Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer with Pre-LN."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN: normalize before sublayer
        # Sublayer 1: Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask)
        x = x + self.dropout(attn_out)
        
        # Sublayer 2: FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with Pre-LN."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,           # target sequence
        memory: torch.Tensor,      # encoder output
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Sublayer 1: Masked self-attention
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + self.dropout(self_attn_out)
        
        # Sublayer 2: Cross-attention (Q=decoder, K=V=encoder)
        x_norm = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(x_norm, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(cross_attn_out)
        
        # Sublayer 3: FFN
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x
```

### 14.8 Full Transformer Model

```python
class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings + positional encoding
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)
    
    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.decoder_norm(x)
    
    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask)
        return self.output_proj(output)  # (batch, tgt_len, tgt_vocab)
```

### 14.9 Parameter Count Analysis

```python
def count_transformer_params(model: Transformer):
    """Break down parameter count by component."""
    def count(module):
        return sum(p.numel() for p in module.parameters())
    
    print(f"Source embedding:   {count(model.src_embed):>10,}")
    print(f"Target embedding:   {count(model.tgt_embed):>10,}")
    print(f"Encoder layers:     {count(model.encoder_layers):>10,}")
    print(f"Decoder layers:     {count(model.decoder_layers):>10,}")
    print(f"Output projection:  {count(model.output_proj):>10,}")
    
    total = count(model)
    print(f"{'Total':>20s}: {total:>10,}")
    return total


# Base transformer config
model = Transformer(src_vocab=32000, tgt_vocab=32000, d_model=512, n_heads=8, d_ff=2048)
count_transformer_params(model)
# Expect ~65M parameters for base config
```

---

## Exercise (45 min)

### E14.1 Parameter Audit

For the base transformer ($d_{\text{model}}=512$, $h=8$, $d_{\text{ff}}=2048$, $N=6$):
1. Compute the parameter count for one encoder layer: MHA ($4 d^2$) + FFN ($2 \cdot d \cdot d_{\text{ff}}$) + LayerNorms ($4d$)
2. Multiply by 6 layers for the encoder total
3. How much of the total is FFN vs attention?
4. Compare with the "large" config ($d=1024$, $h=16$, $d_{\text{ff}}=4096$, $N=6$)

### E14.2 Token Tracing

Trace a single token through the entire forward pass:
1. Start with token ID 42 at position 3 in the source
2. Embedding: $42 \rightarrow \mathbb{R}^{512}$
3. Add positional encoding
4. Through encoder layer 1: self-attention (attending to all 10 source tokens) → FFN
5. Through decoder: masked self-attention → cross-attention → FFN
6. Final linear + softmax → probability distribution over vocab

Write pseudocode tracking the **shape** at each step.

### E14.3 Build and Validate

1. Instantiate your Transformer model with the base config
2. Create dummy source (batch=2, src_len=10) and target (batch=2, tgt_len=8) tensors
3. Generate a causal mask for the decoder
4. Run a forward pass and verify the output shape is (2, 8, vocab_size)
5. Compute a cross-entropy loss with random target labels

---

## Key Takeaways

- The transformer = **encoder** (self-attention + FFN) × N layers + **decoder** (masked self-attention + cross-attention + FFN) × N layers
- **Residual connections** enable gradient flow; **Layer normalization** stabilizes training
- **Pre-LN** (normalize before sublayer) is easier to train than Post-LN
- The **FFN** ($d_{\text{model}} \rightarrow 4 d_{\text{model}} \rightarrow d_{\text{model}}$) actually has more parameters than MHA
- Modern activations (GELU, SwiGLU) outperform ReLU in practice
- The decoder uses a **causal mask** to prevent attending to future tokens

## Connection to the Thread

The transformer is a **compression machine with full connectivity**. Unlike RNNs, which compress information sequentially (and lose distant information), the transformer gives every position direct access to every other position via attention. The FFN then processes each position's attended information independently. This is why transformers scale: they separate the *gathering* of information (attention) from the *processing* of information (FFN), and both can be parallelized across the sequence. We now have all the pieces for modern LLMs — tomorrow we train one.

## Further Reading

- Vaswani et al. **"Attention Is All You Need"** (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Alammar. **"The Illustrated Transformer"** (2018). [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/)
- Xiong et al. **"On Layer Normalization in the Transformer Architecture"** (2020). [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) — Pre-LN vs Post-LN analysis
- Elhage et al. **"A Mathematical Framework for Transformer Circuits"** (2021). [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html) — the residual stream interpretation
- The Annotated Transformer. [nlp.seas.harvard.edu](https://nlp.seas.harvard.edu/annotated-transformer/)
