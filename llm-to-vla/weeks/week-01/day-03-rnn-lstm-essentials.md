# Day 3: RNN/LSTM Essentials

> **Phase I · Week 1 · Day 3 of 112 · 2.5 hours**

*"The vanishing gradient problem in RNNs is not a bug — it's the fundamental reason attention had to be invented. Understanding it deeply is understanding why transformers exist."*

---

## Navigation

| | |
|---|---|
| **Previous** | [← Day 2: CNN & ResNets](day-02-cnn-resnets.md) |
| **Next** | [Day 4: Seq2Seq & The Bottleneck →](day-04-seq2seq-bottleneck.md) |
| **Week** | [Week 1: DL Foundations](README.md) |
| **Phase** | [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md) |
| **Curriculum** | [Full Curriculum](../../CURRICULUM.md) |

---

## Why This Matters

CNNs process spatial data. But language, robot action sequences, and sensor streams are **sequential** — each element depends on what came before. RNNs were the first architectures to model sequences by maintaining a **hidden state** that accumulates information over time.

But RNNs have a fatal flaw: they can't remember long-range dependencies. The gradients literally vanish or explode when backpropagating through time. LSTMs partially fix this with gating mechanisms, but the fundamental problem — compressing an entire sequence into a fixed-size hidden state — remains. Tomorrow (Day 4), you'll see how seq2seq makes this bottleneck explicit. And in Week 2 (Day 10), attention will break through it entirely.

The thread from RNN → LSTM → attention → transformer → VLA is direct and unbroken. You must understand each step to see why the next was necessary.

---

## 1. Theory: Vanilla RNN

### 1.1 The Recurrence Equation

A vanilla RNN processes a sequence $x_1, x_2, \ldots, x_T$ one step at a time:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$
$$y_t = W_{hy} \cdot h_t + b_y$$

where:
- $h_t \in \mathbb{R}^d$ is the hidden state at time $t$
- $W_{hh} \in \mathbb{R}^{d \times d}$ is the recurrent weight matrix
- $W_{xh} \in \mathbb{R}^{d \times n}$ maps input to hidden
- $W_{hy} \in \mathbb{R}^{m \times d}$ maps hidden to output

```
    x₁        x₂        x₃        x₄
    │         │         │         │
    ▼         ▼         ▼         ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│      │─▶│      │─▶│      │─▶│      │
│  h₁  │  │  h₂  │  │  h₃  │  │  h₄  │
│      │  │      │  │      │  │      │
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   │         │         │         │
   ▼         ▼         ▼         ▼
   y₁        y₂        y₃        y₄

Key insight: The SAME weights (W_hh, W_xh) are reused at every time step.
This is weight sharing across time, analogous to CNNs sharing across space.
```

### 1.2 Backpropagation Through Time (BPTT)

To compute gradients, we "unroll" the RNN across time and apply standard backprop. The loss at time $T$ depends on all previous hidden states:

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}$$

The critical term is $\frac{\partial h_T}{\partial h_t}$, which chains through $T - t$ time steps:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} = \prod_{k=t}^{T-1} W_{hh}^T \cdot \text{diag}(\tanh'(z_k))$$

where $z_k = W_{hh} h_{k-1} + W_{xh} x_k + b_h$ and $\tanh'(\cdot) \in (0, 1]$.

---

## 2. Theory: The Vanishing Gradient Problem

### 2.1 The Mathematical Argument

Each factor in the product has norm bounded by:

$$\left\|\frac{\partial h_{k+1}}{\partial h_k}\right\| \leq \|W_{hh}\| \cdot \|\text{diag}(\tanh'(z_k))\| \leq \sigma_{\max}(W_{hh}) \cdot 1$$

where $\sigma_{\max}$ is the largest singular value. Over $T - t$ steps:

$$\left\|\frac{\partial h_T}{\partial h_t}\right\| \leq \sigma_{\max}(W_{hh})^{T-t}$$

This gives us three regimes:

| $\sigma_{\max}(W_{hh})$ | Behavior | Gradient | Result |
|---|---|---|---|
| $< 1$ | **Vanishing** | $\to 0$ exponentially | Can't learn long-range dependencies |
| $= 1$ | Stable | Maintained | Ideal but fragile |
| $> 1$ | **Exploding** | $\to \infty$ exponentially | Training diverges |

### 2.2 What Vanishing Gradients Mean in Practice

When $\frac{\partial L}{\partial h_t} \approx 0$ for small $t$:
- The network **cannot learn** from early tokens in long sequences
- Context from 50+ steps ago is effectively invisible
- The model develops a **recency bias** — only the last few tokens matter

**Gradient clipping** fixes exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

But there is **no simple fix** for vanishing gradients. You need a fundamentally different architecture. Enter LSTM.

---

## 3. Theory: LSTM — Long Short-Term Memory

### 3.1 The Core Idea

LSTM adds a **cell state** $c_t$ — a separate memory path that can carry information across many time steps without being squashed through nonlinearities. Three **gates** control what to forget, what to write, and what to read.

### 3.2 The LSTM Equations

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

where $\sigma$ is sigmoid, $\odot$ is element-wise multiplication, and $[\cdot, \cdot]$ is concatenation.

### 3.3 Gate Intuitions

```
                    Forget Gate (f_t)
                    "What to erase from memory"
                         │
    c_{t-1} ──────(×)────┤
                         │      Input Gate (i_t)
                         │      "What new info to write"
                         │           │
                        (+)────────(×)──── c̃_t (candidate)
                         │
                       c_t ─────────────────────────▶ c_{t+1}
                         │
                       tanh
                         │
                        (×)──── Output Gate (o_t)
                         │      "What to expose as output"
                         │
                       h_t ─────────────────────────▶ h_{t+1}
```

| Gate | Value | Meaning |
|------|-------|---------|
| Forget $f_t \approx 1$ | Keep | "This memory is still relevant" |
| Forget $f_t \approx 0$ | Erase | "This memory is no longer needed" |
| Input $i_t \approx 1$ | Write | "Store this new information" |
| Input $i_t \approx 0$ | Ignore | "Nothing important at this step" |
| Output $o_t \approx 1$ | Read | "This memory is relevant to output now" |
| Output $o_t \approx 0$ | Hide | "Memory exists but isn't needed for output" |

### 3.4 Why LSTM Fixes Vanishing Gradients

The gradient through the cell state:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

If the forget gate learns $f_t \approx 1$, the gradient flows **unchanged** through the cell state — no squashing, no matrix multiplication, just element-wise scaling. This is the cell state's role: a **gradient highway** (same principle as ResNet skip connections!).

Over $T - t$ steps:

$$\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^{T} f_k$$

If forget gates stay near 1, gradients persist. The network *learns* which information to maintain through the gate values.

### 3.5 GRU: A Simpler Alternative

The Gated Recurrent Unit (Cho et al., 2014) merges forget and input gates into a single "update gate" and removes the cell state:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRU has fewer parameters than LSTM and often performs comparably. In practice, LSTM and GRU perform similarly — the choice is empirical.

---

## 4. Implementation: RNN and LSTM from Scratch

### 4.1 Vanilla RNN Cell

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNNCell(nn.Module):
    """RNN cell implemented from scratch (no nn.RNN)."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined weight matrices for efficiency
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
    
    def forward(self, x_t, h_prev):
        """
        x_t: (batch, input_size) — input at time t
        h_prev: (batch, hidden_size) — previous hidden state
        Returns: h_t (batch, hidden_size)
        """
        h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))
        return h_t
```

### 4.2 LSTM Cell

```python
class LSTMCell(nn.Module):
    """LSTM cell implemented from scratch (no nn.LSTM)."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # All four gates in one matrix for efficiency
        # Order: input, forget, candidate, output
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
    
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t: (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        Returns: (h_t, c_t)
        """
        # Compute all four gates simultaneously
        gates = self.W_x(x_t) + self.W_h(h_prev)  # (batch, 4*hidden)
        
        # Split into four gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        i = torch.sigmoid(i)    # input gate
        f = torch.sigmoid(f)    # forget gate
        g = torch.tanh(g)       # candidate
        o = torch.sigmoid(o)    # output gate
        
        # Cell state update
        c_t = f * c_prev + i * g
        
        # Hidden state
        h_t = o * torch.tanh(c_t)
        
        return h_t, c_t
```

### 4.3 Character-Level Language Model

```python
class CharRNN(nn.Module):
    """Character-level language model using our custom LSTM."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cells = nn.ModuleList([
            LSTMCell(embed_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, states=None):
        """
        x: (batch, seq_len) — character indices
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        if states is None:
            states = [(torch.zeros(batch_size, self.hidden_size, device=x.device),
                       torch.zeros(batch_size, self.hidden_size, device=x.device))
                      for _ in range(self.num_layers)]
        
        embeds = self.embed(x)  # (batch, seq_len, embed_size)
        
        outputs = []
        for t in range(seq_len):
            inp = embeds[:, t, :]  # (batch, embed_size)
            new_states = []
            for i, cell in enumerate(self.cells):
                h_prev, c_prev = states[i]
                h_t, c_t = cell(inp, h_prev, c_prev)
                new_states.append((h_t, c_t))
                inp = h_t  # Feed to next layer
            states = new_states
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.fc(outputs)               # (batch, seq_len, vocab)
        return logits, states


# --- Data preparation (tiny Shakespeare) ---
def load_shakespeare():
    """Load a small text corpus. Replace with actual file path."""
    text = """
    First Citizen:
    Before we proceed any further, hear me speak.
    All:
    Speak, speak.
    First Citizen:
    You are all resolved rather to die than to famish?
    All:
    Resolved. resolved.
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, len(chars)
```

### 4.4 Training Loop

```python
def train_char_rnn(model, data, epochs=500, seq_len=64, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Random starting position
        start = torch.randint(0, len(data) - seq_len - 1, (1,)).item()
        x = data[start:start + seq_len].unsqueeze(0)        # (1, seq_len)
        y = data[start + 1:start + seq_len + 1].unsqueeze(0) # (1, seq_len)
        
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f}")

data, stoi, itos, vocab_size = load_shakespeare()
model = CharRNN(vocab_size, embed_size=32, hidden_size=128, num_layers=2)
train_char_rnn(model, data)
```

---

## 5. Exercises

### Exercise 1: Gradient Norm Experiment

This is the most important experiment of the day. Measure how gradients vanish:

```python
def gradient_norm_experiment(hidden_size=64, seq_lengths=[10, 50, 100, 200]):
    """Measure |∂L/∂h₁| vs |∂L/∂hₜ| for increasing sequence lengths."""
    results = {}
    
    for T in seq_lengths:
        rnn = VanillaRNNCell(hidden_size, hidden_size)
        x = torch.randn(1, T, hidden_size)
        h = torch.zeros(1, hidden_size, requires_grad=True)
        
        # Forward pass — save all hidden states
        hiddens = [h]
        for t in range(T):
            h = rnn(x[:, t, :], h)
            hiddens.append(h)
        
        # Loss on final hidden state
        loss = hiddens[-1].sum()
        loss.backward()
        
        # Measure gradient norm at first hidden state
        grad_norm = hiddens[0].grad.norm().item() if hiddens[0].grad is not None else 0
        results[T] = grad_norm
        print(f"T={T:3d} | |∂L/∂h₁| = {grad_norm:.6f}")
    
    return results

# Run and plot
results = gradient_norm_experiment()
```

**Expected**: Gradient norms decrease exponentially with sequence length. At $T = 200$, the gradient at $h_1$ is essentially zero.

### Exercise 2: LSTM vs RNN Gradient Flow

Repeat Exercise 1 with LSTM. Compare the gradient norms. Does LSTM maintain gradients over longer sequences?

### Exercise 3: Forget Gate Bias Initialization

LSTM forget gates are often initialized with a bias of 1.0 (Jozefowicz et al., 2015) so that $f_t \approx \sigma(1) \approx 0.73$ initially — biasing toward *remembering*. Implement this:

```python
# After creating the LSTM cell:
for name, param in lstm_cell.named_parameters():
    if 'bias' in name:
        n = param.size(0) // 4
        param.data[n:2*n].fill_(1.0)  # forget gate bias = 1
```

Train with and without this initialization. Does it help?

### Exercise 4: Generate Shakespeare

After training the CharRNN, implement autoregressive generation:

```python
@torch.no_grad()
def generate(model, itos, stoi, seed_text="First", max_len=200, temperature=0.8):
    model.eval()
    chars = list(seed_text)
    x = torch.tensor([[stoi[ch] for ch in chars]])
    states = None
    
    # Process seed
    logits, states = model(x, states)
    
    # Generate
    for _ in range(max_len):
        last_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_logits, dim=0)
        next_char_idx = torch.multinomial(probs, 1).item()
        chars.append(itos[next_char_idx])
        
        x = torch.tensor([[next_char_idx]])
        logits, states = model(x, states)
    
    return ''.join(chars)
```

What happens with $\text{temperature} \to 0$ vs $\text{temperature} \to \infty$?

---

## 6. Key Takeaways

- **Vanilla RNN**: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$ — simple but fatally flawed
- **Vanishing gradients**: $\|\frac{\partial h_T}{\partial h_t}\| \leq \sigma_{\max}(W_{hh})^{T-t}$ — exponential decay prevents long-range learning
- **LSTM** adds a **cell state** with **gates** that control information flow:
  - Forget gate: what to erase
  - Input gate: what to write
  - Output gate: what to expose
- The cell state gradient $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ can stay near 1 — **gradient highway** through time
- LSTM's cell state is exactly analogous to ResNet's skip connection — both create paths for unimpeded gradient flow
- **Gradient clipping** fixes exploding gradients; **gating** fixes vanishing gradients
- Despite LSTM's improvements, the fundamental bottleneck remains: the entire history must be compressed into a fixed-size vector. Tomorrow's lesson makes this bottleneck painfully explicit.

---

## 7. The Thread: Compression = Prediction = Intelligence

The vanishing gradient problem is fundamentally a **compression failure**. The hidden state $h_t$ tries to compress the *entire* history $(x_1, \ldots, x_t)$ into a fixed-size vector. As the sequence grows, information from early tokens gets overwritten — the compression becomes too lossy.

LSTM improves this by making the compression *selective* (gates choose what to keep), but it's still a fixed-capacity bottleneck. The real breakthrough comes when we abandon the idea of a single compressed state and instead allow the model to **dynamically attend** to any part of the history. That's attention, and it arrives in Day 10.

The deep lesson: **a good compressor must be adaptive**. Fixed-rate compression (RNN) loses information. Variable-rate, content-dependent compression (attention) preserves what matters. This principle will echo through the entire curriculum — from LSTM gates to attention weights to VLA action tokens.

---

## 8. Further Reading

- **Hochreiter & Schmidhuber, "Long Short-Term Memory"** (1997) — The original LSTM paper
- **Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder"** (2014) — GRU introduction
- **Pascanu et al., "On the difficulty of training Recurrent Neural Networks"** (2013) — Formal analysis of vanishing/exploding gradients
- **Jozefowicz et al., "An Empirical Exploration of Recurrent Network Architectures"** (2015) — LSTM initialization tricks
- **Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks"** (2015) — The classic blog post with char-RNN
- **Olah, "Understanding LSTM Networks"** (2015) — The best visual explanation of LSTM gates
