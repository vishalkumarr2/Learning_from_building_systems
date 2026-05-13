# Day 4: Seq2Seq & The Bottleneck

> **Phase I — DL Foundations & Information Theory | Week 1 | 2.5 hours**
> *"You cannot compress 'War and Peace' into a single vector and expect to reconstruct it." — This frustration births attention.*

## Navigation
- **Previous:** [Day 3: RNN/LSTM Essentials](day-03-rnn-lstm-essentials.md)
- **Next:** [Day 5: Information Theory & Compression](day-05-information-theory-compression.md)
- **Week:** [Week 1: DL Foundations](../README.md)
- **Phase:** [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 4.1 The Encoder-Decoder Architecture

The seq2seq (sequence-to-sequence) model, introduced by Sutskever et al. (2014), was
the first neural architecture to tackle **variable-length input → variable-length output**
problems end-to-end. The canonical application: machine translation.

**Architecture overview:**

```
 Source: "the cat sat"          Target: "le chat assis"

 ┌─────┐   ┌─────┐   ┌─────┐
 │ the │──▶│ cat │──▶│ sat │──▶ c = h_T   (context vector)
 └─────┘   └─────┘   └─────┘       │
   ENCODER (processes input)        │
                                    ▼
                              ┌─────────┐
                              │  <SOS>  │──▶ "le"
                              └─────────┘     │
                                    ┌─────────┘
                                    ▼
                              ┌─────────┐
                              │   "le"  │──▶ "chat"
                              └─────────┘     │
                                    ┌─────────┘
                                    ▼
                              ┌─────────┐
                              │  "chat" │──▶ "assis"
                              └─────────┘     │
                                    ┌─────────┘
                                    ▼
                              ┌─────────┐
                              │ "assis" │──▶ <EOS>
                              └─────────┘
                     DECODER (generates output)
```

**Encoder** — An RNN (usually LSTM) reads the source sequence token by token and
produces a sequence of hidden states $h_1, h_2, \ldots, h_T$. Only the **final** hidden
state is kept:

$$c = h_T$$

This single vector $c$ is the **context vector** — the encoder's entire "summary" of the
input.

**Decoder** — A second RNN generates the target sequence one token at a time,
conditioned on the context vector $c$. At each step $t$:

$$s_t = f(s_{t-1}, y_{t-1}, c)$$
$$P(y_t | y_{<t}, x) = \text{softmax}(W_o \cdot s_t)$$

where $s_t$ is the decoder hidden state, $y_{t-1}$ is the previously generated token,
and $c$ is injected at every step (or just the first step, depending on the variant).

### 4.2 Teacher Forcing

During training, instead of feeding the decoder its own (possibly wrong) predictions,
we feed it the **ground-truth** previous token. This is called **teacher forcing**.

| Strategy | Input to decoder at step $t$ | Pros | Cons |
|---|---|---|---|
| Teacher forcing | Ground-truth $y_{t-1}^*$ | Stable gradients, fast convergence | Exposure bias: never sees own mistakes |
| Free running | Model's own $\hat{y}_{t-1}$ | Realistic at test time | Slow, error compounds |
| Scheduled sampling | Mix of both (anneal ratio) | Best of both worlds | Harder to tune |

**Exposure bias** is a real problem: during training the decoder always sees correct
prefixes, but at test time it must consume its own (noisy) outputs. Small errors
compound into catastrophic drift.

### 4.3 The Bottleneck Problem

Here is the central crisis of seq2seq. The context vector $c \in \mathbb{R}^d$ is a
**fixed-size** vector, typically $d = 256$ or $d = 512$. Every bit of information about
the source sentence must be squeezed through this single vector.

**The information-theoretic argument:**

A vector $c \in \mathbb{R}^{512}$ with 32-bit floats carries at most
$512 \times 32 = 16{,}384$ bits. A typical English sentence of 20 words at ~5 bits
per word (Shannon's estimate) requires ~100 bits — fine! But a paragraph of 200 words
needs ~1,000 bits, and a document of 5,000 words needs ~25,000 bits. The fixed vector
literally **cannot carry the information**.

```
 Information in source ──────────────▶ ████████████████████████████████████
                                       (grows with input length)

 Capacity of context c ──────────────▶ ██████████
                                       (fixed at d dimensions)

 When source > capacity ─────────────▶ INFORMATION LOSS! ⚠️
```

**Empirical signature:** BLEU score degrades sharply once input length exceeds ~20 tokens.
This was the key finding that motivated Bahdanau et al. (2015) to invent attention.

$$\text{BLEU}(n) \propto \frac{1}{n^\alpha} \quad \text{for } n > n_{\text{critical}}$$

### 4.4 Why the Bottleneck Matters for the Thread

This is not just a historical curiosity — it is a **compression failure**:

1. **Seq2seq tries to compress** the entire input into $c$ (a lossy compressor)
2. **The compression is too aggressive** — fixed capacity regardless of input complexity
3. **Attention (Day 10)** solves this by allowing the decoder to look at ALL encoder
   states, effectively giving it variable-bandwidth access to the source
4. **Transformers (Day 13)** take this further: every position attends to every other
   position — no bottleneck at all
5. **VLAs** must compress vision + language + proprioception into a shared
   representation — understanding bottlenecks is essential

> The history of deep learning architectures is largely a history of **removing
> information bottlenecks**.

---

## Implementation (60 min)

### 4.5 Hands-On: Build a Seq2Seq Model

We will build an encoder-decoder LSTM for a small number-reversal task (e.g.,
"1 2 3 4 5" → "5 4 3 2 1"), then **intentionally show it failing on long sequences**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# ── Hyperparameters ──────────────────────────────────────────
HIDDEN_DIM = 64        # Intentionally small to expose bottleneck
EMBED_DIM = 32
NUM_TOKENS = 12        # Digits 0-9 + <SOS> + <EOS>
SOS_TOKEN = 10
EOS_TOKEN = 11
MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data generation ──────────────────────────────────────────
def generate_pair(seq_len):
    """Generate a sequence and its reverse."""
    seq = [random.randint(0, 9) for _ in range(seq_len)]
    target = list(reversed(seq))
    return seq, target

def to_tensor(seq, add_eos=True):
    if add_eos:
        seq = seq + [EOS_TOKEN]
    return torch.tensor(seq, dtype=torch.long, device=DEVICE)


# ── Encoder ──────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        # src: (1, seq_len)
        embedded = self.embedding(src)           # (1, seq_len, embed_dim)
        _, (h_n, c_n) = self.lstm(embedded)      # h_n: (1, 1, hidden_dim)
        return h_n, c_n  # This IS the bottleneck — everything in h_n


# ── Decoder ──────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token, hidden, cell):
        # token: (1, 1)
        embedded = self.embedding(token)                  # (1, 1, embed_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))       # (1, vocab_size)
        return prediction, hidden, cell


# ── Seq2Seq wrapper ──────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[1]
        outputs = torch.zeros(trg_len, NUM_TOKENS, device=DEVICE)

        # Encode: squeeze entire source into (h, c)
        h, c = self.encoder(src)

        # First decoder input is <SOS>
        dec_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

        for t in range(trg_len):
            pred, h, c = self.decoder(dec_input, h, c)
            outputs[t] = pred
            top1 = pred.argmax(1)
            # Teacher forcing: use ground truth or own prediction
            if random.random() < teacher_forcing_ratio:
                dec_input = trg[:, t].unsqueeze(0)
            else:
                dec_input = top1.unsqueeze(0)

        return outputs


# ── Training ─────────────────────────────────────────────────
encoder = Encoder(NUM_TOKENS, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
decoder = Decoder(NUM_TOKENS, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train on SHORT sequences (len 3-8)
for epoch in range(2000):
    seq_len = random.randint(3, 8)
    src_seq, trg_seq = generate_pair(seq_len)
    src = to_tensor(src_seq, add_eos=False).unsqueeze(0)  # (1, seq_len)
    trg = to_tensor(trg_seq, add_eos=True).unsqueeze(0)   # (1, seq_len+1)

    outputs = model(src, trg, teacher_forcing_ratio=0.5)
    loss = criterion(outputs, trg.squeeze(0))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 4.6 Demonstrating the Bottleneck

```python
def evaluate(model, seq_len, num_trials=20):
    """Evaluate accuracy at a given sequence length."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(num_trials):
            src_seq, trg_seq = generate_pair(seq_len)
            src = to_tensor(src_seq, add_eos=False).unsqueeze(0)

            h, c = model.encoder(src)
            dec_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)
            predicted = []

            for _ in range(seq_len + 1):
                pred, h, c = model.decoder(dec_input, h, c)
                top1 = pred.argmax(1).item()
                if top1 == EOS_TOKEN:
                    break
                predicted.append(top1)
                dec_input = torch.tensor([[top1]], device=DEVICE)

            if predicted == trg_seq:
                correct += 1
    return correct / num_trials

# ── The bottleneck experiment ────────────────────────────────
print("\n📊 Accuracy vs. Sequence Length (the bottleneck!)")
print("─" * 40)
for length in [3, 5, 8, 10, 15, 20, 30]:
    acc = evaluate(model, length)
    bar = "█" * int(acc * 20)
    print(f"  len={length:2d}  acc={acc:.0%}  {bar}")
```

**Expected output** (your numbers will vary):

```
📊 Accuracy vs. Sequence Length (the bottleneck!)
────────────────────────────────────────
  len= 3  acc=100%  ████████████████████
  len= 5  acc=95%   ███████████████████
  len= 8  acc=70%   ██████████████
  len=10  acc=35%   ███████
  len=15  acc=5%    █
  len=20  acc=0%
  len=30  acc=0%
```

The model trained on lengths 3-8 utterly collapses beyond ~10 tokens. This is the
bottleneck in action: 64 hidden dimensions simply cannot store 20+ tokens of ordered
information.

---

## Exercise (45 min)

1. **Vary hidden dimension:** Re-run the experiment with `HIDDEN_DIM` = 32, 64, 128,
   256, 512. At what hidden size can the model handle length-20 sequences? How does
   this relate to the information capacity argument in §4.3?

2. **Bidirectional encoder:** Replace the encoder LSTM with a bidirectional LSTM
   (`bidirectional=True`). Does this help? Why or why not? (Hint: the bottleneck
   is in the *output* dimensionality, not the direction of processing.)

3. **BLEU score curve:** Implement a function that computes token-level BLEU-1 (unigram
   precision) and plot it against sequence length. You should see a curve similar to
   the one in Cho et al. (2014), Figure 2.

4. **Thought experiment:** If you could design a mechanism that lets the decoder "look
   back" at individual encoder hidden states $h_1, \ldots, h_T$ instead of just $h_T$,
   how would you weight which states to look at for each decoder step? Write pseudocode
   for your idea — you have just reinvented attention (Day 10).

---

## Key Takeaways

- **Seq2seq = encoder (compress) + decoder (decompress)** — the first end-to-end
  neural architecture for sequence transduction
- **The context vector $c = h_T$** carries ALL source information in a fixed-size vector
- **Teacher forcing** accelerates training but causes exposure bias
- **The bottleneck** causes systematic failure on long inputs — this is an information-
  theoretic limitation, not a training bug
- **The bottleneck is a compression failure** — too few bits for the message

## Connection to the Thread

The seq2seq bottleneck is the **first concrete failure of lossy compression** we
encounter in this curriculum. The model is forced to compress arbitrarily long inputs
into a fixed-size vector, and when input information exceeds the vector's capacity,
performance collapses.

This is not just a historical problem:
- **Attention** (Day 10) solves it by giving the decoder variable-bandwidth access
- **Transformers** (Day 13) eliminate sequential bottlenecks entirely
- **VLAs** face a modern version: how to compress high-dimensional visual input and
  language instructions into a representation that supports precise motor control

The thread: **every major architecture advance removes a compression bottleneck**.

## Further Reading

- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning
  with Neural Networks." *NeurIPS 2014.*
- Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder
  for Statistical Machine Translation." *EMNLP 2014.* — Fig. 2 shows the bottleneck.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly
  Learning to Align and Translate." *ICLR 2015.* — The paper that solved the bottleneck.
