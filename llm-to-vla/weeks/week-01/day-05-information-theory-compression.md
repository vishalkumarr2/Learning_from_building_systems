# Day 5: Information Theory & Compression — THE THREAD

> **Phase I — DL Foundations & Information Theory | Week 1 | 2.5 hours**
> *"Compression and intelligence are the same thing." — Ilya Sutskever*

## Navigation
- **Previous:** [Day 4: Seq2Seq & The Bottleneck](day-04-seq2seq-bottleneck.md)
- **Next:** [Day 6: Embeddings & Representation Learning](day-06-embeddings-representations.md)
- **Week:** [Week 1: DL Foundations](../README.md)
- **Phase:** [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

> ⚡ **This is the most important theoretical day in the curriculum.**
> Everything that follows — attention, transformers, scaling laws, emergence,
> multimodal fusion — connects back to the ideas introduced today.

---

## Theory (45 min)

### 5.1 Shannon Entropy — The Fundamental Limit

Claude Shannon (1948) asked: **what is the minimum number of bits needed to encode
messages from a source?** The answer is the **entropy**:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

**Intuition:** Entropy measures **surprise**. If an outcome is certain ($p = 1$), there
is zero surprise, zero entropy. If all outcomes are equally likely, surprise — and
entropy — is maximized.

| Distribution | Entropy | Intuition |
|---|---|---|
| Fair coin: $p(H) = p(T) = 0.5$ | $H = 1$ bit | Maximum uncertainty for 2 outcomes |
| Biased coin: $p(H) = 0.99$ | $H = 0.08$ bits | Almost no surprise |
| Fair die (6 sides) | $H = 2.58$ bits | $\log_2 6$ |
| English text (per character) | $H \approx 1.0$ bit | Highly structured, redundant |
| Random bytes | $H = 8$ bits | No structure to exploit |

**Key property:** You **cannot** compress data below its entropy, on average. Any
lossless encoding must use at least $H(X)$ bits per symbol. This is Shannon's
**source coding theorem**.

### 5.2 Cross-Entropy — How Well Does Your Model Compress?

Now suppose you don't know the true distribution $p$ — you have a model $q$ instead.
If you design a code optimized for $q$ but data comes from $p$, you waste bits.
The **cross-entropy** measures the average code length:

$$H(p, q) = -\sum_{x} p(x) \log_2 q(x)$$

**Critical fact:**

$$H(p, q) \geq H(p)$$

Equality holds if and only if $q = p$. The model's code is always at least as long
as the optimal code. The gap between them is the KL divergence.

**Why cross-entropy is the "right" loss function for language models:**

When we train a language model with cross-entropy loss, we are literally asking:

> "How many bits does your model need to encode each token, on average?"

Minimizing cross-entropy = building a better compressor = learning the true distribution.

```
 True distribution p(x) ─────────────── H(p) bits (optimal, unreachable)
                                              │
                                   gap = D_KL(p||q) ← "wasted bits"
                                              │
 Model distribution q(x) ───────────── H(p,q) bits (our model's cost)
```

### 5.3 KL Divergence — The Wasted Bits

The **Kullback-Leibler divergence** quantifies how many extra bits $q$ wastes:

$$D_{KL}(p \| q) = \sum_{x} p(x) \log_2 \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Properties:
- $D_{KL}(p \| q) \geq 0$ (Gibbs' inequality)
- $D_{KL}(p \| q) = 0$ iff $p = q$
- **Not symmetric:** $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general

**The decomposition that explains everything:**

$$\underbrace{H(p, q)}_{\text{cross-entropy loss}} = \underbrace{H(p)}_{\text{irreducible noise}} + \underbrace{D_{KL}(p \| q)}_{\text{model's ignorance}}$$

When we train a model, $H(p)$ is fixed (property of the data). Minimizing
$H(p, q)$ minimizes $D_{KL}(p \| q)$ — pushing $q$ toward $p$.

### 5.4 Perplexity — Entropy's Friendly Face

Perplexity is the exponential of cross-entropy:

$$\text{PPL} = 2^{H(p, q)}$$

**Interpretation:** A perplexity of 100 means the model is "as confused as if it were
choosing uniformly among 100 options" at each step.

| Model | Perplexity (PTB) | Bits/token | Era |
|---|---|---|---|
| Unigram baseline | ~960 | 9.9 | Static |
| Trigram + KN | ~140 | 7.1 | 1990s |
| LSTM | ~58 | 5.9 | 2016 |
| GPT-2 (1.5B) | ~22 | 4.5 | 2019 |
| GPT-3 (175B) | ~10 | 3.3 | 2020 |
| Human (Shannon game) | ~7-12 | 2.8-3.6 | Biological |

The trend: **larger models compress better**. This is the empirical foundation
of scaling laws (Phase II, Day 22).

### 5.5 The Central Thesis: Compression = Prediction = Intelligence

Here is the conceptual thread that runs through this entire curriculum:

**Claim 1: Prediction is compression.**
If you can predict the next token with probability $q(x_t | x_{<t})$, you can build
an arithmetic code that uses $-\log_2 q(x_t | x_{<t})$ bits for that token. Better
prediction → fewer bits → better compression.

**Claim 2: Compression requires understanding.**
To compress English text well, you must understand grammar, semantics, world knowledge,
and reasoning. A model that achieves 1.0 bits/character on English **must** "understand"
English in a meaningful sense. You cannot shortcut this.

**Claim 3 (Hutter's thesis): Intelligence is compression.**
Marcus Hutter's formal definition: the most intelligent agent is the one that achieves
the best compression of its observations. The **Hutter Prize** offers €500,000 for
compressing a 1GB Wikipedia dump — because doing so requires capturing human knowledge.

```
 ┌────────────────────────────────────────────────────────────────┐
 │                  THE COMPRESSION THREAD                        │
 │                                                                │
 │  Cross-entropy loss ──▶ Compression objective                  │
 │       ↓                                                        │
 │  Scaling laws ────────▶ Bigger models compress better          │
 │       ↓                                                        │
 │  Emergence ───────────▶ Compression forces abstraction         │
 │       ↓                                                        │
 │  Multimodal ──────────▶ Compress vision + language jointly     │
 │       ↓                                                        │
 │  VLAs ────────────────▶ Compress perception + action jointly   │
 └────────────────────────────────────────────────────────────────┘
```

### 5.6 Mutual Information (Preview)

For multimodal models (Phase III), we will need **mutual information**:

$$I(X; Y) = H(X) - H(X|Y) = D_{KL}(p(x, y) \| p(x) p(y))$$

This measures how much knowing $Y$ reduces uncertainty about $X$. A VLA that aligns
vision and language has high $I(\text{image}; \text{text})$ — seeing the image tells
you a lot about the text description, and vice versa. CLIP (Day 30) maximizes this.

---

## Implementation (60 min)

### 5.7 Hands-On: Entropy and Compression in Practice

```python
import torch
import math
import collections
from typing import Dict

# ── 5.7.1 Compute entropy of text ───────────────────────────
def char_entropy(text: str) -> float:
    """Compute Shannon entropy of a character distribution."""
    freq = collections.Counter(text)
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# Test on different texts
random_text = "".join(chr(ord('a') + (i * 7 + 3) % 26) for i in range(1000))
english_text = ("to be or not to be that is the question whether tis nobler "
                "in the mind to suffer the slings and arrows of outrageous "
                "fortune or to take arms against a sea of troubles " * 5)
repetitive = "abababababababababababababababababababababababababab" * 20

print("Character-level entropy (bits/char):")
print(f"  Random-ish:   {char_entropy(random_text):.3f}")
print(f"  English:      {char_entropy(english_text):.3f}")
print(f"  Repetitive:   {char_entropy(repetitive):.3f}")
print(f"  Theoretical max (26 chars): {math.log2(26):.3f}")


# ── 5.7.2 Cross-entropy as compression ──────────────────────
def cross_entropy_bits(true_dist: Dict, model_dist: Dict) -> float:
    """Compute cross-entropy H(p, q) in bits."""
    ce = 0.0
    for symbol, p in true_dist.items():
        q = model_dist.get(symbol, 1e-10)  # Avoid log(0)
        ce -= p * math.log2(q)
    return ce

def kl_divergence(true_dist: Dict, model_dist: Dict) -> float:
    """Compute KL divergence D_KL(p || q) in bits."""
    h_p = -sum(p * math.log2(p) for p in true_dist.values() if p > 0)
    h_pq = cross_entropy_bits(true_dist, model_dist)
    return h_pq - h_p

# True English letter frequencies (approximate)
english_freq = {
    'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
    'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043,
    'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.024,
    'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019, 'b': 0.015,
    'v': 0.010, 'k': 0.008, 'j': 0.002, 'x': 0.002, 'q': 0.001,
    'z': 0.001,
}

# Uniform model (naive)
uniform_dist = {c: 1/26 for c in english_freq}

# Slightly better model (knows vowels are common)
better_dist = {c: 0.08 if c in 'aeiou' else 0.024 for c in english_freq}
total_b = sum(better_dist.values())
better_dist = {c: p/total_b for c, p in better_dist.items()}

h_p = -sum(p * math.log2(p) for p in english_freq.values() if p > 0)
print(f"\nEntropy of English letters:       H(p) = {h_p:.3f} bits")
print(f"Cross-entropy (uniform model):    H(p,q) = {cross_entropy_bits(english_freq, uniform_dist):.3f} bits")
print(f"Cross-entropy (vowel-aware):      H(p,q) = {cross_entropy_bits(english_freq, better_dist):.3f} bits")
print(f"KL divergence (uniform):          D_KL = {kl_divergence(english_freq, uniform_dist):.3f} bits")
print(f"KL divergence (vowel-aware):      D_KL = {kl_divergence(english_freq, better_dist):.3f} bits")


# ── 5.7.3 Huffman coding: making compression tangible ───────
import heapq

def huffman_codes(freq: Dict[str, float]) -> Dict[str, str]:
    """Build Huffman codes from a frequency distribution."""
    heap = [(f, c) for c, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        return {heap[0][1]: "0"}

    while len(heap) > 1:
        f1, left = heapq.heappop(heap)
        f2, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, (left, right)))

    codes = {}
    def traverse(node, prefix=""):
        if isinstance(node, str):
            codes[node] = prefix or "0"
        else:
            traverse(node[0], prefix + "0")
            traverse(node[1], prefix + "1")

    traverse(heap[0][1])
    return codes

codes = huffman_codes(english_freq)
avg_len = sum(english_freq[c] * len(codes[c]) for c in english_freq)

print(f"\nHuffman coding results:")
print(f"  Average code length: {avg_len:.3f} bits/char")
print(f"  Entropy (lower bound): {h_p:.3f} bits/char")
print(f"  Fixed-length (upper bound): {math.log2(26):.3f} bits/char")
print(f"  Compression ratio: {avg_len / math.log2(26):.1%}")
print(f"\nTop-5 codes:")
for c in sorted(codes, key=lambda x: len(codes[x]))[:5]:
    print(f"  '{c}' (p={english_freq[c]:.3f}) → {codes[c]} ({len(codes[c])} bits)")


# ── 5.7.4 Neural model as compressor ────────────────────────
def neural_cross_entropy(text: str, model, tokenizer) -> float:
    """
    Compute bits-per-character for a neural language model.
    (Pseudocode — requires a trained model)
    """
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens[:-1]])
    targets = torch.tensor([tokens[1:]])

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

    # Gather log-probs of actual next tokens
    token_log_probs = log_probs[0, range(len(tokens)-1), targets[0]]
    bits_per_token = -token_log_probs.mean().item() / math.log(2)

    # Convert to bits per character
    chars_per_token = len(text) / len(tokens)
    bits_per_char = bits_per_token / chars_per_token
    return bits_per_char

# With a real GPT-2 model, you'd see ~0.9 bits/char on English
# Shannon estimated humans at ~1.0 bits/char
# GPT compresses BETTER than humans!
```

---

## Exercise (45 min)

1. **Entropy of different languages:** Compute character-level entropy for English,
   Python code, DNA sequences ("ATCG"), and random bytes. Rank them. Which has the
   most structure to exploit?

2. **Verify the decomposition:** Using the English letter frequencies above, verify
   numerically that $H(p, q) = H(p) + D_{KL}(p \| q)$ for the uniform model. Show
   your arithmetic.

3. **The GPT compression puzzle:** GPT-2 achieves ~0.93 bits/char on English text.
   Shannon estimated humans at ~1.0 bits/char. If GPT compresses **better** than
   humans, does that mean GPT "understands" English better than humans? Write a
   paragraph arguing yes, and a paragraph arguing no. Which do you find more convincing?

4. **Perplexity calculation:** A language model assigns the following probabilities to
   a 5-token sequence: $[0.1, 0.3, 0.8, 0.05, 0.6]$. Compute:
   - The log-likelihood
   - The cross-entropy (bits per token)
   - The perplexity
   Show your work step by step.

---

## Key Takeaways

- **Entropy** $H(X)$ is the minimum bits needed to encode messages — the floor of
  compression
- **Cross-entropy** $H(p, q)$ is the bits your model actually uses — the training loss
- **KL divergence** $D_{KL}(p \| q)$ is the gap — the wasted bits your model can
  still eliminate
- **Perplexity** $= 2^{H(p,q)}$ — the "effective vocabulary size" of uncertainty
- **Minimizing cross-entropy = maximizing compression = learning the data distribution**
- **The central thesis: compression = prediction = intelligence**

## Connection to the Thread

Today establishes **THE** conceptual thread of the entire curriculum:

1. **Cross-entropy loss** (every model you train) is a **compression objective**
2. **Scaling laws** (Day 22) show loss decreases as a power law with compute —
   bigger models are better compressors
3. **Emergence** (Day 27) happens when compression forces the model to discover
   abstract concepts — it learns grammar, facts, reasoning because these enable
   shorter codes
4. **Multimodal models** (Phase III) compress vision + language jointly — CLIP
   maximizes mutual information between modalities
5. **VLAs** (Phase VI) compress perception + language + action into a unified
   representation — the ultimate compression challenge

Remember this thread. It connects everything.

## Further Reading

- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System
  Technical Journal.*
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory.* 2nd ed.
  Chapters 1-5.
- Hutter, M. (2005). *Universal Artificial Intelligence.* Springer. — The formal
  argument for compression as intelligence.
- Shannon, C. E. (1951). "Prediction and Entropy of Printed English." *Bell System
  Technical Journal.* — The original ~1 bit/char estimate.
- Deletang, G., et al. (2024). "Language Modeling Is Compression." *ICLR 2024.* —
  Modern empirical validation of the thesis.
