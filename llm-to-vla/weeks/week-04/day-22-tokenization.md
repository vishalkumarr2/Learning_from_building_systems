# Day 22: Tokenization Deep Dive — From Characters to Subwords

> **Phase II — Attention, Transformers & Scaling | Week 4 | 2.5 hours**
> *"How many Rs in strawberry? If you can't answer that, your tokenizer is showing." — @karpathy*

## Navigation
- **Previous:** [Day 21: BERT & Masked LM](../week-03/day-21-bert-masked-lm.md)
- **Next:** [Day 23: GPT & nanoGPT](day-23-gpt-nanogpt.md)
- **Week:** [Week 4 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/03-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 22.1 The Tokenization Problem

Models don't see text. They see integer sequences. How you convert text → integers has profound consequences:

```
Text:      "unhappiness"

Character: ['u','n','h','a','p','p','i','n','e','s','s']  → 11 tokens
Word:      ['unhappiness']                                  → 1 token (but vocab=∞)
Subword:   ['un', 'happiness']                              → 2 tokens ← sweet spot
BPE:       ['un', 'happ', 'iness']                          → 3 tokens (depends on training)
```

| Level | Vocab Size | Seq Length | OOV? | Morphology |
|-------|-----------|------------|------|------------|
| Character | ~256 | Very long | Never | Must learn from scratch |
| Word | 100K+ | Short | Yes, common | Captures full words |
| Subword (BPE) | 32K–100K | Medium | Rare | Learns morphemes |

### 22.2 Byte Pair Encoding (BPE) — Step by Step

BPE is the dominant tokenization algorithm. It's a compression algorithm repurposed for NLP.

**Algorithm:**

1. Start with a vocabulary of individual characters (or bytes)
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until desired vocabulary size is reached

```
Corpus: "low low low low low lowest lowest newer newer newer wider wider"

Step 0 — Character vocabulary:
  {'l','o','w','e','s','t','n','r','i','d',' '}

Step 0 — Tokenized corpus:
  l o w   l o w   l o w   l o w   l o w   l o w e s t   l o w e s t
  n e w e r   n e w e r   n e w e r   w i d e r   w i d e r

Step 1 — Most frequent pair: ('l', 'o') → count: 7
  Merge 'l'+'o' → 'lo'
  lo w   lo w   lo w   lo w   lo w   lo w e s t   lo w e s t
  n e w e r   n e w e r   n e w e r   w i d e r   w i d e r

Step 2 — Most frequent pair: ('lo', 'w') → count: 7
  Merge 'lo'+'w' → 'low'
  low   low   low   low   low   low e s t   low e s t
  n e w e r   n e w e r   n e w e r   w i d e r   w i d e r

Step 3 — Most frequent pair: ('e', 'r') → count: 5
  Merge 'e'+'r' → 'er'
  low   low   low   low   low   low e s t   low e s t
  n e w er   n e w er   n e w er   w i d er   w i d er

Step 4 — Most frequent pair: ('e', 'w') → count: 3
  Merge 'e'+'w' → 'ew'
  ...and so on until vocab size reached
```

### 22.3 WordPiece (BERT) vs BPE (GPT)

Both are subword tokenization but differ in the merge criterion:

**BPE:** Merges the most **frequent** pair.

**WordPiece:** Merges the pair that maximizes the **likelihood** of the training data:
$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This favors merging rare subwords that appear together, rather than just common pairs.

**In practice:** The difference is minor. BPE is more common in modern models.

### 22.4 SentencePiece and Byte-Level BPE

**Problem:** Standard BPE depends on pre-tokenization (splitting on spaces). This fails for languages without spaces (Chinese, Japanese) and creates language bias.

**SentencePiece (Kudo & Richardson, 2018):**
- Treats the input as a raw stream of characters (including spaces as `▁`)
- Language-agnostic — no pre-tokenization needed
- Used by LLaMA, T5, mBART

**Byte-Level BPE (GPT-2/3/4, tiktoken):**
- Start with 256 byte tokens (UTF-8 bytes)
- Build BPE merges on bytes, not characters
- Can represent ANY text (including code, emoji, non-Latin scripts)
- `tiktoken` is OpenAI's fast implementation

```python
# tiktoken example
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

text = "Hello, world! 🌍 こんにちは"
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded pieces: {[enc.decode([t]) for t in tokens]}")
# ['Hello', ',', ' world', '!', ' 🌍', ' ', 'こん', 'にち', 'は']
```

### 22.5 Tokenization Artifacts and Failure Modes

Tokenization creates **blindness** — the model can't reason about sub-token structure:

```
"How many Rs in strawberry?"

GPT-4 tokenization:
  ['How', ' many', ' Rs', ' in', ' str', 'aw', 'berry', '?']
                                         ↑
  'strawberry' is split into 'str' + 'aw' + 'berry'
  The model sees 3 tokens, not individual characters
  It cannot count letters because it never sees them!

Other artifacts:
  " Hello"  ≠  "Hello"     (leading space is a separate token)
  "1+1=2"   tokenized as:  ['1', '+', '1', '=', '2']   ← fine
  "12345"   tokenized as:  ['123', '45']                ← number split!
  
  Python code:
  "    if x:" → ['    ', 'if', ' x', ':']
  vs
  "  if x:"  → ['  ', 'if', ' x', ':']
  Different indentation = different first token!
```

**Why this matters for LLMs:**
- Arithmetic errors (numbers split unpredictably)
- Spelling/character-level tasks fail
- Multilingual efficiency varies wildly (English: ~1 token/word, Japanese: ~3 tokens/character)
- Code formatting sensitivity

### 22.6 Vocabulary Size Trade-offs

| Vocab Size | Sequence Length | Embedding Params | Quality |
|------------|----------------|-------------------|---------|
| 256 (bytes) | Very long | 256 × d | Must learn everything |
| 32K (GPT-2) | Medium | 32K × d (~25M) | Good general balance |
| 50K (BERT) | Shorter | 50K × d (~38M) | Better word coverage |
| 100K (GPT-4) | Shortest | 100K × d (~77M) | Best coverage, most params |

**Rule of thumb:** Larger vocab → shorter sequences → faster training, but more embedding parameters and harder to train rare tokens.

---

## Implementation (60 min)

### 22.7 BPE from Scratch (Karpathy's minbpe)

```python
"""
Minimal BPE implementation.
Based on Karpathy's minbpe: https://github.com/karpathy/minbpe
"""


def get_pair_counts(token_ids):
    """Count frequency of adjacent token pairs."""
    counts = {}
    for a, b in zip(token_ids, token_ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(token_ids, pair, new_id):
    """Replace all occurrences of pair with new_id."""
    result = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(token_ids[i])
            i += 1
    return result


class MinBPE:
    """Minimal BPE tokenizer."""

    def __init__(self):
        self.merges = {}     # (int, int) → int
        self.vocab = {}      # int → bytes

    def train(self, text, vocab_size=276):
        """Train BPE on text. Default: 276 = 256 bytes + 20 merges."""
        assert vocab_size >= 256, "Vocab must include all bytes"
        n_merges = vocab_size - 256

        # Convert text to bytes
        token_ids = list(text.encode("utf-8"))
        print(f"Training BPE: {len(token_ids)} bytes → {n_merges} merges")

        # Build initial vocab: byte → byte
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(n_merges):
            counts = get_pair_counts(token_ids)
            if not counts:
                break

            # Find most frequent pair
            best_pair = max(counts, key=counts.get)
            new_id = 256 + i

            # Merge
            token_ids = merge_pair(token_ids, best_pair, new_id)

            # Record
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if i < 10 or i % 50 == 0:
                piece = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  Merge {i}: {best_pair} → {new_id} "
                      f"('{piece}', count={counts[best_pair]})")

        print(f"Final: {len(token_ids)} tokens "
              f"(compression: {len(text.encode('utf-8'))/len(token_ids):.1f}x)")
        return self

    def encode(self, text):
        """Encode text to token IDs."""
        token_ids = list(text.encode("utf-8"))
        while len(token_ids) >= 2:
            counts = get_pair_counts(token_ids)
            # Find the pair with the lowest merge index (earliest learned)
            best = min(
                counts,
                key=lambda p: self.merges.get(p, float('inf'))
            )
            if best not in self.merges:
                break
            token_ids = merge_pair(token_ids, best, self.merges[best])
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        raw = b"".join(self.vocab[t] for t in token_ids)
        return raw.decode("utf-8", errors="replace")


# Train on a sample corpus
corpus = """
The transformer architecture has revolutionized natural language processing.
Attention is all you need. Self-attention allows each token to attend to every
other token in the sequence. Multi-head attention splits the attention into
multiple heads, each learning different patterns. The transformer uses
positional encodings to inject sequence order information.
""" * 100  # Repeat for more data

bpe = MinBPE()
bpe.train(corpus, vocab_size=356)  # 256 bytes + 100 merges

# Test encoding
test = "The transformer uses attention mechanisms."
encoded = bpe.encode(test)
decoded = bpe.decode(encoded)
print(f"\nOriginal:  '{test}'")
print(f"Encoded:   {encoded}")
print(f"N tokens:  {len(encoded)} (vs {len(test)} chars)")
print(f"Decoded:   '{decoded}'")
print(f"Roundtrip: {test == decoded}")

# Show the learned vocabulary pieces
print("\nLearned merges (first 20):")
for (a, b), new_id in list(bpe.merges.items())[:20]:
    piece = bpe.vocab[new_id].decode("utf-8", errors="replace")
    print(f"  {a} + {b} → {new_id} = '{piece}'")
```

### 22.8 Compare Tokenizers

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "unhappiness",
    "How many Rs in strawberry?",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "こんにちは世界",  # "Hello world" in Japanese
    "🎉🎊🎈",  # Emoji
    "3.14159265358979323846",  # Pi digits
]

print(f"{'Text':<50} {'tiktoken':>10} {'BPE':>10} {'Chars':>10}")
print("-" * 85)
for text in test_texts:
    n_tiktoken = len(enc.encode(text))
    n_bpe = len(bpe.encode(text))
    n_chars = len(text)
    display = text[:47] + "..." if len(text) > 50 else text
    print(f"{display:<50} {n_tiktoken:>10} {n_bpe:>10} {n_chars:>10}")

# Inspect tokenization of tricky cases
for text in ["strawberry", "How many Rs in strawberry?"]:
    tokens = enc.encode(text)
    pieces = [enc.decode([t]) for t in tokens]
    print(f"\ntiktoken({text!r}):")
    print(f"  Pieces: {pieces}")
    print(f"  IDs:    {tokens}")
```

---

## Exercise (45 min)

### E22.1 Compression Ratio Analysis

Train your MinBPE on different corpus types and measure compression:

```python
corpora = {
    "english": "The quick brown fox..." * 1000,
    "python_code": "def func(x):\n    return x * 2\n" * 1000,
    "numbers": "3141592653589793238462643383279" * 100,
    "repetitive": "aaabbbcccaaabbbccc" * 1000,
}

for name, text in corpora.items():
    bpe = MinBPE()
    bpe.train(text, vocab_size=356)
    encoded = bpe.encode(text)
    ratio = len(text.encode('utf-8')) / len(encoded)
    print(f"{name}: {ratio:.1f}x compression")
```

Which corpus compresses best? Why?

### E22.2 Tokenization Blindness Demo

Test multiple tokenizers on character-level tasks:

```python
# Task: Count occurrences of 'r' in these words
words = ["strawberry", "mirror", "raspberry", "error", "corridor"]

for word in words:
    tokens_gpt4 = enc.encode(word)
    pieces = [enc.decode([t]) for t in tokens_gpt4]
    actual_count = word.count('r')
    print(f"'{word}': actual r-count={actual_count}, "
          f"tokens={pieces}, n_tokens={len(tokens_gpt4)}")
    # Can the model count 'r's if it sees these pieces?
    # Does any piece split an 'r' from its context?
```

### E22.3 Vocabulary Efficiency Across Languages

Compare token count for the same meaning in different languages:

```python
sentences = {
    "English": "Artificial intelligence will transform the world.",
    "Japanese": "人工知能は世界を変革するでしょう。",
    "Chinese": "人工智能将改变世界。",
    "Russian": "Искусственный интеллект изменит мир.",
    "Arabic": "سيغير الذكاء الاصطناعي العالم.",
}

for lang, text in sentences.items():
    tokens = enc.encode(text)
    chars = len(text)
    ratio = len(tokens) / len(text.split()) if ' ' in text else len(tokens)
    print(f"{lang:10s}: {len(tokens):3d} tokens, {chars:3d} chars, "
          f"tokens/char={len(tokens)/chars:.2f}")
```

Why is this ratio so different across languages? What are the implications for multilingual LLMs?

---

## Key Takeaways

1. **BPE is a compression algorithm** — it learns the most efficient subword vocabulary for a given corpus
2. **Tokenization determines what the model "sees"** — it can never reason below the token level
3. **"Strawberry" problem:** Tokenization blindness means LLMs struggle with character-level tasks, arithmetic, and spelling
4. **Byte-level BPE (tiktoken)** handles any input text but creates efficiency imbalances across languages
5. **Vocabulary size is a trade-off:** larger vocab → shorter sequences (faster) but more embedding parameters and harder to learn rare tokens
6. **This preprocessing step profoundly affects model behavior** — it's not just an implementation detail

## Connection to the Thread

Tokenization is the invisible foundation under everything you've built. Every attention matrix, every embedding, every generation step operates on tokens — not characters or words. Understanding tokenization explains many "surprising" LLM failures (arithmetic, spelling, multilingual inconsistency). Tomorrow, you'll build GPT from scratch with nanoGPT, and you'll see tokenization in action as the first step of the entire pipeline.

## Further Reading

- **[Karpathy's minbpe](https://github.com/karpathy/minbpe)** — The clearest BPE implementation
- **[Karpathy's Tokenization Video](https://www.youtube.com/watch?v=zduSFxRajkE)** — 2+ hours, essential viewing
- **[SentencePiece Paper](https://arxiv.org/abs/1808.06226)** — Kudo & Richardson, 2018
- **[tiktoken](https://github.com/openai/tiktoken)** — OpenAI's fast BPE implementation
- **[BPE Original Paper](https://arxiv.org/abs/1508.07909)** — Sennrich et al., 2016 (BPE for NMT)
- **[Tokenization Effects](https://arxiv.org/abs/2404.01373)** — Analysis of tokenization artifacts
