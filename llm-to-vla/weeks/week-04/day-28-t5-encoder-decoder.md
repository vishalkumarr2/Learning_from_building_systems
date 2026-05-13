# Day 28: T5 & Encoder-Decoder LMs

> **Phase II — Attention, Transformers & Scaling | Week 4 | 2.5 hours**
> *"Every problem is a text-to-text problem, if you look at it right." — Colin Raffel*

## Navigation
- **Previous:** [Day 27: Sampling & Generation](day-27-sampling-generation.md)
- **Next:** [Day 29: Phase II Capstone Day 1](../week-05/day-29-phase2-capstone-1.md)
- **Week:** [Week 4 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/04-transformer-variants.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 28.1 The Text-to-Text Framework

T5 (Text-to-Text Transfer Transformer) proposed a radical unification: **every NLP task is text-in, text-out.**

```
Traditional approach (task-specific heads):

  Input ──→ [Shared Encoder] ──→ [Classification Head] ──→ Label
                               └→ [Regression Head]    ──→ Number
                               └→ [Seq2Seq Head]       ──→ Text

T5 approach (one format):

  "translate English to German: The house is wonderful" ──→ T5 ──→ "Das Haus ist wunderbar"
  "summarize: <long article>"                           ──→ T5 ──→ "<summary>"
  "sst2 sentence: I loved this movie"                   ──→ T5 ──→ "positive"
  "cola sentence: The dog ran quickly"                  ──→ T5 ──→ "acceptable"
```

**Why this matters:** The task instruction is *part of the input*. The model learns what to do from the text prefix. This is a precursor to instruction-following LLMs (ChatGPT, etc.).

### 28.2 T5 Architecture

T5 uses the original encoder-decoder transformer architecture (Vaswani et al., 2017):

```
T5 Architecture:

  Input tokens                           Target tokens
      │                                      │
      ▼                                      ▼
  [Embedding]                           [Embedding]
      │                                      │
      ▼                                      ▼
  ┌──────────────┐                   ┌──────────────────┐
  │  Encoder      │                   │  Decoder          │
  │  Block ×N     │                   │  Block ×N         │
  │               │                   │                    │
  │  Self-Attn    │     ┌────────────→│  Masked Self-Attn │
  │  (bidirect.)  │     │             │  Cross-Attention   │←─ attends to encoder
  │  FFN          │     │             │  FFN               │
  │  LayerNorm    │─────┘             │  LayerNorm         │
  └──────────────┘                   └──────────────────┘
                                          │
                                          ▼
                                     [Linear + Softmax]
                                          │
                                          ▼
                                     Output tokens
```

Three types of attention in T5:

| Attention Type | Query | Key/Value | Masking |
|---------------|-------|-----------|---------|
| Encoder self-attention | Encoder | Encoder | None (bidirectional) |
| Decoder self-attention | Decoder | Decoder | Causal (can only see past) |
| Cross-attention | Decoder | Encoder | None (can see all input) |

### 28.3 The Three Architectures Compared

This is the most important comparison in modern NLP:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ENCODER-ONLY (BERT)           DECODER-ONLY (GPT)                │
│                                                                   │
│  Input: "The [MASK] sat"       Input: "The cat"                  │
│         ↓                              ↓                         │
│  ┌─────────────┐               ┌──────────────┐                 │
│  │ Bidirectional│               │ Causal (L→R) │                 │
│  │ Self-Attn    │               │ Self-Attn    │                 │
│  └─────────────┘               └──────────────┘                 │
│         ↓                              ↓                         │
│  Output: "cat"                  Output: "sat"                    │
│  (fill in blank)                (predict next)                   │
│                                                                   │
│  ENCODER-DECODER (T5)                                            │
│                                                                   │
│  Input: "translate: The cat sat" ──→ [Encoder] ──→ [Decoder]    │
│                                                       ↓          │
│                                            Output: "Le chat..."  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

| Aspect | BERT (Enc-only) | GPT (Dec-only) | T5 (Enc-Dec) |
|--------|:--:|:--:|:--:|
| Pre-training | Masked LM | Autoregressive LM | Span corruption |
| Bidirectional context | ✅ Yes | ❌ No (causal) | ✅ Encoder, ❌ Decoder |
| Generation | ❌ Poor | ✅ Excellent | ✅ Good |
| Understanding | ✅ Excellent | ⚠️ Good | ✅ Excellent |
| Few-shot learning | ❌ Needs fine-tuning | ✅ In-context | ⚠️ Mixed |
| Parameters (for same quality) | Smallest | Largest | Medium |
| Won in practice | NLU tasks (2018-2020) | Everything (2022+) | Seq2seq tasks |

### 28.4 Why Decoder-Only Won for Generation

GPT-style models dominate today. Why?

1. **Simpler architecture:** One stack of layers, not two. Easier to scale.
2. **Unified pre-training:** Next-token prediction on everything. No need for special denoising objectives.
3. **Scaling efficiency:** For a given parameter count, one big decoder > two smaller stacks.
4. **In-context learning:** Decoder-only models excel at few-shot learning from the prompt.
5. **KV cache:** More efficient autoregressive generation (only cache, no re-encode).

### 28.5 When Encoder-Decoder Still Wins

T5-style models remain better for specific tasks:

1. **Translation:** The encoder processes the full source sentence bidirectionally before the decoder generates the target. This bidirectional encoding captures source context that causal decoders miss.

2. **Summarization:** Full bidirectional understanding of the source document is crucial for selecting what to include.

3. **Structured prediction:** When the output is significantly shorter than and structurally different from the input.

4. **Conditional generation with complex inputs:** When you need to deeply process the "condition" (image, document, etc.) before generating.

**The VLA connection:** Many VLAs use encoder-decoder–like structures! The vision encoder processes the image bidirectionally, and the action decoder generates actions autoregressively. This is essentially a T5 architecture where the "encoder" is a ViT and the "decoder" generates actions instead of text.

### 28.6 T5's Pre-Training: Span Corruption

T5 doesn't use next-token prediction or masked LM. Instead, it uses **span corruption**:

```
Original:  "The quick brown fox jumps over the lazy dog"

Corrupted: "The <X> brown fox <Y> the lazy dog"
                 ↓
Target:    "<X> quick <Y> jumps over"
```

- Random contiguous spans are replaced with sentinel tokens `<X>`, `<Y>`, etc.
- The model must reconstruct the missing spans
- This teaches both understanding (encoder must grasp context) and generation (decoder must produce text)

Mean span length of 3 tokens, corruption rate of 15% worked best in T5's ablation study.

---

## Implementation (60 min)

### Using T5 for Summarization

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load T5-small (60M params — fast for experimentation)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

def summarize_with_t5(text: str, max_length: int = 150) -> str:
    """Summarize text using T5's text-to-text format."""
    # T5 expects the task prefix
    input_text = f"summarize: {text}"
    
    input_ids = tokenizer.encode(
        input_text, 
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,           # beam search for quality
            length_penalty=2.0,    # encourage longer summaries
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Example
article = """
Scaling laws for neural language models show that performance improves 
predictably as a power law with increased compute, data, and parameters. 
The Chinchilla paper demonstrated that most large language models were 
significantly undertrained, recommending approximately 20 training tokens 
per parameter for compute-optimal training. This finding reshaped how 
the industry approaches model training, leading to models like LLaMA 
that deliberately over-train smaller models on more data to optimize 
inference-time efficiency rather than training-time efficiency.
"""

summary = summarize_with_t5(article)
print(f"Summary: {summary}")
```

### Using T5 for Translation

```python
def translate_with_t5(text: str, source: str = "English", target: str = "German"):
    """Translate text using T5."""
    input_text = f"translate {source} to {target}: {text}"
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Examples
sentences = [
    "The cat sits on the mat.",
    "Scaling laws predict model performance.",
    "Robots must learn from experience.",
]

for sent in sentences:
    translation = translate_with_t5(sent)
    print(f"  EN: {sent}")
    print(f"  DE: {translation}\n")
```

### Comparing T5 vs GPT-2 on the Same Task

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def summarize_with_gpt2(text: str, max_new_tokens: int = 100) -> str:
    """Attempt summarization with GPT-2 using prompting."""
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    
    # GPT-2 doesn't have a task prefix — use prompting
    prompt = f"Article: {text}\n\nSummary:"
    input_ids = gpt2_tok.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,
        )
    
    full_text = gpt2_tok.decode(output_ids[0], skip_special_tokens=True)
    # Extract just the generated summary
    summary = full_text[len(prompt):]
    return summary.strip()


# Compare on the same article
print("=== T5 Summary ===")
print(summarize_with_t5(article))
print("\n=== GPT-2 Summary ===")
print(summarize_with_gpt2(article))
```

### Visualizing Cross-Attention

```python
def visualize_cross_attention(model, tokenizer, input_text: str, target_text: str):
    """Visualize which input tokens the decoder attends to."""
    import matplotlib.pyplot as plt
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    target_ids = tokenizer.encode(target_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=target_ids,
            output_attentions=True,
        )
    
    # Cross-attention from last decoder layer, first head
    cross_attn = outputs.cross_attentions[-1][0, 0].numpy()  # (dec_len, enc_len)
    
    # Token labels
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    target_tokens = tokenizer.convert_ids_to_tokens(target_ids[0])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(cross_attn, cmap="Blues", aspect="auto")
    
    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(input_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens, fontsize=8)
    
    ax.set_xlabel("Encoder (Input)")
    ax.set_ylabel("Decoder (Output)")
    ax.set_title("Cross-Attention: What does the decoder look at?")
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("cross_attention.png", dpi=150)
    plt.show()

# visualize_cross_attention(model, tokenizer,
#     "translate English to German: The cat sits on the mat",
#     "Die Katze sitzt auf der Matte")
```

---

## Exercise (45 min)

### E28.1 — Architecture Comparison Table

Fill in this table from your understanding (not from looking it up):

| Feature | GPT (Decoder) | BERT (Encoder) | T5 (Enc-Dec) |
|---------|:---:|:---:|:---:|
| Pre-training objective | ? | ? | ? |
| Bidirectional input context | ? | ? | ? |
| Can generate text | ? | ? | ? |
| Few-shot capable | ? | ? | ? |
| Parameter efficiency | ? | ? | ? |
| Best for classification | ? | ? | ? |
| Best for translation | ? | ? | ? |
| Best for open-ended generation | ? | ? | ? |

### E28.2 — When Would You Choose T5?

List 3 tasks where an encoder-decoder architecture is strictly better than decoder-only:

1. **Task:** ___  
   **Why:** ___

2. **Task:** ___  
   **Why:** ___

3. **Task:** ___  
   **Why:** ___

**Hint:** Think about tasks where the output is much shorter than the input, where bidirectional understanding of the input is critical, or where the output has different structure than the input.

### E28.3 — VLA Architecture Connection

Answer: "Is a VLA more like GPT, BERT, or T5? Why?"

Consider:
- The image encoder processes visual input bidirectionally (like BERT/T5 encoder)
- The action decoder generates actions autoregressively (like GPT/T5 decoder)
- Cross-attention connects vision to action (like T5 cross-attention)

Draw the VLA as a T5-like architecture:
```
Image ──→ [Vision Encoder] ──→ cross-attn ──→ [Action Decoder] ──→ Actions
                                    ↑
Text  ──→ [Language Model ] ────────┘
```

---

## Key Takeaways

1. **Text-to-text unifies all NLP tasks** — every task becomes "input text → output text" with the right prefix
2. **Three transformer architectures:** Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5)
3. **Decoder-only won for generation** — simpler, scales better, supports in-context learning
4. **Encoder-decoder wins for seq2seq** — translation, summarization, and tasks where input ≠ output structure
5. **VLAs are encoder-decoder models** — vision encoder + action decoder connected by cross-attention
6. **Cross-attention is the bridge** — it lets the decoder "look at" the encoded input, essential for conditioned generation

## Connection to the Thread

T5's encoder-decoder architecture is the hidden blueprint of VLAs. When RT-2 processes an image through a ViT encoder and generates actions through a decoder, it's running a T5-like architecture where the "text" is robot commands. Understanding when and why encoder-decoder works better than decoder-only helps you reason about VLA design choices.

## Further Reading

- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020) — the T5 paper
- Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training" (2020) — another encoder-decoder approach
- Tay et al., "Unifying Language Learning Paradigms" (2022) — UL2, unifying all three approaches
- Wang et al., "What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?" (2022)
