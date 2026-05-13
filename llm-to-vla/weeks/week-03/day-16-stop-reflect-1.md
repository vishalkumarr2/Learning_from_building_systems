# Day 16: Stop & Reflect #1 — "Do You Really Understand Attention?"

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"The test of understanding is not whether you can follow a derivation, but whether you can produce one from scratch on a blank page."*

## Navigation
- **Previous:** [Day 15: Training a Transformer](day-15-training-transformer.md)
- **Next:** [Day 17: Efficient Attention](day-17-efficient-attention.md)
- **Week:** [Week 3 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Purpose

This is a **consolidation day**. No new implementation. You've spent six days (10–15) building from Bahdanau attention to training a full transformer. Today you verify that it's all solid in your head — not just recognized, but *retrievable*.

**Format:** Paper and pen. No notes, no code, no peeking. Work through each section cold. Then compare with your materials and identify gaps.

---

## Part 1: Re-Derive the Attention Equation (30 min)

### From first principles, on a blank page:

1. **State the problem:** Given a decoder state (query) and a set of encoder states (keys/values), how do we compute a context vector?

2. **Bahdanau (additive):**
   - Write the energy function $e_{ij}$
   - Write the attention weight formula $\alpha_{ij}$
   - Write the context vector formula $c_i$
   - How many learnable parameters? What are they?

3. **Scaled dot-product:**
   - Write $\text{Attention}(Q, K, V)$ in full
   - Why divide by $\sqrt{d_k}$? Derive the variance argument
   - What happens to softmax when scores are $\gg 1$?

4. **Multi-head:**
   - Write $\text{MultiHead}(Q, K, V)$ in full
   - What is $d_k$ per head if $d_{\text{model}} = 512$ and $h = 8$?
   - How many total parameters in the MHA block (with $W_O$)?

**Self-check:** If you got stuck on any step, that's a gap. Re-read the relevant day and try again tomorrow.

---

## Part 2: Draw the Full Transformer from Memory (30 min)

### On a blank page, draw the complete transformer architecture:

Include:
- [ ] Encoder block with all sublayers labeled
- [ ] Decoder block with all three sublayers labeled
- [ ] Residual connections (drawn as arrows)
- [ ] LayerNorm placement (Pre-LN style)
- [ ] Positional encoding addition
- [ ] The flow from source tokens → encoder → decoder → output logits
- [ ] Causal mask notation on masked self-attention
- [ ] $N \times$ notation showing the layer stack

**Quality test:** Could someone who hasn't taken this course understand the architecture from your diagram alone?

---

## Part 3: Conceptual Questions (30 min)

Answer each in 2–3 sentences, on paper:

### Q1: Why can't the decoder attend to future tokens?
What would break? How does the causal mask prevent it?

### Q2: What is the difference between self-attention and cross-attention?
In the decoder, which sublayer uses which? Where do Q, K, V come from in each case?

### Q3: Why does the FFN exist?
Attention alone is linear in V (it computes weighted sums). What does the FFN add that attention cannot?

### Q4: Why Pre-LN over Post-LN?
Draw the gradient flow path for both. Where does the gradient get modified by LayerNorm in each case?

### Q5: How does RoPE differ from sinusoidal PE?
Both use sinusoidal functions. What's fundamentally different about how they inject position?

### Q6: What happens if you remove positional encoding entirely?
Could the model still learn anything useful? What tasks would break? What might still work?

---

## Part 4: The Essay — "Why Does the Transformer Work?" (45 min)

Write a **one-page essay** (handwritten or typed, ~500 words) answering:

**Why does the transformer architecture work so much better than RNNs for sequence modeling?**

Your essay should touch on:
- **Parallelism:** Why can attention process an entire sequence at once?
- **Path length:** What is the maximum path length between two tokens, and why does it matter for gradient flow and long-range dependencies?
- **Expressiveness:** How do multi-head attention + FFN combine to create a powerful function approximator?
- **The scaling hypothesis:** Why does stacking more layers and using more data keep improving performance?
- **The compression connection:** How does this relate to the Day 5 insight that attention = selective compression?

This is not a literature review — it's a test of your *understanding*. Use your own words and reasoning.

---

## Part 5: Connection Review (15 min)

### Thread check: Trace the narrative

Re-read your notes from Days 5 and 6 (information theory, compression = prediction). Then answer:

1. In what sense is the attention matrix a **learned compression scheme**?
2. The FFN has $4 \times$ expansion — this is the *opposite* of compression. Why is expansion useful inside a compression machine?
3. If you think of the entire transformer as a function $f: \mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d}$, what is it compressing and what is it predicting?

### Re-read Section 3 of "Attention Is All You Need"

Focus on:
- Section 3.1: Encoder and Decoder Stacks — compare with your diagram
- Section 3.2: Attention — compare with your derivation
- Section 3.3: Position-wise FFN — note the dimension choices
- Section 3.5: Positional Encoding — compare with Day 13

Mark anything you understand now that you didn't when you first read it.

---

## Self-Assessment Rubric

| Area | Confident | Shaky | Need to re-study |
|---|---|---|---|
| Bahdanau attention derivation | | | |
| Scaled dot-product equation + scaling argument | | | |
| Multi-head attention mechanics | | | |
| Full transformer architecture (draw from memory) | | | |
| Pre-LN vs Post-LN | | | |
| Positional encoding (sinusoidal, RoPE, ALiBi) | | | |
| Training recipe (warmup, label smoothing) | | | |
| Why transformers > RNNs | | | |

**Rule:** If any row is "Need to re-study," go back to that day *before* continuing to Day 17. Attention and transformers are the foundation for everything that follows — from GPT to vision transformers to VLAs.

---

## Key Takeaways

- **Retrieval ≠ recognition.** You can follow a derivation without being able to reproduce it. Today tests retrieval.
- The transformer's power comes from: $O(1)$ path length, parallel computation, multi-head diversity, and the expand-compress FFN pattern.
- Attention is selective compression — different from but complementary to the information-theoretic view of Days 5–6.
- If you can draw the full transformer, derive the attention equation, and explain *why* it works, you're ready for the next phase.

## Connection to the Thread

This reflection day exists because **understanding is compression**. If you truly understand the transformer, you should be able to compress all of Days 10–15 into: (1) a single equation, (2) a single diagram, and (3) a single paragraph explaining why it works. If you can't compress it, you haven't understood it yet. That's what today tests.

## Further Reading

- Re-read: Vaswani et al. **"Attention Is All You Need"** (2017), Section 3. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Anthropic. **"In-context Learning and Induction Heads"** (2022). [transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — deeper understanding of how transformers work mechanistically
