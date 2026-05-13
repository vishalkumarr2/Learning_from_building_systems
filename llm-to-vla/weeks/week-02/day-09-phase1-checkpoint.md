# Day 9: Phase I Checkpoint

> **Phase I — DL Foundations & Information Theory | Week 2 | 2.5 hours**
> *"You don't truly understand something until you can explain it simply — and draw it from memory."*

## Navigation
- **Previous:** [Day 8: Phase I Mini-Project](day-08-phase1-miniproject.md)
- **Next:** [Day 10: Bahdanau Attention](day-10-bahdanau-attention.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Purpose

This is a **reflection and self-assessment day**. No new content — instead, you verify that the foundations from Days 1–7 are solid before building the attention mechanism on top of them.

Phase II (attention and transformers) assumes fluency with everything below. Gaps here become cracks later.

**Format:** Work through each question on paper or a blank notebook. No peeking at notes until after you've attempted all eight. Then compare, identify weak spots, and re-study.

---

## The Eight Checkpoint Questions

### Q1: Computation Graphs and Backpropagation (Day 1)

**Draw a computation graph** for a 2-layer MLP with ReLU activations processing a single input $x$ to produce output $\hat{y}$:

$$h_1 = \text{ReLU}(W_1 x + b_1), \quad \hat{y} = W_2 h_1 + b_2$$

Then **trace the backward pass**: starting from $\mathcal{L} = (\hat{y} - y)^2$, compute $\frac{\partial \mathcal{L}}{\partial W_1}$ using the chain rule.

**What you should be able to draw:**
```
x ──→ [×W₁ + b₁] ──→ [ReLU] ──→ h₁ ──→ [×W₂ + b₂] ──→ ŷ ──→ [MSE(ŷ,y)] ──→ L
                                                                      │
                backward: ← ∂L/∂ŷ ← ∂ŷ/∂h₁ ← ∂h₁/∂(W₁x+b₁) ← ... ← ∂L/∂W₁
```

**Check:** Can you explain why `∂ReLU/∂x = 1 if x > 0, else 0`? Why does this cause "dead neurons"?

---

### Q2: Why ResNets Train Better (Day 3)

Explain why ResNets train better than plain deep networks from **three distinct perspectives**:

1. **Gradient flow perspective:** How do skip connections affect the Jacobian $\frac{\partial \mathcal{L}}{\partial x_l}$?

2. **Optimization landscape perspective:** What does the loss surface look like for residual vs plain networks?

3. **Ensemble perspective:** How can a ResNet be viewed as an ensemble of shallower networks?

**Check:** Can you write the residual equation $y = F(x) + x$ and explain why learning $F(x) = 0$ (the identity) is easy?

---

### Q3: Vanishing Gradients in RNNs → LSTM (Day 4)

**Derive the vanishing gradient problem** in vanilla RNNs. Show that for a sequence of length $T$:

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_h^\top \cdot \text{diag}(\sigma'(\cdot))$$

Why does this product shrink exponentially when $\|W_h\| < 1$?

**Then explain:** How does the LSTM cell state $c_t$ fix this? What is the role of each gate?

```
LSTM gates (draw from memory):

   forget gate f_t:  what to ERASE from cell state
   input gate i_t:   what to WRITE to cell state
   output gate o_t:  what to READ from cell state

   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t     ← additive updates!
   h_t = o_t ⊙ tanh(c_t)
```

**Check:** Why does the additive update $c_t = f_t \odot c_{t-1} + \ldots$ prevent vanishing gradients? (Hint: $\frac{\partial c_t}{\partial c_{t-1}} = f_t$, a gate value near 1.)

---

### Q4: Seq2Seq Architecture and Its Bottleneck (Day 4)

**Draw the seq2seq architecture** from memory:

```
Encoder:  x₁ → x₂ → x₃ → [h_T]  ← context vector (the bottleneck)
                              ↓
Decoder:              [h_T] → y₁ → y₂ → y₃
```

**Answer:**
- Where exactly is the bottleneck?
- Why is compressing an entire sentence into a single fixed-size vector problematic?
- What information is lost?
- How will attention (Day 10) solve this?

---

### Q5: Information Theory Trinity (Day 5)

**Define precisely:**

1. **Entropy** $H(X)$: What does it measure? Write the formula.

2. **Cross-entropy** $H(p, q)$: What does it measure? Why is it the right loss for classification?

3. **KL divergence** $D_{KL}(p \| q)$: What does it measure? Show that $H(p, q) = H(p) + D_{KL}(p \| q)$.

**Check:** If your model achieves cross-entropy loss = 2.3 on a 10-class problem, how does this compare to random guessing? (Random guessing: $-\ln(0.1) = \ln(10) \approx 2.3$ — your model is no better than random!)

---

### Q6: Compression = Prediction (Day 5)

**Explain the Solomonoff-Hutter thesis in your own words:**

> "Compression = prediction = intelligence"

Give a concrete example. Why does a model that can predict the next token well necessarily understand the structure of the data?

**Check:** How does this connect to autoencoders (Day 8)? What is being compressed, and what is the "prediction"?

---

### Q7: One-Hot vs Dense Embeddings (Day 6)

| Property | One-Hot | Dense Embedding |
|----------|---------|-----------------|
| Dimensionality | ? | ? |
| Similarity between related items | ? | ? |
| Memory requirement | ? | ? |
| Can capture analogies | ? | ? |

**Check:** Why can't one-hot vectors represent that "king" is to "queen" as "man" is to "woman"? How do dense embeddings enable this?

---

### Q8: Training Stability Cookbook (Day 7)

List **five stability techniques** and for each, state:
- **What it does** (one sentence)
- **When to use it** (what problem it solves)
- **The typical setting** (e.g., max_norm=1.0)

| Technique | What | When | Setting |
|-----------|------|------|---------|
| 1. | | | |
| 2. | | | |
| 3. | | | |
| 4. | | | |
| 5. | | | |

---

## Self-Assessment Protocol

After attempting all eight questions from memory:

### Step 1: Grade Yourself

For each question, rate your answer:
- ✅ **Solid** — Could explain to a colleague without hesitation
- ⚠️ **Shaky** — Got the gist but missed details or couldn't derive formulas
- ❌ **Weak** — Couldn't recall or got it wrong

### Step 2: Identify Patterns

- Are your gaps conceptual (don't understand *why*) or mechanical (can't reproduce the formula)?
- Conceptual gaps require re-reading theory. Mechanical gaps require practice.

### Step 3: Re-Study Protocol

For each ⚠️ or ❌:
1. Re-read the corresponding day's Theory section
2. Redo one exercise from that day
3. Try the checkpoint question again tomorrow

### Step 4: Record Your Results

```markdown
## My Phase I Checkpoint Results — [DATE]

| Q# | Topic | Rating | Notes |
|----|-------|--------|-------|
| 1 | Backprop | | |
| 2 | ResNets | | |
| 3 | RNN/LSTM | | |
| 4 | Seq2Seq | | |
| 5 | Info Theory | | |
| 6 | Compression | | |
| 7 | Embeddings | | |
| 8 | Stability | | |

Weakest areas:
Re-study plan:
```

---

## Readiness Criteria

**You're ready for Phase II (Attention & Transformers) when:**

- [ ] You can draw computation graphs and trace backprop without reference
- [ ] You can explain skip connections from at least 2 perspectives
- [ ] You can derive the vanishing gradient problem and explain LSTM gates
- [ ] You understand why the seq2seq bottleneck is a problem (this motivates attention!)
- [ ] You can define entropy, cross-entropy, and KL divergence and relate them
- [ ] You believe the compression=prediction thesis and can give examples
- [ ] You understand the difference between sparse and dense representations
- [ ] You can list 5 stability techniques and know when to apply each

**Minimum bar:** 6/8 questions rated ✅. If you have more than 2 ❌, spend an extra day on review before proceeding.

---

## Key Takeaways

- Phase I gave you the **vocabulary** of deep learning: backprop, architectures, sequences, information theory, embeddings, stability
- The seq2seq bottleneck (Q4) is the *direct motivation* for attention — tomorrow's topic
- Compression = prediction (Q6) is the thread that ties everything together and will continue through transformers, LLMs, and VLAs
- Honest self-assessment now prevents confusion later; foundations matter more than speed

## Connection to the Thread

Phase I established that neural networks are **learned compression functions**. Every architecture (MLP, CNN, RNN, autoencoder) compresses its input into a representation that preserves task-relevant information and discards the rest.

The critical limitation we identified: the seq2seq model compresses *too aggressively* — forcing an entire input sequence through a single vector. Tomorrow, Bahdanau attention breaks this bottleneck by letting the decoder *look back* at the full input sequence. This is the beginning of the transformer revolution.

## Further Reading

- Review your notes from Days 1–7
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Read the first section as a preview of Phase II
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Lilian Weng's overview of attention mechanisms
