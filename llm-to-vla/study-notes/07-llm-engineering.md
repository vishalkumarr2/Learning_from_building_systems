# 07 — LLM Engineering & Robotics Connection

> Phase III · Days 37–44 · ~20 hours
> Prerequisites: 06-llm-training-alignment
> Learning Objectives: Understand quantization, ICL, long context, RAG; connect LLMs to robotics

---

## Table of Contents

1. [Quantization](#1-quantization)
2. [In-Context Learning](#2-in-context-learning)
3. [Long Context & Attention](#3-long-context--attention)
4. [RAG Overview](#4-rag-overview)
5. [LLMs for Robotics](#5-llms-for-robotics)
6. [Phase III Capstone](#6-phase-iii-capstone)
7. [Key Takeaways & Paper References](#7-key-takeaways--paper-references)
8. [Connection to Thread](#8-connection-to-thread)

---

## 1. Quantization

> Day 37 · ~3 hours

### 1.1 Why Quantization?

**The Memory Wall:**

| Model | Params | FP16 Memory | INT8 Memory | INT4 Memory |
|-------|--------|------------|------------|------------|
| 7B | 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 70B | 140 GB | 70 GB | 35 GB |

**Robotics Motivation:**
- Robots have limited onboard compute (Jetson Oort, embedded GPUs)
- Inference latency matters for real-time control
- Power budget constraints on mobile robots
- Can't stream to cloud for every decision
- **Quantization makes LLM-powered robotics feasible on the edge**

### 1.2 Quantization Basics

**What Is Quantization?**
Map high-precision floating-point weights to lower-precision integers.

**Linear (Affine) Quantization:**

$$x_q = \text{round}\left(\frac{x}{s}\right) + z$$

where:
- $s$ = scale factor
- $z$ = zero point
- $x_q$ = quantized integer

**Dequantization:**

$$\hat{x} = s \cdot (x_q - z)$$

**Symmetric vs Asymmetric:**
- Symmetric: $z = 0$, range $[-\alpha, \alpha]$
- Asymmetric: $z \neq 0$, range $[\text{min}, \text{max}]$

### 1.3 Quantization Methods

**Post-Training Quantization (PTQ):**
Quantize after training is complete — no retraining needed.

| Method | Precision | Approach | Quality |
|--------|-----------|----------|---------|
| **GPTQ** | 3-4 bit | Layer-wise, uses calibration data | Good |
| **AWQ** | 4 bit | Activation-aware, protects salient channels | Better |
| **GGUF** | 2-8 bit | CPU-friendly format (llama.cpp) | Good |
| **SqueezeLLM** | 3-4 bit | Sensitivity-based mixed precision | Good |
| **QuIP#** | 2-4 bit | Incoherence processing + lattice codebooks | Excellent at 2-bit |

**GPTQ (GPT-Quantized):**
```
For each layer:
    1. Compute Hessian from calibration data
    2. Quantize weights column-by-column
    3. Update remaining columns to compensate for error
    4. Move to next layer
```

**AWQ (Activation-Aware Weight Quantization):**
- Key insight: 1% of weights are "salient" (much larger activation magnitudes)
- Protect salient channels by scaling them up before quantization
- Searching for optimal per-channel scales using calibration data
- Often better quality than GPTQ at same bit-width

**GGUF Format (llama.cpp):**
- Designed for CPU inference (no GPU required)
- Mixed precision: different layers can use different bit-widths
- Excellent for edge deployment and consumer hardware
- Naming: `Q4_K_M` = 4-bit, K-quant, medium quality

### 1.4 Quality vs Speed vs Memory Tradeoffs

**Benchmark: LLaMA 2 7B on Common Benchmarks**

| Precision | Memory | Tokens/sec (A100) | MMLU | Perplexity |
|-----------|--------|-------------------|------|-----------|
| FP16 | 14 GB | 40 t/s | 45.3 | 5.47 |
| INT8 | 7 GB | 55 t/s | 45.1 | 5.49 |
| INT4 (GPTQ) | 3.5 GB | 65 t/s | 44.2 | 5.68 |
| INT4 (AWQ) | 3.5 GB | 65 t/s | 44.8 | 5.55 |
| INT3 | 2.6 GB | 70 t/s | 41.5 | 6.31 |
| INT2 | 1.8 GB | 75 t/s | 35.2 | 8.44 |

**Key Observations:**
- INT8 is nearly lossless (<0.5% quality drop)
- INT4 is the sweet spot (75% memory reduction, <2% quality drop)
- Below INT4, quality degrades rapidly
- AWQ consistently beats GPTQ at same bit-width

### 1.5 Quantization-Aware Training (QAT)

**Idea:** Simulate quantization during training so the model adapts.

```python
# Straight-Through Estimator (STE)
# Forward: quantize
x_q = quantize(x)
# Backward: pass gradient through as if no quantization happened
x.grad = x_q.grad  # STE approximation
```

**QAT vs PTQ:**
- QAT: better quality, but requires retraining (expensive)
- PTQ: no retraining, but slightly lower quality
- For most LLM use cases, PTQ is sufficient (INT4/INT8)

### 1.6 Practical Quantization Guide

**Decision Tree:**

```
Need to run LLM on edge device?
├── GPU available (Jetson, etc.)
│   ├── Memory > 8GB → INT8 (bitsandbytes)
│   └── Memory < 8GB → INT4 (AWQ or GPTQ)
├── CPU only
│   └── Use GGUF (llama.cpp)
│       ├── RAM > 16GB → Q5_K_M
│       └── RAM < 16GB → Q4_K_M or Q3_K_M
└── Cloud/datacenter
    ├── Latency critical → INT8 or FP8
    └── Quality critical → FP16 or BF16
```

---

## 2. In-Context Learning

> Day 38 · ~3 hours

### 2.1 What Is In-Context Learning (ICL)?

**Definition:** The ability of LLMs to learn new tasks from examples provided in
the prompt, without any gradient updates.

```
Zero-shot:
"Translate English to French: Hello world"

One-shot:
"Translate English to French:
 Good morning → Bonjour
 Hello world →"

Few-shot:
"Translate English to French:
 Good morning → Bonjour
 How are you? → Comment allez-vous?
 Thank you → Merci
 Hello world →"
```

**The Remarkable Property:** The model was never trained on this specific translation
task format, yet it learns the pattern from examples in the prompt.

### 2.2 ICL as Implicit Bayesian Inference

**The Bayesian Interpretation (Xie et al., 2022):**

Pretraining exposes the model to many "concepts" (latent variables). At inference
time, the demonstrations help the model identify the correct concept:

$$P(\text{output} | \text{demos}, \text{query}) = \sum_c P(\text{output} | c, \text{query}) \cdot P(c | \text{demos})$$

where $c$ is the latent concept inferred from demonstrations.

**Intuition:** The model doesn't literally learn a new algorithm. It uses the examples
to identify which of its pretrained capabilities to activate.

### 2.3 Why ICL Emerges at Scale

**Observations:**
- Small models (< 1B) show weak or no ICL ability
- ICL appears as an emergent capability at ~1B+ parameters
- More demonstrations help, but with diminishing returns
- ICL improves continuously with model scale

**Hypotheses for Emergence:**
1. **Mesa-optimization:** The transformer has learned an internal optimization algorithm
   (gradient descent in the forward pass via attention)
2. **Bayesian inference:** Larger models have better concept coverage from pretraining
3. **Distributional matching:** More parameters = better distribution matching from context

### 2.4 Chain-of-Thought (CoT) Prompting

**Standard Prompting:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each.
   How many tennis balls does he have now?
A: 11
```

**Chain-of-Thought:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each.
   How many tennis balls does he have now?
A: Roger starts with 5 balls. He buys 2 cans × 3 balls = 6 balls.
   Total = 5 + 6 = 11.
```

**Why CoT Works:**
- Decomposes complex reasoning into sequential steps
- Each step is within the model's capability
- The chain serves as working memory (since transformers have no explicit memory)
- Dramatically improves math, logic, and multi-step reasoning

**Variants:**
- **Zero-shot CoT:** "Let's think step by step"
- **Few-shot CoT:** Provide examples with reasoning chains
- **Self-consistency:** Sample multiple chains, take majority vote
- **Tree of Thought:** Explore multiple reasoning branches

### 2.5 ICL Sensitivities

**Surprising Findings:**
1. **Label correctness matters less than expected** — random labels still help format
2. **Example order matters** — different orderings can swing accuracy by 20%+
3. **Format matters more than content** — the structure of examples matters
4. **Recency bias** — the model weighs later examples more heavily

**Practical Implications:**
- Carefully curate and order few-shot examples
- Include diverse examples covering edge cases
- Test multiple orderings and templates
- Use self-consistency to reduce variance

---

## 3. Long Context & Attention

> Day 39 · ~3 hours

### 3.1 Context Window Growth

| Model | Year | Context Length |
|-------|------|---------------|
| GPT-3 | 2020 | 2,048 |
| GPT-3.5 | 2023 | 4,096 → 16,384 |
| GPT-4 | 2023 | 8,192 → 32,768 → 128K |
| Claude 3 | 2024 | 200K |
| Gemini 1.5 | 2024 | 1M → 2M |
| LLaMA 3.1 | 2024 | 128K |
| GPT-4.1 | 2025 | 1M |

**Why Long Context Matters:**
- Entire codebases in one prompt
- Full documents without chunking
- Multi-turn conversations without forgetting
- **Robotics:** Full environment history, task specifications, sensor logs

### 3.2 The Attention Bottleneck

**Standard Attention Complexity:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Time: $O(n^2 d)$ where $n$ = sequence length
- Memory: $O(n^2)$ for the attention matrix

**At 1M tokens:** Attention matrix = $10^6 \times 10^6 = 10^{12}$ entries — infeasible.

### 3.3 Efficient Attention Methods

**Flash Attention:**
- Same exact computation as standard attention
- But done in tiles/blocks that fit in GPU SRAM
- IO-aware: minimizes data movement between HBM and SRAM
- Result: 2-4× faster, $O(n)$ memory

**Sliding Window Attention (Mistral):**
- Each token attends to only $w$ nearest tokens
- Window size $w$ = 4096 typically
- Information propagates through layers: effective receptive field = $w \times L$
- Dramatically reduces memory for long sequences

**Multi-Query/Grouped-Query Attention (GQA):**

| Variant | Key-Value Heads | Memory |
|---------|----------------|--------|
| Multi-Head (MHA) | $h$ heads (= query heads) | Full |
| Multi-Query (MQA) | 1 head (shared) | $1/h$ of KV cache |
| Grouped-Query (GQA) | $g$ groups ($1 < g < h$) | $g/h$ of KV cache |

- GQA used in LLaMA 2 70B, LLaMA 3, Mistral
- Reduces KV cache memory without quality loss

### 3.4 RoPE (Rotary Position Embedding) Scaling

**RoPE Review:**
Encodes position by rotating query and key vectors:

$$f(x, m) = x \cdot e^{im\theta}$$

where $m$ is position and $\theta$ depends on dimension.

**Extending Context with RoPE Scaling:**

| Method | Approach | Quality |
|--------|----------|---------|
| **Linear scaling** | Divide position by factor $s$ | Decent, some quality loss |
| **NTK-aware** | Scale the base frequency | Better perplexity |
| **YaRN** | NTK + attention scaling + temperature | Best overall |
| **Dynamic NTK** | Adjust scaling during inference | Adaptive |

**Example (extending 4K → 32K):**
```python
# Linear scaling: simple but lossy
position = position / 8.0  # scale factor = target / original = 32K / 4K

# NTK-aware: scale the base frequency
base = 10000 * (scale_factor ** (dim / (dim - 2)))
```

### 3.5 RAG vs Long Context Tradeoff

| Aspect | RAG | Long Context |
|--------|-----|-------------|
| **Latency** | Retrieval overhead | Higher (quadratic attention) |
| **Cost** | Retrieval infra + short prompt | Long prompt = more tokens |
| **Accuracy** | Depends on retrieval quality | All info available (no retrieval errors) |
| **Freshness** | Can update index anytime | Must re-prompt |
| **Scalability** | Handles millions of docs | Limited by context window |
| **Complexity** | Chunking, embedding, retrieval pipeline | Just a longer prompt |

**Guidance:** Use RAG for large document collections; use long context for
documents that fit in the window and where you need holistic understanding.

---

## 4. RAG Overview

> Day 40 · ~3 hours

### 4.1 What Is RAG?

**Retrieval-Augmented Generation** augments LLM generation with external knowledge
by retrieving relevant documents at inference time.

**Pipeline:**

```
User Query
    │
    ↓
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Embed    │────→│ Vector Search│────→│ Top-K Results│
│ Query    │     │ (similarity) │     │              │
└──────────┘     └──────────────┘     └──────┬───────┘
                                              │
                                              ↓
                                     ┌────────────────┐
                                     │ LLM Generation │
                                     │ (query + docs) │
                                     └────────┬───────┘
                                              │
                                              ↓
                                         Response
```

### 4.2 Embedding & Retrieval

**Text Embedding:**
- Map text chunks to dense vectors
- Models: OpenAI `text-embedding-3-small`, BGE, E5, GTE
- Dimension: 384–3072
- Trained with contrastive learning: similar texts → close vectors

**Vector Databases:**
- FAISS (Meta) — in-memory, fast
- Pinecone — managed cloud
- Weaviate — open source
- Chroma — lightweight, local
- pgvector — PostgreSQL extension

**Similarity Search:**
$$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$

Retrieve top-$k$ documents with highest cosine similarity.

### 4.3 Chunking Strategies

How you split documents matters enormously:

| Strategy | Chunk Size | Overlap | Best For |
|----------|-----------|---------|---------|
| Fixed-size | 512 tokens | 50 tokens | Simple, general purpose |
| Sentence-based | Variable | Sentence boundary | Structured text |
| Paragraph-based | Variable | Paragraph boundary | Well-formatted docs |
| Recursive | 512–1024 | Hierarchical | Code, nested structures |
| Semantic | Variable | Meaning-based | Complex documents |

**Key Principle:** Chunks should be self-contained units of meaning.

### 4.4 Advanced RAG Patterns

**Naive RAG → Advanced RAG → Modular RAG:**

1. **Query Rewriting:** Transform the user query for better retrieval
2. **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer, use it for retrieval
3. **Re-ranking:** Use a cross-encoder to re-rank retrieved chunks
4. **Multi-step retrieval:** Iteratively retrieve and refine
5. **Self-RAG:** Model decides when to retrieve and evaluates retrieval quality

### 4.5 When RAG Beats Long Context

**Use RAG when:**
- Document collection is large (>1M tokens total)
- Documents update frequently
- Cost per query matters (shorter prompts = cheaper)
- Need attribution (which document answered the question)
- Latency requirements are moderate

**Use Long Context when:**
- All documents fit in context window
- Need holistic understanding across documents
- Simplicity is paramount
- Documents are interdependent

---

## 5. LLMs for Robotics

> Day 41 · ~3 hours

### 5.1 The Bridge Problem

**LLMs are powerful reasoners but:**
- They operate in token space (text)
- They can't perceive the physical world
- They can't take physical actions
- They lack grounding (words ≠ physics)

**Robots have complementary strengths:**
- Can sense the environment (cameras, lidar, touch)
- Can take physical actions (motors, grippers)
- But struggle with high-level reasoning and planning

**The opportunity:** Combine LLM reasoning with robot perception and action.

### 5.2 SayCan: Grounding Language in Robot Affordances

**Paper:** "Do As I Can, Not As I Say" (Google, 2022)

**Key Idea:** LLM provides task planning; a learned affordance model grounds
the plan in what the robot can actually do.

**Architecture:**

```
User: "I spilled my drink, can you help?"
         │
         ↓
┌─────────────────┐
│ LLM (SayCan)    │  → suggests actions with probabilities
│ P(useful | task) │     "Pick up sponge" (0.8)
└────────┬────────┘     "Go to kitchen" (0.3)
         │              "Pick up cup" (0.5)
         │
         ↓
┌─────────────────┐
│ Affordance Model │  → which actions are physically feasible
│ P(success | env) │     "Pick up sponge" (0.9) ← sponge visible
└────────┬────────┘     "Go to kitchen" (0.7)
         │              "Pick up cup" (0.1) ← no cup nearby
         │
         ↓
   Combined Score: P(useful) × P(success)
   → "Pick up sponge" (0.72) ✓ SELECTED
   → "Go to kitchen" (0.21)
   → "Pick up cup" (0.05)
```

**Scoring:**

$$a^* = \arg\max_a P_{\text{LLM}}(a | \text{task}) \cdot P_{\text{affordance}}(a | \text{env})$$

### 5.3 Code-as-Policies

**Paper:** "Code as Policies" (Google, 2023)

**Key Idea:** Instead of outputting actions directly, the LLM writes Python code
that calls robot APIs.

```python
# LLM generates this code from natural language instruction
# "Stack the red blocks on the blue plate"

def stack_red_on_blue():
    blue_plate = detect_object("blue plate")
    red_blocks = detect_objects("red block")
    
    # Sort by size for stable stacking
    red_blocks.sort(key=lambda b: b.size, reverse=True)
    
    for i, block in enumerate(red_blocks):
        target_pos = blue_plate.position + [0, 0, i * BLOCK_HEIGHT]
        pick(block)
        place(target_pos)
```

**Advantages:**
- Compositional (can combine primitives)
- Interpretable (code is readable)
- Precise (exact coordinates, loops, conditionals)
- Debuggable (can inspect code before execution)

### 5.4 Inner Monologue

**Paper:** "Inner Monologue: Embodied Reasoning through Planning with Language Models" (Google, 2022)

**Key Idea:** Continuous feedback loop between LLM planner and robot perception.

```
LLM: "First, pick up the red cup"
Robot: [attempts pick] → Success detector: FAILED (cup slipped)
LLM: "The pick failed. Let me try a different grasp angle."
Robot: [re-attempts] → Success detector: SUCCESS
LLM: "Good. Now place the cup on the table."
...
```

**Feedback Types:**
- Success/failure detection
- Scene description (from vision model)
- Human feedback (verbal corrections)
- Object state changes

### 5.5 The Grounding Gap → VLAs

**The Fundamental Problem:**

```
LLM alone:         text → text    (no perception, no action)
SayCan:            text → text → skill selection (limited actions)
Code-as-Policies:  text → code → API calls (requires predefined APIs)
Inner Monologue:   text ↔ text + perception (feedback but still text-mediated)

What we really want:
VLA:               vision + text → continuous actions (end-to-end)
```

**Why VLAs Are the Next Step:**
1. **End-to-end:** No hand-designed skill libraries or APIs
2. **Grounded:** Vision provides direct perception
3. **Continuous:** Output motor commands, not discrete text
4. **Generalizable:** One model, many tasks and environments

**The Bridge from LLMs to VLAs:**
- LLMs provide the language understanding and reasoning backbone
- Vision encoders (from CLIP, SigLIP) provide visual grounding
- Action heads output continuous motor commands
- All three are combined in a Vision-Language-Action model

### 5.6 Current LLM-Robot Systems

| System | Approach | Limitations |
|--------|----------|------------|
| SayCan | LLM + affordances | Needs predefined skills |
| Code-as-Policies | LLM → code | Needs predefined APIs |
| Inner Monologue | LLM + feedback | Text-mediated, slow |
| TidyBot | LLM + preferences | Category-level planning only |
| PROGPROMPT | LLM → programs | Limited to scripted actions |
| Voyager (Minecraft) | LLM + code library | Virtual environment only |

**Common Limitation:** All require predefined action primitives. The LLM plans,
but doesn't act. VLAs will close this gap.

---

## 6. Phase III Capstone

> Days 42–44 · ~8 hours

### 6.1 Project: Robotics-Domain LoRA Assistant

**Goal:** Fine-tune a small LLM (1-3B parameters) to be a knowledgeable robotics
assistant using LoRA, combining all Phase III skills.

**See:** [projects/03-robotics-assistant/README.md](../projects/03-robotics-assistant/README.md)

### 6.2 Capstone Requirements

1. **LoRA Fine-tuning:**
   - Apply QLoRA to a base model (LLaMA 3.2 1B or similar)
   - Train on robotics domain data
   - Compare base vs fine-tuned outputs

2. **Evaluation:**
   - Create a robotics QA evaluation set (50+ questions)
   - Benchmark: MMLU subset + domain-specific questions
   - Compare: base model vs LoRA model vs GPT-4

3. **Deployment:**
   - Quantize the fine-tuned model (INT4)
   - Measure inference latency
   - Run on consumer hardware

4. **Report:**
   - Training curves (loss, eval metrics)
   - Example conversations showing improvement
   - Resource usage analysis

### 6.3 Data Sources for Fine-tuning

- ROS 2 documentation
- OKS system documentation (`knowledge/` directory)
- Robotics textbook excerpts
- Robot troubleshooting conversations
- Navigation algorithm descriptions

---

## 7. Key Takeaways & Paper References

### 7.1 Key Takeaways

1. **Quantization enables edge deployment** — INT4 is the sweet spot (75% memory savings, <2% quality loss)
2. **AWQ > GPTQ** in most scenarios — activation-aware quantization preserves salient channels
3. **ICL is emergent and powerful** — models learn new tasks from prompt examples without gradient updates
4. **Chain-of-thought unlocks reasoning** — decomposing problems into steps dramatically improves performance
5. **Long context and RAG are complementary** — use RAG for large collections, long context for holistic understanding
6. **LLMs for robotics works but is limited** — text-mediated planning requires predefined skill libraries
7. **The grounding gap motivates VLAs** — end-to-end models that see, think, and act
8. **LoRA is the key technique for domain adaptation** — especially in data-scarce robotics settings

### 7.2 Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **SayCan** (Ahn et al.) | 2022 | LLM + affordance grounding for robot planning |
| **Code as Policies** (Liang et al.) | 2023 | LLM generates robot control code |
| **Inner Monologue** (Huang et al.) | 2022 | Feedback loop between LLM and robot |
| **GPTQ** (Frantar et al.) | 2022 | Post-training quantization for LLMs |
| **AWQ** (Lin et al.) | 2023 | Activation-aware weight quantization |
| **Chain-of-Thought** (Wei et al.) | 2022 | Step-by-step reasoning in LLMs |
| **An Explanation of ICL** (Xie et al.) | 2022 | ICL as implicit Bayesian inference |
| **Flash Attention** (Dao et al.) | 2022 | IO-aware exact attention |
| **RoPE** (Su et al.) | 2021 | Rotary position embeddings |
| **YaRN** (Peng et al.) | 2023 | Efficient context extension |
| **Self-RAG** (Asai et al.) | 2023 | Self-reflective RAG |
| **Voyager** (Wang et al.) | 2023 | LLM-powered agent in Minecraft |

### 7.3 Suggested Reading Order

1. **Start:** Chain-of-Thought (clear demonstrations of ICL power)
2. **Then:** SayCan (elegant robotics application)
3. **Then:** Code-as-Policies (practical bridge to robot control)
4. **Then:** AWQ or GPTQ (understand quantization for deployment)
5. **Optional:** Flash Attention, YaRN (for depth on efficiency)

---

## 8. Connection to Thread

### The Compression Lens

**Quantization as Lossy Compression:**
- FP16 → INT4: compress each weight from 16 bits to 4 bits (4× compression)
- The model's knowledge is remarkably robust to this compression
- Why? Because the information is distributed across billions of parameters
- Analogy: JPEG compression — lose some details, preserve the structure

**ICL as Runtime Compression:**
- During pretraining, the model compressed patterns from training data
- At inference, ICL examples activate the right compressed representation
- The model doesn't learn new weights — it selects which compressed knowledge to use
- ICL = indexing into a compressed knowledge base using demonstrations

**The Full Path to VLAs:**

```
Phase I:   Math foundations (the language of compression)
Phase II:  Vision models (compress visual information)
Phase III: LLMs (compress language, reasoning, and knowledge)
           ├── Training pipeline: pretrain → SFT → DPO
           ├── Efficiency: LoRA, quantization
           ├── Capabilities: ICL, CoT, long context
           └── Robotics connection: SayCan, Code-as-Policies
Phase IV:  Multimodal (bridge vision and language) ← NEXT
Phase V:   VLAs (bridge language, vision, and action)
```

### Why This Matters for Robotics

**The progression from LLMs to VLAs:**

1. **LLMs alone** → Can reason about tasks in text
2. **LLMs + APIs** → Can plan and call predefined robot skills
3. **LLMs + Vision** → Can see and reason about the world (VLMs)
4. **LLMs + Vision + Action** → Can see, reason, AND act (VLAs)

**Phase III gives us:**
- Understanding of how LLMs encode knowledge (for VLA backbone)
- LoRA for efficient adaptation (VLAs will be fine-tuned with LoRA)
- Quantization for edge deployment (VLAs must run on robots)
- ICL/CoT for flexible reasoning (VLAs inherit this from the LLM backbone)

**Next: Phase IV will bridge vision and language (CLIP, Flamingo, LLaVA) →
then Phase V combines everything into VLAs (RT-2, OpenVLA, π₀).**

---

## Notes

*These notes are part of the LLM-to-VLA curriculum.*
*Previous: [06 — LLM Training & Alignment](06-llm-training-alignment.md)*
*Next: Phase IV — Multimodal Models*
