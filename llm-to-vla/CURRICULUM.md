# From Attention to VLA: Zero to Robot Intelligence in 16 Weeks

### REVISED v2 — incorporating expert panel feedback (5-persona review)
### 2.5 hours/day × 7 days/week × 16 weeks = 112 days ≈ 280 hours
### From "I know ML basics" to "I can train and deploy a Vision-Language-Action model on a robot"

---

## Changes from v1

This curriculum was reviewed by a 5-persona expert panel (Ilya Sutskever / theory, Andrej Karpathy / implementation, Fei-Fei Li / vision, Chelsea Finn / robot learning, Sergey Levine / deployment). Major changes:

**Cut (~13 days saved):**
- Word2Vec from-scratch implementation → just understand embeddings
- RNN/LSTM compressed to essentials (understand the problem, not build from scratch)
- Seq2seq compressed to 1 day (exists to motivate attention)
- BERT compressed from 2 days to 1
- RAG/tools/assistant project compressed from 7 days to 1 day overview
- Multi-turn conversations + LLM security cut entirely

**Added (~13 days of new critical content):**
- Information Theory & Compression (Sutskever: "compression = prediction = intelligence")
- Training Stability Cookbook (Karpathy: "you WILL hit NaN losses")
- Scaling Laws DEEP dive with compression interpretation (Sutskever)
- nanoGPT as ablation laboratory — 2 full days of experiments (Karpathy)
- T5/encoder-decoder coverage (Karpathy)
- 3D Vision & Depth — Depth Anything, point clouds, NeRF concepts (Li)
- Video Understanding expanded to 2 days (Li)
- Florence-2, SAM 2 coverage (Li)
- PaLI & CoCa as VLM bridge architectures (Li)
- RL Foundations: MDPs, Policy Gradients, PPO — 3 days (Levine)
- Diffusion expanded from 1 → 3 days (Karpathy/Finn)
- ACT, Behavior Transformers, Decision Transformer (Finn)
- Action Tokenization expanded to 2 days (Karpathy/Finn)
- Data Collection for Robot Learning — 2 full days (Finn/Levine)
- Robot Policy Evaluation Methodology (Levine/Finn)
- Debugging Learned Policies (Finn)
- π₀.₅, GR00T N1, PaLM-E, GR-2 coverage (Finn/Karpathy)
- Sim-to-Real expanded to 2 days (Finn)
- Hybrid VLA + Classical Control expanded to 3 days with ROS2 exercise (Levine)
- Deployment expanded to 3 days (Levine)
- 5 "STOP AND REFLECT" markers at key insight moments (Sutskever)
- LeRobot as primary framework for Phases VI–VII (Finn)

---

## Who This Is For

You have:
- ✅ Working knowledge of ML: loss functions, backprop, SGD/Adam, basic neural nets
- ✅ Python + PyTorch experience (can write training loops)
- ✅ Linear algebra, probability, calculus fundamentals
- ✅ Robotics domain knowledge (OKS AMR, ROS, state estimation, control)

You want:
- Understand attention from first principles → transformers → LLMs → vision-language → VLA
- Hands-on implementations at every stage (not just reading papers)
- Connect it all to robotics: end with models that see, reason, and act on physical robots

---

## Design Philosophy

1. **Every concept earns its place through code** — no topic without an implementation exercise
2. **Build before you read** — implement the simplified version first, THEN read the paper
3. **One paper per day max** — depth over breadth; understand the math, run the code
4. **Robotics grounding** — every phase connects back to robot applications
5. **Daily structure**: ~45 min theory → ~60 min implementation → ~45 min exercises/paper reading
6. **Phase gates** — checkpoint quiz before advancing to the next phase
7. **Capstone projects** — each phase ends with a mini-project that proves understanding
8. **"Compression = prediction = intelligence"** — this thread runs through the entire curriculum (Sutskever)
9. **STOP AND REFLECT** — 5 moments where you must pause and internalize a key insight before continuing

---

## Expert Panel

| Expert Persona | Focus | Key Contribution |
|----------------|-------|------------------|
| **Ilya Sutskever** | Theory & Foundations | Information theory, scaling laws, compression interpretation, STOP AND REFLECT markers |
| **Andrej Karpathy** | Implementation & Code | nanoGPT ablation lab, training stability, T5, action tokenization, minbpe |
| **Fei-Fei Li** | Vision & Perception | 3D vision, depth estimation, video understanding, Florence-2, SAM 2, PaLI, CoCa |
| **Chelsea Finn** | Robot Learning | RL foundations, ACT, diffusion policy, LeRobot, data collection, debugging policies |
| **Sergey Levine** | Deployment & Systems | Hybrid VLA+classical control, ROS2 integration, deployment pipeline, evaluation methodology |

---

## Curriculum Overview

| Phase | Days | Count | Topic | Hours | Outcome |
|-------|------|-------|-------|-------|---------|
| **I** | 1–9 | 9 | DL Foundations + Information Theory | 22.5 | Fluent in CNNs, RNNs, seq2seq; understand WHY attention is needed; grasp compression = prediction |
| **II** | 10–30 | 21 | Attention, Transformers & Scaling | 52.5 | Implement transformer from scratch, run nanoGPT ablations, understand scaling laws deeply |
| **III** | 31–44 | 14 | LLMs: Training & Alignment | 35 | Understand the full LLM pipeline: pretrain → SFT → RLHF/DPO; fine-tune with LoRA |
| **IV** | 45–58 | 14 | Vision: ViT, 3D, Video | 35 | ViT from scratch, 3D depth, video understanding, detection with DETR + Florence-2 |
| **V** | 59–70 | 12 | Vision-Language Models | 30 | CLIP, LLaVA, BLIP-2, PaLI; fine-tune a VLM on robotics data |
| **VI** | 71–91 | 21 | Robot Learning: RL, Diffusion & Data | 52.5 | RL foundations, diffusion policy, ACT, action tokenization, data collection, evaluation |
| **VII** | 92–112 | 21 | VLAs: Architecture to Deployment | 52.5 | RT-2, Octo, OpenVLA, π₀/π₀.₅; hybrid control; ROS2 deployment; capstone project |
| | | **112** | | **280** | |

---

## Dependency Graph

```
                            ┌──────────────────────────────────────┐
                            │  Phase VII: VLA Models (Days 92-112) │
                            │  RT-2, Octo, OpenVLA, π₀, π₀.₅     │
                            │  Hybrid Control, ROS2 Deployment     │
                            └───────────────────┬──────────────────┘
                                                │
                   ┌────────────────────────────┼────────────────────────────┐
                   │                            │                            │
       ┌───────────▼──────────┐   ┌─────────────▼──────────────┐  ┌─────────▼────────────────┐
       │ Phase V: VLMs        │   │ Phase VI: Robot Learning   │  │ Your existing            │
       │ CLIP, LLaVA, BLIP-2 │   │ RL, Diffusion, ACT, Data  │  │ robotics knowledge       │
       │ PaLI, CoCa           │   │ LeRobot framework         │  │ (ROS, control, nav, EKF) │
       └───────────┬──────────┘   └──────────┬─────────────────┘  └──────────────────────────┘
                   │                         │
        ┌──────────┴──────────┐              │
        │                     │              │
 ┌──────▼──────────┐  ┌──────▼──────────────────────┐
 │ Phase III: LLMs │  │ Phase IV: Vision            │
 │ Training &      │  │ ViT, 3D Depth, Video,      │
 │ Alignment       │  │ Detection (DETR, Florence)  │
 └──────┬──────────┘  └──────┬──────────────────────┘
        │                    │
        └────────┬───────────┘
                 │
      ┌──────────▼───────────────────┐
      │ Phase II: Attention &        │
      │ Transformers & Scaling       │
      │ nanoGPT ablation lab         │
      └──────────┬───────────────────┘
                 │
      ┌──────────▼───────────────────┐
      │ Phase I: DL Foundations +    │
      │ Information Theory           │
      │ "Compression = Prediction"   │
      └──────────────────────────────┘
```

---

## Tools & Setup (Do Before Day 1)

### Python Stack
```bash
# Create a dedicated environment
python3 -m venv llm-vla-env
source llm-vla-env/bin/activate

# Core
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib
pip install jupyter notebook ipywidgets

# Transformers & LLM
pip install transformers datasets tokenizers
pip install accelerate bitsandbytes peft
pip install sentencepiece tiktoken

# Vision
pip install timm  # PyTorch Image Models
pip install opencv-python pillow

# Training
pip install wandb  # experiment tracking
pip install einops  # tensor manipulation (used everywhere in modern code)

# Diffusion & Generative
pip install diffusers  # diffusion models

# Vision-Language
pip install open_clip_torch  # CLIP

# Robot Learning (Phases VI-VII)
pip install lerobot           # HuggingFace robot learning framework
pip install gymnasium          # RL environments (successor to OpenAI Gym)
pip install mujoco             # Physics simulator for robot learning
pip install robomimic          # Imitation learning benchmark

# Later phases
pip install vllm              # LLM inference (if GPU available)
```

### Hardware
- **Minimum**: GPU with 8GB VRAM (RTX 3060/4060) for Phases I–IV
- **Recommended**: GPU with 16–24GB VRAM (RTX 3090/4090) for Phases V–VII
- **Alternative**: Google Colab Pro ($10/month) for GPU access
- **Phase VI–VII**: MuJoCo (free) for robot simulation

### Key Textbooks & Resources (Free)
| Resource | Use For | Link |
|----------|---------|------|
| Andrej Karpathy's "Neural Networks: Zero to Hero" | Phases I–III (video lectures) | youtube.com |
| "Attention Is All You Need" (Vaswani 2017) | Phase II core paper | arxiv.org/abs/1706.03762 |
| Jay Alammar's "The Illustrated Transformer" | Phase II visual guide | jalammar.github.io |
| Lilian Weng's blog | All phases — phenomenal overviews | lilianweng.github.io |
| Karpathy's nanoGPT | Phase II ablation lab | github.com/karpathy/nanoGPT |
| Karpathy's minbpe | Phase II tokenization | github.com/karpathy/minbpe |
| Hugging Face course | Phases III–V practical | huggingface.co/course |
| LeRobot (Hugging Face) | Phases VI–VII primary framework | github.com/huggingface/lerobot |
| "An Image is Worth 16x16 Words" (Dosovitskiy 2020) | Phase IV core paper | arxiv.org/abs/2010.11929 |
| "Learning Transferable Visual Models" (Radford 2021) | Phase V CLIP paper | arxiv.org/abs/2103.00020 |
| "Denoising Diffusion Probabilistic Models" (Ho 2020) | Phase VI core paper | arxiv.org/abs/2006.11239 |
| "Diffusion Policy" (Chi et al. 2023) | Phase VI robot actions | arxiv.org/abs/2303.04137 |
| Stanford CS231n / CS224n / CS25 | Vision / NLP / Transformers | online lectures |
| "RT-2: Vision-Language-Action Models" (Brohan 2023) | Phase VII core paper | arxiv.org/abs/2307.15818 |

---

# PHASE I: DL Foundations + Information Theory (Days 1–9)

> **Goal:** Solidify DL foundations, understand the "before transformers" world, and grasp the deep connection between compression, prediction, and intelligence. When you finish this phase, you should appreciate WHY attention was invented.

### Phase Gate (answer before starting)
- [ ] Can you write a PyTorch training loop from memory?
- [ ] Do you understand backprop through a computation graph?
- [ ] Can you explain what a loss function measures and name 3?

If you fail any of these: spend 2 days on Karpathy's "micrograd" video first.

---

## Day 1: Computation Graphs & Backprop Refresh

**Theory (45 min):**
- Computation graph representation of neural networks
- Chain rule through graphs — why autodiff works
- PyTorch autograd: `.backward()`, `.grad`, `no_grad()`
- Forward mode vs reverse mode AD — why reverse is O(1) in output dimension

**Implementation (60 min):**
- Implement a tiny autograd engine (Karpathy's micrograd-style)
- Build: Value class with `+`, `*`, `tanh`, backward
- Verify: compare your gradients with PyTorch autograd on a small network

**Exercise (45 min):**
- Hand-compute gradients for a 2-layer network on paper
- Verify with PyTorch autograd
- Bonus: implement `ReLU` and `exp` operations in your autograd

---

## Day 2: CNN & ResNets

**Theory (45 min):**
- Convolution as learnable feature extraction
- Receptive field, stride, padding, dilation
- Feature hierarchy: edges → textures → parts → objects
- Residual connections (ResNet) — why they matter for gradient flow
- Skip connections as "gradient highways" — enabling 100+ layer networks

**Implementation (60 min):**
- Build a small ResNet (ResNet-18 style) from scratch in PyTorch
- Train on CIFAR-10, visualize learned filters at different layers

**Exercise (45 min):**
- Calculate output dimensions for a given conv stack
- Experiment: remove skip connections, measure accuracy drop on a 20-layer net
- Plot gradient norms with and without residuals

---

## Day 3: RNN/LSTM Essentials

> **NOTE (v2 change):** No from-scratch LSTM implementation. The goal is to understand WHY recurrent models fail for long sequences — this motivates attention.

**Theory (45 min):**
- Vanilla RNN: hidden state recurrence, unrolling, backprop through time (BPTT)
- **The vanishing gradient problem** — the fundamental failure mode (draw the gradient flow)
- LSTM gates: forget, input, output — how they create gradient "highways" through time
- GRU as simplified LSTM
- Key takeaway: even LSTMs struggle beyond ~100-200 timesteps in practice

**Implementation (60 min):**
- Implement a vanilla RNN cell from scratch (just the cell, not full LSTM)
- Train character-level language model on Shakespeare text with PyTorch's `nn.LSTM`
- Measure: training loss vs sequence length to see the degradation

**Exercise (45 min):**
- Trace gradients through 10 timesteps of vanilla RNN (show vanishing on paper)
- Compare RNN vs LSTM on a 50-step vs 200-step dependency task
- Write down: "What would an ideal architecture do differently?"

---

## Day 4: Seq2Seq & The Bottleneck

> **NOTE (v2 change):** Compressed to 1 day. This exists to motivate attention — the bottleneck IS the problem that attention solves.

**Theory (45 min):**
- Encoder-decoder architecture for translation
- **The bottleneck problem** — entire source sentence compressed into ONE vector
- Why this is catastrophic for long sequences (information loss)
- Teacher forcing vs free-running decode
- Beam search decoding basics

**Implementation (60 min):**
- Build a simple encoder-decoder for number translation (e.g., "123" → "one hundred twenty three")
- Visualize the hidden state bottleneck: plot reconstruction quality vs input length

**Exercise (45 min):**
- Experiment: vary hidden size, show bottleneck effect on sequences of length 5 vs 20 vs 50
- Write down: "If I could let the decoder LOOK BACK at all encoder states, what would change?"
- This is the question that Bahdanau answered. You're ready for Phase II.

---

## Day 5: Information Theory & Compression (NEW)

> **"Compression is equivalent to prediction, which is equivalent to intelligence."** — Ilya Sutskever

**Theory (45 min):**
- Shannon entropy: H(X) = -Σ p(x) log p(x) — the minimum bits to encode a message
- Cross-entropy loss IS compression efficiency: H(p, q) measures how well model q compresses data from distribution p
- KL divergence: D_KL(p || q) = H(p, q) - H(p) — the "compression gap" between your model and perfection
- **The deep connection**: next-token prediction = optimal compression = understanding
  - A model that perfectly predicts the next token has perfectly compressed the data
  - To compress well, you MUST understand structure, meaning, causality
  - This is why language models trained on "just" next-token prediction learn reasoning
- Bits-per-character / bits-per-token as the fundamental metric
- Why cross-entropy is THE loss function for language models (it's not arbitrary — it's information-theoretically optimal)

**Implementation (60 min):**
- Compute entropy of various text corpora (English, code, random bytes)
- Train your Day 3 character-LM and measure bits-per-character over training
- Plot: BPC improvement = the model is learning to compress = learning to understand
- Compare: BPC of your model vs Shannon's estimate of English entropy (~1.0 bit/char)

**Exercise (45 min):**
- Calculate KL divergence between your model's predictions and the empirical distribution
- Experiment: train on Shakespeare vs random text — which reaches lower BPC? Why?
- Reflect: "If cross-entropy measures compression, then every training step is teaching the model to be a better compressor. What does a perfect compressor know?"

---

## Day 6: Embeddings & Representation Learning

> **NOTE (v2 change):** No Word2Vec from-scratch implementation. Focus on understanding distributed representations and subword tokenization preview.

**Theory (45 min):**
- Why one-hot encoding fails (orthogonal, no similarity)
- Distributed representations: meaning encoded in continuous vectors
- Word2Vec concepts (skip-gram, CBOW) — understand the idea, not the implementation
- Embedding arithmetic: king - man + woman ≈ queen (geometric structure of meaning)
- Subword tokenization preview: BPE, WordPiece — why modern models don't use word-level embeddings
- Connection to information theory: embeddings compress sparse discrete symbols into dense continuous representations

**Implementation (60 min):**
- Load pretrained Word2Vec/GloVe embeddings in PyTorch
- Query nearest neighbors, test analogies
- Visualize embedding clusters with t-SNE (semantic groupings emerge)
- Build a simple embedding layer and train it end-to-end on a classification task

**Exercise (45 min):**
- Compare: random init embeddings vs pretrained embeddings on sentiment classification
- Analyze: what do embedding dimensions "mean"? (they don't have clean interpretations — that's the point)
- Preview: tokenize the same sentence with character-level, word-level, BPE — compare sequence lengths

---

## Day 7: Training Stability Cookbook (NEW)

> **"You WILL hit NaN losses, gradient explosions, and mysterious training collapses. Learn the diagnostic tools NOW before you need them in anger."** — Karpathy

**Theory (45 min):**
- Common training failures: NaN/Inf losses, gradient explosion, loss plateaus, sudden spikes
- Gradient norm monitoring — your most important diagnostic tool
- Learning rate schedules: warmup (why transformers need it), cosine decay, linear decay
- Mixed precision pitfalls: FP16 overflow, loss scaling, when to use BF16
- Loss curve interpretation: what healthy vs unhealthy training looks like
- Weight initialization: Xavier, Kaiming, why it matters more for deep nets
- The "training recipe" mindset: batch size, LR, warmup, weight decay as a coherent system

**Implementation (60 min):**
- Build a training monitor: log gradient norms, loss, learning rate per step
- Deliberately cause training instabilities:
  1. Set LR too high → observe gradient explosion
  2. Remove warmup from a transformer → observe loss spike
  3. Use FP16 without loss scaling → observe NaN
- Implement cosine LR schedule with warmup from scratch
- Plot all diagnostics with matplotlib/wandb

**Exercise (45 min):**
- Given a loss curve (provided), diagnose: is this LR too high? Too low? Warmup too short?
- Create a "training stability checklist" that you'll use for the rest of the curriculum:
  - [ ] Gradient norms < 1.0?
  - [ ] Loss decreasing smoothly?
  - [ ] No NaN/Inf in any tensor?
  - [ ] LR schedule looks correct?
- Save this checklist — you'll use it in every phase that follows

---

## Days 8–9: Phase I Mini-Project + Checkpoint

**Project:** Character-level LM — RNN vs simple attention comparison
- Train on a text corpus of your choice
- Model A: LSTM character-LM (your Day 3 model)
- Model B: Simple self-attention character-LM (preview of Phase II — use a basic attention mechanism)
- Compare: training curves, bits-per-character, sample quality, training stability
- Use your Day 7 training monitor for both
- **Deliverable:** Notebook with comparison plots, BPC analysis, generated samples, stability diagnostics

**Phase I Checkpoint (answer without notes):**
- [ ] Why does a vanilla RNN struggle with long-range dependencies? Draw the gradient flow.
- [ ] What is the "bottleneck" in seq2seq? Why does it fail for long sentences?
- [ ] Explain cross-entropy loss as compression. What does lower loss MEAN in information-theoretic terms?
- [ ] Why do residual connections help deep networks train? (gradient highway)
- [ ] What is the relationship between prediction and compression? (Sutskever's insight)
- [ ] Name 3 training instabilities and how you'd diagnose each.

---

# PHASE II: Attention, Transformers & Scaling (Days 10–30)

> **Goal:** Master attention from first principles, build a full transformer, run systematic experiments with nanoGPT, and deeply understand scaling laws. This is the longest and most important phase.

### Phase Gate
- [ ] Can you explain the seq2seq bottleneck and why it motivates attention?
- [ ] Do you understand cross-entropy as compression?

---

## Week 2: Core Attention + Transformer (Days 10–16)

### Day 10: Bahdanau Attention — "Looking Back at the Source"

**Theory (45 min):**
- Motivation: let the decoder look at ALL encoder states, not just the final one
- Bahdanau attention (2014): learn an alignment function
- Query (decoder state), Key/Value (encoder states) intuition
- Attention weights as soft alignment — visualize as heatmap
- "Attention is a soft dictionary lookup" — this one sentence is the key insight

**Implementation (60 min):**
- Add Bahdanau attention to your Day 4 seq2seq model
- Visualize attention weights as a heatmap over source/target tokens

**Exercise (45 min):**
- Compare BLEU scores: seq2seq vs seq2seq+attention
- Analyze attention heatmaps: does the model learn alignment?
- Measure: does attention fix the long-sequence degradation from Day 4?

### Day 11: Scaled Dot-Product Attention

**Theory (45 min):**
- From Bahdanau (additive) to dot-product attention — simpler, faster
- `score = Q · K^T` — similarity as dot product
- Why scale by `√d_k` — preventing softmax saturation for large d_k
- Full equation: `Attention(Q, K, V) = softmax(Q K^T / √d_k) V`
- Matrix view: attention operates on entire sequences simultaneously (parallelizable!)

**Implementation (60 min):**
- Implement `scaled_dot_product_attention()` from scratch
- Test on random tensors, verify dimensions
- Compare with `torch.nn.functional.scaled_dot_product_attention`

**Exercise (45 min):**
- By hand: compute attention for a 3-token sequence with d_k=2
- What happens without the scaling? Show softmax becoming one-hot at large d_k
- Compute FLOP count: attention is O(n² · d) — why this matters

### Day 12: Multi-Head Attention

**Theory (45 min):**
- Why one attention head isn't enough — different heads attend to different patterns
- Multi-head: split Q, K, V → attend independently → concatenate → project
- `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O`
- Each `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`
- Interpreting heads: some attend to position, syntax, semantics, or rare tokens

**Implementation (60 min):**
- Implement `MultiHeadAttention` class from scratch
- No `nn.MultiheadAttention` — build it yourself with explicit W_Q, W_K, W_V, W_O

**Exercise (45 min):**
- Verify parameter count matches theory: 4 × d_model × d_model (for h heads)
- Visualize different heads' attention patterns on a sample sequence
- Experiment: 1 head vs 4 vs 8 — measure quality on a toy task

### Day 13: Positional Encoding

**Theory (45 min):**
- The problem: attention is permutation-equivariant — it doesn't know token order!
- Sinusoidal positional encoding (Vaswani 2017)
- `PE(pos, 2i) = sin(pos / 10000^(2i/d))` — why this formula works (relative position via dot products)
- Learned positional embeddings (GPT-2 style)
- Rotary Position Embeddings (RoPE) — the modern standard (used in LLaMA, Mistral)
- ALiBi — relative bias approach

**Implementation (60 min):**
- Implement sinusoidal PE from scratch
- Implement RoPE (the modern approach used in all current LLMs)
- Visualize the encoding matrices as heatmaps

**Exercise (45 min):**
- Show that sinusoidal PE can represent relative positions via dot products
- Compare sinusoidal vs learned vs RoPE on a toy task
- Why is RoPE preferred? (generalizes to longer sequences)

### Day 14: The Full Transformer — "Attention Is All You Need"

**Theory (45 min):**
- Read the paper (or Jay Alammar's Illustrated Transformer guide)
- Encoder block: Multi-Head Attention → Add & Norm → FFN → Add & Norm
- Decoder block: Masked Self-Attention → Cross-Attention → FFN (each with Add & Norm)
- Pre-LN (modern) vs Post-LN (original) — why Pre-LN trains more stably
- Feed-forward: `FFN(x) = GELU(xW_1 + b_1)W_2 + b_2` (two linear layers with activation)
- Residual connections everywhere — "the transformer is just attention + FFN + residuals, stacked"

**Implementation (60 min):**
- Implement `TransformerEncoderLayer` from scratch
- Implement `TransformerDecoderLayer` from scratch (with causal mask)
- Stack into full Transformer model

**Exercise (45 min):**
- Count parameters for a given transformer config (L layers, d_model, n_heads, d_ff)
- Trace a single token through the entire forward pass (draw the computation flow)

### Day 15: Training a Transformer

**Theory (45 min):**
- Label smoothing — why it helps generalization
- Learning rate warmup — why transformers need it (cold parameters + attention scaling)
- The "noam" schedule: `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`
- Masked loss for padding tokens
- Use your Day 7 training stability cookbook!

**Implementation (60 min):**
- Train your Day 14 transformer on a small translation dataset (Multi30k EN→DE)
- Implement warmup + cosine decay scheduler
- Log to wandb, use your training monitor

**Exercise (45 min):**
- Compare with your Day 4 seq2seq — how much better is the transformer?
- Experiment: vary number of heads, layers; measure BLEU + training stability
- Attention heatmap analysis: what do different layers attend to?

### Day 16: 🛑 STOP AND REFLECT #1

> **"Attention is just a soft dictionary lookup. You have a query, you match against keys, and you retrieve a weighted sum of values. That's it. The transformer is embarrassingly simple — just attention + FFN + residuals, stacked. No recurrence, no convolution, no explicit memory. Why does something this simple work so well?"**

**Spend this day on reflection and consolidation (2.5 hrs):**
- Re-derive the attention equation from scratch on paper, without notes
- Draw the full transformer architecture from memory
- Write a 1-page essay: "Why does the transformer work?" Your current best understanding.
- Revisit your Day 5 insight: attention lets the model selectively compress — it reads the relevant parts of the input, not all of it. This IS better compression.
- Re-read Section 3 of "Attention Is All You Need" — does it make more sense now?

---

## Week 3: Variants + GPT (Days 17–23)

### Day 17: Efficient Attention — The O(n²) Problem
**Theory:** Standard attention is O(n²) in sequence length. Sparse attention: local windows, strided, random patterns. **Flash Attention** (Dao et al.) — IO-aware exact attention: same result, 2-4x faster by being smart about GPU memory hierarchy. Linear attention approximations.

**Implementation:** Implement sliding-window attention. Benchmark Flash Attention vs naive on sequences of increasing length.

### Day 18: KV Cache — Why Autoregressive Inference Is Fast
**Theory:** During generation, only the new token's Q changes — K, V from all previous tokens can be cached. Memory vs compute tradeoff. Paged attention (vLLM) — virtual memory for KV cache.

**Implementation:** Implement KV cache for autoregressive decoding. Measure speedup: with vs without cache.

### Day 19: Normalization + Activations (Compressed)
**Theory:** Post-LN (original) vs Pre-LN (modern, more stable). RMSNorm (simpler than LayerNorm, used in LLaMA). SwiGLU: `SwiGLU(x) = (xW₁ ⊙ σ(xW_gate)) W₂` — why GLU variants dominate modern LLMs.

**Implementation:** Compare Pre-LN vs Post-LN training stability. Swap activations (ReLU → GELU → SwiGLU), compare convergence.

### Day 20: Mixture of Experts (MoE)
**Theory:** Not all parameters active for every token — sparse activation. Router selects top-k experts (typically top-2). Load balancing loss to prevent expert collapse. Switch Transformer, Mixtral architecture.

**Implementation:** Implement a simple MoE layer with top-2 routing. Train on a small task.

### Day 21: BERT & Masked Language Modeling (Compressed to 1 day)
**Theory:** Bidirectional encoders — see the whole sequence at once. Masked Language Model (MLM): predict [MASK]ed tokens. [CLS] token for classification, fine-tuning paradigm. Why encoder-only models excel at understanding but can't generate.

**Implementation:** Fine-tune BERT-base on sentiment classification (SST-2) using HuggingFace. Visualize attention patterns.

### Day 22: Tokenization Deep Dive
**Theory:** Character-level vs word-level vs subword — the tradeoffs. Byte Pair Encoding (BPE) step by step. WordPiece, Unigram, SentencePiece. tiktoken (GPT tokenizer). Why tokenization matters: multilingual, code, math, "How many Rs in strawberry?"

**Implementation:** Implement BPE from scratch (Karpathy's minbpe). Train on a corpus, inspect vocabulary. Compare with tiktoken.

### Day 23: GPT & nanoGPT — Ablation Laboratory (Day 1)
**Theory:** Decoder-only transformers: autoregressive LM with causal mask. GPT-1 → GPT-2 → GPT-3 progression. Why decoder-only won for generation. The key insight: a single architecture for everything.

**Implementation:** Read and annotate Karpathy's nanoGPT code — understand EVERY line. Set up the training pipeline on Shakespeare. Run baseline training.

---

## Week 4: Scaling + Decoders + Capstone (Days 24–30)

### Day 24: nanoGPT Ablation Experiments (Day 2)
**Theory:** Systematic ablation methodology — change one thing at a time, measure impact.

**Implementation:** Run ablation experiments on nanoGPT:
1. Vary n_layers (2, 4, 6, 8, 12) — plot loss curves
2. Vary n_heads (1, 2, 4, 8) — measure quality
3. Vary d_model (64, 128, 256, 512) — compare
4. Swap activation (ReLU, GELU, SwiGLU)
5. Swap normalization (LayerNorm, RMSNorm, Pre-LN, Post-LN)
6. Plot your own scaling curves from these runs

**Deliverable:** Ablation report with plots showing impact of each architectural choice.

### Day 25: Scaling Laws & Emergence (DEEP DIVE — Sutskever)
**Theory:** Kaplan scaling laws (2020): loss scales as power law with compute, data, parameters — `L(C) ∝ C^(-α)`. Chinchilla (2022): optimal data/param ratio — most models were undertrained. **Compression interpretation** (Sutskever): scaling laws are compression efficiency curves — more compute = better compressor = more "understanding." Emergence debate: do abilities appear suddenly at scale, or is it a measurement artifact? In-context learning as implicit Bayesian inference — the model becomes a better "learner" at scale, not just a better "memorizer."

**Exercise:** Plot scaling curves from published data. Estimate compute for training a 7B model with Chinchilla-optimal data.

### Day 26: 🛑 STOP AND REFLECT #2

> **"The same architecture, more data, more compute, keeps getting better along a predictable power law. No architecture changes needed. It just keeps going. What does this tell us about the nature of learning? If compression = intelligence, then scaling compute = scaling intelligence. Is there a ceiling?"**

**Spend this day on reflection (2.5 hrs):**
- Write: "What do scaling laws mean for robotics?" — if they hold for language, do they hold for robot actions?
- Re-read your Day 5 notes on information theory. Connect: power-law scaling of loss = power-law improvement in compression efficiency.
- Discussion: the "bitter lesson" (Rich Sutton) — general methods that scale beat clever hand-crafted approaches. How does this apply to robot control?

### Day 27: Sampling & Generation
**Theory:** Greedy, top-k, top-p (nucleus sampling), temperature scaling. Repetition penalty. Structured generation (constrained decoding, grammar-guided). How sampling temperature trades off creativity vs coherence.

**Implementation:** Implement all sampling strategies for your nanoGPT. Compare: same prompt with different settings, analyze quality. Implement constrained decoding for JSON output.

### Day 28: T5 & Encoder-Decoder LMs (NEW — Karpathy)
**Theory:** Text-to-text framework: every task as "input text → output text." T5 architecture: encoder-decoder transformer. Why encoder-decoder didn't win for generation but matters for understanding sequence-to-sequence tasks. Comparison: GPT (decoder-only) vs BERT (encoder-only) vs T5 (encoder-decoder) — when to use each.

**Implementation:** Use T5-small from HuggingFace for summarization and translation. Compare generation quality with GPT-2 on the same task.

### Days 29–30: Phase II Capstone

**Project:** Train a mini-LM (10M params) + ablation report + scaling analysis
- Choose a domain corpus (code, robotics papers, or Shakespeare)
- Train a ~10M param GPT from scratch using your nanoGPT setup
- Submit ablation report from Day 24 experiments
- Include scaling analysis: plot your own loss-vs-compute curves
- **Deliverable:** Trained model, training curves, ablation report, scaling analysis

**Phase II Checkpoint (answer without notes):**
- [ ] Write the full attention equation from memory. What does each term do?
- [ ] Why scale by √d_k? What would happen without it?
- [ ] How does the causal mask work in decoder self-attention?
- [ ] Explain KV cache: what is cached, why, and what's the memory cost?
- [ ] What do Chinchilla scaling laws tell us about training LLMs?
- [ ] Explain BPE tokenization algorithm step by step
- [ ] GPT vs BERT vs T5: when would you use each?
- [ ] Implement multi-head attention on a whiteboard (pseudocode)

---

# PHASE III: LLMs — Training & Alignment (Days 31–44)

> **Goal:** Understand the full LLM pipeline from pretraining to deployment. Know how to fine-tune, align, evaluate, and serve models efficiently.

### Phase Gate
- [ ] Have you trained a GPT from scratch (nanoGPT)?
- [ ] Can you explain autoregressive generation with KV cache?
- [ ] Do you understand scaling laws?

---

## Days 31–44: Detailed Plan

### Day 31: The Modern LLM Recipe
**Theory:** The 3-stage pipeline: pretrain → SFT → alignment. Data quality > quantity — why filtering and deduplication matter. Data mix: the ratio of code, math, multilingual, web text. The role of data curation in determining model capabilities.

### Day 32: Supervised Fine-Tuning (SFT)
**Theory:** Instruction-following datasets (Alpaca, FLAN, OpenHermes). Chat templates: system/user/assistant format. SFT as teaching the model to follow instructions.

**Implementation:** Fine-tune a small LLM (TinyLlama or SmolLM) on an instruction dataset using HuggingFace + LoRA.

### Day 33: RLHF — Reinforcement Learning from Human Feedback
**Theory:** The alignment problem — models can be capable but misaligned. Reward model: trained on human preference pairs. PPO to optimize policy against reward model. Reward hacking, mode collapse. **Note:** you'll learn PPO deeply in Phase VI — here, understand the high-level pipeline.

**Implementation:** Implement a simplified RLHF loop using TRL library.

### Day 34: DPO & Modern Alignment
**Theory:** Direct Preference Optimization — simpler than RLHF, no reward model needed. The DPO loss function (derive it). ORPO, KTO — modern alternatives. Constitutional AI: self-critique.

**Implementation:** Fine-tune a model with DPO using TRL. Compare DPO vs RLHF-tuned model quality.

### Day 35: Efficient Fine-Tuning — LoRA & Friends
**Theory:** Full fine-tuning is expensive and often unnecessary. LoRA: low-rank adaptation (W = W₀ + BA where B, A are low-rank). QLoRA: quantized base + LoRA. Adapter layers, prefix tuning, prompt tuning — compare all approaches.

**Implementation:** Compare: full fine-tune vs LoRA vs QLoRA on the same task. Measure: quality, training time, GPU memory.

### Day 36: Evaluation — How Do We Know If an LLM Is Good?
**Theory:** Perplexity as baseline. Benchmark suites: MMLU, HumanEval, GSM8K, HellaSwag. LLM-as-judge. Chatbot Arena and Elo ratings. The gap between benchmarks and real usefulness. Why evaluation is an unsolved problem.

**Exercise:** Evaluate your fine-tuned model on MMLU subset and compare with base model.

### Day 37: Quantization & Inference
**Theory:** FP32 → FP16 → INT8 → INT4 precision reduction. Post-training quantization (PTQ). GPTQ, AWQ, GGUF formats. Quality vs size tradeoff. Speculative decoding: draft small, verify large. vLLM, TensorRT-LLM for serving.

**Implementation:** Quantize a 7B model to 4-bit with GPTQ. Benchmark: latency, memory, quality degradation.

### Day 38: In-Context Learning as Implicit Optimization (NEW — Sutskever)
**Theory:** Why does few-shot prompting work? Zero-shot, few-shot, many-shot. The transformer as a "soft-attention database" — it has seen similar patterns in training. Mesa-optimization: the model learns a learning algorithm inside its forward pass. In-context learning as implicit Bayesian inference. Connection to compression: a better compressor has seen more patterns and can generalize from fewer examples.

**Exercise:** Systematic experiment — vary number of in-context examples (0, 1, 2, 4, 8, 16) on a classification task. Plot accuracy vs examples. Does it look like a learning curve?

### Day 39: Long Context & Reasoning
**Theory:** Context window scaling: 4K → 128K → 1M+. RoPE scaling (NTK-aware, YaRN). Ring attention for distributed long context. o1-style reasoning: extended thinking, chain-of-thought as "compute at inference time." Test-time compute scaling.

### Day 40: RAG & Tool Use Overview (Compressed from 7→1 day)
**Theory:** RAG architecture: retrieve → augment → generate. When to use RAG vs fine-tuning vs larger context. Tool use and function calling. **Robotics applications:** RAG over robot logs, tool use for calling ROS services, code generation for control. This is a 1-day overview — depth comes from practice, not more days.

### Day 41: LLM for Robotics
**Theory:** LLMs as code generators for robot control (SayCan, Code as Policies). Log analysis with LLMs. LLM-based task planning for robot fleets. Connection to OKS: how would an LLM help with ticket triage, log analysis, navigation debugging?

**Implementation:** Build a simple LLM-based robot command translator: "go to the charging station" → ROS action goal.

### Days 42–44: Phase III Capstone

**Project:** Fine-tune a robotics assistant with LoRA + evaluate
- Fine-tune SmolLM or TinyLlama on robotics Q&A data using LoRA
- Evaluate: compare with base model on robotics questions
- Add RAG over OKS documentation for knowledge grounding
- **Deliverable:** Fine-tuned model, evaluation report, comparison analysis

**Phase III Checkpoint (answer without notes):**
- [ ] Describe the 3-stage LLM training pipeline (pretrain → SFT → alignment)
- [ ] What is LoRA? Why is it more efficient than full fine-tuning? Write the equation.
- [ ] Explain DPO in 3 sentences. How does it differ from RLHF?
- [ ] Why does in-context learning work? (Sutskever's compression interpretation)
- [ ] Why does quantization to 4-bit work without catastrophic quality loss?
- [ ] What is speculative decoding? When is it most helpful?

---

# PHASE IV: Vision — ViT, 3D, Video (Days 45–58)

> **Goal:** Bridge from language to vision. Understand how images, depth, video, and 3D scenes become tokens. By the end, you'll see that the transformer doesn't care what the tokens are — text, image patches, video frames, point clouds — it processes them all the same way.

### Phase Gate
- [ ] Understand transformer architecture deeply
- [ ] Have implemented at least one transformer from scratch
- [ ] Understand scaling laws and their implications

---

### Day 45: ViT — An Image Is Worth 16×16 Words
**Theory:** The key insight: cut image into patches → flatten → linear projection → position embedding → standard transformer encoder. [CLS] token for classification. Comparison with CNNs: missing inductive biases (locality, translation invariance) vs learning from data.

**Implementation:** Implement ViT from scratch in PyTorch. No timm, no pretrained — raw implementation.

### Day 46: Training ViT + DeiT
**Theory:** ViT needs more data than CNNs (lack of inductive bias). Data augmentation: RandAugment, CutMix, MixUp. DeiT: training ViT efficiently with knowledge distillation from a CNN teacher.

**Implementation:** Train your ViT on CIFAR-10. Compare with a ResNet of similar parameter count.

### Day 47: Swin Transformer
**Theory:** Shifted windows for computational efficiency — O(n) instead of O(n²). Patch merging for hierarchical multi-scale features. Why Swin is to ViT what ResNet was to AlexNet.

**Implementation:** Implement Swin Transformer attention (window + shift). Compare memory usage with standard ViT.

### Day 48: DINO & Self-Supervised Vision
**Theory:** Self-distillation with no labels. Student-teacher framework with momentum update. Emerging properties: attention maps segment objects automatically! DINO → DINOv2 (scaled up, stabilized).

**Implementation:** Use pretrained DINOv2 to extract features. Visualize self-attention maps — they show object segmentation for free.

### Day 49: MAE — Masked Autoencoders
**Theory:** Mask 75% of patches → reconstruct. Encoder sees only visible patches (very fast). Decoder reconstructs masked patches. Connection to BERT (masked prediction) and compression (predict the missing information).

**Implementation:** Implement MAE: masking, encoder on visible, decoder to reconstruct. Train on a small image dataset.

### Day 50: 🛑 STOP AND REFLECT #3

> **"Images ARE sequences. Cut an image into patches — now it's a sequence of tokens. The transformer doesn't care if tokens came from text or image patches. This one insight unlocks everything that follows: vision-language models, video understanding, robot actions. The same architecture processes them all."**

**Spend this day on reflection (2.5 hrs):**
- Write: list every type of data that can be turned into a sequence of tokens (text, images, audio, video, actions, point clouds…)
- Reflect: what are the implications of one architecture for all modalities?
- Re-read the ViT paper abstract — does the phrase "an image is worth 16x16 words" hit differently now?
- Connect: MAE proves that vision, like language, benefits from predicting the missing parts (compression again!)

### Day 51: 3D Vision & Depth Estimation (NEW — Li)
**Theory:** Why robots need 3D, not just 2D — the depth dimension is critical for manipulation and navigation. Monocular depth estimation: predict depth from a single image. **Depth Anything v2**: foundation model for depth. Stereo vs mono depth. Connection to robotics: how depth maps feed into obstacle avoidance, grasping, SLAM.

**Implementation:** Run Depth Anything v2 on robot camera images. Compare predicted depth with known distances. Visualize depth maps overlaid on RGB.

### Day 52: Point Clouds & 3D Scene Representations (NEW — Li)
**Theory:** Point clouds: unordered 3D coordinate sets from LiDAR or depth cameras. Point cloud transformers: how to apply attention to 3D points. NeRF concepts: neural radiance fields for novel view synthesis. Multi-view geometry basics. Occupancy networks. Relevance to AMR: how OKS robots build world models from sensor data.

**Implementation:** Process a point cloud with a simple point transformer. Visualize 3D occupancy grid from depth data.

### Day 53: Video Understanding Day 1 (EXPANDED — Li)
**Theory:** Video = image sequence with temporal structure. Temporal modeling approaches: 3D convolutions, temporal transformers. TimeSformer: factorized space-time attention. VideoMAE: masked autoencoding for video.

**Implementation:** Run VideoMAE on a video clip. Extract temporal features, visualize attention across frames.

### Day 54: Video Understanding Day 2 (NEW — Li)
**Theory:** Video-language models: connecting video understanding with language. LLaVA-Video, Qwen-VL for video. Temporal token representations. Long video understanding challenges. Action recognition vs temporal grounding.

**Implementation:** Use a video-language model to describe robot actions from video clips.

### Day 55: Object Detection: DETR + Florence-2 + SAM 2 (EXPANDED — Li)
**Theory:** DETR: end-to-end detection with transformers, no NMS, no anchors. Object queries as learned prompts. **Florence-2** (Microsoft): unified vision foundation model for detection, captioning, grounding. **SAM 2**: universal segmentation for images and video. Open-vocabulary detection.

**Implementation:** Run Florence-2 on warehouse images for detection + captioning. Run SAM 2 for interactive segmentation. Build a detection pipeline for robot camera streams.

### Day 56: Vision-Language Bridge
**Theory:** How to connect visual features to language models — this is the central architectural question. Approaches: (1) early fusion — pixel tokens, (2) late fusion — feature concatenation, (3) cross-attention — queries into visual features, (4) visual tokenizer — convert image features to discrete tokens. This sets up Phase V.

### Days 57–58: Phase IV Capstone

**Project:** Robot perception pipeline: depth + video + detection
- Build a perception pipeline using robot camera data
- Component 1: Depth estimation with Depth Anything v2
- Component 2: Object detection with Florence-2 or DETR
- Component 3: Video understanding — describe what the robot sees over a 10-second clip
- **Deliverable:** Working pipeline, visualizations, analysis of where each component fails

**Phase IV Checkpoint (answer without notes):**
- [ ] How does ViT convert an image into a sequence of tokens?
- [ ] Why does ViT need more data than CNNs? What inductive bias is missing?
- [ ] Explain DINO: how does it learn without labels? Why do attention maps segment?
- [ ] Why is depth estimation critical for robotics? Name 2 approaches.
- [ ] What is Florence-2's key architectural innovation?
- [ ] What are the 4 approaches to fusing vision and language features?

---

# PHASE V: Vision-Language Models (Days 59–70)

> **Goal:** Understand how models learn to see AND speak. Master the architectures that bridge vision and language — these are the direct precursors to VLAs.

### Phase Gate
- [ ] Can you extract features from images using ViT/DINOv2?
- [ ] Do you understand the 4 fusion approaches from Day 56?

---

### Day 59: CLIP — Contrastive Vision-Language Learning
**Theory:** Train image encoder + text encoder jointly via contrastive learning. InfoNCE loss: matching image-text pairs closer, pushing non-matching apart. 400M image-text pairs from the internet. Zero-shot classification: compare image embedding to text embeddings of "a photo of a [class]."

**Implementation:** Use OpenCLIP to: classify images zero-shot, compute image-text similarity, build text-to-image search.

### Day 60: CLIP Internals + SigLIP
**Theory:** Image encoder: ViT-L/14. Text encoder: Transformer. Joint embedding space geometry. Temperature parameter in contrastive loss — learned, not fixed. SigLIP: sigmoid loss instead of softmax — scales better. Why CLIP representations are so broadly useful.

**Implementation:** Implement a minimal CLIP training loop on a small image-text dataset (e.g., Flickr8k).

### Day 61: Flamingo & BLIP-2
**Theory:** Flamingo: Perceiver Resampler compresses visual features to fixed-length. Gated cross-attention: interleave visual tokens with language tokens. BLIP-2: Q-Former bridges frozen image encoder to frozen LLM. Three-stage pretraining. InstructBLIP.

**Implementation:** Use BLIP-2 for visual question answering. Analyze: what does the Q-Former attend to?

### Day 62: LLaVA — Visual Instruction Tuning
**Theory:** Elegantly simple: project CLIP features into LLM space with a linear layer (later MLP). Two-stage training: (1) align vision-language features, (2) instruction-tune on visual conversations. LLaVA-1.5: MLP projector, higher resolution. Why simplicity won.

**Implementation:** Fine-tune a small LLaVA-style model on visual instruction data.

### Day 63: PaLI & CoCa (NEW — Li)
**Theory:** PaLI (Pathways Language and Image model): encoder-decoder VLM at scale. CoCa (Contrastive Captioners): combines contrastive + generative objectives. Why these architectures matter: PaLI is the direct ancestor of RT-2 (Phase VII). CoCa unifies the two dominant VLM training paradigms.

**Exercise:** Compare PaLI, CoCa, LLaVA, BLIP-2 architectures in a table: encoder type, fusion method, training objective, strengths.

### Day 64: Open VLM Landscape — Hands-On (RESTRUCTURED — Li)
**Theory:** Survey of 2025-2026 open VLMs: Qwen-VL, InternVL, Molmo, PaliGemma, Llama-Vision. Architecture differences, capability profiles.

**Implementation:** Run Qwen-VL, InternVL, and Molmo on the SAME set of robotics images. Compare: spatial reasoning, object identification, scene description, counting accuracy.

### Day 65: Spatial Reasoning & Visual Grounding
**Theory:** VLMs struggle with precise spatial reasoning (left/right, above/below, counting). Referring expressions: "the red cup to the left of the plate." Grounding DINO: open-vocabulary grounded detection. SAM + language for segmentation from text.

**Implementation:** Test VLMs on spatial reasoning tasks. Use Grounding DINO for language-guided detection on warehouse images.

### Day 66: 🛑 STOP AND REFLECT #4

> **"VLMs are a unified perception-language interface. The same architecture that answers 'What color is the cup?' can tell a robot 'Pick up the blue cup.' The only missing piece is the action output. Everything you've learned — attention, transformers, scaling, vision, language — converges here."**

**Spend this day on reflection (2.5 hrs):**
- Draw the architecture of RT-2 from what you know so far (before reading the paper). What would YOU do to add action output to a VLM?
- List what you'd need: visual encoder, language model, and an action output head. What format should actions be?
- Preview: look at the RT-2 abstract. Were you close?
- The insight: if you can tokenize actions, the VLM is already a robot controller.

### Days 67–68: Phase V Capstone

**Project:** Warehouse visual QA — compare VLMs on OKS scenarios
- Input: warehouse/robot camera images + natural language questions
- Tasks: "What objects are on the shelf?", "Is the path blocked?", "Where should I place this box?"
- Compare: CLIP zero-shot vs LLaVA vs BLIP-2
- **Deliverable:** Evaluation report with per-task accuracy, failure case analysis

### Days 69–70: VLM Fine-Tuning Practice
**Project:** Fine-tune a small VLM on robotics visual data
- Collect or curate a small robotics visual instruction dataset (~1000 examples)
- Fine-tune LLaVA or PaliGemma with LoRA on your data
- Evaluate: does fine-tuning improve robotics-specific performance?
- **Deliverable:** Fine-tuned model, before/after comparison

**Phase V Checkpoint (answer without notes):**
- [ ] Explain CLIP's contrastive training objective in 3 sentences
- [ ] How does LLaVA connect a vision encoder to an LLM? Draw the architecture.
- [ ] What is the Q-Former in BLIP-2? Why is it needed?
- [ ] What is PaLI, and why does it matter for VLAs?
- [ ] Name 3 challenges VLMs face with spatial reasoning
- [ ] How would you add action output to a VLM? (Your Day 66 answer)

---

# PHASE VI: Robot Learning — RL, Diffusion & Data (Days 71–91)

> **Goal:** Build the robot learning foundations needed for VLAs. This phase was MASSIVELY expanded in v2 based on Finn and Levine's feedback. You'll learn RL (needed for RLHF understanding AND robot learning), diffusion (the dominant action generation paradigm), imitation learning, action representations, data collection, and policy evaluation. LeRobot is the primary framework.

### Phase Gate
- [ ] Understand VLMs (CLIP, LLaVA architecture)
- [ ] Comfortable with your existing robotics knowledge (ROS, control, state estimation)
- [ ] Understand cross-entropy as compression (Day 5 — this thread continues)

---

## Week 11: RL + Diffusion Foundations (Days 71–77)

### Day 71: RL Foundations Day 1 (NEW — Levine)
**Theory:** Markov Decision Processes (MDPs): state, action, reward, transition, discount. Bellman equation: V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]. Value functions: V(s) and Q(s,a). Q-learning: model-free, off-policy. Temporal difference learning. Exploration vs exploitation (ε-greedy).

**Implementation:** Implement Q-learning on a simple grid world. Visualize learned value function.

### Day 72: RL Foundations Day 2 (NEW — Levine)
**Theory:** Policy gradients: REINFORCE algorithm. Directly optimize the policy instead of learning values. Log-probability trick: ∇J = E[∇log π(a|s) R]. High variance problem. Baseline subtraction for variance reduction. Actor-critic: combine value function + policy gradient.

**Implementation:** Implement REINFORCE on CartPole. Add baseline, measure variance reduction.

### Day 73: PPO & Connection to RLHF (NEW — Levine)
**Theory:** Proximal Policy Optimization (PPO): the workhorse of modern RL. Clipped objective: prevent too-large policy updates. Trust region motivation. **The bridge to Phase III:** PPO is exactly the algorithm used in RLHF. Now you understand WHY RLHF works — it's policy optimization with a learned reward model. The reward model predicts human preferences, PPO optimizes the LLM policy.

**Implementation:** Implement PPO on CartPole or LunarLander using gymnasium. Connect: "this is the same algorithm that trained ChatGPT."

### Day 74: Diffusion Models Day 1 — DDPM from Scratch (EXPANDED)
**Theory:** Forward process: gradually add Gaussian noise over T steps. Reverse process: learn to denoise step by step. DDPM loss: predict the noise added at each timestep (simple MSE). U-Net architecture for the denoising network. Noise schedule: linear, cosine.

**Implementation:** Implement DDPM from scratch. Train on MNIST. Visualize the denoising process step by step.

### Day 75: Diffusion Day 2 — DDIM + Classifier-Free Guidance (EXPANDED)
**Theory:** DDIM: faster sampling (50 steps → 10 steps) without retraining. Deterministic sampling. Classifier-free guidance: condition generation on text (or robot observations) — the key technique for conditional generation. Guidance scale tradeoff.

**Implementation:** Implement DDIM sampling for your DDPM model. Add classifier-free guidance on class labels (MNIST digits).

### Day 76: Diffusion Day 3 — Latent Diffusion + Score Matching (EXPANDED — Karpathy)
**Theory:** Latent diffusion: compress to latent space first, then diffuse (Stable Diffusion architecture). Score matching perspective: learn the score function ∇log p(x). Connection to flow matching (Day 77). Why the score perspective is mathematically cleaner. Denoising score matching.

**Implementation:** Use HuggingFace diffusers to run Stable Diffusion. Analyze the latent space. Understand the VAE + U-Net + text encoder pipeline.

### Day 77: Flow Matching
**Theory:** Flow matching as a simpler alternative to diffusion. Straight paths from noise to data (vs. curved diffusion paths). Conditional flow matching. Rectified flows. Why robotics is moving toward flow matching: smoother trajectories, faster sampling, more predictable behavior. π₀ uses flow matching for actions.

**Implementation:** Implement basic flow matching on a 2D distribution. Compare with DDPM: speed, quality, trajectory smoothness.

---

## Week 12: Imitation Learning + Action Spaces (Days 78–84)

### Day 78: Imitation Learning: BC, DAgger, Action Chunking (EXPANDED — Finn)
**Theory:** Behavioral cloning (BC): supervised learning on expert demonstrations. Distribution shift: compounding errors when the policy visits states not in the training data. DAgger: interactively collect expert corrections. **Action chunking**: predict a sequence of future actions, not just one — reduces compounding error. Why action chunking is a key insight for robot learning.

**Implementation:** Train a BC agent in a simple simulation task. Demonstrate distribution shift. Implement action chunking (predict next 10 actions).

### Day 79: ACT — Action Chunking with Transformers (NEW — Finn)
**Theory:** Zhao et al. 2023. Transformer-based imitation learning. CVAE (conditional variational autoencoder) for diverse action generation. Action chunking with temporal ensembling. Why ACT works so well for bimanual manipulation. Architecture: observation encoder → transformer → action chunk.

**Implementation:** Study the ACT codebase. Run ACT on a LeRobot benchmark task. Analyze the action distributions.

### Day 80: Behavior Transformers + Decision Transformer (NEW — Finn)
**Theory:** Decision Transformer (Chen et al. 2021): sequence modeling for RL — cast RL as sequence prediction. Input: (return, state, action, return, state, action, ...). Behavior Transformers (Shafiullah et al. 2022): MinGPT for robot actions. The insight: offline RL as a sequence modeling problem. Connection: this is a precursor to VLAs — predict action tokens conditioned on state tokens.

**Implementation:** Study the Decision Transformer architecture. Run on a D4RL benchmark task. Analyze: how does conditioning on desired return affect behavior?

### Day 81: Diffusion Policy (Chi et al. 2023)
**Theory:** Instead of predicting one action, generate action trajectory via denoising diffusion. Observation conditioning: image + robot proprioception → denoised action sequence. Why diffusion beats Gaussian/deterministic policies for multimodal action distributions (when multiple "correct" actions exist). **LeRobot** as primary framework.

**Implementation:** Train a Diffusion Policy on a LeRobot benchmark task (e.g., PushT). Compare with BC baseline. Visualize generated action trajectories — note the multimodal distribution.

### Day 82: Action Representations Day 1 (EXPANDED)
**Theory:** What IS a robot action? Joint positions, joint velocities, end-effector pose (6-DoF), waypoints, force/torque. Absolute vs delta (relative) actions. Continuous action spaces. The action representation choice profoundly affects learning performance. Normalization and scaling.

**Exercise:** For the OKS AMR, define the action space: (linear velocity, angular velocity) — continuous. For a manipulator: joint positions (7-DoF). Compare: what's easier to learn?

### Day 83: Action Tokenization Day 2 (NEW — Karpathy + Finn)
**Theory:** Discretizing continuous actions into tokens — how RT-2 does it. Uniform binning: divide each action dimension into N bins. VQ-VAE for actions: learn a codebook of action tokens. De-tokenization: converting predicted token back to continuous action. The tradeoff: bin resolution vs vocabulary size. **This is the key bridge from LLMs to VLAs**: if actions are tokens, a language model that predicts action tokens IS a robot controller.

**Implementation:** Implement action tokenization: bin continuous actions into 256 bins per dimension. Implement VQ-VAE for action sequences. Measure: reconstruction error vs number of bins.

### Day 84: 🛑 STOP AND REFLECT #5

> **"Actions are just another token type. A VLM that outputs action tokens IS a VLA. The boundary between language model and robot controller just dissolved. You've now seen: text tokens (Phase II), image patch tokens (Phase IV), and action tokens (today). The transformer processes them all with the same attention mechanism. RT-2 proved this works: take a VLM, add action tokens to its vocabulary, fine-tune on robot data, and it controls a robot."**

**Spend this day on reflection (2.5 hrs):**
- Draw the complete picture: text tokens + image tokens + action tokens → one transformer
- Write: "What does it mean that the same architecture handles language, vision, and action?"
- Predict: what would happen if you scale this up? (More data, more robots, more tasks)
- This is the most important insight in the curriculum. Everything in Phase VII builds on it.

---

## Week 13: Data, Evaluation & Capstone (Days 85–91)

### Day 85: Data Collection for Robot Learning Day 1 (NEW — Finn + Levine)
**Theory:** "Data is 80% of robot learning." Teleoperation methods: VR controllers, kinesthetic teaching, puppeteering, shared autonomy. Data quality metrics: smoothness, task completion, consistency. Dataset formats: RLDS (Reverb), Open X-Embodiment, LeRobot format. What makes a good demonstration: coverage of the task space, recovery behaviors, diverse initial conditions.

**Exercise:** Survey Open X-Embodiment: how many robots, tasks, episodes? What does scale look like for robot data vs language data?

### Day 86: Data Collection Hands-On Day 2 (NEW — Finn + Levine)
**Theory:** Recording demonstrations in simulation: scripted policies, human teleoperation in sim. Data augmentation for robotics: image augmentation, action noise injection, camera viewpoint randomization. Dataset analysis: distribution of actions, state coverage, task success.

**Implementation:** Collect demonstrations for a simulated task using a scripted policy. Curate the dataset: filter failures, balance distribution, apply augmentation. Use LeRobot data pipeline.

### Day 87: Robot Policy Evaluation Methodology (NEW — Levine + Finn)
**Theory:** Success rate is not enough — you need: generalization metrics (novel objects, positions, lighting), failure taxonomy (what TYPE of failure?), statistical significance (how many trials?), real vs sim evaluation gap. Evaluation protocols: held-out objects, random initialization, distractor objects, long-horizon composition.

**Exercise:** Design an evaluation protocol for a pick-and-place policy. Define: success criteria, failure categories, number of trials, generalization tests.

### Day 88: Debugging Learned Policies (NEW — Finn)
**Theory:** When the policy fails: visualization tools (attention maps over observations, action distribution plots). Identifying failure modes: perception failure vs planning failure vs execution failure. Systematic diagnosis: test with ground-truth perception → isolate planning → test with oracle actions. Action distribution inspection: is the policy multimodal when it shouldn't be? Is it confident on the wrong action?

**Implementation:** Take a trained diffusion policy, deliberately introduce a failure (e.g., new distractor object). Diagnose: use attention visualization, action distribution plots, systematic ablation.

### Days 89–91: Phase VI Capstone

**Project:** Train diffusion policy on YOUR OWN collected demonstrations + full evaluation
- Choose a simulated task in MuJoCo/gymnasium
- Collect demonstrations yourself (Day 86 pipeline)
- Train a Diffusion Policy using LeRobot
- Full evaluation: success rate, failure taxonomy, generalization tests
- Compare with BC baseline
- **Deliverable:** Trained policy, evaluation report with statistics, failure analysis, visualization of action distributions

**Phase VI Checkpoint (answer without notes):**
- [ ] Explain PPO's clipped objective. Why is it used for RLHF?
- [ ] What is the REINFORCE policy gradient? Write the loss formula.
- [ ] How does Diffusion Policy generate robot actions? Why is it better than BC for multimodal actions?
- [ ] Explain the forward (noising) and reverse (denoising) process in DDPM. What does the model predict?
- [ ] What is flow matching and how does it differ from diffusion? Why is it faster at inference?
- [ ] What is action tokenization? How does RT-2 convert continuous actions to tokens?
- [ ] Explain ACT: what is action chunking and why does it help?
- [ ] What makes a good robot demonstration dataset? Name 4 criteria.
- [ ] A policy fails on a new object. How do you diagnose: is it perception, planning, or execution?
- [ ] "Actions are just another token type." Explain what this means for VLAs.

---

# PHASE VII: VLAs — Architecture to Deployment (Days 92–112)

> **Goal:** Master the state of the art in Vision-Language-Action models: understand every major architecture, train and deploy VLAs, integrate with classical robotics, and build a complete end-to-end pipeline. This is where everything converges.

### Phase Gate
- [ ] Understand VLMs (CLIP, LLaVA, BLIP-2)
- [ ] Understand diffusion/flow matching for action generation
- [ ] Understand action tokenization (Day 83) and why it matters
- [ ] Have trained a policy from your own demonstrations (Phase VI capstone)

---

## Week 14: VLA Architectures (Days 92–98)

### Day 92: RT-1 — The First Large-Scale Robot Transformer
**Theory:** Architecture: image → FiLM-conditioned EfficientNet → TokenLearner (compress visual tokens) → Transformer decoder → discretized actions. Trained on 130K real robot demos at Google. Generalization to new objects, backgrounds, environments. Why RT-1 matters: proved that scaling transformers works for robotics.

### Day 93: RT-2 — The Leap: VLM → VLA
**Theory:** The breakthrough: take a VLM (PaLI-X or PaRT), fine-tune to output actions AS TEXT TOKENS. Actions as strings: "1 128 91 241 5 101 127." Web knowledge transfers to robotics — a model trained on internet data can figure out novel objects it has never manipulated. Same model does VQA AND robot control. This is Day 84's insight made real.

**Implementation:** Study RT-2's action tokenization scheme in detail. Implement the tokenization/de-tokenization.

### Day 94: Octo — Open-Source Generalist Robot Policy
**Theory:** Transformer-based policy pretrained on Open X-Embodiment (800K+ episodes, 22+ robot types). Task specification: language, goal images, or both. Supports multiple embodiments via embodiment tokens. Fine-tuning recipe for new robots.

**Implementation:** Load Octo, run inference in simulation. Fine-tune on a custom task.

### Day 95: OpenVLA — Open Vision-Language-Action
**Theory:** Based on Prismatic VLM (SigLIP + DINOv2 → Llama backbone). Fine-tuned on Open X-Embodiment. 7B parameters. Action prediction as next-token prediction. Action tokenization with 256 bins per dimension.

**Implementation:** Load OpenVLA, analyze architecture layer by layer, run inference on sample robot observations.

### Day 96: π₀ — Flow Matching for Robot Actions
**Theory:** Physical Intelligence's approach: flow matching (not autoregressive tokens) for action generation. Pretrain vision-language on internet data. Fine-tune action head with flow matching on robot data. Multi-task, multi-robot. Zero-shot generalization. Why flow matching: smoother, faster, handles continuous actions natively.

### Day 97: π₀.₅ — Scaling Physical Intelligence (NEW — Finn)
**Theory:** What happens when you scale π₀: more data, more compute, more robot types. Latest results from Physical Intelligence (2025). Emergent capabilities at scale. Comparison with π₀: what changed? Practical recipes for scaling robot learning.

### Day 98: GR00T N1 + PaLM-E (NEW — Finn + Karpathy)
**Theory:** NVIDIA GR00T N1: VLA for humanoid robots. Architecture, training data, sim-to-real pipeline. PaLM-E (Driess et al. 2023): Google's embodied multimodal model — PaLM language model + ViT visual encoder → robot actions AND language AND vision in one model. Why PaLM-E matters: proved that a single model can handle robot control, visual QA, and planning simultaneously.

---

## Week 15: Training, Transfer & Hybrid Systems (Days 99–105)

### Day 99: GR-2 + Frontier VLAs 2026 (NEW — Karpathy)
**Theory:** GR-2 (Cheang et al. 2024): autoregressive world model + action prediction. Survey of 2025-2026 frontier VLAs: bimanual manipulation, humanoid locomotion, mobile manipulation. What's working, what's not. The gap between demos and reliable deployment.

### Day 100: VLA Training Recipes (Expanded)
**Theory:** Open X-Embodiment dataset in detail: 1M+ episodes, 22+ robot types, 527 tasks. Data mixing strategies: balance between robot types, task types, environments. Curriculum learning for VLAs. Compute requirements: what hardware and how long? Co-training on internet + robot data.

### Day 101: Sim-to-Real Day 1 (EXPANDED — Finn)
**Theory:** Domain gap: simulation ≠ real world (textures, physics, lighting, dynamics). Domain randomization: randomize visual appearance, physics parameters. Visual sim-to-real: style transfer, data augmentation. Tactile and force sim-to-real challenges.

**Implementation:** Train a policy in simulation with domain randomization. Measure: performance in original sim vs randomized sim.

**Hands-on:** [Exercise 12: Robot Manipulation in MuJoCo](exercises/12-robot-manipulation-mujoco.md) covers custom env building, reward engineering, and MuJoCo manipulation. [Exercise 13: Sim-to-Real Transfer](exercises/13-sim-to-real-transfer.md) covers domain randomization, adversarial adaptation, and transfer evaluation. [Exercise 14: Data Collection & Teleoperation](exercises/14-data-collection-teleoperation.md) covers keyboard teleop, HDF5 recording, scripted policies, and data quality filtering. [Exercise 15: Kinematics & Trajectory](exercises/15-kinematics-trajectory.md) covers DH parameters, IK solvers, polynomial interpolation, and trajectory planning for VLA deployment.

### Day 102: Sim-to-Real Day 2 (NEW — Finn)
**Theory:** System identification: tune sim parameters to match real robot. Fine-tuning on small real data: sim pretrain + real fine-tune. When sim-to-real works (visual policies, locomotion) and when it fails (contact-rich manipulation, deformable objects). The "sim-to-real gap" vs "sim-to-real chasm." Hybrid: sim for exploration, real for refinement.

**Implementation:** Take your Day 101 policy. Simulate "real-world" by changing physics parameters. Fine-tune on the "real" domain with 10% of the data. Measure transfer efficiency.

### Day 103: Hybrid VLA + Classical Control Day 1 (EXPANDED — Levine)
**Theory:** VLA as high-level planner, classical controller as low-level executor. Architecture: VLA outputs subgoals or waypoints → classical controller (PID, MPC) tracks them. Why hybrid: VLAs are slow (1-10 Hz), classical control is fast (100-1000 Hz). Safety: classical control enforces constraints that learned policies might violate.

**Exercise:** Design a hybrid architecture for OKS AMR: VLA for task planning ("go to shelf B3, pick item") → classical navigation stack → classical motor control.

### Day 104: Hybrid Systems Day 2 — ROS2 Integration (NEW — Levine)
**Theory:** ROS2 architecture for VLA deployment: VLA node → action server → controller manager → hardware interface. Communication patterns: action servers for long-running tasks, topics for sensor data. Latency considerations: how to pipeline VLA inference with robot execution.

**Implementation:** Build a ROS2 node structure (pseudocode or actual ROS2 if available):
- VLA inference node: takes camera image + language command → outputs action/waypoint
- Action server: converts VLA output to robot commands
- Safety monitor node: checks workspace limits, collision proximity

**Hands-on:** [Project 08: ROS 2 + VLA Deployment Pipeline](projects/08-ros2-vla-pipeline/README.md) — build the full pipeline with policy node, safety gate, action interpolator, and monitoring.

### Day 105: Hybrid Systems Day 3 — Safety & Recovery (NEW — Levine)
**Theory:** TAMP (Task and Motion Planning) + LLM: using LLM for task-level planning with classical motion planning. Safety layers: workspace limits, velocity limits, collision checking, force limits. Recovery behaviors: what to do when the VLA outputs a bad action. When to fall back to classical: if VLA confidence is low, or if in a safety-critical regime. Human-in-the-loop: asking for help when uncertain.

**Exercise:** Design a safety system for a VLA-controlled robot:
1. Define 5 safety constraints (workspace, velocity, force, collision, timeout)
2. Design the fallback behavior for each constraint violation
3. Design a confidence-based handoff between VLA and classical controller

---

## Week 16: Deployment & Capstone (Days 106–112)

### Day 106: World Models & Video Prediction
**Theory:** Learning a model of the world from video. Video prediction models (UniSim, Sora-style) for robotics. Planning in imagination: simulate outcomes before acting. Model-based VLA: use world model to evaluate action candidates. The promise: 10x data efficiency via planning.

### Day 107: Deployment Day 1 — Inference Optimization (EXPANDED — Levine)
**Theory:** Inference latency requirements for real-time control (10-50 Hz). Model quantization for edge deployment: INT8, INT4 on robot hardware (Jetson, etc.). ONNX export, TensorRT optimization. Batching strategies for multi-camera input. Latency profiling: where are the bottlenecks?

**Implementation:** Quantize a VLA model (or surrogate) to INT8. Benchmark: latency on GPU vs edge device. Profile: which component is the bottleneck (vision encoder? LLM backbone? action head?).

### Day 108: Deployment Day 2 — Safety & Monitoring (NEW — Levine)
**Theory:** Runtime safety for deployed VLAs: collision avoidance (always-on, classical), workspace limits (hard constraints), velocity/force limits, human detection and stopping. Monitoring: log VLA predictions, detect distribution shift, alert on out-of-distribution inputs. Graceful degradation: if VLA inference is slow or fails, fall back to a safe behavior.

**Exercise:** Design a monitoring dashboard for a deployed VLA:
- Metrics: inference latency, action magnitude, confidence score, success rate
- Alerts: latency > threshold, action out of bounds, confidence below threshold
- Logging: full observation-action pairs for offline analysis

### Day 109: Deployment Day 3 — Full Pipeline (NEW — Levine)
**Theory:** End-to-end deployment pipeline: sensor input → preprocessing → VLA inference → safety check → command execution → state feedback → logging. ROS2 integration for a warehouse robot. Real-time adaptation: fine-tune the VLA on new data collected during deployment (continual learning). The OKS connection: how would you deploy a VLA on an OKS AMR for dynamic obstacle avoidance or manipulation?

**Exercise:** Write a deployment specification document for a VLA on the OKS AMR:
- Hardware requirements (compute, camera, sensors)
- Software stack (ROS2 nodes, inference server, safety layer)
- Evaluation criteria (latency, success rate, safety violations)
- Monitoring and logging plan
- Rollback procedure if VLA underperforms classical baseline

### Days 110–112: CAPSTONE PROJECT

**Project:** End-to-End VLA Pipeline: data → train → deploy → evaluate
1. **Environment:** Set up a simulated robot task in MuJoCo (pick-and-place or navigation)
2. **Data:** Collect demonstrations using your Phase VI pipeline
3. **Train:** Fine-tune a VLA model (Octo or OpenVLA) on your task
4. **Safety:** Add safety layers (workspace limits, velocity limits, fallback behavior)
5. **Deploy:** Build the inference pipeline with real-time constraints
6. **Evaluate:** Full evaluation protocol from Day 87 — success rate, generalization, failure taxonomy
7. **Compare:** VLA approach vs classical approach on the same task
8. **Report:** 5-page technical report covering architecture, training, deployment, evaluation, and lessons learned

**Deliverable:** Code, trained model, evaluation results, safety analysis, technical report

---

## Final Checkpoint — You're Done When You Can:

- [ ] Explain the complete attention mechanism from first principles, on a whiteboard
- [ ] Describe the transformer architecture (encoder, decoder, both) and when to use each
- [ ] Explain how GPT, BERT, and T5 differ architecturally and in their training objectives
- [ ] Walk through the LLM training pipeline: pretraining → SFT → RLHF/DPO
- [ ] Explain PPO and why it's used for RLHF (Phase VI connection)
- [ ] Explain how ViT processes an image, and how DINO learns without labels
- [ ] Describe 3D depth estimation and why it matters for robots
- [ ] Describe CLIP's contrastive training and zero-shot classification mechanism
- [ ] Explain how LLaVA and PaLI connect vision to language
- [ ] Explain how diffusion models generate samples (forward + reverse process)
- [ ] Explain Diffusion Policy and ACT for robot action generation
- [ ] Describe action tokenization: how continuous robot actions become discrete tokens
- [ ] Describe RT-2's key insight: actions as tokens in a VLM → VLA
- [ ] Compare Octo, OpenVLA, π₀, and π₀.₅ architecturally
- [ ] Design a hybrid VLA + classical control system with safety layers
- [ ] Design a data collection and evaluation protocol for a robot learning task
- [ ] Design a VLA deployment pipeline for a real robot (including ROS2 integration)
- [ ] Explain the connection: compression = prediction = intelligence → scaling laws → VLAs

---

## Appendix A: Key Papers (Reading List)

| # | Paper | Year | Phase | Priority |
|---|-------|------|-------|----------|
| 1 | Attention Is All You Need (Vaswani et al.) | 2017 | II | **MUST READ** |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.) | 2018 | II | HIGH |
| 3 | Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al.) | 2019 | II | HIGH |
| 4 | An Image is Worth 16x16 Words (ViT, Dosovitskiy et al.) | 2020 | IV | **MUST READ** |
| 5 | Language Models are Few-Shot Learners (GPT-3, Brown et al.) | 2020 | III | HIGH |
| 6 | Denoising Diffusion Probabilistic Models (Ho et al.) | 2020 | VI | **MUST READ** |
| 7 | Learning Transferable Visual Models (CLIP, Radford et al.) | 2021 | V | **MUST READ** |
| 8 | Masked Autoencoders Are Scalable Vision Learners (He et al.) | 2021 | IV | HIGH |
| 9 | Emerging Properties in Self-Supervised ViTs (DINO, Caron et al.) | 2021 | IV | HIGH |
| 10 | Decision Transformer (Chen et al.) | 2021 | VI | HIGH |
| 11 | LoRA: Low-Rank Adaptation (Hu et al.) | 2021 | III | HIGH |
| 12 | Training language models to follow instructions (InstructGPT, Ouyang et al.) | 2022 | III | HIGH |
| 13 | Training Compute-Optimal LLMs (Chinchilla, Hoffmann et al.) | 2022 | II | **MUST READ** |
| 14 | FlashAttention (Dao et al.) | 2022 | II | HIGH |
| 15 | RT-1: Robotics Transformer (Brohan et al.) | 2022 | VII | HIGH |
| 16 | CoCa: Contrastive Captioners (Yu et al.) | 2022 | V | MEDIUM |
| 17 | Behavior Transformers (Shafiullah et al.) | 2022 | VI | HIGH |
| 18 | LLaMA: Open and Efficient Foundation Models (Touvron et al.) | 2023 | III | HIGH |
| 19 | Visual Instruction Tuning (LLaVA, Liu et al.) | 2023 | V | **MUST READ** |
| 20 | RT-2: Vision-Language-Action Models (Brohan et al.) | 2023 | VII | **MUST READ** |
| 21 | Diffusion Policy (Chi et al.) | 2023 | VI | **MUST READ** |
| 22 | Direct Preference Optimization (Rafailov et al.) | 2023 | III | HIGH |
| 23 | ACT: Learning Fine-Grained Bimanual Manipulation (Zhao et al.) | 2023 | VI | **MUST READ** |
| 24 | PaLM-E: An Embodied Multimodal Language Model (Driess et al.) | 2023 | VII | HIGH |
| 25 | DINOv2 (Oquab et al.) | 2023 | IV | MEDIUM |
| 26 | Scaling Data-Constrained Language Models (Muennighoff et al.) | 2023 | III | MEDIUM |
| 27 | Octo: An Open-Source Generalist Robot Policy (Team et al.) | 2024 | VII | HIGH |
| 28 | OpenVLA (Kim et al.) | 2024 | VII | **MUST READ** |
| 29 | π₀: A Vision-Language-Action Flow Model (Black et al.) | 2024 | VII | HIGH |
| 30 | GR-2: Generative Robot with World Models (Cheang et al.) | 2024 | VII | MEDIUM |
| 31 | Florence-2 (Microsoft) | 2024 | IV | MEDIUM |
| 32 | Depth Anything v2 | 2024 | IV | MEDIUM |
| 33 | SAM 2: Segment Anything in Images and Videos (Meta) | 2024 | IV | MEDIUM |
| 34 | π₀.₅ (Physical Intelligence) | 2025 | VII | HIGH |
| 35 | GR00T N1 (NVIDIA) | 2025 | VII | MEDIUM |

---

## Appendix B: Agents & Skills for Learning

Use these agents for expert guidance during each phase:

| Phase | Agent/Skill | Purpose |
|-------|-------------|---------|
| All | `learning-guide` | Teach concepts, explain code, Socratic questioning |
| All | `socratic-mentor` | Discovery learning through strategic questions |
| I–III | `python-pro` | PyTorch implementation help |
| II | `deep-research` | Find and synthesize transformer papers |
| III | `ai-engineer` | LLM application patterns |
| V | `ai-engineer` | VLM integration patterns |
| VI–VII | `ml-engineer` | Training pipelines, MLOps |
| VII | (Your robotics expertise) | ROS, control, OKS domain |
| All | `ecc-pytorch-patterns` skill | PyTorch best practices |
| III | `ecc-claude-api` skill | Anthropic API patterns |
| All | `ecc-deep-research` skill | Multi-source research |

### Interactive Learning Pattern

For each topic, use this pattern:
```
1. Read the study notes (theory)
2. Ask the socratic-mentor agent probing questions
3. Implement the code
4. Ask the python-pro agent to review your implementation
5. Run experiments, analyze results
6. Ask the learning-guide agent to quiz you
```

---

## Appendix C: File Structure

```
learn/llm-to-vla/
├── CURRICULUM.md                          ← YOU ARE HERE (v2)
├── EXPERT-REVIEW.md                       ← 5-persona review feedback
├── 00-learning-plan.md                    ← Dependency graph, study order
├── study-notes/
│   ├── 01-dl-foundations.md               ← Phase I: backprop, CNN, RNN, info theory
│   ├── 02-attention-mechanism.md          ← Attention from Bahdanau to multi-head
│   ├── 03-transformer-architecture.md     ← Full transformer, positional encoding
│   ├── 04-transformer-variants.md         ← Efficiency, MoE, normalization, activations
│   ├── 05-gpt-scaling.md                  ← GPT, nanoGPT, tokenization, scaling laws
│   ├── 06-llm-training-alignment.md       ← SFT, RLHF, DPO, LoRA, evaluation
│   ├── 07-llm-engineering.md              ← Quantization, ICL, long context, RAG overview
│   ├── 08-vision-transformers.md          ← ViT, Swin, DINO, MAE
│   ├── 09-3d-video-detection.md           ← Depth, point clouds, video, DETR, Florence-2
│   ├── 10-vision-language-models.md       ← CLIP, LLaVA, BLIP-2, PaLI, CoCa
│   ├── 11-rl-foundations.md               ← MDPs, policy gradients, PPO
│   ├── 12-diffusion-flow.md               ← DDPM, DDIM, latent diffusion, flow matching
│   ├── 13-imitation-learning.md           ← BC, DAgger, ACT, Diffusion Policy, action spaces
│   ├── 14-robot-data-eval.md              ← Data collection, evaluation, debugging policies
│   ├── 15-vla-architectures.md            ← RT-1, RT-2, Octo, OpenVLA, π₀, π₀.₅
│   └── 16-deployment-hybrid.md            ← Sim-to-real, hybrid control, ROS2, deployment
├── exercises/
│   ├── 01-autograd-cnn.md                 ← Phase I: micrograd, ResNet
│   ├── 02-attention-from-scratch.md       ← Build attention step by step
│   ├── 03-build-transformer.md            ← Full transformer implementation
│   ├── 04-nanogpt-ablations.md            ← nanoGPT experiments + scaling
│   ├── 05-finetune-llm.md                 ← LoRA fine-tuning
│   ├── 06-implement-vit.md                ← ViT from scratch
│   ├── 07-3d-depth-video.md               ← Depth estimation, video understanding
│   ├── 08-clip-vlm-experiments.md         ← CLIP zero-shot, VLM comparison
│   ├── 09-rl-diffusion-policy.md          ← RL basics, diffusion for actions
│   ├── 10-imitation-act.md                ← ACT, action tokenization
│   ├── 11-vla-evaluation.md               ← VLA deployment + hybrid systems
│   ├── 12-robot-manipulation-mujoco.md    ← MuJoCo manipulation, reward engineering, custom envs
│   ├── 13-sim-to-real-transfer.md         ← Domain randomization, adversarial adaptation, transfer
│   ├── 14-data-collection-teleoperation.md ← Robot teleoperation, HDF5 recording, data curation
│   └── 15-kinematics-trajectory.md        ← DH params, IK solvers, polynomial trajectories
└── projects/
    ├── 01-rnn-vs-attention/               ← Phase I: character LM comparison
    ├── 02-mini-lm-ablations/              ← Phase II: train LM + ablation report
    ├── 03-robotics-assistant/             ← Phase III: fine-tuned LoRA model
    ├── 04-robot-perception/               ← Phase IV: depth + video + detection pipeline
    ├── 05-warehouse-visual-qa/            ← Phase V: VLM comparison + fine-tuning
    ├── 06-diffusion-policy/               ← Phase VI: own demos → diffusion policy + eval
    ├── 07-vla-capstone/                   ← Phase VII: end-to-end VLA pipeline
    └── 08-ros2-vla-pipeline/              ← Phase VII: ROS 2 deployment with safety & monitoring
```

---

## Appendix D: Weekly Time Commitment

| Component | Daily | Weekly |
|-----------|-------|--------|
| Theory (reading, videos, papers) | 45 min | 5.25 hrs |
| Implementation (coding) | 60 min | 7 hrs |
| Exercises & review | 45 min | 5.25 hrs |
| **Total** | **2.5 hrs** | **17.5 hrs** |

**Total curriculum:** 112 days × 2.5 hrs = **280 hours**

**Weekend option:** If weekdays are tight, do theory on weekdays (45 min × 5 = 3.75 hrs) and implementation + exercises on weekends (2 × 6.5 hrs = 13 hrs).

---

## Appendix E: STOP AND REFLECT Summary

| # | Day | Insight |
|---|-----|---------|
| 1 | 16 | "Attention is a soft dictionary lookup. The transformer is embarrassingly simple — just attention + FFN + residuals, stacked." |
| 2 | 26 | "The same architecture, more data, more compute, keeps getting better along a predictable power law. What does this tell us about learning?" |
| 3 | 50 | "Images ARE sequences. The transformer doesn't care if tokens came from text or image patches. This one insight unlocks everything." |
| 4 | 66 | "VLMs are a unified perception-language interface. The missing piece is the action output." |
| 5 | 84 | "Actions are just another token type. A VLM that outputs action tokens IS a VLA. The boundary between language model and robot controller just dissolved." |

---

*Last updated: 2026-04-28*
*Track: LLM-to-VLA*
*Version: 2.0 (post expert panel review)*
*Status: CURRICULUM REVISED — study notes to be updated to match*
