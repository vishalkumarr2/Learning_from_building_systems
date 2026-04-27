# From Attention to VLA: Zero to Robot Intelligence in 16 Weeks

### 2.5 hours/day × 7 days/week × 16 weeks = ~280 hours
### From "I know ML basics" to "I can train and deploy a Vision-Language-Action model on a robot"

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

---

## Expert Panel Review

This curriculum was designed with input from specialists across relevant domains:

| Expert Role | Contribution |
|-------------|-------------|
| **ML Researcher** | Validated learning progression, ensured mathematical rigor |
| **NLP/LLM Engineer** | Reviewed transformer + LLM sections, recommended key papers |
| **Computer Vision Researcher** | Structured ViT → multimodal progression |
| **Robotics ML Researcher** | Ensured VLA coverage reflects 2025–2026 state of the art |
| **Applied AI Engineer** | Added practical deployment, inference optimization content |
| **Educator** | Structured spaced repetition, checkpoint questions, scaffolded difficulty |

---

## Curriculum Overview

| Phase | Weeks | Topic | Hours | Outcome |
|-------|-------|-------|-------|---------|
| **I** | 1–2 | Deep Learning Refresh + Sequence Models | 35 | Fluent in CNNs, RNNs, seq2seq, ready for attention |
| **II** | 3–5 | Attention & Transformers | 52 | Can implement a transformer from scratch, understand every component |
| **III** | 6–8 | Language Models → LLMs | 52 | Understand GPT, BERT, scaling, RLHF; fine-tune a small LM |
| **IV** | 9–10 | Vision Transformers | 35 | Implement ViT, understand DINO/MAE, bridge vision + transformers |
| **V** | 11–12 | Vision-Language Models | 35 | Understand CLIP, LLaVA, multimodal fusion; fine-tune a VLM |
| **VI** | 13–14 | Diffusion, Action Generation & Robot Learning | 35 | Implement diffusion policy, understand imitation learning |
| **VII** | 15–16 | Vision-Language-Action Models | 35 | Understand RT-2, Octo, OpenVLA, π0; build a VLA pipeline |

---

## Dependency Graph

```
                          ┌─────────────────────────────┐
                          │  Phase VII: VLA Models       │
                          │  RT-2, Octo, OpenVLA, π0    │
                          └──────────────┬──────────────┘
                                         │
                    ┌────────────────────┼───────────────────────┐
                    │                    │                       │
        ┌───────────▼──────────┐  ┌──────▼──────────────┐  ┌────▼──────────────────┐
        │ Phase V: VLMs        │  │ Phase VI: Diffusion  │  │ Your existing         │
        │ CLIP, LLaVA, BLIP   │  │ + Robot Learning     │  │ robotics knowledge    │
        └───────────┬──────────┘  └──────┬──────────────┘  │ (ROS, control, nav)   │
                    │                    │                  └───────────────────────┘
         ┌──────────┴──────────┐         │
         │                     │         │
  ┌──────▼──────────┐  ┌──────▼────────────────┐
  │ Phase III: LLMs │  │ Phase IV: ViTs        │
  │ GPT, BERT, RLHF│  │ ViT, DINO, MAE       │
  └──────┬──────────┘  └──────┬────────────────┘
         │                    │
         └────────┬───────────┘
                  │
       ┌──────────▼──────────────┐
       │ Phase II: Attention &   │
       │ Transformers            │
       └──────────┬──────────────┘
                  │
       ┌──────────▼──────────────┐
       │ Phase I: DL Refresh +   │
       │ Sequence Models         │
       └─────────────────────────┘
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

# Later phases
pip install diffusers  # diffusion models
pip install open_clip_torch  # CLIP
```

### Hardware
- **Minimum**: GPU with 8GB VRAM (RTX 3060/4060) for Phases I–IV
- **Recommended**: GPU with 16–24GB VRAM (RTX 3090/4090) for Phases V–VII
- **Alternative**: Google Colab Pro ($10/month) for GPU access
- **Phase VII**: Access to a robot simulator (Isaac Sim, PyBullet, or MuJoCo)

### Key Textbooks & Resources (Free)
| Resource | Use For | Link |
|----------|---------|------|
| Andrej Karpathy's "Neural Networks: Zero to Hero" | Phases I–III (video lectures) | youtube.com |
| "Attention Is All You Need" (Vaswani 2017) | Phase II core paper | arxiv.org/abs/1706.03762 |
| Jay Alammar's "The Illustrated Transformer" | Phase II visual guide | jalammar.github.io |
| Lilian Weng's blog | All phases — phenomenal overviews | lilianweng.github.io |
| Karpathy's nanoGPT | Phase III hands-on | github.com/karpathy/nanoGPT |
| Karpathy's minbpe | Phase III tokenization | github.com/karpathy/minbpe |
| Hugging Face course | Phases III–V practical | huggingface.co/course |
| "An Image is Worth 16x16 Words" (Dosovitskiy 2020) | Phase IV core paper | arxiv.org/abs/2010.11929 |
| "Learning Transferable Visual Models" (Radford 2021) | Phase V CLIP paper | arxiv.org/abs/2103.00020 |
| "Denoising Diffusion Probabilistic Models" (Ho 2020) | Phase VI core paper | arxiv.org/abs/2006.11239 |
| Stanford CS231n / CS224n / CS25 | Vision / NLP / Transformers | online lectures |
| "RT-2: Vision-Language-Action Models" (Brohan 2023) | Phase VII core paper | arxiv.org/abs/2307.15818 |
| LeRobot (Hugging Face) | Phase VII code | github.com/huggingface/lerobot |

---

# PHASE I: Deep Learning Refresh + Sequence Models (Weeks 1–2)

> **Goal:** Solidify DL foundations and understand the "before transformers" world so you appreciate WHY attention was invented.

### Phase Gate (answer before starting)
- [ ] Can you write a PyTorch training loop from memory?
- [ ] Do you understand backprop through a computation graph?
- [ ] Can you explain what a loss function measures and name 3?

If you fail any of these: spend 2 days on Karpathy's "micrograd" video first.

---

## Week 1: Neural Network Foundations (Days 1–7)

### Day 1: Computation Graphs & Backprop Refresh
**Theory (45 min):**
- Computation graph representation of neural networks
- Chain rule through graphs — why autodiff works
- PyTorch autograd: `.backward()`, `.grad`, `no_grad()`

**Implementation (60 min):**
- Implement a tiny autograd engine (Karpathy's micrograd-style)
- Build: Value class with `+`, `*`, `tanh`, backward

**Exercise (45 min):**
- Hand-compute gradients for a 2-layer network on paper
- Verify with PyTorch autograd

### Day 2: CNN Deep Dive
**Theory (45 min):**
- Convolution as learnable feature extraction
- Receptive field, stride, padding, dilation
- Feature hierarchy: edges → textures → parts → objects
- Residual connections (ResNet) — why they matter

**Implementation (60 min):**
- Build a small ResNet from scratch in PyTorch
- Train on CIFAR-10, visualize learned filters

**Exercise (45 min):**
- Calculate output dimensions for a given conv stack
- Experiment: remove skip connections, measure accuracy drop

### Day 3: RNN & LSTM — The Pre-Transformer World
**Theory (45 min):**
- Vanilla RNN: hidden state, unrolling, backprop through time (BPTT)
- Vanishing/exploding gradients — **the fundamental problem**
- LSTM gates: forget, input, output — how they solve vanishing gradients
- GRU as a simplified LSTM

**Implementation (60 min):**
- Implement vanilla RNN cell from scratch
- Implement LSTM cell from scratch
- Character-level language model on Shakespeare text

**Exercise (45 min):**
- Trace gradients through 10 timesteps of vanilla RNN (show vanishing)
- Compare RNN vs LSTM on a 50-step dependency task

### Day 4: Sequence-to-Sequence Models
**Theory (45 min):**
- Encoder-decoder architecture for translation
- The "bottleneck problem" — entire source sentence compressed to one vector
- Teacher forcing vs free-running decode
- Beam search decoding

**Implementation (60 min):**
- Build an encoder-decoder for simple translation (numbers → words)
- Implement beam search

**Exercise (45 min):**
- Measure BLEU score on your model
- Experiment: vary hidden size, show bottleneck effect on long sequences

### Day 5: Embeddings & Representation Learning
**Theory (45 min):**
- Word embeddings: Word2Vec (skip-gram, CBOW)
- Why one-hot fails; distributed representations
- Embedding arithmetic: king - man + woman ≈ queen
- Subword embeddings (FastText)

**Implementation (60 min):**
- Train Word2Vec from scratch (simplified skip-gram with negative sampling)
- Visualize embeddings with t-SNE

**Exercise (45 min):**
- Query nearest neighbors in embedding space
- Test analogies with your trained embeddings

### Day 6–7: Week 1 Mini-Project
**Project:** Character-level text generator with RNN/LSTM
- Train on a corpus of your choice (code, poetry, Shakespeare)
- Compare: vanilla RNN vs LSTM vs GRU
- Plot training curves, sample generations at different temperatures
- **Deliverable:** Notebook with comparison plots and generated samples

**Checkpoint Questions (answer without notes):**
- [ ] Why does a vanilla RNN struggle with long-range dependencies? Draw the gradient flow.
- [ ] What is the "bottleneck" in seq2seq? Why is it a problem for long sentences?
- [ ] Explain LSTM gates in one sentence each (forget, input, output).
- [ ] Why do residual connections help deep networks train?

---

## Week 2: The Attention Mechanism — Birth of Modern AI (Days 8–14)

### Day 8: Attention Intuition — "Looking Back at the Source"
**Theory (45 min):**
- Motivation: why compressing everything into one vector is bad
- Bahdanau attention (2014): learn to align and translate
- Query, Key, Value intuition — like a soft database lookup
- Attention weights as an alignment visualization

**Implementation (60 min):**
- Add Bahdanau attention to your Day 4 seq2seq model
- Visualize attention weights as a heatmap over source/target tokens

**Exercise (45 min):**
- Compare BLEU scores: seq2seq vs seq2seq+attention
- Analyze attention heatmaps: does the model learn alignment?

### Day 9: Scaled Dot-Product Attention
**Theory (45 min):**
- Dot-product attention: `score = Q · K^T`
- Why scale by `√d_k` — preventing softmax saturation
- Attention as weighted sum of values
- Full equation: `Attention(Q, K, V) = softmax(Q K^T / √d_k) V`
- Matrix view: attention operates on entire sequences simultaneously

**Implementation (60 min):**
- Implement `scaled_dot_product_attention()` from scratch
- Test on random tensors, verify dimensions
- Compare with `torch.nn.functional.scaled_dot_product_attention`

**Exercise (45 min):**
- By hand: compute attention for a 3-token sequence with d_k=2
- What happens without the scaling? Show softmax becoming one-hot

### Day 10: Multi-Head Attention
**Theory (45 min):**
- Why one attention head isn't enough — different heads attend to different things
- Multi-head: split Q, K, V → attend independently → concatenate → project
- `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O`
- Each `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`
- Interpreting heads: positional, syntactic, semantic

**Implementation (60 min):**
- Implement `MultiHeadAttention` class from scratch
- No `nn.MultiheadAttention` — build it yourself

**Exercise (45 min):**
- Verify parameter count matches theory
- Visualize different heads' attention patterns on a sample sequence

### Day 11: Positional Encoding
**Theory (45 min):**
- The problem: attention is permutation-equivariant — it doesn't know token order
- Sinusoidal positional encoding (Vaswani 2017)
- `PE(pos, 2i) = sin(pos / 10000^(2i/d))` — why this formula works
- Learned positional embeddings
- Rotary Position Embeddings (RoPE) — the modern approach

**Implementation (60 min):**
- Implement sinusoidal PE from scratch
- Visualize the encoding matrix as a heatmap
- Implement RoPE (used in LLaMA, modern models)

**Exercise (45 min):**
- Show that sinusoidal PE can represent relative positions via dot products
- Compare sinusoidal vs learned PE on a toy task

### Day 12: The Full Transformer — "Attention Is All You Need"
**Theory (45 min):**
- Read the paper (or Illustrated Transformer guide)
- Encoder block: Multi-Head Attention → Add & Norm → FFN → Add & Norm
- Decoder block: Masked Multi-Head Attention → Cross-Attention → FFN
- Layer normalization: Pre-LN vs Post-LN
- Feed-forward network: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`

**Implementation (60 min):**
- Implement `TransformerEncoderLayer` from scratch
- Implement `TransformerDecoderLayer` from scratch (with causal mask)
- Stack into full Transformer model

**Exercise (45 min):**
- Count parameters for a given transformer config (layers, d_model, n_heads)
- Trace a single token through the entire forward pass

### Day 13: Training a Transformer on Translation
**Theory (45 min):**
- Label smoothing — why it helps generalization
- Learning rate warmup — why transformers need it
- The "noam" schedule: `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`
- Masked loss for padding tokens

**Implementation (60 min):**
- Train your Day 12 transformer on a small translation dataset (e.g., Multi30k EN→DE)
- Implement warmup scheduler
- Log to wandb

**Exercise (45 min):**
- Compare with your Day 4 seq2seq — how much better is the transformer?
- Experiment: vary number of heads, layers; measure BLEU

### Day 14: Week 2 Mini-Project + Checkpoint
**Project:** Build a complete annotated transformer
- Clean, documented implementation from scratch
- Train on a task of your choice (translation, summarization, classification)
- Visualize: attention patterns across all heads and layers
- **Deliverable:** Notebook + a blog-style writeup of "What I learned building a transformer"

**Checkpoint Questions (answer without notes):**
- [ ] Write the full attention equation from memory. What does each term do?
- [ ] Why scale by √d_k? What would happen without it?
- [ ] How does the causal mask work in decoder self-attention?
- [ ] What is the purpose of multi-head attention? How many parameters does it add?
- [ ] Why do transformers need positional encoding but RNNs don't?
- [ ] Draw the full encoder block. Label every component.

---

# PHASE II: Attention & Transformers — Advanced (Week 3–5)

> **Goal:** Go deep on transformer variants, efficiency, and the design choices that led to modern LLMs.

### Phase Gate
- [ ] Can you implement multi-head attention from memory?
- [ ] Can you explain the full transformer encoder/decoder architecture?

---

## Week 3: Transformer Variants & Efficiency (Days 15–21)

### Day 15: Self-Attention vs Cross-Attention
**Theory:** Self-attention (Q=K=V from same sequence), cross-attention (Q from decoder, K,V from encoder). When each is used.

**Implementation:** Build both, compare attention maps.

### Day 16: Efficient Attention — The O(n²) Problem
**Theory:** Standard attention is O(n²) in sequence length. Sparse attention (local windows, strided, random). Flash Attention — IO-aware exact attention (Dao et al.). Linear attention approximations.

**Implementation:** Implement sliding-window attention. Benchmark Flash Attention vs naive.

### Day 17: KV Cache — Why LLM Inference Is Fast
**Theory:** Autoregressive generation: only new token needs to compute Q. Cache K, V from all previous tokens. Memory vs compute tradeoff. Paged attention (vLLM).

**Implementation:** Implement KV cache for autoregressive decoding. Measure speedup.

### Day 18: Normalization & Residual Patterns
**Theory:** Post-LN (original) vs Pre-LN (modern) — training stability. RMSNorm (simpler than LayerNorm, used in LLaMA). DeepNorm for very deep transformers.

**Implementation:** Compare Pre-LN vs Post-LN training stability on a deep model.

### Day 19: Activation Functions in Transformers
**Theory:** ReLU → GELU → SwiGLU. Why GLU variants dominate modern LLMs. SwiGLU: `SwiGLU(x) = (xW₁ ⊙ σ(xW_gate)) W₂`.

**Implementation:** Swap activations in your transformer, compare convergence.

### Day 20: Mixture of Experts (MoE)
**Theory:** Not all parameters active for every token. Router selects top-k experts. Load balancing loss. Switch Transformer, Mixtral architecture.

**Implementation:** Implement a simple MoE layer with top-2 routing.

### Day 21: Week 3 Mini-Project
**Project:** Efficient transformer library
- Implement a configurable transformer supporting: Pre-LN/Post-LN, RMSNorm, SwiGLU, KV cache, sliding window attention.
- Benchmark configurations on a language modeling task.

---

## Week 4: Understanding BERT & Encoder Models (Days 22–28)

### Day 22: BERT — Bidirectional Encoders
**Theory:** Masked Language Model (MLM) objective. Next Sentence Prediction (NSP). Why bidirectional > left-to-right for understanding tasks.

**Implementation:** Implement MLM pretraining on a small corpus.

### Day 23: Fine-Tuning BERT
**Theory:** [CLS] token for classification. Task-specific heads. Feature extraction vs fine-tuning all layers.

**Implementation:** Fine-tune BERT-base on sentiment classification (SST-2) using HuggingFace.

### Day 24: Sentence Embeddings
**Theory:** From word embeddings to sentence embeddings. Sentence-BERT, contrastive learning for embeddings. Cosine similarity search.

**Implementation:** Build a semantic search engine over a document corpus.

### Day 25: Tokenization Deep Dive
**Theory:** Character-level vs word-level vs subword. Byte Pair Encoding (BPE) — the algorithm step by step. WordPiece, Unigram, SentencePiece. tiktoken (GPT tokenizer). Why tokenization matters for multilingual, code, numbers.

**Implementation:** Implement BPE from scratch (Karpathy's minbpe). Train on a corpus, inspect vocabulary. Compare with tiktoken.

### Day 26: Tokenization Trade-offs
**Theory:** Vocabulary size vs sequence length tradeoff. Tokenization for code, math, non-English. Byte-level fallback. Token healing. "How many Rs in strawberry?" — tokenizer artifacts.

**Implementation:** Analyze tokenization of code, math, Japanese text across different tokenizers.

### Day 27–28: Week 4 Mini-Project
**Project:** Build a document Q&A system
- Encode documents with a sentence transformer
- Store embeddings in a vector database (FAISS or ChromaDB)
- Query with natural language, retrieve relevant passages
- **Deliverable:** Working retrieval system + analysis of failure cases

---

## Week 5: Understanding GPT & Decoder Models (Days 29–35)

### Day 29: GPT Architecture — Decoder-Only Transformers
**Theory:** Autoregressive language modeling: P(token_t | token_1:t-1). Causal mask → left-to-right only. GPT-1 → GPT-2 → GPT-3 progression. Why decoder-only won over encoder-decoder for generation.

**Implementation:** Read and annotate Karpathy's nanoGPT code. Understand every line.

### Day 30: Building nanoGPT
**Theory:** Training a GPT from scratch on Shakespeare. Data loading, batching, context window. Loss curves and what they tell you.

**Implementation:** Train nanoGPT on Shakespeare (follow Karpathy's tutorial). Modify: try different datasets (code, Wikipedia). Experiment: vary n_layers, n_heads, d_model.

### Day 31: Sampling Strategies
**Theory:** Greedy, top-k, top-p (nucleus), temperature. Repetition penalty. Beam search for text generation. Structured generation (constrained decoding).

**Implementation:** Implement all sampling strategies. Compare: sample the same prompt with different settings, analyze quality.

### Day 32: Scaling Laws — "Bigger Is Better (If You Know How)"
**Theory:** Kaplan scaling laws (2020): loss scales as power law with compute, data, params. Chinchilla (2022): optimal data/param ratio — most models were undertrained! Emergent abilities at scale — or are they a mirage?

**Exercise:** Plot scaling curves from published data. Estimate compute for training a 7B model.

### Day 33: Distributed Training Basics
**Theory:** Data parallelism (DP, DDP). Model parallelism (tensor, pipeline). ZeRO optimization (DeepSpeed). Mixed precision training (FP16, BF16).

**Implementation:** Train a model with PyTorch DDP on 2 GPUs (or simulate). Compare training speed.

### Day 34–35: Week 5 Mini-Project + Phase Checkpoint
**Project:** Train your own mini-LM
- Choose a domain corpus (code, medical, legal, robotics papers)
- Train a ~10M parameter GPT from scratch
- Evaluate: perplexity, sample quality, downstream task performance
- **Deliverable:** Trained model, training curves, sample outputs, analysis

**Phase II Checkpoint (answer without notes):**
- [ ] What are the two pretraining objectives in BERT?
- [ ] Explain BPE tokenization algorithm step by step
- [ ] GPT uses only the decoder. What does it mask and why?
- [ ] What do Chinchilla scaling laws tell us about most LLMs?
- [ ] Explain KV cache: what is cached, why, and what's the memory cost?
- [ ] Implement multi-head attention on a whiteboard (pseudocode)

---

# PHASE III: Large Language Models (Weeks 6–8)

> **Goal:** Understand what makes LLMs "large" — training recipes, alignment, and practical engineering.

### Phase Gate
- [ ] Have you trained a GPT from scratch?
- [ ] Can you explain autoregressive generation with KV cache?

---

## Week 6: LLM Training & Alignment (Days 36–42)

### Day 36: The Modern LLM Training Recipe
**Theory:** Pretraining → Supervised Fine-Tuning (SFT) → Alignment. Data quality > data quantity. Data deduplication, filtering. The importance of data mix (code, math, multilingual).

### Day 37: Supervised Fine-Tuning (SFT)
**Theory:** Instruction-following datasets (Alpaca, FLAN). Chat format and templates. SFT as the "instruction following" step.

**Implementation:** Fine-tune a small LLM (TinyLlama / SmolLM) on an instruction dataset using HuggingFace + LoRA.

### Day 38: RLHF — Reinforcement Learning from Human Feedback
**Theory:** The alignment problem — models can be capable but harmful. Reward model: trained on human preference data. PPO to optimize policy against reward model. Problems: reward hacking, mode collapse.

**Implementation:** Implement a simplified RLHF loop using TRL library.

### Day 39: DPO & Modern Alignment
**Theory:** Direct Preference Optimization — simpler than RLHF, no reward model. ORPO, KTO, IPO — the alignment zoo. Constitutional AI: self-critique and revision.

**Implementation:** Fine-tune a model with DPO using TRL.

### Day 40: Efficient Fine-Tuning — LoRA & Adapters
**Theory:** Full fine-tuning is expensive. LoRA: low-rank adaptation of weight matrices. QLoRA: quantized LoRA (4-bit training). Adapter layers: inserted trainable modules. Prefix tuning, prompt tuning.

**Implementation:** Compare: full fine-tune vs LoRA vs QLoRA on the same task. Measure: quality, training time, GPU memory.

### Day 41: Evaluation — How Do We Know If an LLM Is Good?
**Theory:** Perplexity as a baseline. Benchmark suites: MMLU, HumanEval, GSM8K, HellaSwag. LLM-as-judge evaluation. Chatbot Arena and Elo ratings. The gap between benchmarks and real usefulness.

**Exercise:** Evaluate your fine-tuned model on MMLU (subset) and compare with base model.

### Day 42: Week 6 Mini-Project
**Project:** Fine-tune a robotics assistant
- Fine-tune SmolLM or TinyLlama on robotics Q&A data
- Use LoRA for efficient training
- Evaluate on robotics-specific questions
- **Deliverable:** Fine-tuned model, comparison with base model

---

## Week 7: LLM Engineering & Applications (Days 43–49)

### Day 43: In-Context Learning & Prompting
**Theory:** Zero-shot, few-shot, many-shot learning. Chain-of-thought (CoT) prompting. Self-consistency: sample multiple CoT, take majority vote. ReAct: reasoning + acting.

**Implementation:** Build a prompt engineering testbench. Compare zero-shot vs CoT on math problems.

### Day 44: Retrieval-Augmented Generation (RAG)
**Theory:** LLMs have knowledge cutoff and hallucinate. RAG: retrieve relevant docs → prepend to context → generate. Embedding models for retrieval. Chunk size, overlap, metadata filtering.

**Implementation:** Build a RAG system from scratch (not using LangChain). Components: chunker, embedder, retriever, generator.

### Day 45: Tool Use & Function Calling
**Theory:** LLMs as reasoning engines that invoke tools. Function calling format (OpenAI, Anthropic). Structured output (JSON mode). Planning: breaking complex tasks into tool calls.

**Implementation:** Build an agent that can: search the web, run Python code, query a database.

### Day 46: Quantization — Shrinking Models
**Theory:** FP32 → FP16 → INT8 → INT4. Post-training quantization (PTQ). GPTQ, AWQ, GGUF. Quality vs size tradeoff at each quantization level.

**Implementation:** Quantize a 7B model to 4-bit with GPTQ. Benchmark: latency, memory, quality.

### Day 47: Inference Optimization
**Theory:** Speculative decoding: draft with small model, verify with large. Continuous batching. Tensor parallelism for multi-GPU. vLLM, TensorRT-LLM, llama.cpp.

**Implementation:** Deploy a model with vLLM. Benchmark throughput.

### Day 48–49: Week 7 Mini-Project
**Project:** Build a complete AI assistant
- RAG over your OKS robotics documentation
- Tool use: can search code, query logs, run analysis scripts
- Evaluate: accuracy on real robotics troubleshooting questions
- **Deliverable:** Working assistant, evaluation results

---

## Week 8: Advanced LLM Topics (Days 50–56)

### Day 50: Long Context — Beyond 4K Tokens
**Theory:** Context window scaling: 4K → 128K → 1M+. RoPE scaling (NTK-aware). Ring attention, landmark attention. Needle-in-a-haystack evaluation.

### Day 51: Reasoning Models
**Theory:** o1/o3-style reasoning: extended thinking. Chain-of-thought as "compute at inference time." Reward models for process supervision. Test-time compute scaling.

### Day 52: Multi-Turn Conversations & Memory
**Theory:** Conversation management, context window stuffing. External memory: summarization, retrieval. Episodic vs semantic memory in agents.

### Day 53: LLM Security & Safety
**Theory:** Prompt injection, jailbreaking. Red-teaming and safety evaluation. Output filtering, content policy enforcement.

### Day 54–56: Phase III Capstone + Checkpoint
**Project:** LLM-powered OKS issue analyzer
- Input: ticket description + logs
- RAG over knowledge base
- Chain-of-thought for root cause hypothesis
- Compare with your manual analysis workflow
- **Deliverable:** Working prototype, accuracy evaluation on 10 real tickets

**Phase III Checkpoint (answer without notes):**
- [ ] Describe the 3-stage LLM training pipeline (pretrain → SFT → alignment)
- [ ] What is LoRA? Why is it more efficient than full fine-tuning?
- [ ] Explain DPO in 3 sentences. How does it differ from RLHF?
- [ ] What are the components of a RAG system? Draw the architecture.
- [ ] Why does quantization to 4-bit work without catastrophic quality loss?
- [ ] What is speculative decoding? When is it most helpful?

---

# PHASE IV: Vision Transformers (Weeks 9–10)

> **Goal:** Bridge from language to vision — understand how images become tokens.

### Phase Gate
- [ ] Understand transformer architecture deeply
- [ ] Have implemented at least one transformer from scratch

---

## Week 9: Vision Transformer Foundations (Days 57–63)

### Day 57: ViT — An Image Is Worth 16×16 Words
**Theory:** The key insight: cut image into patches → flatten → linear projection → position embedding → standard transformer encoder. [CLS] token for classification. Comparison with CNNs: inductive biases (locality, translation invariance) vs learned from data. Paper: Dosovitskiy et al. 2020.

**Implementation:** Implement ViT from scratch in PyTorch. No timm, no pretrained — raw implementation.

### Day 58: Training ViT
**Theory:** ViT needs more data than CNNs (lack of inductive bias). Data augmentation strategies: RandAugment, CutMix, MixUp. DeiT: training ViT efficiently with distillation.

**Implementation:** Train your ViT on CIFAR-10. Compare with a ResNet of similar size.

### Day 59: Hierarchical Vision Transformers
**Theory:** Swin Transformer: shifted windows for efficiency. Patch merging for multi-scale features. Why Swin is to ViT what ResNet was to AlexNet.

**Implementation:** Implement Swin Transformer attention (window + shift).

### Day 60: Self-Supervised Vision — DINO
**Theory:** Self-distillation with no labels. Student-teacher framework with momentum. Emerging properties: attention maps segment objects! DINO → DINOv2 (scaled up).

**Implementation:** Use pretrained DINOv2 to extract features. Visualize self-attention maps — they show object segmentation for free.

### Day 61: Masked Autoencoders (MAE)
**Theory:** Mask 75% of patches → reconstruct. Encoder sees only visible patches (fast). Decoder reconstructs masked patches. Incredibly efficient self-supervised pretraining.

**Implementation:** Implement MAE: masking, encoder on visible patches, decoder to reconstruct.

### Day 62: Object Detection with Transformers — DETR
**Theory:** End-to-end object detection, no NMS, no anchors. Bipartite matching for loss computation. Object queries as learned prompts. Transformer decoder as the detection head.

**Implementation:** Use pretrained DETR, visualize detections and attention to objects.

### Day 63: Week 9 Mini-Project
**Project:** Build a visual feature extractor for robotics
- Use DINOv2 to extract features from robot camera images
- Build a zero-shot object classifier using DINO features + kNN
- Compare with a CNN baseline
- **Deliverable:** Feature extraction pipeline, comparison analysis

---

## Week 10: Advanced Vision + Bridge to Multimodal (Days 64–70)

### Day 64: Vision-Language Connection — How to Merge Two Worlds
**Theory:** The bridge problem: how do you connect visual features to language models? Options: early fusion (pixel tokens), late fusion (feature concat), cross-attention. Visual tokenizers: turning images into "language" the LLM can read.

### Day 65: Image Captioning Evolution
**Theory:** Show and Tell (CNN+LSTM) → Show, Attend and Tell (attention) → Transformer captioning → Modern VLMs. Each step removed a bottleneck.

**Implementation:** Implement Show, Attend and Tell with attention visualization.

### Day 66: Contrastive Learning — Foundations for CLIP
**Theory:** Learning representations by contrasting positive vs negative pairs. InfoNCE loss. SimCLR, MoCo — image-image contrastive. Setting the stage for CLIP: image-text contrastive.

**Implementation:** Implement contrastive learning on CIFAR-10. Build a linear probe on learned representations.

### Day 67–68: Phase IV Mini-Project + Checkpoint
**Project:** Build a visual search engine
- Encode images with ViT/DINOv2
- Query with text descriptions (using a text encoder)
- Return similar images
- **Deliverable:** Working visual search, analysis of successes and failures

**Phase IV Checkpoint (answer without notes):**
- [ ] How does ViT convert an image into a sequence of tokens?
- [ ] Why does ViT need more data than CNNs? What inductive bias is missing?
- [ ] Explain the DINO training procedure. Why do attention maps segment objects?
- [ ] What does MAE mask, and why is 75% masking efficient?
- [ ] What are the 3 main approaches to fusing vision and language features?

---

# PHASE V: Vision-Language Models (Weeks 11–12)

> **Goal:** Understand how models learn to see AND speak — the multimodal revolution.

### Phase Gate
- [ ] Can you extract features from images using a ViT?
- [ ] Do you understand contrastive learning (InfoNCE)?

---

## Week 11: CLIP & Contrastive VLMs (Days 69–77)

### Day 69: CLIP — Connecting Images and Text
**Theory:** Train image encoder and text encoder jointly. Contrastive loss: matching image-text pairs closer, pushing non-matching apart. 400M image-text pairs from the internet. Zero-shot classification: compare image embedding to text embeddings of "a photo of a [class]."

**Implementation:** Use OpenCLIP to: classify images zero-shot, compute image-text similarity, build a text-to-image search system.

### Day 70: CLIP Internals
**Theory:** Image encoder: ViT-L/14. Text encoder: Transformer. Joint embedding space. Temperature parameter in contrastive loss. Why CLIP representations are so useful for downstream tasks.

**Implementation:** Implement a minimal CLIP training loop on a small image-text dataset.

### Day 71: SigLIP & Improved Contrastive Models
**Theory:** SigLIP: sigmoid loss instead of softmax — scales better. Patchwise contrastive learning. Region-level alignment.

### Day 72: Flamingo — Few-Shot Visual Understanding
**Theory:** Perceiver Resampler: compress visual features into fixed number of tokens. Gated cross-attention: interleave visual tokens with language tokens. In-context learning with images.

### Day 73: BLIP-2 — Efficient Vision-Language Pretraining
**Theory:** Q-Former: bridge between frozen image encoder and frozen LLM. Three-stage pretraining. Why freezing both encoders is efficient. InstructBLIP: instruction-tuned BLIP-2.

**Implementation:** Use BLIP-2 for visual question answering. Analyze: what does the Q-Former attend to?

### Day 74: LLaVA — Visual Instruction Tuning
**Theory:** Simple and effective: project CLIP features into LLM space with a linear layer. Two-stage training: (1) align vision-language, (2) instruction tune. LLaVA-1.5: MLP projector, higher resolution.

**Implementation:** Fine-tune a small LLaVA-style model on visual instruction data.

### Day 75–77: Week 11 Mini-Project
**Project:** Build a robotics visual assistant
- Input: robot camera image + natural language question
- "What objects are on the shelf?" "Is the path blocked?" "Where should I place this box?"
- Use CLIP or LLaVA as the backbone
- **Deliverable:** Working visual QA for robotics scenarios

---

## Week 12: Advanced VLMs + Grounding (Days 78–84)

### Day 78: Visual Grounding — Seeing What You're Talking About
**Theory:** Referring expression comprehension: "the red cup on the left." Grounding DINO: open-vocabulary object detection. Segment Anything (SAM) + language.

### Day 79: Multi-Image & Video Understanding
**Theory:** Processing multiple images (comparisons, before/after). Video tokenization: temporal patches. Video-LLMs: LLaVA-Video, Qwen-VL.

### Day 80: Spatial Reasoning in VLMs
**Theory:** VLMs struggle with precise spatial reasoning. Coordinate prediction, bounding box output. Set-of-Mark prompting. SpatialVLM.

### Day 81: Open-Source VLM Landscape (2025–2026)
**Theory:** Survey of leading open VLMs: Qwen-VL, InternVL, Llama-Vision, PaliGemma, Molmo. Architecture comparison, capability analysis, robotics applicability.

### Day 82–84: Phase V Capstone + Checkpoint
**Project:** Warehouse scene understanding
- Input: images from a warehouse (or OKS simulation)
- Tasks: identify objects, estimate layout, answer questions about the scene
- Compare: CLIP zero-shot vs LLaVA vs BLIP-2
- **Deliverable:** Evaluation report, failure case analysis

**Phase V Checkpoint (answer without notes):**
- [ ] Explain the CLIP training objective in 3 sentences
- [ ] How does LLaVA connect a vision encoder to an LLM?
- [ ] What is the Q-Former in BLIP-2? Why is it needed?
- [ ] What is "zero-shot classification" with CLIP? How does it work?
- [ ] Name 3 challenges VLMs face with spatial reasoning

---

# PHASE VI: Diffusion, Action Generation & Robot Learning (Weeks 13–14)

> **Goal:** Understand how generative models produce actions — the bridge from "seeing and thinking" to "doing."

### Phase Gate
- [ ] Understand VLMs (CLIP, LLaVA architecture)
- [ ] Comfortable with your existing robotics knowledge (ROS, control, state estimation)

---

## Week 13: Diffusion Models & Robot Learning Foundations (Days 85–91)

### Day 85: Diffusion Models — Basics
**Theory:** Forward process: add noise gradually until pure Gaussian. Reverse process: learn to denoise. DDPM: the denoising diffusion probabilistic model. Loss: predict the noise added at each timestep. U-Net architecture for denoising.

**Implementation:** Implement DDPM from scratch. Train on MNIST. Visualize the denoising process.

### Day 86: Diffusion — Advanced
**Theory:** DDIM: faster sampling (fewer steps). Classifier-free guidance: control generation with text. Latent diffusion: diffuse in a compressed latent space (Stable Diffusion). Score matching perspective.

**Implementation:** Implement classifier-free guidance. Compare DDPM vs DDIM sampling.

### Day 87: Flow Matching
**Theory:** Flow matching as a simpler alternative to diffusion. Straight paths from noise to data. Conditional flow matching. Rectified flows. Why flow matching is gaining traction in robotics.

**Implementation:** Implement basic flow matching on a 2D distribution.

### Day 88: Imitation Learning — Learning from Demonstrations
**Theory:** Behavioral cloning: supervised learning on expert demos. Distribution shift (compounding errors). DAgger: dataset aggregation. Action chunking: predict a sequence of actions, not just one.

**Implementation:** Build a behavioral cloning agent in a simple environment (CartPole or Reacher).

### Day 89: Diffusion Policy — "Diffusion for Robot Actions"
**Theory:** Key paper: Chi et al. 2023. Instead of predicting one action, generate action trajectory via diffusion. Observation conditioning: image + robot state → denoised action sequence. Why diffusion beats Gaussian policies for multimodal action distributions.

**Implementation:** Implement Diffusion Policy on a simple 2D pushing task. Use the LeRobot library.

### Day 90: Action Representations for Robots
**Theory:** What IS an action? Joint positions, velocities, end-effector pose, waypoints. Continuous vs discrete actions. Action tokenization for transformers. Binning, de-tokenization.

### Day 91: Week 13 Mini-Project
**Project:** Diffusion policy for a simulated task
- Environment: MuJoCo pushing task or similar
- Train a diffusion policy from demonstrations
- Compare with behavioral cloning baseline
- **Deliverable:** Trained policy, success rate comparison, visualization of generated trajectories

---

## Week 14: Foundation Models for Robotics (Days 92–98)

### Day 92: RT-1 — Robotics Transformer
**Theory:** First large-scale robot transformer. Architecture: image → FiLM-conditioned EfficientNet → TokenLearner → Transformer decoder → discretized actions. Trained on 130K real robot demos at Google. Generalization to new objects, tasks.

### Day 93: RT-2 — Vision-Language-Action
**Theory:** The leap: take a VLM (PaLI-X) → fine-tune to output actions as text tokens. Actions as strings: "1 128 91 241 5 101 127." Transfer web knowledge to robotics. Same model does VQA AND robot control.

### Day 94: Octo — Open-Source Generalist Robot Policy
**Theory:** Transformer-based policy pre-trained on Open X-Embodiment. Supports multiple robot embodiments. Task specification: language, goal images, or both. Fine-tuning recipe for new robots.

**Implementation:** Use Octo from HuggingFace. Run inference in simulation.

### Day 95: OpenVLA — Open Vision-Language-Action
**Theory:** Based on Prismatic VLM (SigLIP + DINOv2 + Llama backbone). Fine-tuned on Open X-Embodiment dataset. 7B parameters. Action prediction as next-token prediction.

**Implementation:** Load OpenVLA, analyze architecture, run inference on sample robot observations.

### Day 96: π0 (Physical Intelligence) & Frontier VLAs
**Theory:** π0: flow matching for action generation (not autoregressive). Pre-trained on internet data, fine-tuned on robot data. Multi-task, multi-robot, zero-shot generalization. Other frontiers: GR-2, Humanoid VLAs.

### Day 97–98: Week 14 Mini-Project
**Project:** Evaluate VLA models
- Set up a simulated robot environment (SIMPLER or equivalent)
- Run multiple VLA models (Octo, OpenVLA)
- Compare: success rate, generalization, inference speed
- **Deliverable:** Evaluation report with benchmarks

---

# PHASE VII: Vision-Language-Action — Mastery (Weeks 15–16)

> **Goal:** Deep understanding of VLA architectures, training, and deployment for real robotics.

### Phase Gate
- [ ] Understand both VLMs and diffusion/flow matching for actions
- [ ] Have run at least one VLA model in simulation

---

## Week 15: VLA Architecture Deep Dive (Days 99–105)

### Day 99: VLA Design Choices
**Theory:** Action head: autoregressive tokens vs continuous diffusion vs flow matching. Observation encoding: single image vs multi-view vs point cloud. Action spaces: joint vs end-effector, absolute vs delta. Proprioception integration.

### Day 100: Training VLA Models
**Theory:** Data collection: teleoperation, scripted demos, simulation. Open X-Embodiment dataset: 1M+ episodes from 22 robot types. Training recipes: stages, curricula, data mixing. Compute requirements.

### Day 101: Sim-to-Real Transfer
**Theory:** Domain gap: simulation ≠ real world. Domain randomization. System identification. Bridging: train in sim, fine-tune on small real data. When sim-to-real works and when it fails.

### Day 102: World Models for Robotics
**Theory:** Learning a model of the world from video. Video prediction models (Sora-style) for robotics. UniSim: learning universal simulators. Planning in imagination.

### Day 103: Multi-Task & Multi-Robot Generalization
**Theory:** One model for many tasks (pick, place, open, close, etc.). Cross-embodiment: same policy on different robots. Language grounding: "pick up the blue cup" → actions. Few-shot adaptation to new robots.

### Day 104–105: Week 15 Mini-Project
**Project:** Fine-tune a VLA on a custom task
- Choose a simulated environment
- Collect demonstrations (teleoperation or scripted)
- Fine-tune Octo or OpenVLA on your task
- Evaluate zero-shot vs fine-tuned performance
- **Deliverable:** Fine-tuned model, training curves, evaluation

---

## Week 16: Deployment, Frontiers & Capstone (Days 106–112)

### Day 106: Deploying VLAs on Real Robots
**Theory:** Inference latency requirements (10–50Hz control). Model quantization for edge deployment. Safety: collision avoidance, workspace limits, recovery behaviors. Human-in-the-loop: when to ask for help.

### Day 107: VLAs Meet Classical Robotics
**Theory:** VLA as high-level planner + classical low-level controller. Task and motion planning (TAMP) + LLM. Hybrid architectures: LLM brain + traditional control body. Integrating with ROS2.

### Day 108: The OKS Connection — VLAs for Warehouse Robots
**Theory:** How VLAs could apply to OKS AMRs. Navigation: VLA for dynamic obstacle avoidance. Manipulation: VLA for bin picking and placement. Fleet coordination: LLM-based task planning.

### Day 109: Current Frontiers (2026)
**Theory:** Bimanual manipulation VLAs. Humanoid robot policies. Language-conditioned locomotion. Scaling laws for robot data. Synthetic data generation for training. Real-time adaptation and continual learning.

### Day 110–112: CAPSTONE PROJECT
**Project:** End-to-End VLA Pipeline for a Robot Task
- Set up a simulation environment (Isaac Sim, MuJoCo, or PyBullet)
- Define a multi-step task (e.g., "pick up the red object and place it in the blue bin")
- Approach 1: Use a VLM for scene understanding + classical planner for execution
- Approach 2: Fine-tune a VLA model for direct vision-to-action
- Compare both approaches on success rate, generalization, and failure modes
- Write a technical report
- **Deliverable:** Code, trained models, evaluation, 5-page technical report

---

## Final Checkpoint — You're Done When You Can:

- [ ] Explain the complete attention mechanism from first principles, on a whiteboard
- [ ] Describe the transformer architecture (encoder, decoder, both) in detail
- [ ] Explain how GPT, BERT, and T5 differ architecturally and in their training objectives
- [ ] Walk through the LLM training pipeline: pretraining → SFT → RLHF/DPO
- [ ] Explain how ViT processes an image, and how DINO learns without labels
- [ ] Describe CLIP's contrastive training and zero-shot classification mechanism
- [ ] Explain how LLaVA connects vision to language
- [ ] Describe how diffusion models generate samples (forward + reverse process)
- [ ] Explain Diffusion Policy for robot action generation
- [ ] Describe RT-2's key insight: actions as tokens in a VLM
- [ ] Compare Octo, OpenVLA, and π0 architecturally
- [ ] Design a VLA system for a new robot task, choosing appropriate components

---

## Appendix A: Key Papers (Reading List)

| # | Paper | Year | Phase | Priority |
|---|-------|------|-------|----------|
| 1 | Attention Is All You Need (Vaswani et al.) | 2017 | II | **MUST READ** |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.) | 2018 | II | **MUST READ** |
| 3 | Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al.) | 2019 | II | HIGH |
| 4 | An Image is Worth 16x16 Words (ViT, Dosovitskiy et al.) | 2020 | IV | **MUST READ** |
| 5 | Language Models are Few-Shot Learners (GPT-3, Brown et al.) | 2020 | III | HIGH |
| 6 | Denoising Diffusion Probabilistic Models (Ho et al.) | 2020 | VI | **MUST READ** |
| 7 | Learning Transferable Visual Models (CLIP, Radford et al.) | 2021 | V | **MUST READ** |
| 8 | Masked Autoencoders Are Scalable Vision Learners (He et al.) | 2021 | IV | HIGH |
| 9 | Emerging Properties in Self-Supervised ViTs (DINO, Caron et al.) | 2021 | IV | HIGH |
| 10 | Training language models to follow instructions (InstructGPT, Ouyang et al.) | 2022 | III | HIGH |
| 11 | Training Compute-Optimal LLMs (Chinchilla, Hoffmann et al.) | 2022 | III | HIGH |
| 12 | LLaMA: Open and Efficient Foundation Models (Touvron et al.) | 2023 | III | HIGH |
| 13 | Visual Instruction Tuning (LLaVA, Liu et al.) | 2023 | V | **MUST READ** |
| 14 | RT-1: Robotics Transformer (Brohan et al.) | 2022 | VII | HIGH |
| 15 | RT-2: Vision-Language-Action Models (Brohan et al.) | 2023 | VII | **MUST READ** |
| 16 | Diffusion Policy (Chi et al.) | 2023 | VI | **MUST READ** |
| 17 | Octo: An Open-Source Generalist Robot Policy (Team et al.) | 2024 | VII | HIGH |
| 18 | OpenVLA (Kim et al.) | 2024 | VII | **MUST READ** |
| 19 | π0: A Vision-Language-Action Flow Model (Black et al.) | 2024 | VII | HIGH |
| 20 | Direct Preference Optimization (Rafailov et al.) | 2023 | III | HIGH |
| 21 | LoRA: Low-Rank Adaptation (Hu et al.) | 2021 | III | HIGH |
| 22 | FlashAttention (Dao et al.) | 2022 | II | MEDIUM |
| 23 | Scaling Data-Constrained Language Models (Muennighoff et al.) | 2023 | III | MEDIUM |
| 24 | DINOv2 (Oquab et al.) | 2023 | IV | MEDIUM |

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
| III | `prompt-engineer` | Prompt engineering techniques |
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
2. Ask the `socratic-mentor` agent probing questions
3. Implement the code
4. Ask the `python-pro` agent to review your implementation
5. Run experiments, analyze results
6. Ask the `learning-guide` agent to quiz you
```

---

## Appendix C: File Structure

```
learn/llm-to-vla/
├── CURRICULUM.md                          ← YOU ARE HERE
├── 00-learning-plan.md                    ← Dependency graph, study order
├── study-notes/
│   ├── 01-deep-learning-refresh.md        ← Phase I theory
│   ├── 02-attention-mechanism.md          ← Attention from scratch
│   ├── 03-transformer-architecture.md     ← Full transformer
│   ├── 04-transformer-variants.md         ← Efficiency, variants
│   ├── 05-bert-encoders.md                ← BERT, tokenization
│   ├── 06-gpt-decoders.md                 ← GPT, autoregressive LM
│   ├── 07-llm-training.md                 ← SFT, RLHF, DPO
│   ├── 08-llm-engineering.md              ← RAG, tools, prompting
│   ├── 09-vision-transformers.md          ← ViT, DINO, MAE
│   ├── 10-vision-language-models.md       ← CLIP, LLaVA, BLIP-2
│   ├── 11-diffusion-models.md             ← DDPM, diffusion policy
│   ├── 12-robot-learning.md               ← Imitation learning, action spaces
│   ├── 13-vla-models.md                   ← RT-2, Octo, OpenVLA, π0
│   └── 14-deployment-frontiers.md         ← Real robots, hybrid systems
├── exercises/
│   ├── 01-rnn-lstm.md                     ← Phase I exercises
│   ├── 02-attention-from-scratch.md       ← Build attention step by step
│   ├── 03-build-transformer.md            ← Full transformer implementation
│   ├── 04-train-nanogpt.md                ← Train your own LM
│   ├── 05-finetune-llm.md                 ← LoRA fine-tuning
│   ├── 06-build-rag.md                    ← RAG system
│   ├── 07-implement-vit.md                ← ViT from scratch
│   ├── 08-clip-experiments.md             ← CLIP zero-shot, search
│   ├── 09-diffusion-policy.md             ← Diffusion for actions
│   └── 10-vla-evaluation.md               ← Run & compare VLA models
└── projects/
    ├── 01-annotated-transformer/          ← Phase I-II capstone
    ├── 02-mini-lm/                        ← Phase II-III: train your own LM
    ├── 03-robotics-assistant/             ← Phase III: RAG + tools
    ├── 04-visual-search/                  ← Phase IV: image retrieval
    ├── 05-visual-qa/                      ← Phase V: robotics visual QA
    ├── 06-diffusion-policy/               ← Phase VI: robot action generation
    └── 07-vla-capstone/                   ← Phase VII: end-to-end VLA pipeline
```

---

## Appendix D: Weekly Time Commitment

| Component | Daily | Weekly |
|-----------|-------|--------|
| Theory (reading, videos, papers) | 45 min | 5.25 hrs |
| Implementation (coding) | 60 min | 7 hrs |
| Exercises & review | 45 min | 5.25 hrs |
| **Total** | **2.5 hrs** | **17.5 hrs** |

**Weekend option:** If weekdays are tight, do theory on weekdays (45 min × 5 = 3.75 hrs) and implementation + exercises on weekends (2 × 6.5 hrs = 13 hrs).

---

*Last updated: 2026-04-28*
*Track: LLM-to-VLA*
*Status: CURRICULUM PLANNED — study notes to be created*
