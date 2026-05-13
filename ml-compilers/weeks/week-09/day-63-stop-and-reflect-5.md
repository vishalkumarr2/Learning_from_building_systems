# Day 63: Stop & Reflect #5

> **Phase IV · Week 9 · Day 63 of 70 · 2.5 hours**

*"You now have ten levers to pull for inference optimization. The art is knowing which to pull, in what order, and when to stop."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 62: Serving Frameworks Comparison](day-62-serving-frameworks-comparison.md) | [Day 64: Distributed Training Basics](../week-10/day-64-distributed-training-basics.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Phase IV covered 14 days of inference optimization — from model formats and ONNX through knowledge distillation, TensorRT, CPU/edge inference, LLM-specific challenges, KV caching, PagedAttention, speculative decoding, quantization, and serving frameworks. That's a lot of techniques. This reflection synthesizes them into a **decision framework** so you can walk into any deployment scenario and systematically choose the right approach. You'll also test your understanding with a comprehensive quiz and verify readiness for Phase V: Training at Scale.

---

## 1. Phase IV Concept Map

```
Phase IV: Inference & Deployment — Complete Map
═══════════════════════════════════════════════════════════════

  Week 8: Model Optimization Fundamentals
  ────────────────────────────────────────
  Day 50: Model Formats & ONNX     ─── Interchange, protobuf, opset
  Day 51: Weight Compression       ─── Pruning, quantization basics
  Day 52: Knowledge Distillation   ─── Teacher-student, feature matching
  Day 53: TensorRT Optimization    ─── Graph opt, kernel autotuning
  Day 54: Inference on CPU         ─── ONNX Runtime, OpenVINO, vectorization
  Day 55: Inference on Edge        ─── TFLite, Core ML, NPU delegation
  Day 56: Mini-Project             ─── End-to-end optimization pipeline

  Week 9: LLM Serving Systems
  ────────────────────────────
  Day 57: LLM Inference Challenges ─── Prefill/decode, memory-bound
  Day 58: KV Cache Optimization    ─── MQA, GQA, sliding window
  Day 59: vLLM & PagedAttention    ─── Virtual memory for KV, batching
  Day 60: Speculative Decoding     ─── Draft-verify, rejection sampling
  Day 61: LLM Quantization         ─── GPTQ, AWQ, GGUF, outlier handling
  Day 62: Serving Frameworks       ─── vLLM, TRT-LLM, SGLang, llama.cpp

  Key connections:
  ┌─────────────────────────────────────────────────────────┐
  │  Quantization (Day 51) ──extends──► LLM Quant (Day 61) │
  │  TensorRT (Day 53) ────underlies──► TRT-LLM (Day 62)   │
  │  KV Cache (Day 58) ────enables────► PagedAttn (Day 59)  │
  │  Distillation (Day 52) ─provides─► Draft models (Day 60)│
  │  CPU/Edge (Day 54-55) ─realized──► llama.cpp (Day 62)   │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. The Optimization Decision Tree

```
Inference Optimization Decision Tree
═══════════════════════════════════════════════════════════════

  INPUT: Model type, target hardware, latency/throughput requirements

  Q1: What type of model?
  │
  ├─ CNN / Vision model ──────────────────────────────────┐
  │   Q2: Target hardware?                                │
  │   ├─ NVIDIA GPU → TensorRT (Day 53)                   │
  │   ├─ Intel CPU  → OpenVINO (Day 54)                   │
  │   ├─ Mobile     → TFLite / Core ML (Day 55)           │
  │   └─ Any CPU    → ONNX Runtime (Day 54)               │
  │                                                        │
  │   Then apply (in order):                               │
  │   1. Graph optimization (op fusion, constant folding)  │
  │   2. FP16/INT8 quantization (Day 51)                   │
  │   3. Pruning if accuracy budget allows                 │
  │   4. Distillation for >2× compression (Day 52)        │
  │                                                        │
  └─ LLM / Transformer ──────────────────────────────────┐
      Q2: Model size vs available GPU memory?              │
      │                                                    │
      ├─ Fits on 1 GPU (e.g., 7B on 80GB)                │
      │   1. Quantize: AWQ/GPTQ 4-bit (Day 61)           │
      │   2. Enable PagedAttention (Day 59)                │
      │   3. Continuous batching (Day 59)                   │
      │   4. Speculative decoding if latency-critical      │
      │      (Day 60)                                      │
      │   5. Prefix caching if shared prompts (Day 62)     │
      │                                                    │
      ├─ Needs 2-8 GPUs (e.g., 70B)                       │
      │   1. Tensor parallelism (Day 62)                   │
      │   2. FP8 or INT4 quantization (Day 61)            │
      │   3. All of the above optimizations                │
      │   4. Consider TRT-LLM for max throughput           │
      │                                                    │
      ├─ No GPU / edge / laptop                            │
      │   1. GGUF Q4_K_M quantization (Day 61)            │
      │   2. llama.cpp (Day 62)                            │
      │   3. Smaller model + distillation (Day 52)         │
      │                                                    │
      └─ Cost-sensitive (minimize $/token)                 │
          1. Smallest model that meets quality bar          │
          2. Aggressive quantization (3-4 bit)              │
          3. Batching optimization (throughput > latency)   │
          4. Spot instances + autoscaling                   │
```

---

## 3. Key Metrics Reference Card

```
Inference Optimization Metrics — Quick Reference
═══════════════════════════════════════════════════════════════

  LATENCY METRICS
  ───────────────
  TTFT (Time to First Token)   How fast does output start?
    Good: < 200ms              Affected by: prefill time,
    Acceptable: < 500ms        queue depth, model size
    Poor: > 1s

  TBT (Time Between Tokens)   How smooth is streaming?
    Good: < 30ms               Affected by: decode speed,
    Acceptable: < 50ms         batch interference, KV cache
    Poor: > 100ms

  E2E Latency                  Total request time
    = TTFT + (n_tokens × TBT)

  THROUGHPUT METRICS
  ──────────────────
  Tokens/second               Raw generation speed
    Single user: 30-80 tok/s   Batch: 1K-10K tok/s

  Requests/second             Concurrent request capacity
    Depends on: batch size, sequence length, GPU count

  EFFICIENCY METRICS
  ──────────────────
  GPU Utilization              Are you using what you're paying for?
    Target: > 70%              Decode phase often < 30%!

  KV Cache Utilization         Memory efficiency
    PagedAttention: ~95%       Without: 60-70% (fragmentation)

  $/1M tokens                  Cost efficiency
    FP16 70B on A100: ~$1.50   INT4 70B on A100: ~$0.40
    FP16 7B on A100:  ~$0.15   Q4_K_M 7B on CPU: ~$0.05

  QUALITY METRICS
  ───────────────
  Perplexity (PPL)             Lower = better language modeling
    FP16 → INT4: expect +0.1 to +0.3 PPL

  Task accuracy                MMLU, HumanEval, etc.
    FP16 → INT4: expect -0.5% to -2%

  Exact match to FP16          Speculative decoding: 100%
                               Quantization: depends on method
```

---

## 4. Comprehensive Quiz

Test your understanding of Phase IV. Try to answer before checking the solutions.

### Questions

**Q1**: A 13B parameter model in FP16 requires how much memory for weights alone?

**Q2**: During LLM decode, GPU utilization is typically low. Why? What is the hardware bottleneck?

**Q3**: Grouped Query Attention (GQA) with 32 query heads and 8 KV heads reduces KV cache by what factor compared to standard MHA?

**Q4**: In PagedAttention, what is a "block" and why does it eliminate memory fragmentation?

**Q5**: Speculative decoding uses rejection sampling. If a draft token has probability $q(x) = 0.6$ in the draft model and $p(x) = 0.3$ in the target model, what is the acceptance probability?

**Q6**: Why can't you apply standard per-tensor INT8 quantization to LLMs without significant accuracy loss?

**Q7**: GPTQ compensates for rounding error by adjusting un-quantized weights using what mathematical object?

**Q8**: AWQ scales up "salient" weight channels before quantizing. How does it determine which channels are salient?

**Q9**: In GGUF K-quants, Q4_K_M uses more bits for certain layer types. Which layers get higher precision and why?

**Q10**: You have a chatbot with a 2000-token system prompt shared across all users, running on 4×A100s. Which serving framework would maximize throughput, and which specific feature makes it best?

---

### Solutions

<details>
<summary>Click to reveal answers</summary>

**A1**: $13 \times 10^9 \times 2\text{ bytes} = 26\text{ GB}$. Each FP16 parameter is 2 bytes.

**A2**: Decode is **memory-bandwidth-bound**, not compute-bound. Each decode step reads the entire model weights (~26 GB for 13B) but only computes a single token's worth of matrix-vector products. The arithmetic intensity is ~1 FLOP/byte — far below the GPU's compute/bandwidth ratio (~150 FLOP/byte on A100). The GPU cores sit idle waiting for memory.

**A3**: KV cache is reduced by $32/8 = 4\times$. GQA shares each KV head across $32/8 = 4$ query heads, so you store 8 KV pairs instead of 32.

**A4**: A block is a fixed-size chunk of KV cache slots (e.g., 16 tokens). Sequences are allocated blocks from a global pool (like virtual memory pages) rather than reserving contiguous memory for max_seq_len. Blocks can be non-contiguous in physical memory. This eliminates both internal fragmentation (no wasted padding within a sequence) and external fragmentation (freed blocks return to the pool).

**A5**: $\min(1, p(x)/q(x)) = \min(1, 0.3/0.6) = \min(1, 0.5) = 0.5$. The token is accepted with 50% probability.

**A6**: LLMs exhibit **activation outliers** — approximately 0.1% of channels have magnitudes 100× larger than average. Per-tensor quantization sets the scale factor to cover the outlier range, which compresses 99.9% of normal-magnitude activations to just 2-3 quantization levels, losing most information.

**A7**: The **inverse Hessian** ($H^{-1}$). Specifically, when weight $w_q$ is rounded, the error is distributed to remaining weights proportional to $H^{-1}_{:,q}$, which captures the second-order sensitivity of the loss to each weight.

**A8**: By measuring **activation magnitudes** from calibration data: $s_x = \text{mean}(|X|)$ per channel. Channels with large average activation magnitude are "salient" because quantization error in those weights gets amplified by the large activation, contributing more to output error.

**A9**: **Attention value projections and embeddings** get higher precision (5-6 bits), while **FFN gate/up projections** use the minimum (4 bits). The reasoning: embeddings and output layers are on the model's "edges" where errors aren't averaged out; value projections carry content information through attention; FFN matrices are the largest so compression there saves the most memory.

**A10**: **SGLang** with **RadixAttention**. The 2000-token system prompt creates a massive prefix cache opportunity. RadixAttention stores prefix KV states in a radix tree, so the system prompt is computed once and reused for all requests. This can provide 2-5× throughput improvement compared to frameworks without prefix caching.

</details>

---

## 5. Optimization Techniques Summary Table

| Technique | Type | Typical Speedup | Quality Impact | Complexity |
|---|---|---|---|---|
| ONNX export + graph opt | Format | 1.2-1.5× | None | Low |
| FP16 → INT8 quantization | Compression | 1.5-2× | Minimal | Low |
| FP16 → INT4 quantization | Compression | 2.5-3.5× | Low (0.1-0.3 PPL) | Medium |
| Structured pruning | Compression | 1.3-2× | Low-Medium | Medium |
| Knowledge distillation | Architecture | 2-10× | Low-Medium | High |
| TensorRT compilation | Runtime | 1.5-3× | None | Medium |
| Continuous batching | Serving | 2-5× throughput | None | Low |
| PagedAttention | Memory | 2-4× batch size | None | Low |
| Prefix caching | Serving | 1.5-5× (workload) | None | Low |
| Speculative decoding | Algorithm | 2-3× latency | None (exact) | Medium |
| Tensor parallelism | Scaling | Near-linear | None | Medium |
| KV cache compression (GQA) | Architecture | 4-8× cache | Minimal | N/A (model) |

---

## 6. "Ready for Training at Scale" Checklist

Before moving to Phase V (Distributed Training), verify you can answer these confidently:

```
Phase IV Exit Checklist
═══════════════════════════════════════════════════════════════

  Model Formats & Optimization
  ☐ I can export a PyTorch model to ONNX and optimize the graph
  ☐ I understand operator fusion and constant folding
  ☐ I know when to use TensorRT vs ONNX Runtime vs OpenVINO

  Compression Techniques
  ☐ I can explain magnitude pruning, structured pruning, and lottery tickets
  ☐ I understand PTQ vs QAT and when each is appropriate
  ☐ I can set up a knowledge distillation training loop
  ☐ I know why LLM quantization requires special handling (outliers)

  LLM-Specific Optimization
  ☐ I can explain the prefill vs decode phases and their bottlenecks
  ☐ I understand MHA → MQA → GQA and KV cache size tradeoffs
  ☐ I can describe how PagedAttention works and why it matters
  ☐ I understand continuous batching and why it beats static batching
  ☐ I can implement basic speculative decoding with draft-verify
  ☐ I know the difference between GPTQ, AWQ, and GGUF approaches

  Serving & Deployment
  ☐ I can choose the right serving framework for a given scenario
  ☐ I understand tensor parallelism vs pipeline parallelism
  ☐ I know the key metrics: TTFT, TBT, tokens/s, $/1M tokens
  ☐ I can set up a load test and interpret the results

  If you checked all boxes → You're ready for Phase V!
  If you missed 3+       → Review the relevant days
  If you missed 6+       → Re-do the Week 8-9 exercises
```

---

## 7. Phase V Preview — Training at Scale

```
What's Ahead: Phase V (Weeks 10-11)
═══════════════════════════════════════════════════════════════

  Week 10: Distributed Training
  ─────────────────────────────
  Day 64: Distributed Training Basics    ─── AllReduce, ring
  Day 65: Data Parallelism (DDP, FSDP)   ─── Gradient sync
  Day 66: Model Parallelism              ─── Megatron-LM, pipeline
  Day 67: Mixed Precision Training       ─── FP16/BF16, loss scaling
  Day 68: Training Infrastructure        ─── Checkpointing, fault tolerance
  Day 69: Scaling Laws & Efficiency      ─── Chinchilla, compute-optimal
  Day 70: Capstone Project               ─── End-to-end ML system

  Key shift: Inference → Training
  ────────────────────────────────
  Inference optimizes FORWARD pass only.
  Training optimizes forward + BACKWARD + COMMUNICATION.

  New challenges you'll face:
  • Gradient synchronization across 100s of GPUs
  • Memory: activations + gradients + optimizer state = 16× model size
  • Communication: all-reduce can dominate training time
  • Fault tolerance: 1000-GPU jobs fail every few hours
  • Mixed precision: FP16 training requires loss scaling tricks

  ┌─────────────────────────────────────────────────────────┐
  │  Inference skills transfer directly:                     │
  │  • Quantization → Mixed precision training               │
  │  • Tensor parallelism → Model parallelism                │
  │  • Memory management → Activation checkpointing          │
  │  • Batching strategies → Gradient accumulation            │
  └─────────────────────────────────────────────────────────┘
```

---

## Hands-On Exercise: Build Your Decision Matrix

### Exercise: Deployment Scenario Analysis

For each scenario below, specify: (1) framework, (2) quantization method, (3) parallelism strategy, (4) key optimizations, and (5) expected cost/performance.

**Scenario A**: Customer-facing chatbot, LLaMA-3 8B, <200ms TTFT, 500 req/s, 4×A100.

**Scenario B**: Internal document summarizer, LLaMA-3 70B, batch processing 10K docs/day, cost-sensitive, 2×A100.

**Scenario C**: On-device code completion, CodeLlama 7B, MacBook M2 Pro (16GB RAM), <100ms TTFT, single user.

**Scenario D**: Multi-tenant SaaS, 50 different LoRA adapters per customer, LLaMA-3 8B base, 8×A100 cluster.

<details>
<summary>Click to reveal suggested solutions</summary>

**Scenario A**: vLLM, AWQ 4-bit (fits 1 GPU easily → replicate 4×), data parallelism, enable prefix caching for system prompt, speculative decoding for TTFT. Expected: ~600 req/s at p99 <200ms TTFT, ~$0.10/1M tokens.

**Scenario B**: vLLM or TGI, GPTQ 4-bit (~37GB, fits on 2×A100 with TP=2), tensor parallelism, maximize batch size (throughput over latency), no speculative decoding needed. Expected: ~15K tokens/s, ~$0.03/1M tokens (batch pricing).

**Scenario C**: llama.cpp, GGUF Q4_K_M (~4.3GB, fits in 16GB with room for KV cache), Metal acceleration, 8 threads, no batching needed. Expected: ~30-40 tokens/s, TTFT <50ms for short prompts.

**Scenario D**: vLLM with multi-LoRA, FP16 base + LoRA adapters hot-swapped, TP=2 per replica × 4 replicas, enable prefix caching. Expected: ~200 req/s aggregate, LoRA swap overhead <5ms.

</details>

---

## Key Takeaways

1. **Phase IV covered the full inference optimization stack** — from model formats through quantization to production serving frameworks
2. **The decision tree starts with model type and hardware**, then applies optimizations in priority order (quantization → batching → caching → speculation)
3. **LLM inference is fundamentally different from CNN inference** — memory-bound decode, KV cache management, and autoregressive constraints require specialized solutions
4. **Three numbers define deployment quality**: TTFT (user experience), tokens/s (throughput), $/1M tokens (cost)
5. **Start simple, measure, then optimize** — vLLM + AWQ 4-bit covers 80% of use cases; only add complexity when benchmarks prove it's needed
6. **Phase V builds on these foundations** — tensor parallelism becomes model parallelism, quantization becomes mixed precision, and batching becomes gradient accumulation

---

## Further Reading

- **All papers cited in Days 50-62** — revisit any that felt unclear
- **Databricks "LLM Inference Performance Engineering" blog series** (2024)
- **Horace He, "Making Deep Learning Go Brrrr"** — mental models for performance
- **Patterson et al., "Carbon Emissions and Large Neural Network Training"** (2021) — why efficiency matters beyond cost

---

## What's Next

Phase V: **Training at Scale**. Day 64 introduces distributed training fundamentals — AllReduce, ring topology, gradient synchronization — the communication primitives that make multi-GPU training possible. The shift from inference to training means dealing with backwards passes, optimizer states, and the 16× memory multiplier that makes training far more memory-hungry than serving.
