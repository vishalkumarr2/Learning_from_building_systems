# Day 62: Serving Frameworks Comparison

> **Phase IV · Week 9 · Day 62 of 70 · 2.5 hours**

*"The best inference framework is the one that matches your workload. A chatbot, a batch summarizer, and a code completion engine need entirely different serving stacks."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 61: LLM Quantization](day-61-llm-quantization.md) | [Day 63: Stop & Reflect #5](day-63-stop-and-reflect-5.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

You understand KV cache management, continuous batching, speculative decoding, and quantization. Now you need to choose a framework that combines these techniques into a production system. The landscape in 2024-2026 has consolidated around six major players — vLLM, TensorRT-LLM, TGI, SGLang, llama.cpp, and Triton — each with distinct strengths. Picking wrong can mean 3× worse throughput, months of migration pain, or hardware you can't actually use. This lesson maps the decision space so you choose correctly the first time.

---

## 1. Architecture Overview

```
LLM Serving Framework Architectures
═══════════════════════════════════════════════════════════════

  vLLM                          TensorRT-LLM
  ────────────────────          ────────────────────────────
  ┌────────────────┐            ┌──────────────────────────┐
  │  OpenAI-compat │            │  Triton Inference Server │
  │  HTTP API      │            │  (or standalone)         │
  ├────────────────┤            ├──────────────────────────┤
  │  Scheduler     │            │  GptManager              │
  │  (continuous   │            │  (inflight batching)     │
  │   batching)    │            ├──────────────────────────┤
  ├────────────────┤            │  TRT-LLM Engine          │
  │  PagedAttention│            │  (compiled graph +       │
  │  KV Cache Mgr  │            │   custom CUDA kernels)   │
  ├────────────────┤            ├──────────────────────────┤
  │  Model Runner  │            │  FP8/INT4 kernels        │
  │  (PyTorch)     │            │  FP8 KV cache            │
  └────────────────┘            └──────────────────────────┘
  Python-native                 C++/CUDA, compiled graphs
  Easy to hack                  Maximum performance

  TGI (HuggingFace)             SGLang
  ────────────────────          ────────────────────────────
  ┌────────────────┐            ┌──────────────────────────┐
  │  HTTP + gRPC   │            │  SGLang Frontend         │
  ├────────────────┤            │  (structured generation) │
  │  Rust Router   │            ├──────────────────────────┤
  │  (token        │            │  RadixAttention          │
  │   streaming)   │            │  (prefix caching tree)   │
  ├────────────────┤            ├──────────────────────────┤
  │  Python Model  │            │  Scheduler               │
  │  (flash-attn,  │            │  (continuous batching)   │
  │   GPTQ, AWQ)   │            ├──────────────────────────┤
  └────────────────┘            │  Model Runner (PyTorch)  │
  HF ecosystem native           └──────────────────────────┘
  Production-ready Docker        Best for structured output

  llama.cpp                     Triton Inference Server
  ────────────────────          ────────────────────────────
  ┌────────────────┐            ┌──────────────────────────┐
  │  HTTP server   │            │  HTTP/gRPC Frontend      │
  │  (optional)    │            ├──────────────────────────┤
  ├────────────────┤            │  Model Repository        │
  │  GGUF Loader   │            │  (multi-model, A/B test) │
  ├────────────────┤            ├──────────────────────────┤
  │  CPU / Metal / │            │  Backend Orchestrator    │
  │  CUDA / Vulkan │            │  (TRT, PyTorch, ONNX,    │
  │  inference     │            │   vLLM, TRT-LLM)         │
  └────────────────┘            ├──────────────────────────┤
  C/C++, no Python dep          │  Dynamic Batching        │
  Runs everywhere               └──────────────────────────┘
                                Meta-framework, multi-model
```

---

## 2. Feature Comparison Matrix

```
Feature Matrix (as of 2025-2026)
═══════════════════════════════════════════════════════════════

  Feature              vLLM   TRT-LLM  TGI    SGLang  llama  Triton
  ─────────────────────────────────────────────────────────────────
  Continuous batching   ✅      ✅      ✅      ✅      ⚠️      ✅
  PagedAttention        ✅      ✅      ✅      ✅      ❌      via backend
  Prefix caching        ✅      ✅      ✅      ✅★     ✅      via backend
  Speculative decode    ✅      ✅      ❌      ✅      ✅      via backend
  Tensor parallelism    ✅      ✅      ✅      ✅      ❌      ✅
  Pipeline parallelism  ✅      ✅      ❌      ❌      ❌      ✅
  FP8 inference         ✅      ✅      ❌      ✅      ❌      via backend
  GPTQ / AWQ            ✅      ✅      ✅      ✅      ❌      via backend
  GGUF quantization     ❌      ❌      ❌      ❌      ✅★     ❌
  LoRA serving          ✅      ✅      ✅      ✅      ✅      ✅
  Multi-LoRA hot-swap   ✅      ✅      ❌      ✅      ❌      ✅
  Vision-Language       ✅      ✅      ✅      ✅      ✅      ✅
  Structured output     ✅      ❌      ✅      ✅★     ✅      ❌
  OpenAI-compatible API ✅      ✅      ✅      ✅      ✅      ❌
  CPU inference         ❌      ❌      ❌      ❌      ✅★     ✅
  Metal (Mac) support   ❌      ❌      ❌      ❌      ✅★     ❌
  ─────────────────────────────────────────────────────────────────
  ★ = category leader     ⚠️ = partial/limited
```

---

## 3. Performance Benchmarks

Real-world benchmarks vary by model, hardware, and workload pattern. These are representative numbers on A100-80GB:

```
Throughput Benchmarks: LLaMA-2 70B, A100-80GB × 4
═══════════════════════════════════════════════════════════════

  Workload: Chatbot (avg 200 input, 150 output tokens)
  Metric: Requests/second at p99 < 2s TTFT

  ┌────────────┬───────────┬──────────┬──────────┬──────────┐
  │ Framework  │ Req/s     │ TTFT p50 │ TTFT p99 │ Tok/s    │
  ├────────────┼───────────┼──────────┼──────────┼──────────┤
  │ TRT-LLM    │ 42        │ 120 ms   │ 850 ms   │ 6,300    │
  │ SGLang     │ 38        │ 140 ms   │ 920 ms   │ 5,700    │
  │ vLLM       │ 35        │ 160 ms   │ 1,100 ms │ 5,250    │
  │ TGI        │ 28        │ 180 ms   │ 1,400 ms │ 4,200    │
  └────────────┴───────────┴──────────┴──────────┴──────────┘

  Workload: Long-context (4K input, 500 output tokens)

  ┌────────────┬───────────┬──────────┬──────────┐
  │ Framework  │ Req/s     │ TTFT p50 │ Tok/s    │
  ├────────────┼───────────┼──────────┼──────────┤
  │ TRT-LLM    │ 18        │ 380 ms   │ 9,000    │
  │ SGLang     │ 16        │ 410 ms   │ 8,000    │
  │ vLLM       │ 14        │ 480 ms   │ 7,000    │
  │ TGI        │ 11        │ 620 ms   │ 5,500    │
  └────────────┴───────────┴──────────┴──────────┘

  Workload: Shared-prefix (system prompt reuse, 2K shared prefix)

  ┌────────────┬───────────┬────────────────────────────┐
  │ Framework  │ Req/s     │ Notes                      │
  ├────────────┼───────────┼────────────────────────────┤
  │ SGLang     │ 52        │ RadixAttention shines here │
  │ vLLM       │ 38        │ APC (automatic prefix)     │
  │ TRT-LLM    │ 35        │ KV cache reuse             │
  │ TGI        │ 22        │ No prefix caching          │
  └────────────┴───────────┴────────────────────────────┘

  Note: Benchmarks are approximate and version-dependent.
  Always benchmark YOUR workload on YOUR hardware.
```

---

## 4. Framework Deep Dives

### 4.1 vLLM — The Python-Native Default

```python
"""vLLM: simplest path from HuggingFace model to production."""

# Installation
# pip install vllm

# ─── Offline batch inference ───
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,      # 4 GPUs
    max_model_len=4096,
    gpu_memory_utilization=0.90, # reserve 10% for overhead
    quantization="awq",          # or "gptq", "fp8"
    enable_prefix_caching=True,
)

prompts = ["Explain quantum computing", "Write a haiku about AI"]
params = SamplingParams(temperature=0.7, max_tokens=256, top_p=0.9)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(f"{output.prompt[:30]}... → {output.outputs[0].text[:50]}...")

# ─── Online server (OpenAI-compatible) ───
# vllm serve meta-llama/Llama-2-70b-chat-hf \
#   --tensor-parallel-size 4 \
#   --max-model-len 4096 \
#   --quantization awq \
#   --enable-prefix-caching \
#   --port 8000
```

**When to use vLLM**: Default choice for GPU serving. Best ecosystem support, fastest iteration speed, good performance. Use when you want a single `pip install` to production.

### 4.2 TensorRT-LLM — Maximum Performance

```python
"""TensorRT-LLM: compiled engine for maximum throughput."""

# Build engine (one-time, takes 10-30 min)
# python build.py \
#   --model_dir /models/llama-70b \
#   --output_dir /engines/llama-70b-tp4 \
#   --dtype float16 \
#   --tp_size 4 \
#   --pp_size 1 \
#   --use_inflight_batching \
#   --use_paged_context_fmha \
#   --max_batch_size 64 \
#   --max_input_len 2048 \
#   --max_seq_len 4096 \
#   --use_fp8_context_fmha \
#   --quantization fp8

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir(
    engine_dir="/engines/llama-70b-tp4",
    rank=tensorrt_llm.mpi_rank(),
)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Hello, world!")],
    max_new_tokens=128,
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    temperature=0.7,
    top_p=0.9,
)
```

**When to use TensorRT-LLM**: When you need absolute maximum throughput on NVIDIA GPUs and can afford the compilation step. Best for fixed-model production deployments where 20-40% more throughput justifies the complexity.

### 4.3 SGLang — Structured Generation Champion

```python
"""SGLang: best for structured output and prefix-heavy workloads."""

# Server launch
# python -m sglang.launch_server \
#   --model-path meta-llama/Llama-2-70b-chat-hf \
#   --tp 4 \
#   --port 30000

import sglang as sgl

@sgl.function
def structured_extraction(s, document):
    s += sgl.system("Extract structured information from the document.")
    s += sgl.user(document)
    s += sgl.assistant(
        sgl.gen("analysis", max_tokens=200)
    )
    # Constrained generation — forces valid JSON
    s += sgl.user("Now output a JSON with keys: topic, sentiment, entities")
    s += sgl.assistant(
        sgl.gen("json_output", max_tokens=300,
                regex=r'\{[^}]+\}')  # Constrained to JSON
    )

# RadixAttention: prefix tree caching
# If 1000 requests share the same system prompt,
# SGLang computes it ONCE and reuses the KV cache
state = structured_extraction.run(
    document="Apple reported Q4 revenue of $89.5B...",
    backend=sgl.RuntimeEndpoint("http://localhost:30000"),
)
print(state["json_output"])
```

**When to use SGLang**: Multi-turn conversations with shared system prompts, structured/constrained generation (JSON, regex), or any workload with high prefix reuse. RadixAttention can provide 2-5× throughput gains in these scenarios.

### 4.4 llama.cpp — Run Anywhere

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON  # or -DGGML_METAL=ON for Mac
cmake --build build -j

# Run inference
./build/bin/llama-cli \
    -m models/llama-2-7b-Q4_K_M.gguf \
    -p "Explain transformers in simple terms:" \
    -n 256 \
    --temp 0.7 \
    --threads 8 \
    --n-gpu-layers 35    # offload layers to GPU

# Run as OpenAI-compatible server
./build/bin/llama-server \
    -m models/llama-2-7b-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 35 \
    --ctx-size 4096 \
    --parallel 4         # concurrent requests
```

**When to use llama.cpp**: CPU inference, Mac (Metal), edge devices, laptops, or anywhere without NVIDIA GPUs. Also great for local development and testing. The GGUF quantization ecosystem is unmatched for consumer hardware.

---

## 5. Multi-GPU Strategies

```
Multi-GPU Deployment Strategies
═══════════════════════════════════════════════════════════════

  Tensor Parallelism (TP) — Split each layer across GPUs
  ─────────────────────────────────────────────────────

  GPU 0              GPU 1              GPU 2              GPU 3
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Layer 1   │      │ Layer 1   │      │ Layer 1   │      │ Layer 1   │
  │ cols 0-1K │◄────►│ cols 1-2K │◄────►│ cols 2-3K │◄────►│ cols 3-4K │
  │ Layer 2   │      │ Layer 2   │      │ Layer 2   │      │ Layer 2   │
  │ cols 0-1K │◄────►│ cols 1-2K │◄────►│ cols 2-3K │◄────►│ cols 3-4K │
  │    ...    │      │    ...    │      │    ...    │      │    ...    │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘
  All-reduce after each layer. Needs fast interconnect (NVLink).
  Best for: latency-sensitive serving (1 model, multiple GPUs)

  Pipeline Parallelism (PP) — Split layers across GPUs
  ─────────────────────────────────────────────────────

  GPU 0              GPU 1              GPU 2              GPU 3
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Layer 1-8 │─────►│ Layer 9-16│─────►│Layer 17-24│─────►│Layer 25-32│
  └──────────┘      └──────────┘      └──────────┘      └──────────┘
  Sequential through stages. Low interconnect bandwidth OK.
  Best for: throughput (pipeline bubbles filled by batching)

  Data Parallelism — Replicate model, split requests
  ──────────────────────────────────────────────────

  GPU 0: Full Model  │  GPU 1: Full Model  │  GPU 2: Full Model
  Request batch A    │  Request batch B    │  Request batch C
  Independent, no communication needed.
  Best for: small models that fit on 1 GPU, maximum throughput

  ┌──────────────────────────────────────────────────────────┐
  │  Decision tree:                                          │
  │  Model fits 1 GPU?  → Data parallelism (replicate)       │
  │  Model fits 2 GPUs? → TP=2 per replica, replicate more   │
  │  Model needs 4+ GPUs? → TP=4 or TP=2 × PP=2             │
  │  Multi-node?        → PP across nodes, TP within node    │
  └──────────────────────────────────────────────────────────┘
```

---

## 6. Decision Framework

```
Choosing a Serving Framework
═══════════════════════════════════════════════════════════════

  START
    │
    ├─ No NVIDIA GPU? (CPU, Mac, edge)
    │   └─→ llama.cpp (GGUF quantization)
    │
    ├─ Need multi-model serving? (A/B test, ensemble)
    │   └─→ Triton Inference Server (with TRT-LLM or vLLM backend)
    │
    ├─ Need absolute max throughput? (willing to compile engines)
    │   └─→ TensorRT-LLM
    │
    ├─ Heavy prefix reuse? (shared system prompts, multi-turn)
    │   └─→ SGLang (RadixAttention)
    │
    ├─ Need structured/constrained output? (JSON, regex, grammar)
    │   └─→ SGLang or vLLM (both support, SGLang more mature)
    │
    ├─ Want simplest setup? (HuggingFace model → API in 5 min)
    │   └─→ vLLM or TGI
    │
    ├─ Need HuggingFace ecosystem? (latest models day-1)
    │   └─→ TGI (HuggingFace native) or vLLM (fast model support)
    │
    └─ Default choice for GPU serving
        └─→ vLLM (best overall balance)

  ┌────────────────────────────────────────────────────────────┐
  │  Real talk: Most teams should start with vLLM.             │
  │  Move to TRT-LLM only if you've benchmarked and the       │
  │  20-40% throughput gain justifies the operational cost.    │
  │  Move to SGLang if prefix caching is your bottleneck.      │
  └────────────────────────────────────────────────────────────┘
```

### Deployment Considerations

```
Production Deployment Checklist
═══════════════════════════════════════════════════════════════

  1. Health Checks
     ├─ /health endpoint (is the model loaded?)
     ├─ /ready endpoint (is the model warmed up?)
     └─ GPU memory monitoring (OOM prevention)

  2. Scaling
     ├─ Horizontal: multiple replicas behind load balancer
     ├─ Vertical: more GPUs per replica (TP/PP)
     └─ Autoscaling: queue depth or p99 latency triggers

  3. Monitoring
     ├─ TTFT (time to first token) — user-perceived latency
     ├─ TPS (tokens per second) — throughput
     ├─ Queue depth — backpressure signal
     ├─ KV cache utilization — memory pressure
     └─ GPU utilization — cost efficiency

  4. Graceful Degradation
     ├─ Request timeout (kill stuck generations)
     ├─ Max queue size (reject when overloaded)
     ├─ Rate limiting per user/API key
     └─ Fallback to smaller model under load
```

---

## Hands-On Exercises

### Exercise 1: vLLM Benchmark

```python
"""Benchmark vLLM with different configurations."""
import time
import requests
import concurrent.futures

VLLM_URL = "http://localhost:8000/v1/completions"

def single_request(prompt: str, max_tokens: int = 128):
    """Send one completion request, return timing."""
    start = time.perf_counter()
    resp = requests.post(VLLM_URL, json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    })
    elapsed = time.perf_counter() - start
    data = resp.json()
    tokens = data["usage"]["completion_tokens"]
    return {"time": elapsed, "tokens": tokens, "tps": tokens / elapsed}

def load_test(n_concurrent: int = 16, n_total: int = 100):
    """Run concurrent load test."""
    prompts = [f"Write a short paragraph about topic {i}" for i in range(n_total)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        start = time.perf_counter()
        futures = [pool.submit(single_request, p) for p in prompts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        wall_time = time.perf_counter() - start
    
    times = [r["time"] for r in results]
    tps_values = [r["tps"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)
    
    print(f"Concurrency: {n_concurrent}")
    print(f"Total requests: {n_total}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Throughput: {n_total / wall_time:.1f} req/s")
    print(f"Total tokens/s: {total_tokens / wall_time:.0f}")
    print(f"Latency p50: {sorted(times)[len(times)//2]:.2f}s")
    print(f"Latency p99: {sorted(times)[int(len(times)*0.99)]:.2f}s")

# Run: load_test(n_concurrent=1), load_test(n_concurrent=8), load_test(n_concurrent=32)
```

### Exercise Tasks

1. **Framework shootout**: Deploy the same 7B model on vLLM and TGI. Run identical load tests at concurrency 1, 8, and 32. Compare TTFT, throughput, and p99 latency. Which wins at which concurrency level?
2. **Prefix caching measurement**: Create 100 requests that share a 1000-token system prompt. Run on vLLM (with and without prefix caching) and SGLang. Measure the throughput difference.
3. **CPU vs GPU comparison**: Quantize a 7B model to Q4_K_M GGUF. Run on llama.cpp (CPU, 8 threads) and vLLM (A100 GPU, FP16). At what batch size does GPU become worthwhile?

---

## Key Takeaways

1. **vLLM is the default choice** — best balance of performance, ecosystem support, and simplicity for GPU serving
2. **TensorRT-LLM wins on raw throughput** (20-40% over vLLM) but requires compiled engines and NVIDIA-only deployment
3. **SGLang excels at prefix-heavy workloads** — RadixAttention provides 2-5× gains when many requests share system prompts
4. **llama.cpp owns the edge** — only viable option for CPU, Mac Metal, and consumer hardware via GGUF quantization
5. **Triton is a meta-framework** — use it when you need multi-model serving, A/B testing, or ensemble pipelines
6. **Multi-GPU strategy matters**: tensor parallelism for latency, pipeline parallelism for throughput, data parallelism for small models

---

## Further Reading

- **vLLM documentation**: docs.vllm.ai — PagedAttention, continuous batching, quantization
- **TensorRT-LLM GitHub**: github.com/NVIDIA/TensorRT-LLM — build scripts, performance guide
- **SGLang paper**: "Efficiently Programming Large Language Models using SGLang" (Zheng et al., 2024)
- **TGI documentation**: huggingface.co/docs/text-generation-inference
- **llama.cpp GitHub**: github.com/ggerganov/llama.cpp — GGUF, quantization, server mode
- **AnyScale benchmark blog**: "Comparing LLM Serving Frameworks" (2024)

---

## Tomorrow's Preview

We've covered the full inference optimization stack: from attention mechanisms and KV caches to quantization and serving frameworks. **Day 63: Stop & Reflect #5** steps back to synthesize everything from Phase IV (Weeks 8-9). You'll build a decision tree for choosing the right optimization strategy, take a comprehensive quiz, and check your readiness for Phase V: Training at Scale.
