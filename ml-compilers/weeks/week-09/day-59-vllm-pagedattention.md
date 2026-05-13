# Day 59: vLLM & PagedAttention

> **Phase IV · Week 9 · Day 59 of 70 · 2.5 hours**

*"We borrowed the best idea from operating systems — virtual memory — and applied it to attention. The result was a 24× throughput improvement."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 58: KV Cache Optimization](day-58-kv-cache-optimization.md) | [Day 60: Speculative Decoding](day-60-speculative-decoding.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Before vLLM, LLM serving systems wasted 60-80% of GPU memory on internal fragmentation — pre-allocated KV cache blocks that were reserved but unused. A 70B model on 4× A100s might only serve 8 concurrent requests despite having enough raw memory for 40+. PagedAttention solved this by treating KV cache memory like an OS treats RAM: paged, dynamically allocated, and shared when possible. vLLM became the de facto standard for LLM serving, and PagedAttention's ideas have been adopted by TensorRT-LLM, DeepSpeed, and SGLang. Understanding its internals is essential for anyone deploying LLMs at scale.

---

## 1. The Memory Fragmentation Problem

Traditional KV cache allocation pre-reserves a contiguous memory block for each request's maximum possible sequence length:

```
Pre-Allocated KV Cache (Traditional)
═══════════════════════════════════════════════════════════════

  GPU Memory Pool (contiguous allocation):
  ┌──────────────────────────────────────────────────────┐
  │ Request A (max 2048)    │ Request B (max 2048)       │
  │ [████████░░░░░░░░░░░░░] │ [█████████████░░░░░░░░░░] │
  │  actual=512   wasted=1536   actual=832   wasted=1216 │
  ├──────────────────────────────────────────────────────┤
  │ Request C (max 2048)    │ ░░░░░ UNUSABLE ░░░░░░░░░░ │
  │ [██░░░░░░░░░░░░░░░░░░] │ (too fragmented for new   │
  │  actual=128  wasted=1920│  request requiring 2048)   │
  └──────────────────────────────────────────────────────┘

  Total allocated: 6144 slots
  Actually used:   1472 slots (24%)
  Wasted:          4672 slots (76%)    ← catastrophic waste!

  Sources of waste:
  ─────────────────
  1. Internal fragmentation:  reserved slots beyond actual length
  2. External fragmentation:  gaps between allocations
  3. Pre-allocation:          must reserve for max_seq_len upfront
```

---

## 2. PagedAttention — Virtual Memory for KV Cache

PagedAttention maps the KV cache into fixed-size **blocks** (pages), indexed via per-request **block tables**:

```
PagedAttention Architecture
═══════════════════════════════════════════════════════════════

  Physical KV blocks (fixed-size, e.g., 16 tokens each):
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │ P0  │ P1  │ P2  │ P3  │ P4  │ P5  │ P6  │ P7  │
  │ Req │ Req │ Req │ FREE│ Req │ Req │ FREE│ Req │
  │  A  │  B  │  A  │     │  C  │  B  │     │  A  │
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

  Block Tables (logical → physical mapping):
  ┌──────────────────────────────┐
  │  Request A: [P0, P2, P7]    │  3 blocks = 48 tokens allocated
  │  Request B: [P1, P5]        │  2 blocks = 32 tokens allocated
  │  Request C: [P4]            │  1 block  = 16 tokens allocated
  │  Free list: [P3, P6]        │  2 blocks available
  └──────────────────────────────┘

  Key properties:
  ───────────────
  ✓ No contiguous allocation needed (blocks are non-contiguous)
  ✓ No pre-allocation to max length (grow block by block)
  ✓ Internal fragmentation < 1 block per request
  ✓ No external fragmentation (any free block fits any request)
  ✓ Memory utilization: >95% (vs ~24% in traditional)
```

The analogy to OS virtual memory is precise:

| OS Concept | PagedAttention Analog |
|---|---|
| Virtual address space | Sequence positions (0, 1, 2, ...) |
| Physical pages (4KB) | KV blocks (16 tokens × head_dim) |
| Page table | Block table per request |
| Page fault → allocate | New token → allocate next block |
| Free page list | Free block list |
| Copy-on-write (fork) | Shared prefix blocks |

---

## 3. Copy-on-Write & Prefix Caching

PagedAttention enables **copy-on-write** for requests sharing a common prefix (e.g., system prompt):

```
Copy-on-Write for Shared Prefixes
═══════════════════════════════════════════════════════════════

  System prompt: "You are a helpful assistant. Answer concisely."
  (3 blocks)

  Without CoW:
  ──────────────
  Req A: [PA₀][PA₁][PA₂][PA₃][PA₄]    15 blocks total
  Req B: [PB₀][PB₁][PB₂][PB₃]          for 3 requests
  Req C: [PC₀][PC₁][PC₂][PC₃][PC₄][PC₅]
          ^^^^ ^^^^ ^^^^
        identical prefix (duplicated 3×)

  With CoW:
  ──────────
  Shared: [PS₀][PS₁][PS₂]              9 blocks total
                  │                      (saved 6 blocks = 40%)
       ┌──────────┼──────────┐
       ▼          ▼          ▼
  Req A: [PA₃][PA₄]    Req B: [PB₃]    Req C: [PC₃][PC₄][PC₅]

  Block table with ref counts:
  PS₀ (refcount=3), PS₁ (refcount=3), PS₂ (refcount=3)
  PA₃ (refcount=1), PA₄ (refcount=1), ...

  If Req A modifies a shared block → copy it first, then modify
```

**Prefix caching** extends this to cache common prefixes across requests, enabling instant prefill for repeated system prompts.

---

## 4. vLLM Internal Architecture

```
vLLM System Architecture
═══════════════════════════════════════════════════════════════

  ┌─────────────┐    ┌──────────────────────────────────────┐
  │   API       │    │           vLLM Engine                 │
  │   Server    │───▶│                                      │
  │  (FastAPI)  │    │  ┌──────────┐  ┌──────────────────┐  │
  └─────────────┘    │  │Scheduler │  │  Block Manager    │  │
                     │  │          │──│                    │  │
                     │  │ Waiting  │  │  Allocate/Free    │  │
                     │  │ Running  │  │  blocks per req   │  │
                     │  │ Swapped  │  │  CoW ref counting │  │
                     │  └──────────┘  └──────────────────┘  │
                     │       │                               │
                     │       ▼                               │
                     │  ┌──────────────────┐                │
                     │  │   Model Runner    │                │
                     │  │                    │                │
                     │  │  Prepare inputs    │                │
                     │  │  Run model forward │                │
                     │  │  Sample tokens     │                │
                     │  │  PagedAttention    │                │
                     │  │  kernel calls      │                │
                     │  └──────────────────┘                │
                     └──────────────────────────────────────┘

  Scheduler states per request:
  ─────────────────────────────
  WAITING  → queued, no GPU blocks allocated yet
  RUNNING  → has GPU blocks, actively generating
  SWAPPED  → GPU blocks moved to CPU (preemption)
  FINISHED → completed, blocks freed
```

### The Scheduling Loop

Each iteration of the vLLM engine runs one "step":

```python
# Simplified vLLM scheduler loop (pseudocode)
class Scheduler:
    def __init__(self, block_manager, max_num_seqs):
        self.waiting = []    # requests not yet started
        self.running = []    # actively generating
        self.swapped = []    # preempted to CPU
    
    def schedule(self):
        """Decide which requests to run this step."""
        scheduled = []
        
        # 1. Try to swap in previously preempted requests
        while self.swapped and self.block_manager.can_swap_in(self.swapped[0]):
            req = self.swapped.pop(0)
            self.block_manager.swap_in(req)
            scheduled.append(req)
        
        # 2. Continue running existing requests
        for req in self.running:
            if self.block_manager.can_append_token(req):
                scheduled.append(req)
            else:
                # No space! Preempt lowest-priority request
                victim = self._find_preempt_victim(scheduled)
                self.block_manager.swap_out(victim)
                self.swapped.append(victim)
                scheduled.remove(victim)
                scheduled.append(req)
        
        # 3. Admit new requests from waiting queue
        while self.waiting and self.block_manager.can_allocate(self.waiting[0]):
            req = self.waiting.pop(0)
            self.block_manager.allocate(req)
            scheduled.append(req)
        
        return scheduled  # These all run in one batched forward pass
```

---

## 5. Continuous Batching in vLLM

vLLM implements **iteration-level** continuous batching — the batch composition changes every single decode step:

```
Continuous Batching Timeline
═══════════════════════════════════════════════════════════════

  Step 1:  Batch = [A_decode, B_decode, C_prefill]
           ▲                              ▲
           ongoing decodes                new request C enters

  Step 2:  Batch = [A_decode, B_decode, C_decode]
           C now decoding (prefill done)

  Step 3:  Batch = [A_decode, C_decode, D_prefill]
                    ▲                    ▲
                    B finished!          D enters immediately
                    (blocks freed)       (no waiting!)

  Step 4:  Batch = [A_decode, C_decode, D_decode, E_prefill]
                                                   ▲
                                                   E enters too

  Result:
  ─────────
  ✓ GPU never idles — empty slots filled instantly
  ✓ No head-of-line blocking — short requests don't wait for long ones
  ✓ Memory recycled immediately when requests finish
```

Key scheduling decision: **chunked prefill**. Long prefill requests can monopolize the GPU for hundreds of milliseconds, stalling all decode requests. vLLM splits prefill into chunks interleaved with decode steps:

```
Chunked Prefill (vLLM)
═══════════════════════════════════════════════════════════════

  Without chunking:
  |←──── C prefill (2048 tok, 200ms) ────→|  Decode A,B stalled!

  With chunking (chunk=256):
  |C₁|A,B|C₂|A,B|C₃|A,B|C₄|A,B|C₅|A,B|C₆|A,B|C₇|A,B|C₈|A,B|
   ▲                                                         ▲
  256 tok prefill chunk                                   done

  Decode latency for A,B: bounded by chunk processing time
  TTFT for C: slightly higher, but decode requests stay responsive
```

---

## 6. Benchmarking vLLM

```python
# vLLM benchmark script — compare with HuggingFace and TGI
# Requires: pip install vllm

from vllm import LLM, SamplingParams
import time

def benchmark_vllm(model_name="meta-llama/Llama-2-7b-hf",
                   num_prompts=100,
                   input_len=512,
                   output_len=128):
    """Benchmark vLLM throughput and latency."""
    
    # Initialize vLLM engine
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,   # Use 90% of GPU for KV cache
        max_model_len=4096,
        block_size=16,                # PagedAttention block size
        enable_prefix_caching=True,   # Enable prefix sharing
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,    # greedy for deterministic benchmarks
        max_tokens=output_len,
    )
    
    # Generate synthetic prompts
    prompts = [f"Write a detailed essay about topic number {i}. "
               f"{'padding ' * (input_len // 2)}" for i in range(num_prompts)]
    
    # Warm-up
    llm.generate(prompts[:2], sampling_params)
    
    # Benchmark
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    
    total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    print(f"Model:            {model_name}")
    print(f"Requests:         {num_prompts}")
    print(f"Total time:       {elapsed:.2f} sec")
    print(f"Throughput:        {num_prompts/elapsed:.1f} req/sec")
    print(f"Input tok/sec:    {total_input_tokens/elapsed:.0f}")
    print(f"Output tok/sec:   {total_output_tokens/elapsed:.0f}")
    print(f"Total tok/sec:    {(total_input_tokens+total_output_tokens)/elapsed:.0f}")
    
    # Per-request latency distribution
    latencies = [o.metrics.finished_time - o.metrics.arrival_time 
                 for o in outputs if o.metrics]
    if latencies:
        import statistics
        print(f"\nLatency (p50):    {statistics.median(latencies)*1000:.0f} ms")
        print(f"Latency (p99):    {sorted(latencies)[int(0.99*len(latencies))]*1000:.0f} ms")

# Uncomment to run:
# benchmark_vllm()
```

### Performance Comparison Table

| System | Throughput (tok/s) | Memory Util. | Continuous Batch | PagedAttention |
|--------|-------------------|-------------|-----------------|----------------|
| HuggingFace `generate()` | ~500 | ~30% | No | No |
| Text Generation Inference | ~2,000 | ~60% | Yes | No |
| vLLM | ~5,000 | ~95% | Yes | Yes |
| TensorRT-LLM | ~6,000 | ~93% | Yes | Paged KV |
| SGLang | ~5,500 | ~95% | Yes | RadixAttention |

*Approximate numbers for LLaMA-2 7B, A100 80GB, batch of 64, 512 input + 128 output tokens.*

---

## Hands-On Exercise: Build a Mini Block Manager

```python
class MiniBlockManager:
    """Simplified block manager to understand PagedAttention allocation."""
    
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # request_id → list of block indices
        self.ref_counts = {}    # block_idx → reference count
    
    def allocate(self, request_id: str, num_tokens: int):
        """Allocate blocks for a new request."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError(f"Need {num_blocks_needed} blocks, "
                              f"only {len(self.free_blocks)} free")
        
        blocks = [self.free_blocks.pop(0) for _ in range(num_blocks_needed)]
        self.block_tables[request_id] = blocks
        for b in blocks:
            self.ref_counts[b] = 1
        return blocks
    
    def append_token(self, request_id: str):
        """Append a token — may need a new block."""
        blocks = self.block_tables[request_id]
        current_tokens = len(blocks) * self.block_size  # simplified
        # In reality, track partially-filled last block
        if not self.free_blocks:
            raise MemoryError("No free blocks for token append")
        new_block = self.free_blocks.pop(0)
        blocks.append(new_block)
        self.ref_counts[new_block] = 1
    
    def free(self, request_id: str):
        """Free all blocks for a completed request."""
        for block in self.block_tables.pop(request_id):
            self.ref_counts[block] -= 1
            if self.ref_counts[block] == 0:
                self.free_blocks.append(block)
                del self.ref_counts[block]
    
    def share_prefix(self, src_id: str, dst_id: str, n_prefix_blocks: int):
        """CoW: share prefix blocks between two requests."""
        src_blocks = self.block_tables[src_id][:n_prefix_blocks]
        self.block_tables[dst_id] = src_blocks.copy()
        for b in src_blocks:
            self.ref_counts[b] += 1
    
    def utilization(self):
        total = len(self.free_blocks) + sum(1 for _ in self.ref_counts)
        used = sum(1 for _ in self.ref_counts)
        return used / total if total > 0 else 0

# Exercise: simulate 50 concurrent requests arriving and leaving
```

### Exercise Tasks

1. **Run the block manager simulation**: Simulate 50 requests with random input lengths (100-2000 tokens) and output lengths (50-500 tokens). Track utilization over time and compare with a pre-allocated baseline.
2. **Implement swap-out/swap-in**: Add methods to move blocks to/from CPU when GPU is full. Measure the preemption cost.
3. **Prefix sharing benchmark**: Simulate 100 requests all sharing a 512-token system prompt. Calculate memory saved vs independent allocation.

---

## Key Takeaways

1. **Traditional KV cache wastes 60-80% of memory** on internal and external fragmentation due to contiguous pre-allocation
2. **PagedAttention maps KV cache to fixed-size blocks** indexed by per-request block tables — exactly like OS virtual memory
3. **Copy-on-write** enables memory sharing for common prefixes (system prompts), saving 30-50% memory
4. **Continuous batching at iteration level** means requests join and leave every decode step — no head-of-line blocking
5. **Chunked prefill** prevents long prefill requests from starving decode requests
6. **vLLM achieves >95% memory utilization** vs ~24% for naïve allocation, translating to 3-24× higher throughput

---

## Further Reading

- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"** (SOSP 2023)
- **vLLM project**: https://github.com/vllm-project/vllm — source code + architecture docs
- **Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs"** (2024) — RadixAttention
- **NVIDIA TensorRT-LLM**: paged KV cache implementation — https://github.com/NVIDIA/TensorRT-LLM
- **Yu et al., "ORCA: A Distributed Serving System for Transformer-Based Generative Models"** (OSDI 2022) — continuous batching

---

## Tomorrow's Preview

We've optimized *how* we store the cache and *how* we schedule requests. But there's still the fundamental bottleneck: autoregressive generation requires N sequential forward passes for N tokens. What if you could generate 5 tokens in the time of 1? **Day 60: Speculative Decoding** introduces draft-and-verify algorithms that break the sequential barrier by predicting multiple tokens in parallel and verifying them in a single forward pass.
