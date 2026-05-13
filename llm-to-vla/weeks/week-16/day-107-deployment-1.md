# Day 107: Deployment — Latency, Compute & Edge Hardware

> **Phase VII — VLAs: Architecture to Deployment | Week 16 | 2.5 hours**
> *"A 7B parameter model running at 3 Hz is a research demo. The same model running at 20 Hz on edge hardware is a product." — Deployment Reality*

## Navigation
- **Previous:** [Day 106: World Models](day-106-world-models.md)
- **Next:** [Day 108: Deployment — Safety & Monitoring](day-108-deployment-2.md)
- **Week:** [Week 16 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 107.1 Deployment Latency Budget

```
Total control loop: ≤ 50ms (20 Hz) for manipulation
                    ≤ 100ms (10 Hz) for mobile manipulation

Budget breakdown:
  Image capture + transfer:     5 ms
  Vision encoder:              10 ms  ← Bottleneck #1
  VLM reasoning:               20 ms  ← Bottleneck #2
  Action head:                  5 ms
  Safety check:                 2 ms
  Communication:                3 ms
  Motor command:                5 ms
  ────────────────────────────────────
  Total:                       50 ms
```

### 107.2 Model Compression Techniques

| Technique | Compression | Speed Gain | Quality Loss | Best For |
|-----------|------------|------------|--------------|----------|
| FP16 | 2× | 1.5-2× | Negligible | Always use |
| INT8 (PTQ) | 4× | 2-3× | <1% | Standard deployment |
| INT4 (GPTQ) | 8× | 3-5× | 2-5% | Edge devices |
| Pruning | 2-5× | 2-4× | 1-3% | Structured sparsity |
| Distillation | 10-50× | 5-20× | 3-10% | Dedicated student |
| ONNX + TensorRT | — | 2-5× | None | NVIDIA hardware |

### 107.3 Vision Encoder Optimization

The vision encoder is often the bottleneck:

```python
# Strategy 1: Reduce resolution
# ViT-Large at 224×224: ~15ms
# ViT-Large at 128×128: ~6ms  (2.5× faster, ~3% quality drop)

# Strategy 2: Smaller backbone
# ViT-Large (307M): ~15ms
# ViT-Base  (86M):  ~5ms
# ViT-Small (22M):  ~2ms
# MobileNetV3:       ~1ms

# Strategy 3: Cache when scene is static
# Key insight: if no hand/object movement in frame,
# vision features are the same as last step.
# Check: ||frame_t - frame_{t-1}|| < threshold
# If static: reuse cached features (0ms vs 15ms)
```

### 107.4 Edge Deployment Hardware

```
Tier 1: Cloud GPU (best quality, highest latency)
  NVIDIA A100/H100 — unlimited compute
  Latency: 5-20ms model + 10-50ms network
  Best for: fleet management, offline training

Tier 2: On-robot GPU (balanced)
  NVIDIA Jetson Orin (275 TOPS) — $1500
  Latency: 15-50ms
  Best for: autonomous robots

Tier 3: Edge TPU / NPU (low power)
  Google Edge TPU (4 TOPS) — $75
  Intel Myriad X (1 TOPS)  — $50
  Latency: 20-100ms (small models only)
  Best for: simple perception + small action head

Tier 4: Custom ASIC (future)
  Specialized VLA accelerator
  Latency: <5ms (theoretical)
  Best for: mass-produced robots
```

### 107.5 Serving Architecture

```
Option A: On-robot inference
  ┌──────────────────────────┐
  │  Robot (Jetson Orin)     │
  │  ┌──────────────────┐   │
  │  │  VLA model (INT8) │   │
  │  │  TensorRT runtime │   │
  │  └──────────────────┘   │
  │  Latency: 30-50ms       │
  │  No network dependency   │
  └──────────────────────────┘

Option B: Cloud inference
  ┌──────────┐   WiFi/5G   ┌──────────┐
  │  Robot   │ ←─────────→ │  Cloud   │
  │  Camera  │   20-50ms   │  A100    │
  │  Motors  │   network   │  VLA     │
  └──────────┘             └──────────┘
  Latency: 40-100ms (network dependent)

Option C: Hybrid (recommended)
  ┌───────────────────────────────┐
  │  Robot (Jetson)               │
  │  ┌──────────────────┐        │
  │  │  Small action     │←─ VLM │ ← Cloud
  │  │  head (local)     │  feats │   (async)
  │  └──────────────────┘        │
  │  Latency: 10ms local         │
  │  VLM updates: 200ms async    │
  └───────────────────────────────┘
```

---

## Implementation (60 min)

### Model Optimization Pipeline

```python
import torch
import time
import numpy as np

class LatencyProfiler:
    """Profile VLA inference latency."""

    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    def profile(self, input_shape, n_warmup=10, n_runs=100):
        """Measure inference latency."""
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                self.model(dummy_input)

        # Synchronize GPU
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                self.model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "std_ms": np.std(latencies),
            "max_hz": 1000 / np.percentile(latencies, 95),
        }

class ModelQuantizer:
    """Quantize model for deployment."""

    @staticmethod
    def to_fp16(model):
        """Convert to FP16."""
        return model.half()

    @staticmethod
    def dynamic_int8(model):
        """Dynamic INT8 quantization (PyTorch)."""
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    @staticmethod
    def export_onnx(model, dummy_input, path="model.onnx"):
        """Export to ONNX for TensorRT/other runtimes."""
        torch.onnx.export(
            model, dummy_input, path,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"},
                         "action": {0: "batch"}},
            opset_version=14,
        )
        print(f"Exported to {path}")

class VisionCacheManager:
    """Cache vision encoder outputs when scene is static."""

    def __init__(self, encoder, change_threshold=0.02):
        self.encoder = encoder
        self.threshold = change_threshold
        self.cached_features = None
        self.cached_frame = None
        self.cache_hits = 0
        self.total_calls = 0

    @torch.no_grad()
    def encode(self, frame):
        self.total_calls += 1

        if self.cached_frame is not None:
            # Check if scene changed
            diff = (frame - self.cached_frame).abs().mean().item()
            if diff < self.threshold:
                self.cache_hits += 1
                return self.cached_features

        # Scene changed or first call: compute features
        self.cached_features = self.encoder(frame)
        self.cached_frame = frame.clone()
        return self.cached_features

    @property
    def cache_hit_rate(self):
        return self.cache_hits / max(1, self.total_calls)

class HybridInferenceServer:
    """Hybrid local + cloud inference."""

    def __init__(self, local_action_head, vision_cache, cloud_interval_ms=200):
        self.action_head = local_action_head
        self.vision_cache = vision_cache
        self.cloud_interval = cloud_interval_ms / 1000
        self.last_cloud_time = 0
        self.vlm_features = None

    def local_step(self, proprio):
        """Fast local action prediction (~5ms)."""
        if self.vlm_features is None:
            return None

        with torch.no_grad():
            action = self.action_head(
                torch.cat([self.vlm_features, proprio.unsqueeze(0)], dim=-1)
            )
        return action

    def cloud_update(self, frame, instruction):
        """Async VLM update from cloud (~200ms, non-blocking)."""
        # In practice: send to cloud, receive features asynchronously
        # Simulated here:
        self.vlm_features = self.vision_cache.encode(frame)
        self.last_cloud_time = time.time()

    def should_update_cloud(self):
        return (time.time() - self.last_cloud_time) > self.cloud_interval

# Demo: Profiling
print("=== Model Size Comparison ===")
sizes = {
    "ViT-Small (22M)": torch.nn.Sequential(
        torch.nn.Linear(512, 384), torch.nn.ReLU(), torch.nn.Linear(384, 256)),
    "ViT-Base (86M)": torch.nn.Sequential(
        torch.nn.Linear(512, 768), torch.nn.ReLU(), torch.nn.Linear(768, 256)),
    "ViT-Large (307M)": torch.nn.Sequential(
        torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 256)),
}

for name, model in sizes.items():
    params = sum(p.numel() for p in model.parameters())
    profiler = LatencyProfiler(model, device="cpu")
    result = profiler.profile((1, 512), n_warmup=5, n_runs=50)
    print(f"{name}: {params:,} params, {result['mean_ms']:.2f}ms, "
          f"{result['max_hz']:.0f} Hz")

# Vision cache demo
encoder = torch.nn.Linear(512, 256)
cache = VisionCacheManager(encoder, change_threshold=0.01)
frame = torch.randn(1, 512)
_ = cache.encode(frame)
_ = cache.encode(frame + torch.randn(1, 512) * 0.001)  # Static → cache hit
_ = cache.encode(frame + torch.randn(1, 512) * 0.1)    # Changed → recompute
print(f"\nVision cache hit rate: {cache.cache_hit_rate:.0%}")
```

---

## Exercise (45 min)

1. **Quantization comparison:** Take a VLA model. Compare FP32, FP16, INT8, and INT4 quantization. Measure latency and action prediction accuracy on 1000 test samples.

2. **Resolution vs quality:** Train the same model at 224×224, 160×160, 128×128, and 96×96 input resolution. Plot success rate vs latency. Find the Pareto-optimal point.

3. **Vision caching:** On a real manipulation trajectory, measure what percentage of consecutive frames have <2% pixel difference. What's the effective speedup from caching?

4. **Hybrid architecture:** Simulate a hybrid system: VLM updates at 5 Hz, local action head at 50 Hz. Compare with VLA-only at 10 Hz. Which achieves better task success?

---

## Key Takeaways

1. **50ms total budget** for 20 Hz manipulation control
2. **Vision encoder is the bottleneck** — reduce resolution, cache, or use smaller backbone
3. **Quantization (INT8) is free performance** — always deploy with at least FP16
4. **Hybrid cloud+edge** decouples VLM reasoning from fast local control
5. **Profile before optimizing** — measure p95 latency, not just mean

---

## Connection to the Thread

> *Latency and compute are necessary but not sufficient. Tomorrow: deployment safety and monitoring — the systems that prevent your VLA from breaking things, and the dashboards that tell you when something's wrong.*

---

## Further Reading

- NVIDIA TensorRT documentation for model optimization
- Jetson Orin developer guide for edge deployment
- Google Coral Edge TPU for low-power inference
