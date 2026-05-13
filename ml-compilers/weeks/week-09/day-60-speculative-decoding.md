# Day 60: Speculative Decoding

> **Phase IV · Week 9 · Day 60 of 70 · 2.5 hours**

*"Generate 5 tokens for the cost of 1 verification — if you guess right. And with a good draft model, you guess right 70-90% of the time."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 59: vLLM & PagedAttention](day-59-vllm-pagedattention.md) | [Day 61: LLM Quantization](day-61-llm-quantization.md) | [Week 9: LLM Serving Systems](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Autoregressive LLM inference has an unbreakable sequential dependency: you need token $t_n$ before you can compute token $t_{n+1}$. Or do you? Speculative decoding circumvents this by using a small, fast **draft model** to predict multiple tokens in parallel, then **verifying** them all in a single forward pass of the large target model. The mathematical guarantee: the output distribution is **identical** to the target model — zero quality loss, 2-3× speedup. This technique is deployed in production at Google, Meta, and OpenAI, and understanding it is critical because it fundamentally changes the latency-throughput equation for LLM serving.

---

## 1. The Core Algorithm

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) works in three steps:

```
Speculative Decoding — Draft & Verify
═══════════════════════════════════════════════════════════════

  Target model: M_target (e.g., LLaMA-2 70B)   — slow, accurate
  Draft model:  M_draft  (e.g., LLaMA-2 7B)    — fast, approximate

  Context so far: "The capital of France"

  Step 1: DRAFT — Generate K tokens with M_draft (fast)
  ──────────────────────────────────────────────────────
  M_draft("The capital of France") → " is"     (q₁)
  M_draft("... France is")         → " Paris"  (q₂)
  M_draft("... is Paris")          → ","       (q₃)
  M_draft("... Paris,")            → " a"      (q₄)
  M_draft("... , a")               → " city"   (q₅)

  Draft tokens: [" is", " Paris", ",", " a", " city"]
  Draft probs:  [q₁,    q₂,      q₃,  q₄,   q₅   ]

  Step 2: VERIFY — Score all K+1 positions with M_target (one pass!)
  ──────────────────────────────────────────────────────────────────
  M_target("The capital of France is Paris , a city") → [p₁,p₂,p₃,p₄,p₅,p₆]

  One forward pass scores ALL draft tokens simultaneously!

  Step 3: ACCEPT/REJECT — Modified rejection sampling
  ────────────────────────────────────────────────────
  For each draft token i (left to right):
    Accept with probability min(1, p_i(x) / q_i(x))

    " is"    → p₁(" is")=0.95, q₁(" is")=0.90  → accept (0.95/0.90>1)  ✓
    " Paris"  → p₂("Paris")=0.85, q₂("Paris")=0.80 → accept             ✓
    ","       → p₃(",")=0.70,  q₃(",")=0.75   → accept w.p. 0.70/0.75  ✓
    " a"      → p₄(" a")=0.15, q₄(" a")=0.60  → REJECT (p/q=0.25)      ✗
    " city"   → skipped (previous rejected)

  Result: accepted 3 tokens + sample 1 new from adjusted distribution
  Total: 4 tokens generated in ~1 target model forward pass!
```

---

## 2. Why It Works — Rejection Sampling Guarantee

The acceptance criterion ensures the output follows the **exact** target distribution $p$, not the draft distribution $q$. This is a mathematical guarantee, not an approximation:

For each draft token $x$ sampled from $q(x)$:

$$\text{Accept with probability } \min\left(1, \frac{p(x)}{q(x)}\right)$$

If rejected at position $i$, sample a **correction token** from the residual distribution:

$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

**Theorem**: The resulting token sequence is distributed exactly as if it were sampled from $p$ autoregressively.

```
Why Rejection Sampling Preserves the Target Distribution
═══════════════════════════════════════════════════════════════

  Probability of accepting token x from draft:
  P(accept x) = q(x) · min(1, p(x)/q(x))
              = min(q(x), p(x))

  Case 1: p(x) ≥ q(x) → P(accept x) = q(x) · 1 = q(x)
  Case 2: p(x) < q(x) → P(accept x) = q(x) · p(x)/q(x) = p(x)

  So P(accept x) = min(p(x), q(x)) for all x

  Total acceptance rate:
  α = Σ_x min(p(x), q(x))   ← higher when q ≈ p

  On rejection, sample from residual:
  p'(x) ∝ max(0, p(x) - q(x))

  Combined: P(output x) = min(p(x), q(x)) + (1-α) · p'(x)
                         = min(p(x), q(x)) + max(0, p(x) - q(x))
                         = p(x)  ✓  Exact target distribution!
```

### Expected Speedup

The expected number of tokens accepted per speculation round with $K$ draft tokens:

$$E[\text{accepted}] = \sum_{i=1}^{K} \prod_{j=1}^{i} \alpha_j + 1$$

where $\alpha_j$ is the acceptance rate at position $j$. With uniform acceptance rate $\alpha$:

$$E[\text{tokens per round}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

| Acceptance Rate $\alpha$ | K=3 | K=5 | K=7 |
|---|---|---|---|
| 0.5 | 1.9 | 2.0 | 2.0 |
| 0.7 | 2.7 | 3.1 | 3.2 |
| 0.8 | 3.4 | 4.1 | 4.4 |
| 0.9 | 3.8 | 5.2 | 6.1 |

At $\alpha = 0.8$ with $K=5$, you get ~4 tokens per verification — a **4× speedup** for decode-bound generation.

---

## 3. Implementation

```python
import torch
import torch.nn.functional as F

def speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    K: int = 5,  # draft length per round
    temperature: float = 1.0,
):
    """Speculative decoding with rejection sampling.
    
    Args:
        target_model: Large model (e.g., 70B) — slow, accurate
        draft_model:  Small model (e.g., 7B)  — fast, approximate
        input_ids: [1, seq_len] input token IDs
        max_new_tokens: maximum tokens to generate
        K: number of draft tokens per speculation round
        temperature: sampling temperature
    
    Returns:
        Generated token IDs
    """
    device = input_ids.device
    generated = input_ids.clone()
    tokens_generated = 0
    
    while tokens_generated < max_new_tokens:
        # ─── Step 1: Draft K tokens ───
        draft_tokens = []
        draft_probs = []
        draft_input = generated.clone()
        
        for _ in range(K):
            with torch.no_grad():
                logits = draft_model(draft_input).logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            draft_probs.append(probs)
            draft_input = torch.cat([draft_input, token], dim=-1)
        
        draft_tokens = torch.cat(draft_tokens, dim=-1)  # [1, K]
        
        # ─── Step 2: Verify with target model (single forward pass!) ───
        verify_input = torch.cat([generated, draft_tokens], dim=-1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # Get target probs at each draft position
        # Position i verifies draft token i
        n_input = generated.shape[1]
        
        # ─── Step 3: Accept/Reject via rejection sampling ───
        accepted = 0
        for i in range(K):
            target_probs_i = F.softmax(
                target_logits[:, n_input + i - 1, :] / temperature, dim=-1
            )
            draft_prob = draft_probs[i][0, draft_tokens[0, i]]
            target_prob = target_probs_i[0, draft_tokens[0, i]]
            
            # Accept with probability min(1, p/q)
            accept_prob = min(1.0, (target_prob / draft_prob).item())
            
            if torch.rand(1).item() < accept_prob:
                accepted += 1
            else:
                # Reject: sample from residual distribution
                residual = torch.clamp(target_probs_i - draft_probs[i], min=0)
                residual = residual / residual.sum()
                correction = torch.multinomial(residual, num_samples=1)
                
                # Accept tokens up to i, plus correction token
                generated = torch.cat([
                    generated,
                    draft_tokens[:, :i],
                    correction
                ], dim=-1)
                tokens_generated += i + 1
                break
        else:
            # All K draft tokens accepted!
            # Sample one more token from target at position K
            bonus_probs = F.softmax(
                target_logits[:, n_input + K - 1, :] / temperature, dim=-1
            )
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            generated = torch.cat([generated, draft_tokens, bonus_token], dim=-1)
            tokens_generated += K + 1
    
    return generated

# Usage:
# output = speculative_decode(llama_70b, llama_7b, prompt_ids, K=5)
```

---

## 4. Draft Model Selection Strategies

The acceptance rate depends critically on how well the draft approximates the target:

```
Draft Model Selection
═══════════════════════════════════════════════════════════════

  Strategy 1: Smaller model from same family
  ──────────────────────────────────────────
  Target: LLaMA-2 70B  →  Draft: LLaMA-2 7B
  Acceptance rate: ~70-80%
  Pros: Easy setup, shared tokenizer, good alignment
  Cons: Draft model still needs GPU memory

  Strategy 2: Quantized version of target
  ────────────────────────────────────────
  Target: LLaMA-2 70B FP16  →  Draft: LLaMA-2 70B INT4
  Acceptance rate: ~85-95% (very high!)
  Pros: Best acceptance rate (same architecture)
  Cons: Draft is still large (just faster per token)

  Strategy 3: Distilled lightweight model
  ────────────────────────────────────────
  Target: LLaMA-2 70B  →  Draft: Custom 500M model
  Acceptance rate: ~60-70%
  Pros: Very fast drafting, minimal memory
  Cons: Lower acceptance, need training data

  Strategy 4: N-gram / retrieval-based
  ─────────────────────────────────────
  Target: any LLM  →  Draft: suffix tree on training data
  Acceptance rate: ~40-60% (domain-dependent)
  Pros: Zero GPU cost for drafting
  Cons: Low acceptance for creative/novel text

  ┌───────────────────────────────────────────────────────┐
  │  Key tradeoff: faster draft ⇄ lower acceptance rate  │
  │  Sweet spot: draft should be 5-10× faster than target │
  │  with >70% acceptance rate                            │
  └───────────────────────────────────────────────────────┘
```

---

## 5. Advanced Variants

### Medusa — Multiple Decoding Heads

Medusa (Cai et al., 2024) eliminates the separate draft model entirely by adding lightweight prediction heads to the target model:

```
Medusa Architecture
═══════════════════════════════════════════════════════════════

  Standard LLM:
  ┌──────────────────────────┐
  │  Transformer layers      │
  │  (frozen)                │──→ LM Head ──→ token t+1
  └──────────────────────────┘

  Medusa LLM:
  ┌──────────────────────────┐
  │  Transformer layers      │──→ LM Head   ──→ token t+1  (original)
  │  (frozen)                │──→ Medusa H₁ ──→ token t+2  (predicted)
  └──────────────────────────┘──→ Medusa H₂ ──→ token t+3  (predicted)
                               ──→ Medusa H₃ ──→ token t+4  (predicted)

  Each Medusa head: single linear layer + residual
  Parameters added: ~0.5% of model size
  Training: fine-tune heads on target model's output distribution

  Tree-based verification:
  ────────────────────────
  Each head proposes top-k candidates → forms a tree:

         t+1: [A, B]
        ╱       ╲
  t+2: [C,D]   [E,F]
       ╱ ╲     ╱  ╲
  t+3: [G,H] [I,J] [K,L]

  Verify all paths in ONE forward pass using tree attention mask!
  Accept the longest valid path → 3-5 tokens per step.
```

### EAGLE — Extrapolation-based Drafting

EAGLE (Li et al., 2024) uses the target model's own hidden states to draft:

```python
# EAGLE drafting concept (simplified)
class EAGLEDraftHead(nn.Module):
    """Draft future tokens using current hidden states."""
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # Predict next hidden state from current
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_state, embedding):
        """
        hidden_state: [B, D] from target model's last layer
        embedding:    [B, D] embedding of last accepted token
        """
        # Concatenate hidden state + embedding for richer features
        combined = torch.cat([hidden_state, embedding], dim=-1)
        next_hidden = self.fc(combined)
        logits = self.lm_head(next_hidden)
        return logits, next_hidden  # next_hidden used for further drafting
```

### SpecInfer — Tree-Based Speculation

SpecInfer (Miao et al., 2024) uses multiple draft models and combines their predictions into a **speculation tree**:

```
SpecInfer — Multi-Draft Tree
═══════════════════════════════════════════════════════════════

  Draft Model A (7B):     "is" → "Paris" → "," → "the"
  Draft Model B (1.3B):   "is" → "Paris" → "." → "It"
  Draft Model C (n-gram): "is" → "the"   → "capital"

  Merged speculation tree:
                    "is"
                   ╱    ╲
              "Paris"   "the"
              ╱    ╲       ╲
            ","    "."    "capital"
            ╱       ╲
          "the"    "It"

  Verify entire tree in one forward pass!
  → Accept longest valid branch
  → More diverse candidates = higher acceptance rate
```

---

## 6. Speculative Decoding in Production Systems

```python
# vLLM speculative decoding configuration
from vllm import LLM, SamplingParams

def setup_speculative_vllm():
    """Configure vLLM with speculative decoding enabled."""
    
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=4,
        speculative_model="meta-llama/Llama-2-7b-hf",  # draft model
        num_speculative_tokens=5,                        # K value
        # Alternative: use ngram-based speculation (no draft model needed)
        # speculative_model="[ngram]",
        # ngram_prompt_lookup_max=4,
    )
    
    return llm

# TensorRT-LLM speculative decoding setup
def trt_llm_speculative_config():
    """TensorRT-LLM config for speculative decoding (conceptual)."""
    config = {
        "target_model": "llama-2-70b",
        "draft_model": "llama-2-7b",
        "max_draft_len": 5,
        "acceptance_threshold": 0.0,   # pure rejection sampling
        "use_medusa": False,
        # Medusa alternative:
        # "use_medusa": True,
        # "medusa_choices": [[0], [0,0], [1], [0,1], [2]],  # tree structure
        # "num_medusa_heads": 3,
    }
    return config
```

### When Speculative Decoding Helps (and When It Doesn't)

```
When to Use Speculative Decoding
═══════════════════════════════════════════════════════════════

  ✓ USE when:
  ──────────
  • Decode latency matters (interactive chat, real-time)
  • Target model is large (≥13B) and memory-bound in decode
  • Draft model has high acceptance rate (>70%)
  • Batch size is small (1-4) — decode is deeply memory-bound
  • Content is somewhat predictable (code, structured text)

  ✗ AVOID when:
  ─────────────
  • Batch size is large (>16) — decode already memory-efficient
  • Draft model has low acceptance rate (<50%)
  • Target model is small — decode is already fast
  • GPU memory is tight — draft model adds memory pressure
  • Creative/diverse text — low acceptance hurts throughput

  ┌────────────────────────────────────────────────────────┐
  │  Batch Size   Without Spec   With Spec    Improvement  │
  │     1         35 tok/s       105 tok/s    3.0×         │
  │     4         120 tok/s      280 tok/s    2.3×         │
  │     16        400 tok/s      520 tok/s    1.3×         │
  │     64        1200 tok/s     1100 tok/s   0.9× (worse!)│
  └────────────────────────────────────────────────────────┘
  At large batches, verification overhead > drafting benefit
```

---

## Hands-On Exercise: Implement & Benchmark Speculative Decoding

```python
import torch
import time

def measure_acceptance_rate(target_logits, draft_logits, temperature=1.0):
    """Measure empirical acceptance rate between target and draft distributions."""
    target_probs = F.softmax(target_logits / temperature, dim=-1)
    draft_probs = F.softmax(draft_logits / temperature, dim=-1)
    
    # Theoretical acceptance rate: sum of min(p, q) for all tokens
    acceptance = torch.min(target_probs, draft_probs).sum(dim=-1)
    return acceptance.mean().item()


def simulate_speculative_speedup(acceptance_rate, K, 
                                  draft_time_ms, verify_time_ms):
    """Simulate expected speedup from speculative decoding."""
    alpha = acceptance_rate
    
    # Expected tokens per round
    expected_tokens = (1 - alpha**(K+1)) / (1 - alpha)
    
    # Time per round = K * draft_time + 1 * verify_time
    time_per_round = K * draft_time_ms + verify_time_ms
    
    # Baseline: 1 token per verify_time
    baseline_ms_per_token = verify_time_ms
    speculative_ms_per_token = time_per_round / expected_tokens
    
    speedup = baseline_ms_per_token / speculative_ms_per_token
    
    print(f"Acceptance rate:     {alpha:.0%}")
    print(f"Draft length K:      {K}")
    print(f"Expected tokens/rnd: {expected_tokens:.1f}")
    print(f"Time per round:      {time_per_round:.1f} ms")
    print(f"Baseline tok/ms:     {1/baseline_ms_per_token:.3f}")
    print(f"Spec tok/ms:         {expected_tokens/time_per_round:.3f}")
    print(f"Speedup:             {speedup:.2f}×")
    return speedup

# Sweep acceptance rate and K
print("=" * 50)
for alpha in [0.6, 0.7, 0.8, 0.9]:
    for K in [3, 5, 7]:
        print(f"\n--- α={alpha}, K={K} ---")
        simulate_speculative_speedup(
            acceptance_rate=alpha, K=K,
            draft_time_ms=1.5,   # 7B draft: ~1.5ms/token
            verify_time_ms=12.0  # 70B target: ~12ms/token
        )
```

### Exercise Tasks

1. **Acceptance rate sweep**: Use two HuggingFace models (e.g., GPT-2 medium as target, GPT-2 small as draft). Measure acceptance rate across 100 prompts. How does it vary by domain (code vs prose vs dialogue)?
2. **Optimal K finder**: Implement a function that finds the optimal $K$ given measured acceptance rate and draft/target timing. Is $K=5$ always best?
3. **Tree-based verification**: Implement Medusa-style tree verification where each position has top-2 candidates. Build the attention mask, run one forward pass, and find the longest accepted path.

---

## Key Takeaways

1. **Speculative decoding generates multiple tokens per target forward pass** by drafting with a fast model and verifying with the slow target
2. **Rejection sampling guarantees exact target distribution** — zero quality loss, mathematically proven
3. **Expected speedup is $(1-\alpha^{K+1})/(1-\alpha)$ divided by the time ratio** — acceptance rate $\alpha > 0.7$ and target/draft ratio > 5× are needed for benefit
4. **Medusa eliminates the draft model** by adding lightweight heads to the target, predicting tokens $t+2, t+3, \ldots$ directly
5. **EAGLE and SpecInfer** push further with hidden-state extrapolation and multi-draft tree speculation
6. **Speculative decoding helps most at small batch sizes** — at large batches, decode is already efficient and verification overhead dominates

---

## Further Reading

- **Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"** (ICML 2023)
- **Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"** (2023)
- **Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"** (2024)
- **Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"** (2024)
- **Miao et al., "SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference"** (2024)

---

## Tomorrow's Preview

We've covered how to serve LLMs faster and cheaper through PagedAttention, continuous batching, and speculative decoding. But there's one more critical lever: making the model itself smaller and faster through quantization. **Day 61: LLM Quantization** dives into GPTQ, AWQ, SqueezeLLM, and the precision ladder from FP16 down to INT4 and beyond — where you lose 0.5% accuracy but gain 4× memory savings.
