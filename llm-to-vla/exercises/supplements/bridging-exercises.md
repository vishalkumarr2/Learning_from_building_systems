# Bridging Exercises: LLM Concepts → Robot Concepts

> **Purpose**: Explicitly connect abstract LLM/ML concepts to their robotics applications.  
> Each bridge shows: "You learned X in exercise N → here's how it appears in robotics as Y."

---

## Bridge 1: Attention Mechanism → Sensor Fusion

**From**: Exercise 02 (Attention from Scratch)  
**To**: Exercise 12+ (Robot Manipulation)

### The Connection

In NLP, attention learns "which words matter for predicting the next word."  
In robotics, the same mechanism learns "which sensor readings matter for the next action."

```python
"""
Cross-attention for multi-modal sensor fusion in robot control.

A robot has:
  - Camera image (224×224×3) → 196 patch tokens after ViT
  - Proprioception (joint angles, velocities) → 14 values
  - Force/torque sensor → 6 values
  - Language instruction → 20 tokens

The VLA's attention mechanism learns to:
  - Look at the gripper region when grasping
  - Attend to force readings when making contact
  - Focus on instruction tokens when deciding WHAT to do
  - Focus on image tokens when deciding WHERE to do it
"""

import torch
import torch.nn as nn


class SensorFusionAttention(nn.Module):
    """Cross-attention that fuses multiple robot sensor modalities.
    
    Query: action prediction needs information
    Keys/Values: all available sensor data
    
    This IS the same scaled dot-product attention from Exercise 02,
    just applied to robot data instead of text tokens!
    """
    
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Project each modality into shared d_model space
        self.image_proj = nn.Linear(768, d_model)   # ViT features
        self.proprio_proj = nn.Linear(14, d_model)   # Joint state
        self.force_proj = nn.Linear(6, d_model)      # Force/torque
        self.lang_proj = nn.Linear(512, d_model)     # Language embeddings
        
        # Action query (learned)
        self.action_query = nn.Parameter(torch.randn(1, 7, d_model))  # 7 DoF
    
    def forward(self, image_features, proprio, force, lang_features):
        """
        The action query "asks" all sensors: what should I do next?
        Attention weights reveal which sensor is most relevant right now.
        """
        # Project all modalities to same space
        img_tokens = self.image_proj(image_features)   # (B, 196, d)
        prop_token = self.proprio_proj(proprio).unsqueeze(1)  # (B, 1, d)
        force_token = self.force_proj(force).unsqueeze(1)     # (B, 1, d)
        lang_tokens = self.lang_proj(lang_features)    # (B, 20, d)
        
        # Concatenate all sensor tokens
        all_sensors = torch.cat([img_tokens, prop_token, force_token, lang_tokens], dim=1)
        # Shape: (B, 196+1+1+20, d) = (B, 218, d)
        
        # Action queries attend to all sensors
        query = self.action_query.expand(image_features.shape[0], -1, -1)
        
        # This is Exercise 02's attention, applied to robot data!
        action_features, attn_weights = self.attention(query, all_sensors, all_sensors)
        # action_features: (B, 7, d) — one feature per joint
        # attn_weights: (B, 7, 218) — reveals which sensors each joint "looks at"
        
        return action_features, attn_weights


# Exercise: Visualize attn_weights
# - During a reaching motion, which tokens does joint 1 (shoulder) attend to?
# - During grasping, does the gripper joint attend more to force or vision?
# - Plot attention heatmap overlaid on the camera image
```

### Key Insight
> Self-attention in transformers = learned, dynamic sensor fusion for robots.
> The model automatically decides "what information is relevant right now"
> instead of a hand-designed fusion rule.

---

## Bridge 2: Positional Encoding → Temporal Awareness

**From**: Exercise 03 (Build Transformer — Sinusoidal PE, RoPE)  
**To**: Exercises 10, 15 (Imitation Learning, Trajectory Planning)

### The Connection

In language: position tells the model "this word is 5th in the sentence."  
In robotics: position tells the model "this observation is from 0.3 seconds ago"
and "this action should happen 0.1 seconds from now."

```python
"""
Temporal encoding for robot action sequences.

A robot trajectory is a sequence — just like text!
  Text: [token_1, token_2, ..., token_n] with positions [0, 1, ..., n-1]
  Robot: [state_t0, state_t1, ..., state_tn] with timestamps [0, 0.1, ..., n*0.1]

But robot time is CONTINUOUS, not discrete!
→ We need continuous positional encoding (same math, different input).
"""

import torch
import numpy as np


def sinusoidal_time_encoding(timestamps: torch.Tensor, d_model: int = 256):
    """Continuous-time positional encoding for robot trajectories.
    
    Same formula as Exercise 03, but input is real-valued time (seconds)
    instead of integer position!
    
    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    
    Args:
        timestamps: (batch, seq_len) — time in seconds
        d_model: embedding dimension
    """
    # Exactly the same formula you implemented in Exercise 03!
    half_d = d_model // 2
    freqs = torch.exp(torch.arange(half_d) * -(np.log(10000.0) / half_d))
    
    # Outer product: timestamps × frequencies
    # timestamps: (B, T) → (B, T, 1)
    # freqs: (half_d,) → (1, 1, half_d)
    args = timestamps.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
    
    encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return encoding  # (B, T, d_model)


# Why this matters for robots:
# 1. Variable-rate observations: cameras at 30Hz, force at 1000Hz
#    → Continuous time encoding handles irregular spacing naturally
#
# 2. Action chunking: "execute this action 0.15s from now"
#    → Time encoding tells the model WHEN to apply each action
#
# 3. History context: "the robot was here 2 seconds ago"
#    → Temporal encoding captures relative time relationships
#
# Exercise: Generate a 5-second trajectory at irregular timestamps.
# Encode with sinusoidal_time_encoding. Verify that nearby timestamps
# have similar encodings (high cosine similarity) and distant ones don't.
```

---

## Bridge 3: Tokenization → Action Discretization

**From**: Exercise 04 (BPE Tokenizer)  
**To**: Exercise 11 (VLA Experiments — Action Tokenization)

### The Connection

BPE for text: continuous signal (text) → discrete tokens that a transformer can process.  
Action tokenization: continuous signal (joint angles) → discrete tokens for a VLA.

```python
"""
The exact same problem, different domain.

Text:   "Hello world" → [15496, 995]  (BPE tokens)
Action: [0.15, -0.32, 0.87, ...] → [148, 86, 222, ...]  (action tokens)

Both solve: "How do we represent continuous/complex data as a finite vocabulary?"

BPE insight: Frequent character pairs get their own token → efficient encoding
Action insight: Frequent action regions get finer resolution → better control

This is why OpenVLA uses 256 bins — it's the "vocabulary size" for actions!
"""

# Direct analogy table:
#
# | Text Domain          | Robot Domain              |
# |---------------------|---------------------------|
# | Characters          | Raw joint angles (float)  |
# | BPE vocabulary      | Action bin edges          |
# | Subword tokens      | Discretized action tokens |
# | Vocab size (32000)  | Bin count (256)           |
# | OOV handling        | Out-of-range clamping     |
# | Tokenizer training  | Bin edge optimization     |
#
# Key difference: text has long-tail distribution (many rare words),
# while actions are typically concentrated near zero with occasional extremes.
# → μ-law tokenization (Exercise 11, §2.7) is like BPE for actions!
```

---

## Bridge 4: Fine-tuning & LoRA → Domain Adaptation for Robots

**From**: Exercise 05 (Fine-tune LLM)  
**To**: Exercise 11 (VLA Fine-tuning with LoRA)

### The Connection

```python
"""
Same technique, different domain:

Text fine-tuning:
  Base: LLaMA-7B trained on internet text
  Task: Add medical knowledge via LoRA
  Data: 10K medical Q&A pairs
  Result: Medical chatbot (base knowledge preserved!)

VLA fine-tuning:
  Base: OpenVLA-7B trained on Open X-Embodiment
  Task: Add new robot/task via LoRA
  Data: 50 demonstrations of "pick up cup"
  Result: Cup-picking specialist (other skills preserved!)

The math is IDENTICAL:
  W_original + ΔW ≈ W_original + B × A  (low-rank update)
  
  For text: ΔW captures "medical vocabulary" patterns
  For VLA: ΔW captures "how THIS gripper picks up cups"
"""

# Why LoRA works for robots:
# 1. The base model already knows "how objects look" and "what grasping means"
# 2. Adaptation only needs to learn "the specifics of this robot's kinematics"
# 3. Low-rank assumption: the delta between "generic grasping" and 
#    "this-robot grasping" is low-dimensional (just needs kinematics adjustments)
#
# Exercise: Calculate LoRA efficiency for a VLA
# - OpenVLA has 7B params. LoRA rank=16 on attention layers.
# - How many trainable params? (16 × 2 × d_model × n_layers ≈ 12M)
# - How many demos needed? (Rule of thumb: 10-100 for LoRA, 100K for full)
# - What's the fine-tuning time on 1 A100? (minutes, not days!)
```

---

## Bridge 5: KV-Cache → Action Prediction Efficiency

**From**: Exercise 03 (KV-Cache, Exercise 6)  
**To**: Exercise 17 (VLA Systems Engineering)

### The Connection

```python
"""
KV-cache in text generation:
  - Context: "The robot should pick up the red" 
  - KV-cache stores attention states for all past tokens
  - Next token generation only processes 1 new token
  - Speedup: O(1) per new token instead of O(n)

KV-cache in VLA action generation:
  - Context: 256 vision tokens + 20 language tokens = 276 tokens
  - These are FIXED for the current timestep
  - Action tokens (7 values) are generated autoregressively
  - KV-cache: process 276 context once, then 7 cheap forward passes
  
Without cache: 276+1, 276+2, 276+3, ..., 276+7 → process 276 tokens 7 times!
With cache:    276 (once), 1, 1, 1, 1, 1, 1 → process 276 only once!

For VLA: saves ~7× compute on the context encoding step!
This is why KV-cache is CRITICAL for meeting real-time constraints.
"""

# Numerical example:
# VLA with d_model=4096, 32 layers:
# KV-cache size for 276 context tokens:
#   2 (K+V) × 32 layers × 276 tokens × 4096 dim × 2 bytes(fp16)
#   = 2 × 32 × 276 × 4096 × 2 = ~138 MB
#
# But saves: 6 × (2 × 276 × 4096² × 32) FLOPs of recomputation
# At 100 TFLOPS, that's ~26ms saved per action prediction!
# 
# This is the difference between fitting in 100ms budget or not.
```

---

## Bridge 6: Diffusion Models → Action Generation

**From**: Exercise 09 (RL & Diffusion Basics)  
**To**: Exercises 10-11 (Imitation Learning, VLA)

### The Connection

```python
"""
Image diffusion:  noise → pixel values  (iterative denoising)
Action diffusion: noise → action values  (iterative denoising)

Same math, different output space!

DDPM for images: p(x_{t-1} | x_t) — denoise pixels one step
DDPM for actions: p(a_{t-1} | a_t, observation) — denoise actions one step

Why use diffusion for actions?
  1. Multi-modal: can represent multiple valid actions (reach left OR right)
  2. Expressive: doesn't assume Gaussian action distribution
  3. Quality: iterative refinement gives better actions than one-shot prediction

This is how Diffusion Policy works:
  - Start with random noise shaped like (chunk_size, action_dim)
  - Conditioned on observation, iteratively denoise
  - After K steps, get a clean action chunk
  - Uses DDIM (from Exercise 09) for fast sampling (20 steps instead of 1000)
"""

# Direct mapping:
# | Image Diffusion     | Action Diffusion (Diffusion Policy)  |
# |--------------------|-----------------------------------------|
# | Image (256×256×3)  | Action chunk (16×7)                    |
# | Text condition     | Observation condition (image + proprio) |
# | U-Net backbone     | 1D temporal U-Net or Transformer       |
# | 1000 steps (DDPM)  | 10-20 steps (DDIM) for real-time       |
# | FID score          | Success rate + smoothness              |
#
# The DDIM trick from Exercise 09 is ESSENTIAL:
# Without DDIM: 1000 denoising steps × 5ms = 5 seconds (way too slow!)
# With DDIM:    10 steps × 5ms = 50ms (fits in real-time budget!)
```

---

## Bridge 7: Gradient Checkpointing → Training VLAs on Consumer GPUs

**From**: Exercise 03 (Gradient Checkpointing, Exercise 7)  
**To**: Exercise 05 (Fine-tuning), Exercise 11 (VLA Training)

### The Connection

```python
"""
The memory problem for VLA training:

OpenVLA (7B params):
  - Model weights (fp16): 14 GB
  - Gradients: 14 GB (or 0 with LoRA on frozen weights)
  - Optimizer (Adam fp32): 56 GB (or 64 MB for LoRA)
  - Activations: 4-20 GB depending on batch size
  
Full fine-tuning: 14 + 14 + 56 + 8 = 92 GB → needs 2× A100 80GB
LoRA only:        14 + 0.03 + 0.06 + 8 = ~22 GB → fits 1× A100 24GB!
LoRA + checkpointing: 14 + 0.03 + 0.06 + 2 = ~16 GB → fits consumer GPU!

Gradient checkpointing (Exercise 03, Ex 7) is what makes that activation
memory drop from 8 GB to 2 GB possible.

Combined with QLoRA (4-bit base + LoRA + checkpointing):
  4-bit model: 3.5 GB
  LoRA: 0.1 GB  
  Activations (checkpointed): 2 GB
  Total: ~6 GB → fits on an RTX 3060!
"""
```

---

## Bridge 8: Safety Monitoring → Real-Time Robot Constraints

**From**: Exercise 16 (Safety Monitoring & Recovery)  
**To**: Exercise 17 (VLA Systems Engineering)

### The Connection

```python
"""
The safety framework from Exercise 16 plugs directly into 
the failure recovery system in Exercise 17.

Exercise 16 teaches:
  - Monitor joint limits, velocities, torques
  - Detect collisions via force thresholds
  - Implement workspace boundaries
  - Emergency stop protocols

Exercise 17 adds the VLA-specific layer:
  - What if the MODEL predicts an unsafe action? (Validate before execute)
  - What if the model is SLOW? (Stall recovery)
  - What if the model is WRONG? (Confidence estimation)
  
The key insight: ML models are not controllers. They are ADVISORS.
A traditional controller should always be the final safety gate:

    VLA prediction → Safety Filter → Low-level Controller → Robot
         ↑                ↑
    "What to do"    "Is it safe?"
    
If the safety filter rejects an action, the controller falls back to:
  1. Hold position (safest)
  2. Slow stop (decelerate to zero)
  3. Return to home position (recovery)
"""
```

---

## Summary: The LLM→Robot Mapping

| LLM Concept | Robot Application | Exercise Link |
|-------------|-------------------|---------------|
| Self-attention | Sensor fusion | 02 → 12 |
| Positional encoding | Temporal awareness | 03 → 10, 15 |
| KV-cache | Real-time inference | 03 → 17 |
| Gradient checkpointing | Train on consumer GPU | 03 → 05, 11 |
| BPE tokenization | Action discretization | 04 → 11 |
| FLOP counting | Real-time feasibility | 04 → 17 |
| LoRA fine-tuning | Robot-specific adaptation | 05 → 11 |
| Memory profiling | Training budget planning | 05 → 11 |
| ViT | Visual observation encoding | 06 → 08, 11 |
| Depth estimation | 3D workspace understanding | 07 → 12 |
| VLM | Instruction grounding | 08 → 11 |
| PPO / RL | Policy optimization | 09 → 12 |
| DDIM | Fast action sampling | 09 → 10 |
| Diffusion Policy | Multi-modal action generation | 09 → 10, 11 |
| Temporal ensembling | Smooth action execution | 10 → 17 |
| Action chunking | Control rate reduction | 10, 11 → 17 |
| Safety monitoring | Action validation | 16 → 17 |

---

*Use this file as a reference when starting each new exercise — check which bridges are relevant and review the previous exercise's code with the robot application in mind.*
