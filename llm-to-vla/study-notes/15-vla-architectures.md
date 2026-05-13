# 15 — Vision-Language-Action Models
> Phase VII · Days 92–100 · ~24 hours
> Prerequisites: 10-vision-language-models, 13-imitation-learning
> Learning Objectives: Understand RT-2, OpenVLA, π₀, how VLAs unify vision+language+action

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [14 — Diffusion Policy](14-diffusion-policy.md) | [Curriculum](../CURRICULUM.md) | [16 — Deployment & Hybrid](16-deployment-hybrid.md) |

---

## Why This Matters

Every note in this curriculum has been building toward this moment. We started with
backpropagation (Note 1), learned to compress images (Note 10), compress language
(Notes 6–9), compress behavior (Notes 13–14), and now we arrive at the **grand
unification**: a single transformer that compresses vision, language, AND actions
into one shared representation.

**Vision-Language-Action (VLA) models** are the architectural culmination of the
entire deep learning stack applied to robotics. They answer a deceptively simple
question: *can one neural network see, understand language, and output motor
commands?*

The answer, as of 2024, is **yes** — and the implications are profound.

```
The VLA Lineage:

  CNN (2012)           RNN/LSTM (2014)        RL (2015)
     │                      │                    │
     ▼                      ▼                    ▼
  ViT (2020)          Transformer (2017)    Policy Gradient
     │                      │                    │
     ▼                      ▼                    ▼
  CLIP (2021)          GPT/LLM (2020+)     Imitation Learning
     │                      │                    │
     └──────────┬───────────┘                    │
                ▼                                │
          VLM (2022-23)                          │
          PaLM-E, LLaVA                          │
                │                                │
                └────────────┬───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │   VLA (2023+)   │
                    │  RT-2, OpenVLA  │
                    │   π₀, Octo     │
                    └─────────────────┘
```

---

## 1 — The VLA Concept (Day 92)

### 1.1 One Model, Three Modalities

A VLA model processes three types of information in a single forward pass:

| Modality | Input/Output | Representation |
|----------|-------------|----------------|
| **Vision** | Input | Image patches → visual tokens |
| **Language** | Input | Task instruction → text tokens |
| **Action** | Output | Motor commands → action tokens |

The key insight: if a transformer can handle text and images (VLM), then actions
are *just another token type*. The same attention mechanism that relates "pick up
the red cup" to the visual scene can also relate those to the motor commands
needed to execute the task.

$$
\pi_\theta(a_t \mid o_t, l) = \text{Transformer}([\text{visual\_tokens}(o_t); \text{text\_tokens}(l)])
$$

where $o_t$ is the observation (image), $l$ is the language instruction, and
$a_t$ is the action output.

### 1.2 The RT-1 → RT-2 Evolution

**RT-1 (Robotics Transformer 1, 2022)**:
- Custom architecture: FiLM-conditioned EfficientNet + TokenLearner + Transformer
- Trained from scratch on 130k robot demonstrations
- Discrete action bins (256 per dimension)
- Good performance but NO language generalization

**RT-2 (Robotics Transformer 2, 2023)**:
- Takes a pretrained VLM (PaLM-E or PaLI-X) — billions of parameters
- Fine-tunes it to output actions as text tokens
- Internet-scale knowledge transfers to robot behavior
- Emergent semantic generalization: understands "move to the Taylor Swift
  picture" despite never seeing Taylor Swift in robot data

```
RT-1 Architecture:
  Image → EfficientNet → FiLM(language) → TokenLearner → Transformer → Actions
  [custom, trained from scratch, 35M params]

RT-2 Architecture:
  Image + "Pick up the red cup" → PaLM-E (55B) → "1 128 200 45 ..."
  [pretrained VLM, fine-tuned, 55B params]
```

> 💡 **Key Insight**: RT-2 showed that VLMs pretrained on internet data already
> "know" things about the physical world (object properties, spatial relations,
> common sense) that transfer to robot control — even though they never touched a
> robot during pretraining.

### 1.3 Why Not Just Use Separate Models?

The classical pipeline:
```
Image → Perception Model → Scene Description → LLM Planner → Action Plan → Controller
```

Problems with the pipeline approach:
1. **Information bottleneck**: scene description loses spatial detail
2. **Error propagation**: each stage compounds mistakes
3. **Latency**: multiple model calls per timestep
4. **No end-to-end gradient**: can't optimize the whole system jointly

The VLA approach:
```
Image + "pick up red cup" → Single VLA Model → [dx, dy, dz, dθ, gripper]
```

Advantages:
1. **End-to-end**: gradient flows from action loss all the way back to vision
2. **Implicit representations**: the model learns what to attend to
3. **Language grounding**: language tokens directly attend to visual features
4. **Transfer**: internet knowledge bootstraps robot understanding

### 1.4 The Action Token Idea

The fundamental trick: represent continuous robot actions as discrete tokens
that live in the same vocabulary as text tokens.

For a 7-DoF robot arm: $(x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper})$

Each dimension is discretized into bins:
$$
a_{\text{bin}} = \text{round}\left(\frac{a - a_{\min}}{a_{\max} - a_{\min}} \times (N_{\text{bins}} - 1)\right)
$$

With $N_{\text{bins}} = 256$, we add 256 new tokens to the vocabulary. An action
becomes a sequence of 7 tokens, one per dimension:

```
"Pick up the red cup" + [image] → "128 200 145 128 100 130 255"
                                    x    y    z   r    p    w   grip
```

> ⚠️ **Pitfall**: Discretization resolution matters enormously. With 256 bins
> over a 1-meter workspace, each bin spans ~4mm. For precision tasks (threading
> a needle), you may need 1024+ bins or continuous outputs.

---

## 2 — RT-2: The Breakthrough (Day 93)

### 2.1 Architecture

RT-2 co-opts a pretrained Vision-Language Model and fine-tunes it to also output
actions. The paper explored two base models:

| Base Model | Parameters | Vision Encoder | Language Model |
|-----------|-----------|----------------|----------------|
| PaLI-X | 55B | ViT-22B | UL2 32B |
| PaLM-E | 12B | ViT-4B | PaLM 8B |

```
┌─────────────────────────────────────────────────────────┐
│                    RT-2 Architecture                     │
│                                                         │
│  ┌──────────┐    ┌───────────────────────────────────┐  │
│  │  Camera   │    │  "Pick up the object that is      │  │
│  │  Image    │    │   the same color as a fire truck"  │  │
│  │ 320×512   │    │                                   │  │
│  └─────┬─────┘    └──────────────┬────────────────────┘  │
│        │                         │                       │
│        ▼                         ▼                       │
│  ┌───────────┐          ┌──────────────┐                 │
│  │ ViT-22B   │          │  Tokenizer   │                 │
│  │ (frozen   │          │  (text)      │                 │
│  │  or LoRA) │          │              │                 │
│  └─────┬─────┘          └──────┬───────┘                 │
│        │                       │                         │
│        ▼                       ▼                         │
│  ┌─────────────────────────────────────────────────┐     │
│  │         Transformer Decoder (UL2 / PaLM)        │     │
│  │                                                 │     │
│  │  [vis₁ vis₂ ... visₙ text₁ text₂ ... textₘ]   │     │
│  │              ↓  autoregressive  ↓               │     │
│  │         [act₁ act₂ ... act₇ <EOS>]             │     │
│  └─────────────────────────────────────────────────┘     │
│        │                                                 │
│        ▼                                                 │
│  Action: [1, 128, 200, 45, 128, 128, 255]               │
│          terminate  x    y   z    r    p   gripper       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Action Representation in RT-2

RT-2 represents actions as **text strings of integers** in the range [0, 255].

The action space:
- 7 dimensions: $(x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper})$
- Plus 1 termination token (episode mode or continue)
- Each dimension discretized to 256 bins
- Actions are **relative** (delta from current pose)

The model generates these as autoregressive text:
```
Input:  "What action should the robot take to pick up the red cup?"
        + [image tokens]

Output: "1 128 200 45 128 128 255"
```

The genius: these action tokens share the same embedding space as natural
language tokens. The model doesn't need a separate action head.

### 2.3 Training Details

```python
# RT-2 training pseudocode
class RT2:
    def __init__(self):
        self.vlm = load_pretrained_palm_e()  # 55B params
        # Extend vocabulary with action tokens 0-255
        self.vlm.extend_vocab(256)  # tokens for action bins
    
    def format_action_as_text(self, action_vector):
        """Convert continuous action to text token string."""
        bins = []
        for i, a in enumerate(action_vector):
            # Discretize each dimension to [0, 255]
            lo, hi = ACTION_BOUNDS[i]
            bin_idx = int((a - lo) / (hi - lo) * 255)
            bin_idx = max(0, min(255, bin_idx))
            bins.append(str(bin_idx))
        return " ".join(bins)
    
    def train_step(self, image, instruction, action):
        # Format: image tokens + text tokens → action text tokens
        action_text = self.format_action_as_text(action)
        prompt = f"Q: What action should the robot take to {instruction}?"
        
        # Standard language modeling loss on action tokens
        loss = self.vlm.compute_loss(
            image=image,
            input_text=prompt,
            target_text=action_text
        )
        return loss
```

Training recipe:
1. Start with pretrained PaLM-E (55B) checkpoint
2. Co-fine-tune on mixture: 50% robot data + 50% original VLM data
3. Robot data: ~130k demonstrations from RT-1 dataset
4. Train for ~50k steps with batch size 2048
5. Learning rate: 1e-4 with cosine decay

### 2.4 Emergent Capabilities

The most remarkable result of RT-2 is **emergent semantic generalization**:

| Capability | Example | Why It Works |
|-----------|---------|--------------|
| Semantic reasoning | "Pick up the extinct animal" → picks up dinosaur | VLM knows "dinosaur = extinct" |
| Visual reasoning | "Move to the colorful object" → finds rainbow toy | Visual grounding from CLIP-like training |
| Novel concepts | "Put the banana on the Taylor Swift picture" | Recognizes celebrities from web data |
| Math/logic | "Move to the fruit with most vitamin C" → orange | Factual knowledge from pretraining |

```
Performance on novel semantic concepts:

                     RT-1    RT-2 (PaLI-X)
Seen tasks:          87%        87%
Unseen objects:      32%        62%    ← +30%!
Semantic reasoning:   0%        29%    ← completely new!
Verbal reasoning:     0%        34%    ← completely new!
```

> 💡 **Key Insight**: RT-2 demonstrated that the "common sense" embedded in VLMs
> during internet pretraining is *directly useful* for robot manipulation. A
> model that has read about fire trucks being red can figure out which object
> to pick up when asked for "the object the color of a fire truck."

### 2.5 Limitations of RT-2

1. **Enormous compute**: 55B parameters, requires TPU pods
2. **Slow inference**: ~1-3 Hz action generation (real robots need 10-20 Hz)
3. **Discrete actions**: 256 bins may not be precise enough
4. **Single-arm**: only tested on one robot morphology
5. **Autoregressive actions**: 7 sequential token predictions per timestep

---

## 3 — OpenVLA: Open-Source VLA (Day 94)

### 3.1 Motivation

RT-2 proved VLAs work, but:
- Closed-source (Google proprietary)
- 55B parameters (impractical for most labs)
- Single robot, single dataset

**OpenVLA** (2024) delivers an open-source, reproducible VLA at 7B scale.

### 3.2 Architecture

```
┌──────────────────────────────────────────────────────┐
│                  OpenVLA Architecture                  │
│                                                        │
│  ┌───────────┐       ┌──────────────────────────┐     │
│  │  Image     │       │  "Pick up the red block" │     │
│  │  224×224   │       │                          │     │
│  └─────┬──────┘       └───────────┬──────────────┘     │
│        │                          │                    │
│        ▼                          │                    │
│  ┌───────────┐                    │                    │
│  │  SigLIP   │                    │                    │
│  │  ViT-SO   │                    │                    │
│  │  400M     │                    │                    │
│  └─────┬──────┘                    │                    │
│        │ visual tokens             │                    │
│        ▼                          ▼                    │
│  ┌──────────────────────────────────────────────┐     │
│  │           Llama 2 7B Decoder                  │     │
│  │                                               │     │
│  │  Concatenation + Self-attention:              │     │
│  │  visual_tokens + text_tokens → action_tokens  │     │
│  │                                               │     │
│  └───────────────────┬──────────────────────────┘     │
│                      │                                │
│                      ▼                                │
│  Action tokens: [a₁, a₂, a₃, a₄, a₅, a₆, a₇]       │
│  De-tokenize via 256-bin lookup                       │
│  → [dx, dy, dz, droll, dpitch, dyaw, gripper]        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

Components:
1. **Vision Encoder**: SigLIP ViT-SO/400M (superior to CLIP for grounding)
2. **Language Model**: Llama 2 7B
3. **Projector**: 2-layer MLP mapping visual tokens into LLM's embedding space
4. **Action Head**: Shared vocabulary, 256 discrete bins per action dimension

### 3.3 Action Tokenization

OpenVLA discretizes each continuous action dimension into 256 bins:

$$
\text{token}_i = \text{vocab\_offset} + \text{round}\left(\frac{a_i - \mu_i}{k \cdot \sigma_i} \cdot 127 + 128\right)
$$

where $\mu_i, \sigma_i$ are per-dimension statistics from the dataset, and
$k$ controls the range (typically $k = 4$ for 4σ coverage).

```python
class ActionTokenizer:
    """OpenVLA-style action tokenization."""
    
    def __init__(self, stats, n_bins=256, vocab_offset=32000):
        self.means = stats['mean']      # per-dimension mean
        self.stds = stats['std']        # per-dimension std
        self.n_bins = n_bins
        self.vocab_offset = vocab_offset
        self.k = 4.0  # number of std devs to cover
    
    def encode(self, actions):
        """Continuous actions → discrete tokens."""
        # Normalize to [-1, 1] range (approximately)
        normalized = (actions - self.means) / (self.k * self.stds)
        # Map to [0, n_bins-1]
        bins = torch.round(normalized * (self.n_bins // 2 - 1) + self.n_bins // 2)
        bins = bins.clamp(0, self.n_bins - 1).long()
        # Offset into vocabulary
        return bins + self.vocab_offset
    
    def decode(self, tokens):
        """Discrete tokens → continuous actions."""
        bins = tokens - self.vocab_offset
        normalized = (bins.float() - self.n_bins // 2) / (self.n_bins // 2 - 1)
        actions = normalized * (self.k * self.stds) + self.means
        return actions
```

### 3.4 Training Data: Open X-Embodiment

OpenVLA is trained on the **Open X-Embodiment (OXE)** dataset — a massive
collection of robot demonstrations across many institutions:

| Dataset Component | Robot | Demos | Tasks |
|------------------|-------|-------|-------|
| Bridge V2 | WidowX | 60k | tabletop manipulation |
| RT-1 Robot Action | Everyday Robots | 130k | mobile manipulation |
| TACO Play | Franka | 3.6k | language-conditioned |
| BC-Z | Sawyer | 26k | 100+ tasks |
| ... and many more | various | 1M+ total | diverse |

### 3.5 Training Recipe

```
Phase 1: VLM Pretraining (done by Prismatic team)
  SigLIP + Llama 2 7B on image-text data
  → Produces Prismatic-7B VLM checkpoint

Phase 2: VLA Fine-tuning
  Prismatic-7B → OpenVLA-7B
  - Data: Open X-Embodiment mixture
  - Batch size: 2048
  - Learning rate: 2e-5
  - Training: ~100k steps on TPU v4-128
  - LoRA rank: 32 (optional for efficiency)
```

### 3.6 Key Results

```
Success rates on WidowX evaluation:

Task                    | RT-1   | RT-2-X | OpenVLA
------------------------|--------|--------|--------
Put carrot on plate     | 56%    | 74%    | 82%
Stack blocks            | 24%    | 42%    | 54%
Wipe table              | 68%    | 72%    | 76%
Novel objects           | 18%    | 48%    | 58%
Language variations     | 22%    | 52%    | 60%
```

> 💡 **Key Insight**: OpenVLA shows that a 7B VLA can match or exceed a 55B
> RT-2 on many tasks, likely because SigLIP provides better visual grounding
> than PaLI-X's encoder, and the OXE dataset is more diverse than RT-1 alone.

---

## 4 — π₀: Flow Matching Meets VLA (Day 95)

### 4.1 The Problem with Discrete Actions

Both RT-2 and OpenVLA discretize actions into bins. This has limitations:

1. **Resolution ceiling**: 256 bins across a 1m workspace = ~4mm resolution
2. **No multimodality**: autoregressive tokens can't easily represent
   multi-modal action distributions (multiple valid ways to grasp)
3. **Sequential prediction**: 7 autoregressive steps add latency
4. **Compounding errors**: error in early dimensions propagates

**π₀** (pronounced "pi-zero", Physical Intelligence, 2024) solves this with
a **flow matching action head** that outputs continuous action chunks.

### 4.2 Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     π₀ Architecture                        │
│                                                            │
│  ┌───────────┐       ┌──────────────────────┐             │
│  │  Image(s)  │       │  Language instruction │             │
│  │  multi-view│       │                      │             │
│  └─────┬──────┘       └──────────┬───────────┘             │
│        │                         │                         │
│        ▼                         ▼                         │
│  ┌───────────┐          ┌──────────────┐                   │
│  │ ViT       │          │ Tokenizer    │                   │
│  │ Encoder   │          │              │                   │
│  └─────┬──────┘          └──────┬───────┘                   │
│        │                        │                          │
│        ▼                        ▼                          │
│  ┌────────────────────────────────────────────┐            │
│  │         VLM Backbone (PaLM-E / Gemma)      │            │
│  │    (processes vision + language jointly)    │            │
│  │              frozen or LoRA                │            │
│  └──────────────────┬─────────────────────────┘            │
│                     │ context embedding                    │
│                     ▼                                      │
│  ┌────────────────────────────────────────────┐            │
│  │        Flow Matching Action Expert          │            │
│  │                                             │            │
│  │  ┌────────┐    ┌──────────┐    ┌────────┐  │            │
│  │  │ Noise  │ →  │ Denoise  │ →  │ Clean  │  │            │
│  │  │ aₜ~N(0,I)   │ Network  │    │ Action │  │            │
│  │  └────────┘    │ v_θ(aₜ,t,c)  │ Chunk  │  │            │
│  │                └──────────┘    └────────┘  │            │
│  │                                             │            │
│  │  Cross-attention with VLM embeddings        │            │
│  │  Outputs: action chunk [a₁, a₂, ..., aH]   │            │
│  └─────────────────────┬──────────────────────┘            │
│                        │                                   │
│                        ▼                                   │
│  Continuous actions: [H × 7-DoF] action chunk              │
│  Execute first K actions, re-plan                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.3 Flow Matching Refresher

Flow matching (from Note 14 on Diffusion Policy) learns a velocity field that
transforms noise into actions:

$$
v_\theta(a_t, t, c) \approx \frac{d}{dt} a_t
$$

where $a_t = (1-t) \cdot \epsilon + t \cdot a_{\text{data}}$ is the interpolation
between noise $\epsilon$ and the target action $a_{\text{data}}$.

The loss:
$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \epsilon, a} \left[ \|v_\theta(a_t, t, c) - (a - \epsilon)\|^2 \right]
$$

At inference, integrate the ODE from $t=0$ (noise) to $t=1$ (action):
$$
a_1 = a_0 + \int_0^1 v_\theta(a_t, t, c) \, dt
$$

### 4.4 The Action Expert

The key innovation in π₀ is the **action expert** — a separate transformer
that specializes in action generation while receiving context from the VLM:

```python
class PiZeroActionExpert(nn.Module):
    """Flow matching action expert with VLM cross-attention."""
    
    def __init__(self, d_model=1024, n_layers=8, n_heads=16,
                 action_dim=7, chunk_size=16):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Noisy action embedding
        self.action_proj = nn.Linear(action_dim * chunk_size, d_model)
        
        # Timestep embedding (sinusoidal)
        self.time_embed = SinusoidalPosEmb(d_model)
        
        # Transformer layers with cross-attention to VLM
        self.layers = nn.ModuleList([
            ActionExpertLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Output projection: predict velocity field
        self.out_proj = nn.Linear(d_model, action_dim * chunk_size)
    
    def forward(self, noisy_actions, timestep, vlm_context):
        """
        noisy_actions: (B, chunk_size * action_dim)
        timestep: (B,) in [0, 1]
        vlm_context: (B, N, d_model) from VLM backbone
        
        Returns: velocity field v_θ, same shape as noisy_actions
        """
        # Embed noisy action chunk
        h = self.action_proj(noisy_actions)
        
        # Add timestep embedding
        t_emb = self.time_embed(timestep)
        h = h + t_emb
        
        # Process through transformer layers
        for layer in self.layers:
            h = layer(
                h,
                context=vlm_context,      # cross-attend to VLM output
                time_emb=t_emb
            )
        
        # Predict velocity field
        velocity = self.out_proj(h)
        return velocity
```

### 4.5 Training π₀

The two-stage training:

**Stage 1: VLM pretraining** (standard)
- Train VLM backbone on web-scale image-text data
- Result: strong visual and language understanding

**Stage 2: Action expert fine-tuning**
- Freeze (or LoRA) the VLM backbone
- Train action expert with flow matching loss
- Cross-attention bridges VLM context → action expert

$$
\mathcal{L} = \mathbb{E}_{t \sim U(0,1)} \left[ \| v_\theta(a_t, t, \text{VLM}(o, l)) - (a - \epsilon) \|^2 \right]
$$

### 4.6 Advantages of π₀

| Property | RT-2 / OpenVLA | π₀ |
|----------|---------------|-----|
| Action space | Discrete (256 bins) | Continuous |
| Multimodality | Single mode | Multi-modal distributions |
| Action output | 7 sequential tokens | Full chunk in parallel |
| Precision | Limited by bin size | Arbitrary precision |
| Latency | 7 autoregressive steps | Single forward pass + ODE |

> 💡 **Key Insight**: π₀ separates what the VLM is good at (understanding
> scenes and language) from what a specialized module is good at (generating
> precise, possibly multi-modal action distributions). The VLM provides
> context; the action expert provides precision.

---

## 5 — Octo & Other Generalist Policies (Day 96)

### 5.1 Octo

**Octo** (2024) is an open-source generalist robot policy from UC Berkeley:

```
┌────────────────────────────────────────────┐
│             Octo Architecture               │
│                                             │
│  Inputs (flexible):                         │
│  ├── Image(s): wrist cam, overhead, ...     │
│  ├── Language: task instruction              │
│  └── Goal image (optional)                  │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │    Transformer Backbone              │   │
│  │    (multi-task, multi-embodiment)    │   │
│  │                                      │   │
│  │    Tokenize all inputs uniformly     │   │
│  │    Self-attention over all tokens    │   │
│  │    Readout tokens for actions        │   │
│  └──────────────────┬───────────────────┘   │
│                     │                       │
│                     ▼                       │
│  ┌──────────────────────────────────────┐   │
│  │    Diffusion Action Head             │   │
│  │    (DDPM-style, predicts action      │   │
│  │     chunk conditioned on backbone    │   │
│  │     embeddings)                      │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Output: [a₁, ..., aH] action chunk        │
│                                             │
└─────────────────────────────────────────────┘
```

Key features:
- **Multi-embodiment**: trained on OXE dataset (800k+ episodes, 20+ robots)
- **Flexible inputs**: handles different camera configurations via tokenization
- **Diffusion head**: outputs continuous action chunks (like Diffusion Policy)
- **93M parameters**: much smaller than RT-2 or OpenVLA
- **Fine-tunable**: designed to be adapted to new robots with small datasets

### 5.2 RT-X and the Open X-Embodiment Project

**RT-X** is not a single model but a collaborative effort:

```
Open X-Embodiment Dataset:
  22 institutions
  22 robot types
  160k+ unique tasks  
  1M+ episodes
  527 skills

Goal: Enable cross-robot transfer learning
```

Key finding from RT-X paper: models trained on **diverse** cross-robot data
often outperform models trained on **only** the target robot's data:

$$
\text{Perf}(\text{OXE mix}) > \text{Perf}(\text{single robot data})
$$

This is the robot equivalent of the "scaling + diversity" lesson from NLP.

### 5.3 Other Notable Models

| Model | Year | Key Innovation |
|-------|------|---------------|
| **GR-1** | 2024 | Video prediction + action, generative world model |
| **SuSIE** | 2024 | Subgoal image generation → low-level policy |
| **LLARVA** | 2024 | LLaVA-based, visual trace actions |
| **RoboFlamingo** | 2024 | Flamingo backbone for robot control |
| **ManipLLM** | 2024 | Affordance-aware VLA |

> 💡 **Key Insight**: The field is converging on a shared recipe:
> pretrained VLM + action head (discrete tokens, diffusion, or flow matching)
> + large-scale diverse robot data. The differences are in the details.

---

## 6 — VLA Training Pipeline (Day 97)

### 6.1 The Standard VLA Training Pipeline

```
Stage 0: Collect Robot Data
  └── Teleoperation, sim, autonomous → (obs, language, action) tuples

Stage 1: VLM Pretraining (usually borrowed)
  └── SigLIP/CLIP + LLM on web-scale image-text
  └── Result: vision-language backbone

Stage 2: VLM → VLA Fine-tuning
  ├── Option A: Full fine-tune (expensive, best performance)
  ├── Option B: LoRA (efficient, nearly as good)
  └── Option C: Freeze VLM, train action head only (fastest)

Stage 3: Task-specific Fine-tuning (optional)
  └── Small dataset of target task, further LoRA
```

### 6.2 Data Mixture Strategy

The data mixture during VLA fine-tuning is critical:

```python
# Typical VLA training data mixture
data_mixture = {
    # Keep VLM capabilities alive
    "image_caption": 0.20,      # maintain vision-language
    "vqa": 0.10,                # maintain reasoning
    
    # Robot data (the new stuff)
    "bridge_v2": 0.25,          # WidowX manipulation
    "rt1_data": 0.20,           # Everyday Robots
    "taco_play": 0.10,          # Franka language-conditioned
    "bc_z": 0.10,               # Sawyer multi-task
    "other_oxe": 0.05,          # smaller datasets
}
```

> ⚠️ **Pitfall**: If you fine-tune on robot data only, the model loses VLM
> capabilities (catastrophic forgetting). Always mix in original VLM data.
> OpenVLA uses roughly 50/50 mix of robot and VLM data.

### 6.3 LoRA for VLA Fine-tuning

Low-Rank Adaptation is the dominant efficient fine-tuning strategy:

$$
W' = W + \Delta W = W + BA
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, and $r \ll d$.

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=32, alpha=64):
        super().__init__()
        self.original = original_layer
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize B to zero → start from pretrained weights
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_normal_(self.lora_A.weight)
    
    def forward(self, x):
        # Original path (frozen) + LoRA path (trainable)
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling
```

Typical LoRA configuration for VLA:
- Rank: 32-64
- Alpha: 64-128
- Applied to: Q, K, V, and output projections in attention layers
- Trainable params: ~2-5% of total

### 6.4 Complete Training Script (Pseudocode)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def train_vla():
    # Load pretrained VLM
    model = AutoModelForCausalLM.from_pretrained("prismatic-vlm-7b")
    tokenizer = AutoTokenizer.from_pretrained("prismatic-vlm-7b")
    
    # Extend vocabulary for action tokens
    num_action_bins = 256
    tokenizer.add_tokens([f"<act_{i}>" for i in range(num_action_bins)])
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            images, instructions, actions = batch
            
            # Tokenize actions
            action_tokens = action_tokenizer.encode(actions)
            
            # Format input
            input_ids = format_vla_input(
                tokenizer, images, instructions
            )
            
            # Forward pass with action token targets
            outputs = model(
                input_ids=input_ids,
                labels=action_tokens,
                images=images
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

---

## 7 — Action Tokenization Deep Dive (Day 98)

### 7.1 The Central Design Choice

How you represent actions determines everything about your VLA's capabilities:

```
                    Action Representation Spectrum
                    
  Discrete Bins          Autoregressive         Flow/Diffusion
  (RT-2, OpenVLA)        Chunks                  (π₀, Octo)
       │                    │                        │
       ▼                    ▼                        ▼
  Simple, uses LLM      Ordered sequence        Continuous, can be
  vocabulary natively    of action tokens         multi-modal
  
  ◀─── Simplicity ──────────────────────── Expressiveness ───▶
```

### 7.2 Method 1: Uniform Discrete Bins

Used by RT-2, OpenVLA.

$$
\text{bin}(a) = \left\lfloor \frac{a - a_{\min}}{a_{\max} - a_{\min}} \times (N - 1) + 0.5 \right\rfloor
$$

```
Action dimension range: [-0.5m, 0.5m]
N_bins = 256
Resolution = 1.0m / 255 ≈ 3.9mm per bin

Example:
  Continuous: a = 0.123m
  Bin index:  floor((0.123 + 0.5) / 1.0 * 255) = 158
  Recovered:  158 / 255 * 1.0 - 0.5 = 0.120m
  Error:      0.003m = 3mm
```

**Pros**: Simple, uses LLM's autoregressive machinery directly
**Cons**: Fixed resolution, can't represent multimodal distributions

### 7.3 Method 2: Gaussian Mixture Bins (Learned)

Instead of uniform bins, learn a non-uniform binning that places more bins
where actions are more likely:

$$
p(a) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(a \mid \mu_k, \sigma_k^2)
$$

Fit a GMM to the action distribution, then assign bins proportional to density.

### 7.4 Method 3: Continuous Diffusion/Flow Matching

Used by π₀, Octo, Diffusion Policy.

```python
class DiffusionActionHead(nn.Module):
    """DDPM-style action generation."""
    
    def __init__(self, context_dim, action_dim, chunk_size, 
                 n_steps=100):
        super().__init__()
        self.n_steps = n_steps
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Noise prediction network
        self.noise_pred = TransformerDenoiser(
            input_dim=action_dim * chunk_size,
            context_dim=context_dim,
            n_layers=4,
            n_heads=8
        )
        
        # Noise schedule
        self.betas = cosine_beta_schedule(n_steps)
        self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)
    
    def training_loss(self, actions, context):
        """Standard DDPM training loss."""
        B = actions.shape[0]
        flat_actions = actions.reshape(B, -1)
        
        # Sample random timestep
        t = torch.randint(0, self.n_steps, (B,))
        
        # Add noise
        noise = torch.randn_like(flat_actions)
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
        noisy = torch.sqrt(alpha_t) * flat_actions + \
                torch.sqrt(1 - alpha_t) * noise
        
        # Predict noise
        pred_noise = self.noise_pred(noisy, t, context)
        
        return F.mse_loss(pred_noise, noise)
    
    def sample(self, context, n_steps=20):
        """DDIM-style fast sampling."""
        B = context.shape[0]
        action_size = self.action_dim * self.chunk_size
        
        # Start from noise
        x = torch.randn(B, action_size, device=context.device)
        
        # Denoise
        for t in reversed(range(0, self.n_steps, self.n_steps // n_steps)):
            t_tensor = torch.full((B,), t, device=context.device)
            pred_noise = self.noise_pred(x, t_tensor, context)
            x = ddim_step(x, pred_noise, t, self.alphas_cumprod)
        
        return x.reshape(B, self.chunk_size, self.action_dim)
```

### 7.5 Method 4: Autoregressive Chunked Tokens

A hybrid approach: predict a *sequence* of action chunks autoregressively,
where each chunk contains multiple timesteps.

```
Standard autoregressive (RT-2):
  t=0: predict a₁ᵡ, a₁ʸ, a₁ᶻ, a₁ʳ, a₁ᵖ, a₁ʷ, a₁ᵍ  (7 tokens)

Chunked autoregressive:
  t=0: predict [a₁...a₄] as one token, [a₅...a₈] as next token, ...
  
  Each "token" encodes a mini-sequence via VQ-VAE:
  [a₁, a₂, a₃, a₄] → VQ encode → single token index
```

### 7.6 Comprehensive Comparison

| Dimension | Discrete Bins | Diffusion/Flow | AR Chunked |
|-----------|:------------:|:--------------:|:----------:|
| **Resolution** | Fixed (256 bins) | Arbitrary | VQ-dependent |
| **Multimodality** | ✗ single mode | ✓ multi-modal | Partial |
| **Inference speed** | 7 AR steps | 10-50 denoise steps | Fewer AR steps |
| **Implementation** | Simple (reuse LLM) | Moderate (new head) | Complex (VQ-VAE) |
| **Precision tasks** | Limited | Excellent | Good |
| **LLM compatibility** | Native | Requires adapter | Requires VQ codebook |
| **Training stability** | Stable | Can be tricky | Moderate |

> 💡 **Key Insight**: The choice of action representation is the single most
> important architectural decision in VLA design. For "good enough" manipulation,
> 256 discrete bins work fine. For dexterous or precision tasks, you need
> continuous outputs via diffusion or flow matching.

---

## 8 — Scaling & Emergence in VLAs (Day 99)

### 8.1 Scaling Dimensions

VLAs can be scaled along multiple axes:

```
                        VLA Scaling Axes
                        
  ┌─────────────────────────────────────────────┐
  │                                             │
  │  Model Size           Data Size             │
  │  7B → 55B → ?        100k → 1M → 10M      │
  │                                             │
  │  Task Diversity       Robot Diversity        │
  │  10 → 100 → 1000+    1 → 22 → 100+        │
  │                                             │
  │  Modality Count       Compute               │
  │  1cam → multi-cam     TPU v4 → v5 → ?      │
  │  + proprio + tactile                        │
  │                                             │
  └─────────────────────────────────────────────┘
```

### 8.2 Evidence for Scaling Laws

From RT-2 and subsequent work:

$$
\text{Performance} \propto \log(\text{model\_size}) + \log(\text{data\_size})
$$

Empirical observations:
- RT-2 (55B) > RT-2 (12B) by 5-15% across benchmarks
- OXE mix (1M episodes) > single-robot data by 10-30%
- More task diversity → better zero-shot to novel tasks

```
Scaling curve (approximate, from multiple papers):

Performance
   100% ┤
    90% ┤                                    ●───────
    80% ┤                          ●────────╱
    70% ┤                   ●────╱
    60% ┤            ●────╱
    50% ┤      ●───╱
    40% ┤   ●╱
    30% ┤  ╱
    20% ┤╱
        └─┬────┬────┬────┬────┬────┬────┬──
         1B   3B   7B   12B  30B  55B  100B+
                    Model Parameters
```

### 8.3 Language-Conditioned Behavior

Perhaps the most exciting emergent property: VLAs develop **compositional
understanding** of language instructions:

```
Training data includes:
  "pick up the red block"
  "put the blue cup on the plate"
  
At test time, novel compositions work:
  "pick up the blue block"  ← recombined color + object
  "put the red block on the cup"  ← recombined object + location
  
Even more abstract:
  "pick up the object that doesn't belong"  ← requires reasoning
  "stack them in order of size"  ← requires comparison
```

This is the language model's compositional generalization transferring to
robot behavior — a direct benefit of the VLM backbone.

### 8.4 Cross-Embodiment Transfer

When trained on data from multiple robots, VLAs show transfer:

$$
\text{Robot A data} + \text{Robot B data} \rightarrow \text{Better on Robot C}
$$

This suggests VLAs learn **abstract manipulation concepts** (reach, grasp,
place) that transfer across specific kinematic configurations.

```python
# Embodiment-aware VLA (conceptual)
class EmbodimentAwareVLA(nn.Module):
    def __init__(self, vlm_backbone, action_dims_per_robot):
        super().__init__()
        self.vlm = vlm_backbone
        
        # Shared understanding of manipulation
        self.shared_encoder = TransformerEncoder(...)
        
        # Robot-specific action decoders
        self.action_heads = nn.ModuleDict({
            robot: ActionHead(dim)
            for robot, dim in action_dims_per_robot.items()
        })
        
        # Robot identity token
        self.robot_tokens = nn.Embedding(
            len(action_dims_per_robot), d_model
        )
    
    def forward(self, image, instruction, robot_id):
        # VLM processes scene + language (shared)
        context = self.vlm(image, instruction)
        
        # Add robot identity
        robot_emb = self.robot_tokens(robot_id)
        context = context + robot_emb
        
        # Robot-specific action decoding
        actions = self.action_heads[robot_id](context)
        return actions
```

> ⚠️ **Pitfall**: Cross-embodiment transfer is promising but not solved.
> Morphologically very different robots (humanoid vs. wheeled) show limited
> transfer. The benefit is largest among robots with similar kinematics.

---

## 9 — Architecture Comparison (Day 100)

### 9.1 The Landscape

```
                    VLA Architecture Map (2024)
                    
  Large (55B+)   │  RT-2
                  │  PaLM-E Robot
                  │
  Medium (7-12B)  │  OpenVLA        π₀
                  │  RoboFlamingo
                  │
  Small (<1B)     │  Octo (93M)
                  │  
                  └──────────────────────────────
                     Discrete        Continuous
                     Actions         Actions
```

### 9.2 Comprehensive Comparison Table

| Dimension | **RT-2** | **OpenVLA** | **π₀** | **Octo** |
|-----------|----------|-------------|--------|----------|
| **Year** | 2023 | 2024 | 2024 | 2024 |
| **Lab** | Google DeepMind | Stanford/Berkeley | Physical Intelligence | Berkeley |
| **Vision Encoder** | ViT-22B (PaLI-X) | SigLIP ViT-SO 400M | ViT (undisclosed) | ViT-B (86M) |
| **LLM Backbone** | PaLM/UL2 (32B) | Llama 2 7B | Gemma-based | Custom (93M total) |
| **Total Params** | 55B | 7.6B | ~3B (est.) | 93M |
| **Action Repr.** | 256 discrete bins | 256 discrete bins | Flow matching (continuous) | Diffusion (continuous) |
| **Action Output** | 7 AR tokens | 7 AR tokens | Action chunk (parallel) | Action chunk (parallel) |
| **Training Data** | RT-1 (130k) | OXE (970k+) | Proprietary (~1M) | OXE (800k+) |
| **Robots Trained** | 1 (Everyday Robot) | 7+ robots | 7+ robots | 20+ robots |
| **Open Source** | ✗ | ✓ | ✗ | ✓ |
| **Open Weights** | ✗ | ✓ | ✗ | ✓ |
| **Inference Speed** | ~1-3 Hz | ~5-8 Hz | ~10-15 Hz | ~15-20 Hz |
| **Precision** | ~4mm (bin limited) | ~4mm (bin limited) | Sub-mm (continuous) | Sub-mm (continuous) |
| **Multimodal Actions** | ✗ | ✗ | ✓ | ✓ |
| **Key Strength** | Semantic reasoning | Reproducibility | Precision + speed | Efficiency + flexibility |
| **Key Weakness** | Compute cost | Bin resolution | Closed source | Smaller backbone |

### 9.3 When to Use What

```
Decision Tree for VLA Architecture Selection:

  Do you need semantic generalization?
  ├── YES → Need large VLM backbone
  │         ├── Open source required? → OpenVLA
  │         └── Max performance? → RT-2 / π₀
  └── NO → Octo (smaller, faster)

  Do you need precision < 1mm?
  ├── YES → Need continuous actions → π₀ or Octo
  └── NO → Any VLA works

  Do you need multi-modal actions (multiple valid grasps)?
  ├── YES → Diffusion/flow matching → π₀ or Octo
  └── NO → Discrete bins are fine → OpenVLA

  Edge deployment (limited compute)?
  ├── YES → Octo (93M) or quantized OpenVLA
  └── NO → Any VLA works
```

### 9.4 Cost-Performance Analysis

```
                    Cost vs Performance

  Performance
       │
  100% ┤                                    ● RT-2 (55B)
       │                              ● π₀
   80% ┤                     ● OpenVLA
       │
   60% ┤          ● Octo
       │
   40% ┤
       │
       └──┬────────┬────────┬────────┬────
        $100    $1,000   $10,000  $100,000
                   Training Cost (GPU hours)
```

### 9.5 The Architecture Zoo (Extended)

For reference, additional models in the VLA ecosystem:

| Model | Params | Action Type | Notable Feature |
|-------|--------|-------------|----------------|
| GR-1 | 1.6B | Continuous | Video prediction + action |
| SuSIE | 7B VLM + small policy | Subgoal images | Two-stage: imagine → act |
| LLARVA | 7B | Visual traces | 2D trajectory as action |
| RoboFlamingo | 9B | Discrete | Flamingo cross-attention |
| ManipLLM | 7B | Contact points | Affordance-based |
| HPT | 1B | Continuous | Heterogeneous pre-training |
| CrossFormer | 130M | Continuous | Cross-embodiment |

---

## 10 — Equations, Diagrams, and References

### 10.1 Core Equations Summary

**VLA as conditional language model (RT-2 / OpenVLA):**
$$
p(a_{1:T} \mid o, l) = \prod_{t=1}^{T} \prod_{d=1}^{D} p_\theta(a_t^d \mid a_t^{<d}, a_{<t}, o, l)
$$

**VLA with flow matching action head (π₀):**
$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t,\epsilon} \left[ \left\| v_\theta\!\left((1-t)\epsilon + t \cdot a, \; t, \; f_\text{VLM}(o, l)\right) - (a - \epsilon) \right\|^2 \right]
$$

**VLA with diffusion action head (Octo):**
$$
\mathcal{L}_\text{diff} = \mathbb{E}_{t,\epsilon} \left[ \left\| \epsilon_\theta\!\left(\sqrt{\bar\alpha_t} \, a + \sqrt{1-\bar\alpha_t} \, \epsilon, \; t, \; c\right) - \epsilon \right\|^2 \right]
$$

**Action tokenization (discretization):**
$$
\text{tok}(a^d) = \text{clip}\!\left(\left\lfloor \frac{a^d - \mu_d}{k \sigma_d} \cdot \frac{N}{2} + \frac{N}{2} \right\rfloor, \; 0, \; N-1\right)
$$

**LoRA update for efficient VLA fine-tuning:**
$$
h = W_0 x + \frac{\alpha}{r} B A x, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}
$$

### 10.2 Full VLA Forward Pass (Unified View)

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified VLA Forward Pass                     │
│                                                              │
│  Step 1: Encode observation                                  │
│    v = ViT(image)                    # visual tokens         │
│                                                              │
│  Step 2: Encode language                                     │
│    l = Tokenizer(instruction)        # text tokens           │
│                                                              │
│  Step 3: Joint reasoning (VLM backbone)                      │
│    c = Transformer([v; l])           # context embedding     │
│                                                              │
│  Step 4: Generate actions                                    │
│    ┌─────────────────────────────────────────────────┐       │
│    │ Option A: Discrete tokens (RT-2, OpenVLA)       │       │
│    │   a = argmax p(token | c), for each dimension   │       │
│    ├─────────────────────────────────────────────────┤       │
│    │ Option B: Diffusion (Octo)                      │       │
│    │   a = denoise(randn(), c, T_steps)              │       │
│    ├─────────────────────────────────────────────────┤       │
│    │ Option C: Flow matching (π₀)                    │       │
│    │   a = ODE_solve(v_θ, randn(), c)                │       │
│    └─────────────────────────────────────────────────┘       │
│                                                              │
│  Step 5: Execute                                             │
│    robot.step(a)                                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 Key Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **RT-1**: Robotics Transformer (Brohan et al.) | 2022 | First large-scale robot transformer |
| **PaLM-E**: An Embodied Multimodal Language Model (Driess et al.) | 2023 | VLM for embodied reasoning |
| **RT-2**: Vision-Language-Action Models (Brohan et al.) | 2023 | VLM fine-tuned for robot actions |
| **Octo**: An Open-Source Generalist Robot Policy (Team Octo) | 2024 | Open-source, multi-robot, diffusion head |
| **OpenVLA**: An Open-Source VLA (Kim et al.) | 2024 | Open-source 7B VLA |
| **π₀**: A Vision-Language-Action Flow Model (Black et al.) | 2024 | Flow matching action head |
| **Open X-Embodiment** (RT-X Collaboration) | 2024 | Cross-robot dataset & transfer |
| **Diffusion Policy** (Chi et al.) | 2023 | Diffusion for robot action generation |
| **ALOHA / ACT** (Zhao et al.) | 2023 | Action Chunking Transformer |
| **GR-1** (Wu et al.) | 2024 | Video generation + actions |

### 10.4 Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $o_t$ | Observation (image) at time $t$ |
| $l$ | Language instruction |
| $a_t$ | Action at time $t$ |
| $\pi_\theta$ | Policy parameterized by $\theta$ |
| $v_\theta$ | Velocity field (flow matching) |
| $\epsilon_\theta$ | Noise predictor (diffusion) |
| $c$ | Context embedding from VLM backbone |
| $H$ | Action chunk horizon |
| $D$ | Action dimensionality (e.g., 7 for a robot arm) |
| $N$ | Number of discretization bins |

---

## 11 — Connection to Thread

```
The Full Compression Journey:

  Note 1:  Backprop          → learn ANY differentiable function
  Note 2:  Optimizers        → learn it EFFICIENTLY
  Note 3:  Regularization    → learn it ROBUSTLY
  Note 4:  CNNs              → compress SPATIAL patterns
  Note 5:  RNNs/Attention    → compress TEMPORAL patterns
  Note 6:  Transformers      → compress EVERYTHING with attention
  Note 7:  LLM Pretraining   → compress ALL OF LANGUAGE
  Note 8:  Fine-tuning       → compress to YOUR task
  Note 9:  RLHF              → compress HUMAN preferences
  Note 10: VLMs              → compress VISION + LANGUAGE
  Note 11: Reinforcement     → compress REWARD-seeking behavior
  Note 12: Robot Learning    → compress PHYSICAL interaction
  Note 13: Imitation         → compress DEMONSTRATED behavior
  Note 14: Diffusion Policy  → compress MULTI-MODAL action distributions
  
  ★ Note 15: VLAs            → compress VISION + LANGUAGE + ACTIONS
                                into ONE model that can SEE, UNDERSTAND,
                                and ACT.
```

VLAs are the **grand unification** of this curriculum. They show that the
compression hypothesis applies across all modalities simultaneously: a single
transformer, trained on internet text, images, and robot demonstrations, can
develop an integrated understanding of the physical world that enables it to
follow natural language instructions to manipulate objects.

The key equation that captures everything:

$$
\boxed{\pi_\theta(a \mid o, l) = \text{Transformer}(\text{compress}(o) \oplus \text{compress}(l)) \rightarrow \text{decompress}(a)}
$$

One model. Three modalities. Compression all the way down.

---

## Exercises

### Day 92: VLA Concept
1. Explain why representing actions as tokens in a language model is a natural extension of VLMs
2. List three advantages of end-to-end VLAs over pipeline approaches
3. Draw the evolution: CNN → ViT → CLIP → VLM → VLA

### Day 93: RT-2
4. Implement a simple action tokenizer that converts 7-DoF continuous actions to 256-bin discrete tokens
5. Explain how RT-2 achieves "semantic generalization" with no explicit symbolic reasoning
6. Calculate the spatial resolution of 256 bins over a 1-meter workspace

### Day 94: OpenVLA
7. Compare SigLIP vs CLIP as vision encoders for VLAs — why might SigLIP be better?
8. Implement a LoRA wrapper for a linear layer
9. Explain why the 50/50 VLM/robot data mix is important during fine-tuning

### Day 95: π₀
10. Implement the flow matching loss for an action expert
11. Explain why continuous action outputs are superior for precision tasks
12. Draw the π₀ architecture showing VLM → cross-attention → action expert

### Day 96: Octo & Others
13. Compare Octo's diffusion head with π₀'s flow matching head
14. Explain why cross-embodiment data helps even for a single target robot
15. Design a flexible input tokenization scheme for variable camera configurations

### Day 97: Training Pipeline
16. Write pseudocode for the full VLA training pipeline (data → model → eval)
17. Design a data mixture strategy for a new VLA targeting kitchen tasks
18. Calculate the number of trainable LoRA parameters for a 7B model with rank 32

### Day 98: Action Tokenization
19. Implement all three action representations (discrete, diffusion, flow) for 7-DoF actions
20. Create a comparison benchmark: measure precision and inference speed for each
21. Propose a hybrid: discrete bins for coarse + continuous refinement for fine

### Day 99: Scaling
22. Sketch the expected scaling curve for VLA performance vs model size
23. Design an experiment to test compositional language generalization in a VLA
24. Propose a curriculum learning schedule for training a VLA on 1M+ episodes

### Day 100: Synthesis
25. Build a comparison table of all VLA architectures on 10 dimensions
26. For each of the 5 OKS subsystems (nav, sensorbar, lifter, BEC, guardian),
    propose whether a VLA or classical approach is better and why
27. Design a "VLA v2" architecture that combines the best of RT-2, OpenVLA, and π₀

---

## Self-Check Questions

- [ ] Can you explain why actions can be treated as "just another token type"?
- [ ] Can you draw the RT-2 architecture from memory?
- [ ] Can you implement action tokenization (continuous → discrete bins)?
- [ ] Do you understand the flow matching loss used in π₀?
- [ ] Can you explain why VLMs transfer knowledge to robot control?
- [ ] Can you compare discrete vs continuous action representations?
- [ ] Do you understand LoRA and why it's used for VLA fine-tuning?
- [ ] Can you explain cross-embodiment transfer and its limitations?
- [ ] Can you choose the right VLA architecture for a given problem?

---

*Next: [16 — Deployment, Hybrid Systems & The Road Ahead](16-deployment-hybrid.md)*
