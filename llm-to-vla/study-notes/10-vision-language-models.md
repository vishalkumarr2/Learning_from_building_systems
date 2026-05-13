# 10 — Vision-Language Models

> **Phase V · Days 59–70 · ~30 hours**
> Prerequisites: [05-gpt-scaling](05-gpt-scaling.md), [08-vision-transformers](08-vision-transformers.md)
> Learning Objectives: Understand how vision and language unify, CLIP, multimodal architectures, why this is the foundation for VLAs

---

## The Big Picture

We've built two towers:
- **Language**: tokenize text → transformer → predict next token (Phases I–III)
- **Vision**: patchify images → ViT → classify/segment (Phase IV)

Phase V fuses them. The central question: **can a single model understand both images and text?** The answer is yes — and the mechanism is surprisingly simple. Both modalities compress into the same transformer bottleneck. The model doesn't "see" and "read" differently; it processes token sequences regardless of origin.

This unification is not just academically interesting. It's the **direct precursor to VLAs**: if vision and language share a representation, adding action tokens is a natural extension.

```
Phase V Progression:

Day 59: CLIP — contrastive alignment of vision + language
Day 60: LLaVA — visual instruction tuning
Day 61: Architecture deep dive — design space exploration
Day 62: Flamingo — cross-attention for multimodal
Day 63: VLM training pipeline — stages and data
Day 64: Grounding — from understanding to localization
Day 65: VLM applications for robotics
Day 66: 🛑 STOP AND REFLECT #4
Day 67-68: Advanced VLMs — frontier models
Day 69-70: Phase V Capstone — build a scene analyzer
```

---

## 1. CLIP — Contrastive Vision-Language (Day 59)

### 1.1 The Contrastive Pretraining Idea

Before CLIP, vision models were trained on fixed label sets (ImageNet's 1000 classes). This created a fundamental limitation: the model could only recognize categories it was trained on.

CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) broke this paradigm by training on **400 million image-text pairs** from the internet. Instead of classifying images into fixed categories, CLIP learns to **align images and text in a shared embedding space**.

### 1.2 Architecture: Two Encoders

```
Image Encoder (ViT-L/14 or ResNet)        Text Encoder (Transformer)
         │                                          │
    image embedding                            text embedding
    z_I ∈ ℝ^d                                 z_T ∈ ℝ^d
         │                                          │
         └──────── cosine similarity ───────────────┘
                   sim(I, T) = z_I · z_T / (‖z_I‖ ‖z_T‖)
```

Both encoders map their inputs to the **same d-dimensional space**. The training objective ensures that matching image-text pairs have high cosine similarity, while non-matching pairs have low similarity.

### 1.3 Contrastive Loss (InfoNCE)

Given a batch of $N$ image-text pairs $(I_i, T_i)$, the loss for a single image $I_i$ is:

$$\mathcal{L}_{\text{image}}^{(i)} = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

And symmetrically for text:

$$\mathcal{L}_{\text{text}}^{(i)} = -\log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)}$$

Total loss:

$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \left[ \mathcal{L}_{\text{image}}^{(i)} + \mathcal{L}_{\text{text}}^{(i)} \right]$$

where $\tau$ is a **learned temperature parameter** that controls the sharpness of the distribution.

**Intuition**: In a batch of $N$ pairs, each image must pick its matching text from $N$ candidates (and vice versa). This is an $N$-way classification problem where the batch itself creates the negative examples.

### 1.4 Why Large Batches Matter

CLIP uses batch sizes of **32,768**. This is critical because:
- More negatives = harder contrastive task = better representations
- With batch size $N$, each sample has $N-1$ negatives
- The effective "label space" is the entire batch

### 1.5 Zero-Shot Transfer

CLIP's breakthrough: **classify any image without training on that specific task**.

```
Step 1: Create text prompts for each class
   "a photo of a dog"
   "a photo of a cat"
   "a photo of a bird"

Step 2: Encode all prompts with text encoder
   T_dog, T_cat, T_bird ∈ ℝ^d

Step 3: Encode the test image
   z_I ∈ ℝ^d

Step 4: Pick the highest similarity
   prediction = argmax_c sim(z_I, T_c)
```

This works because CLIP learned a rich concept space from 400M image-text pairs. The text encoder encodes **semantic meaning**, not just word identity.

### 1.6 Why CLIP Representations Are Powerful

CLIP embeddings capture:
1. **Semantic content**: what objects are present
2. **Style and domain**: "a painting of" vs "a photo of"
3. **Spatial relationships**: some spatial reasoning emerges
4. **Abstract concepts**: "danger", "beautiful", "expensive"

These properties make CLIP the de facto vision backbone for many downstream tasks.

### 1.7 SigLIP — Sigmoid Loss for Language-Image Pretraining

SigLIP (Zhai et al. 2023) replaces CLIP's softmax-based loss with a **sigmoid loss** applied to each image-text pair independently:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma\!\bigl(y_{ij} \cdot (\mathbf{z}_i^{I\top} \mathbf{z}_j^T \cdot e^t + b)\bigr)$$

where $y_{ij} = +1$ if $i = j$ (matching pair) and $y_{ij} = -1$ otherwise. Here $\mathbf{z}_i^{I\top} \mathbf{z}_j^T$ is the dot-product similarity between L2-normalised image and text embeddings, $t$ is a learnable log-temperature, and $b$ is a learnable bias.

**Key advantage**: No need for all-to-all similarity computation across devices. Each pair is classified independently (match or not), enabling:
- Better scaling to very large batches
- Simpler distributed training
- Slightly better performance in practice

SigLIP is now the preferred vision encoder for many VLMs (PaLI, LLaVA-NeXT).

### 1.8 Limitations of CLIP

- **No generative capability**: CLIP can match and classify, but can't generate descriptions
- **Weak at counting**: "two dogs" vs "three dogs" often indistinguishable
- **Limited compositionality**: struggles with attribute binding ("red car and blue truck" vs "blue car and red truck")
- **Bag-of-words tendency**: text encoder treats "dog bites man" ≈ "man bites dog"

These limitations motivate generative VLMs (next section).

---

## 2. Visual Instruction Tuning — LLaVA (Day 60)

### 2.1 The Key Insight

CLIP aligns vision and language but can't hold conversations about images. What if we **project visual features into an LLM's input space** and fine-tune the LLM to understand them?

This is the LLaVA (Visual Instruction Tuning, Liu et al. 2023) approach.

### 2.2 LLaVA Architecture

```
┌──────────────┐
│  Input Image  │
└──────┬───────┘
       │
┌──────▼───────┐
│  CLIP ViT-L   │  (frozen vision encoder)
│  Image Encoder │
└──────┬───────┘
       │  visual features: [v₁, v₂, ..., v_M]  (M=256 patch tokens)
       │
┌──────▼───────┐
│  Projection   │  W · v_i → h_i  (linear or MLP)
│  Layer         │
└──────┬───────┘
       │  visual tokens: [h₁, h₂, ..., h_M]
       │
       ▼
┌─────────────────────────────────────────┐
│            LLM (e.g., Vicuna-13B)        │
│                                           │
│  [SYS] [h₁ h₂ ... h_M] [USER] question  │
│  → generates answer autoregressively      │
└───────────────────────────────────────────┘
```

The visual tokens are treated **identically to text tokens** by the LLM's attention mechanism. The LLM doesn't know (or care) that some tokens came from an image.

### 2.3 Two-Stage Training

**Stage 1: Feature Alignment (Pretraining)**
- Freeze: vision encoder + LLM
- Train: only the projection layer
- Data: 558K image-caption pairs (CC3M subset)
- Objective: standard next-token prediction
- Purpose: teach the projection to map visual features into the LLM's "language" of token embeddings

**Stage 2: Visual Instruction Tuning**
- Freeze: vision encoder
- Train: projection layer + LLM (LoRA or full fine-tune)
- Data: 158K multimodal instruction-following samples (LLaVA-Instruct)
- Objective: standard next-token prediction on instruction-response pairs
- Purpose: teach the model to follow complex visual instructions

### 2.4 Data Generation via GPT-4

A key innovation: LLaVA's instruction tuning data was generated by:
1. Feeding image captions + bounding boxes to GPT-4 (text-only)
2. Asking GPT-4 to generate visual question-answer pairs
3. Creating three types: conversation, detailed description, complex reasoning

This "language-only teacher generates multimodal data" trick enabled high-quality training data at scale.

### 2.5 Interleaving Visual and Text Tokens

The input sequence to the LLM looks like:

```
[BOS] <system_prompt> [IMG₁ IMG₂ ... IMG_M] [USER] What is in this image? [ASSISTANT]
```

The LLM's self-attention operates over ALL tokens uniformly:
- Text tokens attend to visual tokens (and vice versa)
- The model learns cross-modal relationships through standard attention
- No special cross-attention mechanism needed

### 2.6 Why This Works

The LLM already has:
- Rich world knowledge from text pretraining
- Instruction-following ability
- Reasoning capabilities

By projecting visual features into the LLM's embedding space, we're giving the LLM "eyes" without retraining its core capabilities.

---

## 3. Multimodal Architecture Deep Dive (Day 61)

### 3.1 The Three Components

Every VLM has three key components:

```
┌─────────────────┐    ┌────────────┐    ┌──────────────┐
│  Vision Encoder  │───▶│  Connector  │───▶│  LLM Backbone │
└─────────────────┘    └────────────┘    └──────────────┘
```

The design space is the combinatorial product of choices for each.

### 3.2 Vision Encoder Choices

| Encoder | Pretraining | Strengths | Used In |
|---------|-------------|-----------|---------|
| CLIP ViT-L/14 | Contrastive (image-text) | Semantic alignment | LLaVA, OpenFlamingo |
| SigLIP | Sigmoid contrastive | Better scaling | PaLI, LLaVA-NeXT |
| DINOv2 | Self-supervised (image-only) | Fine-grained spatial | Some research models |
| EVA-CLIP | Enhanced CLIP training | Better features | InternVL |
| ConvNeXt | Supervised ImageNet | Efficiency | Some hybrid models |

**Key tradeoff**: CLIP/SigLIP encoders have semantic alignment with text (good for VLMs), but DINOv2 has better spatial features (good for grounding/robotics).

**Dual encoder strategy**: Some models use **both** — CLIP for semantics + DINOv2 for spatial.

### 3.3 Connector Types

The connector bridges the vision encoder's output space to the LLM's input space.

**Linear Projection** (simplest):
$$h_i = W \cdot v_i + b, \quad W \in \mathbb{R}^{d_\text{LLM} \times d_\text{vision}}$$

Used in: LLaVA v1. Simple but effective.

**MLP Projection** (most common):
$$h_i = W_2 \cdot \text{GELU}(W_1 \cdot v_i + b_1) + b_2$$

Used in: LLaVA v1.5+. 2-layer MLP is the sweet spot.

**Q-Former** (BLIP-2):
```
Learnable queries Q ∈ ℝ^{K×d}
       │
  Cross-attend to vision features V
       │
  K output tokens (K << M, typically K=32)
```

Compresses $M$ visual tokens to $K$ query tokens. Reduces computation in the LLM but may lose spatial detail.

**Perceiver Resampler** (Flamingo):
Similar to Q-Former — learned latent tokens cross-attend to visual features. More flexible architecture.

**C-Abstractor** (Honeybee):
Convolutional abstractor that preserves spatial relationships while reducing token count.

### 3.4 Token Count Tradeoff

| Approach | Visual Tokens | Pros | Cons |
|----------|--------------|------|------|
| Full patches | 256-576 | Maximum information | Expensive LLM inference |
| Q-Former | 32-64 | Efficient | May lose spatial detail |
| Downsampled | 64-144 | Good balance | Some information loss |
| Dynamic | varies | Adaptive | Complex implementation |

For robotics, spatial detail matters → lean toward more tokens.

### 3.5 LLM Backbone Choices

| Model | Size | Strength | Used In |
|-------|------|----------|---------|
| Vicuna/LLaMA | 7B/13B | Good instruction following | LLaVA |
| Mistral | 7B | Efficient, strong reasoning | LLaVA-NeXT |
| Qwen-2 | 7B/72B | Multilingual, strong | Qwen-VL |
| InternLM2 | 7B/20B | Strong reasoning | InternVL |
| Phi-3 | 3.8B | Small but capable | Phi-3-Vision |
| Gemma | 2B/7B | Google's compact LLM | PaliGemma |

### 3.6 Where to Invest Parameters?

Research insight: **most of the capability comes from the LLM backbone**. The vision encoder and connector contribute, but a bigger/better LLM improves VLM performance more than a bigger vision encoder.

Priority ranking for parameter allocation:
1. LLM backbone quality (most important)
2. Vision encoder quality
3. Training data quality and diversity
4. Connector architecture (least important — linear vs MLP matters little)

---

## 4. Flamingo & Cross-Attention (Day 62)

### 4.1 Flamingo's Innovation

DeepMind's Flamingo (Alayrac et al. 2022) introduced a different approach to combining vision and language: **gated cross-attention layers** inserted into a frozen LLM.

### 4.2 Architecture

```
                    Frozen LLM Layer N
                          │
              ┌───────────▼──────────┐
              │  Gated Cross-Attention │◄── visual features
              │  (new, trainable)      │
              └───────────┬──────────┘
                          │
                    Frozen LLM Layer N+1
```

Cross-attention layers are inserted between existing LLM layers. Only the cross-attention parameters are trained; the LLM remains frozen.

### 4.3 Gated Cross-Attention

The gate ensures smooth integration:

$$\text{output} = \text{LLM\_output} + \tanh(\alpha) \cdot \text{CrossAttn}(\text{LLM\_output}, \text{visual\_features})$$

where $\alpha$ is initialized to **zero**, so at the start of training, the cross-attention contributes nothing. The model gradually learns to incorporate visual information.

### 4.4 Perceiver Resampler

Before cross-attention, Flamingo compresses visual features:

```
Input: variable-length visual features from N images
  ↓
Perceiver Resampler: learned latent queries cross-attend to visual features
  ↓
Output: fixed-length (64 tokens per image) visual representation
```

This handles variable numbers of images gracefully.

### 4.5 Few-Shot Visual Learning

Flamingo's architecture enables **interleaved image-text prompting**:

```
Image₁: "This is a German Shepherd"
Image₂: "This is a Golden Retriever"  
Image₃: "This is a ___"
→ Model: "Labrador Retriever"
```

The cross-attention mechanism naturally handles multiple images in sequence, enabling few-shot visual learning without any fine-tuning.

### 4.6 Cross-Attention vs Concatenation

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| Concatenation (LLaVA) | Visual tokens prepended to text | Simple, no new parameters | Fixed overhead, all layers process visual tokens |
| Cross-Attention (Flamingo) | Visual info injected via cross-attn | Flexible, LLM stays frozen | New parameters, more complex |

**For VLAs**: Cross-attention is interesting because action tokens could similarly be injected without modifying the core LLM.

---

## 5. VLM Training Pipeline (Day 63)

### 5.1 The Standard Recipe

Modern VLM training follows a consistent pipeline:

```
Stage 0: Component Pretraining (done by others)
  - Vision encoder: CLIP/SigLIP training
  - LLM: standard language pretraining + RLHF

Stage 1: Feature Alignment
  - Freeze: vision encoder + LLM
  - Train: connector only
  - Data: image-caption pairs (558K–1.2M)
  - Duration: ~1 epoch, few hours on 8 GPUs

Stage 2: Visual Instruction Tuning
  - Freeze: vision encoder
  - Train: connector + LLM (full or LoRA)
  - Data: multimodal instruction data (150K–1.5M)
  - Duration: 1 epoch, ~24 hours on 8 GPUs

(Optional) Stage 3: RLHF/DPO for VLMs
  - Train with human preference data on visual responses
  - Reduces hallucination, improves factual grounding
```

### 5.2 Stage 1: Feature Alignment Pretraining

**Goal**: Teach the projection layer to map visual features into the LLM's token embedding space.

Data format:
```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image briefly."},
    {"from": "gpt", "value": "A dog sitting on a park bench on a sunny day."}
  ]
}
```

The LLM generates captions for images. Since the LLM is frozen, the only thing that changes is how visual features map to the LLM's space.

### 5.3 Stage 2: Visual Instruction Tuning

**Goal**: Teach the model to follow complex visual instructions.

Data types:
1. **Visual conversations**: Multi-turn Q&A about images
2. **Detailed descriptions**: Long, detailed image descriptions
3. **Complex reasoning**: Requires multi-step inference about visual content

Key datasets:
| Dataset | Size | Type |
|---------|------|------|
| LLaVA-Instruct-150K | 150K | GPT-4 generated conversations |
| ShareGPT4V | 100K | GPT-4V generated descriptions |
| LVIS-Instruct-4V | 220K | Object-focused instructions |
| ALLaVA | 715K | Mixed instruction data |
| Cambrian-10M | 10M | Large-scale mixed |

### 5.4 Scaling VLM Training

The scaling laws for VLMs follow similar patterns to LLMs:
- More data → better generalization (but with diminishing returns)
- Larger LLM backbone → better reasoning about images
- Higher resolution → better fine-grained understanding
- More diverse data → less hallucination

**Resolution scaling**: LLaVA v1.5 uses 336×336, LLaVA-NeXT uses dynamic resolution up to 672×672 by splitting into tiles.

### 5.5 Data Quality > Data Quantity

Key finding from LLaVA v1.5: **665K carefully curated samples outperform millions of noisy samples**. Quality markers:
- Diverse instruction types
- Accurate ground truth
- Complex reasoning chains
- Spatial relationship descriptions

---

## 6. Grounding — From Understanding to Localization (Day 64)

### 6.1 The Grounding Problem

Standard VLMs can describe images but can't **point to** specific objects. Grounding bridges this gap: connecting language to spatial locations.

```
Input: "the red mug on the left side of the table"
Output: bounding box [x₁, y₁, x₂, y₂] = [0.12, 0.34, 0.28, 0.56]
```

### 6.2 Grounding DINO

Grounding DINO (Liu et al. 2023) combines:
- **DINO** (self-supervised ViT features)
- **BERT** (text understanding)
- **Cross-modality fusion** at multiple stages

Architecture:
```
Image → Backbone → Multi-scale features ─┐
                                          ├── Cross-Modality Fusion → Detection Head → Boxes
Text  → BERT → Text features ────────────┘
```

The model takes a text prompt and an image, outputting bounding boxes for objects matching the description.

### 6.3 Florence-2 — Unified Visual Understanding

Florence-2 (Xiao et al. 2024) is Microsoft's unified model handling:
- Image captioning
- Object detection
- Grounding (text → boxes)
- Referring expression comprehension
- OCR
- Region-to-text

All through a **single sequence-to-sequence architecture**:

```
Input:  <task_token> + image tokens + text prompt
Output: text sequence encoding locations as "<loc_x1><loc_y1><loc_x2><loc_y2>"
```

Locations are discretized into vocabulary tokens (e.g., 1000 bins per axis).

### 6.4 Grounded VLMs

Models that combine conversational VLM capability with grounding:

| Model | Approach | Grounding Method |
|-------|----------|-----------------|
| Kosmos-2 | Text tokens for coordinates | `<loc_42><loc_78>` |
| Shikra | Text tokens for coordinates | Referral dialogue |
| Ferret | Region-text alignment | Any-shape region input |
| CogVLM-Grounding | Dedicated grounding head | Box regression |

### 6.5 Why Grounding Matters for Robotics

For a robot to act, it must ground language to physical locations:

```
"Pick up the blue cup"
  1. Understand "blue cup" (vision-language)
  2. Locate it in the image (grounding)  ← THIS
  3. Convert pixel coordinates to 3D pose (calibration)
  4. Plan and execute grasp (action)
```

Without grounding, VLMs can only describe — they can't act.

### 6.6 Coordinate Representation Approaches

| Method | Representation | Pros | Cons |
|--------|---------------|------|------|
| Vocabulary tokens | `<loc_123>` | Simple, unified | Limited precision |
| Continuous regression | [0.123, 0.456] | High precision | Separate head needed |
| Heatmap | Spatial probability map | Rich spatial info | Memory intensive |
| Normalized text | "(0.12, 0.45)" | Pure text, any LLM | Requires parsing |

---

## 7. VLM Applications for Robotics (Day 65)

### 7.1 Scene Understanding

VLMs provide rich scene understanding that classical vision systems lack:

```python
# Classical approach
objects = detector.detect(image)  # ["cup", "plate", "fork"]

# VLM approach
description = vlm.describe(image)
# "A cluttered kitchen counter with a white ceramic mug near the edge,
#  a blue plate with leftover food in the center, and a silver fork
#  resting on the plate. The mug is dangerously close to falling."
```

The VLM provides **context, relationships, and reasoning** beyond object lists.

### 7.2 Spatial Reasoning

VLMs can answer spatial questions:
- "Is the cup to the left or right of the plate?"
- "Which object is closest to the robot gripper?"
- "Is there enough space between objects to place the box?"

**But spatial reasoning is still weak**: VLMs often fail on precise spatial relationships, relative distances, and 3D reasoning from 2D images.

### 7.3 Affordance Detection

VLMs can reason about what actions are possible:

```
Q: "Where can the robot grasp this mug?"
A: "The mug has a handle on the right side. The best grasp point 
    is the handle. Alternatively, a top-down grasp on the rim is 
    possible, but the mug is full of liquid, so the handle grasp 
    is safer."
```

This is **affordance detection through reasoning** — going beyond geometric features.

### 7.4 Task Decomposition

VLMs can break down complex instructions:

```
Q: "Clean up the table"
A: Step 1: Pick up the empty water bottle and place in recycling bin
   Step 2: Stack the dirty plates and move to the sink
   Step 3: Wipe the table surface with the cloth on the counter
   Step 4: Organize the remaining items neatly
```

This is high-level planning that leverages common sense from language pretraining.

### 7.5 Why VLMs Are Necessary But Not Sufficient

VLMs provide:
✅ Scene understanding
✅ Spatial reasoning (approximate)
✅ Common sense
✅ Task decomposition
✅ Natural language interface

VLMs lack:
❌ Motor control knowledge
❌ Physics understanding (at action level)
❌ Embodied experience
❌ Real-time action generation
❌ Precise spatial reasoning

**The gap from VLM to VLA**: VLMs understand but can't act. Adding action tokens and training on robot data bridges this gap (Phase VI).

---

## 8. 🛑 STOP AND REFLECT #4 — The Unification Thesis (Day 66)

### The Reflection

> **Vision and language share the same transformer. The model doesn't distinguish between "seeing" and "reading." This is NOT a metaphor — the attention mechanism literally compresses visual and text tokens identically.**

Let's unpack this:

### What the Transformer Sees

When a VLM processes an image alongside text, the input to the LLM is:

```
[h₁, h₂, ..., h_M, t₁, t₂, ..., t_N]
 ├── visual tokens ──┤ ├── text tokens ─┤
```

Inside the transformer:
- Self-attention computes $\text{Attn}(Q, K, V)$ over **all tokens together**
- Visual token $h_3$ can attend to text token $t_7$ with the same mechanism as $t_7$ attending to $t_3$
- There is **no special "visual attention" vs "text attention"** — it's all the same attention

### The Compression View

Each transformer layer applies the same compression to all tokens:

$$\text{Layer}(x_i) = x_i + \text{FFN}(\text{LayerNorm}(x_i + \text{Attn}(x_i, X, X)))$$

Whether $x_i$ originated from an image patch or a text token is irrelevant. The transformer treats all tokens as points in a shared high-dimensional space and applies the same learned transformations.

### The Key Implication for VLAs

If the transformer treats visual and text tokens identically, then **action tokens can be added in exactly the same way**:

```
[visual_tokens, text_tokens, action_tokens]
 ├── from image ─┤ ├── from text ──┤ ├── robot actions ─┤
```

The model processes all three modalities with the same attention mechanism. This is the architectural insight that makes VLAs possible (Phase VI).

### Why This Matters

Consider the evolution:
1. **LLM**: [text_tokens] → predict next text token
2. **VLM**: [visual_tokens, text_tokens] → predict next text token
3. **VLA**: [visual_tokens, text_tokens, action_tokens] → predict next action token

Each step adds a modality by projecting into the same embedding space. The transformer doesn't need modification — only the tokenization and training data change.

### Write in Your Notebook

Answer these questions before proceeding:

1. **Why doesn't the transformer need separate mechanisms for vision and language?**
   > (Your answer here)

2. **What does "the same attention mechanism" mean concretely?** Write out the attention equation and identify where modality appears (hint: it doesn't).
   > (Your answer here)

3. **If I add a new modality (audio, touch, proprioception), what needs to change?** (Just the encoder and projection — the transformer core is unchanged.)
   > (Your answer here)

4. **What does this imply about the representations inside the transformer?** (Vision, language, and action converge to a shared representation — a "thought" space that is modality-agnostic.)
   > (Your answer here)

5. **Draw the full trajectory from LLM to VLA as a diagram**, showing how each phase adds a modality while keeping the transformer core unchanged.
   > (Your diagram here)

---

## 9. Advanced VLMs (Days 67–68)

### 9.1 Proprietary Frontier Models

**GPT-4V/4o (OpenAI)**
- Architecture: undisclosed, likely vision encoder + large LLM
- Capabilities: exceptional visual reasoning, OCR, chart understanding
- Limitations: closed-source, API-only, expensive

**Gemini (Google DeepMind)**
- Architecture: natively multimodal — trained from scratch on text + image + audio + video
- Key difference: not a "vision encoder bolted onto an LLM" but a unified multimodal model
- Gemini 1.5 Pro: 1M token context, long-video understanding

**Claude (Anthropic)**
- Vision added to Claude 3
- Strong at document understanding, chart analysis
- Constitutional AI training may improve safety on visual content

### 9.2 PaLM-E — Embodied Multimodal

PaLM-E (Driess et al. 2023) is particularly relevant for our VLA trajectory:

```
Input: [robot state tokens, visual tokens, text instruction tokens]
Output: text plan for the robot
```

Key innovations:
- Ingests **robot state** (joint positions, gripper state) as input tokens alongside vision and language
- Trained on robotics data + general multimodal data
- 562B parameter model that can plan robot actions from visual observations

**PaLM-E bridges VLMs and VLAs**: it's a VLM that understands robot state, producing text plans that a lower-level policy can execute.

### 9.3 Open-Source VLMs

**LLaVA-NeXT / LLaVA-OneVision**
- Dynamic high-resolution: splits image into tiles, processes each
- AnyRes: handles arbitrary aspect ratios
- Strong performance at 7B-34B scale

**InternVL (Shanghai AI Lab)**
- InternVL 2.5: competitive with GPT-4V on many benchmarks
- Progressive training strategy
- Open weights, strong community

**Qwen-VL (Alibaba)**
- Multilingual VLM
- Strong OCR and document understanding
- Available from 2B to 72B

**Phi-3-Vision (Microsoft)**
- Only 4.2B parameters
- Surprisingly strong for its size
- Efficient deployment possible

**PaliGemma (Google)**
- Based on SigLIP + Gemma
- Designed for fine-tuning on specific tasks
- Good for research and specialized applications

### 9.4 Comparative Analysis

| Model | Size | Resolution | Grounding | Open | Robot-Relevant |
|-------|------|-----------|-----------|------|---------------|
| GPT-4V | ~1T? | High | No | No | Scene understanding |
| Gemini 1.5 | ~? | Dynamic | Some | No | Video understanding |
| LLaVA-NeXT | 7-34B | Dynamic | Partial | Yes | Base for VLA |
| InternVL2.5 | 2-76B | Dynamic | Yes | Yes | Spatial reasoning |
| PaLM-E | 562B | 224 | Via text | No | Direct robot planning |
| Florence-2 | 0.23-0.77B | 384-768 | Yes | Yes | Grounding + captioning |

### 9.5 Evaluation Benchmarks

| Benchmark | What It Tests |
|-----------|--------------|
| MMBench | Comprehensive VLM evaluation |
| MMMU | Multi-discipline understanding |
| MathVista | Mathematical reasoning with visuals |
| SeedBench | Spatial and temporal understanding |
| RealWorldQA | Real-world visual reasoning |
| GQA | Compositional visual reasoning |
| TextVQA | OCR + reasoning |
| POPE | Hallucination evaluation |

---

## 10. Phase V Capstone — VLM Scene Analysis (Days 69–70)

### 10.1 Project Overview

Build a **VLM-powered robot workspace analyzer** that:
1. Takes camera images from a robot workspace
2. Answers questions about the scene
3. Provides grounded responses (with spatial locations)
4. Compares multiple VLM backends

### 10.2 System Architecture

```
┌──────────────┐
│ Robot Camera   │
│ (or test set)  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  VLM Scene        │
│  Analyzer          │
│                    │
│  ┌──────────────┐ │    ┌─────────────────────┐
│  │ CLIP Backbone │──────│ Similarity Search     │
│  └──────────────┘ │    │ (object queries)      │
│                    │    └─────────────────────┘
│  ┌──────────────┐ │    ┌─────────────────────┐
│  │ VLM (LLaVA)  │──────│ Scene Description     │
│  └──────────────┘ │    │ Spatial Reasoning     │
│                    │    └─────────────────────┘
│  ┌──────────────┐ │    ┌─────────────────────┐
│  │ Florence-2    │──────│ Grounded Detection    │
│  └──────────────┘ │    │ (text → boxes)        │
│                    │    └─────────────────────┘
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Structured Output  │
│ - Scene description│
│ - Object locations │
│ - Spatial relations│
│ - Action proposals │
└──────────────────┘
```

### 10.3 Key Evaluation Metrics

1. **Scene description accuracy**: Does the description match ground truth?
2. **Spatial reasoning**: Can the model correctly answer "left of", "above", "between"?
3. **Grounding accuracy**: IoU of predicted vs actual bounding boxes
4. **Action relevance**: Are proposed actions reasonable for a robot?

Full project details in [projects/05-vlm-scene-analysis/README.md](../projects/05-vlm-scene-analysis/README.md).

---

## Key Equations

### CLIP Contrastive Loss

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{\text{sim}(I_i,T_i)/\tau}}{\sum_{j}e^{\text{sim}(I_i,T_j)/\tau}} + \log\frac{e^{\text{sim}(T_i,I_i)/\tau}}{\sum_{j}e^{\text{sim}(T_j,I_i)/\tau}}\right]$$

### SigLIP Sigmoid Loss

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2}\sum_{i,j}\log\sigma\!\bigl(y_{ij}\cdot(\mathbf{z}_i^{I\top}\mathbf{z}_j^T\cdot e^t + b)\bigr), \quad y_{ij} = \begin{cases}+1 & i=j \\ -1 & i\neq j\end{cases}$$

### Cosine Similarity

$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

### Gated Cross-Attention (Flamingo)

$$\text{out} = x + \tanh(\alpha) \cdot \text{CrossAttn}(x, v), \quad \alpha \text{ init } 0$$

### Cross-Attention

$$\text{CrossAttn}(Q, K_v, V_v) = \text{softmax}\!\left(\frac{Q K_v^\top}{\sqrt{d}}\right)V_v$$

where $Q$ comes from text/LLM tokens and $K_v, V_v$ come from visual features.

### Visual Token Projection

$$h_i = W_2 \cdot \text{GELU}(W_1 \cdot v_i + b_1) + b_2$$

---

## Architecture Diagrams

### CLIP Training

```
Batch of N image-text pairs:
(I₁,T₁), (I₂,T₂), ..., (I_N,T_N)

┌─────────┐              ┌─────────┐
│ Image    │              │ Text    │
│ Encoder  │              │ Encoder │
│ (ViT)    │              │ (Trans) │
└────┬─────┘              └────┬────┘
     │ z_I₁...z_I_N           │ z_T₁...z_T_N
     │                        │
     └───────┐    ┌───────────┘
             ▼    ▼
     ┌──────────────────┐
     │  N×N Similarity   │
     │  Matrix            │
     │  S_ij = z_Ii·z_Tj │
     └──────────────────┘
             │
     maximize diagonal
     minimize off-diagonal
```

### LLaVA Full Pipeline

```
Stage 1: Alignment          Stage 2: Instruction Tuning
┌─────────────────┐         ┌─────────────────┐
│ ViT (frozen) ❄️  │         │ ViT (frozen) ❄️  │
└────────┬────────┘         └────────┬────────┘
         │                           │
┌────────▼────────┐         ┌────────▼────────┐
│ Projection 🔥    │         │ Projection 🔥    │
└────────┬────────┘         └────────┬────────┘
         │                           │
┌────────▼────────┐         ┌────────▼────────┐
│ LLM (frozen) ❄️  │         │ LLM 🔥 (or LoRA) │
└─────────────────┘         └─────────────────┘

Data: captions              Data: instructions
```

### Evolution: LLM → VLM → VLA

```
LLM:  [t₁ t₂ ... tₙ] → predict t_{n+1}          (text only)
       └── text ──┘

VLM:  [v₁ ... vₘ | t₁ ... tₙ] → predict t_{n+1}  (vision + text)
       └─ visual ─┘ └─ text ─┘

VLA:  [v₁ ... vₘ | t₁ ... tₙ] → predict a₁ ... aₖ  (vision + text → action)
       └─ visual ─┘ └─ text ─┘   └── action ──┘

Same transformer. Same attention. Different tokens.
```

---

## Key Takeaways

1. **CLIP proved vision and language can share an embedding space** through contrastive pretraining on 400M image-text pairs — enabling zero-shot transfer.

2. **LLaVA showed that a simple projection layer can connect a vision encoder to an LLM**, creating a multimodal conversational model with minimal new parameters.

3. **The connector design space** (linear, MLP, Q-Former, perceiver) offers tradeoffs between token count and information retention — but the LLM backbone matters most.

4. **Flamingo's cross-attention** provides an alternative to concatenation, enabling few-shot visual learning and more flexible multimodal integration.

5. **Grounding bridges understanding and localization** — VLMs must point, not just describe, for robotics applications.

6. **VLMs provide scene understanding, spatial reasoning, and common sense** for robots, but lack motor control and embodied experience — the gap that VLAs will fill.

7. **The transformer treats all tokens identically** regardless of modality — this architectural property is what makes extending to action tokens (VLAs) natural and principled.

---

## Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | 2021 | Contrastive vision-language pretraining |
| [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | 2022 | Gated cross-attention, few-shot visual learning |
| [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | 2023 | Simple but effective visual instruction tuning |
| [LLaVA v1.5: Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) | 2023 | MLP connector, data quality > quantity |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | 2023 | VLM with robot state input |
| [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | 2023 | Better contrastive loss for VLM training |
| [Grounding DINO: Marrying DINO with Grounded Pre-Training](https://arxiv.org/abs/2303.05499) | 2023 | Open-set object detection with text |
| [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242) | 2024 | Unified vision model via sequence-to-sequence |
| [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 2024 | Dynamic resolution, AnyRes |
| [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238) | 2024 | Open-source GPT-4V competitor |
| [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966) | 2023 | Multilingual VLM |

---

## Connection to the Thread

```
Phase I:   Transformers compress text sequences → language understanding
Phase II:  Autoregressive prediction → generation
Phase III: GPT scaling → emergent abilities
Phase IV:  ViTs compress image patches → visual understanding
Phase V:   VLMs fuse vision + language → multimodal compression ← YOU ARE HERE
Phase VI:  VLAs add action tokens → embodied AI
```

VLMs are **multimodal compression engines**. The model compresses vision and language into a shared representation space where:
- Similar concepts cluster together regardless of modality
- Cross-modal reasoning emerges from shared attention
- The representation is ready for action extension

The path to VLAs is now clear: take a VLM, add action tokens projected into the same space, train on robot demonstration data, and the transformer's modality-agnostic attention handles the rest.

**Next**: [11-robot-learning-foundations.md](11-robot-learning-foundations.md) — Phase VI begins.

---

*Previous: [09-detection-segmentation.md](09-detection-segmentation.md) · Next: [11-robot-learning-foundations.md](11-robot-learning-foundations.md)*
