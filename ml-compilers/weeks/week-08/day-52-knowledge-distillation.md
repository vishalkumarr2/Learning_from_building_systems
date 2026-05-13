# Day 52: Knowledge Distillation

> **Phase IV · Week 8 · Day 52 of 70 · 2.5 hours**

*"The teacher whispers probabilities across classes; the student hears structure the labels alone could never reveal."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 51: Weight Compression & Pruning](day-51-weight-compression.md) | [Day 53: TensorRT Optimization](day-53-tensorrt-optimization.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Hard labels say "this is a cat." Soft labels say "this is 90% cat, 5% lynx, 3% tiger, 2% dog." Those secondary probabilities — the **dark knowledge** — encode similarity structure the teacher learned over millions of examples. Knowledge distillation transfers this rich information from a large, accurate **teacher** model to a small, deployable **student** model. DistilBERT (66M params) matches 97% of BERT-base's accuracy (110M params) while being 60% smaller and 2× faster. Progressive distillation can compress a 1000-step diffusion model to 4 steps. Combined with pruning and quantization, distillation is the third pillar of model compression — and often the most powerful one because it doesn't just compress weights, it compresses *knowledge*.

---

## 1. The Teacher-Student Framework

```
Knowledge Distillation — Architecture Overview
══════════════════════════════════════════════════════════════

  Input x ─────────────────────┬──────────────────────────┐
                               │                          │
                               ▼                          ▼
                    ┌──────────────────┐      ┌──────────────────┐
                    │  Teacher Model   │      │  Student Model   │
                    │  (Large, frozen) │      │  (Small, trainable)
                    │  ResNet-152      │      │  ResNet-18       │
                    │  BERT-large      │      │  TinyBERT        │
                    └────────┬─────────┘      └────────┬─────────┘
                             │                          │
                             ▼                          ▼
                    ┌────────────────┐        ┌────────────────┐
                    │  Soft logits   │        │  Soft logits   │
                    │  z_t / T       │        │  z_s / T       │
                    └────────┬───────┘        └────────┬───────┘
                             │                          │
                             └──────────┬───────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  L_distill =      │
                              │  KL(σ(z_t/T) ‖    │
                              │      σ(z_s/T))    │
                              │  × T²             │
                              └──────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐     Hard label y
                              │  L_total =        │◀────────────┐
                              │  α·L_distill +    │             │
                              │  (1-α)·L_CE(s,y)  │   ┌────────┴────┐
                              └──────────────────┘   │ L_CE =       │
                                                      │ CE(σ(z_s), y)│
                                                      └──────────────┘
```

---

## 2. Hinton's Soft Label Distillation

The foundational technique from Hinton, Vinyals & Dean (2015):

### 2.1 Temperature Scaling

Softmax with temperature $T$ controls the entropy of the output distribution:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T = 1$: standard softmax (peaked distribution)
- $T > 1$: softer distribution, reveals inter-class similarities  
- $T \to \infty$: uniform distribution

```
Temperature Effect on Softmax Output
══════════════════════════════════════

  Class:        Cat    Lynx   Tiger  Dog    Fish
  Logits:       5.2    2.1    1.8    0.3   -1.5

  T=1 (hard):   0.91   0.04   0.03   0.01   0.00
  T=5 (soft):   0.38   0.18   0.16   0.10   0.06
  T=10 (softer):0.27   0.17   0.16   0.13   0.09
               ──▲──                          ──▲──
               Still dominant           Now visible
```

### 2.2 The Distillation Loss

$$\mathcal{L}_{\text{distill}} = T^2 \cdot \text{KL}\left(\sigma\left(\frac{z_t}{T}\right) \;\bigg\|\; \sigma\left(\frac{z_s}{T}\right)\right)$$

The $T^2$ factor compensates for the reduced gradient magnitudes at high temperature:

$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial z_s} \propto \frac{1}{T^2} \quad \Rightarrow \quad T^2 \cdot \mathcal{L}_{\text{KL}} \text{ restores gradient scale}$$

### 2.3 Total Training Objective

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{distill}}(z_t, z_s; T) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(z_s, y)$$

Typical hyperparameters: $T \in [4, 20]$, $\alpha \in [0.5, 0.9]$.

### 2.4 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft distillation loss (KL divergence on softened distributions)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        distill_loss = F.kl_div(
            soft_student, soft_teacher,
            reduction="batchmean"
        ) * (self.T ** 2)

        # Hard label loss (standard cross-entropy)
        ce_loss = self.ce_loss(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss


# Training loop
teacher = load_pretrained_teacher().eval()
student = create_small_student()
criterion = DistillationLoss(temperature=4.0, alpha=0.7)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)

        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 3. Feature-Based Distillation

Instead of (or in addition to) matching output logits, match **intermediate representations**:

```
Feature-Based Distillation
══════════════════════════════════════════════════════════

  Teacher                              Student
  ┌────────────────┐                   ┌────────────────┐
  │  Layer 1       │ ────hint────────▶ │  Layer 1       │
  │  [B,256,56,56] │   projection      │  [B,64,56,56]  │
  ├────────────────┤                   ├────────────────┤
  │  Layer 2       │ ────hint────────▶ │  Layer 2       │
  │  [B,512,28,28] │   projection      │  [B,128,28,28] │
  ├────────────────┤                   ├────────────────┤
  │  Layer 3       │ ────hint────────▶ │  Layer 3       │
  │  [B,1024,14,14]│   projection      │  [B,256,14,14] │
  ├────────────────┤                   ├────────────────┤
  │  Output        │ ────KL div───────▶│  Output        │
  └────────────────┘                   └────────────────┘

  Projection: 1×1 conv to match teacher dimensions
  Loss: L_total = L_logit + β·Σ L_feature(f_t^l, proj(f_s^l))
```

```python
class FeatureDistillationLoss(nn.Module):
    def __init__(self, teacher_channels, student_channels):
        super().__init__()
        # 1x1 conv projections to match teacher dimensions
        self.projections = nn.ModuleList([
            nn.Conv2d(sc, tc, kernel_size=1)
            for sc, tc in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_features, teacher_features):
        loss = 0.0
        for proj, sf, tf in zip(
            self.projections, student_features, teacher_features
        ):
            projected = proj(sf)
            # Normalize features before comparing
            sf_norm = F.normalize(projected.flatten(2), dim=-1)
            tf_norm = F.normalize(tf.flatten(2), dim=-1)
            loss += F.mse_loss(sf_norm, tf_norm)
        return loss
```

---

## 4. Attention Transfer

For transformer models, transfer the **attention patterns** themselves:

$$\mathcal{L}_{\text{attn}} = \sum_{l=1}^{L} \left\| \frac{A_t^l}{\|A_t^l\|_F} - \frac{A_s^l}{\|A_s^l\|_F} \right\|_F^2$$

where $A^l \in \mathbb{R}^{H \times N \times N}$ is the attention map at layer $l$, with $H$ heads and $N$ tokens.

```python
def attention_transfer_loss(teacher_attentions, student_attentions):
    """
    teacher_attentions: list of [B, H_t, N, N] attention maps
    student_attentions: list of [B, H_s, N, N] attention maps
    """
    loss = 0.0
    # Map student layers to teacher layers (e.g., every other)
    layer_mapping = list(zip(
        range(0, len(teacher_attentions), 2),  # Teacher: layers 0, 2, 4, ...
        range(len(student_attentions))           # Student: layers 0, 1, 2, ...
    ))

    for t_idx, s_idx in layer_mapping:
        # Average across heads
        t_attn = teacher_attentions[t_idx].mean(dim=1)  # [B, N, N]
        s_attn = student_attentions[s_idx].mean(dim=1)

        # Frobenius-normalize
        t_norm = t_attn / (t_attn.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        s_norm = s_attn / (s_attn.norm(dim=(-2, -1), keepdim=True) + 1e-8)

        loss += F.mse_loss(s_norm, t_norm)

    return loss / len(layer_mapping)
```

---

## 5. Case Study: DistilBERT

The most successful distillation in NLP — a blueprint for efficient deployment:

```
DistilBERT Architecture Comparison
══════════════════════════════════════════════════════════

  BERT-base               DistilBERT
  ┌──────────────┐        ┌──────────────┐
  │ 12 layers    │        │ 6 layers     │   ← 50% layers
  │ 768 hidden   │        │ 768 hidden   │   ← Same width
  │ 12 heads     │        │ 12 heads     │   ← Same heads
  │ 110M params  │        │ 66M params   │   ← 40% smaller
  ├──────────────┤        ├──────────────┤
  │ GLUE: 79.6   │        │ GLUE: 77.0   │   ← 97% accuracy
  │ Latency: 1x  │        │ Latency: 0.5x│   ← 2× faster
  └──────────────┘        └──────────────┘

  Distillation recipe:
  ┌────────────────────────────────────────────┐
  │  1. Initialize student from every-other     │
  │     teacher layer (layers 0,2,4,6,8,10)    │
  │  2. Triple loss:                            │
  │     - L_ce: masked LM cross-entropy        │
  │     - L_mlm: soft distillation (T=8)       │
  │     - L_cos: cosine embedding loss         │
  │  3. Train on same data as BERT             │
  │  4. No NSP task (removed for simplicity)   │
  └────────────────────────────────────────────┘
```

The triple loss combines: (1) cross-entropy on hard MLM labels ($\alpha_{ce} = 0.5$), (2) KL divergence on softened logits at $T=8$ ($\alpha_{mlm} = 0.33$), and (3) cosine embedding loss between hidden states ($\alpha_{cos} = 0.17$).

---

## 6. Progressive Distillation for Diffusion Models

Distillation applied to iterative sampling — halving the number of denoising steps repeatedly:

```
Progressive Distillation for Diffusion
══════════════════════════════════════════════════════════

  Round 1: Teacher uses 1024 steps → Student learns 512 steps
  Round 2: Student becomes teacher → New student learns 256 steps
  Round 3: 256-step teacher → 128-step student
  Round 4: 128 → 64 → 32 → 16 → 8 → 4 steps

  ┌─────────┐   distill   ┌─────────┐   distill   ┌─────────┐
  │ 1024     │ ──────────▶ │  512    │ ──────────▶ │  256    │ ──▶ ...
  │ steps    │  merge      │  steps  │  merge      │  steps  │
  │ (teacher)│  2 → 1      │(student)│  2 → 1      │(student)│
  └─────────┘             └─────────┘             └─────────┘
  
  Each round: student predicts the result of 2 teacher steps in 1 step
  
  v-prediction parameterization:
  v_θ(x_t, t) = α_t · ε − σ_t · x_0
  
  Student loss: ‖v_student(x_t, t) − v_teacher_2steps(x_t, t)‖²
```

This technique (Salimans & Ho, 2022) enabled **4-step generation** from Stable Diffusion with minimal quality loss.

---

## Hands-On Exercises

### Exercise 1: Basic Logit Distillation (25 min)
1. Train a ResNet-50 teacher on CIFAR-10 to ~93% accuracy
2. Define a small 4-layer CNN student (~100K params)
3. Train the student with distillation ($T=4$, $\alpha=0.7$)
4. Compare accuracy vs. student trained with hard labels only
5. Sweep temperature $T \in \{1, 2, 4, 8, 20\}$ — plot accuracy vs. $T$

### Exercise 2: Feature Distillation (20 min)
1. Extract intermediate features from both teacher and student at 3 layer points
2. Add 1×1 conv projections to align channel dimensions
3. Train with combined logit + feature loss — does it help over logit-only?

### Exercise 3: Self-Distillation (15 min)
```python
# Teacher = Student (same architecture, different training)
# Step 1: Train model normally
# Step 2: Use the trained model as teacher for a fresh copy
# Does this improve over standard training? Why?
```

### Exercise 4: Distillation + Quantization + Pruning (20 min)
1. Start with a teacher ResNet-50 (93% accuracy)
2. Distill to ResNet-18 student → measure accuracy
3. Prune student to 50% sparsity → fine-tune → measure accuracy
4. Quantize pruned student to INT8 → measure accuracy and size
5. What's the total compression ratio and final accuracy?

---

## Key Takeaways

1. **Dark knowledge is powerful** — soft labels carry inter-class similarity information that hard labels discard; this is why distillation works
2. **Temperature controls information flow** — higher $T$ reveals more structure in the teacher's predictions; typical sweet spot is $T \in [4, 10]$
3. **Feature distillation adds depth** — matching intermediate representations helps the student learn richer internal representations, not just output mimicry
4. **DistilBERT = the blueprint** — triple loss (CE + KL + cosine), layer initialization from teacher, same architecture width with half the depth
5. **Progressive distillation enables fast sampling** — repeatedly halving diffusion steps via distillation achieves 4-step generation
6. **Compression techniques compose** — distillation → pruning → quantization can yield 10–100× smaller models at 95%+ accuracy

---

## Further Reading

- [Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)
- [Sanh et al., "DistilBERT" (NeurIPS 2019 Workshop)](https://arxiv.org/abs/1910.01108)
- [Salimans & Ho, "Progressive Distillation for Fast Sampling" (ICLR 2022)](https://arxiv.org/abs/2202.00512)
- [Romero et al., "FitNets: Hints for Thin Deep Nets" (ICLR 2015)](https://arxiv.org/abs/1412.6550)
- [Jiao et al., "TinyBERT" (EMNLP 2020)](https://arxiv.org/abs/1909.10351)
- [Zagoruyko & Komodakis, "Paying More Attention to Attention" (ICLR 2017)](https://arxiv.org/abs/1612.03928)

---

## Tomorrow's Preview

**Day 53: TensorRT Optimization** — You've compressed and distilled models. Now it's time to make them *fly*. TensorRT fuses layers, calibrates precision, plans memory, and profiles kernels to extract every last TFLOP from NVIDIA GPUs. You'll learn the full pipeline from ONNX parsing through INT8 calibration to TensorRT-LLM for transformer inference.
