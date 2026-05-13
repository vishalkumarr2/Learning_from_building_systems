# Day 58: Phase IV Capstone — Day 2

> **Phase IV — Vision: ViT, 3D, Video | Week 9 | 2.5 hours**
> *"Before you build VLMs, prove you understand vision transformers." — Phase IV checkpoint*

## Navigation
- **Previous:** [Day 57: Phase IV Capstone Day 1](day-57-phase4-capstone-1.md)
- **Next:** [Day 59: CLIP — Contrastive VL Learning](day-59-clip.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Evaluation & Ablation (60 min)

### Ablation Study

Run your perception pipeline with different configurations and compare:

```python
import time


def benchmark_pipeline(pipeline, image_paths, config_name="default"):
    """Benchmark perception pipeline on a set of images."""
    times = []
    n_objects = []
    
    for path in image_paths:
        start = time.time()
        result = pipeline.process(path)
        elapsed = time.time() - start
        
        times.append(elapsed)
        n_objects.append(len(result['objects']))
    
    print(f"\n=== {config_name} ===")
    print(f"Avg time: {sum(times)/len(times):.2f}s")
    print(f"Avg objects: {sum(n_objects)/len(n_objects):.1f}")
    print(f"Total time: {sum(times):.2f}s for {len(image_paths)} images")
    
    return {
        'config': config_name,
        'avg_time': sum(times) / len(times),
        'avg_objects': sum(n_objects) / len(n_objects),
    }


# Ablation configurations
ablations = {
    'Full pipeline': {'scene': True, 'depth': True, 'detection': True},
    'No depth': {'scene': True, 'depth': False, 'detection': True},
    'No scene features': {'scene': False, 'depth': True, 'detection': True},
    'Detection only': {'scene': False, 'depth': False, 'detection': True},
}
```

### Quality Assessment

| Metric | How to Measure |
|--------|---------------|
| Detection precision | Manual review: are detected objects correct? |
| Depth quality | Relative ordering: are near objects darker in depth map? |
| Feature coherence | PCA visualization: do semantically similar regions cluster? |
| 3D accuracy | Do object centroids make spatial sense? |

---

## Phase IV Checkpoint (60 min)

**Answer these 6 questions without looking at notes.** Write 3-5 sentences per answer.

### 1. ViT Architecture

> *Explain how ViT processes an image. Cover: patch creation, embedding, positional encoding, and classification.*

**Your answer:**

_Space for writing. Key points: split into 16×16 patches, linear projection to embeddings, add learned positional embeddings, prepend [CLS] token, transformer encoder, classify from [CLS] output._

### 2. DINO Self-Supervision

> *How does DINO learn without labels? What emerges in the attention maps?*

**Your answer:**

_Key points: student-teacher self-distillation, teacher is EMA of student, multi-crop strategy (student gets local crops, teacher gets global), centering + sharpening prevent collapse, attention maps discover object boundaries._

### 3. Depth Estimation

> *Why can a single camera estimate depth? What cues does the model learn?*

**Your answer:**

_Key points: monocular cues — relative size, occlusion, texture gradient, perspective, atmospheric haze. ViT's global attention helps reason about scene-wide cues. Training on diverse datasets with scale-invariant loss._

### 4. Video Understanding

> *How does TimeSformer adapt ViT for video? What's the computational trick?*

**Your answer:**

_Key points: divided space-time attention — temporal attention (same patch across frames) + spatial attention (all patches within a frame). Reduces O((TN)²) to O(T²N + N²T). Separable position embeddings._

### 5. DETR Innovation

> *How is DETR different from traditional detectors like YOLO? What's the role of Hungarian matching?*

**Your answer:**

_Key points: no anchors, no NMS. Learned object queries predict a set of (class, box) pairs. Hungarian matching provides optimal bipartite assignment between predictions and ground truth. End-to-end differentiable._

### 6. Vision-Language Bridge

> *Describe three ways to connect a vision encoder to an LLM. Which is simplest? Which compresses most?*

**Your answer:**

_Key points: (1) MLP projection (LLaVA) — simplest, N tokens. (2) Cross-attention (Flamingo) — gated layers, 64 resampled tokens. (3) Q-Former (BLIP-2) — 32 compressed queries, most compression. Modern trend: MLP is enough._

---

## Self-Assessment Rubric

| Score | Meaning |
|-------|---------|
| 6/6 | Ready for Phase V — you understand vision deeply |
| 4-5/6 | Solid — review the weak areas, then proceed |
| 2-3/6 | Revisit Days 45-56 before continuing |
| 0-1/6 | Re-do the week; don't rush |

---

## Reflection (30 min)

### What Phase IV Taught You

Write a paragraph connecting Phase IV back to the curriculum thread:

```
Phase I:   Neural network foundations
Phase II:  Attention + Transformers (text)
Phase III: LLM training + alignment
Phase IV:  Vision transformers + 3D + video + detection  ← YOU ARE HERE
Phase V:   Vision-Language Models (connecting vision + language)
Phase VI:  VLAs (vision + language + actions)
```

**Key insight:** The transformer architecture didn't change between Phase II and Phase IV. What changed was the **tokenization** — words became patches. The same architecture that generates text also classifies images, estimates depth, and detects objects.

### Looking Ahead to Phase V

Phase V connects everything:
- **CLIP** (Day 59): Align vision and language in shared space
- **BLIP-2/Flamingo** (Day 61): Add visual understanding to LLMs
- **LLaVA** (Day 62): Enable visual conversations

> *VLMs can see and describe. After Phase V, you'll understand how to make a model that both sees and speaks. Phase VI will add the final piece: acting.*

---

## Key Takeaways

1. **Phase IV verified.** You can build a full perception pipeline from components
2. **Modularity.** Each vision module is independently useful and combinable
3. **Checkpoint passed.** If you answered 4+ questions correctly, proceed to Phase V
4. **The thread continues.** Same transformer, new modality — now let's connect them

## Connection to the Thread

> *Phase IV is complete. You can make transformers see. Tomorrow starts Phase V: making transformers see AND understand language simultaneously. CLIP is the foundation.*
