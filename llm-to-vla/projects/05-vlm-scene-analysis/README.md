# Project 05 — VLM-Powered Robot Scene Analyzer

> **Phase V Capstone · Days 69–70 · ~8 hours**
> Prerequisites: Exercises 08 (VLM Experiments)
> Study Notes: [10-vision-language-models.md](../../study-notes/10-vision-language-models.md)

---

## Objective

Build a system that takes **robot camera images** and answers questions about the scene using multiple VLM backends. Evaluate which approach works best for robotics scene understanding.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Robot Camera Feed   │
                    │   (or test image set) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   SceneAnalyzer       │
                    │                       │
                    │   Backend A: CLIP+LLM │──▶ Similarity-based Q&A
                    │   Backend B: LLaVA    │──▶ Generative description
                    │   Backend C: Florence │──▶ Grounded detection
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Structured Output   │
                    │   - Scene description  │
                    │   - Object list + locs │
                    │   - Spatial relations  │
                    │   - Action proposals   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Evaluation Suite    │
                    │   - Description acc   │
                    │   - Spatial reasoning  │
                    │   - Grounding IoU      │
                    │   - Latency / cost     │
                    └───────────────────────┘
```

---

## Tasks

### Task 1: Scene Analyzer Core (Day 69, ~3 hours)

Build `scene_analyzer.py` with a unified interface:

```python
class SceneAnalyzer:
    def __init__(self, backend: str):  # "clip", "llava", "florence"
        ...
    
    def describe(self, image) -> str:
        """Generate a natural language scene description."""
    
    def list_objects(self, image) -> list[dict]:
        """Return [{"name": str, "bbox": [x1,y1,x2,y2], "confidence": float}]."""
    
    def answer(self, image, question: str) -> str:
        """Answer a free-form question about the scene."""
    
    def ground(self, image, text_query: str) -> list[dict]:
        """Locate objects matching the text query. Return bboxes."""
    
    def propose_actions(self, image, task: str) -> list[str]:
        """Suggest robot actions for the given task."""
```

### Task 2: Evaluation Benchmark (Day 69, ~2 hours)

Create `benchmark.py` with a test suite:

| Metric | How to Measure | Target |
|--------|----------------|--------|
| Description accuracy | Manual rating (1–5) on 20 images | ≥ 3.5 avg |
| Object detection | Precision/Recall at IoU=0.5 | ≥ 0.6 |
| Spatial reasoning | Accuracy on 40 spatial questions | ≥ 0.5 |
| Grounding accuracy | mAP@0.5 on text queries | ≥ 0.4 |
| Latency | Seconds per image (GPU) | Report |

Test images: 20 robot workspace scenes (real or synthetic). Annotate ground truth for objects and spatial relationships.

### Task 3: Comparison Report (Day 70, ~3 hours)

Run all three backends through the benchmark. Produce `REPORT.md`:

1. **Quantitative comparison table** — all metrics, all backends
2. **Qualitative examples** — 5 images showing each backend's strengths/weaknesses
3. **Failure analysis** — categorize failure modes per backend
4. **Recommendation** — which backend (or combination) is best for which robotics sub-task?
5. **Connection to VLAs** — what's still missing for a robot to act on this understanding?

---

## Deliverables

```
projects/05-vlm-scene-analysis/
├── README.md              ← this file
├── scene_analyzer.py      ← unified analyzer with 3 backends
├── benchmark.py           ← evaluation harness
├── data/
│   ├── images/            ← 20 test images
│   └── annotations.json   ← ground truth objects, relations
├── results/
│   ├── scores.json        ← quantitative results
│   └── examples/          ← qualitative comparison images
└── REPORT.md              ← comparison analysis
```

---

## Evaluation Criteria

| Criterion | Weight | Pass Threshold |
|-----------|--------|---------------|
| All 3 backends working | 30% | All produce output |
| Benchmark runs end-to-end | 20% | Scores computed for all metrics |
| Quantitative comparison | 20% | Table with all metrics |
| Qualitative analysis | 15% | ≥5 example comparisons |
| Robotics connection | 15% | Clear articulation of VLM→VLA gap |

---

## Tips

- Start with Florence-2 (smallest, easiest to run) then add LLaVA and CLIP
- Use the `supervision` library for bounding box visualization
- If GPU memory is tight, run backends sequentially, not in parallel
- For annotations, keep it simple: 5–10 objects per image, 3–4 spatial relations
- The comparison report matters more than the code — focus on insights

---

*Previous: [04-efficient-finetuning/README.md](../04-efficient-finetuning/README.md) · Next: [06-vla-integration/README.md](../06-vla-integration/README.md)*
