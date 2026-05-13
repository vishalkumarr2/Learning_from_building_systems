# Project 07 — Full VLA Pipeline: Fine-tune, Deploy, Evaluate

> Phase VII Capstone · Days 107–112 · ~15 hours
> The culmination of the 112-day LLM-to-VLA curriculum

---

## Objective

Fine-tune OpenVLA on a custom manipulation task, deploy it as a real-time
control loop, and rigorously evaluate performance. This project ties together
every skill from the curriculum: deep learning foundations, vision-language
understanding, imitation learning, and robot control.

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 16 GB (with 4-bit quant) | A100 40 GB |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB free | 100 GB free |
| Python | 3.10+ | 3.10 |
| Key libs | transformers, peft, lerobot, simple-pid | + wandb, tensorboard |

---

## Task 1 — Select Task & Prepare Data (2 h)

1. Browse the [LeRobot dataset hub](https://huggingface.co/lerobot) and select
   a manipulation task (e.g., `pusht`, `aloha_sim_transfer_cube`).
2. Inspect the dataset: observation shape, action dimensionality, episode lengths.
3. Split into train / val / test (80 / 10 / 10 by episode).
4. Build a `DataLoader` with proper collation (image, instruction, action).
5. Visualize 5 sample episodes: image grid + action trajectory plot.

**Deliverable**: `data_pipeline.py` with verified DataLoader and sample plots.

## Task 2 — Fine-tune OpenVLA with LoRA (4 h)

1. Load OpenVLA-7B with 4-bit quantization.
2. Attach LoRA adapters (rank 16, α=32) to Q/K/V/O projections.
3. Freeze the vision encoder; train only LoRA + action head.
4. Train for 10–20 epochs with cosine LR schedule, gradient accumulation.
5. Log to wandb: train loss, val loss, learning rate, GPU memory.
6. Save the best LoRA adapter by validation loss.

**Deliverable**: `finetune.py`, saved adapter in `checkpoints/`, wandb run link.

## Task 3 — Deploy Real-Time Inference (3 h)

1. Build an inference server that loads the fine-tuned model once.
2. Implement a control loop running at ≥10 Hz:
   - Capture observation → run VLA → output action → step environment.
3. Measure per-step latency; optimize with:
   - Batched KV-cache reuse across steps
   - CUDA graph capture for repeated forward passes
   - Action chunking (predict K future actions, execute sequentially)
4. Verify the loop sustains target frequency for 100+ steps.

**Deliverable**: `deploy.py` with latency profiling output.

## Task 4 — Evaluate Rigorously (3 h)

1. **Success rate**: Roll out 100 episodes; report mean ± std.
2. **Generalization**: Test with novel objects / colours not in training set (20 episodes).
3. **Latency profile**: Histogram of per-step inference times.
4. **Ablation**: Compare base OpenVLA vs your fine-tuned adapter.
5. **Failure analysis**: Categorize the top-3 failure modes with attention visualizations.

**Deliverable**: `evaluate.py`, evaluation report with plots.

## Task 5 — Hybrid Safety Layer (2 h)

1. Wrap the VLA with a classical safety controller:
   - Workspace bounds check (Cartesian limits)
   - Velocity / acceleration clamping
   - Collision proximity threshold (if sim provides distance)
2. Compare: pure VLA vs VLA + safety on 50 episodes.
3. Report: safety interventions count, success rate delta, smoothness metric.

**Deliverable**: `safety_layer.py`, comparison table.

---

## Deliverables Summary

| # | Artifact | Format |
|---|----------|--------|
| 1 | Fine-tuned LoRA adapter | `checkpoints/best_lora_adapter/` |
| 2 | Deployment code | `deploy.py` with ≥10 Hz control loop |
| 3 | Evaluation report | `REPORT.md` with embedded plots |
| 4 | Plots | success rate bar, latency histogram, attention heatmaps |
| 5 | Reflection essay | `reflection.md` — 2 pages: "From Backprop to VLA" |

---

## Evaluation Rubric (100 points)

| Category | Points | Criteria |
|----------|--------|----------|
| **Implementation quality** | 30 | Clean code, modular design, reproducible with a single script. LoRA fine-tuning converges and adapter loads correctly. |
| **Evaluation rigor** | 25 | 100-episode evaluation, generalization test, ablation vs base model, statistical reporting (mean ± std). |
| **Hybrid safety system** | 15 | Safety layer catches ≥90 % of out-of-bounds actions. Comparison table shows trade-off clearly. |
| **Analysis & visualization** | 15 | Attention heatmaps on ≥3 scenes, failure mode categorization, latency histogram with percentiles. |
| **Reflection essay** | 15 | Connects curriculum arc (backprop → transformers → VLMs → IL → VLA). Honest about limitations and surprises. |

### Grade Thresholds

| Grade | Points | Description |
|-------|--------|-------------|
| A | 90–100 | Publication-quality evaluation, novel insight in reflection |
| B | 75–89 | All tasks complete, solid analysis |
| C | 60–74 | Core pipeline works, evaluation is thin |
| F | <60 | Pipeline incomplete or does not run |

---

## Getting Started

```bash
# 1. Activate environment from Exercise 11
conda activate vla

# 2. Create project directory
mkdir -p project-07-vla-capstone && cd project-07-vla-capstone

# 3. Copy your Exercise 11 utilities
cp ../exercises/11_ex1_openvla_inference.py utils_inference.py
cp ../exercises/11_ex3_vla_finetune.py     utils_finetune.py
cp ../exercises/11_ex4_hybrid_controller.py utils_hybrid.py

# 4. Scaffold project files
touch data_pipeline.py finetune.py deploy.py evaluate.py safety_layer.py
touch REPORT.md reflection.md

# 5. Start with Task 1 — data exploration
python data_pipeline.py --task pusht --visualize
```

### Recommended Timeline

| Day | Focus | Hours |
|-----|-------|-------|
| 107 | Task 1: data pipeline + Task 2: start fine-tuning | 3 |
| 108 | Task 2: finish fine-tuning + Task 3: deploy | 3 |
| 109 | Task 3: optimize latency + Task 4: evaluation | 3 |
| 110 | Task 4: finish evaluation + Task 5: safety layer | 3 |
| 111 | Polish report, generate final plots | 2 |
| 112 | Write reflection essay, final review | 1 |

---

## References

1. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," 2024.
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
3. Cadene et al., "LeRobot: State-of-the-art ML for Robotics," HuggingFace, 2024.
4. Brohan et al., "RT-2: Vision-Language-Action Models," 2023.

---

*Project 07 — the final milestone of the LLM-to-VLA curriculum.*
