# Project 03 — Robotics Domain LoRA Assistant

> Phase III Capstone · Days 42–44 · ~8 hours
> Prerequisites: Exercises 05 (LoRA fine-tuning), Study notes 06–07

---

## Objective

Fine-tune a small LLM (1–3B parameters) to be a knowledgeable robotics domain
assistant using QLoRA. The model should answer questions about ROS 2, navigation,
sensors, OKS robot systems, and common failure modes significantly better than
the base model.

---

## Architecture

```
┌──────────────────────────────────────────┐
│       Robotics LoRA Assistant            │
│                                          │
│  ┌────────────────┐  ┌───────────────┐  │
│  │ Base Model      │  │ LoRA Adapter  │  │
│  │ (LLaMA 3.2 1B) │ +│ (rank 32)     │  │
│  │ 4-bit quantized │  │ ~2.5M params  │  │
│  └────────────────┘  └───────────────┘  │
│                                          │
│  Input:  "What causes sensorbar drift?"  │
│  Output: Domain-accurate explanation     │
└──────────────────────────────────────────┘
```

---

## Data Sources

| Source | Content | Estimated QA Pairs |
|--------|---------|-------------------|
| `knowledge/systems/` | OKS subsystem docs | 100+ |
| `knowledge/errors.json` | Error codes + causes | 50+ |
| `knowledge/oks-system-overview.md` | System architecture | 30+ |
| ROS 2 documentation | Concepts, nodes, topics | 100+ |
| `docs/rca/` | Past RCA reports | 50+ |
| Navigation algorithms | SLAM, path planning, localization | 50+ |
| Sensor fusion | IMU, odometry, sensorbar | 30+ |

**Target:** 400–500 high-quality instruction-response pairs.

---

## Steps

### 1. Data Preparation (Day 42, ~3h)

```bash
mkdir -p data/robotics-qa
```

- Extract QA pairs from knowledge base documents
- Write a script to convert RCA reports into QA format
- Include error code lookups, troubleshooting steps, system explanations
- Format: `{"instruction": "...", "output": "..."}`
- Split: 90% train, 10% eval
- Validate: no duplicates, consistent formatting

### 2. Training (Day 43, ~3h)

- Base model: `meta-llama/Llama-3.2-1B` (or `TinyLlama-1.1B`)
- QLoRA: 4-bit NF4 quantization + LoRA rank 32
- Target modules: `q_proj, v_proj, k_proj, o_proj`
- Epochs: 5–10 (small dataset → more epochs)
- Learning rate: 2e-4 with cosine schedule
- Monitor with wandb

### 3. Evaluation (Day 43–44, ~2h)

Create an evaluation set of 50+ robotics questions:

| Category | Example Question | Count |
|----------|-----------------|-------|
| Error codes | "What does NAV_ESTIMATED_STATE_NOT_FINITE mean?" | 10 |
| Troubleshooting | "Robot stopped with MOTOR_OVERCURRENT, what to check?" | 10 |
| ROS 2 | "How do ROS 2 lifecycle nodes work?" | 10 |
| Navigation | "Explain the difference between DWA and TEB planners" | 10 |
| System architecture | "What components make up the OKS navigation stack?" | 10 |

**Metrics:**
- Qualitative: side-by-side comparison (base vs LoRA vs GPT-4)
- Quantitative: perplexity on held-out robotics text
- Domain accuracy: manual scoring of answer correctness (1–5 scale)

### 4. Deployment Test (Day 44, ~2h)

- Merge LoRA into base weights
- Quantize merged model to INT4 (GGUF)
- Benchmark inference latency on consumer hardware
- Test with interactive chat

---

## Deliverables

| Deliverable | Description |
|-------------|-------------|
| `data/robotics-qa/train.json` | Training dataset (400+ QA pairs) |
| `data/robotics-qa/eval.json` | Evaluation dataset (50+ questions) |
| `adapters/robotics-lora/` | Trained LoRA adapter weights |
| `eval_results.md` | Evaluation results with comparisons |
| `examples/` | 10 example conversations showing improvement |
| `README.md` | This file (updated with results) |

---

## Evaluation Template

```markdown
## Results

### Training
- Base model: ___
- LoRA rank: ___
- Training examples: ___
- Epochs: ___
- Final training loss: ___
- Training time: ___

### Quality Comparison (1-5 scale, 5=best)

| Question Category | Base Model | LoRA Model | GPT-4 |
|-------------------|-----------|-----------|-------|
| Error codes       |           |           |       |
| Troubleshooting   |           |           |       |
| ROS 2 concepts    |           |           |       |
| Navigation        |           |           |       |
| System arch       |           |           |       |
| **Average**       |           |           |       |

### Inference Benchmarks

| Metric        | FP16    | INT8    | INT4 (GGUF) |
|---------------|---------|---------|-------------|
| Memory (GB)   |         |         |             |
| Tokens/sec    |         |         |             |
| Latency (s)   |         |         |             |
```

---

## Success Criteria

- [ ] LoRA model scores ≥3.5/5 average on domain questions
- [ ] LoRA model scores ≥1.0 higher than base model on average
- [ ] Inference runs at ≥10 tokens/sec on consumer GPU
- [ ] INT4 quantized model fits in <4 GB memory
- [ ] 10 example conversations demonstrate clear improvement

---

## Stretch Goals

- **Multi-task LoRA:** Train separate adapters for error diagnosis vs ROS 2 tutoring
- **RAG + LoRA:** Combine the LoRA model with RAG over the full knowledge base
- **Chatbot UI:** Wrap in a simple Gradio interface for interactive testing
- **Continuous learning:** Add new QA pairs from resolved tickets, retrain adapter

---

*This project is the Phase III capstone of the LLM-to-VLA curriculum.*
*Next: Phase IV — Multimodal Models (CLIP, LLaVA, Flamingo)*
