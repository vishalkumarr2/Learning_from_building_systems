# LLM-to-VLA Curriculum — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Create all 34 content files (16 study notes, 11 exercises, 7 project scaffolds) for the v2 curriculum.

**Architecture:** Each study note is a standalone deep-dive document covering theory, key equations, implementation guidance, and paper references for its topic. Exercise files provide step-by-step coding exercises. Project directories contain README specs for milestone projects.

**Tech Stack:** Markdown, PyTorch code snippets, paper citations, architecture diagrams (ASCII art)

---

## Scope

```
study-notes/    16 files  (~800-1500 lines each, the BULK of the work)
exercises/      11 files  (~300-600 lines each)
projects/       7 dirs    (~100-200 lines each README)
```

**Total:** ~34 files, estimated ~25,000-35,000 lines of educational content.

**Constraint:** Each study note must be self-contained — a learner reads ONLY that file plus the curriculum for context. No forward references to unwritten notes.

---

## File Manifest

### Study Notes (create in order — each may reference previous)

| # | File | Phase | Days | Key Content |
|---|------|-------|------|-------------|
| 1 | `study-notes/01-dl-foundations.md` | I | 1-7 | Backprop, CNN, RNN essentials, info theory, embeddings, training stability |
| 2 | `study-notes/02-attention-mechanism.md` | II | 10-12 | Bahdanau, scaled dot-product, multi-head attention |
| 3 | `study-notes/03-transformer-architecture.md` | II | 13-16 | Full transformer, positional encoding, STOP AND REFLECT #1 |
| 4 | `study-notes/04-transformer-variants.md` | II | 17-22 | Efficient attention, KV cache, normalization, MoE, BERT, tokenization |
| 5 | `study-notes/05-gpt-scaling.md` | II | 23-30 | GPT, nanoGPT ablations, scaling laws, T5, STOP AND REFLECT #2 |
| 6 | `study-notes/06-llm-training-alignment.md` | III | 31-36 | SFT, RLHF, DPO, LoRA, evaluation |
| 7 | `study-notes/07-llm-engineering.md` | III | 37-44 | ICL, long context, RAG overview, quantization, LLM for robotics |
| 8 | `study-notes/08-vision-transformers.md` | IV | 45-50 | ViT, Swin, DINO, MAE, STOP AND REFLECT #3 |
| 9 | `study-notes/09-3d-video-detection.md` | IV | 51-58 | Depth, point clouds, video, DETR, Florence-2, SAM 2 |
| 10 | `study-notes/10-vision-language-models.md` | V | 59-70 | CLIP, LLaVA, BLIP-2, PaLI, CoCa, STOP AND REFLECT #4 |
| 11 | `study-notes/11-rl-foundations.md` | VI | 71-73 | MDPs, policy gradients, PPO, connection to RLHF |
| 12 | `study-notes/12-diffusion-flow.md` | VI | 74-77 | DDPM, DDIM, latent diffusion, flow matching |
| 13 | `study-notes/13-imitation-learning.md` | VI | 78-84 | BC, ACT, Diffusion Policy, action tokenization, STOP AND REFLECT #5 |
| 14 | `study-notes/14-robot-data-eval.md` | VI | 85-91 | Data collection, evaluation, debugging policies |
| 15 | `study-notes/15-vla-architectures.md` | VII | 92-100 | RT-1→RT-2→Octo→OpenVLA→π₀→π₀.₅→GR00T→PaLM-E→GR-2 |
| 16 | `study-notes/16-deployment-hybrid.md` | VII | 101-112 | Sim-to-real, hybrid control, ROS2, deployment, capstone |

### Exercises

| # | File | Covers |
|---|------|--------|
| 1 | `exercises/01-autograd-cnn.md` | Phase I: micrograd, ResNet, training monitor |
| 2 | `exercises/02-attention-from-scratch.md` | Bahdanau, scaled dot-product, multi-head |
| 3 | `exercises/03-build-transformer.md` | Full transformer implementation + training |
| 4 | `exercises/04-nanogpt-ablations.md` | nanoGPT experiments, scaling curves, BPE |
| 5 | `exercises/05-finetune-llm.md` | LoRA fine-tuning, DPO, evaluation |
| 6 | `exercises/06-implement-vit.md` | ViT from scratch, DINO features |
| 7 | `exercises/07-3d-depth-video.md` | Depth Anything, video understanding |
| 8 | `exercises/08-clip-vlm-experiments.md` | CLIP zero-shot, VLM comparison |
| 9 | `exercises/09-rl-diffusion-policy.md` | Q-learning, PPO, DDPM, flow matching |
| 10 | `exercises/10-imitation-act.md` | BC, ACT, action tokenization |
| 11 | `exercises/11-vla-evaluation.md` | VLA inference, hybrid systems, deployment |

### Projects

| # | Directory | Deliverable |
|---|-----------|-------------|
| 1 | `projects/01-rnn-vs-attention/` | Character LM comparison notebook |
| 2 | `projects/02-mini-lm-ablations/` | Trained GPT + ablation report |
| 3 | `projects/03-robotics-assistant/` | LoRA fine-tuned robotics LLM |
| 4 | `projects/04-robot-perception/` | Depth + video + detection pipeline |
| 5 | `projects/05-warehouse-visual-qa/` | VLM comparison on OKS scenarios |
| 6 | `projects/06-diffusion-policy/` | Own demos → diffusion policy |
| 7 | `projects/07-vla-capstone/` | End-to-end VLA pipeline |

---

## Implementation Order

### Batch 1: Phase I (study-notes 01 + exercises 01 + project 01)
These are foundational — everything else builds on them.

**Task 1.1:** Create `study-notes/01-dl-foundations.md`
- Covers Days 1-7 of curriculum
- Sections: Computation Graphs & Backprop, CNN & ResNets, RNN/LSTM Essentials, Seq2Seq & Bottleneck, Information Theory & Compression, Embeddings, Training Stability Cookbook
- Key equations: chain rule, cross-entropy as compression, Shannon entropy
- Must include the compression = prediction = intelligence thread
- ~1200-1500 lines

**Task 1.2:** Create `exercises/01-autograd-cnn.md`
- Step-by-step: micrograd implementation, ResNet from scratch, training monitor
- Include starter code templates and expected outputs
- ~400-500 lines

**Task 1.3:** Create `projects/01-rnn-vs-attention/README.md`
- Project spec: RNN vs simple attention character LM comparison
- Requirements, evaluation criteria, deliverables
- ~100-150 lines

### Batch 2: Phase II-A (study-notes 02-03 + exercises 02-03)
Core attention and transformer — the most critical study notes.

**Task 2.1:** Create `study-notes/02-attention-mechanism.md`
- Bahdanau attention, scaled dot-product, multi-head
- Full derivation of attention equation
- ~800-1000 lines

**Task 2.2:** Create `study-notes/03-transformer-architecture.md`
- Full encoder-decoder transformer, positional encoding (sinusoidal + RoPE)
- Training a transformer, STOP AND REFLECT #1
- ~1000-1200 lines

**Task 2.3:** Create `exercises/02-attention-from-scratch.md`
- Build attention step by step, verify against PyTorch
- ~300-400 lines

**Task 2.4:** Create `exercises/03-build-transformer.md`
- Full transformer implementation from scratch
- ~400-500 lines

### Batch 3: Phase II-B (study-notes 04-05 + exercises 04 + project 02)
Variants, GPT, scaling — the nanoGPT ablation lab.

**Task 3.1:** Create `study-notes/04-transformer-variants.md`
- Efficient attention, KV cache, normalization, MoE, BERT, tokenization
- ~1000-1200 lines

**Task 3.2:** Create `study-notes/05-gpt-scaling.md`
- GPT, nanoGPT, scaling laws deep dive, T5, STOP AND REFLECT #2
- ~1200-1500 lines

**Task 3.3:** Create `exercises/04-nanogpt-ablations.md`
- nanoGPT setup, systematic ablation experiments, BPE implementation
- ~500-600 lines

**Task 3.4:** Create `projects/02-mini-lm-ablations/README.md`
- Mini-LM training + ablation report spec
- ~100-150 lines

### Batch 4: Phase III (study-notes 06-07 + exercises 05 + project 03)
LLM training pipeline.

**Task 4.1:** Create `study-notes/06-llm-training-alignment.md`
- SFT, RLHF, DPO, LoRA, evaluation
- ~1000-1200 lines

**Task 4.2:** Create `study-notes/07-llm-engineering.md`
- ICL theory, long context, RAG overview, quantization, LLM for robotics
- ~800-1000 lines

**Task 4.3:** Create `exercises/05-finetune-llm.md`
- LoRA fine-tuning exercise, DPO comparison
- ~400-500 lines

**Task 4.4:** Create `projects/03-robotics-assistant/README.md`
- Robotics LoRA assistant project spec
- ~100-150 lines

### Batch 5: Phase IV (study-notes 08-09 + exercises 06-07 + project 04)
Vision transformers, 3D, video.

**Task 5.1:** Create `study-notes/08-vision-transformers.md`
- ViT, Swin, DINO, MAE, STOP AND REFLECT #3
- ~1000-1200 lines

**Task 5.2:** Create `study-notes/09-3d-video-detection.md`
- Depth Anything, point clouds, video, DETR, Florence-2, SAM 2
- ~1000-1200 lines

**Task 5.3:** Create `exercises/06-implement-vit.md`
- ViT from scratch, DINO feature extraction
- ~400-500 lines

**Task 5.4:** Create `exercises/07-3d-depth-video.md`
- Depth estimation, video understanding exercises
- ~300-400 lines

**Task 5.5:** Create `projects/04-robot-perception/README.md`
- Robot perception pipeline project spec
- ~100-150 lines

### Batch 6: Phase V (study-notes 10 + exercises 08 + project 05)
Vision-language models.

**Task 6.1:** Create `study-notes/10-vision-language-models.md`
- CLIP, LLaVA, BLIP-2, PaLI, CoCa, STOP AND REFLECT #4
- ~1200-1500 lines

**Task 6.2:** Create `exercises/08-clip-vlm-experiments.md`
- CLIP zero-shot, VLM comparison exercises
- ~400-500 lines

**Task 6.3:** Create `projects/05-warehouse-visual-qa/README.md`
- Warehouse visual QA project spec
- ~100-150 lines

### Batch 7: Phase VI (study-notes 11-14 + exercises 09-10 + project 06)
Robot learning — the biggest batch. RL, diffusion, imitation, data.

**Task 7.1:** Create `study-notes/11-rl-foundations.md`
- MDPs, policy gradients, PPO, connection to RLHF
- ~800-1000 lines

**Task 7.2:** Create `study-notes/12-diffusion-flow.md`
- DDPM, DDIM, latent diffusion, flow matching
- ~1000-1200 lines

**Task 7.3:** Create `study-notes/13-imitation-learning.md`
- BC, ACT, Behavior/Decision Transformers, Diffusion Policy, action tokenization, STOP AND REFLECT #5
- ~1200-1500 lines

**Task 7.4:** Create `study-notes/14-robot-data-eval.md`
- Data collection, evaluation methodology, debugging policies
- ~800-1000 lines

**Task 7.5:** Create `exercises/09-rl-diffusion-policy.md`
- Q-learning, PPO, DDPM, flow matching exercises
- ~500-600 lines

**Task 7.6:** Create `exercises/10-imitation-act.md`
- BC, ACT, action tokenization exercises
- ~400-500 lines

**Task 7.7:** Create `projects/06-diffusion-policy/README.md`
- Diffusion policy from own demonstrations project spec
- ~100-150 lines

### Batch 8: Phase VII (study-notes 15-16 + exercises 11 + project 07)
VLA architectures and deployment — the finale.

**Task 8.1:** Create `study-notes/15-vla-architectures.md`
- RT-1→RT-2→Octo→OpenVLA→π₀→π₀.₅→GR00T N1→PaLM-E→GR-2
- ~1200-1500 lines

**Task 8.2:** Create `study-notes/16-deployment-hybrid.md`
- Sim-to-real, hybrid VLA+classical, ROS2 deployment, safety, capstone
- ~1200-1500 lines

**Task 8.3:** Create `exercises/11-vla-evaluation.md`
- VLA inference, hybrid system design, deployment exercises
- ~400-500 lines

**Task 8.4:** Create `projects/07-vla-capstone/README.md`
- End-to-end VLA capstone project spec
- ~150-200 lines

---

## Quality Requirements

### Every Study Note Must Include:
1. **Header** with phase, days covered, prerequisites, learning objectives
2. **Theory sections** with key equations in LaTeX-style (use `$...$` for inline, `$$...$$` for blocks)
3. **Architecture diagrams** (ASCII art for key models)
4. **Code snippets** (PyTorch) for key concepts — pseudocode where full implementation is in exercises
5. **Paper references** with arxiv links and reading priority
6. **Key takeaways** section at the end (bullet points)
7. **Connection to the thread**: how does this topic connect to "compression = prediction = intelligence"?
8. **STOP AND REFLECT** marker where applicable (notes 03, 05, 08, 10, 13)

### Every Exercise Must Include:
1. **Prerequisites** (which study note to read first)
2. **Setup** (pip installs, imports)
3. **Step-by-step tasks** with starter code templates
4. **Expected outputs** (what the correct result looks like)
5. **Stretch goals** for going deeper
6. **Self-check** questions

### Every Project Must Include:
1. **Overview** and learning objectives
2. **Requirements** (what to build, what to measure)
3. **Suggested approach** (high-level steps)
4. **Evaluation rubric** (how to know it's good)
5. **Deliverables** list

---

## Execution Strategy

**Use subagents for parallelism:** Each batch is independent (except Batch 1 which is foundation). Within a batch, study notes and exercises can be written in parallel by subagents.

**Estimated effort per batch:**
- Batch 1 (Phase I): 3 files → 1 subagent
- Batch 2 (Phase II-A): 4 files → 2 subagents (study notes || exercises)
- Batch 3 (Phase II-B): 4 files → 2 subagents
- Batch 4 (Phase III): 4 files → 2 subagents
- Batch 5 (Phase IV): 5 files → 2 subagents
- Batch 6 (Phase V): 3 files → 1 subagent
- Batch 7 (Phase VI): 7 files → 3 subagents (LARGEST batch)
- Batch 8 (Phase VII): 4 files → 2 subagents

---

*Created: 2026-04-28*
*Source: CURRICULUM.md v2*
