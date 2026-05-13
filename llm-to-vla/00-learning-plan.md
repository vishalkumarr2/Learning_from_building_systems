# LLM-to-VLA — Learning Plan (v2)
### For: Engineer with ML basics + deep robotics domain knowledge (OKS AMR, ROS, control, state estimation)
### Goal: From attention mechanism to deploying Vision-Language-Action models on robots
### Revised after 5-persona expert panel review

---

## Why This Track Exists

You investigate robot failures daily — you know ROS, estimators, controllers, sensorbar physics.
The missing piece: the AI/ML revolution that's transforming robotics. VLAs (Vision-Language-Action
models) are the frontier — robots that see, understand language, and act. This track builds you
from "I know ML basics" to "I can train and deploy a VLA on a robot."

**What you already have:** Python, PyTorch, ML fundamentals, robotics domain expertise.
**What you're building:** Attention → Transformers → LLMs → Vision → VLMs → RL + Diffusion → VLAs.

**Key rebalancing (v2):** The curriculum was restructured so that the "A" in VLA gets as much
weight as the "V" and "L". Your existing robotics knowledge is an asset — the curriculum now
explicitly leverages it in Phases VI–VII with ROS2 integration and hybrid control exercises.

---

## Dependency Graph

```
                             ┌──────────────────────────────────────┐
                             │  16-deployment-hybrid.md             │
                             │  Sim-to-real, ROS2, hybrid control   │
                             └───────────────────┬──────────────────┘
                                                 │
                             ┌───────────────────▼──────────────────┐
                             │  15-vla-architectures.md             │
                             │  RT-2, Octo, OpenVLA, π₀, π₀.₅     │
                             └───┬───────────────────────────┬──────┘
                                 │                           │
               ┌─────────────────▼──────┐   ┌───────────────▼─────────────┐
               │  10-vision-language-    │   │  13-imitation-learning.md   │
               │  models.md             │   │  + 14-robot-data-eval.md    │
               │  CLIP, LLaVA, PaLI    │   │  BC, ACT, Diffusion Policy  │
               └────┬──────────────┬────┘   └──────────────┬──────────────┘
                    │              │                        │
       ┌────────────▼────┐  ┌─────▼──────────┐  ┌─────────▼───────────────┐
       │ 07-llm-eng.md  │  │ 08+09-vision   │  │ 11-rl-foundations.md    │
       │ Quant, ICL     │  │ ViT, 3D, video │  │ + 12-diffusion-flow.md  │
       └────────┬───────┘  └─────┬──────────┘  └─────────────────────────┘
                │                │
       ┌────────▼───────┐       │
       │ 06-llm-train   │       │
       │ SFT, RLHF, DPO│       │
       └────────┬───────┘       │
                │               │
       ┌────────▼───────┐      │
       │ 05-gpt-scaling │      │
       │ GPT, nanoGPT   │      │
       │ Scaling laws    │      │
       └────────┬───────┘      │
                │               │
       ┌────────▼──────────────┘
       │
  ┌────▼────────────────────┐
  │ 04-transformer-          │
  │ variants.md              │
  │ Efficiency, MoE, KV$    │
  └──────┬───────────────────┘
         │
  ┌──────▼──────────────────┐
  │ 03-transformer-          │
  │ architecture.md          │
  │ Full encoder-decoder     │
  └──────┬───────────────────┘
         │
  ┌──────▼──────────────────┐
  │ 02-attention-            │
  │ mechanism.md             │
  │ Bahdanau → Multi-Head    │
  └──────┬───────────────────┘
         │
  ┌──────▼──────────────────┐
  │ 01-dl-foundations.md     │
  │ CNN, RNN, Info Theory,   │
  │ Training Stability       │
  └──────────────────────────┘
```

**Reading order:** Strictly sequential (01 → 02 → ... → 16). Each builds on the previous.

---

## Topic Breakdown

| # | Study Note | Phase | Est. Time | What It Gives You |
|---|-----------|-------|-----------|-------------------|
| 01 | `01-dl-foundations.md` | I | 10 hrs | CNN, RNN essentials, info theory (compression = prediction), training stability cookbook |
| 02 | `02-attention-mechanism.md` | II | 8 hrs | Attention from first principles; Q/K/V intuition |
| 03 | `03-transformer-architecture.md` | II | 10 hrs | Build a full transformer from scratch; understand every component |
| 04 | `04-transformer-variants.md` | II | 10 hrs | Efficiency (Flash, KV cache), MoE, normalization, activations |
| 05 | `05-gpt-scaling.md` | II | 14 hrs | GPT, nanoGPT as ablation lab, tokenization (BPE), scaling laws deep dive, T5 |
| 06 | `06-llm-training-alignment.md` | III | 12 hrs | SFT, RLHF, DPO, LoRA — the full alignment pipeline |
| 07 | `07-llm-engineering.md` | III | 8 hrs | Quantization, ICL theory, long context, RAG overview (compressed) |
| 08 | `08-vision-transformers.md` | IV | 10 hrs | ViT from scratch, Swin, DINO, MAE — images as tokens |
| 09 | `09-3d-video-detection.md` | IV | 10 hrs | Depth Anything, point clouds, video understanding, DETR, Florence-2, SAM 2 |
| 10 | `10-vision-language-models.md` | V | 12 hrs | CLIP, LLaVA, BLIP-2, PaLI, CoCa; fine-tune a VLM |
| 11 | `11-rl-foundations.md` | VI | 8 hrs | MDPs, policy gradients, PPO, connection to RLHF |
| 12 | `12-diffusion-flow.md` | VI | 10 hrs | DDPM (3 days), DDIM, latent diffusion, flow matching |
| 13 | `13-imitation-learning.md` | VI | 14 hrs | BC, DAgger, ACT, Behavior/Decision Transformers, Diffusion Policy, action tokenization |
| 14 | `14-robot-data-eval.md` | VI | 10 hrs | Data collection, teleoperation, evaluation methodology, debugging policies |
| 15 | `15-vla-architectures.md` | VII | 14 hrs | RT-1→RT-2→Octo→OpenVLA→π₀→π₀.₅→GR00T N1→GR-2 |
| 16 | `16-deployment-hybrid.md` | VII | 12 hrs | Sim-to-real (2 days), hybrid VLA+classical (3 days), ROS2 deployment (3 days) |

**Total: ~162 hours of study notes + ~118 hours of exercises/projects = ~280 hours (16 weeks)**

---

## Milestone Projects

| Phase | Project | Skills Demonstrated |
|-------|---------|---------------------|
| I | RNN vs Attention comparison | Character LM: show why attention wins |
| II | Mini-LM + ablation report | Train GPT, systematic ablation experiments |
| III | Robotics LoRA assistant | Fine-tune LLM on robotics Q&A with LoRA |
| IV | Robot perception pipeline | 3D depth + video + object detection on camera stream |
| V | Warehouse visual QA | Compare CLIP/LLaVA/BLIP-2 on OKS scenarios |
| VI | Diffusion policy from own demos | Collect demos → train diffusion policy → evaluate |
| VII | **VLA Capstone** | End-to-end: collect data → train VLA → hybrid deploy + safety |

---

## STOP AND REFLECT Moments

These 5 moments require you to **pause and journal** before continuing:

| # | Day | Insight to Internalize |
|---|-----|----------------------|
| 🛑 1 | 16 | "Attention is a soft dictionary lookup. The transformer is embarrassingly simple — just attention + FFN + residuals, stacked." |
| 🛑 2 | 26 | "Same architecture, more data, more compute, keeps getting better along a power law. What does this tell us about learning?" |
| 🛑 3 | 50 | "Images ARE sequences. The transformer doesn't care if tokens came from text or image patches." |
| 🛑 4 | 66 | "VLMs are a unified perception-language interface. The missing piece is the action output." |
| 🛑 5 | 84 | "Actions are just another token type. A VLM that outputs action tokens IS a VLA. The boundary between LM and robot controller just dissolved." |

---

## Connection to Existing Tracks

This track connects to your completed learning tracks:

| Existing Track | Connection Point |
|---------------|------------------|
| `navigation-estimator/` | EKF → how VLAs replace/augment classical estimation |
| `control-systems/` | PID/MPC → Phase VII hybrid VLA+classical control exercises |
| `ros2-handson/` | ROS2 integration in Phase VII deployment (Days 104, 109) |
| `optimization/` | Gradient descent, loss landscapes — same math in DL |
| `python-oks/` | Python proficiency for all implementations |
| `electronics/` | Understanding sensor inputs that feed into VLAs |

---

## How to Use the Agents

For expert instruction during each phase, invoke these agents:

```bash
# Socratic questioning on a new concept
→ Use `socratic-mentor` agent: "Explain why self-attention is O(n²)"

# Get implementation reviewed
→ Use `python-pro` agent: "Review my attention implementation"

# Deep research on a paper
→ Use `deep-research` agent: "Summarize the RT-2 paper's key contributions"

# Quiz yourself
→ Use `learning-guide` agent: "Quiz me on transformer architecture"

# Debug training issues
→ Use `debugger` agent: "My ViT loss is not decreasing"

# RL/robot learning questions
→ Use `ml-engineer` agent: "Why does diffusion policy handle multimodal actions?"
```

---

*Last updated: 2026-04-28*
*Version: 2.0 (post expert panel review)*
