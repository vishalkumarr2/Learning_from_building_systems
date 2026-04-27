# LLM-to-VLA — Learning Plan
### For: Engineer with ML basics + deep robotics domain knowledge (OKS AMR, ROS, control, state estimation)
### Goal: From attention mechanism to deploying Vision-Language-Action models on robots

---

## Why This Track Exists

You investigate robot failures daily — you know ROS, estimators, controllers, sensorbar physics.
The missing piece: the AI/ML revolution that's transforming robotics. VLAs (Vision-Language-Action
models) are the frontier — robots that see, understand language, and act. This track builds you
from "I know ML basics" to "I can train and deploy a VLA on a robot."

**What you already have:** Python, PyTorch, ML fundamentals, robotics domain expertise.
**What you're building:** Attention → Transformers → LLMs → Vision → VLMs → VLAs.

---

## Dependency Graph

```
                                    ┌─────────────────────────────────┐
                                    │  14-deployment-frontiers.md     │
                                    │  Real robots, hybrid, OKS apps  │
                                    └────────────────┬────────────────┘
                                                     │
                                    ┌────────────────▼────────────────┐
                                    │  13-vla-models.md               │
                                    │  RT-2, Octo, OpenVLA, π0       │
                                    └───┬────────────────────────┬────┘
                                        │                        │
                      ┌─────────────────▼──────┐   ┌─────────────▼────────────┐
                      │  10-vision-language-    │   │  12-robot-learning.md    │
                      │  models.md              │   │  Imitation, diffusion    │
                      │  CLIP, LLaVA, BLIP-2   │   │  policy, action spaces   │
                      └────┬──────────────┬─────┘   └─────────────┬───────────┘
                           │              │                        │
              ┌────────────▼────┐  ┌──────▼──────────┐  ┌─────────▼───────────┐
              │ 08-llm-eng.md  │  │ 09-vision-       │  │ 11-diffusion.md     │
              │ RAG, tools     │  │ transformers.md  │  │ DDPM, flow match    │
              └────────┬───────┘  │ ViT, DINO, MAE  │  └─────────────────────┘
                       │          └──────┬───────────┘
              ┌────────▼───────┐         │
              │ 07-llm-        │         │
              │ training.md    │         │
              │ SFT, RLHF     │         │
              └────────┬───────┘         │
                       │                 │
         ┌─────────────▼──────┐          │
         │ 06-gpt-decoders.md │          │
         │ GPT, autoregressive│          │
         └─────────┬──────────┘          │
                   │                     │
         ┌─────────▼──────────┐          │
         │ 05-bert-encoders.md│          │
         │ BERT, tokenization │          │
         └─────────┬──────────┘          │
                   │                     │
         ┌─────────▼─────────────────────┘
         │
  ┌──────▼──────────────────┐
  │ 04-transformer-          │
  │ variants.md              │
  │ Efficiency, KV cache     │
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
  │ 01-deep-learning-        │
  │ refresh.md               │
  │ CNN, RNN, LSTM, Seq2Seq  │
  └──────────────────────────┘
```

**Reading order:** Strictly sequential (01 → 02 → ... → 14). Each builds on the previous.

---

## Topic Breakdown

| # | Study Note | Est. Time | What It Gives You |
|---|-----------|-----------|-------------------|
| 01 | `01-deep-learning-refresh.md` | 12 hrs | CNN, RNN/LSTM fluency; understand WHY attention was needed |
| 02 | `02-attention-mechanism.md` | 8 hrs | Attention from first principles; Q/K/V intuition |
| 03 | `03-transformer-architecture.md` | 10 hrs | Build a full transformer from scratch; understand every component |
| 04 | `04-transformer-variants.md` | 12 hrs | Efficiency (Flash, KV cache), MoE, modern design choices |
| 05 | `05-bert-encoders.md` | 10 hrs | BERT, tokenization (BPE), bidirectional understanding |
| 06 | `06-gpt-decoders.md` | 12 hrs | GPT, autoregressive LM, nanoGPT, sampling, scaling laws |
| 07 | `07-llm-training.md` | 12 hrs | SFT, RLHF, DPO, LoRA — the full LLM training pipeline |
| 08 | `08-llm-engineering.md` | 15 hrs | RAG, tools, prompting, quantization, deployment |
| 09 | `09-vision-transformers.md` | 12 hrs | ViT from scratch, DINO, MAE — images as tokens |
| 10 | `10-vision-language-models.md` | 12 hrs | CLIP, LLaVA, BLIP-2 — the multimodal bridge |
| 11 | `11-diffusion-models.md` | 10 hrs | DDPM, flow matching — generative models for actions |
| 12 | `12-robot-learning.md` | 10 hrs | Imitation learning, diffusion policy, action spaces |
| 13 | `13-vla-models.md` | 12 hrs | RT-2, Octo, OpenVLA, π0 — robots that see, think, act |
| 14 | `14-deployment-frontiers.md` | 10 hrs | Real deployment, hybrid architectures, OKS applications |

**Total: ~157 hours of study notes + ~123 hours of exercises/projects = ~280 hours (16 weeks)**

---

## Milestone Projects

| Phase | Project | Skills Demonstrated |
|-------|---------|---------------------|
| I-II | Annotated Transformer | Implement transformer from scratch |
| II-III | Train your own mini-LM | GPT architecture, training loops |
| III | Robotics RAG Assistant | RAG, tool use, LLM engineering |
| IV | Visual Search Engine | ViT, contrastive learning |
| V | Robotics Visual QA | CLIP/LLaVA for robot scenes |
| VI | Diffusion Policy Agent | Diffusion for robot actions |
| VII | **VLA Capstone** | End-to-end VLA pipeline on a robot task |

---

## Connection to Existing Tracks

This track connects to your completed learning tracks:

| Existing Track | Connection Point |
|---------------|------------------|
| `navigation-estimator/` | EKF → how VLAs replace/augment classical estimation |
| `control-systems/` | PID/MPC → how VLAs replace/augment classical control |
| `ros2-handson/` | ROS2 integration for VLA deployment |
| `optimization/` | Gradient descent, loss landscapes — same math in DL |
| `python-oks/` | Python proficiency for implementations |
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
```

---

*Last updated: 2026-04-28*
