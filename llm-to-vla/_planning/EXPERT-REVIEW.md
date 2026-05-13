# Expert Panel Review — "From Attention to VLA" Curriculum

> **Date:** July 2025
> **Reviewed by:** 5 simulated world-class expert personas
> **Status:** Review complete — awaiting decision on revisions

## The Panel

| # | Persona | Domain | Overall Verdict |
|---|---------|--------|-----------------|
| 1 | **Andrej Karpathy** | Transformer/LLM Education | Skeleton strong, time allocation wrong |
| 2 | **Chelsea Finn** | Robot Learning / VLAs | Solid survey, insufficient for *doing* robot learning (3/5) |
| 3 | **Fei-Fei Li** | Computer Vision / Multimodal | B overall — missing 3D vision entirely |
| 4 | **Ilya Sutskever** | Deep Learning Foundations | Competent engineering curriculum; lacks deep understanding |
| 5 | **Sergey Levine** | Applied Robotics ML | "LLM curriculum with a robotics tail" — needs rebalancing |

---

## Consensus: What ALL 5 Experts Agree On

1. The dependency graph (attention → transformers → LLMs → VLMs → VLAs) is **correct**
2. "Build before read" philosophy is **excellent pedagogy**
3. Phase I is **too long** for someone who already knows ML basics — compress it
4. Phase VI-VII (robot learning) is **too compressed** relative to its difficulty
5. Too much LLM application engineering (RAG, tool use, assistant projects) that doesn't serve the VLA goal

---

## CRITICAL Findings (Must Fix)

| # | Finding | Experts | Action |
|---|---------|---------|--------|
| C1 | **No RL content at all** — can't understand online VLA fine-tuning, reward shaping, or why diffusion beats Gaussian policies | Levine, Finn | Add 3 days: MDP, policy gradients, PPO, connection to RLHF |
| C2 | **Data collection treated as a paragraph** — 60-80% of real robot learning is data, curriculum gives it ~1% | Finn, Levine | Add 2 days: teleoperation, data quality, dataset formats, hands-on demo collection |
| C3 | **Compress Phase I (save ~5 days)** — RNNs/LSTMs/Word2Vec are historical, not prerequisite | Karpathy, Levine | Cut to ~8 days, skip Word2Vec, compress seq2seq |
| C4 | **Add information theory + compression** — the mathematical foundation of everything is absent | Sutskever | Add 1 day: Shannon entropy, cross-entropy, compression ↔ prediction equivalence |
| C5 | **Add π₀.₅ and GR00T N1** — curriculum frozen at mid-2024 | Finn | Update VLA survey to include 2025-2026 frontier models |

## HIGH Findings (Should Fix)

| # | Finding | Experts | Action |
|---|---------|---------|--------|
| H1 | **3D vision completely missing** — no depth estimation, point clouds, or NeRF for a robotics engineer | Li | Add 2 days: Depth Anything, point cloud transformers, 3D scene representations |
| H2 | **Hybrid VLA + classical control underweighted** — 1 day for the most realistic deployment path | Levine | Expand to 3-4 days with ROS2 integration exercise |
| H3 | **Diffusion foundations too compressed** — 1 day for DDPM is insufficient | Karpathy, Finn | Expand to 3 days (saved from Phase I) |
| H4 | **Video understanding gets only 1 day** — VLAs process frame sequences | Li | Split to 2 days with implementation |
| H5 | **Training stability not taught** — student will hit NaN losses with no tools to debug | Karpathy | Add "Training Stability Cookbook" before first real training |
| H6 | **LLM application block (RAG/tools/assistant) is ~7 days that don't serve VLA goal** | Sutskever, Levine | Compress to 1-2 days, redistribute to robot learning |
| H7 | **Deployment only gets 1 day** — latency profiling, safety layers, ROS2 integration missing | Levine | Expand to 3 days |
| H8 | **No scaling laws deep dive** — treated as "plot some curves" | Sutskever | Add 1-2 days: why power laws, emergence debate, compression interpretation |
| H9 | **nanoGPT used passively** — should be an ablation laboratory | Karpathy | Restructure as systematic ablation exercises |

## MEDIUM Findings (Consider Fixing)

| # | Finding | Experts |
|---|---------|---------|
| M1 | T5/encoder-decoder models missing | Karpathy |
| M2 | Action tokenization (Day 90) needs 2 days, not 1 | Karpathy, Finn |
| M3 | Add ACT (Action Chunking with Transformers) and Behavior Transformers | Finn |
| M4 | Add PaLI as bridge to understanding RT-2 | Li |
| M5 | Capstone project too compressed (3 days for 2 approaches) | Karpathy, Levine |
| M6 | Missing robot policy evaluation methodology | Levine, Finn |
| M7 | Add "STOP AND REFLECT" markers at 5 key aha moments | Sutskever |
| M8 | Missing "timeless principles" summaries per phase | Sutskever |
| M9 | Add debugging learned policies day | Finn |
| M10 | Make LeRobot the primary implementation framework for Phases VI-VII | Finn |

---

## Proposed Time Reallocation

| Source (Cut/Compress) | Days Saved | Destination | Expert |
|---|---|---|---|
| Phase I: RNNs, Word2Vec, overlong seq2seq | **5 days** | Diffusion foundations (+2), RL foundations (+3) | Karpathy, Levine |
| LLM app engineering: RAG, tools, assistant project | **5 days** | Hybrid architectures (+3), data collection (+2) | Sutskever, Levine |
| Phase III Week 8: multi-turn, LLM security | **2 days** | 3D vision (+2) | Karpathy, Li |
| BERT/embeddings deep dive | **1 day** | Video understanding (+1) | Levine, Li |
| **Total rebalanced** | **~13 days** | Redistributed to robot learning + foundations | |

---

## Key Expert Quotes

**Karpathy:** "The biggest risk is the student running out of gas in Phases VI-VII because too much time was spent on historical foundations in Phase I."

**Chelsea Finn:** "Data is the single biggest bottleneck in robot learning, and this curriculum treats it as a paragraph."

**Fei-Fei Li:** "The curriculum treats vision as a self-contained discipline rather than as perception in service of action."

**Ilya Sutskever:** "The student who completes this curriculum as-is will be able to fine-tune a VLA model and deploy it. But they will not understand *why* it works."

**Sergey Levine:** "The curriculum reflects a common bias in the VLA field: over-indexing on the 'V' and 'L' and under-indexing on the 'A.' The student's existing control systems knowledge is a massive advantage that this curriculum currently underexploits."

---

## 5 Aha Moments to Scaffold (Sutskever)

1. **Week 2:** "Attention is just a soft dictionary lookup"
2. **Week 2:** "The transformer is embarrassingly simple"
3. **Week 5:** "Scaling is all you need — same arch, more data/compute, just keeps getting better"
4. **Week 9:** "Images ARE sequences — and the transformer doesn't care"
5. **Week 14:** "Actions are just another token type" (RT-2 insight)

---

## Additional Papers to Add

| Paper | Expert | Phase |
|-------|--------|-------|
| π₀.₅ (Physical Intelligence, 2025) | Finn | VII |
| GR00T N1 (NVIDIA, 2025) | Finn | VII |
| PaLM-E (Driess et al., 2023) | Karpathy | VII |
| ACT (Zhao et al., 2023) | Finn | VI |
| Behavior Transformers (Shafiullah et al., 2022) | Finn | VI |
| Decision Transformer (Chen et al., 2021) | Finn | VI |
| Florence-2 (Microsoft) | Li | IV |
| Depth Anything v2 | Li | IV |
| SAM 2 | Li | IV |
| CoCa (Yu et al., 2022) | Li | V |
| GR-2 (Cheang et al., 2024) | Karpathy | VII |

---

## Decision Required

- [x] Accept review and revise curriculum ✅ (CURRICULUM.md v2 + 00-learning-plan.md v2 — 2026-04-28)
- [ ] Accept partially — pick which findings to address
- [ ] Keep current curriculum as-is
