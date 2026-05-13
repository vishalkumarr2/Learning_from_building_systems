# Day 105: Stop & Reflect #6 — VLA Architectures & Deployment

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2 hours**
> *"You've surveyed 8+ VLA architectures, training recipes, sim-to-real, and deployment. Before the final capstone, consolidate."*

## Navigation
- **Previous:** [Day 104: Hybrid Architectures Day 2](day-104-hybrid-2.md)
- **Next:** [Day 106: World Models for Robot Control](../week-16/day-106-world-models.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## VLA Architecture Map

Fill in this comparison from memory:

```
Model       Year  Params  Action Rep   VLM Backbone  Key Innovation
─────────── ───── ─────── ──────────── ───────────── ──────────────────
RT-1        2022  ___M    ___________  ___________   ___________________
RT-2        2023  ___B    ___________  ___________   ___________________
Octo        2024  ___M    ___________  ___________   ___________________
OpenVLA     2024  ___B    ___________  ___________   ___________________
π₀          2024  ___B    ___________  ___________   ___________________
π₀.5        2025  ___B    ___________  ___________   ___________________
GR-2        2024  ___B    ___________  ___________   ___________________
GROOT N1    2025  ___B    ___________  ___________   ___________________
```

---

## Concept Check (30 min)

### 6 Questions — Answer Without Notes

**Q1.** Draw the RT-2 pipeline from image input to action output. Label each component.

**Q2.** Explain why Octo uses blockwise causal attention. What would go wrong with full attention?

**Q3.** π₀ uses flow matching instead of diffusion. Name 3 practical advantages for robot action generation.

**Q4.** In coarse-to-fine action prediction, the coarse head uses 64 bins. For $\Delta x \in [-5\text{cm}, 5\text{cm}]$, what's the coarse resolution? Why is refinement needed for sub-millimeter tasks?

**Q5.** You have a 7B VLA running at 3 Hz. You need 20 Hz for dexterous manipulation. List 3 techniques from Day 104 to close this gap.

**Q6.** Your deployed VLA achieves 85% success in the lab but 60% in a new kitchen. List 3 potential causes and the corresponding fix for each.

---

## Architecture Selection Flowchart (30 min)

Build a decision tree for choosing the right VLA design:

```
START: What's your task?
  │
  ├── Simple pick-and-place
  │     └── How much real data?
  │           ├── >10K episodes → RT-1 style (small, fast)
  │           └── <1K episodes → OpenVLA + LoRA fine-tune
  │
  ├── Dexterous manipulation
  │     └── How many DOF?
  │           ├── 7 DOF arm → π₀ (flow matching)
  │           └── >14 DOF (bimanual/hand) → π₀ (long chunk)
  │
  ├── Long-horizon tasks
  │     └── Need language reasoning?
  │           ├── Yes → π₀.5 (plan + act)
  │           └── No → Hierarchical controller
  │
  ├── Multi-robot deployment
  │     └── Same task?
  │           ├── Yes → Octo (multi-embodiment)
  │           └── No → OpenVLA + per-robot LoRA
  │
  └── Novel objects / scenes
        └── Need reasoning about objects?
              ├── Yes → RT-2 / OpenVLA (VLM reasoning)
              └── No → Domain randomization + BC
```

---

## Synthesis Exercise (30 min)

### Design Your Ideal VLA

Given everything you've learned, design a VLA for this scenario:

**Scenario:** A warehouse robot that must:
- Pick diverse products from shelves (novel objects)
- Follow natural language instructions from workers
- Operate at 15+ Hz control frequency
- Work across 3 different robot models
- Improve from deployment corrections

Document your design:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| VLM backbone | | |
| Vision encoder | | |
| Action representation | | |
| Training data | | |
| Transfer approach | | |
| Inference optimization | | |
| Safety mechanisms | | |
| Adaptation strategy | | |

---

## Reflection Prompt (30 min)

**Write 300+ words:** "What surprised me most about the VLA landscape, and what do I think the next breakthrough will be?"

Consider:
- The rapid pace of progress (RT-1 to π₀.5 in 3 years)
- The convergence on hybrid designs
- The open-source movement (Octo, OpenVLA)
- What's still missing (generalization, robustness, speed)
- Your prediction for the next 2 years

---

## Weeks 14-15 Review Checklist

- [ ] Can explain the architecture of RT-1, RT-2, Octo, OpenVLA, π₀, π₀.5
- [ ] Understand tokenized vs continuous vs flow matching action representations
- [ ] Know the three-stage VLA training recipe (align → co-fine-tune → specialize)
- [ ] Can design a domain randomization strategy for sim-to-real transfer
- [ ] Understand teacher-student distillation for deployable policies
- [ ] Can choose between VLA architectures for a given deployment scenario
- [ ] Know the deployment stack: safety + confidence + recycling + adaptation

---

## Connection to the Thread

> *Final week begins. Day 106: world models for robot control — can a VLA that predicts the future make better decisions? Days 107-109: deployment deep dive. Days 110-112: the final capstone, where you design, build, and evaluate a complete VLA system. Then: curriculum complete.*

---

## Further Reading

- Re-read the original papers for any VLA where you're uncertain
- Preview: Ha & Schmidhuber (2018), "World Models" — foundation for Day 106
