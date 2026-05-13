# Day 110: Final Capstone — Day 1: System Design

> **Phase VII — VLAs: Architecture to Deployment | Week 16 | 3 hours**
> *"Design a complete VLA system. Not a toy — a production-viable architecture with every component specified."*

## Navigation
- **Previous:** [Day 109: Fleet Management](day-109-deployment-3.md)
- **Next:** [Day 111: Final Capstone — Day 2](day-111-final-capstone-2.md)
- **Week:** [Week 16 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## The Final Capstone

Over three days, you'll design, implement, and evaluate a complete VLA system.

**Scenario:** Design a VLA for a kitchen robot that can:
- Follow natural language cooking instructions
- Manipulate diverse kitchen objects (utensils, containers, food)
- Operate safely around humans
- Learn from human corrections during deployment
- Run at ≥15 Hz on edge hardware (Jetson Orin)

---

## Day 1: System Design (3 hours)

### Part 1: Architecture Specification (60 min)

Complete this design document:

```
VLA SYSTEM DESIGN DOCUMENT
═══════════════════════════

1. MODEL ARCHITECTURE
   1.1 VLM Backbone:
       Model: _______________
       Parameters: ___________
       Why this choice: ______
   
   1.2 Vision Encoder:
       Architecture: __________
       Input resolution: ______
       Output dim: ____________
       Freeze strategy: _______
   
   1.3 Action Representation:
       Type: [tokens / continuous / flow / hybrid]
       Dimension: _____________
       Chunk size: ____________
       Frequency: _____________
       Why this choice: _______
   
   1.4 Action Head:
       Architecture: __________
       Parameters: ____________
       Conditioning: __________

2. TRAINING PIPELINE
   2.1 Data:
       Robot data source: ____
       Web data source: ______
       Robot/web ratio: ______
       Total training examples: ___
       Data augmentation: _____
   
   2.2 Training Schedule:
       Stage 1 (align): ______
       Stage 2 (co-fine-tune): ___
       Stage 3 (specialize): ___
       Total compute: ________
   
   2.3 Multi-task Strategy:
       Task list: ____________
       Sampling temperature: __
       Loss weighting: _______

3. DEPLOYMENT STACK
   3.1 Hardware:
       Compute: ______________
       Sensors: ______________
       Robot: ________________
   
   3.2 Inference:
       Quantization: _________
       Target latency: _______
       Throughput: ___________
       Caching strategy: _____
   
   3.3 Safety:
       Layer 1 (physical): ___
       Layer 2 (hardware): ___
       Layer 3 (motion): _____
       Layer 4 (behavior): ___
       Layer 5 (task): _______
   
   3.4 Monitoring:
       Key metrics: __________
       Alert thresholds: _____
       Dashboard: ____________

4. ADAPTATION
   4.1 Online learning:
       Strategy: _____________
       Buffer size: __________
       Fine-tune frequency: __
   
   4.2 Fleet management:
       Rollout strategy: _____
       A/B testing: __________
       Rollback plan: ________
```

### Part 2: Component Diagram (30 min)

Draw a detailed system diagram showing data flow:

```
Fill in:

Camera ──→ [___________] ──→ [___________] ──→ [___________]
                                                      │
Proprio ──→ [___________] ──────────────────────→ [___________]
                                                      │
Language ──→ [___________] ──────────────────────→ [___________]
                                                      │
                                                      ▼
                                               [___________]
                                                      │
                                                      ▼
                                               [___________]
                                                      │
                                                      ▼
                                              Robot Controller
```

### Part 3: Risk Analysis (30 min)

Identify the top 5 risks and mitigations:

| Rank | Risk | Probability | Impact | Mitigation |
|------|------|-------------|--------|------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Part 4: Trade-off Analysis (30 min)

For each design decision, document:

**Decision 1: Action representation**
- Option A: Tokenized (256 bins)
  - Pro: ________, ________
  - Con: ________, ________
- Option B: Flow matching
  - Pro: ________, ________
  - Con: ________, ________
- Option C: Hybrid coarse-to-fine
  - Pro: ________, ________
  - Con: ________, ________
- **Chosen:** _____ because _____

**Decision 2: Model size**
- Option A: 3B VLM + LoRA
- Option B: 7B VLM frozen + small action head
- Option C: 300M end-to-end
- **Chosen:** _____ because _____

**Decision 3: Training data strategy**
- Option A: Real-only (expensive but high quality)
- Option B: Sim-to-real (cheap but gap)
- Option C: Web pretrain + real fine-tune
- **Chosen:** _____ because _____

### Part 5: Evaluation Plan (30 min)

Define success metrics:

```
Offline Metrics:
  - Action prediction loss: target < ___
  - Action accuracy (within 5mm): target > ___%
  - Language grounding accuracy: target > ___%

Online Metrics (simulation):
  - Task success rate: target > ___%
  - Mean episode length: target < ___ steps
  - Safety violation rate: target < ___%

Online Metrics (real):
  - Task success rate: target > ___%
  - Human correction rate: target < ___%
  - Control frequency: target ≥ ___ Hz
  - Time to first failure: target > ___ minutes

Generalization Tests:
  - Novel objects success: target > ___%
  - Novel instructions success: target > ___%
  - Distractor robustness: target > ___%
```

---

## Deliverables for Day 1

By end of session, you should have:
- [ ] Complete architecture specification (all fields filled)
- [ ] System diagram with all components and data flows
- [ ] Risk analysis with mitigations
- [ ] Documented trade-off rationale for 3+ key decisions
- [ ] Evaluation plan with concrete numeric targets

---

## Connection to the Thread

> *Architecture designed. Tomorrow (Day 111): implement the core components — the VLA model, training loop, and basic evaluation. Day 112: full integration, evaluation, and final reflection.*
