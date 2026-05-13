# Day 42: Phase III Capstone Day 1 — Robotics Assistant with LoRA + RAG

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"The capstone isn't about building something perfect — it's about integrating everything you've learned."*

## Navigation
- **Previous:** [Day 41: LLM for Robotics](day-41-llm-for-robotics.md)
- **Next:** [Day 43: Phase III Capstone Day 2](../week-07/day-43-phase3-capstone-2.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Capstone Overview

Build a **robotics assistant** that combines:
1. **LoRA fine-tuning** on robot domain data (Days 32, 35)
2. **RAG** over technical documentation (Day 40)
3. **Function calling** for robot status queries (Day 40)
4. **Evaluation** with a custom benchmark (Day 36)

```
Architecture:
┌────────────────────────────────────────────────────┐
│                Robotics Assistant                    │
│                                                      │
│  User Query ──→ Intent Router                        │
│                    │                                 │
│          ┌─────────┼─────────┐                       │
│          ▼         ▼         ▼                       │
│     Knowledge   Command    Status                   │
│     Question    Execution   Query                   │
│          │         │         │                       │
│     RAG Search  Parse +   Tool Call                  │
│          │      Validate     │                       │
│          ▼         ▼         ▼                       │
│     LoRA-tuned LLM (TinyLlama + robotics adapter)   │
│          │                                           │
│          ▼                                           │
│     Response + Sources/Actions                       │
└────────────────────────────────────────────────────┘
```

---

## Implementation (120 min)

### Step 1: Prepare Training Data (30 min)

```python
"""
Day 42 Capstone: Robotics Assistant with LoRA + RAG.
Step 1 — Prepare instruction-following dataset.
"""
from datasets import Dataset

def create_robotics_sft_dataset() -> Dataset:
    """Create instruction dataset for robotics domain SFT."""
    examples = [
        # Navigation
        {
            "instruction": "How does the robot handle obstacles during navigation?",
            "response": "The robot uses a two-layer planning approach: a global "
                        "planner (A* on the occupancy grid) computes the overall "
                        "path, while a local planner (DWA - Dynamic Window "
                        "Approach) handles real-time obstacle avoidance. LiDAR "
                        "scans at 10Hz detect obstacles, and the local planner "
                        "adjusts velocity commands to maintain a safety margin "
                        "of at least 0.3m from any obstacle.",
        },
        # Error diagnosis
        {
            "instruction": "What causes NAV_ESTIMATED_STATE_NOT_FINITE?",
            "response": "This error occurs when the navigation estimator "
                        "produces NaN or Inf values. Common causes: 1) IMU "
                        "sensor failure producing invalid readings, 2) Wheel "
                        "encoder signal loss causing division by zero in "
                        "odometry, 3) Sensorbar SPI communication timeout "
                        "propagating stale data. Check /diagnostics for the "
                        "first sensor to report errors in the timeline.",
        },
        # Battery/BEC
        {
            "instruction": "Describe the battery exchange process.",
            "response": "The AMR approaches the BEC station using IR-guided "
                        "docking. Process: 1) Alignment using IR beacon pair, "
                        "2) Final approach at 0.05m/s, 3) Dock confirmation "
                        "via contact sensors, 4) Mechanical latch release of "
                        "depleted battery, 5) Conveyor extraction (15s), "
                        "6) Fresh battery insertion and electrical verification "
                        "(voltage + CAN handshake), 7) Undock. Total: ~45s.",
        },
        # Sensorbar
        {
            "instruction": "How to troubleshoot sensorbar stiction issues?",
            "response": "Sensorbar stiction (encoder lockup) symptoms: constant "
                        "readings despite robot motion, large jumps when "
                        "unsticking. Diagnosis: 1) Check /sensorbar/raw_data "
                        "for flat sections, 2) Inspect physical sensor for "
                        "debris or moisture, 3) Verify SPI clock at 10MHz "
                        "with oscilloscope, 4) Check firmware version ≥ v1.2.4. "
                        "Resolution: clean sensor surface, replace if worn.",
        },
        # Fleet management
        {
            "instruction": "How do you optimize robot fleet throughput?",
            "response": "Fleet throughput optimization: 1) Minimize empty travel "
                        "by assigning tasks nearest to each robot's current "
                        "position (nearest-neighbor heuristic), 2) Stagger "
                        "charging schedules to keep ≥80% of fleet active, "
                        "3) Use zone-based traffic management to prevent "
                        "congestion at intersections, 4) Monitor via OWM "
                        "(OKS World Model) for real-time bin/station status, "
                        "5) Set velocity limits by zone density.",
        },
        # Safety
        {
            "instruction": "What happens during an emergency stop?",
            "response": "Guardian node triggers e-stop sequence: 1) Publish "
                        "zero velocity to /cmd_vel immediately, 2) Engage "
                        "motor brakes within 100ms, 3) Activate warning lights "
                        "and buzzer, 4) Log timestamp + trigger reason to "
                        "/emergency_stop_log, 5) Notify fleet management "
                        "system. Robot requires manual intervention to resume. "
                        "Common triggers: obstacle within 0.15m, IMU impact "
                        "detection, software watchdog timeout, physical e-stop.",
        },
    ]
    return Dataset.from_list(examples)


# Knowledge base for RAG
KNOWLEDGE_BASE = [
    {
        "text": "The OKS robot uses differential drive kinematics with two "
                "powered wheels (200mm diameter) and two passive caster wheels. "
                "Maximum linear velocity: 1.5 m/s. Maximum angular velocity: "
                "1.0 rad/s. The drive controller runs at 100Hz.",
        "source": "hardware_spec",
    },
    {
        "text": "The navigation estimator fuses IMU (100Hz), wheel encoders "
                "(50Hz), and LiDAR-based localization (10Hz) using an Extended "
                "Kalman Filter. The estimator state includes position (x,y), "
                "heading (θ), and velocities (vx, vθ).",
        "source": "nav_spec",
    },
    {
        "text": "Guardian node health monitoring thresholds: CPU temperature > "
                "85°C triggers warning, > 95°C triggers shutdown. Battery "
                "voltage < 22.0V triggers low battery alert, < 20.5V forces "
                "return to charging station. Network latency > 500ms triggers "
                "autonomous mode.",
        "source": "guardian_spec",
    },
    {
        "text": "The sensorbar communicates via SPI at 10MHz with the main "
                "controller. It provides wheel odometry, floor detection, and "
                "cliff sensing. Firmware supports self-test mode activated "
                "via diagnostic service call /sensorbar/self_test.",
        "source": "sensorbar_spec",
    },
    {
        "text": "OKS World Model (OWM) maintains the warehouse digital twin: "
                "bin states (empty/occupied/reserved), station statuses, tile "
                "traversability, and flow assignments. REST API at /api/v1/owm/.",
        "source": "owm_spec",
    },
]

# Evaluation benchmark
EVAL_QUESTIONS = [
    {
        "question": "What sensor fusion algorithm does the navigation use?",
        "expected_keywords": ["EKF", "Extended Kalman", "IMU", "encoder", "LiDAR"],
        "category": "knowledge",
    },
    {
        "question": "The robot shows sensorbar SPI errors. What should I check first?",
        "expected_keywords": ["SPI", "wiring", "firmware", "10MHz", "debris"],
        "category": "diagnosis",
    },
    {
        "question": "How do I move robot OKS-42 to the charging station?",
        "expected_keywords": ["navigate", "charging", "command"],
        "category": "command",
    },
]


if __name__ == "__main__":
    dataset = create_robotics_sft_dataset()
    print(f"SFT Dataset: {len(dataset)} examples")
    print(f"Knowledge Base: {len(KNOWLEDGE_BASE)} documents")
    print(f"Eval Benchmark: {len(EVAL_QUESTIONS)} questions")

    # Preview
    print("\nSample instruction:")
    print(f"  Q: {dataset[0]['instruction']}")
    print(f"  A: {dataset[0]['response'][:100]}...")
```

### Step 2: LoRA Fine-Tuning (30 min)

```python
"""
Step 2 — Fine-tune with LoRA (refer to Day 32 + Day 35 patterns).
"""
# Use the SFTTrainer pattern from Day 32:
# 1. Load TinyLlama with QLoRA config (Day 35)
# 2. Apply LoRA to q_proj, k_proj, v_proj, o_proj with r=16
# 3. Format with ChatML template
# 4. Train for 3 epochs with lr=2e-4
# 5. Save adapter weights
#
# See day-32-supervised-finetuning.md and day-35-lora-finetuning.md
# for the complete training code.
```

### Step 3: RAG Integration (30 min)

```python
"""
Step 3 — Add RAG over robot documentation (refer to Day 40 patterns).
"""
# 1. Index KNOWLEDGE_BASE using SimpleEmbedder + VectorStore from Day 40
# 2. For knowledge questions: retrieve top-3 docs, augment prompt
# 3. For commands: bypass RAG, use command parser
# 4. For diagnosis: retrieve relevant docs + use fine-tuned model
```

### Step 4: Evaluation (30 min)

```python
"""
Step 4 — Evaluate the assistant.
"""
def evaluate_response(response: str, expected_keywords: list[str]) -> dict:
    """Simple keyword-based evaluation."""
    response_lower = response.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in response_lower]
    return {
        "keyword_recall": len(hits) / len(expected_keywords),
        "matched": hits,
        "missed": [kw for kw in expected_keywords if kw not in hits],
    }

# Run evaluation across all EVAL_QUESTIONS
# Compare: base model vs LoRA-tuned vs LoRA+RAG
```

---

## Exercise (Remaining time)

### E42.1 — Full Pipeline Assembly
Combine all components into a single `RoboticsAssistant` class:
1. `__init__`: load model, RAG index, command parser
2. `query(text) → response`: route to appropriate handler
3. `evaluate(benchmark) → scores`: run full evaluation

### E42.2 — Prepare for Day 43
Document what works and what doesn't:
- Which question categories does the assistant handle best?
- Where does RAG improve over pure fine-tuning?
- What failure modes did you observe?

---

## Key Takeaways

1. **Integration is harder than components** — making LoRA, RAG, and function calling work together requires careful routing
2. **Domain-specific SFT data** is the highest-leverage improvement for robotics applications
3. **RAG provides freshness** — the fine-tuned model can't know about new robot configurations
4. **Evaluation must be multi-dimensional** — keyword recall, response quality, safety, latency

---

## Connection to the Thread

This capstone integrates nearly every concept from Phase III into a practical robotics tool. Tomorrow we evaluate, compare against baselines, and refine. The full pipeline (domain SFT + RAG + tool use) is exactly how production robot AI assistants are built — you're implementing the architecture used by companies deploying LLMs for fleet management.

---

## Further Reading

- [TRL: Transformer Reinforcement Learning Library](https://huggingface.co/docs/trl) — Training infrastructure
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — Production RAG patterns
- [Evaluation of LLM-based Robotic Assistants](https://arxiv.org/abs/2404.07997) — Benchmarking framework
