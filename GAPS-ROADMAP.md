# Learning Gaps Roadmap
### Track completion status and remaining work

---

## Status Overview

```
TRACK                           STATUS
──────────────────────────────  ────────────────────────────────
electronics/                    ✅ COMPLETED — 7 lessons + 7 exercises
zephyr/                         ✅ COMPLETED — 17 reference docs, 5 study notes, 12 deep-dive labs, 11 exercises
cpp-advanced/                   ✅ COMPLETED — 15 of 18 chapters (09, 10 are exercise-only stubs)
navigation-estimator/           ✅ COMPLETED — 6 lessons + 5 exercises
ros2-handson/                   ✅ COMPLETED — 5 lessons + 8 exercises
python-oks/                     ✅ COMPLETED — 3 lessons + 3 exercises
control-systems/                ✅ COMPLETED — 10 lessons, 3 debugging, 8 exercises, 3 senior Q&A
optimization/                   ✅ COMPLETED — 10 lessons + 9 exercises + code implementations
llm-to-vla/                     ✅ COMPLETED — 16 study notes, 11 exercises, 7 projects, 16 weekly schedules
```

---

## Track 1: Navigation & State Estimation — ✅ COMPLETED
**Folder:** `learn/navigation-estimator/`

6 core lessons (dead reckoning → Ceres sensorbar deep dive), 5 exercises.

---

## Track 2: ROS 2 Hands-On — ✅ COMPLETED
**Folder:** `learn/ros2-handson/`

5 lessons (nodes/topics → holonomic motion), 8 exercises including search algorithms
and interview-prep traps.

---

## Track 3: Python / OKS Scripting — ✅ COMPLETED
**Folder:** `learn/python-oks/`

3 lessons (types → time-series analysis), 3 exercises.

---

## Track 4: Control Systems — ✅ COMPLETED
**Folder:** `learn/control-systems/`

10 core lessons (PID fundamentals → advanced control), 3 debugging sessions (motor oscillation,
integral windup, cmd_vel gap), 8 exercise sets, and 3 senior interview question banks across
control theory, embedded control, and system integration.

---

## Track 5: Zephyr Deep-Dive — ✅ COMPLETED
**Folder:** `learn/zephyr/deep-dive/`

12 hardware lab guides (logic analyzer → HardFault decode), plus 5 study notes and
11 exercises covering foundations through Jetson RT profiling.

---

## Track 6: Linux RT Internals — 🟢 Partially covered
**Status:** Covered in `cpp-advanced/03-rt-linux-programming/`
**Action:** Extend with `perf`, `ftrace`, `/proc/latency_stats`, PREEMPT_RT patch model
when needed.

---

## Track 7: Optimization — ✅ COMPLETED
**Folder:** `learn/optimization/`

10 core lessons (foundations → game theory), 9 exercise sets, Python/C++ code
implementations, and weekly schedules covering Lie groups and game theory.

---

## Track 8: LLM → VLA — ✅ COMPLETED
**Folder:** `learn/llm-to-vla/`

16 study notes (DL foundations → deployment), 11 exercise sets, 7 milestone projects,
and 16 weekly day-by-day schedules. Bridges control/estimation/ROS knowledge with
modern AI for robotics.

---

## Remaining Gaps

| Gap | Priority | Action |
|-----|----------|--------|
| `cpp-advanced/09-hardware-lessons` | LOW | Stub — exercises only, no notes.md. Add content or remove |
| `cpp-advanced/10-safety-lessons` | LOW | Stub — exercises only, no notes.md. Add content or remove |
| Linux RT internals | LOW | Extend `cpp-advanced/03` when needed |

## Quick-Start: Next Session

Start here → [navigation-estimator/00-learning-plan.md](navigation-estimator/00-learning-plan.md)

**Day 1 goal (2 hrs):** Read `01-dead-reckoning.md` and be able to derive
`(x', y', θ')` from two encoder counts for a differential-drive robot.

This directly ties to `doPrediction()` in OKS estimator — you'll recognize every variable.
