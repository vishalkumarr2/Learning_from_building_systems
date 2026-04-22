# Learning Gaps Roadmap
### What's missing, why it matters, and in what order to build it

---

## The Gap Analysis

All seven tracks have been written. The table below shows each track's current status.
Future work: Linux RT internals (extend cpp-advanced/03) and deeper hardware exercises once
the STM32 + logic analyzer are in hand.

```
EXISTING (COMPLETE)                     STATUS
────────────────────────────────────    ────────────────────────────────────────
electronics/    (hardware theory)       🟢 7 lessons + exercises
zephyr/         (firmware + deep-dive)  🟢 17 tutorials + 12 deep-dives
cpp-advanced/   (C++ depth)             🟢 18 modules + exercises
control-systems/ (PID → advanced)       🟢 10 lessons + exercises
navigation-estimator/ (EKF/IMU)         🟢 5 lessons + 5 exercises
ros2-handson/   (nodes/TF/Nav2)         🟢 3 lessons + 4 exercises
python-scripting/ (typing/test/polars)  🟢 3 lessons + 3 exercises
```

---

## Track 1: Navigation & State Estimation
**Priority: HIGHEST**
**Why now:** You investigate `ESTIMATOR_STATE_INVALID`, slip detection, line-sensor covariance
blow-up, and EKF divergence tickets daily. You know the robot code deeply (repo memory) but
lack the mathematical theory to immediately see *why* a covariance grows / explodes / converges.

**Goal:** Be able to look at any estimator log and reason from first principles about whether
the observed covariance trajectory is expected or anomalous.

**Timeline:** 5 weeks · 3–4 hrs/week · no hardware needed

**Status:** � Complete — 5 lessons + 5 exercises written
**Folder:** `learn/navigation-estimator/`

| Week | Topic | Practical Relevance |
|------|-------|---------------|
| 1 | Dead-reckoning & wheel odometry math | `predict()` noise model |
| 2 | EKF theory: prediction step, covariance propagation | Covariance INF = slipped/collision |
| 3 | EKF measurement update: Mahalanobis, innovation gating | Line-Sensor update in `update()` |
| 4 | IMU fusion, gyro bias, line-sensor as line-constraint measurement | Theta update + gyro correction |
| 5 | Diagnosing estimator failures from logs/bags | Apply to real tickets |

**Files to create:**
```
navigation-estimator/
├── 00-learning-plan.md        ← dependency graph, study order, checkpoints
├── 01-dead-reckoning.md       ← odometry math, unicycle model, arc integration
├── 02-kalman-filter.md        ← KF → EKF derivation, predict/update equations
├── 03-measurement-models.md   ← line sensors as constraints, Mahalanobis gating
├── 04-imu-fusion.md           ← gyro integration, bias estimation, line-sensor theta
├── 05-failure-modes.md        ← covariance blow-up patterns, slip vs collision vs delocalize
└── exercises/
    ├── 01-odometry-math.md
    ├── 02-kalman-1d.md        ← build a 1D Kalman filter by hand
    ├── 03-ekf-unicycle.md     ← implement EKF for a unicycle robot in Python
    ├── 04-log-diagnosis.md    ← given a covariance trajectory, identify the failure
    └── 05-robot-specific.md     ← navigation estimator parameter tuning exercises
```

---

## Track 2: ROS2 Hands-On Exercises
**Priority: HIGH**
**Why:** `zephyr/study-notes/05-jetson-ros2.md` (1524 lines) is theory-only. You work with
ROS2 topics, nodes, tf2, nav2 every day but haven't built a systematic exercise set.

**Goal:** Be able to write, debug, and tune any ROS2 node from scratch. Understand DDS/QoS
well enough to explain frame drop bugs.

**Timeline:** 3 weeks · 3–4 hrs/week

**Status:** � Complete — 3 lessons + 4 exercises written
**Folder:** `learn/ros2-handson/`

| Week | Topic | Practical Relevance |
|------|-------|---------------|
| 1 | Nodes, topics, services, actions, lifecycle | AMR node architecture |
| 2 | tf2 transforms, time sync, QoS profiles | TF lookup failures in nav logs |
| 3 | Nav2 BT architecture, costmaps, planner/controller | RCS navigation stack |

**Files to create:**
```
ros2-handson/
├── 00-learning-plan.md
├── 01-nodes-topics-actions.md
├── 02-tf2-time-qos.md
├── 03-nav2-architecture.md
└── exercises/
    ├── 01-pub-sub-timer.md
    ├── 02-tf2-broadcaster.md
    ├── 03-action-server.md
    └── 04-nav2-custom-plugin.md
```

---

## Track 3: Python / Robot Scripting
**Priority: MEDIUM**
**Why:** You write Python scripts daily (RCA analysis, KB tools, log parsing) but there's no
structured skill-building. Gaps: async patterns, dataclass typing, pandas/polars for log analysis,
test coverage for scripts.

**Goal:** Write analysis scripts that are testable, typed, and maintainable — not just
"it works on my machine."

**Timeline:** 3 weeks · 2–3 hrs/week

**Status:** � Complete — 3 lessons + 3 exercises written
**Folder:** `learn/python-scripting/`

| Week | Topic | Practical Relevance |
|------|-------|---------------|
| 1 | Type hints, dataclasses, Pydantic for config validation | `session_tracker.py`, `knowledge_search.py` |
| 2 | pytest, fixtures, mocking for CLI scripts | Script tests in `tests/` |
| 3 | pandas/polars for time-series log analysis | Bag CSV analysis patterns |

**Files to create:**
```
python-scripting/
├── 00-learning-plan.md
├── 01-types-dataclasses.md
├── 02-testing-cli-scripts.md
├── 03-timeseries-analysis.md
└── exercises/
    ├── 01-type-annotations.py
    ├── 02-pytest-fixtures.py
    └── 03-log-analysis-polars.py
```

---

## Track 4: Zephyr Deep-Dive
**Priority: MEDIUM (blocked by hardware)**
**Why:** `zephyr/deep-dive/` exists but is empty. Meant for hardware-stage deep dives once
the STM32 + logic analyzer are in hand.

**Timeline:** Add content incrementally as hardware is acquired (see `zephyr/00-mastery-plan.md` hardware list)

**Status:** � Complete — 12 deep-dives + README written
**Folder:** `learn/zephyr/deep-dive/`

**Files to create (when hardware arrives):**
```
zephyr/deep-dive/
├── 01-first-build-flash-debug.md   ← blinky to shell, GDB, Segger RTT
├── 02-spi-slave-first-frame.md     ← wire up logic analyzer, capture first byte
├── 03-dma-cache-gotchas.md         ← reproduce the D-cache coherency bug on real HW
├── 04-imu-i2c-reads.md             ← ICM-42688 register read via Zephyr I2C API
└── 05-100hz-spi-bridge.md          ← the full capstone
```

---

## Track 5: Linux RT Internals
**Priority: LOW**
**Why:** `cpp-advanced/03-rt-linux-programming` partially covers this (SCHED_FIFO, mutex, priority inversion).
Not a separate track — extend cpp-advanced/03 instead.

**Status:** 🟢 Partially covered in `cpp-advanced/03-rt-linux-programming/`
**Action:** After completing cpp-advanced weeks 1–5, add:
- `perf` and `ftrace` for latency profiling
- `/proc/latency_stats` interpretation
- PREEMPT_RT patch model

---

## Recommended Study Order

All content is written. Suggested study order for the best learning flow:

```
CORE (do these first, in parallel):
│
├── navigation-estimator/   ← highest RCA value, 5 weeks
├── cpp-advanced/           ← 18 modules, work through progressively
│
THEN:
│
├── ros2-handson/           ← builds on navigation-estimator theory
├── python-scripting/       ← can interleave with anything
│
ALWAYS AVAILABLE:
│
├── electronics/            ← reference as needed
├── control-systems/        ← PID → advanced control theory
└── zephyr/                 ← firmware + 12 deep-dives
```

---

## Quick-Start: Next Session

Start here → [navigation-estimator/00-learning-plan.md](navigation-estimator/00-learning-plan.md)

**Day 1 goal (2 hrs):** Read `01-dead-reckoning.md` and be able to derive
`(x', y', θ')` from two encoder counts for a differential-drive robot.

This directly ties to `predict()` in navigation estimator — you'll recognize every variable.
