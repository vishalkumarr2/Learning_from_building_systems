# Learning Gaps Roadmap
### What's missing, why it matters, and in what order to build it

---

## The Gap Analysis

Existing tracks cover **hardware/firmware foundations** (electronics → protocols → Zephyr) and
**advanced C++**. Missing tracks are all on the **software/algorithm side** — which is where
the daily RCA work actually happens.

```
EXISTING                            MISSING
────────────────────────────────    ────────────────────────────────────────
electronics/    (hardware theory)   navigation-estimator/  ← COMPLETED ✅
zephyr/         (firmware)          ros2-handson/           ← COMPLETED ✅
cpp-advanced/   (C++ depth)         python-oks/             ← COMPLETED ✅
navigation-estimator/ (EKF/odom)    control-systems/        ← COMPLETED ✅
ros2-handson/   (ROS2 practice)     optimization/           ← COMPLETED ✅
python-oks/     (scripting)         zephyr/deep-dive/       ← MEDIUM (stub only)
control-systems/ (PID→advanced)     linux-rt/               ← LOW (cpp-advanced/03 partial)
optimization/   (solvers/MPC/SLAM)
```

---

## Track 1: Navigation & State Estimation
**Priority: HIGHEST**
**Status:** ✅ **COMPLETED**
**Folder:** `learn/navigation-estimator/`

| Week | Topic | OKS Relevance |
|------|-------|---------------|
| 1 | Dead-reckoning & wheel odometry math | `doPrediction()` noise model |
| 2 | EKF theory: prediction step, covariance propagation | Covariance INF = slipped/collision |
| 3 | EKF measurement update: Mahalanobis, innovation gating | Sensorbar update in `update()` |
| 4 | IMU fusion, gyro bias, sensorbar as line-constraint measurement | Theta update + gyro correction |
| 5 | Diagnosing estimator failures from logs/bags | Apply to real tickets |

**Files to create:**
```
navigation-estimator/
├── 00-learning-plan.md        ← dependency graph, study order, checkpoints
├── 01-dead-reckoning.md       ← odometry math, unicycle model, arc integration
├── 02-kalman-filter.md        ← KF → EKF derivation, predict/update equations
├── 03-measurement-models.md   ← line sensors as constraints, Mahalanobis gating
├── 04-imu-fusion.md           ← gyro integration, bias estimation, sensorbar theta
├── 05-failure-modes.md        ← covariance blow-up patterns, slip vs collision vs delocalize
└── exercises/
    ├── 01-odometry-math.md
    ├── 02-kalman-1d.md        ← build a 1D Kalman filter by hand
    ├── 03-ekf-unicycle.md     ← implement EKF for a unicycle robot in Python
    ├── 04-log-diagnosis.md    ← given a covariance trajectory, identify the failure
    └── 05-oks-specific.md     ← OKS estimator parameter tuning exercises
```

---

## Track 2: ROS2 Hands-On Exercises
**Priority: HIGH**
**Why:** `zephyr/study-notes/05-jetson-ros2.md` (1524 lines) is theory-only. You work with
ROS2 topics, nodes, tf2, nav2 every day but haven't built a systematic exercise set.

**Goal:** Be able to write, debug, and tune any ROS2 node from scratch. Understand DDS/QoS
well enough to explain frame drop bugs.

**Timeline:** 3 weeks · 3–4 hrs/week

**Status:** 🔴 Not started
**Folder:** `learn/ros2-handson/`

| Week | Topic | OKS Relevance |
|------|-------|---------------|
| 1 | Nodes, topics, services, actions, lifecycle | OKS node architecture |
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

## Track 3: Python / OKS Scripting
**Priority: MEDIUM**
**Why:** You write Python scripts daily (RCA analysis, KB tools, log parsing) but there's no
structured skill-building. Gaps: async patterns, dataclass typing, pandas/polars for log analysis,
test coverage for scripts.

**Goal:** Write analysis scripts that are testable, typed, and maintainable — not just
"it works on my machine."

**Timeline:** 3 weeks · 2–3 hrs/week

**Status:** 🔴 Not started
**Folder:** `learn/python-oks/`

| Week | Topic | OKS Relevance |
|------|-------|---------------|
| 1 | Type hints, dataclasses, Pydantic for config validation | `rca_session.py`, `kb_search.py` |
| 2 | pytest, fixtures, mocking for CLI scripts | Script tests in `tests/` |
| 3 | pandas/polars for time-series log analysis | Bag CSV analysis patterns |

**Files to create:**
```
python-oks/
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

## Track 4: Control Systems
**Priority: HIGH**
**Status:** ✅ **COMPLETED**
**Folder:** `learn/control-systems/`

10 core lessons (PID fundamentals → advanced control), 3 debugging sessions (motor oscillation,
integral windup, cmd_vel gap), 8 exercise sets, and 30 senior interview questions across
control theory, embedded control, and system integration. Covers MCU inner loop (STM32, 10 kHz,
fixed-point) through Jetson outer loop (50 Hz, floating-point, ROS2 Nav2).

---

## Track 5: Zephyr Deep-Dive
**Priority: MEDIUM (blocked by hardware)**
**Why:** `zephyr/deep-dive/` exists but is empty. Meant for hardware-stage deep dives once
the STM32 + logic analyzer are in hand.

**Timeline:** Add content incrementally as hardware is acquired (see `zephyr/00-mastery-plan.md` hardware list)

**Status:** 🟡 Stub created, blocked on hardware
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

## Track 6: Linux RT Internals
**Priority: LOW**
**Why:** `cpp-advanced/03-rt-linux-programming` partially covers this (SCHED_FIFO, mutex, priority inversion).
Not a separate track — extend cpp-advanced/03 instead.

**Status:** 🟢 Partially covered in `cpp-advanced/03-rt-linux-programming/`
**Action:** After completing cpp-advanced weeks 1–5, add:
- `perf` and `ftrace` for latency profiling
- `/proc/latency_stats` interpretation
- PREEMPT_RT patch model

---

## Track 7: Optimization
**Priority: MEDIUM-HIGH**
**Status:** ✅ **COMPLETED**
**Folder:** `learn/optimization/`

7 core lessons covering the optimization stack used in robotics — from gradient
fundamentals through least-squares (Ceres), constrained/convex optimization (MPC/QP),
graph-based SLAM backends, and numerical methods for debugging ill-conditioned problems.
6 exercise sets with Python implementations, Ceres C++ patterns, and OKS-specific
debugging scenarios.

| Week | Topic | OKS Relevance |
|------|-------|---------------|
| 1 | Gradients, Hessians, Taylor expansions, convexity | Foundation for everything below |
| 2 | Gradient descent, Newton's method, line search | Understanding solver internals |
| 3 | Nonlinear least squares, Gauss-Newton, LM, Ceres Solver | Estimator uses Ceres (LM) for calibration |
| 4 | KKT conditions, penalty/barrier methods, LP/QP/NLP | Controller constraint handling |
| 5 | Convex optimization, QP solvers, SOCP, OSQP, MPC | MPC velocity/acceleration planning |
| 6 | Pose graphs, factor graphs, g2o/GTSAM, sparsity | SLAM backends, loop closure |
| 7 | Conditioning, finite differences, sparse solvers, debugging | Diagnosing numerical failures in production |

**Files:**
```
optimization/
├── 00-learning-plan.md            ← dependency graph, study order, milestones
├── 01-foundations.md              ← gradients, Hessians, Taylor, convexity
├── 02-unconstrained-optimization.md ← GD, Newton, BFGS, line search, trust region
├── 03-least-squares.md            ← NLS, Gauss-Newton, LM, Ceres API
├── 04-constrained-optimization.md ← KKT, penalty, barrier, LP/QP/NLP taxonomy
├── 05-convex-optimization.md      ← convex sets/functions, QP, SOCP, SDP, OSQP, MPC
├── 06-graph-optimization.md       ← pose graphs, factor graphs, g2o, GTSAM, sparsity
├── 07-numerical-methods.md        ← conditioning, finite diff, sparse factorization
└── exercises/
    ├── 01-gradients-convexity.md  ← gradient/Hessian computation, Rosenbrock, Newton vs GD
    ├── 02-least-squares-ceres.md  ← circle fitting LM, Huber loss, Ceres cost functors
    ├── 03-constrained-optimization.md ← penalty method, LP resource allocation, KKT
    ├── 04-convex-mpc.md           ← CVXPY MPC, 1D/2D, obstacle avoidance, OSQP
    ├── 05-graph-slam.md           ← 1D/2D pose graph from scratch, loop closure
    └── 06-numerical-debugging.md  ← gradient checking, condition number, EKF debugging
```

---

## Recommended Study Order

```
PARALLEL TRACKS (can interleave):

NOW
│
├── navigation-estimator/   ← start immediately, highest RCA value
│   Week 1: Dead-reckoning math
│   Week 2: EKF theory
│   Week 3: Measurement models
│   Week 4: IMU fusion
│   Week 5: Failure mode diagnosis
│
├── cpp-advanced/ (continue from wherever you are)
│
IN 6–8 WEEKS
│
├── ros2-handson/           ← after navigation-estimator (builds on the theory)
├── python-oks/             ← can start in parallel with navigation-estimator
├── optimization/           ← after navigation-estimator + control-systems
│   Week 1-3: Foundations → least-squares (unlocks Ceres understanding)
│   Week 4-5: Constrained + convex (unlocks MPC reasoning)
│   Week 6-7: Graph optimization + numerical methods (unlocks SLAM debugging)
│
WHEN HARDWARE ARRIVES
│
└── zephyr/deep-dive/       ← hardware-gated, builds on existing zephyr/ theory
```

---

## Quick-Start: Next Session

Start here → [navigation-estimator/00-learning-plan.md](navigation-estimator/00-learning-plan.md)

**Day 1 goal (2 hrs):** Read `01-dead-reckoning.md` and be able to derive
`(x', y', θ')` from two encoder counts for a differential-drive robot.

This directly ties to `doPrediction()` in OKS estimator — you'll recognize every variable.
