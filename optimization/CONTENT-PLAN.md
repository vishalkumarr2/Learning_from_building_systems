# Optimization Mastery — Content Creation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate all 70 daily lesson files + 10 weekly exercise sets + 5 capstone projects for the 10-week Optimization Mastery curriculum, each with theory, runnable code, and expert-level challenges.

**Architecture:** Each week becomes a folder under `learn/optimization/weeks/`. Each day is a standalone `.md` file with theory + implementation exercises (complete runnable code) + expert questions. Weekly review days include comprehensive exercise sets and capstone projects. A `code/` sibling directory holds runnable `.py` and `.cpp` files.

**Tech Stack:** Markdown, Python (NumPy/SciPy/JAX/CVXPY/OSQP/CasADi), C++ (Ceres/g2o/GTSAM/Eigen)

---

## File Structure

```
learn/optimization/
├── CURRICULUM.md                          ← EXISTS (master syllabus)
├── 00-learning-plan.md                    ← EXISTS (keep as-is, quick-ref)
├── 01-foundations.md ... 07-numerical-methods.md  ← EXISTS (condensed notes)
├── exercises/01-..06-                     ← EXISTS (condensed exercises)
│
├── weeks/                                 ← NEW: daily deep-dive content
│   ├── README.md                          ← navigation index
│   │
│   ├── week-01/                           ← Phase I: Math Foundations
│   │   ├── day-01-vectors-matrices.md
│   │   ├── day-02-eigenvalues-svd.md
│   │   ├── day-03-gradients-jacobians.md
│   │   ├── day-04-hessians-taylor.md
│   │   ├── day-05-convexity.md
│   │   ├── day-06-optimality-conditions.md
│   │   └── day-07-week-review.md          ← review + Exercise Set 1
│   │
│   ├── week-02/                           ← Phase I: Advanced Foundations
│   │   ├── day-08-matrix-decompositions.md
│   │   ├── day-09-sparse-schur.md
│   │   ├── day-10-numerical-differentiation.md
│   │   ├── day-11-automatic-differentiation.md
│   │   ├── day-12-conditioning.md
│   │   ├── day-13-iterative-linear-solvers.md
│   │   └── day-14-week-review.md          ← review + Exercise Set 2
│   │
│   ├── week-03/                           ← Phase II: Unconstrained I
│   │   ├── day-15-gradient-descent.md
│   │   ├── day-16-line-search.md
│   │   ├── day-17-newtons-method.md
│   │   ├── day-18-trust-region.md
│   │   ├── day-19-quasi-newton-bfgs.md
│   │   ├── day-20-conjugate-gradient-optim.md
│   │   └── day-21-week-review.md          ← review + Optimizer Shootout
│   │
│   ├── week-04/                           ← Phase II: Unconstrained II
│   │   ├── day-22-convergence-theory.md
│   │   ├── day-23-momentum-acceleration.md
│   │   ├── day-24-stochastic-methods-enrichment.md
│   │   ├── day-25-regularization.md
│   │   ├── day-26-global-optimization.md
│   │   ├── day-27-scipy-optimize.md
│   │   └── day-28-week-review.md          ← Capstone: miniopt library
│   │
│   ├── week-05/                           ← Phase III: NLS (Python)
│   │   ├── day-29-linear-least-squares.md
│   │   ├── day-30-nls-formulation.md
│   │   ├── day-31-gauss-newton.md
│   │   ├── day-32-levenberg-marquardt.md
│   │   ├── day-33-robust-estimation.md
│   │   ├── day-34-scipy-least-squares.md
│   │   └── day-35-week-review.md          ← Capstone: Wheel Calibration + Fisher Info
│   │
│   ├── week-06/                           ← Phase III: Ceres (C++)
│   │   ├── day-36-ceres-intro.md
│   │   ├── day-37-ceres-autodiff.md
│   │   ├── day-38-ceres-loss-functions.md
│   │   ├── day-39-ceres-robotics.md
│   │   ├── day-40-ceres-solver-config.md
│   │   ├── day-41-lie-groups-manifolds.md  ← CRITICAL: Lie groups for SLAM/calibration
│   │   └── day-42-week-review.md          ← Capstone: Multi-Sensor Calibration
│   │
│   ├── week-07/                           ← Phase IV: Constrained
│   │   ├── day-43-lagrange-multipliers.md
│   │   ├── day-44-kkt-conditions.md
│   │   ├── day-45-penalty-barrier.md
│   │   ├── day-46-linear-programming.md
│   │   ├── day-47-quadratic-programming.md
│   │   ├── day-48-nonlinear-programming.md
│   │   └── day-49-week-review.md          ← Capstone: Robot Arm Motion
│   │
│   ├── week-08/                           ← Phase IV: Convex + MPC
│   │   ├── day-50-convex-theory.md
│   │   ├── day-51-cvxpy.md
│   │   ├── day-52-osqp-realtime.md
│   │   ├── day-53-mpc-theory.md
│   │   ├── day-54-mpc-2d-robot.md
│   │   ├── day-55-nmpc-casadi.md
│   │   └── day-56-week-review.md          ← Capstone: Warehouse MPC
│   │
│   ├── week-09/                           ← Phase V: SLAM
│   │   ├── day-57-factor-graphs.md
│   │   ├── day-58-1d-pose-graph.md
│   │   ├── day-59-2d-pose-graph.md
│   │   ├── day-60-g2o.md
│   │   ├── day-61-gtsam-isam2.md
│   │   ├── day-62-robust-slam.md
│   │   └── day-63-week-review.md          ← Capstone: Full SLAM Pipeline
│   │
│   └── week-10/                           ← Phase V: Applications + Final
│       ├── day-64-calibration.md
│       ├── day-65-trajectory-optimization.md
│       ├── day-66-numerical-debugging.md
│       ├── day-67-realtime-optimization.md
│       ├── day-68-software-landscape.md
│       └── day-69-70-final-capstone.md    ← 2-day final project
│
└── code/                                  ← NEW: runnable implementations
    ├── README.md                          ← setup instructions, run guide
    ├── requirements.txt                   ← Python dependencies
    ├── CMakeLists.txt                     ← C++ build for Ceres/g2o exercises
    │
    ├── week01/                            ← Python scripts for Week 1
    │   ├── gram_schmidt.py
    │   ├── power_iteration.py
    │   ├── numerical_gradient.py
    │   ├── dual_numbers.py                ← (Week 2 Day 11)
    │   └── ...
    ├── week02/
    ├── week03/
    │   ├── gradient_descent.py
    │   ├── newton_method.py
    │   ├── bfgs.py
    │   ├── trust_region.py
    │   └── ...
    ├── week04/
    │   └── miniopt/                       ← capstone library
    │       ├── __init__.py
    │       ├── minimize.py
    │       ├── line_search.py
    │       └── test_miniopt.py
    ├── week05/
    │   ├── gauss_newton.py
    │   ├── levenberg_marquardt.py
    │   ├── wheel_calibration.py           ← capstone
    │   └── ...
    ├── week06_cpp/                        ← C++ Ceres exercises
    │   ├── CMakeLists.txt
    │   ├── hello_ceres.cpp
    │   ├── circle_fit.cpp
    │   ├── scan_matching.cpp
    │   ├── multi_sensor_calib.cpp         ← capstone
    │   ├── lie_group_so3.py               ← Day 41: SO(3)/SE(3) implementations
    │   └── ...
    ├── week07/
    │   ├── lagrange.py
    │   ├── kkt_solver.py
    │   ├── penalty_method.py
    │   └── ...
    ├── week08/
    │   ├── cvxpy_examples.py
    │   ├── osqp_mpc.py
    │   ├── casadi_nmpc.py
    │   ├── warehouse_mpc.py              ← capstone
    │   └── ...
    ├── week09/
    │   ├── pose_graph_1d.py
    │   ├── pose_graph_2d.py
    │   ├── gtsam_slam.py
    │   ├── slam_pipeline.py              ← capstone
    │   └── ...
    └── week10/
        ├── camera_calibration.py
        ├── trajectory_optimization.py
        ├── final_capstone/               ← integrated system
        │   ├── calibration_module.py
        │   ├── slam_backend.py
        │   ├── mpc_controller.py
        │   └── run_pipeline.py
        └── ...
```

**Total file count:** ~70 daily `.md` files + ~60 `.py` files + ~10 `.cpp` files + 4 phase-gate quiz files + 3 index files = **~147 files**

---

## Content Template — Every Daily File MUST Contain

Each `day-XX-topic.md` file follows this structure:

```markdown
# Day XX: [Topic Title]
> Phase [N] — Week [N] | Estimated time: 2.5 hours
> Prerequisites: Day XX-1

---

## Learning Objectives
- [ ] Objective 1 (measurable — "can implement X", "can derive Y")
- [ ] Objective 2
- [ ] Objective 3

## Key Concepts (theory — 45 min)
### [Subsection]
Theory with equations, diagrams, intuition.
Every concept tied to a robotics motivation.

## Implementation Exercises (hands-on — 60 min)
### Exercise 1: [Name]
Complete runnable code with:
- Problem setup + data generation
- Skeleton with TODO markers
- Solution in <details> block
- Expected output / plot description

### Exercise 2: [Name]
...

## Expert Challenge Problems (45 min)
### 🔴 Challenge 1: [Hard conceptual question]
Requires deep understanding, proof, or derivation.
<details><summary>Solution</summary>...</details>

### 🔴 Challenge 2: [Production debugging scenario]
"The solver diverged on this real problem. Diagnose and fix."
<details><summary>Solution</summary>...</details>

### 🔴 Challenge 3: [Design question]
"You're building X for a warehouse robot. Choose the algorithm and justify."
<details><summary>Solution</summary>...</details>

## Connections
- **Back-link:** How this builds on Day XX-1
- **Forward-link:** What this enables in Day XX+1
- **OKS relevance:** Specific connection to robot subsystem

## Self-Check
- [ ] Can I explain [concept] in one sentence?
- [ ] Did my code produce correct output?
- [ ] Can I solve the expert challenges without looking?

## Further Reading
- [Link 1] — for deeper theory
- [Link 2] — for production examples
```

---

## Expert-Level Question Categories

Every day includes **3 expert challenges** drawn from these categories (rotate):

| Category | Example | Appears In |
|----------|---------|------------|
| **Proof/Derivation** | "Prove BFGS maintains PD if Wolfe conditions hold" | Theory-heavy days |
| **Debugging Scenario** | "Ceres reports NUMERICAL_FAILURE. Given this Jacobian, diagnose why." | Implementation days |
| **Design Decision** | "MPC at 50Hz on ARM Cortex-M7. QP solver choice and justify." | Application days |
| **Complexity Analysis** | "What's the per-iteration cost of your SLAM backend with N poses, M landmarks, K observations?" | Algorithm days |
| **Edge Case** | "Your GN solver works on synthetic data but diverges on real. List 5 possible causes, ranked by likelihood." | Every review day |
| **Interview Question** | "Explain the difference between Gauss-Newton and Levenberg-Marquardt to a senior engineer in 2 minutes." | Every week review |
| **Code Review** | "Here's a colleague's Ceres cost function. Find 3 bugs." | C++ weeks |
| **Paper Reading** | "Read Section 3 of Grisetti's SLAM tutorial. Summarize the key insight in 3 sentences." | SLAM week |

---

## Execution Plan — Task Breakdown

### Batch 1: Infrastructure + Phase I (Weeks 1–2) — 16 files

- [ ] **Task 1.0: Create directory structure + boilerplate**
  - Create: `learn/optimization/weeks/README.md`
  - Create: `learn/optimization/code/README.md`
  - Create: `learn/optimization/code/requirements.txt`
  - Create: all empty week folders

- [ ] **Task 1.1: Week 1 — Day 1 (Vectors, Matrices)**
  - Create: `weeks/week-01/day-01-vectors-matrices.md`
  - Create: `code/week01/gram_schmidt.py`
  - Content: linalg fundamentals, Gram-Schmidt impl, 10 hand problems, 3 expert Qs

- [ ] **Task 1.2: Week 1 — Day 2 (Eigenvalues, PD, SVD)**
  - Create: `weeks/week-01/day-02-eigenvalues-svd.md`
  - Create: `code/week01/power_iteration.py`, `code/week01/pd_classifier.py`

- [ ] **Task 1.3: Week 1 — Day 3 (Gradients, Jacobians)**
  - Create: `weeks/week-01/day-03-gradients-jacobians.md`
  - Create: `code/week01/numerical_gradient.py`

- [ ] **Task 1.4: Week 1 — Day 4 (Hessians, Taylor)**
  - Create: `weeks/week-01/day-04-hessians-taylor.md`
  - Create: `code/week01/numerical_hessian.py`, `code/week01/taylor_viz.py`

- [ ] **Task 1.5: Week 1 — Day 5 (Convexity)**
  - Create: `weeks/week-01/day-05-convexity.md`
  - Create: `code/week01/convexity_checker.py`

- [ ] **Task 1.6: Week 1 — Day 6 (Optimality Conditions)**
  - Create: `weeks/week-01/day-06-optimality-conditions.md`
  - Create: `code/week01/critical_point_finder.py`

- [ ] **Task 1.7: Week 1 — Day 7 (Review + Exercise Set 1)**
  - Create: `weeks/week-01/day-07-week-review.md`
  - Create: `code/week01/exercise_set_01.py` (10 comprehensive problems)

- [ ] **Task 1.8: Week 2 — Day 8 (Matrix Decompositions)**
  - Create: `weeks/week-02/day-08-matrix-decompositions.md`
  - Create: `code/week02/cholesky.py`

- [ ] **Task 1.9: Week 2 — Day 9 (Sparse Matrices + Schur Complement)**
  - Create: `weeks/week-02/day-09-sparse-schur.md`
  - Create: `code/week02/sparse_benchmark.py`, `code/week02/schur_complement.py`
  - Content: CSR/CSC, Cholesky, AMD/METIS, Schur complement, BA structure, marginalization preview

- [ ] **Task 1.10: Week 2 — Day 10 (Numerical Differentiation)**
  - Create: `weeks/week-02/day-10-numerical-differentiation.md`
  - Create: `code/week02/finite_differences.py`

- [ ] **Task 1.11: Week 2 — Day 11 (Automatic Differentiation)**
  - Create: `weeks/week-02/day-11-automatic-differentiation.md`
  - Create: `code/week02/dual_numbers.py`, `code/week02/jax_ad_demo.py`

- [ ] **Task 1.12: Week 2 — Day 12 (Conditioning)**
  - Create: `weeks/week-02/day-12-conditioning.md`
  - Create: `code/week02/conditioning_demo.py`

- [ ] **Task 1.13: Week 2 — Day 13 (Iterative Solvers / CG)**
  - Create: `weeks/week-02/day-13-iterative-linear-solvers.md`
  - Create: `code/week02/conjugate_gradient.py`

- [ ] **Task 1.14: Week 2 — Day 14 (Review + Exercise Set 2)**
  - Create: `weeks/week-02/day-14-week-review.md`
  - Create: `code/week02/exercise_set_02.py`

---

### Batch 2: Phase II (Weeks 3–4) — 14 files

- [ ] **Task 2.1: Week 3 — Day 15 (Gradient Descent)**
  - Create: `weeks/week-03/day-15-gradient-descent.md`
  - Create: `code/week03/gradient_descent.py`

- [ ] **Task 2.2: Week 3 — Day 16 (Line Search)**
  - Create: `weeks/week-03/day-16-line-search.md`
  - Create: `code/week03/line_search.py`

- [ ] **Task 2.3: Week 3 — Day 17 (Newton's Method)**
  - Create: `weeks/week-03/day-17-newtons-method.md`
  - Create: `code/week03/newton_method.py`

- [ ] **Task 2.4: Week 3 — Day 18 (Trust Region)**
  - Create: `weeks/week-03/day-18-trust-region.md`
  - Create: `code/week03/trust_region.py`

- [ ] **Task 2.5: Week 3 — Day 19 (BFGS / L-BFGS)**
  - Create: `weeks/week-03/day-19-quasi-newton-bfgs.md`
  - Create: `code/week03/bfgs.py`, `code/week03/lbfgs.py`

- [ ] **Task 2.6: Week 3 — Day 20 (Nonlinear CG)**
  - Create: `weeks/week-03/day-20-conjugate-gradient-optim.md`
  - Create: `code/week03/nonlinear_cg.py`

- [ ] **Task 2.7: Week 3 — Day 21 (Review + Optimizer Shootout)**
  - Create: `weeks/week-03/day-21-week-review.md`
  - Create: `code/week03/optimizer_shootout.py`

- [ ] **Task 2.8: Week 4 — Day 22 (Convergence Theory)**
  - Create: `weeks/week-04/day-22-convergence-theory.md`
  - Create: `code/week04/convergence_rates.py`

- [ ] **Task 2.9: Week 4 — Day 23 (Momentum / Nesterov)**
  - Create: `weeks/week-04/day-23-momentum-acceleration.md`
  - Create: `code/week04/momentum.py`

- [ ] **Task 2.10: Week 4 — Day 24 (SGD / Adam — Enrichment)**
  - Create: `weeks/week-04/day-24-stochastic-methods-enrichment.md`
  - Create: `code/week04/sgd_adam.py`
  - Note: marked as optional enrichment — lighter treatment, add proximal operator connections

- [ ] **Task 2.11: Week 4 — Day 25 (Regularization)**
  - Create: `weeks/week-04/day-25-regularization.md`
  - Create: `code/week04/regularization.py`

- [ ] **Task 2.12: Week 4 — Day 26 (Global Optimization)**
  - Create: `weeks/week-04/day-26-global-optimization.md`
  - Create: `code/week04/global_optim.py`

- [ ] **Task 2.13: Week 4 — Day 27 (scipy.optimize)**
  - Create: `weeks/week-04/day-27-scipy-optimize.md`
  - Create: `code/week04/scipy_showcase.py`

- [ ] **Task 2.14: Week 4 — Day 28 (Review + miniopt Capstone)**
  - Create: `weeks/week-04/day-28-week-review.md`
  - Create: `code/week04/miniopt/__init__.py`, `minimize.py`, `line_search.py`, `test_miniopt.py`

---

### Batch 3: Phase III (Weeks 5–6) — 14 files

- [ ] **Task 3.1: Week 5 — Day 29 (Linear Least Squares)**
  - Create: `weeks/week-05/day-29-linear-least-squares.md`
  - Create: `code/week05/linear_ls.py`

- [ ] **Task 3.2: Week 5 — Day 30 (NLS Formulation)**
  - Create: `weeks/week-05/day-30-nls-formulation.md`
  - Create: `code/week05/circle_fit_nls.py`

- [ ] **Task 3.3: Week 5 — Day 31 (Gauss-Newton)**
  - Create: `weeks/week-05/day-31-gauss-newton.md`
  - Create: `code/week05/gauss_newton.py`

- [ ] **Task 3.4: Week 5 — Day 32 (Levenberg-Marquardt)**
  - Create: `weeks/week-05/day-32-levenberg-marquardt.md`
  - Create: `code/week05/levenberg_marquardt.py`

- [ ] **Task 3.5: Week 5 — Day 33 (Robust Estimation)**
  - Create: `weeks/week-05/day-33-robust-estimation.md`
  - Create: `code/week05/robust_fitting.py`

- [ ] **Task 3.6: Week 5 — Day 34 (scipy.optimize.least_squares)**
  - Create: `weeks/week-05/day-34-scipy-least-squares.md`
  - Create: `code/week05/scipy_nls.py`

- [ ] **Task 3.7: Week 5 — Day 35 (Review + Fisher Info + Wheel Calibration Capstone)**
  - Create: `weeks/week-05/day-35-week-review.md`
  - Create: `code/week05/wheel_calibration.py`, `code/week05/fisher_information.py`
  - Content: Cramér-Rao bound, Fisher info matrix, covariance from NLS, observability, capstone

- [ ] **Task 3.8: Week 6 — Day 36 (Ceres Intro)**
  - Create: `weeks/week-06/day-36-ceres-intro.md`
  - Create: `code/week06_cpp/hello_ceres.cpp`, `code/week06_cpp/CMakeLists.txt`

- [ ] **Task 3.9: Week 6 — Day 37 (Ceres AutoDiff)**
  - Create: `weeks/week-06/day-37-ceres-autodiff.md`
  - Create: `code/week06_cpp/triangulation.cpp`

- [ ] **Task 3.10: Week 6 — Day 38 (Ceres Loss Functions)**
  - Create: `weeks/week-06/day-38-ceres-loss-functions.md`
  - Create: `code/week06_cpp/circle_fit_robust.cpp`, `code/week06_cpp/scan_matching.cpp`

- [ ] **Task 3.11: Week 6 — Day 39 (Ceres Robotics)**
  - Create: `weeks/week-06/day-39-ceres-robotics.md`
  - Create: `code/week06_cpp/imu_odom_calib.cpp`, `code/week06_cpp/pose_graph_ceres.cpp`

- [ ] **Task 3.12: Week 6 — Day 40 (Solver Config / Debugging)**
  - Create: `weeks/week-06/day-40-ceres-solver-config.md`
  - Create: `code/week06_cpp/solver_benchmark.cpp`

- [ ] **Task 3.13: Week 6 — Day 41 (Lie Groups & Manifold Optimization)**
  - Create: `weeks/week-06/day-41-lie-groups-manifolds.md`
  - Create: `code/week06_cpp/lie_group_so3.py`, `code/week06_cpp/bundle_adjustment.cpp`
  - Content: SO(2)/SO(3)/SE(3), exp/log maps, tangent space, manifold GN, Ceres manifolds
  - **CRITICAL** — this is the #1 gap identified by expert review

- [ ] **Task 3.14: Week 6 — Day 42 (Review + Multi-Sensor Calib Capstone)**
  - Create: `weeks/week-06/day-42-week-review.md`
  - Create: `code/week06_cpp/multi_sensor_calib.cpp`

---

### Batch 4: Phase IV (Weeks 7–8) — 14 files

- [ ] **Task 4.1: Week 7 — Day 43 (Lagrange Multipliers)**
  - Create: `weeks/week-07/day-43-lagrange-multipliers.md`
  - Create: `code/week07/lagrange.py`

- [ ] **Task 4.2: Week 7 — Day 44 (KKT Conditions)**
  - Create: `weeks/week-07/day-44-kkt-conditions.md`
  - Create: `code/week07/kkt_solver.py`

- [ ] **Task 4.3: Week 7 — Day 45 (Penalty / Barrier)**
  - Create: `weeks/week-07/day-45-penalty-barrier.md`
  - Create: `code/week07/penalty_method.py`, `code/week07/barrier_method.py`

- [ ] **Task 4.4: Week 7 — Day 46 (Linear Programming)**
  - Create: `weeks/week-07/day-46-linear-programming.md`
  - Create: `code/week07/simplex.py`

- [ ] **Task 4.5: Week 7 — Day 47 (Quadratic Programming)**
  - Create: `weeks/week-07/day-47-quadratic-programming.md`
  - Create: `code/week07/qp_solver.py`

- [ ] **Task 4.6: Week 7 — Day 48 (Nonlinear Programming)**
  - Create: `weeks/week-07/day-48-nonlinear-programming.md`
  - Create: `code/week07/nlp_scipy.py`

- [ ] **Task 4.7: Week 7 — Day 49 (Review + Robot Arm Capstone)**
  - Create: `weeks/week-07/day-49-week-review.md`
  - Create: `code/week07/robot_arm_motion.py`

- [ ] **Task 4.8: Week 8 — Day 50 (Convex Theory)**
  - Create: `weeks/week-08/day-50-convex-theory.md`
  - Create: `code/week08/convex_hierarchy.py`

- [ ] **Task 4.9: Week 8 — Day 51 (CVXPY)**
  - Create: `weeks/week-08/day-51-cvxpy.md`
  - Create: `code/week08/cvxpy_examples.py`

- [ ] **Task 4.10: Week 8 — Day 52 (OSQP Real-Time)**
  - Create: `weeks/week-08/day-52-osqp-realtime.md`
  - Create: `code/week08/osqp_warmstart.py`

- [ ] **Task 4.11: Week 8 — Day 53 (MPC Theory)**
  - Create: `weeks/week-08/day-53-mpc-theory.md`
  - Create: `code/week08/mpc_1d.py`

- [ ] **Task 4.12: Week 8 — Day 54 (MPC 2D Robot)**
  - Create: `weeks/week-08/day-54-mpc-2d-robot.md`
  - Create: `code/week08/mpc_diff_drive.py`

- [ ] **Task 4.13: Week 8 — Day 55 (NMPC + CasADi)**
  - Create: `weeks/week-08/day-55-nmpc-casadi.md`
  - Create: `code/week08/casadi_nmpc.py`

- [ ] **Task 4.14: Week 8 — Day 56 (Review + Warehouse MPC Capstone)**
  - Create: `weeks/week-08/day-56-week-review.md`
  - Create: `code/week08/warehouse_mpc.py`

---

### Batch 5: Phase V (Weeks 9–10) — 13 files

- [ ] **Task 5.1: Week 9 — Day 57 (Factor Graphs)**
  - Create: `weeks/week-09/day-57-factor-graphs.md`
  - Create: `code/week09/factor_graph_viz.py`

- [ ] **Task 5.2: Week 9 — Day 58 (1D Pose Graph)**
  - Create: `weeks/week-09/day-58-1d-pose-graph.md`
  - Create: `code/week09/pose_graph_1d.py`

- [ ] **Task 5.3: Week 9 — Day 59 (2D Pose Graph)**
  - Create: `weeks/week-09/day-59-2d-pose-graph.md`
  - Create: `code/week09/pose_graph_2d.py`

- [ ] **Task 5.4: Week 9 — Day 60 (g2o)**
  - Create: `weeks/week-09/day-60-g2o.md`
  - Create: `code/week09/g2o_slam.py`

- [ ] **Task 5.5: Week 9 — Day 61 (GTSAM + iSAM2)**
  - Create: `weeks/week-09/day-61-gtsam-isam2.md`
  - Create: `code/week09/gtsam_slam.py`

- [ ] **Task 5.6: Week 9 — Day 62 (Robust SLAM)**
  - Create: `weeks/week-09/day-62-robust-slam.md`
  - Create: `code/week09/robust_slam.py`

- [ ] **Task 5.7: Week 9 — Day 63 (Review + SLAM Capstone)**
  - Create: `weeks/week-09/day-63-week-review.md`
  - Create: `code/week09/slam_pipeline.py`

- [ ] **Task 5.8: Week 10 — Day 64 (Calibration)**
  - Create: `weeks/week-10/day-64-calibration.md`
  - Create: `code/week10/camera_calibration.py`

- [ ] **Task 5.9: Week 10 — Day 65 (Trajectory Optimization)**
  - Create: `weeks/week-10/day-65-trajectory-optimization.md`
  - Create: `code/week10/trajectory_optimization.py`

- [ ] **Task 5.10: Week 10 — Day 66 (Numerical Debugging)**
  - Create: `weeks/week-10/day-66-numerical-debugging.md`
  - Create: `code/week10/debug_exercises.py`

- [ ] **Task 5.11: Week 10 — Day 67 (Real-Time Optimization)**
  - Create: `weeks/week-10/day-67-realtime-optimization.md`
  - Create: `code/week10/realtime_benchmark.py`

- [ ] **Task 5.12: Week 10 — Day 68 (Software Landscape)**
  - Create: `weeks/week-10/day-68-software-landscape.md`
  - Create: `code/week10/multi_tool_comparison.py`

- [ ] **Task 5.13: Week 10 — Day 69-70 (Final Capstone)**
  - Create: `weeks/week-10/day-69-70-final-capstone.md`
  - Create: `code/week10/final_capstone/calibration_module.py`
  - Create: `code/week10/final_capstone/slam_backend.py`
  - Create: `code/week10/final_capstone/mpc_controller.py`
  - Create: `code/week10/final_capstone/run_pipeline.py`

---

### Batch 6: Phase-Gate Quizzes + Index Files + Integration

- [ ] **Task 6.0: Phase-Gate Self-Assessments**
  - Create: `weeks/phase-gate-quiz-II.md` — 10 questions testing Phase I prerequisites
  - Create: `weeks/phase-gate-quiz-III.md` — 10 questions testing Phase II prerequisites
  - Create: `weeks/phase-gate-quiz-IV.md` — 10 questions testing Phase III prerequisites
  - Create: `weeks/phase-gate-quiz-V.md` — 10 questions testing Phase IV prerequisites
  - Each quiz: questions + answer key + "review Day X" pointers for wrong answers

- [ ] **Task 6.1: Navigation README**
  - Create: `weeks/README.md` — links to all 70 days, progress tracker with checkboxes
  - Create: `code/README.md` — setup, how to run, dependency install

- [ ] **Task 6.2: requirements.txt + CMakeLists.txt**
  - Create: `code/requirements.txt`
  - Create: `code/week06_cpp/CMakeLists.txt`

- [ ] **Task 6.3: Update CURRICULUM.md cross-references**
  - Modify: `CURRICULUM.md` — add links from each Day section to the corresponding `weeks/` file

---

## Quality Gates Per File

Before marking a daily file complete, verify:

| Gate | Check |
|------|-------|
| **Theory depth** | ≥3 key concepts with equations, ≥1 diagram/visualization description |
| **Runnable code** | Every code block specifies language, has correct imports, produces output |
| **Hands-on count** | ≥3 implementation exercises per day (5 on capstone days) |
| **Expert questions** | ≥3 expert/senior challenges per day, with hidden solutions |
| **Robotics tie-in** | ≥1 exercise explicitly references a real robotics use case |
| **Connections** | Back-link + forward-link + OKS relevance filled in |
| **Self-check** | 3+ checklist items for self-assessment |
| **Code file** | Companion `.py` or `.cpp` file exists and runs |

---

## Execution Strategy

**Recommended: Subagent-driven, batch-per-week.**

Each batch = 1 week (7 daily files + companion code files). Run as one subagent task per day to keep context manageable.

**Estimated creation time per daily file:** ~15-20 min agent time
**Total estimated creation:** ~70 daily files × 18 min ≈ 21 hours of agent time
**Recommended batching:** 1 week per session (7 files) = 10 sessions

### Execution Order

```
Session 1:  Task 1.0 (infra) + Tasks 1.1–1.7  (Week 1)
Session 2:  Tasks 1.8–1.14                     (Week 2)
Session 3:  Tasks 2.1–2.7                      (Week 3)
Session 4:  Tasks 2.8–2.14                     (Week 4)
Session 5:  Tasks 3.1–3.7                      (Week 5)
Session 6:  Tasks 3.8–3.14                     (Week 6 — C++)
Session 7:  Tasks 4.1–4.7                      (Week 7)
Session 8:  Tasks 4.8–4.14                     (Week 8)
Session 9:  Tasks 5.1–5.7                      (Week 9)
Session 10: Tasks 5.8–5.13 + Task 6.*          (Week 10 + index)
Session 11: Tasks 7.1–7.7                      (Week 11 — Lie Groups: Rotations)
Session 12: Tasks 7.8–7.14                     (Week 12 — Lie Groups: Lie Algebra)
Session 13: Tasks 7.15–7.21                    (Week 13 — Lie Groups: Manifold Opt + Capstone)
Session 14: Tasks 8.1–8.7                      (Week 14 — Game Theory: Normal Form)
Session 15: Tasks 8.8–8.14                     (Week 15 — Game Theory: Cooperative + Dynamic)
Session 16: Tasks 8.15–8.21                    (Week 16 — Game Theory: Algorithmic + Capstone)
```

---

## Batch 7: Phase VI — Lie Groups & Manifold Optimization (Weeks 11–13)

**Status: ✅ COMPLETE**

### Reference + Code + Exercises
| Task | File | Status |
|------|------|--------|
| 7.0a | `09-lie-groups.md` — reference guide (~830 lines) | ✅ |
| 7.0b | `code/lie_groups/lie_groups.py` — full implementation (~1328 lines) | ✅ |
| 7.0c | `exercises/08-lie-groups.md` — 30 exercises | ✅ |

### Week 11 — Group Theory & Rotations (Days 71–77)
| Task | File | Status |
|------|------|--------|
| 7.1 | `day-71-group-axioms.md` | ✅ |
| 7.2 | `day-72-so2-rotations.md` | ✅ |
| 7.3 | `day-73-so3-rodrigues.md` | ✅ |
| 7.4 | `day-74-quaternions.md` | ✅ |
| 7.5 | `day-75-quaternion-operations.md` | ✅ |
| 7.6 | `day-76-rotation-representations.md` | ✅ |
| 7.7 | `day-77-week11-review.md` | ✅ |

### Week 12 — Rigid Motions & Lie Algebra (Days 78–84)
| Task | File | Status |
|------|------|--------|
| 7.8 | `day-78-se2-rigid-motions.md` | ✅ |
| 7.9 | `day-79-se3-rigid-body.md` | ✅ |
| 7.10 | `day-80-exponential-map.md` | ✅ |
| 7.11 | `day-81-lie-algebra.md` | ✅ |
| 7.12 | `day-82-adjoint-bch.md` | ✅ |
| 7.13 | `day-83-jacobians.md` | ✅ |
| 7.14 | `day-84-week12-review.md` | ✅ |

### Week 13 — Manifold Optimization + Capstone (Days 85–91)
| Task | File | Status |
|------|------|--------|
| 7.15 | `day-85-manifold-gauss-newton.md` | ✅ |
| 7.16 | `day-86-pose-graph-se2.md` | ✅ |
| 7.17 | `day-87-pose-graph-se3.md` | ✅ |
| 7.18 | `day-88-ceres-gtsam-manifold.md` | ✅ |
| 7.19 | `day-89-imu-preintegration.md` | ✅ |
| 7.20 | `day-90-calibration-interpolation.md` | ✅ |
| 7.21 | `day-91-week13-capstone.md` — Mini VIO | ✅ |

---

## Batch 8: Phase VII — Game Theory for Optimization & Robotics (Weeks 14–16)

**Status: ✅ COMPLETE**

### Reference + Code + Exercises
| Task | File | Status |
|------|------|--------|
| 8.0a | `10-game-theory.md` — reference guide (~1089 lines) | ✅ |
| 8.0b | `code/game_theory/game_theory.py` — full implementation (~1500 lines) | ✅ |
| 8.0c | `exercises/09-game-theory.md` — 30 exercises | ✅ |

### Week 14 — Normal Form & Zero-Sum Games (Days 92–98)
| Task | File | Status |
|------|------|--------|
| 8.1 | `day-92-normal-form-games.md` | ✅ |
| 8.2 | `day-93-pure-nash.md` | ✅ |
| 8.3 | `day-94-mixed-nash.md` | ✅ |
| 8.4 | `day-95-equilibrium-selection.md` | ✅ |
| 8.5 | `day-96-zero-sum-minimax.md` | ✅ |
| 8.6 | `day-97-zero-sum-lp.md` | ✅ |
| 8.7 | `day-98-week14-review.md` | ✅ |

### Week 15 — Cooperative & Dynamic Games (Days 99–105)
| Task | File | Status |
|------|------|--------|
| 8.8 | `day-99-cooperative-core.md` | ✅ |
| 8.9 | `day-100-shapley-value.md` | ✅ |
| 8.10 | `day-101-bargaining.md` | ✅ |
| 8.11 | `day-102-dynamic-games.md` | ✅ |
| 8.12 | `day-103-repeated-games.md` | ✅ |
| 8.13 | `day-104-mechanism-design.md` | ✅ |
| 8.14 | `day-105-week15-review.md` | ✅ |

### Week 16 — Algorithmic GT + Capstone (Days 106–112)
| Task | File | Status |
|------|------|--------|
| 8.15 | `day-106-computing-nash.md` | ✅ |
| 8.16 | `day-107-fictitious-play.md` | ✅ |
| 8.17 | `day-108-no-regret.md` | ✅ |
| 8.18 | `day-109-congestion-games.md` | ✅ |
| 8.19 | `day-110-multi-robot-adversarial.md` | ✅ |
| 8.20 | `day-111-robust-optimization.md` | ✅ |
| 8.21 | `day-112-week16-capstone.md` — Multi-Robot Delivery | ✅ |

---

## Relationship to Existing Content

The existing `01-foundations.md` through `07-numerical-methods.md` and `exercises/01-06` are **condensed reference notes** (~50-60 hours). They remain as quick-reference material.

The new `weeks/` content is the **expanded daily curriculum** (~175 hours) that:
- Goes deeper on every topic (3-7× more content per concept)
- Has complete runnable code (not just code skeletons)
- Includes expert-level challenges (none in existing exercises)
- Has daily structure (theory → implementation → practice)
- Includes 5 capstone projects (vs 0 in existing content)
- Covers additional topics: AD, sparse matrices, momentum methods, SGD, CasADi, real-time optimization, robust SLAM, trajectory optimization

The `00-learning-plan.md` and `CURRICULUM.md` serve as high-level maps; the `weeks/` content is the actual study material.
