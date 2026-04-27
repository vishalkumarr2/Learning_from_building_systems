# Optimization for Robotics Engineers
### From gradient descent to pose-graph SLAM — the math behind Ceres, MPC, and Nav2
### Total: ~50–60 hours across 6 weeks

---

## Why This Track Exists

Every major OKS subsystem runs an optimizer:

| Subsystem | Optimizer | What it solves |
|-----------|-----------|----------------|
| Navigation estimator | Ceres Solver (LM) | Sensorbar line-fit, pose correction |
| Control (MPC path) | QP / NLP solver | Trajectory tracking with constraints |
| SLAM / localization | Pose-graph optimizer (g2o) | Loop closure, map consistency |
| Path planner | Graph search + cost minimization | Shortest/safest path through costmap |
| Calibration | Nonlinear least-squares | Camera-lidar-wheel extrinsics |

If you can't read an optimizer's cost function, you can't debug why the robot drives wrong.

---

## Dependency Graph

```
01-foundations
│  (gradients, Hessians, convexity)
│
├──→ 02-unconstrained-optimization
│      (gradient descent, Newton, line search, trust region)
│      │
│      ├──→ 03-least-squares  ← HIGHEST OKS VALUE
│      │      (Gauss-Newton, Levenberg-Marquardt, Ceres Solver)
│      │
│      └──→ 07-numerical-methods
│             (convergence, conditioning, sparsity)
│
├──→ 04-constrained-optimization
│      (Lagrange, KKT, duality, penalty/barrier)
│      │
│      └──→ 05-convex-optimization
│             (LP, QP, SDP, interior-point, CVXPy)
│
└──→ 06-graph-optimization  ← SECOND HIGHEST OKS VALUE
       (factor graphs, pose-graph SLAM, g2o, GTSAM)
```

**Critical path for OKS work:** 01 → 02 → 03 (least-squares / Ceres) — this is 60% of the RCA value.

---

## Study Materials

```
learn/optimization/
├── 00-learning-plan.md                    ← YOU ARE HERE
├── 01-foundations.md                       ← Gradients, Hessians, convexity, optimality conditions
├── 02-unconstrained-optimization.md        ← GD, Newton, line search, trust region, convergence
├── 03-least-squares.md                     ← Linear LS, NLLS, Gauss-Newton, LM, Ceres deep dive
├── 04-constrained-optimization.md          ← Lagrange multipliers, KKT, duality, penalty methods
├── 05-convex-optimization.md               ← LP, QP, SOCP, SDP, interior-point, CVXPy
├── 06-graph-optimization.md                ← Factor graphs, pose-graph SLAM, g2o, GTSAM, Ceres
├── 07-numerical-methods.md                 ← Conditioning, convergence rates, sparsity, Cholesky
└── exercises/
    ├── 01-gradients-convexity.md           ← Hand-derive gradients, classify convexity
    ├── 02-gradient-descent.md              ← Implement GD/Newton from scratch in Python
    ├── 03-least-squares-ceres.md           ← Fit curves with scipy + Ceres, reproduce sensorbar fit
    ├── 04-constrained-kkt.md               ← Solve KKT by hand, verify with solver
    ├── 05-convex-cvxpy.md                  ← Model LP/QP problems in CVXPy
    └── 06-pose-graph.md                    ← Build a mini pose-graph optimizer
```

---

## Week-by-Week Schedule

### Week 1 — Foundations + Unconstrained (10 hrs)
**Read:** `01-foundations.md` + `02-unconstrained-optimization.md`
**Do:** Exercises 01 + 02

**Checkpoint — answer without looking:**
- [ ] What are the first-order and second-order necessary conditions for a minimum?
- [ ] When is a function convex? Give the Hessian condition.
- [ ] Gradient descent step: $x_{k+1} = x_k - \alpha \nabla f(x_k)$. What happens if $\alpha$ is too large?
- [ ] Newton's method converges quadratically near a minimum. Why? What's the downside?
- [ ] What is the Armijo condition for line search?

### Week 2 — Least Squares (10 hrs) 🎯
**Read:** `03-least-squares.md`
**Do:** Exercise 03 (the Ceres one is critical)

**Checkpoint:**
- [ ] Derive the normal equations for linear least squares
- [ ] Gauss-Newton approximates the Hessian as $J^T J$. Why is this valid for small residuals?
- [ ] LM adds $\lambda I$ to $J^T J$. What happens as $\lambda → 0$? As $\lambda → ∞$?
- [ ] In Ceres, what is a `CostFunction`, a `LossFunction`, and a `ParameterBlock`?
- [ ] The sensorbar uses `CauchyLoss(c=40.0)`. What does this do to outliers?

### Week 3 — Constrained Optimization (8 hrs)
**Read:** `04-constrained-optimization.md`
**Do:** Exercise 04

**Checkpoint:**
- [ ] State the KKT conditions for $\min f(x)$ s.t. $g(x) \leq 0$, $h(x) = 0$
- [ ] What is complementary slackness? Give a geometric interpretation.
- [ ] What is strong duality? When does it hold?
- [ ] How does a penalty method convert a constrained problem to unconstrained?

### Week 4 — Convex Optimization (8 hrs)
**Read:** `05-convex-optimization.md`
**Do:** Exercise 05

**Checkpoint:**
- [ ] Classify: LP ⊂ QP ⊂ SOCP ⊂ SDP ⊂ general convex. Why does this hierarchy matter?
- [ ] Write the standard-form LP. What does "vertex of the feasible polytope" mean?
- [ ] Interior-point methods use a barrier function. What is it? Why $O(\sqrt{n})$ iterations?
- [ ] Model a simple robot resource allocation as a QP in CVXPy.

### Week 5 — Graph Optimization + SLAM (10 hrs) 🎯
**Read:** `06-graph-optimization.md`
**Do:** Exercise 06

**Checkpoint:**
- [ ] What is a factor graph? How does it differ from a Bayesian network?
- [ ] In pose-graph SLAM, what are the nodes and edges? What is the cost function?
- [ ] How does Gauss-Newton apply to the pose-graph problem?
- [ ] What is the role of the information matrix (inverse covariance) in the edge cost?
- [ ] How do robust kernels (Huber, Cauchy) handle outlier loop closures?

### Week 6 — Numerical Methods + Integration (6 hrs)
**Read:** `07-numerical-methods.md`
**Do:** Review and integrate all exercises

**Checkpoint:**
- [ ] What is the condition number of a matrix? Why does it matter for optimization?
- [ ] Sparse vs dense Cholesky: when does sparsity win? (SLAM problems!)
- [ ] What is the difference between convergence rate: linear, superlinear, quadratic?
- [ ] Given a 10000-variable sparse least-squares problem, which solver would you pick?

---

## Key External Resources

See [RESOURCES.md — Optimization section](../RESOURCES.md#optimization) for the full curated list.

**Start with these three:**
1. **Boyd & Vandenberghe — Convex Optimization** (free PDF) — the bible, chapters 1–5 + 9–11
2. **Kochenderfer & Wheeler — Algorithms for Optimization** (free PDF) — practical, code-first
3. **Ceres Solver documentation** — the actual tool you debug against daily

**Video courses (pick one):**
- Stanford EE364A (Boyd) — rigorous, theory-heavy, gold standard
- CMU Convex Optimization (Tibshirani) — slightly more applied
- Optimization Methods for ML (KIT) — engineering-oriented

---

## OKS Relevance Map

| Study note | Directly used in OKS for... |
|------------|-----------------------------|
| 01 Foundations | Reading any cost function in the codebase |
| 02 Unconstrained | Understanding why GD is used in ML tuning but Newton in estimator |
| 03 Least squares | **Ceres sensorbar fit**, calibration, every NLLS in the stack |
| 04 Constrained | Understanding MPC constraint formulation |
| 05 Convex | QP in MPC, understanding solver guarantees |
| 06 Graph optimization | **Pose-graph SLAM**, loop closure, localization |
| 07 Numerical methods | Why Ceres uses sparse Cholesky, conditioning issues |
