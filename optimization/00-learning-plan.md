# Optimization for Robotics Engineers
### From gradient descent to pose-graph SLAM — the math behind Ceres, MPC, and Nav2
### Total: ~50–60 hours across 6 weeks
### Expanded: ~280 hours across 16 weeks (includes Lie Groups & Game Theory)

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
├── 08-matrix-essentials.md                 ← Matrix Lie groups, exponential map, adjoint
├── 09-lie-groups.md                        ← SO(3)/SE(3) geometry, manifold optimization
├── 10-game-theory.md                       ← Nash equilibria, mechanism design, multi-agent
├── exercises/
│   ├── 01-gradients-convexity.md           ← Hand-derive gradients, classify convexity
│   ├── 02-gradient-descent.md              ← Implement GD/Newton from scratch in Python
│   ├── 03-least-squares-ceres.md           ← Fit curves with scipy + Ceres, reproduce sensorbar fit
│   ├── 04-constrained-kkt.md               ← Solve KKT by hand, verify with solver
│   ├── 05-convex-cvxpy.md                  ← Model LP/QP problems in CVXPy
│   ├── 06-pose-graph.md                    ← Build a mini pose-graph optimizer
│   ├── 07-08-09/                           ← Lie groups & manifold exercises
│   └── 10/                                 ← Game theory exercises
└── code/
    ├── lie_groups/                          ← SO(3)/SE(3) Python implementations
    └── game_theory/                         ← Nash, minimax, auction modules
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
| 08 Matrix essentials | Matrix decompositions, SVD, pseudoinverse in calibration |
| 09 Lie groups | **SE(3) pose-graph SLAM**, manifold optimization, IMU preintegration |
| 10 Game theory | Fleet coordination, congestion-aware routing, auction-based allocation |

---

## Phase VI — Lie Groups & Manifold Optimization (Weeks 11–13)

### Week 11 — Group Theory & Rotations (17.5 hrs)
**Read:** `09-lie-groups.md` (Sections 1–4)
**Code:** `code/lie_groups/lie_groups.py`

| Day | Topic | Key concepts |
|-----|-------|-------------|
| 71 | Group Axioms | Closure, associativity, identity, inverse; SO(2), SO(3) |
| 72 | SO(2) Rotations | 2D rotation matrix, angle wrap, composition |
| 73 | SO(3) & Rodrigues | Axis-angle, Rodrigues formula, exp map |
| 74 | Quaternions | Unit quaternions ↔ SO(3), Hamilton product |
| 75 | Quaternion Ops | SLERP, interpolation, avoiding gimbal lock |
| 76 | Rotation Representations | Euler angles, rotation vectors, comparison chart |
| 77 | Week 11 Review | Synthesis + self-assessment |

### Week 12 — Rigid Motions & Lie Algebra (17.5 hrs)
**Read:** `09-lie-groups.md` (Sections 5–8)

| Day | Topic | Key concepts |
|-----|-------|-------------|
| 78 | SE(2) Rigid Motions | Homogeneous coords, composition, inverse |
| 79 | SE(3) Rigid Body | 4×4 transformation, adjoint action |
| 80 | Exponential Map | se(3) → SE(3), hat/vee operators |
| 81 | Lie Algebra | Tangent space, generators, bracket |
| 82 | Adjoint & BCH | Adjoint representation, BCH formula |
| 83 | Jacobians on Manifolds | Left/right Jacobians, perturbation models |
| 84 | Week 12 Review | Synthesis + self-assessment |

### Week 13 — Manifold Optimization & Capstone (17.5 hrs)
**Read:** `09-lie-groups.md` (Sections 9–10)

| Day | Topic | Key concepts |
|-----|-------|-------------|
| 85 | Manifold Gauss-Newton | Retraction, tangent-space linearization |
| 86 | Pose-Graph SLAM SE(2) | Graph formulation, anchoring, sparsity |
| 87 | Pose-Graph SLAM SE(3) | 3D extension, robust kernels (Huber, Cauchy) |
| 88 | Ceres & GTSAM | Local parameterization, factor graphs |
| 89 | IMU Preintegration | Forster et al., bias correction |
| 90 | Calibration & Interpolation | Hand-eye AX=XB, SLERP, trajectory interp |
| 91 | **Capstone: Mini VIO** | SE(3) + IMU + landmarks, sliding-window GN |

---

## Phase VII — Game Theory for Optimization & Robotics (Weeks 14–16)

### Week 14 — Normal Form & Zero-Sum Games (17.5 hrs)
**Read:** `10-game-theory.md` (Sections 1–4)
**Code:** `code/game_theory/game_theory.py`

| Day | Topic | Key concepts |
|-----|-------|-------------|
| 92 | Normal Form Games | Payoff matrices, dominant strategies, IESDS |
| 93 | Pure Nash Equilibria | Best response, multiple equilibria, coordination |
| 94 | Mixed Nash Equilibria | Support enumeration, indifference principle |
| 95 | Equilibrium Selection | Pareto, risk dominance, focal points |
| 96 | Zero-Sum & Minimax | Minimax theorem, saddle points |
| 97 | Zero-Sum LP | LP formulation, dual interpretation |
| 98 | Week 14 Review | Synthesis + self-assessment |

### Week 15 — Cooperative & Dynamic Games (17.5 hrs)
**Read:** `10-game-theory.md` (Sections 5–7)

| Day | Topic | Key concepts |
|-----|-------|-------------|
| 99 | Cooperative Games & Core | Coalitional games, core stability |
| 100 | Shapley Value | Axioms, computation, allocation fairness |
| 101 | Nash Bargaining | Bargaining solution, axiomatic approach |
| 102 | Dynamic Games | Extensive form, backward induction |
| 103 | Repeated Games | Folk theorem, trigger strategies |
| 104 | Mechanism Design | VCG, incentive compatibility, revelation principle |
| 105 | Week 15 Review | Synthesis + self-assessment |

### Week 16 — Algorithmic GT & Capstone (17.5 hrs)
**Read:** `10-game-theory.md` (Sections 8–10)

| Day | Topic | Key concepts |
|-----|-------|-------------|
| 106 | Computing Nash | Lemke-Howson, PPAD complexity |
| 107 | Fictitious Play | Empirical strategy learning, convergence |
| 108 | No-Regret Learning | Regret matching, multiplicative weights |
| 109 | Congestion Games | Potential games, Braess's paradox, PoA |
| 110 | Multi-Robot & Adversarial | Auctions, pursuit-evasion, GAN minimax |
| 111 | Robust Optimization | Minimax as game against nature, H∞ control |
| 112 | **Capstone: Multi-Robot Delivery** | Auction + congestion + robust rerouting |
