# Curriculum Plan: Lie Groups & Game Theory
### Two new modules under the Optimization Mastery track
### Draft v2 — April 2026 (expanded to 6 weeks)

---

## How These Fit the Existing Track

```
Existing (Weeks 1-10):
  01-foundations → 02-unconstrained → 03-least-squares → ... → 08-matrix-essentials

New module placement:
  09-lie-groups.md           ← Reference guide (like 08-matrix-essentials)
  10-game-theory.md          ← Reference guide
  exercises/08-lie-groups.md
  exercises/09-game-theory.md
  code/lie_groups/            ← Python implementations
  code/game_theory/           ← Python implementations
  weeks/week-11/              ← Daily: Lie Groups — Group Theory + SO(2)/SO(3) (7 days)
  weeks/week-12/              ← Daily: Lie Groups — SE(2)/SE(3) + Lie Algebra Deep Dive (7 days)
  weeks/week-13/              ← Daily: Lie Groups — Manifold Optimization + Applications (7 days)
  weeks/week-14/              ← Daily: Game Theory — Foundations + Nash + Zero-Sum (7 days)
  weeks/week-15/              ← Daily: Game Theory — Cooperative + Dynamic + Mechanism Design (7 days)
  weeks/week-16/              ← Daily: Game Theory — Algorithms + Applications + Capstone (7 days)
```

**Prerequisites:**
- Lie Groups requires: Chapters 01 (foundations), 08 (matrix essentials), and Day 41 material
- Game Theory requires: Chapters 01 (foundations), 04 (constrained), 05 (convex)

---

# MODULE A: LIE GROUPS & MANIFOLD OPTIMIZATION

## Why This Module Exists

| Robotics subsystem | Lie group usage |
|---|---|
| SLAM / pose-graph | Poses live on SE(2)/SE(3) — can't add deltas in Euclidean space |
| Navigation estimator | Quaternion rotations, SO(3) for IMU integration |
| Calibration | Camera-lidar extrinsics are SE(3) transforms |
| Motion planning | Configuration spaces include SO(2) joint angles |
| Visual odometry | Frame-to-frame SE(3) estimation |
| Control (MPC) | Orientation tracking requires manifold-aware error |

**Current gap:** Day 41 in the 10-week curriculum covers Lie groups in 2.5 hours — enough for a taste, not for fluency. This module gives 3 full weeks (~52.5 hours) from group theory basics to production-level manifold optimization — enough time for derivations by hand, debugging manifold code, and building real applications.

---

## Dependency Graph

```
09-lie-groups (Reference Guide)
│
├── Part I: Algebraic Foundations
│   (groups, rings, fields, homomorphisms)
│
├── Part II: Topology & Manifolds
│   (topological spaces, smooth manifolds, tangent spaces, charts)
│
├── Part III: Core Lie Groups
│   │
│   ├── SO(2): 2D rotations
│   ├── SO(3): 3D rotations (Rodrigues, quaternions)
│   ├── SE(2): 2D rigid motions
│   ├── SE(3): 3D rigid motions
│   └── Sim(3): similarity transforms (monocular SLAM)
│
├── Part IV: Lie Algebra & Exp/Log Maps
│   (tangent space, generators, BCH formula, adjoint)
│
├── Part V: Optimization on Manifolds
│   (retraction, Riemannian gradient, manifold GN/LM)
│
└── Part VI: Applications
    (SLAM Jacobians, IMU preintegration, calibration, interpolation)
```

---

## Reference Guide: `09-lie-groups.md`

### Section 1 — Why Rotations Break Euclidean Math
- The "add delta" failure: $R + \Delta R$ is not a rotation matrix
- Gimbal lock, angle wrapping, quaternion double cover
- Motivating example: averaging 10 rotations — wrong way vs manifold way

### Section 2 — Group Theory Crash Course
- 2.1 What is a group: (G, ·) with closure, associativity, identity, inverse
- 2.2 Subgroups, cosets, normal subgroups (brief)
- 2.3 Group homomorphisms and isomorphisms
- 2.4 Matrix groups: GL(n), O(n), SO(n), SE(n)
- 2.5 Worked examples: integers under addition, $\mathbb{Z}/n\mathbb{Z}$, permutation groups

### Section 3 — Manifolds and Tangent Spaces
- 3.1 Topological space → smooth manifold (intuition: the Earth is locally flat)
- 3.2 Charts, atlases, coordinate maps
- 3.3 Tangent space $T_p M$ — velocity vectors of curves through $p$
- 3.4 Why tangent space matters: optimization perturbations live here
- 3.5 Riemannian metric: measuring distances and angles on manifolds

### Section 4 — The Core Lie Groups for Robotics
- 4.1 **SO(2)** — Circle group
  - Matrix form: $R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$
  - Lie algebra $\mathfrak{so}(2)$: skew-symmetric $2\times 2$ matrices ↔ scalars $\theta$
  - Exp map: $\theta \mapsto R(\theta)$, Log map: $R \mapsto \theta$
  - Worked example: compose two rotations, take the inverse

- 4.2 **SO(3)** — 3D Rotation Group
  - Representations: rotation matrix (9 params, 6 constraints), quaternion (4 params, 1 constraint), angle-axis (3 params, minimal), Euler angles (3 params, gimbal lock)
  - Lie algebra $\mathfrak{so}(3)$: skew-symmetric $3\times 3$ matrices ↔ $\mathbb{R}^3$ via hat/vee
  - Hat operator: $\omega^\wedge = \begin{bmatrix}0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0\end{bmatrix}$
  - Exponential map: **Rodrigues' formula** $\exp(\omega^\wedge) = I + \frac{\sin\theta}{\theta}\omega^\wedge + \frac{1-\cos\theta}{\theta^2}(\omega^\wedge)^2$
  - Logarithmic map: extract angle-axis from rotation matrix
  - Quaternion ↔ rotation matrix ↔ angle-axis conversions (all with code)
  - Worked example: interpolation between two rotations (SLERP)

- 4.3 **SE(2)** — 2D Rigid Motions
  - Homogeneous coordinates: $T = \begin{bmatrix}R & t \\ 0 & 1\end{bmatrix}$
  - Lie algebra $\mathfrak{se}(2)$: $(\omega, v_x, v_y) \in \mathbb{R}^3$
  - Exp/Log maps with closed-form expressions
  - Composition: $T_{AC} = T_{AB} \cdot T_{BC}$
  - Worked example: robot drives forward 1m then turns 30° — compute final pose

- 4.4 **SE(3)** — 3D Rigid Body Transformations
  - 4×4 homogeneous matrix
  - Lie algebra $\mathfrak{se}(3)$: twist coordinates $\xi = (\omega, v) \in \mathbb{R}^6$
  - Exp map via Rodrigues + V matrix
  - Adjoint representation: $\text{Ad}_T$ transforms twists between frames
  - Worked example: transform a point cloud from camera frame to world frame

- 4.5 **Sim(3)** — Similarity Transformations (brief)
  - Adds scale: $T = \begin{bmatrix}sR & t \\ 0 & 1\end{bmatrix}$
  - Why monocular SLAM needs Sim(3): scale is unobservable
  - 7 DoF Lie algebra

### Section 5 — The Lie Algebra–Group Connection
- 5.1 Exponential map derivation: matrix exponential of generator
- 5.2 Baker-Campbell-Hausdorff (BCH) formula: $\log(\exp(A)\exp(B)) \neq A + B$
  - First-order: $\approx A + B$; second-order: $\approx A + B + \frac{1}{2}[A,B]$
  - Why this matters: composing small perturbations
- 5.3 Adjoint representation: $\text{Ad}_X: \mathfrak{g} \to \mathfrak{g}$
  - Left vs right perturbation models
  - Adjoint of SE(3): the 6×6 matrix that transforms twists
- 5.4 Small-angle approximations: when $\exp(\delta) \approx I + \delta^\wedge$

### Section 6 — Optimization on Lie Groups
- 6.1 The manifold optimization framework
  - State $x \in \mathcal{M}$ (manifold), perturbation $\delta \in T_x\mathcal{M}$ (tangent space ≅ $\mathbb{R}^n$)
  - Update: $x \leftarrow x \oplus \delta = \text{Exp}(\delta) \circ x$ (or $x \circ \text{Exp}(\delta)$)
  - Cost function: $f(x) = \sum_i \| r_i(x) \|^2$
  - Jacobian: $\frac{\partial r_i}{\partial \delta}$ — derivative w.r.t. tangent space perturbation
- 6.2 Gauss-Newton on manifolds
  - Linearize: $r(x \oplus \delta) \approx r(x) + J\delta$
  - Solve normal equations: $J^TJ\delta^* = -J^Tr$
  - Update on manifold: $x \leftarrow x \oplus \delta^*$
  - Same as Euclidean GN, but the ⊕ is different!
- 6.3 Jacobian derivation on SE(3) — the hard part
  - Chain rule through exp map
  - Left Jacobian of SO(3): $J_l(\theta)$
  - Practical shortcut: first-order Jacobian (valid for small perturbations)
- 6.4 Solver support
  - Ceres: `SetManifold()`, `EigenQuaternionManifold`, `ProductManifold`
  - g2o: vertex types with built-in manifold ops
  - GTSAM: manifold traits, `between()`, `retract()`, `localCoordinates()`
  - manif library (C++): generic Lie group operations
- 6.5 Common pitfalls
  - Left vs right perturbation — inconsistent convention = wrong Jacobians
  - Quaternion sign ambiguity ($q$ and $-q$ represent the same rotation)
  - Gimbal lock in Euler angles — why angle-axis/quaternion preferred
  - Forgetting to normalize quaternions after update

### Section 7 — Applications Gallery
- 7.1 Rotation averaging (Hartley et al.)
- 7.2 IMU preintegration on SO(3) × $\mathbb{R}^3$
- 7.3 Pose-graph SLAM with SE(2)/SE(3) edges
- 7.4 Camera-IMU extrinsic calibration on SE(3)
- 7.5 Smooth trajectory interpolation: cumulative B-splines on SE(3)
- 7.6 Uncertainty propagation on manifolds (covariance on tangent space)

### Section 8 — Cheat Sheet
- Table: Group | Dim | Algebra Dim | Exp Map | Log Map | Compose | Inverse | Adjoint

---

## Exercises: `exercises/08-lie-groups.md` (~30 graded problems)

### Section A — Group Theory Basics (⭐, 4 problems)
- A1: Verify SO(2) group axioms (closure, associativity, identity, inverse)
- A2: Show matrix multiplication is not commutative for SO(3) with specific counter-example
- A3: Prove $\det(R) = 1$ and $R^T R = I$ for any $R \in SO(3)$
- A4: Homomorphism: show the determinant map $\det: GL(n) \to \mathbb{R}^*$ is a group homomorphism

### Section B — SO(2) and SO(3) Mechanics (⭐⭐, 6 problems)
- B1: Implement SO(2) exp/log, verify round-trip for 20 random angles
- B2: Implement Rodrigues' formula, verify exp(log(R)) = R for 50 random rotations
- B3: Compute the rotation that aligns vector $a$ to vector $b$ (two methods)
- B4: SLERP: interpolate between two quaternions at $t = 0, 0.25, 0.5, 0.75, 1$
- B5: Hat/Vee operators: verify $(\omega^\wedge)^T = -\omega^\wedge$ (skew-symmetry)
- B6: Quaternion double cover: show $q$ and $-q$ give the same rotation matrix

### Section C — SE(2) and SE(3) (⭐⭐, 5 problems)
- C1: Implement SE(2) class with compose, inverse, exp, log — test on robot path
- C2: Transform a 2D point cloud through a sequence of SE(2) poses, plot result
- C3: Implement SE(3) exp map using Rodrigues + V matrix
- C4: Compute $T_{AC} = T_{AB} \cdot T_{BC}$ for given transforms, verify inverse
- C5: Relative pose: given $T_{WA}$ and $T_{WB}$, compute $T_{AB}$

### Section D — Lie Algebra & Adjoint (⭐⭐⭐, 5 problems)
- D1: Verify BCH first-order: $\log(\exp(A)\exp(B)) \approx A + B$ for small $A, B$
- D2: Compute the adjoint matrix of SE(3) for a given transform
- D3: Left Jacobian of SO(3): implement $J_l(\theta)$ and verify numerically
- D4: Show: $\exp(\text{Ad}_T \cdot \xi) = T \cdot \exp(\xi) \cdot T^{-1}$
- D5: Small-angle: compare $\exp(\delta^\wedge)$ vs $I + \delta^\wedge$ for $\|\delta\| = 0.01, 0.1, 1.0$

### Section E — Manifold Optimization (⭐⭐⭐, 5 problems)
- E1: Rotation averaging: given 10 noisy SO(3) measurements, find Fréchet mean via manifold GN
- E2: 2D pose-graph: 5 poses with SE(2) edges, optimize with manifold Gauss-Newton
- E3: Derive the Jacobian of $r(T) = \text{Log}(T_{\text{meas}}^{-1} \cdot T)$ w.r.t. left perturbation $\delta$
- E4: Compare: Euler-angle parameterization vs manifold for optimizing a rotation (show singularity)
- E5: Implement manifold LM: add damping $\lambda I$ to the normal equations on SE(3)

### Section F — Real-World Applications (⭐⭐⭐⭐, 5 problems)
- F1: IMU preintegration: integrate 100 gyroscope samples on SO(3), compare with ground truth
- F2: Mini visual odometry: estimate SE(3) from 2D-3D point correspondences (PnP as optimization)
- F3: Camera-lidar extrinsic calibration: optimize SE(3) to align point clouds
- F4: Trajectory interpolation: fit a cubic SE(3) spline to 5 keyframe poses
- F5: Covariance on SE(3): propagate uncertainty through a 3-pose chain, visualize as ellipsoids

---

## Python Code: `code/lie_groups/`

### `lie_groups.py` — Core implementations (~600 lines)
```
Planned functions/classes:
- hat(omega) / vee(skew)          — R³ ↔ so(3) conversion
- SO2: exp, log, compose, inverse, matrix
- SO3: exp (Rodrigues), log, compose, inverse, to_quaternion, from_quaternion, slerp
- SE2: exp, log, compose, inverse, transform_point
- SE3: exp, log, compose, inverse, transform_point, adjoint
- Sim3: exp, log (brief)
- left_jacobian_so3(omega)
- right_jacobian_so3(omega)
- adjoint_se3(T)
- manifold_gauss_newton(residual_fn, jacobian_fn, x0, retract_fn, max_iter=50)
- rotation_averaging(rotations, weights=None)
- pose_graph_optimize_2d(poses, edges)
```

### `__init__.py` — Module docstring

### Self-tests: 20+ test cases covering round-trips, identities, and numerical verification

---

## Weekly Lessons: `weeks/week-11/`, `weeks/week-12/`, and `weeks/week-13/`

### Week 11 — Group Theory + SO(2)/SO(3) + Quaternions (2.5 hrs/day × 7 = 17.5 hrs)

| Day | Topic | Theory | Implementation | Practice |
|-----|-------|--------|----------------|----------|
| **Day 71** | Why Rotations Break Euclidean Math | The "add delta" failure, gimbal lock, angle wrapping, motivation for manifolds | Plot: Euler angle interpolation fails vs SLERP | 6 "what goes wrong" thought experiments |
| **Day 72** | Group Theory Crash Course | Group axioms, subgroups, homomorphisms, matrix groups GL/O/SO | Verify SO(2) and SO(3) satisfy all group axioms in code | 8 group classification exercises |
| **Day 73** | SO(2) — 2D Rotations and the Circle Group | Matrix form, Lie algebra so(2), exp/log maps, composition | Implement SO2 class: exp, log, compose, inverse | 10 rotation computation problems |
| **Day 74** | SO(3) — Rotation Matrices and Rodrigues' Formula | Rotation matrix properties, angle-axis, Rodrigues' formula, log map | Implement SO3 class: exp (Rodrigues), log, compose, inverse | Verify exp(log(R)) = R for 50 random rotations |
| **Day 75** | Quaternions Deep Dive | Unit quaternion algebra, Hamilton product, double cover, conversions | Implement quaternion class: multiply, to/from matrix, to/from angle-axis | 8 quaternion chain puzzles |
| **Day 76** | SLERP and Rotation Interpolation | Spherical interpolation, SLERP derivation, squad interpolation | Implement SLERP, compare with naive interpolation, plot trajectories | Interpolation between 5 orientations |
| **Day 77** | Week 11 Review: From Groups to Rotations | Flashcard drill, concept map, representation comparison table | Complete SO2/SO3 self-test suite (20+ tests) | Exercise Set 8 Section A+B |

### Week 12 — SE(2)/SE(3) + Lie Algebra Deep Dive + Jacobians (2.5 hrs/day × 7 = 17.5 hrs)

| Day | Topic | Theory | Implementation | Practice |
|-----|-------|--------|----------------|----------|
| **Day 78** | Manifolds and Tangent Spaces | Topological space → smooth manifold, charts, atlases, tangent space, Riemannian metric | Visualize tangent planes on S², implement chart/atlas for SO(3) | 6 manifold classification problems |
| **Day 79** | SE(2) — 2D Rigid Body Motions | Homogeneous coordinates, Lie algebra se(2), exp/log closed-form, composition | Implement SE2 class: exp, log, compose, inverse, transform_point | Robot path: drive + turn sequences |
| **Day 80** | SE(3) — 3D Rigid Body Transformations | 4×4 homogeneous matrix, twist coordinates ξ, exp via Rodrigues + V matrix | Implement SE3 class: exp, log, compose, inverse, adjoint | Transform point clouds between frames |
| **Day 81** | Sim(3) and the Full Lie Group Zoo | Sim(3) for monocular SLAM, comparative anatomy of all groups, cheat sheet | Implement Sim3 exp/log, build complete cheat sheet table | 6 "which group?" classification problems |
| **Day 82** | Lie Algebra, BCH Formula, and Adjoint | Generators, matrix exponential derivation, BCH to 2nd order, adjoint representation | BCH numerical verification, adjoint_se3() implementation | Hand-derive BCH for so(3) |
| **Day 83** | Jacobians on Lie Groups | Left/right perturbation models, chain rule through exp, left/right Jacobian of SO(3) | Implement left_jacobian_so3, right_jacobian_so3, numerical checker | 6 Jacobian derivation problems (by hand + verify) |
| **Day 84** | Week 12 Review: From SE to Jacobians | Left vs right convention comparison, adjoint properties drill | Complete SE2/SE3/adjoint self-test suite | Exercise Set 8 Section C+D |

### Week 13 — Manifold Optimization + Applications + Capstone (2.5 hrs/day × 7 = 17.5 hrs)

| Day | Topic | Theory | Implementation | Practice |
|-----|-------|--------|----------------|----------|
| **Day 85** | Manifold Gauss-Newton | Retraction, tangent-space linearization, normal equations on manifold, update rule | Implement manifold_gauss_newton solver, test on SO(3) rotation averaging | 4 optimization problems: rotation averaging, pose fitting |
| **Day 86** | Manifold Levenberg-Marquardt + Convergence | Damping on manifolds, trust region, convergence analysis, Euclidean vs manifold comparison | Implement manifold LM, compare convergence with/without damping | Singularity demo: Euler angles fail, manifold succeeds |
| **Day 87** | Pose-Graph Optimization on SE(2) | Pose-graph structure, odometry + loop closure edges, information matrix, SE(2) Jacobians | Build a 10-pose 2D pose-graph optimizer from scratch | Compare output with g2o reference |
| **Day 88** | SE(3) Optimization and 3D SLAM | 3D pose-graph, quaternion parameterization, Schur complement for efficiency | 3D pose-graph optimizer (20 poses + loop closures) | Visualize trajectory before/after correction |
| **Day 89** | Ceres, g2o, and GTSAM Manifold APIs | Ceres SetManifold, GTSAM retract/localCoordinates, g2o vertex types, manif library | Rewrite Day 87 in Ceres C++ or GTSAM, benchmark vs hand-rolled | Library comparison report |
| **Day 90** | IMU Preintegration + Calibration + Splines | Gyro integration on SO(3), preintegration theory, bias correction, hand-eye AX=XB, SE(3) B-splines | Implement preintegration from IMU data, camera-lidar calibration, cubic SE(3) spline | Covariance propagation + uncertainty ellipsoid visualization |
| **Day 91** | Week 13 Capstone: Mini Visual-Inertial Odometry | Full pipeline: SE(3) frame estimation, IMU preintegration, manifold optimization, trajectory output | **Capstone:** Build mini VIO system combining PnP + IMU + manifold GN | Exercise Set 8 Section E+F completion |

---

# MODULE B: GAME THEORY FOR ENGINEERS

## Why This Module Exists

| Engineering domain | Game theory usage |
|---|---|
| Multi-robot coordination | Robots are "players" choosing paths — avoid collisions, share resources |
| Adversarial robotics | Pursuit-evasion, adversarial search, robust control against worst-case |
| Mechanism design | Auction-based task allocation, incentive-compatible resource sharing |
| Network optimization | Routing games, congestion pricing, distributed resource allocation |
| Robust optimization | Minimax = game against nature, worst-case optimization |
| Machine learning | GANs (min-max), adversarial training, multi-agent RL |
| Economics of engineering | Pricing, market equilibria, contract theory |

**Connection to optimization:** Game theory is optimization with multiple competing objective functions. Nash equilibrium is a fixed point of coupled optimization problems. Minimax and saddle points connect directly to duality theory (Chapter 04-05). This module gives 3 full weeks (~52.5 hours) from normal-form games to multi-robot coordination and adversarial ML.

---

## Dependency Graph

```
10-game-theory (Reference Guide)
│
├── Part I: Foundations
│   (normal-form games, strategies, dominance, Nash equilibrium)
│
├── Part II: Zero-Sum Games & Minimax
│   (minimax theorem, LP connection, saddle points)
│
├── Part III: Cooperative Games
│   (coalitions, Shapley value, core, bargaining)
│
├── Part IV: Dynamic & Repeated Games
│   (extensive form, subgame perfection, repeated games, folk theorem)
│
├── Part V: Mechanism Design & Auctions
│   (incentive compatibility, VCG, auction theory)
│
├── Part VI: Algorithmic Game Theory
│   (computing Nash, fictitious play, no-regret learning, correlated equilibrium)
│
└── Part VII: Applications
    (multi-robot, adversarial ML, congestion games, robust optimization)
```

---

## Reference Guide: `10-game-theory.md`

### Section 1 — What Is a Game?
- 1.1 Players, strategies, payoffs — the normal form
- 1.2 Pure vs mixed strategies
- 1.3 Dominant strategy, dominated strategy, iterated elimination
- 1.4 Classic games: Prisoner's Dilemma, Battle of the Sexes, Chicken, Stag Hunt, Matching Pennies
- 1.5 Notation: $G = (N, S_1 \times \cdots \times S_n, u_1, \ldots, u_n)$

### Section 2 — Nash Equilibrium
- 2.1 Definition: no player can improve by unilateral deviation
- 2.2 Pure strategy NE: best response analysis
- 2.3 Mixed strategy NE: indifference principle
- 2.4 **Nash's theorem:** every finite game has at least one (possibly mixed) NE
- 2.5 Worked example: find all NE of a 3×3 game
- 2.6 Multiple equilibria and equilibrium selection (Pareto, risk dominance, focal points)

### Section 3 — Zero-Sum Games and the Minimax Theorem
- 3.1 Definition: $u_1 + u_2 = 0$ — pure competition
- 3.2 Minimax strategy: $\max_x \min_y f(x,y)$
- 3.3 **Von Neumann's minimax theorem:** $\max_x \min_y = \min_y \max_x$ for mixed strategies
- 3.4 Connection to LP: solving zero-sum games via linear programming
- 3.5 Connection to saddle points and convex-concave duality
- 3.6 Worked example: solve a 3×3 zero-sum game graphically and via LP
- 3.7 Minimax in optimization: robust optimization as a game against nature

### Section 4 — Cooperative Games
- 4.1 Coalitional games: characteristic function $v(S)$
- 4.2 The core: set of stable allocations no coalition wants to block
- 4.3 **Shapley value:** fair allocation based on marginal contributions
  - Axioms: efficiency, symmetry, dummy, additivity
  - Formula: $\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}[v(S \cup \{i\}) - v(S)]$
- 4.4 Bargaining: Nash bargaining solution, Kalai-Smorodinsky
- 4.5 Applications: cost sharing, voting power, feature attribution in ML (SHAP!)

### Section 5 — Dynamic and Repeated Games
- 5.1 Extensive form: game trees, information sets
- 5.2 Backward induction and subgame perfect equilibrium
- 5.3 Repeated games: strategy in iterated Prisoner's Dilemma
- 5.4 Folk theorem: cooperation emerges with patient enough players
- 5.5 Stochastic games: state transitions between stages
- 5.6 Worked example: Stackelberg game (leader-follower), centipede game

### Section 6 — Mechanism Design and Auctions
- 6.1 Mechanism design: "inverse game theory" — design rules to get desired outcomes
- 6.2 Revelation principle
- 6.3 Incentive compatibility and individual rationality
- 6.4 VCG mechanism: Vickrey-Clarke-Groves for truthful auctions
- 6.5 Auction types: English, Dutch, first-price sealed-bid, Vickrey (second-price)
- 6.6 Revenue equivalence theorem
- 6.7 Application: task allocation in multi-robot systems via auctions

### Section 7 — Algorithmic Game Theory
- 7.1 Computing Nash equilibrium: Lemke-Howson algorithm, support enumeration
- 7.2 PPAD complexity: finding NE is hard in general
- 7.3 Fictitious play: iterative best response
- 7.4 No-regret learning: regret matching, multiplicative weights
- 7.5 Correlated equilibrium: traffic lights as correlation devices
- 7.6 Online learning connection: bandit algorithms, exploration-exploitation
- 7.7 Convergence guarantees: when do dynamics lead to equilibrium?

### Section 8 — Connections to Optimization
- 8.1 Nash equilibrium as coupled optimization / variational inequality
- 8.2 Zero-sum games and LP duality (direct map)
- 8.3 Minimax and robust optimization
- 8.4 Saddle-point problems: $\min_x \max_y \mathcal{L}(x,y)$ — the Lagrangian!
- 8.5 Potential games: a single potential function whose optima are NE
- 8.6 GANs as zero-sum games: generator vs discriminator
- 8.7 Multi-agent reinforcement learning

### Section 9 — Engineering Applications Gallery
- 9.1 Multi-robot task allocation via auction mechanisms
- 9.2 Pursuit-evasion: minimax trajectory planning
- 9.3 Congestion games: traffic routing as a potential game (Braess's paradox)
- 9.4 Wireless channel allocation: spectrum sharing as a game
- 9.5 Supply chain: Stackelberg pricing between manufacturer and retailer
- 9.6 Robust MPC: minimax control against bounded disturbances
- 9.7 Adversarial ML: evasion attacks as zero-sum games

---

## Exercises: `exercises/09-game-theory.md` (~30 graded problems)

### Section A — Normal Form Games (⭐, 5 problems)
- A1: Find all pure-strategy NE of Prisoner's Dilemma, Battle of the Sexes, Chicken
- A2: Iterated elimination of dominated strategies on a 4×4 game
- A3: Best response correspondence: plot and find intersections for a 2-player game
- A4: Verify Nash's theorem: find a mixed NE for Matching Pennies
- A5: Model a real-world scenario (parking, traffic merge) as a 2-player normal-form game

### Section B — Mixed Strategies & Nash Equilibrium (⭐⭐, 5 problems)
- B1: Compute the mixed NE of a general 2×2 game — derive the indifference equations
- B2: Support enumeration: find all NE of a 3×3 game (pure and mixed)
- B3: Implement `find_nash_2x2(payoff_matrix)` — returns all NE with Python
- B4: Graphical solution: plot best response polyhedra for a 2×3 game
- B5: Multiple equilibria: game with 3 NE — which is Pareto-dominant? risk-dominant?

### Section C — Zero-Sum Games & Minimax (⭐⭐, 5 problems)
- C1: Solve a 3×3 zero-sum game using the graphical method
- C2: Formulate the same game as an LP, solve with `scipy.optimize.linprog`
- C3: Verify the minimax theorem: $\max_p \min_q p^T A q = \min_q \max_p p^T A q$
- C4: Implement minimax for a simple pursuit-evasion on a grid (3×3 grid, 2 steps)
- C5: Connect minimax to duality: show the LP dual gives the other player's strategy

### Section D — Cooperative Games (⭐⭐, 4 problems)
- D1: Compute the Shapley value for a 3-player voting game
- D2: Implement `shapley_value(v)` for arbitrary characteristic function
- D3: Check if an allocation is in the core for a given game
- D4: Cost sharing: 3 towns sharing a water pipeline — find the fair allocation via Shapley

### Section E — Dynamic Games & Mechanism Design (⭐⭐⭐, 5 problems)
- E1: Backward induction: solve a 3-stage sequential game, find SPE
- E2: Implement Vickrey (second-price) auction simulator, verify truthfulness
- E3: Repeated Prisoner's Dilemma: implement tit-for-tat, grim trigger, random; run tournament
- E4: Design a VCG mechanism for allocating 3 tasks to 3 robots with private costs
- E5: Stackelberg game: leader-follower pricing — solve by backward induction and verify with optimization

### Section F — Algorithmic & Applied (⭐⭐⭐⭐, 6 problems)
- F1: Implement fictitious play for a 2-player game, plot convergence to NE
- F2: Implement regret matching (no-regret learning), compare with fictitious play
- F3: Potential game: model a congestion game (3 routes, 100 drivers), find equilibrium, show Braess's paradox
- F4: Multi-robot task allocation: 4 robots, 6 tasks, implement auction-based allocation
- F5: GAN equilibrium: implement a toy 1D GAN (generator and discriminator are single-layer), visualize training dynamics as a zero-sum game
- F6: Robust optimization: $\min_x \max_{\delta \in \Delta} f(x, \delta)$ — solve for simple quadratic with bounded perturbation

---

## Python Code: `code/game_theory/`

### `game_theory.py` — Core implementations (~500 lines)
```
Planned functions/classes:
- NormalFormGame(payoff_matrices)        — create n-player game
- find_pure_nash(game)                   — enumerate pure-strategy NE
- find_mixed_nash_2x2(game)              — analytic mixed NE for 2×2
- support_enumeration(game)              — all NE via support enumeration
- solve_zero_sum_lp(A)                   — zero-sum game via LP
- minimax_value(A)                       — compute minimax value
- shapley_value(v, n)                    — Shapley value for coalitional game
- is_in_core(allocation, v)              — check core membership
- fictitious_play(game, iterations)      — iterative best response
- regret_matching(game, iterations)      — no-regret dynamics
- backward_induction(game_tree)          — solve extensive-form game
- vickrey_auction(bids)                  — second-price auction
- vcg_mechanism(valuations, allocation_fn) — VCG payments
- congestion_game(routes, cost_fns, n_players) — compute equilibrium
- best_response(game, player, opponent_strategy) — BR computation
```

### `__init__.py` — Module docstring

### Self-tests: 20+ test cases

---

## Weekly Lessons: `weeks/week-14/`, `weeks/week-15/`, and `weeks/week-16/`

### Week 14 — Game Theory: Normal Form + Nash + Zero-Sum (2.5 hrs/day × 7 = 17.5 hrs)

| Day | Topic | Theory | Implementation | Practice |
|-----|-------|--------|----------------|----------|
| **Day 92** | What Is a Game? Normal Form and Dominance | Players, strategies, payoffs, dominant/dominated strategies, IESDS | Implement NormalFormGame class with dominance checking | Solve 6 classic games (PD, BoS, Chicken, Stag Hunt) by hand |
| **Day 93** | Pure-Strategy Nash Equilibrium | Best response analysis, pure NE, best response correspondence | Implement find_pure_nash, best_response functions | 8 pure NE problems on 2×2, 3×3 games |
| **Day 94** | Mixed-Strategy Nash Equilibrium | Indifference principle, Nash's existence theorem, support enumeration | Implement find_mixed_nash_2x2, support_enumeration | Graphical: plot best response polyhedra for 2×3 game |
| **Day 95** | Equilibrium Selection and Multiple Equilibria | Pareto dominance, risk dominance, focal points (Schelling), correlated equilibrium | Implement equilibrium ranker (Pareto, risk), analyze games with 3+ NE | Model real scenarios as games: parking, traffic merge |
| **Day 96** | Zero-Sum Games and the Minimax Theorem | Minimax strategy, saddle points, von Neumann's theorem, graphical method | Implement minimax_value, solve_zero_sum_graphical | Solve 3 zero-sum games by graphical method |
| **Day 97** | Zero-Sum Games via LP and Duality | LP formulation of zero-sum games, dual gives opponent's strategy, connection to saddle points | Implement solve_zero_sum_lp using scipy.optimize.linprog | 4 problems: LP ↔ duality ↔ minimax connection |
| **Day 98** | Week 14 Review: Foundations Consolidated | Flashcard drill, concept map (dominance → NE → minimax → LP duality) | Support enumeration for 3×3 game from scratch | Exercise Set 9 Section A+B+C |

### Week 15 — Cooperative + Dynamic Games + Mechanism Design (2.5 hrs/day × 7 = 17.5 hrs)

| Day | Topic | Theory | Implementation | Practice |
|-----|-------|--------|----------------|----------|
| **Day 99** | Cooperative Games and the Core | Coalitional games, characteristic function v(S), the core, stability | Implement is_in_core, visualize core for 3-player games | 4 core membership problems |
| **Day 100** | Shapley Value and Fair Division | Shapley axioms, formula, marginal contribution, connection to SHAP in ML | Implement shapley_value for arbitrary characteristic functions | Cost sharing: 3 towns + water pipeline |
| **Day 101** | Bargaining Theory | Nash bargaining solution, Kalai-Smorodinsky, alternating offers (Rubinstein) | Implement Nash bargaining solver, compare with Kalai-Smorodinsky | Contract negotiation scenario |
| **Day 102** | Dynamic Games and Backward Induction | Extensive form, game trees, information sets, subgame perfect equilibrium | Implement backward_induction, game tree solver and visualizer | Solve centipede game, ultimatum game |
| **Day 103** | Repeated Games and the Folk Theorem | Iterated Prisoner's Dilemma, tit-for-tat, grim trigger, folk theorem, discount factor | Implement strategy tournament: TFT, grim, random, pavlov, etc. | Axelrod tournament reproduction |
| **Day 104** | Mechanism Design and Auctions | Incentive compatibility, revelation principle, VCG, auction types, revenue equivalence | Implement vickrey_auction, vcg_mechanism, first_price_auction | Design 2 mechanisms: truthful task auction |
| **Day 105** | Week 15 Review: Cooperation to Mechanism Design | Stackelberg game deep dive (leader-follower), stochastic games intro | Stackelberg pricing optimization, connect to bilevel programming | Exercise Set 9 Section D+E |

### Week 16 — Algorithmic GT + Applications + Capstone (2.5 hrs/day × 7 = 17.5 hrs)

| Day | Topic | Theory | Implementation | Practice |
|-----|-------|--------|----------------|----------|
| **Day 106** | Computing Nash Equilibria | Lemke-Howson algorithm, PPAD complexity, support enumeration revisited | Implement Lemke-Howson for 2-player games | Complexity analysis: why NE is PPAD-complete |
| **Day 107** | Learning in Games (Day 1) | Fictitious play, convergence properties, best response dynamics | Implement fictitious_play, plot convergence to NE over 1000 iterations | Compare convergence speed across game types |
| **Day 108** | Learning in Games (Day 2) — No-Regret | Regret matching, multiplicative weights, external vs internal regret, bandit connection | Implement regret_matching, multiplicative_weights, convergence plots | Regret matching vs fictitious play tournament |
| **Day 109** | Potential Games, Congestion, and Price of Anarchy | Potential functions, congestion games, Wardrop equilibrium, Braess's paradox, PoA | Implement congestion_game, find equilibrium, demonstrate Braess's paradox | Traffic routing: 3 routes, 100 drivers |
| **Day 110** | Multi-Robot Coordination and Adversarial ML | Task allocation via auctions, pursuit-evasion as minimax, GANs as zero-sum games | Auction-based multi-robot allocator (4 robots, 6 tasks), toy 1D GAN | GAN training dynamics visualization |
| **Day 111** | Robust Optimization as Games Against Nature | Minimax control, robust MPC, worst-case QP, connection to H-infinity control | Robust QP: $\min_x \max_\delta f(x,\delta)$ via duality, compare robust vs nominal | Robust portfolio optimization |
| **Day 112** | Week 16 Capstone: Multi-Robot Delivery Game | Full pipeline: auction-based task allocation + congestion-aware routing + robustness to failures | **Capstone:** 6-robot warehouse delivery — auction allocation, congestion game for routing, robust rerouting on failure | Exercise Set 9 Section F completion |

---

# SUMMARY: What Gets Created

## New Files (Estimated)

| Component | Files | Est. Lines |
|---|---|---|
| `09-lie-groups.md` | 1 reference guide | ~800 |
| `10-game-theory.md` | 1 reference guide | ~700 |
| `exercises/08-lie-groups.md` | 1 exercise set (30 problems) | ~600 |
| `exercises/09-game-theory.md` | 1 exercise set (30 problems) | ~600 |
| `code/lie_groups/lie_groups.py` | Core Python + tests | ~700 |
| `code/lie_groups/__init__.py` | Module init | ~5 |
| `code/game_theory/game_theory.py` | Core Python + tests | ~600 |
| `code/game_theory/__init__.py` | Module init | ~5 |
| `weeks/week-11/` (7 daily files) | Lie Groups: groups + SO(2)/SO(3) | ~2800 |
| `weeks/week-12/` (7 daily files) | Lie Groups: SE + Lie algebra | ~2800 |
| `weeks/week-13/` (7 daily files) | Lie Groups: manifold opt + capstone | ~2800 |
| `weeks/week-14/` (7 daily files) | Game Theory: foundations + Nash | ~2800 |
| `weeks/week-15/` (7 daily files) | Game Theory: cooperative + dynamic | ~2800 |
| `weeks/week-16/` (7 daily files) | Game Theory: algorithms + capstone | ~2800 |
| **Total** | **~46 files** | **~19,800 lines** |

## Updates to Existing Files
- `00-learning-plan.md` — add Weeks 11-16 to schedule
- `CURRICULUM.md` — add Phase VI (Lie Groups) and Phase VII (Game Theory)
- `CONTENT-PLAN.md` — add the new creation tasks
- Converter script — add new pages/code dirs

## Suggested Build Order
1. Reference guides first (09, 10) — establishes the content
2. Exercise sets (08, 09) — graded problems with answers
3. Python code modules — standalone implementations with self-tests
4. Weekly daily lessons — expand into 2.5 hr/day format (6 weeks)
5. Update learning plan, curriculum, converter
6. Convert to HTML and push

## Pacing Rationale (v2)

The original 4-week plan (v1) crammed too much into Weeks 12 and 14:
- Week 12 packed manifold GN, pose-graph, 3D SLAM, IMU preintegration, calibration, AND trajectory interpolation
- Week 14 packed algorithmic GT, 2 days of learning dynamics, congestion, multi-robot, adversarial ML, robust opt, AND capstone

Expanding to 6 weeks (3 + 3) provides:
- **Dedicated days for derivations** — BCH formula, Jacobians, Shapley formula deserve pen-and-paper time
- **Debugging room** — manifold optimization code has subtle sign/convention bugs that need full days to find
- **Richer capstones** — mini VIO (Lie Groups) and multi-robot delivery game (Game Theory) each get a full day
- **2 days for learning dynamics** — fictitious play and regret matching each get their own day instead of being crammed
- **Separate SE/Lie algebra week** — SE(3) adjoint + Jacobians were rushed; now they get proper breathing room
- **Total: 105 hours** (vs 70 hours in v1) for 42 daily lessons

---

## Key Design Decisions

### Lie Groups
- **Depth:** Much deeper than Day 41 — full algebraic foundations through production use
- **Robotics-first:** Every abstraction justified by a robotics application
- **Code-first:** `lie_groups.py` is standalone — students can `import lie_groups` and use it
- **Bridge to existing:** References Chapter 06 (graph optimization), Chapter 08 (matrix essentials)

### Game Theory
- **Engineering angle:** Not an economics course — focus on algorithmic and engineering applications
- **Optimization connection:** Every section explicitly links back to optimization concepts (duality, minimax, saddle points, LP)
- **Multi-robot grounding:** Primary application domain is multi-robot systems (relevant to OKS fleet)
- **Hands-on:** Implement every algorithm — no "prove this theorem" without a corresponding code exercise
- **Modern ML connection:** GANs, adversarial robustness, SHAP values tie to current practice
