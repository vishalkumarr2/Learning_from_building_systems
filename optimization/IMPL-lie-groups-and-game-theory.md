# Implementation Plan: Lie Groups & Game Theory (6 Weeks)
### Based on PLAN-lie-groups-and-game-theory.md v2

---

## Scope

Create ~46 new files (~19,800 lines) for 2 modules:
- **Module A** — Lie Groups & Manifold Optimization (Weeks 11-13, Days 71-91)
- **Module B** — Game Theory (Weeks 14-16, Days 92-112)

Plus updates to 4 existing files.

---

## Pre-Implementation Status

| Component | State |
|-----------|-------|
| PLAN file (v2) | ✅ Complete — full 6-week daily breakdown |
| Weeks 1-4 daily content | ✅ Exist (28 files, pattern established) |
| Weeks 5-10 daily content | ❌ Empty dirs — not blocking, but no immediate predecessors |
| Reference guides 01-08 | ✅ Exist |
| Exercise sets 01-07 | ✅ Exist |
| Code modules (week01-04, matrix_mastery) | ✅ Exist |
| Converter script | ✅ Works for weeks 1-4, needs config additions |

---

## Established Patterns (from existing files)

### Daily Lesson File (`weeks/week-XX/day-NN-slug.md`)
```
# Day N: Title

> **Phase X — Module Name | Week Y | 2.5 hours**
> *One-line motivation.*

## Navigation
- Previous / Next / Week / Curriculum links

## OKS Relevance
Short paragraph connecting to OKS robot systems.

---

## Theory (45 min)
### N.1 First concept — equations, explanations
### N.2 Second concept — equations, explanations

---

## Implementation (60 min)
Code blocks with full imports, runnable, produces output.

---

## Practice Problems (45 min)
≥3 problems with <details><summary>Answer</summary> blocks.

---

## Expert Challenges
≥3 advanced problems.

---

## Connections
- Back-link to prereq days
- Forward-link to dependent days
- OKS relevance recap

## Self-Check
- [ ] Checklist items
```

**Typical size:** ~140-200 lines per daily file.

### Reference Guide (`09-lie-groups.md`)
- Long-form reference with sections, equations, tables
- Companion to daily lessons (not a replacement)
- ~500-800 lines

### Exercise Set (`exercises/08-lie-groups.md`)
- 30 problems, difficulty-graded (⭐ to ⭐⭐⭐⭐)
- Sections A-F, with `<details>` answer blocks
- ~600-1400 lines

### Code Module (`code/lie_groups/`)
- `__init__.py` + main implementation file
- Docstrings, self-tests (`--test` flag), demo mode
- numpy/scipy only, matplotlib for plots
- ~500-700 lines

---

## Build Phases

### Phase 0: Infrastructure (1 session, ~10 min)

| # | Task | Output |
|---|------|--------|
| 0.1 | Create directory structure | `weeks/week-{11..16}/`, `code/lie_groups/`, `code/game_theory/` |
| 0.2 | Create placeholder `__init__.py` files | 2 files |

**No dependencies. Can run immediately.**

---

### Phase 1: Reference Guides (1 session, ~45 min)

These establish the theoretical backbone — all daily lessons reference back to them.

| # | Task | Output | Lines |
|---|------|--------|-------|
| 1.1 | Write `09-lie-groups.md` | Reference guide: groups → SO/SE → Lie algebra → manifold opt → applications | ~800 |
| 1.2 | Write `10-game-theory.md` | Reference guide: normal form → Nash → cooperative → dynamic → algorithmic → applications | ~700 |

**Depends on:** Phase 0
**Pattern:** Follow `08-matrix-essentials.md` structure (sections, equations, tables, OKS relevance).

---

### Phase 2: Python Code Modules (1 session, ~40 min)

Standalone implementations that daily lessons import and extend.

| # | Task | Output | Lines |
|---|------|--------|-------|
| 2.1 | Write `code/lie_groups/lie_groups.py` | SO2, SO3, SE2, SE3, exp/log maps, Jacobians, manifold GN | ~700 |
| 2.2 | Write `code/game_theory/game_theory.py` | NormalFormGame, Nash solvers, Shapley, backward induction, auctions, learning dynamics | ~600 |

**Depends on:** Phase 0
**Pattern:** Follow `code/matrix_mastery/matrix_essentials.py` (docstring, self-tests, demo mode).
**Can run in parallel with Phase 1.**

---

### Phase 3: Exercise Sets (1 session, ~40 min)

| # | Task | Output | Lines |
|---|------|--------|-------|
| 3.1 | Write `exercises/08-lie-groups.md` | 30 problems: Sections A (groups), B (SO/SE), C (exp/log), D (Jacobians), E (manifold opt), F (applications) | ~600 |
| 3.2 | Write `exercises/09-game-theory.md` | 30 problems: Sections A (normal form), B (Nash), C (zero-sum/LP), D (cooperative), E (dynamic), F (algorithmic + applications) | ~600 |

**Depends on:** Phase 1 (reference guides define the content)
**Pattern:** Follow `exercises/07-matrix-mastery.md` (difficulty stars, `<details>` answers).

---

### Phase 4: Weekly Daily Lessons — Lie Groups (3 sessions, ~60 min each)

Each session = 1 week = 7 daily files.

#### Session 4A: Week 11 — Group Theory + SO(2)/SO(3) (Days 71-77)

| # | File | Topic | Key Content |
|---|------|-------|-------------|
| 4A.1 | `day-71-group-axioms.md` | Group Axioms and Matrix Groups | Closure, associativity, identity, inverse; GL, O, SO |
| 4A.2 | `day-72-so2-rotations.md` | SO(2) — 2D Rotations | Rotation matrices, angle representation, group structure |
| 4A.3 | `day-73-so3-rodrigues.md` | SO(3) — Rodrigues' Formula | Axis-angle, Rodrigues, rotation composition |
| 4A.4 | `day-74-quaternions.md` | Quaternions for Rotation | Hamilton's quaternions, unit quaternions ↔ SO(3), double cover |
| 4A.5 | `day-75-quaternion-operations.md` | Quaternion Interpolation and Averaging | SLERP, quaternion averaging, Karcher mean |
| 4A.6 | `day-76-rotation-representations.md` | Rotation Representations Compared | Euler angles, axis-angle, quaternion, matrix — tradeoffs, singularities |
| 4A.7 | `day-77-week11-review.md` | Week 11 Review + Exercises | Flashcard drill, concept map, Exercise Set 8 Section A+B |

#### Session 4B: Week 12 — SE(2)/SE(3) + Lie Algebra (Days 78-84)

| # | File | Topic | Key Content |
|---|------|-------|-------------|
| 4B.1 | `day-78-se2-rigid-motions.md` | SE(2) — 2D Rigid Motions | Homogeneous coordinates, composition, inverse |
| 4B.2 | `day-79-se3-rigid-body.md` | SE(3) — 3D Rigid Body Transforms | 4×4 matrices, twist representation, screw motions |
| 4B.3 | `day-80-exponential-map.md` | Exponential and Logarithmic Maps | $\exp: \mathfrak{g} \to G$, $\log: G \to \mathfrak{g}$, closed-form for SO(3)/SE(3) |
| 4B.4 | `day-81-lie-algebra.md` | Lie Algebra Deep Dive | Generators, Lie bracket, structure constants, BCH formula |
| 4B.5 | `day-82-adjoint-bch.md` | Adjoint Representation and BCH | Adjoint action, adjoint matrix, BCH truncation |
| 4B.6 | `day-83-jacobians.md` | Left and Right Jacobians | Left/right Jacobians of SO(3)/SE(3), perturbation models |
| 4B.7 | `day-84-week12-review.md` | Week 12 Review + Exercises | Concept map (algebra ↔ group ↔ Jacobians), Exercise Set 8 Section C+D |

#### Session 4C: Week 13 — Manifold Optimization + Capstone (Days 85-91)

| # | File | Topic | Key Content |
|---|------|-------|-------------|
| 4C.1 | `day-85-manifold-gauss-newton.md` | Manifold Gauss-Newton | Retraction, tangent-space linearization, update rule |
| 4C.2 | `day-86-pose-graph-se2.md` | Pose-Graph SLAM on SE(2) | 2D SLAM with Lie group poses, compare to Euclidean |
| 4C.3 | `day-87-pose-graph-se3.md` | Pose-Graph SLAM on SE(3) | 3D SLAM, robust kernels, manifold LM |
| 4C.4 | `day-88-ceres-gtsam-manifold.md` | Ceres & GTSAM for Manifold Optimization | Local parameterization in Ceres, GTSAM manifold factors |
| 4C.5 | `day-89-imu-preintegration.md` | IMU Preintegration | Forster's method, preintegrated measurements on SO(3)×ℝ⁶ |
| 4C.6 | `day-90-calibration-interpolation.md` | Calibration + Trajectory Interpolation | Hand-eye AX=XB on SE(3), Slerp/Squad for SE(3) trajectories |
| 4C.7 | `day-91-week13-capstone.md` | Week 13 Capstone: Mini VIO | SE(3) frame estimation + IMU preintegration + manifold GN |

**Depends on:** Phase 1 (reference guide 09), Phase 2 (lie_groups.py)

---

### Phase 5: Weekly Daily Lessons — Game Theory (3 sessions, ~60 min each)

#### Session 5A: Week 14 — Normal Form + Nash + Zero-Sum (Days 92-98)

| # | File | Topic | Key Content |
|---|------|-------|-------------|
| 5A.1 | `day-92-normal-form-games.md` | Normal Form and Dominance | Players, strategies, payoffs, IESDS |
| 5A.2 | `day-93-pure-nash.md` | Pure-Strategy Nash Equilibrium | Best response, pure NE, best response correspondence |
| 5A.3 | `day-94-mixed-nash.md` | Mixed-Strategy Nash Equilibrium | Indifference principle, support enumeration |
| 5A.4 | `day-95-equilibrium-selection.md` | Equilibrium Selection | Pareto/risk dominance, focal points, correlated equilibrium |
| 5A.5 | `day-96-zero-sum-minimax.md` | Zero-Sum Games and Minimax | Minimax strategy, saddle points, graphical method |
| 5A.6 | `day-97-zero-sum-lp.md` | Zero-Sum via LP and Duality | LP formulation, dual ↔ opponent's strategy |
| 5A.7 | `day-98-week14-review.md` | Week 14 Review | Concept map, Exercise Set 9 Section A+B+C |

#### Session 5B: Week 15 — Cooperative + Dynamic + Mechanism Design (Days 99-105)

| # | File | Topic | Key Content |
|---|------|-------|-------------|
| 5B.1 | `day-99-cooperative-core.md` | Cooperative Games and the Core | Coalitional games, characteristic function, core |
| 5B.2 | `day-100-shapley-value.md` | Shapley Value and Fair Division | Axioms, formula, SHAP connection |
| 5B.3 | `day-101-bargaining.md` | Bargaining Theory | Nash bargaining, Kalai-Smorodinsky, Rubinstein |
| 5B.4 | `day-102-dynamic-games.md` | Dynamic Games and Backward Induction | Extensive form, SPE, game trees |
| 5B.5 | `day-103-repeated-games.md` | Repeated Games and Folk Theorem | IPD, tit-for-tat, grim trigger, Axelrod tournament |
| 5B.6 | `day-104-mechanism-design.md` | Mechanism Design and Auctions | VCG, revelation principle, auction types |
| 5B.7 | `day-105-week15-review.md` | Week 15 Review | Stackelberg, Exercise Set 9 Section D+E |

#### Session 5C: Week 16 — Algorithmic GT + Capstone (Days 106-112)

| # | File | Topic | Key Content |
|---|------|-------|-------------|
| 5C.1 | `day-106-computing-nash.md` | Computing Nash Equilibria | Lemke-Howson, PPAD complexity |
| 5C.2 | `day-107-fictitious-play.md` | Learning in Games: Fictitious Play | Convergence, best response dynamics |
| 5C.3 | `day-108-no-regret.md` | No-Regret Learning | Regret matching, multiplicative weights |
| 5C.4 | `day-109-congestion-games.md` | Potential Games and Congestion | Braess's paradox, Price of Anarchy |
| 5C.5 | `day-110-multi-robot-adversarial.md` | Multi-Robot + Adversarial ML | Auction-based allocation, GAN as zero-sum |
| 5C.6 | `day-111-robust-optimization.md` | Robust Optimization as Games | Minimax control, robust MPC, H∞ |
| 5C.7 | `day-112-week16-capstone.md` | Capstone: Multi-Robot Delivery Game | Auction + congestion routing + robust rerouting |

**Depends on:** Phase 1 (reference guide 10), Phase 2 (game_theory.py)

---

### Phase 6: Integration & Existing File Updates (1 session, ~20 min)

| # | Task | File | Change |
|---|------|------|--------|
| 6.1 | Add Weeks 11-16 to learning plan | `00-learning-plan.md` | Append Phase VI + VII schedule |
| 6.2 | Add Phase VI + VII to curriculum | `CURRICULUM.md` | Add ~200 lines after Day 70 |
| 6.3 | Add creation tasks to content plan | `CONTENT-PLAN.md` | Append Batch 7 + Batch 8 |
| 6.4 | Update converter script | `scripts/convert_optimization_to_html.py` | Add to OVERVIEW_PAGES, CODE_DIRS, WEEK_DAYS, WEEK_NAMES, PHASE_INFO |

**Depends on:** Phases 4-5 (need filenames finalized)

---

### Phase 7: HTML Conversion & Push (1 session, ~15 min)

| # | Task | Command |
|---|------|---------|
| 7.1 | Run converter | `python3 scripts/convert_optimization_to_html.py` |
| 7.2 | Verify HTML output | Open in browser, check links |
| 7.3 | Push to GitHub Pages | `cd ~/Learning_from_building_systems && git add . && git commit && git push` |

**Depends on:** Phase 6

---

## Dependency Graph

```
Phase 0 (infra)
├── Phase 1 (reference guides)
│   ├── Phase 3 (exercise sets)
│   ├── Phase 4 (Lie Groups weekly lessons) ← also needs Phase 2
│   └── Phase 5 (Game Theory weekly lessons) ← also needs Phase 2
├── Phase 2 (code modules) ← parallel with Phase 1
│   ├── Phase 4
│   └── Phase 5
│
Phase 4 + 5 complete
└── Phase 6 (integration)
    └── Phase 7 (HTML + push)
```

**Parallelizable pairs:**
- Phase 1 ∥ Phase 2 (reference guides and code modules are independent)
- Phase 4 ∥ Phase 5 (Lie Groups and Game Theory weekly lessons are independent)
- Sessions 4A, 4B, 4C are sequential (each week builds on prior)
- Sessions 5A, 5B, 5C are sequential (each week builds on prior)

---

## Session Plan (Recommended Execution Order)

| Session | Phase | Content | Est. Time | Subagents |
|---------|-------|---------|-----------|-----------|
| **S1** | 0 + 1 + 2 | Infra + reference guides + code modules | ~90 min | 2 parallel (guides ∥ code) |
| **S2** | 3 | Exercise sets | ~40 min | 2 parallel (lie groups ∥ game theory) |
| **S3** | 4A | Week 11 daily lessons (7 files) | ~60 min | sequential |
| **S4** | 4B | Week 12 daily lessons (7 files) | ~60 min | sequential |
| **S5** | 4C | Week 13 daily lessons (7 files) | ~60 min | sequential |
| **S6** | 5A | Week 14 daily lessons (7 files) | ~60 min | sequential |
| **S7** | 5B | Week 15 daily lessons (7 files) | ~60 min | sequential |
| **S8** | 5C | Week 16 daily lessons (7 files) | ~60 min | sequential |
| **S9** | 6 + 7 | Integration + HTML + push | ~35 min | sequential |

**Total: 9 sessions**

---

## File Naming Convention

### Daily Lessons
```
weeks/week-11/day-71-group-axioms.md
weeks/week-11/day-72-so2-rotations.md
...
weeks/week-16/day-112-week16-capstone.md
```

Pattern: `day-{DD}-{kebab-case-slug}.md`

### Code Modules
```
code/lie_groups/__init__.py
code/lie_groups/lie_groups.py
code/game_theory/__init__.py
code/game_theory/game_theory.py
```

---

## Quality Gates (per daily file)

Inherited from CONTENT-PLAN.md:

| Gate | Check |
|------|-------|
| Theory depth | ≥3 key concepts with equations, ≥1 diagram description |
| Runnable code | Every code block has correct imports, produces output |
| Hands-on count | ≥3 implementation exercises (5 on capstone days) |
| Expert challenges | ≥3 advanced problems with hidden solutions |
| Robotics tie-in | ≥1 exercise references a real robotics use case |
| Connections | Back-link + forward-link + OKS relevance |
| Self-check | 3+ checklist items |

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Lie algebra notation inconsistency | Establish convention in reference guide Section 4 before daily files |
| Game theory LP formulation varies by source | Use Boyd & Vandenberghe convention (consistent with Chapter 05) |
| Manifold code has subtle sign bugs | Include self-tests in lie_groups.py that verify $\exp(\log(R)) = R$ |
| Exercise difficulty calibration | Review against existing exercise sets (07) for consistent difficulty |
| Weeks 5-10 don't exist yet | Not a blocker — Weeks 11-16 reference Chapters 01-08, not Week 5-10 daily files |
