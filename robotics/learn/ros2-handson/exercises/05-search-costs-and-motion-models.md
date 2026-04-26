# Exercise05 — Search Costs and Motion Models

## Companion exercises for `03-nav2-architecture.md`

**Estimated time:** 35 minutes  
**Prerequisite:** [03-nav2-architecture.md](../03-nav2-architecture.md)

**Self-assessment guide:** If you can explain why a planner chose a path using
`g(n)`, `h(n)`, and the selected motion model, you are no longer memorizing Nav2 terms —
you are reasoning like the planner.

---

## Overview

This exercise set builds intuition for two separate but easy-to-mix ideas:

1. **Search cost functions** — why Dijkstra and A* expand different nodes
2. **Motion models** — why Dubins and Reeds-Shepp produce different feasible paths

In practice, field debugging gets easier once you stop asking only "Did the planner find
a path?" and start asking:

- What state space was it searching?
- What cost function pushed it toward that path?
- Was reverse motion allowed?

---

## Section A — Dijkstra vs A*

Use this tiny grid for the first two questions:

```text
S . . .
. X . .
. X . G
. . . .
```

Movement is 4-connected: up, down, left, right. Each move costs 1.

---

### Question 1 — What does Dijkstra optimize?

1. Write the scoring function used by Dijkstra.
2. Why does it expand outward like a ripple instead of moving directly toward the goal?
3. In what situation is that still useful?

### Question 1 Answer

- Dijkstra uses:

$$
f(n) = g(n)
$$

- It only cares about the exact cost from the start to the current node. It has no
   goal-directed heuristic, so every node at cost 2 is considered before any node at cost
   3, regardless of whether it points toward or away from the goal.

- It is useful when you want the true shortest-path cost field over the whole map, or
   when you have no good admissible heuristic.

- [ ] Done

---

### Question 2 — Why does A* usually get there faster?

1. Write the A* scoring function.
2. If you use Manhattan distance on this grid, what is the heuristic at the start cell
   if `S = (0,0)` and `G = (3,2)` using `(x, y) = (column, row)`?
3. Why can A* still return the optimal path when the heuristic is admissible?

### Question 2 Answer

- A* uses:

$$
f(n) = g(n) + h(n)
$$

- Manhattan distance:

$$
h(S) = |0 - 3| + |0 - 2| = 5
$$

- If the heuristic never overestimates the true remaining path cost, A* never becomes
   over-optimistic in a way that would skip a cheaper valid solution. It stays both
   goal-directed and optimal.

- [ ] Done

---

## Section B — Choosing the Motion Model

### Question 3 — Dubins or Reeds-Shepp?

For each platform, choose the better motion model and explain why.

1. A small fixed-wing UAV that must keep moving forward
2. A forklift placing a pallet in a tight bay
3. A car-like tugger that is allowed to reverse but mostly operates in wide aisles

### Question 3 Answer

- **Dubins** — forward-only motion matches the platform constraint.
- **Reeds-Shepp** — tight maneuvering with reversing is exactly what the model adds.
- **Usually Reeds-Shepp**, if reverse maneuvers are operationally allowed. Even if most
   motion is forward, the ability to back up can shorten or rescue tight maneuvers.
   If policy forbids reversing during normal operation, then Dubins is the correct model
   even though the platform could reverse physically.

- [ ] Done

---

### Question 4 — Why can Reeds-Shepp find shorter paths?

Answer all three:

1. What new capability does Reeds-Shepp add relative to Dubins?
2. What is a **cusp** in this context?
3. Why does allowing reverse motion often reduce the total path length in tight spaces?

### Question 4 Answer

- Reverse motion.
- A cusp is a point where the vehicle changes travel direction, for example from forward
   to reverse.
- Because the robot does not need to commit to a wide forward-only turning maneuver. It
   can back up, reorient, and continue, which often creates a shorter feasible path than
   any purely forward alternative.

- [ ] Done

---

## Section C — Nav2 Configuration Reasoning

### Scenario 1 — Tight Parking Stall

You are tuning SmacPlanner for a robot that must enter a narrow charging bay by backing
in. Current settings:

```yaml
motion_model_for_search: DUBIN
minimum_turning_radius: 1.2
```

**Questions:**

1. What is the likely planning limitation here?
2. Which parameter should change first?
3. Why might the robot fail even if the goal pose is reachable in the real world?

### Scenario 1 Answer

- The planner is restricted to forward-only maneuvers, so it cannot search backing-in
   trajectories.
- Change `motion_model_for_search` from `DUBIN` to `REEDS_SHEPP`.
- Because the search space excludes valid reverse maneuvers. The planner may report no
   path or return an awkward long path even though a human driver would simply reverse
   into the bay.

- [ ] Done

---

### Scenario 2 — Planner Is Too Slow

A warehouse robot operates in wide, simple corridors. Reverse is disabled by policy.
SmacPlanner with `REEDS_SHEPP` works, but planning latency is high.

**Questions:**

1. What simpler search choice could reduce planner complexity while matching policy?
2. What is the tradeoff?
3. When would you keep Reeds-Shepp anyway?

### Scenario 2 Answer

- Use `DUBIN` instead of `REEDS_SHEPP`.
- You reduce the search space by removing reverse maneuvers, which can improve planning
   speed, but you also lose the ability to exploit reverse motion when it would rescue a
   tight situation.
- Keep Reeds-Shepp if reverse motion may be needed for recovery, docking, or confined
   goal poses even if the nominal task is mostly forward.

- [ ] Done

---

## Stretch Exercise

Write a short answer to this prompt in 5 sentences or fewer:

> "NavFn with A* and Smac with Reeds-Shepp are both path planners. Why are they not
> interchangeable?"

**Target answer shape:** mention grid vs kinematic state space, heuristic search,
turn-radius constraints, and reverse motion.

- [ ] Done

---

## Practical Takeaway

When a Nav2 path looks strange, diagnose it in this order:

1. **Search space:** 2D grid or SE2 kinematic search?
2. **Scoring rule:** Dijkstra-style or A*-style?
3. **Motion model:** Dubins or Reeds-Shepp?
4. **Robot limits:** does `minimum_turning_radius` reflect reality?
5. **Operational policy:** is reverse motion allowed in the actual deployment?

That checklist usually gets you to the root cause faster than staring at the path alone.