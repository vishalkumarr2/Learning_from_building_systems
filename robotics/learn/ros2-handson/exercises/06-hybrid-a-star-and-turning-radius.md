# Exercise06 — Hybrid-A* and Minimum Turning Radius

## Companion exercises for `03-nav2-architecture.md` and `04-navfn-vs-smac-search-spaces.md`

**Estimated time:** 30 minutes  
**Prerequisite:** [03-nav2-architecture.md](../03-nav2-architecture.md), [04-navfn-vs-smac-search-spaces.md](../04-navfn-vs-smac-search-spaces.md)

**Self-assessment guide:** If you can look at a path, a corridor width, and a
`minimum_turning_radius` value and predict whether Smac will struggle, you are thinking in
vehicle kinematics rather than only in grid cells.

---

## Overview

This exercise focuses on one idea only:

> A path can be free in the map and still be infeasible for the robot.

Hybrid-A* exists to close exactly that gap.

---

## Section A — State-Space Reasoning

### Question 1 — Why isn't `(x, y)` enough?

Answer in your own words:

1. Why can two states at the same map location still be different for a car-like robot?
2. Why does heading matter near narrow turns or docking goals?
3. Why is this less critical for simple grid planning?

### Question 1 Answer

- Because a car-like robot also has orientation, so `(x, y, 0°)` and `(x, y, 180°)` are
  different physical situations.
- Heading matters because the robot cannot instantly rotate into any direction while
  moving; it must approach with a feasible curvature.
- In plain grid planning, the planner only cares whether cells are traversable, not
  whether the robot's steering geometry can realize that path smoothly.

- [ ] Done

---

## Section B — Turning Radius Intuition

### Question 2 — What does `minimum_turning_radius` really mean?

Explain these three points:

1. What physical robot limitation does `minimum_turning_radius` represent?
2. What happens if you set it too small in SmacPlanner?
3. What happens if you set it too large?

### Question 2 Answer

- It represents the tightest circle the robot can physically follow while steering.
- If it is too small, the planner may return paths that look executable in simulation but
  ask the real robot to turn more sharply than it actually can.
- If it is too large, the planner becomes overly conservative and may reject feasible
  maneuvers or produce unnecessarily long detours.

- [ ] Done

---

## Section C — Corridor Scenario

### Scenario 1 — Narrow aisle turn

A car-like AMR must turn into a charging bay from a corridor.

- corridor width: `2.4 m`
- charging bay entrance offset: sharp right turn
- robot minimum turning radius in reality: `1.1 m`
- SmacPlanner configured `minimum_turning_radius: 0.5`

**Questions:**

1. What planning mistake is likely here?
2. Why might simulation look better than reality?
3. What should you fix first?

### Scenario 1 Answer

- The planner is assuming the robot can turn much tighter than it really can.
- Simulation may appear fine because the path is geometrically possible in the planner's
  model, even though the real vehicle cannot physically achieve that curvature.
- Fix `minimum_turning_radius` first so the planner's search space matches the robot's
  actual steering limits.

- [ ] Done

---

### Scenario 2 — Planner says no path, operator can drive it

Same robot, new configuration:

```yaml
motion_model_for_search: DUBIN
minimum_turning_radius: 2.0
```

The operator says the robot can make the maneuver with a short reverse correction, but
Smac reports no path.

**Questions:**

1. What two reasons could explain the planner failure?
2. Which parameter reflects the missing operational behavior most directly?
3. What does this teach you about policy vs physics?

### Scenario 2 Answer

- The turning radius may be configured too conservatively, and the planner is also limited
  to forward-only motion.
- `motion_model_for_search`; changing from `DUBIN` to `REEDS_SHEPP` adds reverse
  maneuvers to the search.
- A robot can be physically capable of a maneuver that the current planning policy still
  forbids. You must model both the hardware limits and the allowed behavior.

- [ ] Done

---

## Section D — `Hybrid-A*` interview questions

### Question 3 — Short interview answers

Answer each in 1 to 3 sentences.

1. Why is `Hybrid-A*` more expensive than plain A*?
2. Why is it often still worth using?
3. What kind of bug do you get if `minimum_turning_radius` is wrong?

### Question 3 Answer

- `Hybrid-A*` searches over position plus heading, often with motion primitives, so the
  state space is much larger than a plain `(x, y)` grid.
- It is worth using because it returns paths that are far closer to what a real
  non-holonomic robot can actually follow.
- You usually get a model mismatch: either the planner returns infeasible turns, or it
  becomes too conservative and rejects maneuvers the robot could really do.

- [ ] Done

---

## Practical Takeaway

When tuning SmacPlanner, check these in order:

1. Is the robot model forward-only or forward-and-reverse?
2. Is `minimum_turning_radius` measured from the actual platform, not guessed?
3. Does the goal require approach orientation, not just goal position?
4. Is the map free but the maneuver still kinematically impossible?

That is the quickest way to diagnose why NavFn succeeds on paper while Smac rejects the
same route.