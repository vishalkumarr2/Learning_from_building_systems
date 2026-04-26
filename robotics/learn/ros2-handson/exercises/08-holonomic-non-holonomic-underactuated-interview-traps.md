# Exercise08 — Holonomic, Non-Holonomic, and Underactuated Interview Traps

## Companion exercises for `05-holonomic-vs-non-holonomic.md`

**Estimated time:** 30 minutes  
**Prerequisite:** [05-holonomic-vs-non-holonomic.md](../05-holonomic-vs-non-holonomic.md), [04-navfn-vs-smac-search-spaces.md](../04-navfn-vs-smac-search-spaces.md)

**Self-assessment guide:** If you can separate motion constraints from actuation limits without mixing them up, you are answering like an engineer instead of guessing from buzzwords.

---

## Overview

This exercise targets the questions that usually trip people up:

1. Why is a car non-holonomic even though it can still reach many poses?
2. Why is a quadrotor usually called underactuated rather than holonomic?
3. Why does sideways motion matter so much in planning?

---

## Section A — Fast Comparison Sheet

| Term | Main question | Safe interview answer |
| --- | --- | --- |
| Holonomic | Can the robot move directly in each local direction allowed by its configuration? | Yes, it can move instantaneously in all independent local DOFs that matter |
| Non-holonomic | Are there velocity directions that are forbidden even though the pose may still be reachable later? | Yes, local motion is constrained by heading or curvature |
| Underactuated | Do I have fewer independent controls than the motion variables I care about? | Yes, control authority is indirect or incomplete |

### Comparison checklist

Your comparison is strong if it separates:

- motion constraints from control-input limitations
- local feasible motion from eventual reachability
- non-holonomic examples from underactuated examples

- [ ] Done

---

## Section B — Trap Questions with Model Answers

### Question 1 — Why is a differential-drive robot non-holonomic?

**Model answer:**

- Because it cannot generate lateral velocity directly.
- Its instantaneous motion is tied to its heading.
- It can rotate and move forward, but not slide sideways.

### Question 2 — If it can still reach the pose eventually, why is that not holonomic?

**Model answer:**

- Holonomy is about local feasible motion directions, not just final reachability.
- A robot may reach the same pose through a sequence of maneuvers while still being unable to move there directly.

### Question 3 — What is the most important mathematical clue for non-holonomic behavior?

**Model answer:**

- A non-integrable velocity constraint, usually written in a form like `A(q)qdot = 0`.
- The key point is that the constraint lives at the velocity level and cannot be reduced to a pure position constraint.

### Question 4 — Why is a mecanum robot often treated as holonomic?

**Model answer:**

- Because in the common planar model it can command translation in both planar directions plus rotation.
- That means it can correct lateral error directly instead of maneuvering to do it.

### Question 5 — Why is a quadrotor usually called underactuated rather than holonomic?

**Model answer:**

- Because the important issue is that translational motion is achieved indirectly through thrust and attitude control.
- The system has fewer independent direct controls than the full motion variables people care about.

### Question 6 — Is underactuated the same thing as non-holonomic?

**Model answer:**

- No.
- Underactuation is about missing independent control inputs.
- Non-holonomy is about motion constraints on feasible local velocities.

### Question 7 — Why do non-holonomic robots need different planners?

**Model answer:**

- Because free cells on the map are not enough.
- The planner must also respect heading, turning radius, and sometimes reverse motion.

### Question 8 — What is the clean one-sentence distinction?

**Model answer:**

- A holonomic robot can move directly in every allowed local direction, while a non-holonomic robot must maneuver because some local velocity directions are forbidden.

### Interview-answer checklist

Your interview answers are strong if they mention:

- sideways motion as the easiest intuition test for non-holonomy
- that non-holonomic does not mean unreachable, only locally constrained
- that underactuated is a different axis from holonomic vs non-holonomic
- a clean example for each category, such as mecanum, car, and quadrotor

- [ ] Done

---

## Section C — Scenario Questions

### Scenario 1 — Docking correction

Two robots miss a docking pose by 15 cm to the left.

- Robot A: mecanum base
- Robot B: car-like platform

**Questions:**

1. Which robot can correct laterally more directly?
2. Which robot is more likely to need a re-approach?
3. What concept is being tested here?

### Scenario 1 Answer

- The mecanum base can correct more directly.
- The car-like platform is more likely to need a re-approach.
- The concept is local motion feasibility: holonomic correction versus non-holonomic maneuvering.

- [ ] Done

---

### Scenario 2 — Interview trap: car vs quadrotor

An interviewer asks:

> "Which one is non-holonomic: a car or a quadrotor?"

**Best answer shape:**

1. What should you say about the car?
2. What should you say about the quadrotor?
3. Why is this a trap question?

### Scenario 2 Answer

- A car is the clean classical non-holonomic example because it cannot move sideways instantaneously.
- A quadrotor is more cleanly described as underactuated in its full dynamic model.
- It is a trap because it checks whether you confuse motion constraints with actuation structure.

### Scenario checklist

Your scenario answers are strong if they identify:

- which property is being tested in the scenario
- why the car example is kinematic and the quadrotor example is actuation-related
- what mistake the interviewer is hoping you make

- [ ] Done

---

## Practical Takeaway

Before choosing a planner or controller, ask:

1. Can the robot translate directly in the direction of its error?
2. Does heading constrain local feasible motion?
3. Are the dominant limits kinematic, dynamic, or actuation-related?

That is how you avoid mixing up holonomic, non-holonomic, and underactuated systems.