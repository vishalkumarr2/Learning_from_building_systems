# Exercise07 — Bellman-Ford, Dijkstra, and A* Interview Traps

## Companion exercises for `03-nav2-architecture.md`

**Estimated time:** 35 minutes  
**Prerequisite:** [03-nav2-architecture.md](../03-nav2-architecture.md), [Exercise05 — Search Costs and Motion Models](05-search-costs-and-motion-models.md)

**Self-assessment guide:** If you can explain when Dijkstra fails, why Bellman-Ford keeps fixing distances, and why A* needs an admissible heuristic, you understand search behavior instead of memorizing algorithm names.

---

## Overview

This page is designed for exactly the kind of interview follow-up questions that catch people off guard:

1. Why does A* fail when the heuristic overestimates?
2. Why does consistency matter even if admissibility already gives correctness?
3. Why does Dijkstra break on negative edges?
4. Why can Bellman-Ford recover from those cases?

In Nav2, Bellman-Ford is not the planner you usually deploy. But if you understand Bellman-Ford, you understand something deeper:

> When can a shortest-path algorithm trust its first answer, and when must it keep correcting itself?

That is a very common interview axis.

---

## Section A — Bellman-Ford vs Dijkstra vs A* Cheat Sheet

| Algorithm | Scoring idea | Handles negative edges? | Needs heuristic? | Main strength | Main failure mode |
| --- | --- | --- | --- | --- | --- |
| Dijkstra | Expand the node with smallest `g(n)` | No | No | Exact shortest paths for non-negative graphs | Breaks if a later negative edge makes a finalized node cheaper |
| A* | Expand the node with smallest `f(n) = g(n) + h(n)` | Not in the standard shortest-path guarantee sense | Yes | Goal-directed search that can stay optimal with a good heuristic | Overestimating heuristic can hide the true optimal path |
| Bellman-Ford | Relax every edge repeatedly | Yes | No | Works with negative edges and detects negative cycles | Slower because it keeps revisiting estimates |

### Quick memory hooks

- **Dijkstra**: "I trust the cheapest known frontier node now."
- **A***: "I trust the cheapest frontier node after adjusting for promise."
- **Bellman-Ford**: "I do not trust early answers, so I keep repairing them."

### When each one is the right answer

- Use **Dijkstra** when edges are non-negative and you want exact shortest paths.
- Use **A*** when you have a good admissible heuristic and care about one goal.
- Use **Bellman-Ford** when negative edges may exist or you must detect negative cycles.

### Cheat-sheet checklist

Your comparison is strong if it clearly states:

- which algorithm is greedy and which one keeps repairing estimates
- which one depends on a heuristic
- which one tolerates negative edges
- the main tradeoff between speed and generality

- [ ] Done

---

## Section B — Trap-Style Interview Questions with Model Answers

### Question 1 — Why can A* return a wrong path if `h(n)` overestimates?

**Model answer:**

- A* ranks nodes using `f(n) = g(n) + h(n)`.
- If `h(n)` overestimates, the truly optimal branch can look artificially expensive.
- That can cause A* to prefer a worse branch and lose its optimality guarantee.
- That is why admissibility requires `h(n)` to never exceed the true remaining cost.

### Question 2 — Admissible vs consistent: what is the real difference?

**Model answer:**

- **Admissible** means the heuristic never overestimates the true remaining cost.
- **Consistent** means the heuristic also behaves smoothly across edges: `h(n) <= c(n, n') + h(n')`.
- Admissibility protects optimality.
- Consistency makes A* cleaner operationally because it avoids many node re-openings.

### Question 3 — Why does Dijkstra fail with negative weights?

**Model answer:**

- Dijkstra assumes once the current cheapest node is finalized, no later path can improve it.
- That assumption is only valid if all edge weights are non-negative.
- A negative edge can create a cheaper path after a node was already considered done.
- So the greedy "finalize once" logic breaks.

### Question 4 — Why does Bellman-Ford not have the same problem?

**Model answer:**

- Bellman-Ford does not trust early estimates.
- It relaxes all edges repeatedly, so a better path discovered later can still repair an earlier answer.
- That is exactly why it handles negative edges correctly.

### Question 5 — Why exactly `V - 1` rounds in Bellman-Ford?

**Model answer:**

- A simple shortest path cannot contain more than `V - 1` edges in a graph with `V` vertices.
- Each relaxation pass can correctly propagate shortest-path information across one more edge.
- So after `V - 1` rounds, every valid shortest path has had enough passes to fully propagate.

### Question 6 — Why does one extra Bellman-Ford pass detect a negative cycle?

**Model answer:**

- After `V - 1` rounds, all shortest simple paths should already be stable.
- If a further relaxation still improves a distance, the only explanation is that a cycle is making the path cheaper.
- If that cycle has negative total cost, you can keep looping and reduce the path forever.
- So a finite shortest path no longer exists.

### Question 7 — If the heuristic in A* is too small, is that bad?

**Model answer:**

- It is not wrong, but it is less useful.
- A* stays optimal if the heuristic remains admissible.
- But as the heuristic becomes less informative, A* explores more nodes and drifts toward Dijkstra.

### Question 8 — What happens if the heuristic is perfect?

**Model answer:**

- If `h(n)` equals the exact remaining cost, A* becomes maximally informed.
- It expands almost only the nodes that matter for the optimal path.
- That is the ideal case for search efficiency.

### Question 9 — Why isn't greedy best-first search equivalent to A*?

**Model answer:**

- Greedy best-first search uses only the heuristic, effectively `f(n) = h(n)`.
- It ignores the real cost already paid.
- A* balances both past cost and estimated future cost, which is why it can stay optimal under the right conditions.

### Question 10 — In one sentence, compare the three algorithms

**Model answer:**

- Dijkstra trusts accumulated cost, A* trusts accumulated cost plus a safe estimate, and Bellman-Ford keeps correcting estimates until no edge can improve them.

### Interview-answer checklist

Your interview answers are strong if they mention:

- A* loses optimality when the heuristic overestimates
- consistency is about smooth heuristic behavior across edges
- Dijkstra fails because greedy finalization assumes no later cheaper path appears
- Bellman-Ford succeeds because repeated relaxation can repair earlier estimates
- `V - 1` rounds correspond to the maximum number of edges in a simple shortest path
- one extra round is used to expose a negative cycle

- [ ] Done

---

## Section C — Mini Scenarios

### Scenario 1 — Negative edge trap

Graph:

```text
A -> B = 5
A -> C = 2
C -> B = -10
```

**Questions:**

1. Why is this graph a trap for Dijkstra?
2. What result should Bellman-Ford eventually find for `dist(B)`?
3. What interview lesson is hidden here?

### Scenario 1 Answer

- It is a trap because a later negative edge can improve the cost to `B` after an earlier estimate looked final.
- Bellman-Ford should find `dist(B) = -8` through `A -> C -> B`.
- The lesson is that greedy finalization only works when future edges cannot make old answers cheaper.

- [ ] Done

---

### Scenario 2 — Bad heuristic trap

Suppose the real shortest path from `S` to `G` has total cost `10`, but your heuristic assigns a large overestimate on that optimal branch and a tiny estimate on a longer branch.

**Questions:**

1. What wrong behavior can A* show here?
2. Which formal property was broken?
3. What is the safest interview answer for how to fix it?

### Scenario 2 Answer

- A* can prefer the longer branch because the optimal branch looks falsely expensive under `g(n) + h(n)`.
- The heuristic is not admissible because it overestimates the true remaining cost.
- Fix the heuristic so it never overestimates, and ideally make it consistent as well.

### Scenario checklist

Your scenario answers are strong if they identify:

- the exact hidden assumption that breaks in the scenario
- the corrected distance or branch preference, not just a vague description
- whether the problem is caused by edge weights, heuristic quality, or cycle structure
- the safest formal fix: admissible heuristic, consistent heuristic, or Bellman-Ford-style repeated relaxation

- [ ] Done

---

## Section D — Robotics Translation

### Why this still matters in Nav2 interviews

Even though Nav2 does not typically use Bellman-Ford for robot path planning, these questions matter because they test whether you understand search assumptions:

- **NavFn / A***: why a heuristic helps and when it is still safe
- **Grid search vs kinematic search**: why `A*` on `(x, y)` is different from `Hybrid-A*` on `(x, y, θ)`
- **Planner debugging**: whether the failure is coming from the cost function, the state space, or the robot model

If you can explain Bellman-Ford, you usually give better answers about why a planner trusts or distrusts partial solutions.

### Interview bridge answer

If someone asks, "Why should a robotics engineer care about Bellman-Ford if Nav2 mostly uses NavFn or Smac?" a strong answer is:

> Because Bellman-Ford exposes the hidden assumption behind greedy shortest-path methods: whether a distance estimate can safely be finalized early. That same reasoning helps when explaining Dijkstra, A*, and why kinematic planners must reason more carefully about feasibility.

### Robotics-translation checklist

Your robotics translation is strong if it connects Bellman-Ford back to:

- why NavFn or A* can trust some partial answers
- why planner assumptions matter as much as formulas
- why search state, heuristic choice, and feasibility constraints should be discussed separately

- [ ] Done

---

## Practical Takeaway

Memorize this progression:

1. **Dijkstra**: exact, greedy, non-negative edges only
2. **A***: Dijkstra plus heuristic guidance
3. **Bellman-Ford**: repeated correction instead of greedy finalization

And memorize this failure map:

1. **Bad heuristic** -> A* may lose optimality
2. **Inconsistent heuristic** -> A* may reopen nodes and get messy
3. **Negative edge** -> Dijkstra's greedy assumption breaks
4. **Negative cycle** -> no finite shortest path exists

That is enough to answer most interview follow-ups cleanly.