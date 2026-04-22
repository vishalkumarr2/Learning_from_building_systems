# Exercise 5 — Propositional Logic

**Covers:** Lesson 05
**Difficulty:** Intermediate

---

## Problem 1: Symbolize

Let: p = "The battery is charged," q = "The motors are enabled," r = "The robot is operational," s = "The firmware is current."

Translate each statement into symbolic form.

**A.** "The robot is operational only if the battery is charged and the motors are enabled."

**B.** "If the firmware is not current, then the robot is not operational."

**C.** "Either the battery is charged or the motors are not enabled."

**D.** "The robot is operational if and only if the battery is charged, the motors are enabled, and the firmware is current."

**E.** "Unless the firmware is current, the motors are not enabled."

**F.** "It's not the case that both the battery is uncharged and the robot is operational."

---

## Problem 2: Truth Tables

Construct truth tables and determine: tautology, contradiction, or contingency?

**A.** p ∨ ~p

**B.** (p ⊃ q) · (p · ~q)

**C.** (p ⊃ q) ∨ (q ⊃ p)

**D.** p ⊃ (q ⊃ p)

---

## Problem 3: Argument Validity (Truth Table Method)

Test each argument for validity using truth tables.

**A.** Modus Tollens:
```
p ⊃ q
~q
∴ ~p
```

**B.** Hypothetical Syllogism:
```
p ⊃ q
q ⊃ r
∴ p ⊃ r
```

**C.** A suspicious argument:
```
p ⊃ q
r ⊃ q
∴ p ⊃ r
```

**D.** Debugging scenario:
```
(s · b) ⊃ r        (If firmware is current and battery is charged, robot is operational)
~r                   (Robot is not operational)
s                    (Firmware is current)
∴ ~b                (Therefore, battery is not charged)
```

---

## Problem 4: Logical Equivalence

Use truth tables to verify whether each pair is logically equivalent.

**A.** p ⊃ q  and  ~q ⊃ ~p  (Contraposition)

**B.** ~(p · q)  and  ~p ∨ ~q  (De Morgan's)

**C.** p ⊃ q  and  ~p ⊃ ~q

---

## Problem 5: Identify the Form

Name the valid argument form (or identify the fallacy) used in each.

**A.** "If the SPI clock is unstable, the DMA transfers corrupt. The SPI clock is unstable. Therefore, DMA transfers corrupt."

**B.** "If the motor overheats, it shuts down. The motor didn't shut down. Therefore, it didn't overheat."

**C.** "If the firmware is updated, the bug is fixed. The bug is fixed. Therefore, the firmware was updated."

**D.** "Either the problem is hardware or software. It's not hardware. Therefore, it's software."

**E.** "If the battery dies, the robot stops. If the robot stops, the order is delayed. Therefore, if the battery dies, the order is delayed."

**F.** "If the firmware is v1.18, the robot has the SPI bug. The firmware is not v1.18. Therefore, the robot doesn't have the SPI bug."

---

## Problem 6: Complex Argument

Symbolize and test for validity:

"If the sensorbar cable is loose, then either the readings are intermittent or the sensorbar is offline. The readings are not intermittent and the sensorbar is not offline. Therefore, the sensorbar cable is not loose."

Let: p = cable is loose, q = readings are intermittent, r = sensorbar is offline.

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** r ⊃ (p · q)

**B.** ~s ⊃ ~r

**C.** p ∨ ~q

**D.** r ≡ (p · q · s)

**E.** ~s ⊃ ~q (or equivalently: q ⊃ s)

**F.** ~(~p · r)  or equivalently: p ∨ ~r

**Problem 2:**

**A.** p ∨ ~p — **Tautology** (always true).
| p | ~p | p ∨ ~p |
|---|---|---|
| T | F | T |
| F | T | T |

**B.** (p ⊃ q) · (p · ~q) — **Contradiction** (always false).
This says "if p then q" AND "p and not q" — these can never both be true.

**C.** (p ⊃ q) ∨ (q ⊃ p) — **Tautology**.
| p | q | p⊃q | q⊃p | (p⊃q)∨(q⊃p) |
|---|---|---|---|---|
| T | T | T | T | T |
| T | F | F | T | T |
| F | T | T | F | T |
| F | F | T | T | T |

**D.** p ⊃ (q ⊃ p) — **Tautology**.
| p | q | q⊃p | p⊃(q⊃p) |
|---|---|---|---|
| T | T | T | T |
| T | F | T | T |
| F | T | F | T |
| F | F | T | T |

**Problem 3:**

**A.** Modus Tollens — **Valid**. No row has both premises T and conclusion F.

| p | q | p⊃q | ~q | ~p |
|---|---|---|---|---|
| T | T | T | F | — |
| T | F | F | — | — |
| F | T | T | F | — |
| **F** | **F** | **T** | **T** | **T** ✓ |

Only row 4 has both premises true, and conclusion (~p = T) is true. Valid.

**B.** Hypothetical Syllogism — **Valid**. Check all 8 rows (3 variables). In every row where both p⊃q and q⊃r are true, p⊃r is also true.

**C.** **Invalid**.

| p | q | r | p⊃q | r⊃q | p⊃r |
|---|---|---|---|---|---|
| T | T | F | T | T | **F** ✗ |

Row where p=T, q=T, r=F: Both premises true (T⊃T=T, F⊃T=T), but conclusion T⊃F=F. **Invalid.**

**D.** **Valid**.
(s · b) ⊃ r, ~r, s ∴ ~b

From ~r and (s · b) ⊃ r: by Modus Tollens, ~(s · b).
By De Morgan's: ~s ∨ ~b.
We have s. By Disjunctive Syllogism: ~b. ✓

Truth table confirms: in every row where all three premises are true, ~b is also true.

**Problem 4:**

**A.** p ⊃ q and ~q ⊃ ~p: **Equivalent** (same truth values in all rows — this is contraposition).

**B.** ~(p · q) and ~p ∨ ~q: **Equivalent** (De Morgan's Law).

**C.** p ⊃ q and ~p ⊃ ~q: **NOT equivalent**.
| p | q | p⊃q | ~p⊃~q |
|---|---|---|---|
| T | T | T | T |
| T | F | F | T |
| F | T | T | **F** |
| F | F | T | T |

Row 3 differs. These are NOT equivalent. (This is the inverse, not the contrapositive.)

**Problem 5:**

**A.** Modus Ponens (valid).
**B.** Modus Tollens (valid).
**C.** Affirming the Consequent (INVALID — the bug could have been fixed another way).
**D.** Disjunctive Syllogism (valid).
**E.** Hypothetical Syllogism (valid).
**F.** Denying the Antecedent (INVALID — other firmware versions might also have the SPI bug).

**Problem 6:**

Symbolized:
```
P1: p ⊃ (q ∨ r)
P2: ~q · ~r
C:  ~p
```

P2 gives us ~q · ~r. By De Morgan's: ~(q ∨ r).
From P1 and ~(q ∨ r): by Modus Tollens, ~p. ✓

**Valid.** This is Modus Tollens applied to a compound consequent.

</details>
