# Exercise 7 — Predicate Logic

**Covers:** Lesson 07
**Difficulty:** Intermediate to Advanced

---

## Problem 1: Translation (English to Symbolic)

Domain: robots, sensors, firmware versions, tests.
Predicates: Rx = x is a robot, Sx = x is a sensor, Fx = x is a firmware version, Ox = x is operational, Dx = x is defective, Ux = x needs updating, Pxy = x passed test y, Hxy = x has component y.

**A.** "All robots are operational."

**B.** "Some sensors are defective."

**C.** "No firmware version needs updating."

**D.** "Every robot has at least one sensor."

**E.** "Some robots passed every test."

**F.** "No robot passed all tests."

**G.** "If any sensor is defective, then some robot is not operational."

**H.** "Only operational robots passed the safety test." (Let s = the safety test)

---

## Problem 2: Translation (Symbolic to English)

Using the same predicates as Problem 1:

**A.** ∃x(Rx · Dx)

**B.** ∀x(Sx ⊃ ~Dx)

**C.** ∀x∀y((Rx · Sy · Hxy) ⊃ ~Dy)

**D.** ~∃x(Rx · ∀y(Sy ⊃ Hxy))

---

## Problem 3: Quantifier Negation

Rewrite each statement using the equivalent quantifier.

**A.** ~∀x(Rx ⊃ Ox)  → 

**B.** ~∃x(Sx · Dx)  → 

**C.** ∀x~(Rx · Dx)  → 

**D.** "It's not the case that every test was passed." → 

---

## Problem 4: Proofs with Quantifiers

Derive the conclusion from the premises.

**A.**
```
1. ∀x(Rx ⊃ Sx)          (All robots have sensors)
2. ∀x(Sx ⊃ Cx)          (All things with sensors need calibration)
∴ ∀x(Rx ⊃ Cx)          (All robots need calibration)
```

**B.**
```
1. ∀x(Fx ⊃ Bx)          (All devices with old firmware have the bug)
2. ∃x(Rx · Fx)           (Some robot has old firmware)
∴ ∃x(Rx · Bx)           (Some robot has the bug)
```

**C.**
```
1. ∀x(Rx ⊃ Ox)          (All robots are operational)
2. ∃x(Rx · ~Tx)          (Some robot didn't pass the test)
∴ ∃x(Ox · ~Tx)          (Something operational didn't pass the test)
```

---

## Problem 5: Invalid Quantifier Arguments

Show that each argument is invalid by providing a counterexample (a domain where premises are true but conclusion is false).

**A.**
```
∀x(Px ⊃ Qx)            (All P are Q)
∀x(Rx ⊃ Qx)            (All R are Q)
∴ ∀x(Px ⊃ Rx)          (All P are R)
```

**B.**
```
∃x(Rx · Dx)             (Some robot is defective)
∃x(Rx · Ox)             (Some robot is operational)
∴ ∃x(Rx · Dx · Ox)     (Some robot is defective and operational)
```

---

## Problem 6: Identity and Counting

Symbolize each statement.

**A.** "Exactly one robot failed the test." (Use Fx = x is a robot, Tx = x failed the test)

**B.** "At least two sensors are defective." (Use Sx = x is a sensor, Dx = x is defective)

**C.** "Only Robot-42 has the SPI bug." (Use Bx = x has the SPI bug, r = Robot-42)

---

## Problem 7: Engineering Specification

Translate this system requirement into predicate logic:

"For every robot in the fleet, if its battery level drops below 15%, then either it must return to dock within 5 minutes or an operator must be notified, unless the robot is currently carrying a critical payload."

Let: Rx = x is a robot, Bx = x has battery below 15%, Dx = x returns to dock within 5 min, Nx = an operator is notified about x, Px = x is carrying a critical payload.

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** ∀x(Rx ⊃ Ox)

**B.** ∃x(Sx · Dx)

**C.** ~∃x(Fx · Ux) or equivalently ∀x(Fx ⊃ ~Ux)

**D.** ∀x(Rx ⊃ ∃y(Sy · Hxy))

**E.** ∃x(Rx · ∀y(Ty ⊃ Pxy)) where Ty = y is a test

**F.** ~∃x(Rx · ∀y(Ty ⊃ Pxy)) or equivalently ∀x(Rx ⊃ ∃y(Ty · ~Pxy))

**G.** ∃x(Sx · Dx) ⊃ ∃x(Rx · ~Ox)

Note: The quantifier scopes are independent — the "some robot" that's not operational need not be the same as anything related to the defective sensor.

**H.** ∀x(Pxs ⊃ (Rx · Ox))

"Only operational robots passed" = "anything that passed is an operational robot."

**Problem 2:**

**A.** "Some robot is defective."

**B.** "No sensor is defective" or "All sensors are non-defective."

**C.** "For every robot and every sensor, if the robot has that sensor, then the sensor is not defective." (i.e., "No robot has a defective sensor.")

**D.** "There is no robot that has every sensor." or "No single robot has all the sensors."

**Problem 3:**

**A.** ~∀x(Rx ⊃ Ox) → ∃x(Rx · ~Ox)
"Not all robots are operational" → "Some robot is not operational."

**B.** ~∃x(Sx · Dx) → ∀x(Sx ⊃ ~Dx)
"No sensor is defective" → "All sensors are non-defective."

**C.** ∀x~(Rx · Dx) → ~∃x(Rx · Dx)  (by quantifier exchange)
Also equivalent to: ∀x(Rx ⊃ ~Dx) — "No robot is defective."

**D.** ~∀x(Tx ⊃ Px) → ∃x(Tx · ~Px)
"Some test was not passed."

**Problem 4A:**
```
1. ∀x(Rx ⊃ Sx)          Premise
2. ∀x(Sx ⊃ Cx)          Premise
   [For arbitrary a:]
3. Ra ⊃ Sa              UI 1
4. Sa ⊃ Ca              UI 2
5. Ra ⊃ Ca              HS 3, 4
6. ∀x(Rx ⊃ Cx)          UG 5
```

**Problem 4B:**
```
1. ∀x(Fx ⊃ Bx)          Premise
2. ∃x(Rx · Fx)           Premise
3. Ra · Fa               EI 2        (EI first — fresh name 'a')
4. Fa ⊃ Ba              UI 1        (instantiate for 'a')
5. Fa                    Simp 3 (Com then Simp)
6. Ba                    MP 4, 5
7. Ra                    Simp 3
8. Ra · Ba               Conj 7, 6
9. ∃x(Rx · Bx)          EG 8
```

**Problem 4C:**
```
1. ∀x(Rx ⊃ Ox)          Premise
2. ∃x(Rx · ~Tx)          Premise
3. Ra · ~Ta              EI 2
4. Ra ⊃ Oa              UI 1
5. Ra                    Simp 3
6. Oa                    MP 4, 5
7. ~Ta                   Simp 3 (Com, Simp)
8. Oa · ~Ta              Conj 6, 7
9. ∃x(Ox · ~Tx)          EG 8
```

**Problem 5A:**

Counterexample: Let the domain be {1, 2, 3}.
- P = {1, 2}, Q = {1, 2, 3}, R = {3}
- All P are Q ✓ (1∈Q, 2∈Q)
- All R are Q ✓ (3∈Q)
- All P are R? NO — 1∈P but 1∉R. **Invalid.**

(Robots and motors can both need maintenance without all robots being motors.)

**Problem 5B:**

Counterexample: Domain = {robot-1, robot-2}.
- robot-1 is a robot, defective, NOT operational.
- robot-2 is a robot, NOT defective, operational.
- ∃x(Rx · Dx) ✓ (robot-1)
- ∃x(Rx · Ox) ✓ (robot-2)
- ∃x(Rx · Dx · Ox)? NO — robot-1 is defective but not operational; robot-2 is operational but not defective. **Invalid.**

**Problem 6:**

**A.** ∃x(Fx · Tx · ∀y((Fy · Ty) ⊃ y = x))
"There exists a robot that failed, and any robot that failed is that same one."

**B.** ∃x∃y(Sx · Sy · Dx · Dy · x ≠ y)
"There exist two different sensors that are both defective."

**C.** ∀x(Bx ⊃ x = r) or equivalently Br · ∀x(Bx ⊃ x = r)
"Anything with the SPI bug is Robot-42" (and Robot-42 has it).

**Problem 7:**

∀x(Rx · Bx · ~Px) ⊃ (Dx ∨ Nx))

"For every x, if x is a robot and x has battery below 15% and x is NOT carrying a critical payload, then either x returns to dock within 5 minutes or an operator is notified about x."

</details>
