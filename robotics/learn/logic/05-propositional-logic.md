# 05 — Propositional Logic
### Truth tables, connectives, and argument validity

**Covers:** Chapter 6 — Propositional Logic (§6.1–6.6)
**Prerequisite:** Lessons 01–04
**Unlocks:** Lesson 06 (Natural Deduction)

---

## Why Should I Care?

Propositional logic is the foundation of Boolean logic — the same `AND`, `OR`, `NOT`, `IF-THEN` you use in code every day. But in code, you apply it intuitively. This lesson makes it rigorous: you'll learn to prove whether complex conditional chains are valid, build truth tables to verify edge cases, and recognize the named argument forms that appear everywhere in debugging.

---

# PART 1 — LOGICAL CONNECTIVES

## 1.1 Propositions and Variables

A **simple proposition** is one that contains no logical connectives. We represent them with lowercase letters:

- p = "The battery is charged"
- q = "The robot is operational"
- r = "The firmware is updated"

A **compound proposition** is built from simple propositions using connectives.

## 1.2 The Five Connectives

| Connective | Symbol | Name | English | Code equivalent |
|---|---|---|---|---|
| Negation | ~p (or ¬p) | Tilde | "not p" | `!p` |
| Conjunction | p · q (or p ∧ q) | Dot | "p and q" | `p && q` |
| Disjunction | p ∨ q | Wedge | "p or q" | `p \|\| q` |
| Conditional | p ⊃ q (or p → q) | Horseshoe | "if p then q" | `!p \|\| q` |
| Biconditional | p ≡ q (or p ↔ q) | Triple bar | "p if and only if q" | `p == q` |

## 1.3 Truth Tables for Each Connective

**Negation (~):**

| p | ~p |
|---|---|
| T | F |
| F | T |

**Conjunction (·):** True only when BOTH are true.

| p | q | p · q |
|---|---|---|
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | F |

**Disjunction (∨):** False only when BOTH are false. (Inclusive or)

| p | q | p ∨ q |
|---|---|---|
| T | T | T |
| T | F | T |
| F | T | T |
| F | F | F |

**Conditional (⊃):** False ONLY when the antecedent is true and the consequent is false.

| p | q | p ⊃ q |
|---|---|---|
| T | T | T |
| T | F | **F** |
| F | T | T |
| F | F | T |

**This is the most counterintuitive one.** "If p then q" is considered true whenever p is false, regardless of q. Why? Because the conditional only makes a promise about what happens when p is true. If p is false, the promise isn't broken.

**Engineering analogy:** "If `battery < 10%`, then `robot docks`." When battery is 50%, the rule isn't violated whether the robot docks or not — the condition doesn't apply.

**Biconditional (≡):** True when both have the SAME truth value.

| p | q | p ≡ q |
|---|---|---|
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | T |

## 1.4 Translating English to Symbols

| English | Symbolic |
|---|---|
| "The robot stops and the alarm sounds" | p · q |
| "Either the hardware failed or the software crashed" | p ∨ q |
| "If the sensor is stale, then the estimator diverges" | p ⊃ q |
| "The robot moves if and only if both motors are running" | p ≡ (q · r) |
| "It's not the case that the firmware is current" | ~p |
| "Unless the battery is charged, the robot won't start" | ~p ⊃ ~q (or equivalently: q ⊃ p) |

**Tricky words:**
- "unless" = "if not" → ~p ⊃ q
- "only if" → p ⊃ q ("p only if q" means "if p then q")
- "if" → q ⊃ p ("p if q" means "if q then p")
- "sufficient condition" → p ⊃ q (p is sufficient for q)
- "necessary condition" → q ⊃ p (q is necessary for p, equivalently p ⊃ q)

---

# PART 2 — TRUTH TABLES FOR ARGUMENTS

## 2.1 Testing Validity

To test an argument's validity with a truth table:
1. List all propositional variables.
2. Create all possible truth-value combinations (2^n rows for n variables).
3. Compute the truth value of each premise and the conclusion.
4. Check: is there ANY row where all premises are true but the conclusion is false?
   - If NO such row → **Valid**
   - If YES → **Invalid** (that row is the counterexample)

**Example: Modus Ponens**

```
P1: p ⊃ q    (If the sensor fails, the robot stops)
P2: p          (The sensor failed)
C:  q          (Therefore, the robot stops)
```

| p | q | p ⊃ q (P1) | p (P2) | q (C) |
|---|---|---|---|---|
| T | T | T | T | **T** ✓ |
| T | F | F | T | — |
| F | T | T | F | — |
| F | F | T | F | — |

Row 1 is the only row where both premises are true, and the conclusion is also true. **Valid.**

**Example: Affirming the Consequent (INVALID)**

```
P1: p ⊃ q    (If the sensor fails, the robot stops)
P2: q          (The robot stopped)
C:  p          (Therefore, the sensor failed)
```

| p | q | p ⊃ q (P1) | q (P2) | p (C) |
|---|---|---|---|---|
| T | T | T | T | T |
| T | F | F | F | — |
| **F** | **T** | **T** | **T** | **F** ✗ |
| F | F | T | F | — |

Row 3: Both premises true, conclusion false. **Invalid.** The robot could have stopped for another reason.

## 2.2 Tautologies, Contradictions, Contingencies

- **Tautology:** True in every row. Example: p ∨ ~p (law of excluded middle).
- **Contradiction:** False in every row. Example: p · ~p.
- **Contingency:** True in some rows, false in others. Most propositions are contingencies.

Two statements are **logically equivalent** if they have the same truth value in every row:
- p ⊃ q ≡ ~p ∨ q (material conditional = negated antecedent or consequent)
- ~(p · q) ≡ ~p ∨ ~q (De Morgan's Law)
- ~(p ∨ q) ≡ ~p · ~q (De Morgan's Law)

---

# PART 3 — NAMED ARGUMENT FORMS

## 3.1 Valid Forms (Know These Cold)

| Name | Form | Pattern |
|---|---|---|
| **Modus Ponens** (MP) | p ⊃ q, p ∴ q | "If P then Q. P. So Q." |
| **Modus Tollens** (MT) | p ⊃ q, ~q ∴ ~p | "If P then Q. Not Q. So not P." |
| **Hypothetical Syllogism** (HS) | p ⊃ q, q ⊃ r ∴ p ⊃ r | "If P then Q. If Q then R. So if P then R." |
| **Disjunctive Syllogism** (DS) | p ∨ q, ~p ∴ q | "P or Q. Not P. So Q." |
| **Constructive Dilemma** (CD) | (p ⊃ q) · (r ⊃ s), p ∨ r ∴ q ∨ s | |
| **Absorption** (Abs) | p ⊃ q ∴ p ⊃ (p · q) | |
| **Simplification** (Simp) | p · q ∴ p | "P and Q. So P." |
| **Conjunction** (Conj) | p, q ∴ p · q | "P. Q. So P and Q." |
| **Addition** (Add) | p ∴ p ∨ q | "P. So P or Q." |

## 3.2 Invalid Forms (Common Fallacies)

| Name | Form | Why It's Wrong |
|---|---|---|
| **Affirming the Consequent** | p ⊃ q, q ∴ p | Q could be true for other reasons |
| **Denying the Antecedent** | p ⊃ q, ~p ∴ ~q | Q could be true even without P |

**Debugging trap — Affirming the Consequent:**
> "If the motor driver is fried (p), the motor won't spin (q). The motor won't spin (q). Therefore the motor driver is fried (p)."

Wrong! The motor might not spin because: loose connector, dead battery, firmware crash, broken gearbox, emergency stop pressed...

**Debugging trap — Denying the Antecedent:**
> "If the firmware is v1.18 (p), the robot has the SPI bug (q). The firmware is not v1.18 (~p). Therefore the robot doesn't have the SPI bug (~q)."

Wrong! The SPI bug might exist in other firmware versions too.

## 3.3 Modus Tollens: The Debugger's Best Friend

Modus Tollens is the logical foundation of **elimination-based debugging:**

> "If hypothesis H were true, we would observe symptom S. We do NOT observe symptom S. Therefore, hypothesis H is false."

This is how you rule things out:
- "If the SPI bus were locked, we'd see frozen register values. Register values are updating. → SPI bus is not locked."
- "If the battery were dead, the system wouldn't boot. The system boots. → Battery is not dead."
- "If memory were corrupted, the checksum would fail. Checksum passes. → Memory is not corrupted."

Each application of Modus Tollens eliminates one hypothesis. Systematic debugging is essentially repeated Modus Tollens until you're left with the one hypothesis you can't eliminate.

---

# PART 4 — LOGICAL EQUIVALENCES

## 4.1 Important Equivalences

These are like algebraic identities — you can substitute freely:

| Name | Equivalence |
|---|---|
| De Morgan's | ~(p · q) ≡ ~p ∨ ~q |
| De Morgan's | ~(p ∨ q) ≡ ~p · ~q |
| Double Negation | ~~p ≡ p |
| Contraposition | (p ⊃ q) ≡ (~q ⊃ ~p) |
| Material Conditional | (p ⊃ q) ≡ (~p ∨ q) |
| Material Equivalence | (p ≡ q) ≡ (p ⊃ q) · (q ⊃ p) |
| Distribution | p · (q ∨ r) ≡ (p · q) ∨ (p · r) |
| Distribution | p ∨ (q · r) ≡ (p ∨ q) · (p ∨ r) |
| Commutation | p · q ≡ q · p |
| Association | (p · q) · r ≡ p · (q · r) |

**De Morgan's in code:** `!(a && b)` is the same as `!a || !b`. You use this constantly when refactoring conditionals.

**Contraposition in debugging:** "If the firmware update causes the crash, then reverting the firmware stops the crash" is equivalent to "If reverting the firmware does NOT stop the crash, then the firmware update did NOT cause the crash."

---

## Key Takeaways

1. **Five connectives** (~, ·, ∨, ⊃, ≡) cover all compound propositions.
2. **The conditional (⊃) is only false when T → F.** This is counterintuitive but essential.
3. **Truth tables mechanically prove validity.** No row can have all-true premises and a false conclusion.
4. **Modus Ponens and Modus Tollens are the workhorses of reasoning.** MP moves forward; MT eliminates.
5. **Affirming the consequent is the most common debugging fallacy.** "Symptom matches hypothesis ≠ hypothesis is proven."
6. **De Morgan's and Contraposition** are the most useful equivalences for everyday reasoning and code.
