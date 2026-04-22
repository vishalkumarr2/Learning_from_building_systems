# 06 — Natural Deduction in Propositional Logic
### Building proofs step by step

**Covers:** Chapter 7 — Natural Deduction in Propositional Logic (§7.1–7.7)
**Prerequisite:** Lesson 05 (Propositional Logic)
**Unlocks:** Lesson 07 (Predicate Logic)

---

## Why Should I Care?

Truth tables work but get unwieldy fast — n variables means 2^n rows. With 5 variables, that's 32 rows. With 10, it's 1024. Natural deduction lets you prove validity by deriving the conclusion from the premises step by step, using rules of inference — like building a proof in mathematics or constructing a chain of evidence in an investigation.

---

# PART 1 — RULES OF INFERENCE

## 1.1 The Eight Basic Rules

These rules let you derive new lines from existing lines in a proof. Each rule has a specific pattern — you can only apply it when the pattern matches exactly.

### 1. Modus Ponens (MP)
```
n.  p ⊃ q
m.  p
∴   q                    MP n, m
```

### 2. Modus Tollens (MT)
```
n.  p ⊃ q
m.  ~q
∴   ~p                   MT n, m
```

### 3. Hypothetical Syllogism (HS)
```
n.  p ⊃ q
m.  q ⊃ r
∴   p ⊃ r                HS n, m
```

### 4. Disjunctive Syllogism (DS)
```
n.  p ∨ q
m.  ~p
∴   q                    DS n, m
```

### 5. Constructive Dilemma (CD)
```
n.  (p ⊃ q) · (r ⊃ s)
m.  p ∨ r
∴   q ∨ s                CD n, m
```

### 6. Simplification (Simp)
```
n.  p · q
∴   p                    Simp n
```
(You can also get q from p · q — just use commutativity first.)

### 7. Conjunction (Conj)
```
n.  p
m.  q
∴   p · q                Conj n, m
```

### 8. Addition (Add)
```
n.  p
∴   p ∨ q                Add n
```
(You can add *anything* as a disjunct. This seems odd but is logically sound.)

## 1.2 Example Proof

**Prove:** From premises (1) p ⊃ q, (2) q ⊃ r, (3) p — derive r.

```
1.  p ⊃ q          Premise
2.  q ⊃ r          Premise
3.  p               Premise
4.  q               MP 1, 3
5.  r               MP 2, 4
```

**Engineering reading:** "If the sensor fails (p), the data goes stale (q). If data goes stale (q), the estimator diverges (r). The sensor failed (p). Therefore: the data goes stale (step 4), and the estimator diverges (step 5)."

## 1.3 Strategy Tips

1. **Work backward** from the conclusion. What do you need to derive it? What rule would produce it?
2. **Look for Modus Ponens opportunities.** If you have `p ⊃ q` and can get `p`, you get `q`.
3. **Look for Modus Tollens opportunities.** If you have `p ⊃ q` and `~q`, you get `~p`.
4. **Break conjunctions apart** with Simplification to get individual components.
5. **Build conjunctions** when the conclusion needs them.

---

# PART 2 — RULES OF REPLACEMENT

## 2.1 The Ten Replacement Rules

Unlike inference rules (which go one direction), replacement rules are **equivalences** — you can substitute in either direction, and you can apply them to *part* of a line.

| Name | Equivalence |
|---|---|
| **De Morgan's (DM)** | ~(p · q) ≡ ~p ∨ ~q, ~(p ∨ q) ≡ ~p · ~q |
| **Double Negation (DN)** | ~~p ≡ p |
| **Commutation (Com)** | p ∨ q ≡ q ∨ p, p · q ≡ q · p |
| **Association (Assoc)** | (p ∨ q) ∨ r ≡ p ∨ (q ∨ r) |
| **Distribution (Dist)** | p · (q ∨ r) ≡ (p · q) ∨ (p · r) |
| **Transposition (Trans)** | (p ⊃ q) ≡ (~q ⊃ ~p) |
| **Material Implication (Impl)** | (p ⊃ q) ≡ (~p ∨ q) |
| **Material Equivalence (Equiv)** | (p ≡ q) ≡ (p ⊃ q) · (q ⊃ p) |
| **Exportation (Exp)** | (p · q) ⊃ r ≡ p ⊃ (q ⊃ r) |
| **Tautology (Taut)** | p ≡ p ∨ p, p ≡ p · p |

## 2.2 Example Using Replacement Rules

**Prove:** From (1) ~p ∨ q and (2) ~q — derive ~p.

```
1.  ~p ∨ q          Premise
2.  ~q               Premise
3.  p ⊃ q           Impl 1     (replaced ~p ∨ q with p ⊃ q)
4.  ~p               MT 3, 2
```

Without replacement rules, this would be much harder. Material Implication let us convert the disjunction into a conditional, then apply Modus Tollens.

## 2.3 Longer Proof

**Prove:** From (1) p ⊃ (q · r), (2) ~q — derive ~p.

```
1.  p ⊃ (q · r)     Premise
2.  ~q               Premise
3.  ~q ∨ ~r          Add 2         (adding ~r as disjunct)
4.  ~(q · r)         DM 3          (De Morgan's: ~q ∨ ~r ≡ ~(q · r))
5.  ~p               MT 1, 4
```

**Engineering reading:** "If the robot is in autonomous mode (p), then both the planner is running (q) and the motors are enabled (r). The planner is not running (~q). Therefore the robot is not in autonomous mode (~p)."

---

# PART 3 — CONDITIONAL AND INDIRECT PROOF

## 3.1 Conditional Proof (CP)

To prove a conditional conclusion (p ⊃ q):
1. **Assume** the antecedent (p) as a temporary hypothesis.
2. From that assumption plus the premises, **derive** the consequent (q).
3. **Discharge** the assumption: conclude p ⊃ q.

**Example:** From (1) p ⊃ q, (2) q ⊃ r — prove p ⊃ r.

```
1.  p ⊃ q            Premise
2.  q ⊃ r            Premise
    ┌──────────────────
3.  │ p               Assume (CP)
4.  │ q               MP 1, 3
5.  │ r               MP 2, 4
    └──────────────────
6.  p ⊃ r            CP 3–5
```

Lines 3–5 are inside the conditional proof "box." They depend on the assumption. Line 6 discharges the assumption.

## 3.2 Indirect Proof (IP) — Proof by Contradiction

To prove a conclusion q:
1. **Assume** the negation (~q).
2. From that assumption plus the premises, **derive a contradiction** (any r · ~r).
3. **Discharge** the assumption: conclude q (since ~q leads to contradiction).

**Example:** From (1) p ∨ q, (2) ~p — prove q.

```
1.  p ∨ q             Premise
2.  ~p                Premise
    ┌──────────────────
3.  │ ~q              Assume (IP)
4.  │ p               DS 1, 3      (from p ∨ q and ~q, get p)
5.  │ p · ~p          Conj 4, 2    (contradiction!)
    └──────────────────
6.  q                 IP 3–5
```

**When to use IP:** When the conclusion doesn't obviously follow from the premises, try assuming its negation and see if you reach a contradiction.

## 3.3 Proof Strategy Flowchart

```
What form is the conclusion?

p ⊃ q  → Try Conditional Proof (assume p, derive q)
p · q  → Derive p and q separately, then Conj
p ∨ q  → Try to derive p or q, then Add
~p     → Try MT or IP
p      → Try MP, DS, or Simp from premises
```

**General strategy:**
1. Look at the conclusion's main connective.
2. Choose the strategy that matches.
3. If stuck, try Indirect Proof — assume the opposite of what you want and derive a contradiction.

---

# PART 4 — PROOF PRACTICE PATTERNS

## 4.1 Chain Reasoning (HS + MP)

Very common in debugging — tracing a causal chain:
```
1.  a ⊃ b            "SPI jitter → DMA corruption"
2.  b ⊃ c            "DMA corruption → stale sensorbar data"
3.  c ⊃ d            "stale data → estimator divergence"
4.  a                 "SPI jitter occurred"
5.  a ⊃ c            HS 1, 2
6.  a ⊃ d            HS 5, 3
7.  d                 MP 6, 4
```

## 4.2 Elimination Reasoning (DS chain)

```
1.  a ∨ b ∨ c        "It's hardware, firmware, or software"
2.  ~a                "Ruled out hardware"
3.  ~b                "Ruled out firmware"
4.  b ∨ c            DS 1, 2     (with a eliminated)
5.  c                 DS 4, 3
```

## 4.3 Proof by Cases (CD)

When you have a disjunction and need to show both cases lead to the same (or compatible) conclusions:
```
1.  p ∨ q            "Either the motor overheated or the driver failed"
2.  p ⊃ r            "If motor overheated → replace motor"
3.  q ⊃ r            "If driver failed → also replace motor (it may be damaged)"
4.  (p ⊃ r) · (q ⊃ r)  Conj 2, 3
5.  r ∨ r            CD 4, 1
6.  r                 Taut 5
```

---

## Key Takeaways

1. **8 inference rules + 10 replacement rules** give you complete proof capability for propositional logic.
2. **Work backward** from the conclusion to figure out what you need.
3. **Conditional Proof** is the go-to method when the conclusion is a conditional (if-then).
4. **Indirect Proof** (contradiction) is the escape hatch when nothing else works.
5. **Chain reasoning** (HS + MP) models causal chain analysis in debugging.
6. **Elimination reasoning** (DS) models hypothesis elimination in RCA.
7. **Practice is essential.** These rules become second nature with repetition — like learning keyboard shortcuts.
