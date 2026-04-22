# 07 — Predicate Logic
### Quantifiers, variables, and reasoning about "all" and "some"

**Covers:** Chapter 8 — Predicate Logic (§8.1–8.7)
**Prerequisite:** Lessons 05–06 (Propositional Logic + Natural Deduction)
**Unlocks:** Lesson 08 (Inductive Reasoning)

---

## Why Should I Care?

Propositional logic treats entire statements as units. But many arguments depend on the *internal structure* of statements — specifically, on quantifiers like "all," "some," "every," and "no."

- "**Every** robot in the fleet needs the firmware update."
- "**Some** sensorbar readings are invalid."
- "**No** robot with firmware v1.24 exhibits the bug."

Predicate logic lets you formalize and reason about these precisely — and it's the foundation for database queries, type systems, formal specifications, and any time you write `for all x in collection` or `there exists x such that`.

---

# PART 1 — PREDICATES AND QUANTIFIERS

## 1.1 From Propositions to Predicates

In propositional logic: `p` = "Socrates is mortal" (an atomic, indivisible unit).

In predicate logic, we break this apart:
- **Subject:** Socrates (an individual)
- **Predicate:** "is mortal" (a property)

We write: **Mx** where M = "is mortal" and x = a variable that can be any individual.
- **Ms** = "Socrates is mortal" (s = Socrates)
- **Ma** = "Aristotle is mortal"

**Multi-place predicates:**
- **Lxy** = "x loves y" (a relation between two individuals)
- **Bxyz** = "x is between y and z"

## 1.2 The Universal Quantifier (∀)

**∀x** means "for all x" or "for every x."

| English | Symbolic |
|---|---|
| "All robots are autonomous" | ∀x(Rx ⊃ Ax) |
| "Every device needs updating" | ∀x(Dx ⊃ Ux) |
| "Each sensorbar has a code strip" | ∀x(Sx ⊃ Cx) |

**Important:** "All S are P" is translated as ∀x(Sx ⊃ Px) — with a **conditional**, not a conjunction. Why?

∀x(Rx · Ax) would mean "everything in the universe is both a robot and autonomous" — that includes chairs, trees, and numbers. We want: "for everything, IF it's a robot, THEN it's autonomous."

## 1.3 The Existential Quantifier (∃)

**∃x** means "there exists an x" or "for some x."

| English | Symbolic |
|---|---|
| "Some robots are malfunctioning" | ∃x(Rx · Mx) |
| "There is a firmware version that fixes the bug" | ∃x(Fx · Bx) |
| "At least one sensor is stale" | ∃x(Sx · Tx) |

**Important:** "Some S are P" is translated as ∃x(Sx · Px) — with a **conjunction**, not a conditional. Why?

∃x(Sx ⊃ Px) would be true if there exists anything that's NOT S (since a false antecedent makes the conditional true). That's almost always true and says almost nothing.

**Summary of the critical pattern:**
- Universal → use **conditional** (⊃): ∀x(Sx ⊃ Px)
- Existential → use **conjunction** (·): ∃x(Sx · Px)

## 1.4 Translating Complex Statements

| English | Symbolic |
|---|---|
| "No robots are perfect" | ~∃x(Rx · Px) or equivalently ∀x(Rx ⊃ ~Px) |
| "Not all tests passed" | ~∀x(Tx ⊃ Px) or equivalently ∃x(Tx · ~Px) |
| "Only robots have sensorbars" | ∀x(Sx ⊃ Rx) (having a sensorbar implies being a robot) |
| "All and only robots are autonomous" | ∀x(Rx ≡ Ax) |
| "Some but not all devices failed" | ∃x(Dx · Fx) · ∃x(Dx · ~Fx) |

## 1.5 Quantifier Negation (Crucial!)

These equivalences let you push negation through quantifiers:

| Original | Equivalent |
|---|---|
| ~∀xPx | ∃x~Px |
| ~∃xPx | ∀x~Px |
| ∀xPx | ~∃x~Px |
| ∃xPx | ~∀x~Px |

**In English:**
- "Not all robots passed" = "Some robot didn't pass"
- "No robot failed" = "All robots didn't fail" = "It's not the case that some robot failed"

---

# PART 2 — INFERENCE RULES FOR QUANTIFIERS

## 2.1 The Four Quantifier Rules

### Universal Instantiation (UI)
From "all x have property P," conclude that any *specific* individual has property P.
```
∀xPx
∴ Pa        (for any constant a)
```

**Example:** "All robots in fleet X need updating. Robot-42 is in fleet X. Therefore Robot-42 needs updating."

### Universal Generalization (UG)
From "an arbitrary individual has property P," conclude that all individuals have property P.
```
Pa          (where a is an arbitrary individual, not previously assumed to be special)
∴ ∀xPx
```

**Restrictions:** The individual `a` must be truly arbitrary — not introduced by existential instantiation, and not assumed to have any special properties.

### Existential Instantiation (EI)
From "some x has property P," introduce a *new* name for that individual.
```
∃xPx
∴ Pa        (where a is a NEW constant not yet used in the proof)
```

**Key restriction:** You must use a fresh name. You can't assume the "some x" is any previously mentioned individual.

### Existential Generalization (EG)
From "this specific individual has property P," conclude that some individual has property P.
```
Pa
∴ ∃xPx
```

**Example:** "Robot-42 has the SPI bug. Therefore, some robot has the SPI bug."

## 2.2 Order of Application

**Critical rule:** If you need to use both UI and EI in a proof, apply **EI first**. Why? EI requires a fresh name. If you apply UI first and introduce a name, that name is no longer fresh for EI.

## 2.3 Example Proof

**Prove:** From (1) ∀x(Rx ⊃ Sx) and (2) ∃x(Rx · Bx) — derive ∃x(Sx · Bx).

Translation: "All robots have sensors. Some robots have bugs. Therefore, some things with sensors have bugs."

```
1.  ∀x(Rx ⊃ Sx)         Premise  (all robots have sensors)
2.  ∃x(Rx · Bx)         Premise  (some robot has a bug)
3.  Ra · Ba              EI 2     (call it 'a' — EI first!)
4.  Ra ⊃ Sa             UI 1     (instantiate for 'a')
5.  Ra                   Simp 3
6.  Sa                   MP 4, 5
7.  Ba                   Simp 3   (right conjunct: Com then Simp)
8.  Sa · Ba              Conj 6, 7
9.  ∃x(Sx · Bx)         EG 8
```

---

# PART 3 — IDENTITY AND RELATIONS

## 3.1 Identity (=)

The identity predicate states that two terms refer to the same individual:
- a = b means "a is identical to b"
- a ≠ b means ~(a = b)

**Uses:**
- "**Exactly one** robot failed": ∃x(Fx · ∀y(Fy ⊃ y = x))
  - "There exists an x that failed, and for any y that failed, y is that same x."

- "**At least two** robots failed": ∃x∃y(Fx · Fy · x ≠ y)
  - "There exist x and y, both failed, and they're different."

## 3.2 Translating "Only," "Except," "At Most"

| English | Symbolic |
|---|---|
| "Only Alice passed" | ∀x(Px ⊃ x = a) |
| "Everyone except Bob failed" | ∀x(x ≠ b ⊃ Fx) · ~Fb |
| "At most one robot crashed" | ∀x∀y((Cx · Cy) ⊃ x = y) |

---

# PART 4 — PRACTICAL APPLICATIONS

## 4.1 Formalizing Fleet Policies

Policy: "Every robot with battery below 20% must return to dock, unless it's actively carrying a payload."

Let:
- Rx = x is a robot
- Bx = x has battery below 20%
- Dx = x must return to dock
- Px = x is carrying a payload

Formalization: ∀x((Rx · Bx · ~Px) ⊃ Dx)

## 4.2 Database Query Connection

SQL `WHERE` clauses are essentially predicate logic:

```sql
-- "Find all robots that have firmware < v1.20 and are operational"
SELECT * FROM robots WHERE firmware_version < 'v1.20' AND status = 'operational';
```

This is: ∃x(Rx · Fx · Ox) — "there exists a robot that has old firmware and is operational"

```sql
-- "Check that ALL robots have been updated"  
SELECT COUNT(*) = 0 FROM robots WHERE firmware_version < 'v1.20';
```

This is checking: ~∃x(Rx · Fx) which is equivalent to ∀x(Rx ⊃ ~Fx) — "no robot has old firmware."

## 4.3 Formal Specifications

When writing system requirements:
- "The system shall ensure that **every** request receives a response within 100ms."
  - ∀x(Request(x) ⊃ ∃y(Response(y) · PairsTo(y,x) · Time(y) ≤ 100))

- "**No** two robots shall occupy the same tile simultaneously."
  - ∀x∀y∀t∀tile((Robot(x) · Robot(y) · x ≠ y · At(x, tile, t)) ⊃ ~At(y, tile, t))

---

## Key Takeaways

1. **Universal = conditional (⊃); Existential = conjunction (·).** This is the most common translation error. Get it right.
2. **∀ means "for all"; ∃ means "there exists."** Together they can express any quantity.
3. **Quantifier negation** (~∀ = ∃~, ~∃ = ∀~) is used constantly — "not all" = "some... not."
4. **Four quantifier rules** (UI, UG, EI, EG) extend natural deduction to quantified statements. **EI before UI.**
5. **Identity** lets you express uniqueness, "exactly one," "at most," and "only."
6. **Predicate logic maps directly to** database queries, formal specs, and for/exists loops in code.
