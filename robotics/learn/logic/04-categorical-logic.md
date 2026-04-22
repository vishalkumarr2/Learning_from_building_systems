# 04 — Categorical Logic
### All, some, and none — reasoning about categories

**Covers:** Chapters 4–5 — Categorical Propositions & Categorical Syllogisms (§4.1–5.7)
**Prerequisite:** Lessons 01–03
**Unlocks:** Lesson 05 (Propositional Logic)

---

## Why Should I Care?

Categorical logic is about reasoning with "all," "some," and "none." In engineering:
- "**All** robots with firmware < v1.20 have the SPI bug."
- "**Some** sensorbar readings are stale."
- "**No** robot in the fleet passed the thermal test."

These statements have precise logical structure, and you can combine them into syllogisms to draw valid conclusions — or detect invalid ones.

---

# PART 1 — CATEGORICAL PROPOSITIONS

## 1.1 Standard Form

Every categorical proposition relates two categories (called **terms**). There are exactly four standard forms:

| Form | Code | Structure | Example |
|---|---|---|---|
| Universal Affirmative | **A** | All S are P | All OKS robots are differential-drive |
| Universal Negative | **E** | No S are P | No firmware builds passed testing |
| Particular Affirmative | **I** | Some S are P | Some sensorbar readings are stale |
| Particular Negative | **O** | Some S are not P | Some robots are not operational |

**Memory aid:** **A**ff**I**rmo (A and I are affirmative), n**E**g**O** (E and O are negative).

- **S** = subject term (what we're talking about)
- **P** = predicate term (what we're saying about it)
- **Quantifier** = All / No / Some
- **Copula** = "are" or "are not"

## 1.2 Quality and Quantity

| | Affirmative | Negative |
|---|---|---|
| **Universal** | A (All S are P) | E (No S are P) |
| **Particular** | I (Some S are P) | O (Some S are not P) |

- **Quality:** Affirmative or Negative (does it include or exclude?)
- **Quantity:** Universal or Particular (all or some?)

## 1.3 Distribution

A term is **distributed** when the proposition makes a claim about *every member* of that term's category.

| Form | Subject distributed? | Predicate distributed? |
|---|---|---|
| **A**: All S are P | Yes (talks about all S) | No (not all P need be S) |
| **E**: No S are P | Yes | Yes (excludes all of both) |
| **I**: Some S are P | No | No |
| **O**: Some S are not P | No | Yes (excludes from all of P) |

**Memory trick:** Universal → subject distributed. Negative → predicate distributed.

**Why distribution matters:** It's the key to detecting invalid syllogisms. If a term is distributed in the conclusion but not in the premises, the syllogism is invalid (the fallacy of illicit distribution).

## 1.4 Venn Diagrams for Propositions

Represent each proposition visually with two overlapping circles (S and P):

**A: All S are P** — Shade out the S-only region (no S exists outside P)
```
    S         P
  ┌───┐   ┌───┐
  │///│   │   │
  │///│───│   │     /// = shaded (empty)
  │///│   │   │
  └───┘   └───┘
  [S outside P is empty → all S must be inside P]
```

**E: No S are P** — Shade out the overlap region
```
    S         P
  ┌───┐   ┌───┐
  │   │   │   │
  │   │///│   │     /// = shaded (empty)
  │   │   │   │
  └───┘   └───┘
  [overlap is empty → no S is P]
```

**I: Some S are P** — Place an X in the overlap region
```
    S         P
  ┌───┐   ┌───┐
  │   │   │   │
  │   │ X │   │     X = at least one member exists here
  │   │   │   │
  └───┘   └───┘
```

**O: Some S are not P** — Place an X in the S-only region
```
    S         P
  ┌───┐   ┌───┐
  │ X │   │   │
  │   │   │   │     X = at least one S exists outside P
  │   │   │   │
  └───┘   └───┘
```

## 1.5 The Modern Square of Opposition

The relationships between A, E, I, O:

```
         A ──── contradictory ──── O
         │                         │
    contrary                  subcontrary
         │                         │
         E ──── contradictory ──── I
```

- **Contradictories** (A/O, E/I): One is true, the other is false. Always.
- **Contraries** (A/E): Can both be false, but cannot both be true.
- **Subcontraries** (I/O): Can both be true, but cannot both be false.

**Practical use:** If you know "All robots passed the test" (A) is FALSE, you immediately know "Some robots did not pass the test" (O) is TRUE (contradictory). But you *cannot* conclude "No robots passed the test" (E) — that might also be false.

## 1.6 Conversion, Obversion, Contraposition

Operations that transform propositions while preserving truth value (or not):

**Conversion** (switch S and P):
- E: "No S are P" → "No P are S" ✓ (always valid)
- I: "Some S are P" → "Some P are S" ✓ (always valid)
- A: "All S are P" → "All P are S" ✗ (INVALID! "All dogs are mammals" ≠ "All mammals are dogs")

**Obversion** (change quality, replace P with non-P):
- A: "All S are P" → "No S are non-P" ✓
- E: "No S are P" → "All S are non-P" ✓
- I: "Some S are P" → "Some S are not non-P" ✓
- O: "Some S are not P" → "Some S are non-P" ✓

**Contraposition** (switch S and P, replace both with complements):
- A: "All S are P" → "All non-P are non-S" ✓
- O: "Some S are not P" → "Some non-P are not non-S" ✓
- E and I: NOT generally valid by contraposition.

---

# PART 2 — CATEGORICAL SYLLOGISMS

## 2.1 What Is a Syllogism?

A **categorical syllogism** is an argument with:
- Exactly **two premises** and **one conclusion**
- Exactly **three terms**, each appearing exactly twice
- Each proposition in standard form (A, E, I, or O)

**Terms:**
- **Major term (P):** The predicate of the conclusion
- **Minor term (S):** The subject of the conclusion  
- **Middle term (M):** Appears in both premises but NOT in the conclusion

**Standard form layout:**
```
Major premise:  [contains M and P]
Minor premise:  [contains M and S]
Conclusion:     [S ... P]
```

**Example:**
```
All SPI-connected devices (M) are susceptible to clock jitter (P).     [A]
All sensorbars (S) are SPI-connected devices (M).                       [A]
∴ All sensorbars (S) are susceptible to clock jitter (P).               [A]
```

This is Figure **AAA-1** — valid. (We'll explain figures and moods shortly.)

## 2.2 Mood and Figure

**Mood** = the letter combination of the three propositions (e.g., AAA, EIO, AOO).

**Figure** = the position of the middle term:

| Figure | Major Premise | Minor Premise |
|---|---|---|
| 1 | M — P | S — M |
| 2 | P — M | S — M |
| 3 | M — P | M — S |
| 4 | P — M | M — S |

There are 4 × 4 × 4 = 64 possible moods, and 4 figures, giving 256 possible syllogistic forms. Only **15** are valid (unconditionally, in the Boolean interpretation).

## 2.3 Venn Diagram Test for Validity

To test a syllogism with Venn diagrams:
1. Draw three overlapping circles (S, M, P).
2. Diagram the **universal** premise first (shading).
3. Then diagram the **particular** premise (X mark).
4. Check: does the diagram already show the conclusion?
   - If yes → **Valid**
   - If no → **Invalid**

**Rule:** Only diagram the premises. Never diagram the conclusion. Then check if the conclusion appears automatically.

## 2.4 The Five Rules for Valid Syllogisms

Instead of drawing Venn diagrams every time, you can apply these rules:

1. **The middle term must be distributed at least once.**
   - Violation → Undistributed Middle
   - "All P are M. All S are M. ∴ All S are P." — M is never distributed (predicate of A propositions).

2. **If a term is distributed in the conclusion, it must be distributed in a premise.**
   - Violation → Illicit Major or Illicit Minor
   - If P is distributed in conclusion but not in the premise → Illicit Major.

3. **Two negative premises are not allowed.**
   - Violation → Exclusive Premises

4. **A negative premise requires a negative conclusion (and vice versa).**
   - Violation → Drawing Affirmative from Negative (or Negative from Affirmative)

5. **If both premises are universal, the conclusion cannot be particular.**
   - (Boolean interpretation — no existential import for universals)

**Practical workflow:** Check rules 3 → 4 → 1 → 2 → 5 in order. First violation found = invalid.

## 2.5 Common Valid Forms (Named Syllogisms)

| Name | Figure | Mood | Example Pattern |
|---|---|---|---|
| Barbara | 1 | AAA | All M are P. All S are M. ∴ All S are P. |
| Celarent | 1 | EAE | No M are P. All S are M. ∴ No S are P. |
| Darii | 1 | AII | All M are P. Some S are M. ∴ Some S are P. |
| Camestres | 2 | AEE | All P are M. No S are M. ∴ No S are P. |
| Ferio | 1 | EIO | No M are P. Some S are M. ∴ Some S are not P. |
| Disamis | 3 | IAI | Some M are P. All M are S. ∴ Some S are P. |

**Engineering example (Barbara):**
```
All devices with firmware < v1.20 (M) have the SPI timing bug (P).
All robots in building 3 (S) have firmware < v1.20 (M).
∴ All robots in building 3 (S) have the SPI timing bug (P).
```

**Engineering example (Camestres):**
```
All robots that passed testing (P) have firmware ≥ v1.24 (M).
No robots in fleet X (S) have firmware ≥ v1.24 (M).
∴ No robots in fleet X (S) passed testing (P).
```

## 2.6 Common Invalid Forms (Fallacies)

**Undistributed Middle:**
```
All robots with the bug (P) run Linux (M).
All OKS robots (S) run Linux (M).
∴ All OKS robots (S) have the bug (P).    ← INVALID!
```
M (run Linux) is predicate of two A-propositions → never distributed. Lots of things run Linux without having the bug.

**Illicit Major:**
```
All SPI devices (M) are affected by the bug (P).
No I2C devices (S) are SPI devices (M).
∴ No I2C devices (S) are affected by the bug (P).    ← INVALID!
```
P is distributed in the E conclusion but not in the A premise. The bug might affect I2C devices for a different reason.

---

## Key Takeaways

1. **Four standard forms:** A (All S are P), E (No S are P), I (Some S are P), O (Some S are not P). Every categorical claim can be put in one of these forms.
2. **Distribution is the key concept** for checking validity — track which terms talk about "all" of their category.
3. **The five rules** let you check any syllogism quickly without Venn diagrams.
4. **Undistributed middle** is the most common syllogistic fallacy in everyday reasoning.
5. **Translate engineering claims into standard form** to check their logic: "All X are Y" / "No X are Y" / "Some X are Y" / "Some X are not Y."
