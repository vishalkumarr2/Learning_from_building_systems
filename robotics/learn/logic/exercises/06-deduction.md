# Exercise 6 — Natural Deduction

**Covers:** Lesson 06
**Difficulty:** Intermediate to Advanced

---

## Problem 1: Basic Proofs (Inference Rules Only)

Derive the conclusion from the given premises. Use only the 8 rules of inference.

**A.**
```
1. p ⊃ q
2. q ⊃ r
3. p
∴ r
```

**B.**
```
1. p · q
2. p ⊃ r
∴ r · q
```

**C.**
```
1. p ∨ q
2. ~p
3. q ⊃ r
∴ r
```

**D.**
```
1. (p ⊃ q) · (r ⊃ s)
2. p ∨ r
∴ q ∨ s
```

---

## Problem 2: Proofs with Replacement Rules

Derive the conclusion using both inference and replacement rules.

**A.**
```
1. ~p ∨ q
2. ~q
∴ ~p
```

**B.**
```
1. p ⊃ (q · r)
2. ~r
∴ ~p
```

**C.**
```
1. ~(p · q)
2. p
∴ ~q
```

**D.**
```
1. p ≡ q
2. ~q
∴ ~p
```

---

## Problem 3: Conditional Proof

Use conditional proof to derive the conclusion.

**A.**
```
1. p ⊃ q
2. q ⊃ r
∴ p ⊃ r
```

**B.**
```
1. p ⊃ (q ⊃ r)
∴ (p · q) ⊃ r
```

**C.**
```
1. p ⊃ q
∴ (p · r) ⊃ q
```

---

## Problem 4: Indirect Proof

Use indirect proof (proof by contradiction) to derive the conclusion.

**A.**
```
1. p ⊃ q
2. p ⊃ ~q
∴ ~p
```

**B.**
```
1. p ∨ q
2. ~q
3. ~p ⊃ r
∴ p
```

---

## Problem 5: Engineering Proofs

Translate the engineering scenario into propositional logic, then construct a formal proof.

**A.** "If the SPI clock is unstable (s), then the DMA transfer corrupts (d). If the DMA transfer corrupts (d), the sensorbar reads stale data (b). If the sensorbar reads stale data (b), the estimator diverges (e). The SPI clock is unstable. Prove: the estimator diverges."

**B.** "The failure is either hardware (h), firmware (f), or software (w). If it's hardware, bench testing would fail (b). Bench testing passed (~b). If it's software, the same binary on another robot would fail (a). The same binary works fine on another robot (~a). Prove: the failure is firmware."

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1A:**
```
1. p ⊃ q          Premise
2. q ⊃ r          Premise
3. p               Premise
4. q               MP 1, 3
5. r               MP 2, 4
```

**Problem 1B:**
```
1. p · q           Premise
2. p ⊃ r          Premise
3. p               Simp 1
4. r               MP 2, 3
5. q               Simp 1 (via Com: q · p, then Simp)
6. r · q           Conj 4, 5
```

(For step 5, formally: apply Com to get q · p from p · q, then Simp. Or use replacement rules if available.)

**Problem 1C:**
```
1. p ∨ q           Premise
2. ~p              Premise
3. q ⊃ r          Premise
4. q               DS 1, 2
5. r               MP 3, 4
```

**Problem 1D:**
```
1. (p ⊃ q) · (r ⊃ s)  Premise
2. p ∨ r                Premise
3. q ∨ s                CD 1, 2
```

**Problem 2A:**
```
1. ~p ∨ q          Premise
2. ~q              Premise
3. p ⊃ q          Impl 1
4. ~p              MT 3, 2
```

**Problem 2B:**
```
1. p ⊃ (q · r)    Premise
2. ~r              Premise
3. ~r ∨ ~q         Add 2
4. ~q ∨ ~r         Com 3
5. ~(q · r)        DM 3  (or: ~(q · r) from ~r ∨ ~q by DM)
6. ~p              MT 1, 5
```

**Problem 2C:**
```
1. ~(p · q)        Premise
2. p               Premise
3. ~p ∨ ~q         DM 1
4. ~~p             DN 2
5. ~q              DS 3, 4
```

**Problem 2D:**
```
1. p ≡ q           Premise
2. ~q              Premise
3. (p ⊃ q) · (q ⊃ p)  Equiv 1
4. p ⊃ q          Simp 3
5. ~p              MT 4, 2
```

**Problem 3A:**
```
1. p ⊃ q          Premise
2. q ⊃ r          Premise
   ┌──────────────
3. │ p             ACP (Assume for Conditional Proof)
4. │ q             MP 1, 3
5. │ r             MP 2, 4
   └──────────────
6. p ⊃ r          CP 3-5
```

**Problem 3B:**
```
1. p ⊃ (q ⊃ r)   Premise
   ┌──────────────
2. │ p · q         ACP
3. │ p             Simp 2
4. │ q ⊃ r        MP 1, 3
5. │ q             Simp 2 (Com, then Simp)
6. │ r             MP 4, 5
   └──────────────
7. (p · q) ⊃ r   CP 2-6
```

**Problem 3C:**
```
1. p ⊃ q          Premise
   ┌──────────────
2. │ p · r         ACP
3. │ p             Simp 2
4. │ q             MP 1, 3
   └──────────────
5. (p · r) ⊃ q   CP 2-4
```

**Problem 4A:**
```
1. p ⊃ q          Premise
2. p ⊃ ~q         Premise
   ┌──────────────
3. │ p             AIP (Assume for Indirect Proof, assume ~(~p) = p)
4. │ q             MP 1, 3
5. │ ~q            MP 2, 3
6. │ q · ~q        Conj 4, 5 (Contradiction!)
   └──────────────
7. ~p              IP 3-6
```

**Problem 4B:**
```
1. p ∨ q           Premise
2. ~q              Premise
3. ~p ⊃ r         Premise
4. p               DS 1, 2
```

(This one is straightforward with DS — no need for IP. But if you wanted to use IP:)

```
   ┌──────────────
4. │ ~p            AIP
5. │ r             MP 3, 4
6. │ q             DS 1, 4  (Hmm, this gives q)
7. │ q · ~q        Conj 6, 2 (Contradiction)
   └──────────────
8. p               IP 4-7
```

Wait — DS with p ∨ q and ~p gives q, and we have ~q. That's the contradiction. The simpler approach is just DS 1, 2.

**Problem 5A:**
```
1. s ⊃ d          Premise (SPI unstable → DMA corrupts)
2. d ⊃ b          Premise (DMA corrupts → stale data)
3. b ⊃ e          Premise (stale data → estimator diverges)
4. s               Premise (SPI is unstable)
5. d               MP 1, 4
6. b               MP 2, 5
7. e               MP 3, 6
```

Or more elegantly using HS:
```
1. s ⊃ d          Premise
2. d ⊃ b          Premise
3. b ⊃ e          Premise
4. s               Premise
5. s ⊃ b          HS 1, 2
6. s ⊃ e          HS 5, 3
7. e               MP 6, 4
```

**Problem 5B:**
```
1. h ∨ f ∨ w       Premise (hardware or firmware or software)
2. h ⊃ b          Premise (hardware → bench test fails)
3. ~b              Premise (bench test passed)
4. w ⊃ a          Premise (software → other robot fails)
5. ~a              Premise (other robot works)
6. ~h              MT 2, 3  (not hardware)
7. ~w              MT 4, 5  (not software)
8. f ∨ w           DS 1, 6  (eliminate h)
9. f               DS 8, 7  (eliminate w)
```

This is the formal proof behind elimination-based debugging!

</details>
