# Day 71: Group Axioms and Matrix Groups

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *Every symmetry in physics and robotics hides a group. Today we learn to recognize them.*

## Navigation
- **Previous:** None — this is the start of Phase VI
- **Next:** [Day 72: SO(2) — 2D Rotations](day-72-so2-rotations.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Robot configurations form groups under composition. The set of all planar poses (x, y, θ) with the composition operation is SE(2) — a matrix Lie group. Understanding group axioms tells us *why* composing two valid poses always gives a valid pose, and *why* every pose has a unique inverse. These guarantees underpin every frame transform in the OKS navigation stack.

---

## Theory (45 min)

### 71.1 Group Axioms

A **group** $(G, \cdot)$ is a set $G$ with a binary operation $\cdot$ satisfying four axioms:

| Axiom | Statement | Meaning |
|-------|-----------|---------|
| **Closure** | $a, b \in G \Rightarrow a \cdot b \in G$ | Combining elements stays in the set |
| **Associativity** | $(a \cdot b) \cdot c = a \cdot (b \cdot c)$ | Grouping doesn't matter |
| **Identity** | $\exists\, e \in G : e \cdot a = a \cdot e = a$ | There is a "do nothing" element |
| **Inverse** | $\forall\, a \in G,\;\exists\, a^{-1} : a \cdot a^{-1} = e$ | Every action is undoable |

If additionally $a \cdot b = b \cdot a$ for all $a, b$, the group is **abelian** (commutative).

**Examples:**
- $(\mathbb{Z}, +)$: integers under addition. Identity = 0, inverse of $n$ is $-n$. Abelian.
- $(\mathbb{R}^*, \times)$: nonzero reals under multiplication. Identity = 1. Abelian.
- $(S_3, \circ)$: permutations of 3 elements. **Not** abelian — order matters.

### 71.2 Matrix Groups

Matrix groups use matrix multiplication as the group operation:

$$\text{GL}(n) = \{A \in \mathbb{R}^{n \times n} : \det(A) \neq 0\}$$

The **general linear group** — all invertible $n \times n$ matrices. Closure holds because $\det(AB) = \det(A)\det(B) \neq 0$.

$$\text{O}(n) = \{R \in \mathbb{R}^{n \times n} : R^T R = I\}$$

The **orthogonal group** — matrices preserving lengths. $\det(R) = \pm 1$.

$$\text{SO}(n) = \{R \in \text{O}(n) : \det(R) = +1\}$$

The **special orthogonal group** — pure rotations without reflections.

### 71.3 Subgroups and Hierarchy

A **subgroup** $H \leq G$ is a subset that is itself a group under the same operation. The hierarchy is:

$$\text{SO}(n) \leq \text{O}(n) \leq \text{GL}(n)$$

**Subgroup test:** $H \leq G$ if and only if $H \neq \emptyset$ and $\forall\, a, b \in H : a \cdot b^{-1} \in H$.

### 71.4 Group Homomorphisms

A map $\phi: G \to H$ is a **homomorphism** if $\phi(a \cdot b) = \phi(a) \cdot \phi(b)$. Example: $\det: \text{GL}(n) \to \mathbb{R}^*$ is a homomorphism since $\det(AB) = \det(A)\det(B)$.

---

## Implementation (60 min)

```python
import numpy as np

def verify_closure(elements, operation):
    """Check if operation on any two elements stays in the set."""
    for a in elements:
        for b in elements:
            result = operation(a, b)
            found = any(np.allclose(result, e) for e in elements)
            if not found:
                return False, (a, b, result)
    return True, None

def verify_associativity(elements, operation, n_tests=20):
    """Check (a*b)*c == a*(b*c) for random triples."""
    rng = np.random.default_rng(42)
    for _ in range(n_tests):
        idx = rng.integers(0, len(elements), size=3)
        a, b, c = elements[idx[0]], elements[idx[1]], elements[idx[2]]
        lhs = operation(operation(a, b), c)
        rhs = operation(a, operation(b, c))
        if not np.allclose(lhs, rhs):
            return False, (a, b, c)
    return True, None

def verify_identity(elements, operation, identity):
    """Check e*a == a*e == a for all a."""
    for a in elements:
        if not np.allclose(operation(identity, a), a):
            return False, a
        if not np.allclose(operation(a, identity), a):
            return False, a
    return True, None

def verify_inverse(elements, operation, identity, inv_fn):
    """Check a * a^{-1} == identity for all a."""
    for a in elements:
        a_inv = inv_fn(a)
        if not np.allclose(operation(a, a_inv), identity):
            return False, a
    return True, None

# --- Test SO(2) with 8 evenly-spaced rotations ---
def make_rotation_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
so2_elements = [make_rotation_2d(a) for a in angles]
mat_mul = lambda A, B: A @ B

print("=== SO(2) with 8 discrete rotations ===")
ok, _ = verify_closure(so2_elements, mat_mul)
print(f"Closure:       {'PASS' if ok else 'FAIL'}")
ok, _ = verify_associativity(so2_elements, mat_mul)
print(f"Associativity: {'PASS' if ok else 'FAIL'}")
ok, _ = verify_identity(so2_elements, mat_mul, np.eye(2))
print(f"Identity:      {'PASS' if ok else 'FAIL'}")
ok, _ = verify_inverse(so2_elements, mat_mul, np.eye(2), lambda R: R.T)
print(f"Inverse:       {'PASS' if ok else 'FAIL'}")

# --- Test GL(2): random invertible matrices ---
rng = np.random.default_rng(0)
gl2 = [rng.standard_normal((2, 2)) + 0.5 * np.eye(2) for _ in range(5)]
gl2 = [A for A in gl2 if abs(np.linalg.det(A)) > 0.1]

print("\n=== GL(2) sample check ===")
for A in gl2:
    d = np.linalg.det(A)
    orth = np.allclose(A.T @ A, np.eye(2))
    print(f"det={d:+.3f}, orthogonal={orth}, in_SO2={orth and d > 0}")
```

---

## Practice Problems (45 min)

**Problem 1:** Is $(\{1, -1, i, -i\}, \times)$ a group? Verify all four axioms.

<details><summary>Answer</summary>

Yes. Closure: product of any two is in the set (e.g., $i \times i = -1$). Associativity: inherited from $\mathbb{C}$. Identity: 1. Inverses: $1^{-1}=1$, $(-1)^{-1}=-1$, $i^{-1}=-i$, $(-i)^{-1}=i$. This is the cyclic group $C_4$.
</details>

**Problem 2:** Show that the set of $2 \times 2$ upper-triangular matrices with 1s on the diagonal forms a group under multiplication.

<details><summary>Answer</summary>

$$\begin{pmatrix} 1 & a \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & b \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & a+b \\ 0 & 1 \end{pmatrix}$$

Closure ✓ (still upper-triangular with 1s on diagonal). Associativity ✓ (matrix multiplication). Identity: $a=0$. Inverse: $a \mapsto -a$. This is isomorphic to $(\mathbb{R}, +)$.
</details>

**Problem 3:** Does the set of $2\times 2$ matrices with $\det = 2$ form a group under multiplication?

<details><summary>Answer</summary>

No. The identity matrix has $\det = 1 \neq 2$, so no identity exists. Also, closure fails: $\det(AB) = \det(A)\det(B) = 4 \neq 2$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that in any group, the identity is unique and every element has a unique inverse.

<details><summary>Answer</summary>

**Unique identity:** Suppose $e$ and $f$ are both identities. Then $e = e \cdot f = f$, since $f$ is an identity for the first equality and $e$ is an identity for the second.

**Unique inverse:** Suppose $b$ and $c$ are both inverses of $a$. Then $b = b \cdot e = b \cdot (a \cdot c) = (b \cdot a) \cdot c = e \cdot c = c$.
</details>

**Challenge 2:** Show that $\text{O}(n)$ is a subgroup of $\text{GL}(n)$ using the one-step subgroup test.

<details><summary>Answer</summary>

$\text{O}(n) \neq \emptyset$ since $I \in \text{O}(n)$. For $A, B \in \text{O}(n)$: $AB^{-1} = AB^T$. Check: $(AB^T)^T(AB^T) = B A^T A B^T = B I B^T = I$. So $AB^{-1} \in \text{O}(n)$. ✓
</details>

**Challenge 3:** The quaternion group $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$ has 8 elements. Write out its multiplication table and verify it is non-abelian.

<details><summary>Answer</summary>

Key products: $ij = k$, $ji = -k$. Since $ij \neq ji$, $Q_8$ is non-abelian. The full table is an $8 \times 8$ grid following Hamilton's rules: $i^2 = j^2 = k^2 = ijk = -1$.
</details>

---

## Connections
- **Prerequisites:** Linear algebra (Day 1–2) — matrix operations, determinants
- **Forward:** [Day 72](day-72-so2-rotations.md) deep-dives into SO(2), [Day 73](day-73-so3-rodrigues.md) into SO(3)
- **OKS:** SE(2) and SE(3) are the groups that describe robot poses; understanding the abstract axioms makes their properties predictable

## Self-Check
- [ ] I can state all four group axioms and test them on a given set
- [ ] I can explain the hierarchy GL(n) ⊃ O(n) ⊃ SO(n) and what each constraint adds
- [ ] I can use the one-step subgroup test to prove a subset is a subgroup
- [ ] I can distinguish abelian from non-abelian groups with a concrete example
