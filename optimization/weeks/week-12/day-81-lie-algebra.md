# Day 81: Lie Algebra Deep Dive

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *The Lie algebra is where calculus lives — the flat tangent space where you can add, subtract, and differentiate.*

## Navigation
- **Previous:** [Day 80: Exponential and Logarithmic Maps](day-80-exponential-map.md)
- **Next:** [Day 82: Adjoint Representation and BCH](day-82-adjoint-bch.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When the navigation estimator linearizes motion models, it works in the Lie algebra $\mathfrak{se}(2)$ — the tangent space where ordinary linear algebra applies. The Lie bracket captures how rotational and translational perturbations interact, which matters when composing uncertain incremental motions from wheel encoders and sensorbar.

---

## Theory (45 min)

### 81.1 Tangent Space at the Identity

The Lie algebra $\mathfrak{g}$ of a Lie group $G$ is the tangent space at the identity $T_e G$. For matrix groups:

$$\mathfrak{g} = \left\{ \left.\frac{d}{dt}\right|_{t=0} \gamma(t) : \gamma(0) = I, \; \gamma(t) \in G \right\}$$

For SO(3): $\mathfrak{so}(3) = \{A \in \mathbb{R}^{3\times3} : A^T = -A\}$ — the space of $3\times3$ skew-symmetric matrices.

### 81.2 Generators of $\mathfrak{so}(3)$

The standard basis (generators) of $\mathfrak{so}(3)$:

$$E_1 = \begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix}, \quad
E_2 = \begin{pmatrix}0&0&1\\0&0&0\\-1&0&0\end{pmatrix}, \quad
E_3 = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}$$

These satisfy $E_i = [\mathbf{e}_i]_\times$ where $\mathbf{e}_i$ are standard basis vectors. Any $\Omega \in \mathfrak{so}(3)$ decomposes as $\Omega = \omega_1 E_1 + \omega_2 E_2 + \omega_3 E_3$.

### 81.3 The Lie Bracket

The **Lie bracket** is the commutator of matrix multiplication:

$$[X, Y] = XY - YX$$

For $\mathfrak{so}(3)$, the bracket of generators follows the **cross product** rule:

$$[E_i, E_j] = \varepsilon_{ijk} E_k$$

where $\varepsilon_{ijk}$ is the Levi-Civita symbol. Explicitly:

$$[E_1, E_2] = E_3, \quad [E_2, E_3] = E_1, \quad [E_3, E_1] = E_2$$

In vector form: $[\hat{\mathbf{a}}, \hat{\mathbf{b}}] = \widehat{\mathbf{a} \times \mathbf{b}}$. The Lie bracket of $\mathfrak{so}(3)$ *is* the cross product.

### 81.4 Structure Constants

The **structure constants** $c_{ij}^k$ define the bracket in terms of generators:

$$[E_i, E_j] = \sum_k c_{ij}^k E_k$$

For $\mathfrak{so}(3)$: $c_{ij}^k = \varepsilon_{ijk}$ (the Levi-Civita symbol).

### 81.5 Baker-Campbell-Hausdorff (BCH) Formula

For the product of two exponentials:

$$\exp(X)\exp(Y) = \exp(Z)$$

where $Z$ is given by the BCH series:

$$Z = X + Y + \frac{1}{2}[X, Y] + \frac{1}{12}\bigl([X,[X,Y]] - [Y,[X,Y]]\bigr) + \cdots$$

**First-order approximation** (valid for small $X, Y$):

$$Z \approx X + Y + \frac{1}{2}[X, Y]$$

---

## Implementation (60 min)

```python
import numpy as np

def hat3(omega):
    """R^3 -> so(3)."""
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def vee3(Omega):
    """so(3) -> R^3."""
    return np.array([Omega[2,1], Omega[0,2], Omega[1,0]])

# --- Generators ---
E1 = hat3([1, 0, 0])
E2 = hat3([0, 1, 0])
E3 = hat3([0, 0, 1])
generators = [E1, E2, E3]

def lie_bracket(X, Y):
    """Compute [X, Y] = XY - YX."""
    return X @ Y - Y @ X

# Verify bracket relations
print("Verify [E1, E2] = E3:")
print(np.allclose(lie_bracket(E1, E2), E3))

print("Verify [E2, E3] = E1:")
print(np.allclose(lie_bracket(E2, E3), E1))

print("Verify [E3, E1] = E2:")
print(np.allclose(lie_bracket(E3, E1), E2))

# --- Structure constants ---
print("\nStructure constants c_{ij}^k:")
for i in range(3):
    for j in range(3):
        bracket = lie_bracket(generators[i], generators[j])
        coeffs = vee3(bracket)
        if np.linalg.norm(coeffs) > 1e-10:
            for k in range(3):
                if abs(coeffs[k]) > 1e-10:
                    print(f"  c_{i+1}{j+1}^{k+1} = {coeffs[k]:+.0f}")

# --- BCH approximation ---
def so3_exp(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + hat3(omega)
    K = hat3(omega / theta)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K

def so3_log(R):
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos_angle)
    if theta < 1e-10:
        return vee3(R - R.T) / 2
    return theta / (2 * np.sin(theta)) * vee3(R - R.T)

omega_a = np.array([0.1, 0.2, -0.15])
omega_b = np.array([-0.05, 0.1, 0.3])

# Exact composition
R_exact = so3_exp(omega_a) @ so3_exp(omega_b)
omega_exact = so3_log(R_exact)

# BCH first-order
bracket_ab = np.cross(omega_a, omega_b)  # [a,b] in vector form
omega_bch1 = omega_a + omega_b + 0.5 * bracket_ab

# BCH second-order
term2 = (1/12) * (np.cross(omega_a, bracket_ab) - np.cross(omega_b, bracket_ab))
omega_bch2 = omega_bch1 + term2

print(f"\nExact:  {np.round(omega_exact, 6)}")
print(f"BCH-1:  {np.round(omega_bch1, 6)}  err={np.linalg.norm(omega_exact-omega_bch1):.6f}")
print(f"BCH-2:  {np.round(omega_bch2, 6)}  err={np.linalg.norm(omega_exact-omega_bch2):.6f}")
```

---

## Practice Problems (45 min)

**Problem 1:** Compute $[E_1, E_1]$ and $[E_2, E_1]$. What does anti-symmetry of the bracket imply?

<details><summary>Answer</summary>

$[E_1, E_1] = E_1 E_1 - E_1 E_1 = 0$ (always). $[E_2, E_1] = -[E_1, E_2] = -E_3$. Anti-symmetry: $[X,Y] = -[Y,X]$ for all $X,Y$.
</details>

**Problem 2:** Verify the **Jacobi identity**: $[E_1, [E_2, E_3]] + [E_2, [E_3, E_1]] + [E_3, [E_1, E_2]] = 0$.

<details><summary>Answer</summary>

$[E_1, [E_2, E_3]] = [E_1, E_1] = 0$. $[E_2, [E_3, E_1]] = [E_2, E_2] = 0$. $[E_3, [E_1, E_2]] = [E_3, E_3] = 0$. Sum $= 0$. ✓
</details>

**Problem 3:** For rotations of 5° about x and 5° about y, compare the exact composition against BCH first-order. What is the error?

<details><summary>Answer</summary>

$\omega_a = (5\pi/180, 0, 0)$, $\omega_b = (0, 5\pi/180, 0)$. Exact via $\log(\exp(\omega_a)\exp(\omega_b))$ gives a result with a small z-component. BCH-1 predicts $z = \frac{1}{2}\omega_{a_x}\omega_{b_y} \approx 3.8 \times 10^{-3}$ rad. Error is $O(\theta^3) \approx 10^{-5}$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that for $\mathfrak{so}(3)$, $[\hat{a}, \hat{b}] = \widehat{a \times b}$ using the identity $[\omega]_\times \mathbf{v} = \omega \times \mathbf{v}$.

<details><summary>Answer</summary>

$[\hat{a}, \hat{b}] = \hat{a}\hat{b} - \hat{b}\hat{a}$. Apply to arbitrary vector $\mathbf{v}$: $(\hat{a}\hat{b} - \hat{b}\hat{a})\mathbf{v} = a\times(b\times v) - b\times(a\times v)$. Using the BAC-CAB rule: $= b(a\cdot v) - v(a\cdot b) - a(b\cdot v) + v(b\cdot a) = (a\times b)\times v$. So $[\hat{a}, \hat{b}] = \widehat{a\times b}$.
</details>

**Challenge 2:** Show that $\mathfrak{so}(2)$ is abelian (all brackets vanish) and explain why this is consistent with SO(2) being commutative.

<details><summary>Answer</summary>

$\mathfrak{so}(2)$ is 1-dimensional: every element is $\theta \begin{pmatrix}0&-1\\1&0\end{pmatrix}$. For a 1D space, $[X,Y] = \alpha\beta[G,G] = 0$ always. An abelian Lie algebra corresponds to a commutative (abelian) group near the identity, and for connected groups, globally.
</details>

**Challenge 3:** Implement BCH to third order and plot the approximation error vs. $\|\omega\|$ for $\omega$ from $0.01$ to $2.0$ rad.

<details><summary>Answer</summary>

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

angles = np.linspace(0.01, 2.0, 50)
errs1, errs2 = [], []
for a in angles:
    w1 = np.array([a, 0.3*a, -0.2*a])
    w2 = np.array([-0.1*a, a, 0.4*a])
    exact = so3_log(so3_exp(w1) @ so3_exp(w2))
    b = np.cross(w1, w2)
    bch1 = w1 + w2 + 0.5*b
    bch2 = bch1 + (1/12)*(np.cross(w1,b) - np.cross(w2,b))
    errs1.append(np.linalg.norm(exact - bch1))
    errs2.append(np.linalg.norm(exact - bch2))

plt.semilogy(angles, errs1, label='BCH order 1')
plt.semilogy(angles, errs2, label='BCH order 2')
plt.xlabel('||omega||'); plt.ylabel('Error')
plt.legend(); plt.title('BCH Approximation Error')
plt.savefig('bch_error.png')
```
</details>

---

## Connections
- **Back:** [Day 72: SO(2)](../week-11/day-72-so2-rotations.md) — $\mathfrak{so}(2)$ is the simplest Lie algebra (1D, abelian)
- **Back:** [Day 80: Exp/Log](day-80-exponential-map.md) — exp maps Lie algebra elements to group elements
- **Forward:** [Day 82: Adjoint & BCH](day-82-adjoint-bch.md) — adjoint action on the Lie algebra
- **Forward:** [Day 83: Jacobians](day-83-jacobians.md) — derivatives of the exp map
- **OKS:** BCH explains why composing small rotations is approximately additive — and when the approximation breaks down

## Self-Check
- [ ] I can write down the generators of $\mathfrak{so}(3)$ and compute their brackets
- [ ] I understand that the Lie bracket of $\mathfrak{so}(3)$ is the cross product
- [ ] I can state the BCH formula to first order and know when it is accurate
- [ ] I know what structure constants are and can compute them for $\mathfrak{so}(3)$
