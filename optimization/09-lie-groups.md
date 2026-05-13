# Lie Groups & Manifold Optimization

> **A comprehensive reference — Lie groups, Lie algebras, exponential maps, and
> optimization on manifolds for robotics and SLAM.**
>
> Rotations and rigid-body poses live on curved spaces (manifolds), not in Euclidean
> vector spaces. This guide develops the theory from first principles and connects
> every concept to concrete robotics applications in the OKS AMR system.

## Prerequisites
- [Day 1: Vectors, Matrices, Transformations](weeks/week-01/day-01-vectors-matrices.md)
- [Day 2: Eigenvalues, PD, SVD](weeks/week-01/day-02-eigenvalues-svd.md)
- [Day 6: Nonlinear Least Squares](weeks/week-01/day-06-nonlinear-ls.md)
- [08: Matrix Essentials](08-matrix-essentials.md) — orthogonal matrices, skew-symmetric matrices
- [06: Graph-Based Optimization](06-graph-optimization.md) — pose graphs, factor graphs

---

## 1. Why Rotations Break Euclidean Math

### 1.1 The "Add Delta" Failure

In Euclidean space, we update a parameter by simple addition: $x \leftarrow x + \Delta x$.
For a rotation matrix $R \in \mathbb{R}^{3 \times 3}$, this fails catastrophically:

$$R_\text{new} = R + \Delta R \quad \text{(WRONG)}$$

The result is **not a rotation matrix**. Rotation matrices must satisfy $R^T R = I$ and
$\det(R) = 1$. Adding an arbitrary perturbation destroys both properties. Even if $\Delta R$
is small, $R + \Delta R$ will have a determinant that differs from 1, and its columns will no
longer be orthonormal.

**OKS relevance:** The navigation estimator tracks robot orientation continuously. If the
filter update simply added a correction to the rotation matrix, the orientation estimate
would drift off the rotation manifold within seconds, producing physically impossible states.

### 1.2 Gimbal Lock and Angle Wrapping

**Euler angles** suffer from gimbal lock — a singular configuration where two rotation axes
align and one degree of freedom is lost. For the common ZYX convention, gimbal lock occurs
at pitch $= \pm 90°$:

$$R(\phi, \theta, \psi) \to R(\phi, \pm 90°, \psi) \quad \Rightarrow \quad \text{only } (\phi \mp \psi) \text{ is observable}$$

**Angle wrapping** is another nuisance. The angles $359°$ and $1°$ are only $2°$ apart, but
their arithmetic mean is $180°$, which is diametrically wrong.

**Quaternion double cover:** Every rotation corresponds to **two** unit quaternions: $q$ and
$-q$ represent the same rotation. Naïve interpolation between $q$ and $-q$ traverses a $360°$
path instead of staying still.

### 1.3 Motivating Example — Averaging Rotations

Given $N$ rotation matrices $R_1, \ldots, R_N$, find their "average" rotation. The Euclidean
mean $\bar{R} = \frac{1}{N} \sum_i R_i$ is **not** a rotation matrix.

The correct approach uses the **Riemannian (Karcher) mean** on $SO(3)$:

$$\bar{R} = \arg\min_{R \in SO(3)} \sum_{i=1}^{N} \| \text{Log}(R^T R_i) \|^2$$

This requires the Lie group logarithmic map (Section 5.4) and iterative optimization on the
manifold (Section 7). The algorithm:

1. Initialize $\bar{R} = R_1$
2. Compute $\delta = \frac{1}{N} \sum_{i=1}^{N} \text{Log}(\bar{R}^T R_i)$
3. Update $\bar{R} \leftarrow \bar{R} \cdot \text{Exp}(\delta)$
4. Repeat until $\|\delta\| < \varepsilon$

**OKS relevance:** When fusing multiple orientation estimates (wheel odometry, IMU, lidar
matching), the estimator must average rotations correctly — this exact algorithm applies.

---

## 2. Group Theory Crash Course

### 2.1 Group Axioms

A **group** $(G, \cdot)$ is a set $G$ with a binary operation $\cdot$ satisfying four axioms:

| Axiom | Statement | Rotation Example ($SO(3)$) |
|-------|-----------|---------------------------|
| **Closure** | $a, b \in G \implies a \cdot b \in G$ | Product of two rotations is a rotation |
| **Associativity** | $(a \cdot b) \cdot c = a \cdot (b \cdot c)$ | Matrix multiplication is associative |
| **Identity** | $\exists\, e \in G : e \cdot a = a \cdot e = a$ | The identity matrix $I_{3 \times 3}$ |
| **Inverse** | $\forall\, a \in G,\, \exists\, a^{-1} : a \cdot a^{-1} = e$ | $R^{-1} = R^T$ for rotations |

A group is **abelian** (commutative) if $a \cdot b = b \cdot a$ for all $a, b$. Notably:
- $SO(2)$ is abelian — 2D rotations commute
- $SO(3)$ is **not** abelian — 3D rotations do not commute in general

### 2.2 Subgroups, Homomorphisms, Isomorphisms

A **subgroup** $H \leq G$ is a subset that is itself a group under the same operation.

| Concept | Definition | Example |
|---------|-----------|---------|
| **Subgroup** | $H \subseteq G$, closed under $\cdot$ and $^{-1}$ | $SO(n) \leq O(n) \leq GL(n)$ |
| **Homomorphism** | $\phi: G \to H$ with $\phi(ab) = \phi(a)\phi(b)$ | $\det: GL(n) \to \mathbb{R}^*$ is a homomorphism |
| **Isomorphism** | Bijective homomorphism | $SO(2) \cong U(1) \cong S^1$ (the unit circle) |
| **Kernel** | $\ker(\phi) = \{g \in G : \phi(g) = e_H\}$ | $\ker(\det) = SL(n)$ (unit determinant matrices) |

### 2.3 Matrix Groups

The groups relevant to robotics are all **matrix Lie groups** — groups whose elements are
invertible matrices and whose group operation is matrix multiplication.

| Group | Definition | Dimension | Elements |
|-------|-----------|-----------|----------|
| $GL(n)$ | General linear: all invertible $n \times n$ matrices | $n^2$ | Any invertible matrix |
| $SL(n)$ | Special linear: $\det(A) = 1$ | $n^2 - 1$ | Volume-preserving transforms |
| $O(n)$ | Orthogonal: $A^T A = I$ | $n(n-1)/2$ | Rotations and reflections |
| $SO(n)$ | Special orthogonal: $A^T A = I$, $\det = 1$ | $n(n-1)/2$ | Pure rotations |
| $SE(n)$ | Special Euclidean: rotation + translation | $n(n+1)/2$ | Rigid body motions |
| $Sim(3)$ | Similarity: rotation + translation + scale | 7 | Monocular SLAM poses |

The containment hierarchy:

$$SO(n) \subset O(n) \subset GL(n) \qquad \text{and} \qquad SO(n) \subset SE(n)$$

### 2.4 Worked Examples

**Example 1: Verifying $SO(2)$ is a group**

Elements: $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$

- **Closure:** $R(\alpha) R(\beta) = R(\alpha + \beta)$ — still a rotation ✓
- **Associativity:** Inherited from matrix multiplication ✓
- **Identity:** $R(0) = I_2$ ✓
- **Inverse:** $R(\theta)^{-1} = R(-\theta) = R(\theta)^T$ ✓

**Example 2: $SO(3)$ is non-abelian**

Let $R_x(90°)$ be a $90°$ rotation about $x$ and $R_z(90°)$ be about $z$:

$$R_x(90°) R_z(90°) \neq R_z(90°) R_x(90°)$$

You can verify this by matrix multiplication. The non-commutativity of 3D rotations is why
the order of rotations matters in robotics — applying yaw then pitch differs from pitch then yaw.

---

## 3. Manifolds and Tangent Spaces

### 3.1 Smooth Manifold Intuition

A **smooth manifold** is a space that locally resembles Euclidean space but may have global
curvature. The classic analogy: the Earth's surface is a 2D manifold — locally flat (use a
street map), globally curved (it's a sphere).

Formally, an $n$-dimensional manifold $\mathcal{M}$ is a topological space where every point
$p \in \mathcal{M}$ has a neighborhood homeomorphic to an open subset of $\mathbb{R}^n$.

Key manifolds in robotics:

| Manifold | Dimension | Topology | Robotics Use |
|----------|-----------|----------|-------------|
| $S^1$ (circle) | 1 | Compact, no boundary | 2D heading angle |
| $S^2$ (sphere) | 2 | Compact, no boundary | Gravity direction |
| $SO(3)$ | 3 | Compact, $\cong \mathbb{RP}^3$ | 3D orientation |
| $SE(2)$ | 3 | Non-compact | 2D robot pose |
| $SE(3)$ | 6 | Non-compact | 6-DOF pose |
| $S^3$ (unit quaternions) | 3 | Double cover of $SO(3)$ | Orientation with interpolation |

### 3.2 Charts, Atlases, Coordinate Maps

A **chart** $(U, \varphi)$ is an open set $U \subset \mathcal{M}$ together with a homeomorphism
$\varphi: U \to \mathbb{R}^n$ (a coordinate map). An **atlas** is a collection of charts covering
the entire manifold.

For $SO(3)$, common charts include:
- **Euler angles** $(\phi, \theta, \psi) \to R$ — singular at gimbal lock (not a global chart)
- **Axis-angle** $(\hat{n}, \theta) \to R$ — singular at $\theta = 0$ and $\theta = \pi$
- **Exponential coordinates** $\omega \in \mathbb{R}^3 \to \text{Exp}(\omega) \in SO(3)$ — covers a neighborhood of $I$

No single chart can cover $SO(3)$ globally without singularity. This is a topological fact:
$SO(3)$ is not homeomorphic to $\mathbb{R}^3$. Unit quaternions provide a double cover via $S^3$.

### 3.3 Tangent Space $T_p \mathcal{M}$

At each point $p$ on a manifold, the **tangent space** $T_p\mathcal{M}$ is the vector space of all
velocity vectors of curves passing through $p$. It is always a flat, Euclidean vector space of
the same dimension as $\mathcal{M}$.

For $SO(3)$ at the identity $I$:

$$T_I SO(3) = \{ \dot{R}(0) : R(t) \text{ is a curve in } SO(3) \text{ with } R(0) = I \}$$

Since $R(t)^T R(t) = I$ for all $t$, differentiating:

$$\dot{R}^T R + R^T \dot{R} = 0 \implies \dot{R}^T + \dot{R} = 0 \text{ at } t=0$$

So $\dot{R}(0)$ is a **skew-symmetric** matrix. The tangent space at the identity of $SO(3)$ is
the space of $3 \times 3$ skew-symmetric matrices, which is exactly the Lie algebra
$\mathfrak{so}(3)$ (Section 5).

### 3.4 Why Tangent Space Matters for Optimization

Optimization on a manifold proceeds by:

1. **Lift** the problem to the tangent space (a vector space where calculus works normally)
2. **Solve** a linear system in the tangent space (standard Gauss-Newton, LM, etc.)
3. **Retract** the solution back onto the manifold (using the exponential map)

This is the "optimize in tangent space, retract to manifold" pattern that underpins every
modern SLAM solver (g2o, GTSAM, Ceres with manifold parameterization).

```
Manifold M:     ──── p ──── q ────          (curved)
                     ↕ Exp/Log
Tangent T_p M:  ──── 0 ──── δ ────          (flat, Euclidean)
                     ↑ solve here
```

**OKS relevance:** The OKS navigation estimator's state includes an orientation on $SO(3)$
and a 2D pose on $SE(2)$. All filter corrections are computed in tangent space and applied
via the exponential map — never by direct matrix addition.

---

## 4. Core Lie Groups for Robotics

### 4.1 SO(2): 2D Rotations

**Dimension:** 1 (one angle $\theta$)

**Matrix form:**
$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \in SO(2)$$

**Lie algebra** $\mathfrak{so}(2)$: $2 \times 2$ skew-symmetric matrices

$$\omega^\wedge = \begin{pmatrix} 0 & -\omega \\ \omega & 0 \end{pmatrix}, \quad \omega \in \mathbb{R}$$

**Exponential map:** $\text{Exp}(\omega) = R(\omega) = \begin{pmatrix} \cos\omega & -\sin\omega \\ \sin\omega & \cos\omega \end{pmatrix}$

**Logarithmic map:** $\text{Log}(R) = \text{atan2}(\sin\theta, \cos\theta) = \theta$

$SO(2)$ is abelian and isomorphic to the unit circle $S^1$. The exponential map wraps the
real line around the circle — it is a covering map with period $2\pi$.

**OKS relevance:** The OKS AMR operates primarily on a flat warehouse floor. Its heading
angle lives on $SO(2)$. The sensorbar-based heading correction must use the $SO(2)$
exponential map to avoid angle-wrapping discontinuities.

### 4.2 SO(3): 3D Rotations

**Dimension:** 3 (three rotation parameters)

**Representations:**

| Representation | Parameters | Singularities | Interpolation | OKS Use |
|---------------|-----------|--------------|---------------|---------|
| Rotation matrix $R$ | 9 (with 6 constraints) | None | Hard | Internal representation |
| Euler angles $(\phi, \theta, \psi)$ | 3 | Gimbal lock | Easy (but wrong) | Human-readable output |
| Axis-angle $(\hat{n}, \theta)$ | 3 (or 4) | $\theta = 0, \pi$ | SLERP-like | Perturbation model |
| Quaternion $q$ | 4 (with 1 constraint) | None (double cover) | SLERP | IMU integration |
| Rodrigues vector $\rho = \theta \hat{n}$ | 3 | $\|\rho\| = \pi$ | Via exp/log | Lie algebra element |

**Rodrigues' rotation formula:**

Given axis $\hat{n}$ ($\|\hat{n}\| = 1$) and angle $\theta$:

$$R = I + \sin\theta\, [\hat{n}]_\times + (1 - \cos\theta)\, [\hat{n}]_\times^2$$

where $[\hat{n}]_\times$ is the skew-symmetric cross-product matrix:

$$\hat{n} = \begin{pmatrix} n_1 \\ n_2 \\ n_3 \end{pmatrix} \implies [\hat{n}]_\times = \begin{pmatrix} 0 & -n_3 & n_2 \\ n_3 & 0 & -n_1 \\ -n_2 & n_1 & 0 \end{pmatrix}$$

**Quaternion-to-rotation:**

For unit quaternion $q = (w, x, y, z)$ with $w^2 + x^2 + y^2 + z^2 = 1$:

$$R = \begin{pmatrix} 1 - 2(y^2+z^2) & 2(xy - wz) & 2(xz + wy) \\ 2(xy + wz) & 1 - 2(x^2+z^2) & 2(yz - wx) \\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2+y^2) \end{pmatrix}$$

### 4.3 SE(2): 2D Rigid Motions

**Dimension:** 3 (one rotation + two translations)

**Homogeneous coordinates:**

$$T = \begin{pmatrix} R & t \\ 0^T & 1 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{pmatrix} \in SE(2)$$

**Group operations:**

| Operation | Formula |
|-----------|---------|
| **Composition** | $T_1 T_2 = \begin{pmatrix} R_1 R_2 & R_1 t_2 + t_1 \\ 0^T & 1 \end{pmatrix}$ |
| **Inverse** | $T^{-1} = \begin{pmatrix} R^T & -R^T t \\ 0^T & 1 \end{pmatrix}$ |
| **Action on point** | $T \begin{pmatrix} p \\ 1 \end{pmatrix} = \begin{pmatrix} Rp + t \\ 1 \end{pmatrix}$ |

**Lie algebra** $\mathfrak{se}(2)$: $3 \times 3$ matrices of the form

$$\xi^\wedge = \begin{pmatrix} [\omega]_\times & v \\ 0^T & 0 \end{pmatrix} = \begin{pmatrix} 0 & -\omega & v_1 \\ \omega & 0 & v_2 \\ 0 & 0 & 0 \end{pmatrix}, \quad \xi = \begin{pmatrix} v \\ \omega \end{pmatrix} \in \mathbb{R}^3$$

**OKS relevance:** This is the primary state representation for the OKS robot's 2D pose.
The pose graph in the OKS navigation estimator operates on $SE(2)$ — each node is a
robot pose $(x, y, \theta)$ encoded as an element of $SE(2)$.

### 4.4 SE(3): 3D Rigid Body Motions

**Dimension:** 6 (three rotation + three translation)

**Homogeneous coordinates:**

$$T = \begin{pmatrix} R & t \\ 0^T & 1 \end{pmatrix} \in SE(3), \quad R \in SO(3), \; t \in \mathbb{R}^3$$

**Lie algebra** $\mathfrak{se}(3)$: The tangent space at the identity consists of **twists**:

$$\xi^\wedge = \begin{pmatrix} [\omega]_\times & v \\ 0^T & 0 \end{pmatrix} \in \mathbb{R}^{4 \times 4}$$

where $\xi = \begin{pmatrix} v \\ \omega \end{pmatrix} \in \mathbb{R}^6$ is the twist coordinates, $v$ is translational
velocity, and $\omega$ is angular velocity.

**Screw interpretation:** A twist $\xi$ can be interpreted as a screw motion — simultaneous
rotation about and translation along a screw axis. This is Chasles' theorem: every rigid body
displacement is a screw motion.

**Exponential map** (see Section 5.3 for closed form):

$$T = \text{Exp}(\xi) = \begin{pmatrix} \text{Exp}(\omega) & V(\omega) v \\ 0^T & 1 \end{pmatrix}$$

where $V(\omega) = I + \frac{1 - \cos\theta}{\theta^2} [\omega]_\times + \frac{\theta - \sin\theta}{\theta^3} [\omega]_\times^2$ and $\theta = \|\omega\|$.

### 4.5 Sim(3): Similarity Transforms

**Dimension:** 7 (three rotation + three translation + one scale)

$$S = \begin{pmatrix} sR & t \\ 0^T & 1 \end{pmatrix} \in Sim(3), \quad s > 0, \; R \in SO(3), \; t \in \mathbb{R}^3$$

**Lie algebra** $\mathfrak{sim}(3)$: $\zeta = \begin{pmatrix} v \\ \omega \\ \sigma \end{pmatrix} \in \mathbb{R}^7$

**Why needed:** In **monocular SLAM**, the absolute scale is unobservable. Poses can only
be determined up to a similarity transformation. The $Sim(3)$ group allows scale drift to be
corrected during loop closure. ORB-SLAM2/3 uses $Sim(3)$ for this purpose.

**OKS relevance:** While the OKS AMR has wheel encoders (providing scale), if only a
monocular camera were used for SLAM, $Sim(3)$ optimization would be essential.

---

## 5. Lie Algebra and Exponential Map

### 5.1 The Tangent Space at Identity = Lie Algebra $\mathfrak{g}$

The **Lie algebra** $\mathfrak{g}$ of a Lie group $G$ is the tangent space at the identity element,
equipped with a bilinear operation called the **Lie bracket** $[\cdot, \cdot]$.

For matrix Lie groups, the Lie bracket is the matrix commutator:

$$[A, B] = AB - BA$$

| Lie Group | Lie Algebra | Elements | Dimension |
|-----------|------------|----------|-----------|
| $SO(2)$ | $\mathfrak{so}(2)$ | $2 \times 2$ skew-symmetric | 1 |
| $SO(3)$ | $\mathfrak{so}(3)$ | $3 \times 3$ skew-symmetric | 3 |
| $SE(2)$ | $\mathfrak{se}(2)$ | $3 \times 3$ twist matrices | 3 |
| $SE(3)$ | $\mathfrak{se}(3)$ | $4 \times 4$ twist matrices | 6 |
| $Sim(3)$ | $\mathfrak{sim}(3)$ | $4 \times 4$ extended twists | 7 |

### 5.2 Generators and Basis

**$\mathfrak{so}(3)$ generators:**

$$G_1 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}, \quad G_2 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}, \quad G_3 = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

Any element of $\mathfrak{so}(3)$ is a linear combination: $[\omega]_\times = \omega_1 G_1 + \omega_2 G_2 + \omega_3 G_3$

**Hat operator** $(\cdot)^\wedge$: maps a vector to its Lie algebra element:

$$\omega = \begin{pmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \end{pmatrix} \xrightarrow{\wedge} [\omega]_\times = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}$$

**Vee operator** $(\cdot)^\vee$: the inverse of hat, extracting the vector from a skew-symmetric matrix.

$$[\omega]_\times^\vee = \omega$$

**$\mathfrak{se}(3)$ generators:** Six generators, three for rotation ($G_1, G_2, G_3$) and three
for translation ($G_4, G_5, G_6$):

$$G_4 = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}, \quad G_5 = \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}, \quad G_6 = \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Any twist: $\xi^\wedge = v_1 G_4 + v_2 G_5 + v_3 G_6 + \omega_1 G_1 + \omega_2 G_2 + \omega_3 G_3$

### 5.3 Exponential Map: $\mathfrak{g} \to G$ (Closed Forms)

The exponential map connects the Lie algebra to the Lie group:

$$\text{Exp}: \mathfrak{g} \to G, \quad \text{Exp}(X) = \sum_{k=0}^{\infty} \frac{X^k}{k!} = I + X + \frac{X^2}{2!} + \frac{X^3}{3!} + \cdots$$

For our groups, closed-form expressions exist (no infinite series needed).

**$SO(3)$ exponential map (Rodrigues):**

Given $\omega \in \mathbb{R}^3$ with $\theta = \|\omega\|$ and $\hat{\omega} = \omega / \theta$ (unit axis):

$$\text{Exp}(\omega) = I + \frac{\sin\theta}{\theta} [\omega]_\times + \frac{1 - \cos\theta}{\theta^2} [\omega]_\times^2$$

Special case when $\theta \approx 0$ (use Taylor expansion to avoid division by zero):

$$\text{Exp}(\omega) \approx I + [\omega]_\times + \frac{1}{2} [\omega]_\times^2$$

**$SE(3)$ exponential map:**

Given twist $\xi = (v, \omega)^T \in \mathbb{R}^6$, with $\theta = \|\omega\|$:

$$\text{Exp}(\xi^\wedge) = \begin{pmatrix} \text{Exp}([\omega]_\times) & V v \\ 0^T & 1 \end{pmatrix}$$

where $V$ is the **left Jacobian** of $SO(3)$:

$$V = J_l(\omega) = I + \frac{1 - \cos\theta}{\theta^2} [\omega]_\times + \frac{\theta - \sin\theta}{\theta^3} [\omega]_\times^2$$

When $\omega = 0$: $\text{Exp}(\xi^\wedge) = \begin{pmatrix} I & v \\ 0^T & 1 \end{pmatrix}$ (pure translation).

**$SE(2)$ exponential map:**

Given $\xi = (v_1, v_2, \omega)^T$:

If $\omega \neq 0$:
$$\text{Exp}(\xi^\wedge) = \begin{pmatrix} \cos\omega & -\sin\omega & \frac{v_1 \sin\omega + v_2(1 - \cos\omega)}{\omega} \\ \sin\omega & \cos\omega & \frac{v_2 \sin\omega - v_1(1 - \cos\omega)}{\omega} \\ 0 & 0 & 1 \end{pmatrix}$$

If $\omega = 0$: $\text{Exp}(\xi^\wedge) = \begin{pmatrix} 1 & 0 & v_1 \\ 0 & 1 & v_2 \\ 0 & 0 & 1 \end{pmatrix}$ (pure translation).

### 5.4 Logarithmic Map: $G \to \mathfrak{g}$

The logarithmic map is the inverse of the exponential map.

**$SO(3)$ logarithm:**

Given $R \in SO(3)$:

1. Compute $\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right)$
2. If $\theta \approx 0$: $[\omega]_\times \approx \frac{1}{2}(R - R^T)$
3. If $\theta \approx \pi$: Extract $\hat{\omega}$ from the column of $R + I$ with largest norm, then $\omega = \pi \hat{\omega}$
4. Otherwise: $[\omega]_\times = \frac{\theta}{2 \sin\theta} (R - R^T)$

The result $\omega = [\omega]_\times^\vee \in \mathbb{R}^3$ is the axis-angle representation.

**$SE(3)$ logarithm:**

Given $T = \begin{pmatrix} R & t \\ 0^T & 1 \end{pmatrix}$:

1. Compute $\omega = \text{Log}(R)$ using the $SO(3)$ logarithm
2. Compute $V^{-1} = I - \frac{1}{2}[\omega]_\times + \frac{1}{\theta^2}\left(1 - \frac{\theta \sin\theta}{2(1-\cos\theta)}\right) [\omega]_\times^2$
3. Recover translational part: $v = V^{-1} t$
4. Return $\xi = (v, \omega)^T$

### 5.5 Baker-Campbell-Hausdorff (BCH) Formula

For matrix exponentials, in general $e^A e^B \neq e^{A+B}$ unless $A$ and $B$ commute.
The BCH formula gives the exact correction:

$$\text{Log}(\text{Exp}(A) \cdot \text{Exp}(B)) = A + B + \frac{1}{2}[A, B] + \frac{1}{12}([A, [A, B]] + [B, [B, A]]) + \cdots$$

In practice, for small perturbations $\delta$ to a group element $\text{Exp}(\phi)$:

**First-order BCH approximation for $SO(3)$:**

$$\text{Log}(\text{Exp}(\phi) \cdot \text{Exp}(\delta)) \approx \phi + J_r^{-1}(\phi)\, \delta$$

where $J_r(\phi)$ is the right Jacobian of $SO(3)$ (Section 6.1). This approximation is crucial
for deriving Jacobians in pose-graph optimization.

### 5.6 Adjoint Representation

The **adjoint representation** $\text{Ad}_T$ describes how a Lie algebra element transforms
under conjugation by a group element.

For a group element $T$ and Lie algebra element $\xi^\wedge$:

$$T \cdot \text{Exp}(\xi^\wedge) \cdot T^{-1} = \text{Exp}(\text{Ad}_T \cdot \xi)$$

**$SO(3)$ adjoint:**

$$\text{Ad}_R = R \quad \in \mathbb{R}^{3 \times 3}$$

A tangent vector transforms by the rotation itself: $R \cdot \text{Exp}(\omega) \cdot R^T = \text{Exp}(R\omega)$.

**$SE(3)$ adjoint:**

$$\text{Ad}_T = \begin{pmatrix} R & [t]_\times R \\ 0 & R \end{pmatrix} \quad \in \mathbb{R}^{6 \times 6}$$

**Why it matters:** The adjoint maps perturbations between different coordinate frames. If
you apply a perturbation on the left vs. the right, the adjoint converts between them:

$$T \cdot \text{Exp}(\delta_L) = \text{Exp}(\text{Ad}_T \, \delta_L) \cdot T$$

---

## 6. Jacobians on Lie Groups

### 6.1 Left and Right Jacobians of $SO(3)$

The **left Jacobian** $J_l(\omega)$ relates a perturbation in the Lie algebra to a change in the
group element when the perturbation is applied on the **left**:

$$\text{Exp}(\omega + \delta) \approx \text{Exp}(J_l(\omega) \delta) \cdot \text{Exp}(\omega)$$

$$J_l(\omega) = \frac{\sin\theta}{\theta} I + \left(1 - \frac{\sin\theta}{\theta}\right) \hat{\omega} \hat{\omega}^T + \frac{1 - \cos\theta}{\theta} [\hat{\omega}]_\times$$

where $\theta = \|\omega\|$ and $\hat{\omega} = \omega / \theta$.

The **right Jacobian** $J_r(\omega)$ is for perturbation on the **right**:

$$\text{Exp}(\omega + \delta) \approx \text{Exp}(\omega) \cdot \text{Exp}(J_r(\omega) \delta)$$

$$J_r(\omega) = J_l(-\omega)$$

Explicitly:

$$J_r(\omega) = \frac{\sin\theta}{\theta} I + \left(1 - \frac{\sin\theta}{\theta}\right) \hat{\omega} \hat{\omega}^T - \frac{1 - \cos\theta}{\theta} [\hat{\omega}]_\times$$

**Inverse Jacobians** (needed for optimization):

$$J_l^{-1}(\omega) = \frac{\theta/2}{\tan(\theta/2)} I + \left(1 - \frac{\theta/2}{\tan(\theta/2)}\right) \hat{\omega}\hat{\omega}^T - \frac{\theta}{2} [\hat{\omega}]_\times$$

$$J_r^{-1}(\omega) = J_l^{-1}(-\omega)$$

**Small angle limit** ($\theta \to 0$):

$$J_l \approx I + \frac{1}{2}[\omega]_\times, \quad J_r \approx I - \frac{1}{2}[\omega]_\times$$

### 6.2 Left and Right Jacobians of $SE(3)$

The $SE(3)$ Jacobians are $6 \times 6$ matrices with a block structure:

$$\mathcal{J}_l(\xi) = \begin{pmatrix} J_l(\omega) & Q_l(\xi) \\ 0 & J_l(\omega) \end{pmatrix} \in \mathbb{R}^{6 \times 6}$$

where $Q_l$ is a complex expression involving $\omega$ and $v$ (the translational part of the twist).
The full formula for $Q_l$ involves terms up to $[\omega]_\times^3 [v]_\times$ and is given in Barfoot's
*State Estimation for Robotics*, Appendix A.

In practice, for small perturbations, the approximation $\mathcal{J}_l \approx I_6$ is often
sufficiently accurate, especially in iterative optimization where the perturbation decreases
each iteration.

### 6.3 Perturbation Models: Left vs. Right

When computing Jacobians for optimization, we must choose where to apply the perturbation.

**Left perturbation (body-frame):**

$$T \oplus \delta = \text{Exp}(\delta) \cdot T$$

The error function Jacobian under left perturbation:

$$\frac{\partial e}{\partial \delta}\bigg|_{\delta = 0} = \lim_{\delta \to 0} \frac{e(\text{Exp}(\delta) \cdot T) - e(T)}{\delta}$$

**Right perturbation (global-frame):**

$$T \oplus \delta = T \cdot \text{Exp}(\delta)$$

$$\frac{\partial e}{\partial \delta}\bigg|_{\delta = 0} = \lim_{\delta \to 0} \frac{e(T \cdot \text{Exp}(\delta)) - e(T)}{\delta}$$

The two are related by the adjoint:

$$\frac{\partial e}{\partial \delta_L} = \frac{\partial e}{\partial \delta_R} \cdot \text{Ad}_{T^{-1}}$$

**Convention choice:** Both are valid. GTSAM uses right perturbation. Ceres Solver is
flexible. The key is consistency — mixing conventions produces wrong Jacobians.

### 6.4 Chain Rule on Manifolds

For a function $f: G \to \mathbb{R}^m$ composed with a group operation $h(X) = f(A \cdot X \cdot B)$:

$$\frac{\partial h}{\partial \delta}\bigg|_{\delta = 0} = \frac{\partial f}{\partial Y}\bigg|_{Y=AXB} \cdot \text{Ad}_A$$

(under left perturbation of $X$).

More generally, if $g: G_1 \times G_2 \to G_3$ is a smooth map between Lie groups and
$f: G_3 \to \mathbb{R}^m$, the chain rule uses the differential (pushforward) of $g$ evaluated
at the current point, composed with the Jacobian of $f$.

**Practical rule:** For each pose variable in the error function:
1. Write the error in terms of group operations
2. Perturb one variable at a time using $\text{Exp}(\delta)$
3. Linearize using first-order BCH and the Jacobians from 6.1–6.2
4. Extract the Jacobian matrix that multiplies $\delta$

---

## 7. Optimization on Manifolds

### 7.1 Why Standard Gauss-Newton Fails on Manifolds

Standard Gauss-Newton computes a parameter update $\Delta x$ and applies it additively:

$$x_{k+1} = x_k + \Delta x_k$$

For a state $X$ on a manifold, this has two problems:
1. **$X + \Delta X$ may leave the manifold** (e.g., $R + \Delta R \notin SO(3)$)
2. **The Jacobian computation assumes Euclidean tangent directions**, which are only
   valid in the tangent space

### 7.2 Retraction-Based Optimization

The solution is the **retraction** (or exponential map) pattern:

$$X_{k+1} = X_k \oplus \Delta x_k = X_k \cdot \text{Exp}(\Delta x_k)$$

where:
- $\Delta x_k \in \mathbb{R}^n$ lives in the tangent space (a Euclidean vector)
- $\text{Exp}(\Delta x_k)$ maps it to a group element (on the manifold)
- Multiplication keeps us on the manifold

A **retraction** $\mathcal{R}_X: T_X \mathcal{M} \to \mathcal{M}$ is any smooth map satisfying:
- $\mathcal{R}_X(0) = X$ (zero perturbation = no change)
- $D\mathcal{R}_X(0) = \text{Id}$ (first-order match to exponential)

The exponential map is the "gold standard" retraction, but cheaper alternatives exist for
some groups (e.g., Cayley transform for $SO(3)$: $R \cdot (I - \frac{1}{2}[\delta]_\times)^{-1}(I + \frac{1}{2}[\delta]_\times)$).

### 7.3 Manifold Gauss-Newton Algorithm

For a nonlinear least-squares problem on manifolds:

$$\min_{X_1, \ldots, X_N} \sum_{k} \| r_k(X_{i_k}, X_{j_k}) \|_{\Sigma_k}^2$$

where each $X_i$ lives on a Lie group and $r_k$ is a residual between connected nodes.

**Algorithm:**

```
Input: Initial values X_1^(0), ..., X_N^(0), residuals r_k, Jacobians
Output: Optimized X_1*, ..., X_N*

repeat:
    1. Linearize: For each residual r_k, compute
       J_k = d r_k / d delta  (Jacobian w.r.t. tangent space perturbation)
       b_k = r_k(X_i, X_j)   (current residual value)

    2. Build normal equations:
       H = sum_k J_k^T Sigma_k^{-1} J_k    (Hessian approximation)
       g = sum_k J_k^T Sigma_k^{-1} b_k     (gradient)

    3. Solve linear system:  H * delta = -g

    4. Retract (update on manifold):
       For each variable i:
           X_i^{new} = X_i * Exp(delta_i)

    5. Check convergence: |delta| < epsilon or |cost change| < epsilon
until converged
```

This is exactly what g2o, GTSAM, and Ceres (with manifold parameterization) implement.

**OKS relevance:** The OKS pose-graph optimizer runs this algorithm on $SE(2)$ with
odometry and loop-closure constraints. Each iteration linearizes in tangent space and
retracts onto $SE(2)$.

### 7.4 Manifold Levenberg-Marquardt

The Levenberg-Marquardt modification adds a damping term to the normal equations:

$$(H + \lambda D) \, \delta = -g$$

where $D$ is typically the diagonal of $H$ (Marquardt's choice) or the identity (Levenberg's
choice). The damping parameter $\lambda$ is adjusted based on the actual-to-predicted cost
reduction ratio $\rho$:

$$\rho = \frac{\text{cost}(X) - \text{cost}(X \oplus \delta)}{\Delta x^T (g + \frac{1}{2} H \Delta x)}$$

- $\rho > 0.75$: decrease $\lambda$ (trust region grows, behave more like Gauss-Newton)
- $\rho < 0.25$: increase $\lambda$ (trust region shrinks, behave more like gradient descent)
- $\rho < 0$: reject step, increase $\lambda$

The only modification from Euclidean LM is the retraction step: $X_{k+1} = X_k \oplus \delta_k$.

### 7.5 Convergence Properties

| Property | Gauss-Newton | Levenberg-Marquardt |
|----------|-------------|-------------------|
| **Rate near minimum** | Quadratic (if Jacobian is full rank) | Superlinear to quadratic |
| **Global convergence** | Not guaranteed (may diverge) | Guaranteed descent |
| **Per-iteration cost** | Solve $H\delta = -g$ | Solve $(H + \lambda D)\delta = -g$ |
| **Sensitivity to init** | High | Medium (damping helps) |

On manifolds, the convergence proofs carry over from Euclidean space because the
retraction is a diffeomorphism near the identity. The local geometry in the tangent space is
Euclidean, so standard convergence theory applies.

**Key insight:** The manifold structure does **not** slow down convergence. The overhead
is only in computing the exponential/logarithmic maps, which are $O(1)$ for all our groups.

---

## 8. Applications in Robotics

### 8.1 Pose-Graph SLAM ($SE(2)$ and $SE(3)$)

In pose-graph SLAM, the state is a collection of robot poses $\{T_0, T_1, \ldots, T_n\} \subset SE(n)$.
Constraints come from odometry and loop closures.

**Odometry edge between $T_i$ and $T_j$:**

Measurement: $\Delta T_{ij}$ (relative transform from pose $i$ to $j$)

Residual (right perturbation):
$$r_{ij} = \text{Log}(\Delta T_{ij}^{-1} \cdot T_i^{-1} \cdot T_j)$$

The Jacobians of this residual w.r.t. perturbations $\delta_i$ and $\delta_j$:

$$\frac{\partial r_{ij}}{\partial \delta_i} = -J_r^{-1}(r_{ij}) \cdot \text{Ad}_{T_j^{-1}}, \quad \frac{\partial r_{ij}}{\partial \delta_j} = J_r^{-1}(r_{ij})$$

For small residuals (near convergence), $J_r^{-1}(r_{ij}) \approx I$ and the Jacobians simplify.

**OKS pose graph:** The OKS navigation system maintains a local pose graph in $SE(2)$ with:
- Wheel odometry edges (high frequency, moderate accuracy)
- Sensorbar correction edges (when crossing tile boundaries)
- Occasionally loop-closure edges from re-observing known landmarks

```python
import numpy as np

def se2_exp(xi):
    """Exponential map for SE(2). xi = [v1, v2, omega]."""
    v1, v2, omega = xi
    if abs(omega) < 1e-10:
        return np.array([[1, 0, v1],
                         [0, 1, v2],
                         [0, 0, 1]])
    c, s = np.cos(omega), np.sin(omega)
    V = np.array([[s / omega, -(1 - c) / omega],
                  [(1 - c) / omega, s / omega]])
    t = V @ np.array([v1, v2])
    return np.array([[c, -s, t[0]],
                     [s,  c, t[1]],
                     [0,  0, 1]])

def se2_log(T):
    """Logarithmic map for SE(2). Returns xi = [v1, v2, omega]."""
    c, s = T[0, 0], T[1, 0]
    omega = np.arctan2(s, c)
    t = T[:2, 2]
    if abs(omega) < 1e-10:
        return np.array([t[0], t[1], omega])
    half_omega = omega / 2.0
    V_inv = half_omega / np.tan(half_omega) * np.eye(2)
    V_inv += np.array([[0, half_omega], [-half_omega, 0]])
    v = V_inv @ t
    return np.array([v[0], v[1], omega])
```

### 8.2 IMU Preintegration (Forster's Method)

IMU preintegration is a landmark technique (Forster et al., 2017) that compounds many
high-rate IMU measurements between two keyframes into a single relative motion constraint
on $SO(3) \times \mathbb{R}^3 \times \mathbb{R}^3$ (rotation, velocity, position).

**The preintegration quantities:**

Between keyframe times $i$ and $j$, with IMU readings $(\tilde{\omega}_k, \tilde{a}_k)$:

$$\Delta R_{ij} = \prod_{k=i}^{j-1} \text{Exp}\left((\tilde{\omega}_k - b_g^i) \Delta t\right) \in SO(3)$$

$$\Delta v_{ij} = \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a}_k - b_a^i) \Delta t$$

$$\Delta p_{ij} = \sum_{k=i}^{j-1} \left( \Delta v_{ik} \Delta t + \frac{1}{2} \Delta R_{ik} (\tilde{a}_k - b_a^i) \Delta t^2 \right)$$

**Key innovation:** These preintegrated quantities are computed in a **reference frame
independent of the keyframe states**, so they don't need to be recomputed when the
optimizer updates the keyframe poses. Only when the bias estimate changes significantly
do they need a first-order correction (using the right Jacobian).

**Bias correction (first-order):**

$$\Delta \tilde{R}_{ij}(b_g) \approx \Delta \tilde{R}_{ij}(\bar{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta R_{ij}}{\partial b_g} \delta b_g\right)$$

### 8.3 Camera-Lidar Calibration (Hand-Eye $AX = XB$)

The classic hand-eye calibration problem: find the rigid transform $X \in SE(3)$ between two
sensors given pairs of motions $(A_i, B_i)$ where:

$$A_i X = X B_i$$

$A_i$ is the motion in the first sensor's frame, $B_i$ in the second.

**Rotation-first approach:**

Taking the rotation part: $R_A R_X = R_X R_B$, or equivalently:

$$(R_A - I) R_X = R_X (R_B - I)$$

Using the axis-angle form and the properties of $SO(3)$ logarithm, this can be converted to
a linear system in the Rodrigues parameters.

**Full $SE(3)$ approach:** Solve directly on $SE(3)$ using manifold optimization:

$$\min_{X \in SE(3)} \sum_i \| \text{Log}(X^{-1} A_i^{-1} X B_i) \|^2$$

### 8.4 Trajectory Interpolation (SLERP, Squad)

**SLERP (Spherical Linear Interpolation)** on $SO(3)$:

Given two rotations $R_0, R_1$ and parameter $t \in [0, 1]$:

$$R(t) = R_0 \cdot \text{Exp}(t \cdot \text{Log}(R_0^T R_1))$$

Equivalently using quaternions $q_0, q_1$:

$$q(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} q_0 + \frac{\sin(t\Omega)}{\sin\Omega} q_1$$

where $\Omega = \arccos(q_0 \cdot q_1)$ and we ensure $q_0 \cdot q_1 > 0$ (pick the shorter path).

**Squad (Spherical Quadrangle)** for smooth interpolation through multiple rotations:

Given keyframe quaternions $q_{i-1}, q_i, q_{i+1}$, compute an inner control point:

$$s_i = q_i \cdot \text{Exp}\left(-\frac{1}{4}(\text{Log}(q_i^{-1} q_{i+1}) + \text{Log}(q_i^{-1} q_{i-1}))\right)$$

Then interpolate between $q_i$ and $q_{i+1}$ with a double SLERP:

$$\text{Squad}(q_i, q_{i+1}, s_i, s_{i+1}, t) = \text{SLERP}(\text{SLERP}(q_i, q_{i+1}, t), \text{SLERP}(s_i, s_{i+1}, t), 2t(1-t))$$

**$SE(3)$ interpolation:**

For full rigid-body trajectory interpolation, interpolate rotation (SLERP) and translation
(linear) independently, or use the $SE(3)$ exponential map:

$$T(t) = T_0 \cdot \text{Exp}(t \cdot \text{Log}(T_0^{-1} T_1))$$

**OKS relevance:** The motion planner interpolates between waypoints. For smooth 2D
trajectories, $SE(2)$ SLERP ensures the robot follows a natural arc rather than an awkward
straight-line-then-rotate path.

### 8.5 Visual Odometry Frame Estimation

In visual odometry, the goal is to estimate the camera pose $T_{cw} \in SE(3)$ (camera-to-world)
from 2D-3D correspondences.

**Reprojection error on $SE(3)$:**

Given a 3D point $p_w$ in world coordinates and its observed pixel $z$:

$$r = z - \pi(T_{cw} \cdot p_w)$$

where $\pi$ is the camera projection function.

**Jacobian of reprojection w.r.t. pose (left perturbation):**

$$\frac{\partial r}{\partial \delta} = -\frac{\partial \pi}{\partial p_c} \cdot \frac{\partial (T_{cw} p_w)}{\partial \delta}$$

For a point $p_c = T_{cw} p_w = Rp_w + t$ in camera coordinates:

$$\frac{\partial (T_{cw} p_w)}{\partial \delta} = \begin{pmatrix} I & -[p_c]_\times \end{pmatrix} \in \mathbb{R}^{3 \times 6}$$

(under left perturbation convention where $\delta = (v, \omega)^T$).

This Jacobian is the foundation of all direct and feature-based visual odometry systems
(ORB-SLAM, DSO, LSD-SLAM).

```python
import numpy as np
from scipy.spatial.transform import Rotation

def so3_exp(omega):
    """Exponential map for SO(3) using Rodrigues formula."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + skew(omega)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def so3_log(R):
    """Logarithmic map for SO(3)."""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return 0.5 * vee(R - R.T)
    return (theta / (2 * np.sin(theta))) * vee(R - R.T)

def skew(v):
    """Hat operator: R^3 -> so(3)."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

def vee(S):
    """Vee operator: so(3) -> R^3."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def se3_exp(xi):
    """Exponential map for SE(3). xi = [v1, v2, v3, w1, w2, w3]."""
    v, omega = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    R = so3_exp(omega)
    if theta < 1e-10:
        t = v
    else:
        K = skew(omega / theta)
        V = (np.eye(3)
             + ((1 - np.cos(theta)) / theta**2) * skew(omega)
             + ((theta - np.sin(theta)) / theta**3) * skew(omega) @ skew(omega))
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def se3_log(T):
    """Logarithmic map for SE(3). Returns xi = [v; omega]."""
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log(R)
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        v = t
    else:
        K = skew(omega)
        half_theta = theta / 2
        V_inv = (np.eye(3)
                 - 0.5 * K
                 + (1.0 / theta**2) * (1 - (half_theta * np.cos(half_theta) / np.sin(half_theta))) * K @ K)
        v = V_inv @ t
    return np.concatenate([v, omega])

def left_jacobian_so3(omega):
    """Left Jacobian of SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * skew(omega)
    K = skew(omega / theta)
    return ((np.sin(theta) / theta) * np.eye(3)
            + (1 - np.sin(theta) / theta) * np.outer(omega, omega) / theta**2
            + ((1 - np.cos(theta)) / theta) * K)

def right_jacobian_so3(omega):
    """Right Jacobian of SO(3)."""
    return left_jacobian_so3(-omega)
```

---

## 9. Notation Reference Table

| Symbol | Space | Dimension | Description |
|--------|-------|-----------|-------------|
| $R$ | $SO(3)$ | $3 \times 3$ | Rotation matrix |
| $T$ | $SE(3)$ | $4 \times 4$ | Rigid body transform (homogeneous) |
| $q$ | $S^3$ | 4 | Unit quaternion (double covers $SO(3)$) |
| $\omega$ | $\mathbb{R}^3$ | 3 | Rotation vector (axis-angle, Rodrigues) |
| $\xi$ | $\mathbb{R}^6$ | 6 | Twist coordinates for $SE(3)$: $\xi = (v, \omega)^T$ |
| $[\omega]_\times$ | $\mathfrak{so}(3)$ | $3 \times 3$ | Skew-symmetric (hat of $\omega$) |
| $\omega^\wedge$ | $\mathfrak{so}(3)$ | $3 \times 3$ | Same as $[\omega]_\times$ (hat notation) |
| $\xi^\wedge$ | $\mathfrak{se}(3)$ | $4 \times 4$ | Twist matrix (hat of $\xi$) |
| $(\cdot)^\vee$ | $\mathbb{R}^n$ | — | Vee operator: inverse of hat |
| $\text{Exp}(\cdot)$ | $G$ | — | Lie group exponential map: $\mathfrak{g} \to G$ |
| $\text{Log}(\cdot)$ | $\mathfrak{g}$ | — | Lie group logarithmic map: $G \to \mathfrak{g}$ |
| $\exp(\cdot)$ | $\mathbb{R}^{n\times n}$ | — | Matrix exponential (infinite series) |
| $J_l(\omega)$ | $\mathbb{R}^{3 \times 3}$ | $3 \times 3$ | Left Jacobian of $SO(3)$ |
| $J_r(\omega)$ | $\mathbb{R}^{3 \times 3}$ | $3 \times 3$ | Right Jacobian of $SO(3)$ |
| $\text{Ad}_T$ | $\mathbb{R}^{6 \times 6}$ | $6 \times 6$ | Adjoint representation of $T \in SE(3)$ |
| $\oplus$ | — | — | Manifold "plus": $X \oplus \delta = X \cdot \text{Exp}(\delta)$ |
| $\ominus$ | — | — | Manifold "minus": $X \ominus Y = \text{Log}(Y^{-1} X)$ |
| $T_p \mathcal{M}$ | — | $\dim(\mathcal{M})$ | Tangent space at point $p$ |
| $\mathfrak{so}(3)$ | — | 3 | Lie algebra of $SO(3)$ |
| $\mathfrak{se}(3)$ | — | 6 | Lie algebra of $SE(3)$ |
| $\mathfrak{sim}(3)$ | — | 7 | Lie algebra of $Sim(3)$ |
| $\Sigma$ | $\mathbb{R}^{n \times n}$ | — | Covariance or information matrix |
| $\theta$ | $\mathbb{R}$ | 1 | Rotation angle: $\theta = \|\omega\|$ |
| $\hat{\omega}$ | $S^2$ | 3 | Unit rotation axis: $\hat{\omega} = \omega / \|\omega\|$ |

---

## 10. Recommended Resources

### Textbooks

| Book | Authors | Focus | Level |
|------|---------|-------|-------|
| *State Estimation for Robotics* | T. D. Barfoot | Lie groups for estimation, Jacobians, SLAM | Graduate, applied |
| *A Micro Lie Theory for State Estimation in Robotics* | J. Solà et al. (tech report) | Compact reference for Lie groups in robotics | Reference |
| *Introduction to Smooth Manifolds* | J. M. Lee | Rigorous manifold theory | Graduate, theoretical |
| *Lie Groups, Lie Algebras, and Representations* | B. C. Hall | Pure Lie theory with matrix groups | Graduate, mathematical |
| *Multiple View Geometry* | R. Hartley, A. Zisserman | Projective geometry, SE(3) for vision | Graduate, applied |
| *Probabilistic Robotics* | S. Thrun et al. | EKF/UKF/PF with rotation states | Graduate, applied |

### Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| Forster et al., "On-Manifold Preintegration for Real-Time VIO" | 2017 | IMU preintegration on $SO(3)$ |
| Carlone et al., "Lagrangian Duality in 3D SLAM" | 2015 | Certifiably optimal rotation averaging |
| Kümmerle et al., "g2o: A General Framework for Graph Optimization" | 2011 | The g2o solver used in ORB-SLAM |
| Solà et al., "A Micro Lie Theory for State Estimation" | 2018 | Practical Lie group reference for roboticists |
| Boumal et al., "Manopt: A Matlab Toolbox for Optimization on Manifolds" | 2014 | Manifold optimization algorithms |
| Absil et al., *Optimization Algorithms on Matrix Manifolds* | 2008 | Theoretical foundation for manifold optimization |

### Open-Source Libraries

| Library | Language | Features |
|---------|----------|----------|
| **Sophus** | C++ | SO(2), SO(3), SE(2), SE(3), Sim(3) with autodiff support |
| **Manif** | C++ | Lie group operations with analytic Jacobians |
| **GTSAM** | C++ / Python | Factor graph optimization on Lie groups |
| **g2o** | C++ | Pose-graph and bundle adjustment on manifolds |
| **Ceres Solver** | C++ | General NLS with manifold (local parameterization) support |
| **jaxlie** | Python (JAX) | Differentiable Lie group operations with JAX |
| **lietorch** | Python (PyTorch) | Differentiable Lie groups for deep learning |
| **pymanopt** | Python | Manifold optimization (Riemannian gradient descent, etc.) |
| **scipy.spatial.transform** | Python | Rotation class with conversions (not full Lie theory) |

### Quick Start: Sophus (C++)

```cpp
#include <sophus/se3.hpp>

// Create an SE(3) element from rotation and translation
Sophus::SE3d T_world_body(
    Eigen::Quaterniond(1, 0, 0, 0),  // identity rotation
    Eigen::Vector3d(1.0, 2.0, 3.0)   // translation
);

// Exponential map from twist
Eigen::Matrix<double, 6, 1> xi;
xi << 0.1, 0.2, 0.3, 0.01, 0.02, 0.03;  // [v; omega]
Sophus::SE3d T_delta = Sophus::SE3d::exp(xi);

// Composition
Sophus::SE3d T_new = T_world_body * T_delta;

// Logarithmic map
Eigen::Matrix<double, 6, 1> log_T = T_new.log();

// Adjoint
Eigen::Matrix<double, 6, 6> Ad = T_world_body.Adj();
```

### Quick Start: GTSAM (Python)

```python
import gtsam
import numpy as np

# Create poses
pose0 = gtsam.Pose2(0.0, 0.0, 0.0)
pose1 = gtsam.Pose2(1.0, 0.0, np.pi / 4)

# Relative transform (Lie group operation)
delta = pose0.between(pose1)      # pose0^{-1} * pose1
print(delta)                       # (1.0, 0.0, 0.7854)

# Exponential / Logarithmic maps
xi = gtsam.Pose2.Logmap(delta)    # R^3
recovered = gtsam.Pose2.Expmap(xi)

# Build a factor graph
graph = gtsam.NonlinearFactorGraph()
noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))

# Add odometry factor
graph.add(gtsam.BetweenFactorPose2(0, 1, delta, noise))

# Add prior
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
graph.add(gtsam.PriorFactorPose2(0, pose0, prior_noise))

# Initial values
initial = gtsam.Values()
initial.insert(0, gtsam.Pose2(0.0, 0.0, 0.0))
initial.insert(1, gtsam.Pose2(0.9, 0.1, 0.8))  # noisy initial

# Optimize with Levenberg-Marquardt
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

print("Optimized pose 1:", result.atPose2(1))
```

---

*This reference is part of the [Optimization Learning Plan](00-learning-plan.md). Next: [10: Game Theory & Multi-Agent Optimization](10-game-theory.md).*
