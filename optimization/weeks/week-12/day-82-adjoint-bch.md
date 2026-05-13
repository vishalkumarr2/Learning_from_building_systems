# Day 82: Adjoint Representation and BCH

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *The adjoint maps velocities between frames — the Lie-theoretic heart of rigid body mechanics.*

## Navigation
- **Previous:** [Day 81: Lie Algebra Deep Dive](day-81-lie-algebra.md)
- **Next:** [Day 83: Left and Right Jacobians](day-83-jacobians.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When the navigation estimator needs to express a velocity measured in the body frame as a velocity in the world frame, it uses the **adjoint** of SE(3). The 6×6 adjoint matrix maps twists (linear + angular velocity) between frames — this is how IMU measurements (body frame) get fused with wheel odometry (mixed frame) in the EKF.

---

## Theory (45 min)

### 82.1 The Adjoint Representation $\mathrm{Ad}(T)$

For a Lie group $G$ with element $T$ and Lie algebra $\mathfrak{g}$, the **adjoint map** is:

$$\mathrm{Ad}(T) : \mathfrak{g} \to \mathfrak{g}, \quad \hat{\xi} \mapsto T \hat{\xi} T^{-1}$$

This conjugation maps Lie algebra elements to Lie algebra elements. In matrix form:

$$\mathrm{Ad}(T) \xi = (T \hat{\xi} T^{-1})^\vee$$

### 82.2 Adjoint of SE(3): The 6×6 Matrix

For $T = \begin{pmatrix} R & \mathbf{t} \\ 0 & 1 \end{pmatrix} \in SE(3)$ and twist $\xi = \begin{pmatrix}\mathbf{v} \\ \boldsymbol{\omega}\end{pmatrix} \in \mathbb{R}^6$:

$$\mathrm{Ad}(T) = \begin{pmatrix} R & [\mathbf{t}]_\times R \\ \mathbf{0} & R \end{pmatrix} \in \mathbb{R}^{6 \times 6}$$

This maps body-frame twists to spatial-frame twists:

$$\xi_s = \mathrm{Ad}(T_{sb}) \, \xi_b$$

### 82.3 The Small Adjoint $\mathrm{ad}(\xi)$

The **adjoint of the Lie algebra** (lowercase "ad") is the derivative of Ad at the identity:

$$\mathrm{ad}(\xi) : \mathfrak{g} \to \mathfrak{g}, \quad \mathrm{ad}(\xi)\eta = [\xi, \eta]$$

For $\mathfrak{se}(3)$ with $\xi = (\mathbf{v}, \boldsymbol{\omega})$:

$$\mathrm{ad}(\xi) = \begin{pmatrix} [\boldsymbol{\omega}]_\times & [\mathbf{v}]_\times \\ \mathbf{0} & [\boldsymbol{\omega}]_\times \end{pmatrix} \in \mathbb{R}^{6 \times 6}$$

Key identity: $\mathrm{ad}(\xi)\eta = [\hat{\xi}, \hat{\eta}]^\vee$ — the bracket in vector form.

### 82.4 BCH with Adjoint Notation

The BCH formula can be written compactly using adjoint:

$$\log(\exp(X)\exp(Y)) = X + \frac{\mathrm{ad}(X)}{1 - e^{-\mathrm{ad}(X)}} Y + O(Y^2)$$

For small $Y$, the first-order approximation is:

$$\log(\exp(X)\exp(Y)) \approx X + J_l^{-1}(X) Y$$

where $J_l$ is the left Jacobian (Day 83 preview).

### 82.5 Key Identities

| Identity | Expression |
|----------|-----------|
| Ad is a homomorphism | $\mathrm{Ad}(T_1 T_2) = \mathrm{Ad}(T_1)\mathrm{Ad}(T_2)$ |
| Ad-exp relation | $\mathrm{Ad}(\exp(\hat{\xi})) = \exp(\mathrm{ad}(\xi))$ |
| Inverse | $\mathrm{Ad}(T^{-1}) = \mathrm{Ad}(T)^{-1}$ |
| Bracket | $[\xi, \eta] = \mathrm{ad}(\xi)\eta$ |

---

## Implementation (60 min)

```python
import numpy as np

def hat3(w):
    return np.array([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

def vee3(W):
    return np.array([W[2,1], W[0,2], W[1,0]])

def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.eye(3) + hat3(w)
    K = hat3(w/theta)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K

def so3_log(R):
    c = np.clip((np.trace(R)-1)/2, -1, 1)
    theta = np.arccos(c)
    if theta < 1e-10:
        return vee3(R - R.T) / 2
    return theta/(2*np.sin(theta)) * vee3(R - R.T)

def se3_adjoint(T):
    """Compute 6x6 adjoint matrix of SE(3) element T."""
    R = T[:3, :3]
    t = T[:3, 3]
    t_hat = hat3(t)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = t_hat @ R
    Ad[3:, 3:] = R
    return Ad

def se3_small_adjoint(xi):
    """Compute 6x6 ad(xi) matrix for se(3)."""
    v, w = xi[:3], xi[3:]
    ad = np.zeros((6, 6))
    ad[:3, :3] = hat3(w)
    ad[:3, 3:] = hat3(v)
    ad[3:, 3:] = hat3(w)
    return ad

def hat4(xi):
    """R^6 -> se(3): 4x4 matrix."""
    v, w = xi[:3], xi[3:]
    X = np.zeros((4, 4))
    X[:3, :3] = hat3(w)
    X[:3, 3] = v
    return X

def vee4(X):
    """se(3) -> R^6."""
    return np.concatenate([X[:3, 3], vee3(X[:3, :3])])

# --- Build test SE(3) element ---
R = so3_exp([0.3, -0.2, 0.5])
t = np.array([1.0, -0.5, 2.0])
T = np.eye(4); T[:3,:3] = R; T[:3,3] = t

# --- Verify Ad(T) xi = (T xi_hat T^{-1})^vee ---
xi = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.2])
T_inv = np.eye(4); T_inv[:3,:3] = R.T; T_inv[:3,3] = -R.T@t

# Method 1: 6x6 adjoint
Ad_xi_1 = se3_adjoint(T) @ xi

# Method 2: direct conjugation
xi_hat = hat4(xi)
Ad_xi_2 = vee4(T @ xi_hat @ T_inv)

print("Ad(T) @ xi:     ", np.round(Ad_xi_1, 6))
print("(T xi T^-1)^vee:", np.round(Ad_xi_2, 6))
print("Match:", np.allclose(Ad_xi_1, Ad_xi_2))

# --- Verify Ad is a homomorphism ---
R2 = so3_exp([0.1, 0.4, -0.3])
t2 = np.array([-0.5, 1.0, 0.3])
T2 = np.eye(4); T2[:3,:3] = R2; T2[:3,3] = t2

Ad_T1T2 = se3_adjoint(T @ T2)
Ad_T1_Ad_T2 = se3_adjoint(T) @ se3_adjoint(T2)
print("\nAd(T1 T2) == Ad(T1) Ad(T2):", np.allclose(Ad_T1T2, Ad_T1_Ad_T2))

# --- Verify ad(xi)eta = [xi, eta] ---
eta = np.array([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1])
ad_xi_eta = se3_small_adjoint(xi) @ eta
bracket = vee4(hat4(xi) @ hat4(eta) - hat4(eta) @ hat4(xi))
print("\nad(xi)eta:   ", np.round(ad_xi_eta, 6))
print("[xi, eta]:   ", np.round(bracket, 6))
print("Match:", np.allclose(ad_xi_eta, bracket))

# --- BCH numerical test ---
w1 = np.array([0.1, -0.05, 0.2, 0.3, -0.1, 0.15])
w2 = np.array([-0.08, 0.12, -0.06, 0.05, 0.2, -0.1])

T1 = np.eye(4); R1 = so3_exp(w1[3:]); T1[:3,:3] = R1
V1 = np.eye(3) + ((1-np.cos(np.linalg.norm(w1[3:])))/np.linalg.norm(w1[3:])**2)*hat3(w1[3:]) + \
     ((np.linalg.norm(w1[3:])-np.sin(np.linalg.norm(w1[3:])))/np.linalg.norm(w1[3:])**3)*(hat3(w1[3:])@hat3(w1[3:]))
T1[:3,3] = V1 @ w1[:3]

# Simple BCH test: compare [xi, eta] via ad vs direct bracket
bracket_direct = vee4(hat4(w1) @ hat4(w2) - hat4(w2) @ hat4(w1))
bracket_ad = se3_small_adjoint(w1) @ w2
print(f"\nBracket match: {np.allclose(bracket_direct, bracket_ad)}")
```

---

## Practice Problems (45 min)

**Problem 1:** Compute $\mathrm{Ad}(T)$ for $T$ with $R = I$ and $\mathbf{t} = (1, 0, 0)$ (pure translation). What does it do to a twist?

<details><summary>Answer</summary>

$\mathrm{Ad}(T) = \begin{pmatrix}I & [\mathbf{t}]_\times \\ 0 & I\end{pmatrix}$. For twist $(\mathbf{v}, \boldsymbol{\omega})$, the result is $(\mathbf{v} + \mathbf{t}\times\boldsymbol{\omega}, \boldsymbol{\omega})$. Angular velocity is unchanged; linear velocity gains a $\mathbf{t}\times\boldsymbol{\omega}$ correction (the "lever arm" effect).
</details>

**Problem 2:** Verify that $\mathrm{Ad}(T^{-1}) = \mathrm{Ad}(T)^{-1}$ numerically for a random SE(3) element.

<details><summary>Answer</summary>

```python
Ad_Tinv = se3_adjoint(T_inv)
Ad_T_inv = np.linalg.inv(se3_adjoint(T))
print(np.allclose(Ad_Tinv, Ad_T_inv))  # True
```
</details>

**Problem 3:** Compute $\mathrm{ad}(\xi)$ for $\xi = (0, 0, 0, 0, 0, 1)$ (pure rotation about z at unit rate). What is $\mathrm{ad}(\xi)^2$?

<details><summary>Answer</summary>

$\mathrm{ad}(\xi) = \begin{pmatrix}E_3 & 0 \\ 0 & E_3\end{pmatrix}$ where $E_3 = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}$. $\mathrm{ad}(\xi)^2 = \begin{pmatrix}E_3^2 & 0 \\ 0 & E_3^2\end{pmatrix}$ where $E_3^2 = \begin{pmatrix}-1&0&0\\0&-1&0\\0&0&0\end{pmatrix}$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove the Ad-exp relation: $\mathrm{Ad}(\exp(\hat{\xi})) = \exp(\mathrm{ad}(\xi))$.

<details><summary>Answer</summary>

Start from $\mathrm{Ad}(\exp(\hat{\xi}))\eta = \exp(\hat{\xi})\hat{\eta}\exp(-\hat{\xi})$. Expand both sides as power series. The LHS uses the matrix exponential conjugation identity: $e^A B e^{-A} = B + [A,B] + \frac{1}{2!}[A,[A,B]] + \cdots = e^{\mathrm{ad}(A)}B$. Setting $A = \hat{\xi}$, $B = \hat{\eta}$, we get $\mathrm{Ad}(\exp(\hat{\xi}))\eta = \exp(\mathrm{ad}(\xi))\eta$.
</details>

**Challenge 2:** Show that $\mathrm{ad}(\xi)$ is nilpotent for $\mathfrak{se}(3)$ when $\boldsymbol{\omega} = 0$ (pure translation twist).

<details><summary>Answer</summary>

When $\omega = 0$: $\mathrm{ad}(\xi) = \begin{pmatrix}0 & [\mathbf{v}]_\times \\ 0 & 0\end{pmatrix}$. Then $\mathrm{ad}(\xi)^2 = \begin{pmatrix}0 & 0 \\ 0 & 0\end{pmatrix} = 0$. So $\mathrm{ad}(\xi)$ is nilpotent of order 2. This means $\exp(\mathrm{ad}(\xi)) = I + \mathrm{ad}(\xi)$ — the exponential terminates exactly.
</details>

**Challenge 3:** For the BCH formula applied to $\mathfrak{se}(3)$, compute $[(\mathbf{v}_1, \boldsymbol{\omega}_1), (\mathbf{v}_2, \boldsymbol{\omega}_2)]$ explicitly in component form.

<details><summary>Answer</summary>

Using $\mathrm{ad}$: $[(\mathbf{v}_1, \boldsymbol{\omega}_1), (\mathbf{v}_2, \boldsymbol{\omega}_2)] = (\boldsymbol{\omega}_1 \times \mathbf{v}_2 + \mathbf{v}_1 \times \boldsymbol{\omega}_2, \; \boldsymbol{\omega}_1 \times \boldsymbol{\omega}_2)$. The angular part is the SO(3) bracket (cross product), and the linear part has two cross-product terms reflecting the coupling between rotation and translation.
</details>

---

## Connections
- **Back:** [Day 81: Lie Algebra](day-81-lie-algebra.md) — Lie bracket is the core operation; adjoint formalizes it
- **Back:** [Day 80: Exp/Log](day-80-exponential-map.md) — Ad connects group and algebra via $\mathrm{Ad}(\exp) = \exp(\mathrm{ad})$
- **Forward:** [Day 83: Jacobians](day-83-jacobians.md) — left Jacobian $J_l$ appears in BCH formula
- **OKS:** adjoint transforms velocities/forces between body and world frames in the EKF

## Self-Check
- [ ] I can write the 6×6 adjoint matrix for a given SE(3) element
- [ ] I understand the difference between Ad (group) and ad (algebra)
- [ ] I can verify $\mathrm{Ad}(T)\xi = (T\hat{\xi}T^{-1})^\vee$ numerically
- [ ] I know that $\mathrm{ad}(\xi)\eta = [\xi, \eta]$ and can compute it
