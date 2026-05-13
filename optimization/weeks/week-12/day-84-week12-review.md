# Day 84: Week 12 Review + Exercises

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *Assemble the full SE(3) toolkit — from matrix construction to optimization-ready Jacobians.*

## Navigation
- **Previous:** [Day 83: Left and Right Jacobians](day-83-jacobians.md)
- **Next:** [Day 85: Manifold Gauss-Newton (Week 13)](../week-13/day-85-manifold-gauss-newton.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
This review synthesizes all Lie group machinery needed for robot state estimation and SLAM. The complete pipeline — build SE(3) from sensor data, chain transforms, compute residuals via log map, linearize via Jacobians, update via exp map — runs in every pose-graph optimizer. Mastering this toolkit is prerequisite for understanding the OKS navigation estimator's manifold-aware EKF.

---

## Theory (45 min)

### Week 12 Concept Map

```
SE(2) (Day 78)                         SE(3) (Day 79)
├── 3×3 homogeneous matrix              ├── 4×4 homogeneous matrix
├── SO(2) ⋉ ℝ²                          ├── SO(3) ⋉ ℝ³
├── Planar pose (x, y, θ)               ├── 6-DOF pose (R, t)
└── V matrix (2D)                       ├── Screw motions (Chasles)
                                        └── Transform chains
        │                                       │
        └──────── Exp / Log Maps (Day 80) ──────┘
                  ├── Rodrigues for SO(3)
                  ├── V matrix for SE(3)
                  ├── Small-angle Taylor forms
                  └── Wrapping at θ = π
                          │
                Lie Algebra (Day 81)
                ├── Generators E₁, E₂, E₃
                ├── Bracket [X,Y] = XY - YX
                ├── Structure constants εᵢⱼₖ
                └── BCH: log(exp X · exp Y) ≈ X + Y + ½[X,Y]
                          │
                Adjoint (Day 82)
                ├── Ad(T): 6×6 frame change for twists
                ├── ad(ξ): bracket as matrix multiply
                └── Ad(exp ξ) = exp(ad ξ)
                          │
                Jacobians (Day 83)
                ├── J_l: left Jacobian
                ├── J_r = J_l(-φ): right Jacobian
                ├── Closed forms + series
                └── SLAM residual linearization
```

### Key Formula Summary

| Concept | Formula |
|---------|---------|
| SE(3) compose | $T_1 T_2 = \begin{pmatrix}R_1R_2 & R_1\mathbf{t}_2+\mathbf{t}_1 \\ 0 & 1\end{pmatrix}$ |
| SE(3) inverse | $T^{-1} = \begin{pmatrix}R^T & -R^T\mathbf{t} \\ 0 & 1\end{pmatrix}$ |
| Rodrigues | $\exp([\omega]_\times) = I + \sin\theta\,K + (1-\cos\theta)K^2$ |
| SE(3) exp | $\exp(\hat\xi) = \begin{pmatrix}\exp([\omega]_\times) & V\mathbf{v} \\ 0 & 1\end{pmatrix}$ |
| $V$ matrix | $V = I + \frac{1-\cos\theta}{\theta^2}K + \frac{\theta-\sin\theta}{\theta^3}K^2$ |
| Lie bracket (so(3)) | $[\hat{a},\hat{b}] = \widehat{a \times b}$ |
| BCH (1st order) | $\log(e^X e^Y) \approx X + Y + \tfrac{1}{2}[X,Y]$ |
| Adjoint SE(3) | $\mathrm{Ad}(T) = \begin{pmatrix}R & [t]_\times R \\ 0 & R\end{pmatrix}$ |
| Left Jacobian | $J_l = \frac{\sin\theta}{\theta}I + (1-\frac{\sin\theta}{\theta})aa^T + \frac{1-\cos\theta}{\theta}[a]_\times$ |
| Right Jacobian | $J_r(\phi) = J_l(-\phi)$ |

---

## Implementation (60 min)

### Complete SE(3) Toolkit

```python
import numpy as np

# === Core utilities ===
def hat3(w):
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

def vee3(W):
    return np.array([W[2,1], W[0,2], W[1,0]])

# === SO(3) ===
def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-10: return np.eye(3) + hat3(w)
    K = hat3(w/theta)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K

def so3_log(R):
    c = np.clip((np.trace(R)-1)/2, -1, 1)
    theta = np.arccos(c)
    if theta < 1e-10: return vee3(R-R.T)/2
    return theta/(2*np.sin(theta)) * vee3(R-R.T)

# === SE(3) ===
def se3_from_Rt(R, t):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t; return T

def se3_inverse(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T@t; return Ti

def V_matrix(w):
    theta = np.linalg.norm(w); K = hat3(w)
    if theta < 1e-10: return np.eye(3) + 0.5*K
    return np.eye(3) + ((1-np.cos(theta))/theta**2)*K + ((theta-np.sin(theta))/theta**3)*K@K

def V_inv(w):
    theta = np.linalg.norm(w); K = hat3(w)
    if theta < 1e-10: return np.eye(3) - 0.5*K
    ht = theta/2
    return np.eye(3) - 0.5*K + (1/theta**2 - (1+np.cos(theta))/(2*theta*np.sin(theta)))*K@K

def se3_exp(xi):
    v, w = xi[:3], xi[3:]
    T = np.eye(4); T[:3,:3] = so3_exp(w); T[:3,3] = V_matrix(w)@v; return T

def se3_log(T):
    w = so3_log(T[:3,:3]); v = V_inv(w) @ T[:3,3]
    return np.concatenate([v, w])

def se3_adjoint(T):
    R, t = T[:3,:3], T[:3,3]
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R; Ad[:3,3:] = hat3(t)@R; Ad[3:,3:] = R
    return Ad

def so3_left_jacobian(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-10: return np.eye(3) + 0.5*hat3(phi)
    a = phi/theta; K = hat3(a)
    return (np.sin(theta)/theta)*np.eye(3) + (1-np.sin(theta)/theta)*np.outer(a,a) + ((1-np.cos(theta))/theta)*K

# === Demo: full transform pipeline ===
# Sensor chain: world -> odom -> base -> imu -> camera
T_w_o = se3_from_Rt(so3_exp([0, 0, 0.1]), [5.0, 3.0, 0.0])
T_o_b = se3_from_Rt(so3_exp([0, 0, 0.3]), [0.2, -0.05, 0.0])
T_b_i = se3_from_Rt(so3_exp([0, 0.01, 0]), [0.1, 0.0, 0.3])
T_b_c = se3_from_Rt(so3_exp([0, -np.pi/2, 0]), [0.15, 0.0, 0.4])

T_w_c = T_w_o @ T_o_b @ T_b_c
print("World -> Camera:\n", np.round(T_w_c, 4))

# Relative pose between two robots
T_w_A = se3_from_Rt(so3_exp([0, 0, 0.5]), [10, 5, 0])
T_w_B = se3_from_Rt(so3_exp([0, 0, -0.3]), [12, 7, 0])
T_A_B = se3_inverse(T_w_A) @ T_w_B
print("\nRelative A->B:\n", np.round(T_A_B, 4))
xi_rel = se3_log(T_A_B)
print("Relative twist:", np.round(xi_rel, 4))

# Extract Euler angles (ZYX)
def to_euler_zyx(T):
    R = T[:3,:3]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0.0
    return np.degrees(np.array([roll, pitch, yaw]))

print("Euler (R,P,Y) of camera:", np.round(to_euler_zyx(T_w_c), 2), "deg")
```

---

## Practice Problems (45 min)

**Problem 1:** Given world poses of a robot at times $t_1, t_2, t_3$, compute the incremental transforms $\Delta T_{12}$ and $\Delta T_{23}$, verify $T_3 = T_1 \cdot \Delta T_{12} \cdot \Delta T_{23}$.

<details><summary>Answer</summary>

$\Delta T_{12} = T_1^{-1} T_2$, $\Delta T_{23} = T_2^{-1} T_3$. Chain: $T_1 \cdot \Delta T_{12} \cdot \Delta T_{23} = T_1 \cdot T_1^{-1} T_2 \cdot T_2^{-1} T_3 = T_3$. ✓
</details>

**Problem 2:** Implement a function that computes the geodesic distance between two SE(3) elements: $d(T_1, T_2) = \|\log(T_1^{-1} T_2)\|$.

<details><summary>Answer</summary>

```python
def se3_distance(T1, T2):
    xi = se3_log(se3_inverse(T1) @ T2)
    return np.linalg.norm(xi)
```

This weighs rotation and translation equally; in practice, use weighted norm $\sqrt{w_t\|v\|^2 + w_r\|\omega\|^2}$.
</details>

**Problem 3:** Verify that the adjoint of the identity is $I_{6\times6}$ and that $\mathrm{Ad}(T_1 T_2) = \mathrm{Ad}(T_1)\mathrm{Ad}(T_2)$.

<details><summary>Answer</summary>

$\mathrm{Ad}(I) = \begin{pmatrix}I&0\\0&I\end{pmatrix} = I_6$. For the product: compute both $\mathrm{Ad}(T_1 T_2)$ and $\mathrm{Ad}(T_1)\mathrm{Ad}(T_2)$ numerically and verify `np.allclose()`.
</details>

---

## Expert Challenges

**Challenge 1:** Derive the SE(3) exponential map closed form from the matrix exponential Taylor series. Start from $\hat{\xi}^k$ and show the translation block sums to $V\mathbf{v}$.

<details><summary>Answer</summary>

$\hat{\xi} = \begin{pmatrix}[\omega]_\times & \mathbf{v} \\ 0 & 0\end{pmatrix}$. Compute $\hat{\xi}^k$: the rotation block gives $[\omega]_\times^k$ and the translation block gives $[\omega]_\times^{k-1}\mathbf{v}$. Summing $\sum_{k=0}^\infty \frac{1}{(k+1)!}[\omega]_\times^k = V$ by comparison with the Rodrigues series. Therefore the translation block is $V\mathbf{v}$.
</details>

**Challenge 2:** Prove that $J_l(\phi) J_l^{-1}(\phi) = I$ by substituting the closed forms and simplifying.

<details><summary>Answer</summary>

Write both in terms of $\{I, \mathbf{a}\mathbf{a}^T, [\mathbf{a}]_\times\}$ (these form a closed algebra under multiplication). The product evaluates to $I$ using: $[\mathbf{a}]_\times^2 = \mathbf{a}\mathbf{a}^T - I$, $[\mathbf{a}]_\times \mathbf{a}\mathbf{a}^T = 0$, and $\mathbf{a}\mathbf{a}^T [\mathbf{a}]_\times = 0$. Collect coefficients of each basis element and verify they sum to $(1, 0, 0)$ for $(I, \mathbf{a}\mathbf{a}^T, [\mathbf{a}]_\times)$.
</details>

**Challenge 3:** Build a mini pose-graph optimizer. Given 3 poses connected by noisy SE(3) measurements, implement one Gauss-Newton iteration using $J_r^{-1}$.

<details><summary>Answer</summary>

```python
def pose_graph_step(poses, measurements):
    """One GN step for a 3-pose chain."""
    n = len(poses)
    H = np.zeros((6*n, 6*n))
    b = np.zeros(6*n)
    for (i, j), T_meas in measurements:
        T_err = se3_inverse(T_meas) @ se3_inverse(poses[i]) @ poses[j]
        e = se3_log(T_err)
        Jr_inv = np.linalg.inv(
            np.block([[so3_left_jacobian(-e[3:]), np.zeros((3,3))],
                      [np.zeros((3,3)), so3_left_jacobian(-e[3:])]]))
        Ji = -Jr_inv
        Jj = Jr_inv
        # Accumulate into H and b (fixing pose 0)
        si, sj = 6*i, 6*j
        H[si:si+6, si:si+6] += Ji.T @ Ji
        H[sj:sj+6, sj:sj+6] += Jj.T @ Jj
        H[si:si+6, sj:sj+6] += Ji.T @ Jj
        H[sj:sj+6, si:si+6] += Jj.T @ Ji
        b[si:si+6] -= Ji.T @ e
        b[sj:sj+6] -= Jj.T @ e
    # Fix pose 0
    H[:6,:6] += 1e6 * np.eye(6)
    dx = np.linalg.solve(H, b)
    for k in range(n):
        poses[k] = poses[k] @ se3_exp(dx[6*k:6*k+6])
    return poses
```
</details>

---

## Connections
- **Back:** All of Week 12 — this review ties together SE(2/3), exp/log, Lie algebra, adjoint, and Jacobians
- **Back:** [Week 11](../week-11/day-77-week11-review.md) — SO(3) foundations underpin everything here
- **Forward:** [Week 13: Manifold Optimization](../week-13/day-85-manifold-gauss-newton.md) — applying this toolkit to real optimization problems
- **OKS:** The complete world→sensor transform pipeline and pose-graph SLAM use every concept from this week

## Self-Check
- [ ] I can chain SE(3) transforms for a multi-frame robot system
- [ ] I can compute exp, log, adjoint, and Jacobians for SE(3)
- [ ] I understand the full optimization loop: residual via log → linearize via Jacobian → update via exp
- [ ] I can extract Euler angles from SE(3) and know when gimbal lock occurs
- [ ] I can explain every entry in the formula summary table above
