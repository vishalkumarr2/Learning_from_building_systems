# Day 90: Calibration + Trajectory Interpolation

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Solve AX = XB for sensor calibration and interpolate smooth trajectories on SE(3).*

## Navigation
- **Previous:** [Day 89: IMU Preintegration](day-89-imu-preintegration.md)
- **Next:** [Day 91: Week 13 Capstone — Mini VIO](day-91-week13-capstone.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
OKS robots mount multiple sensors (cameras, lidars, IMUs) that must be precisely calibrated relative to each other. Hand-eye calibration (AX = XB) determines the extrinsic transform between sensor pairs. Smooth trajectory interpolation on SE(3) is needed for motion planning and for interpolating between discrete pose estimates when fusing asynchronous sensors.

---

## Theory (45 min)

### 90.1 Hand-Eye Calibration: AX = XB

Given a sensor (eye) rigidly mounted on a robot hand, we observe:
- $A_k \in \text{SE}(3)$: relative hand motion between pose $k$ and $k+1$
- $B_k \in \text{SE}(3)$: relative sensor motion between pose $k$ and $k+1$

The unknown $X \in \text{SE}(3)$ is the hand-to-eye transform satisfying:

$$A_k X = X B_k \quad \text{for all } k = 1, \ldots, N$$

This decomposes into a rotation part and a translation part:

$$R_A R_X = R_X R_B \quad \Rightarrow \quad R_X = ?$$
$$(R_A - I)t_X = R_X t_B - t_A$$

### 90.2 Tsai-Lenz Method

**Step 1 — Solve for rotation** using the axis-angle decomposition:

For each pair, the rotation constraint $R_A R_X = R_X R_B$ implies that the rotation axes satisfy:

$$(\hat{a}_k + \hat{b}_k) \times r = \hat{a}_k - \hat{b}_k$$

where $\hat{a}_k = \text{axis}(A_k) \cdot \tan(\theta_A/2)$ and similarly for $\hat{b}_k$. This gives a linear system for the modified rotation axis $r$, from which $R_X$ is recovered.

**Step 2 — Solve for translation**: with $R_X$ known, each pair gives:

$$(R_{A_k} - I) t_X = R_X t_{B_k} - t_{A_k}$$

Stack $N$ equations into an overdetermined linear system and solve via least squares.

### 90.3 SLERP for SO(3) Interpolation

**Spherical Linear Interpolation** between rotations $R_0, R_1$ at parameter $t \in [0,1]$:

$$\text{SLERP}(R_0, R_1, t) = R_0 \cdot \text{Exp}\!\left(t \cdot \text{Log}(R_0^T R_1)\right)$$

Properties:
- Constant angular velocity (geodesic on SO(3))
- $\text{SLERP}(R_0, R_1, 0) = R_0$, $\text{SLERP}(R_0, R_1, 1) = R_1$
- Shortest-path rotation (assuming $\|\text{Log}(R_0^T R_1)\| < \pi$)

### 90.4 SE(3) Trajectory Interpolation

For full rigid-body trajectories, combine rotation and translation interpolation:

**Method 1 — Decoupled**: SLERP for rotation + linear interpolation for translation. Simple but doesn't produce a geodesic on SE(3).

**Method 2 — Coupled (Geodesic)**: 

$$T(t) = T_0 \cdot \text{Exp}_{SE(3)}\!\left(t \cdot \text{Log}_{SE(3)}(T_0^{-1} T_1)\right)$$

This is the geodesic on SE(3) — constant body-frame velocity (screw motion).

### 90.5 SQUAD: Spherical Quadratic Interpolation

For smooth interpolation through multiple waypoints $\{R_k\}$, SQUAD provides $C^1$ continuity:

$$\text{SQUAD}(R_k, R_{k+1}, S_k, S_{k+1}, t) = \text{SLERP}(\text{SLERP}(R_k, R_{k+1}, t), \text{SLERP}(S_k, S_{k+1}, t), 2t(1-t))$$

where the control rotations are:

$$S_k = R_k \cdot \text{Exp}\!\left(-\tfrac{1}{4}\left[\text{Log}(R_k^T R_{k+1}) + \text{Log}(R_k^T R_{k-1})\right]\right)$$

This is the rotation analogue of cubic Hermite interpolation.

---

## Implementation (60 min)

```python
"""Hand-Eye Calibration (Tsai-Lenz) + SE(3) Interpolation."""
import numpy as np

def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def so3_exp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-10: return np.eye(3) + skew(phi)
    K = skew(phi/a)
    return np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K@K)

def so3_log(R):
    ca = np.clip((np.trace(R)-1)/2, -1, 1)
    a = np.arccos(ca)
    if a < 1e-10: return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/2
    return a/(2*np.sin(a))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def se3_exp(xi):
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi); a = np.linalg.norm(phi)
    if a < 1e-10: V = np.eye(3)
    else:
        K = skew(phi/a)
        V = np.eye(3)+(1-np.cos(a))/a**2*K+(a-np.sin(a))/a**3*(K@K)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = V @ rho
    return T

def se3_log(T):
    R, t = T[:3,:3], T[:3,3]; phi = so3_log(R); a = np.linalg.norm(phi)
    if a < 1e-10: return np.concatenate([t, phi])
    K = skew(phi)
    V_inv = np.eye(3) - 0.5*K + (1-a/(2*np.tan(a/2)))/a**2*(K@K)
    return np.concatenate([V_inv @ t, phi])

def se3_inv(T):
    Ti = np.eye(4); Ti[:3,:3] = T[:3,:3].T; Ti[:3,3] = -T[:3,:3].T @ T[:3,3]
    return Ti

# === Hand-Eye Calibration (Tsai-Lenz) ===
def hand_eye_calibration(A_list, B_list):
    """Solve AX = XB given lists of relative motions A_k, B_k in SE(3)."""
    n = len(A_list)

    # Step 1: Solve for rotation via modified Rodrigues
    M = np.zeros((3 * n, 3))
    rhs = np.zeros(3 * n)
    lhs_rows = []
    rhs_rows = []

    for k in range(n):
        Ra = A_list[k][:3, :3]; Rb = B_list[k][:3, :3]
        alpha = so3_log(Ra); beta = so3_log(Rb)
        a_norm = np.linalg.norm(alpha); b_norm = np.linalg.norm(beta)
        if a_norm < 1e-10 or b_norm < 1e-10:
            continue
        a_prime = alpha / a_norm * np.tan(a_norm / 2)
        b_prime = beta / b_norm * np.tan(b_norm / 2)
        lhs_rows.append(skew(a_prime + b_prime))
        rhs_rows.append(b_prime - a_prime)

    if len(lhs_rows) < 2:
        raise ValueError("Need at least 2 non-trivial motions")

    A_mat = np.vstack(lhs_rows)
    b_vec = np.concatenate(rhs_rows)
    r, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    # Recover rotation from modified Rodrigues
    Rx = (1 - r @ r) * np.eye(3) + 2 * np.outer(r, r) + 2 * skew(r)
    Rx /= (1 + r @ r)

    # Step 2: Solve for translation
    C_rows, d_rows = [], []
    for k in range(n):
        Ra = A_list[k][:3, :3]; ta = A_list[k][:3, 3]; tb = B_list[k][:3, 3]
        C_rows.append(Ra - np.eye(3))
        d_rows.append(Rx @ tb - ta)

    C = np.vstack(C_rows)
    d = np.concatenate(d_rows)
    tx, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx
    return X

# === SE(3) Interpolation ===
def slerp_so3(R0, R1, t):
    """SLERP between two rotations."""
    delta = so3_log(R0.T @ R1)
    return R0 @ so3_exp(t * delta)

def interp_se3_decoupled(T0, T1, t):
    """Decoupled: SLERP rotation + lerp translation."""
    R = slerp_so3(T0[:3,:3], T1[:3,:3], t)
    p = (1 - t) * T0[:3, 3] + t * T1[:3, 3]
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = p
    return T

def interp_se3_geodesic(T0, T1, t):
    """Geodesic interpolation on SE(3)."""
    delta = se3_log(se3_inv(T0) @ T1)
    return T0 @ se3_exp(t * delta)

# --- Demo: Hand-Eye Calibration ---
np.random.seed(42)
X_true = se3_exp(np.array([0.1, -0.05, 0.2, 0.3, -0.1, 0.15]))
n_motions = 10

A_list, B_list = [], []
for _ in range(n_motions):
    A = se3_exp(np.random.randn(6) * 0.5)
    B = se3_inv(X_true) @ A @ X_true  # AX = XB => B = X^{-1}AX
    A_noisy = A @ se3_exp(np.random.randn(6) * 0.005)
    B_noisy = B @ se3_exp(np.random.randn(6) * 0.005)
    A_list.append(A_noisy)
    B_list.append(B_noisy)

X_est = hand_eye_calibration(A_list, B_list)
err = np.linalg.norm(se3_log(se3_inv(X_true) @ X_est))
print(f"Hand-eye calibration error: {err:.6f}")

# --- Demo: SE(3) Interpolation ---
T0 = np.eye(4)
T1 = se3_exp(np.array([2.0, 1.0, 0.5, 0.0, 0.0, np.pi/3]))
ts = np.linspace(0, 1, 5)

print("\nSE(3) interpolation comparison:")
for t in ts:
    Td = interp_se3_decoupled(T0, T1, t)
    Tg = interp_se3_geodesic(T0, T1, t)
    diff = np.linalg.norm(se3_log(se3_inv(Td) @ Tg))
    print(f"  t={t:.2f}: decoupled-geodesic diff = {diff:.4f}")
```

---

## Practice Problems (45 min)

### Problem 1: Minimum Motions
What is the minimum number of motion pairs $(A_k, B_k)$ needed to solve AX = XB? Why?

<details><summary>Answer</summary>

Minimum 2 motions with non-parallel rotation axes. The rotation part requires at least 2 equations with linearly independent $\text{skew}(a_k + b_k)$ rows to determine the 3 unknowns. A single motion constrains only the component of $R_X$ along the rotation axis. Translation requires 2 motions with $R_{A_k} - I$ having rank ≥ 2 in aggregate.
</details>

### Problem 2: SLERP Verification
Show that $\text{SLERP}(R_0, R_1, t)$ produces constant angular velocity by computing $\omega(t) = \text{Log}(\text{SLERP}(t)^T \cdot \text{SLERP}(t + \epsilon)) / \epsilon$ for several values of $t$.

<details><summary>Answer</summary>

$\omega(t) = \text{Log}(R_0^T R_1)$ for all $t$ (constant). This is because $\text{SLERP}(t) = R_0 \text{Exp}(t\phi)$ where $\phi = \text{Log}(R_0^T R_1)$, so $\text{SLERP}(t)^T \text{SLERP}(t+\epsilon) = \text{Exp}(-t\phi) R_0^T R_0 \text{Exp}((t+\epsilon)\phi) = \text{Exp}(\epsilon\phi)$. Angular velocity $= \phi / 1 = \text{const}$.
</details>

### Problem 3: Geodesic vs Decoupled
For what configurations is the difference between geodesic and decoupled SE(3) interpolation largest? Design a test case maximizing the difference.

<details><summary>Answer</summary>

Largest difference when rotation and translation are strongly coupled — e.g., a large rotation (close to $\pi$) combined with a large translation perpendicular to the rotation axis. The geodesic traces a screw motion (helical path) while the decoupled version translates linearly and rotates independently, producing different midpoints. Maximum divergence at $t = 0.5$ for $\theta \approx \pi$.
</details>

---

## Expert Challenges

### Challenge 1: SQUAD Implementation
Implement full SQUAD interpolation through 5 waypoints on SO(3). Compare the angular velocity profile with piecewise SLERP. Does SQUAD achieve $C^1$ continuity?

<details><summary>Answer</summary>

```python
def squad(R_prev, R_k, R_next, R_kp1, t):
    phi_fwd = so3_log(R_k.T @ R_kp1)
    phi_bwd = so3_log(R_k.T @ R_prev)
    S_k = R_k @ so3_exp(-0.25 * (phi_fwd + phi_bwd))
    phi_fwd2 = so3_log(R_kp1.T @ R_next) if R_next is not None else np.zeros(3)
    phi_bwd2 = so3_log(R_kp1.T @ R_k)
    S_kp1 = R_kp1 @ so3_exp(-0.25 * (phi_fwd2 + phi_bwd2))
    return slerp_so3(slerp_so3(R_k, R_kp1, t), slerp_so3(S_k, S_kp1, t), 2*t*(1-t))
```
SQUAD achieves angular velocity continuity at waypoints, unlike piecewise SLERP which has velocity discontinuities.
</details>

### Challenge 2: Nonlinear Hand-Eye
Formulate AX = XB as a nonlinear least-squares problem on SE(3) and solve with manifold GN. Compare accuracy with Tsai-Lenz for high-noise data.

<details><summary>Answer</summary>

Residual for pair $k$: $r_k = \text{Log}(X^{-1} A_k^{-1} X B_k) \in \mathbb{R}^6$. Jacobian w.r.t. perturbation $\delta X$: $\frac{\partial r_k}{\partial \delta} \approx -\text{Ad}(B_k^{-1}) + I_6$. Manifold GN converges to a better solution than Tsai-Lenz for $\sigma > 0.01$ rad because it optimizes the full SE(3) cost jointly.
</details>

### Challenge 3: Cumulative B-Spline on SE(3)
Implement a cubic cumulative B-spline on SE(3) through $n$ control poses. Verify $C^2$ continuity by computing angular acceleration at knot points.

<details><summary>Answer</summary>

A cumulative B-spline on SE(3) uses the formula: $T(t) = T_0 \prod_{j=1}^{p} \text{Exp}(\tilde{B}_j(t) \cdot \Omega_j)$ where $\Omega_j = \text{Log}(T_{j-1}^{-1}T_j)$ and $\tilde{B}_j(t)$ are the cumulative basis functions. For cubic ($p=3$), this gives $C^2$ continuity in body-frame velocity and acceleration. The key advantage over SQUAD: full $C^2$ smoothness, not just $C^1$.
</details>

---

## Connections
- **Prerequisites**: Day 79 (SE(3) basics), Day 80 (Exp/Log), Day 85 (manifold optimization)
- **Forward**: Day 91 capstone uses calibrated sensor transforms and smooth trajectories
- **OKS**: camera-lidar extrinsic calibration, smooth trajectory generation for motion planning

## Self-Check
- [ ] I can formulate and solve AX = XB using the Tsai-Lenz method
- [ ] I can implement SLERP on SO(3) and verify constant angular velocity
- [ ] I understand the difference between geodesic and decoupled SE(3) interpolation
- [ ] I can explain when SQUAD provides smoother interpolation than piecewise SLERP
