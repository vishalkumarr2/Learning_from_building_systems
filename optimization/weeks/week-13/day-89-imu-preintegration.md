# Day 89: IMU Preintegration

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Preintegrate hundreds of IMU samples into a single factor — the key to efficient visual-inertial odometry.*

## Navigation
- **Previous:** [Day 88: Ceres & GTSAM for Manifold Optimization](day-88-ceres-gtsam-manifold.md)
- **Next:** [Day 90: Calibration + Trajectory Interpolation](day-90-calibration-interpolation.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The OKS robot fuses wheel odometry with IMU data for state estimation. Between visual keyframes (or sensorbar readings), the IMU fires at 200+ Hz. Preintegration compresses all those samples into a single constraint between keyframe states, avoiding re-integration when the bias estimate changes — a crucial efficiency for real-time operation.

---

## Theory (45 min)

### 89.1 IMU Measurement Model

An IMU provides two measurements at high rate ($\ge 200$ Hz):

**Accelerometer** (specific force in body frame):

$$\tilde{a}_t = R_t^T(a_t - g) + b_a + \eta_a$$

**Gyroscope** (angular velocity in body frame):

$$\tilde{\omega}_t = \omega_t + b_g + \eta_g$$

where $R_t \in \text{SO}(3)$ is body orientation, $g$ is gravity, $b_a, b_g$ are slowly-varying biases, and $\eta_a, \eta_g$ are white noise.

### 89.2 Direct Integration (Naive Approach)

Between keyframes $i$ and $j$ (times $t_i$ to $t_j$), integrate:

$$R_j = R_i \prod_{k=i}^{j-1} \text{Exp}\!\left((\tilde{\omega}_k - b_g)\Delta t\right)$$

$$v_j = v_i + g \Delta t_{ij} + \sum_{k=i}^{j-1} R_k (\tilde{a}_k - b_a) \Delta t$$

$$p_j = p_i + v_i \Delta t_{ij} + \tfrac{1}{2}g\Delta t_{ij}^2 + \sum_{k=i}^{j-1}\left[R_k(\tilde{a}_k - b_a)\tfrac{\Delta t^2}{2}\right]$$

**Problem**: when the optimizer updates $R_i, v_i, b_a, b_g$, we must re-integrate everything. With 200 Hz IMU and 10 Hz keyframes, that's 20 re-integrations per iteration per edge.

### 89.3 Preintegrated Measurements (Forster et al., 2017)

The key insight: factor out the **initial state** by defining quantities relative to frame $i$:

$$\Delta R_{ij} = R_i^T R_j = \prod_{k=i}^{j-1} \text{Exp}\!\left((\tilde{\omega}_k - b_g)\Delta t\right)$$

$$\Delta v_{ij} = R_i^T(v_j - v_i - g\Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta R_{ik}(\tilde{a}_k - b_a)\Delta t$$

$$\Delta p_{ij} = R_i^T(p_j - p_i - v_i\Delta t_{ij} - \tfrac{1}{2}g\Delta t_{ij}^2) = \sum_{k=i}^{j-1}\left[\Delta v_{ik}\Delta t + \Delta R_{ik}(\tilde{a}_k - b_a)\tfrac{\Delta t^2}{2}\right]$$

These depend only on IMU measurements and biases — **not** on $R_i, v_i, p_i$. They can be precomputed once.

### 89.4 Bias Correction via First-Order Update

When the bias estimate changes from $\bar{b}$ to $\bar{b} + \delta b$, instead of re-integrating, use a first-order correction:

$$\Delta R_{ij}(\bar{b}_g + \delta b_g) \approx \Delta R_{ij}(\bar{b}_g) \cdot \text{Exp}\!\left(\frac{\partial \Delta R_{ij}}{\partial b_g}\delta b_g\right)$$

$$\Delta v_{ij}(\bar{b} + \delta b) \approx \Delta v_{ij}(\bar{b}) + \frac{\partial \Delta v_{ij}}{\partial b_g}\delta b_g + \frac{\partial \Delta v_{ij}}{\partial b_a}\delta b_a$$

$$\Delta p_{ij}(\bar{b} + \delta b) \approx \Delta p_{ij}(\bar{b}) + \frac{\partial \Delta p_{ij}}{\partial b_g}\delta b_g + \frac{\partial \Delta p_{ij}}{\partial b_a}\delta b_a$$

The Jacobians $\frac{\partial \Delta R}{\partial b_g}$, etc., are accumulated during preintegration.

### 89.5 Preintegration Factor

The preintegration factor connects states $\mathcal{X}_i = (R_i, v_i, p_i, b_i)$ and $\mathcal{X}_j$. The residual:

$$r_{ij} = \begin{pmatrix} \text{Log}(\Delta\tilde{R}_{ij}^T \cdot R_i^T R_j) \\ R_i^T(v_j - v_i - g\Delta t) - \Delta\tilde{v}_{ij} \\ R_i^T(p_j - p_i - v_i\Delta t - \tfrac{1}{2}g\Delta t^2) - \Delta\tilde{p}_{ij} \end{pmatrix} \in \mathbb{R}^9$$

This 9D residual is a single factor in the graph — independent of the number of IMU samples.

---

## Implementation (60 min)

```python
"""IMU Preintegration (Forster et al., 2017)."""
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

def right_jacobian_so3(phi):
    a = np.linalg.norm(phi)
    if a < 1e-10: return np.eye(3) - 0.5*skew(phi)
    K = skew(phi/a)
    return np.eye(3) - (1-np.cos(a))/a**2*K + (a-np.sin(a))/a**3*(K@K)

class IMUPreintegrator:
    """Preintegrate IMU measurements between two keyframes."""

    def __init__(self, bias_a=np.zeros(3), bias_g=np.zeros(3)):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.dt_total = 0.0
        self.bias_a = bias_a.copy()
        self.bias_g = bias_g.copy()
        # Jacobians for bias correction
        self.dR_dbg = np.zeros((3, 3))
        self.dv_dba = np.zeros((3, 3))
        self.dv_dbg = np.zeros((3, 3))
        self.dp_dba = np.zeros((3, 3))
        self.dp_dbg = np.zeros((3, 3))

    def integrate(self, accel, gyro, dt):
        """Add one IMU measurement."""
        a_unbiased = accel - self.bias_a
        w_unbiased = gyro - self.bias_g

        dR_k = so3_exp(w_unbiased * dt)
        Jr_k = right_jacobian_so3(w_unbiased * dt)

        # Update bias Jacobians (before updating delta quantities)
        self.dp_dba += self.dv_dba * dt - 0.5 * self.delta_R @ np.eye(3) * dt**2
        self.dp_dbg += self.dv_dbg * dt - 0.5 * self.delta_R @ skew(a_unbiased) @ self.dR_dbg * dt**2

        self.dv_dba += -self.delta_R * dt
        self.dv_dbg += -self.delta_R @ skew(a_unbiased) @ self.dR_dbg * dt

        # Update preintegrated quantities
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ a_unbiased * dt**2
        self.delta_v += self.delta_R @ a_unbiased * dt

        # Update rotation bias Jacobian before rotation
        self.dR_dbg = dR_k.T @ self.dR_dbg - Jr_k * dt

        self.delta_R = self.delta_R @ dR_k
        self.dt_total += dt

    def correct_bias(self, new_ba, new_bg):
        """First-order bias correction without re-integration."""
        dba = new_ba - self.bias_a
        dbg = new_bg - self.bias_g
        delta_R_corrected = self.delta_R @ so3_exp(self.dR_dbg @ dbg)
        delta_v_corrected = self.delta_v + self.dv_dba @ dba + self.dv_dbg @ dbg
        delta_p_corrected = self.delta_p + self.dp_dba @ dba + self.dp_dbg @ dbg
        return delta_R_corrected, delta_v_corrected, delta_p_corrected

    def residual(self, Ri, vi, pi, Rj, vj, pj, gravity):
        """Compute 9D preintegration residual."""
        dt = self.dt_total
        r_R = so3_log(self.delta_R.T @ Ri.T @ Rj)
        r_v = Ri.T @ (vj - vi - gravity * dt) - self.delta_v
        r_p = Ri.T @ (pj - pi - vi * dt - 0.5 * gravity * dt**2) - self.delta_p
        return np.concatenate([r_R, r_v, r_p])

# --- Synthetic IMU sequence ---
np.random.seed(42)
gravity = np.array([0, 0, -9.81])
dt_imu = 0.005  # 200 Hz
n_samples = 100  # 0.5 seconds between keyframes

# True trajectory: constant velocity + rotation
R_true = np.eye(3)
v_true = np.array([1.0, 0.0, 0.0])  # 1 m/s forward
omega_true = np.array([0.0, 0.0, 0.1])  # yaw rate
bias_a_true = np.array([0.02, -0.01, 0.03])
bias_g_true = np.array([0.001, -0.002, 0.001])
sigma_a, sigma_g = 0.1, 0.01

# Preintegrate
preint = IMUPreintegrator(bias_a=bias_a_true, bias_g=bias_g_true)
R_k = R_true.copy()
for k in range(n_samples):
    a_world = np.array([0, 0, 0])  # no acceleration in world frame
    a_body = R_k.T @ (a_world - gravity) + bias_a_true + np.random.randn(3)*sigma_a
    g_body = omega_true + bias_g_true + np.random.randn(3)*sigma_g
    preint.integrate(a_body, g_body, dt_imu)
    R_k = R_k @ so3_exp(omega_true * dt_imu)

# Compute true final state
dt_total = n_samples * dt_imu
R_final = R_true @ so3_exp(omega_true * dt_total)
v_final = v_true  # constant velocity, no acceleration
p_final = v_true * dt_total

# Residual should be near zero
r = preint.residual(R_true, v_true, np.zeros(3), R_final, v_final, p_final, gravity)
print(f"Preintegration residual norm: {np.linalg.norm(r):.6f}")
print(f"  Rotation residual:    {np.linalg.norm(r[:3]):.6f}")
print(f"  Velocity residual:    {np.linalg.norm(r[3:6]):.6f}")
print(f"  Position residual:    {np.linalg.norm(r[6:9]):.6f}")
print(f"Total preintegrated time: {preint.dt_total:.3f}s ({n_samples} samples)")
```

---

## Practice Problems (45 min)

### Problem 1: Preintegration vs Direct Integration
Integrate the same IMU sequence directly (propagating full state) and via preintegration. Compare final states. When does preintegration save computation?

<details><summary>Answer</summary>

Both produce identical results for a fixed bias. Preintegration saves computation during optimization: if $R_i, v_i$ change, direct integration must re-process all $N$ IMU samples. Preintegration only needs to recompute the 9D residual from precomputed $\Delta R, \Delta v, \Delta p$ — $O(1)$ vs $O(N)$ per optimization iteration.
</details>

### Problem 2: Bias Correction Test
Perturb the bias by $\delta b_g = [0.01, 0, 0]^T$. Compare the first-order corrected $\Delta R$ with full re-integration at the new bias. What is the approximation error?

<details><summary>Answer</summary>

For small bias perturbations ($\|\delta b\| < 0.05$ rad/s), the first-order correction matches re-integration to within $\sim 10^{-4}$ rad. Error grows quadratically with $\|\delta b\|$. When bias changes significantly (e.g., after a long period without updates), re-integration may be necessary.
</details>

### Problem 3: Noise Propagation
Derive the covariance of the preintegrated measurements $(\Delta R, \Delta v, \Delta p)$ given noise parameters $\sigma_a, \sigma_g$. How does uncertainty grow with the number of samples?

<details><summary>Answer</summary>

Covariance propagates as $\Sigma_{k+1} = F_k \Sigma_k F_k^T + G_k Q G_k^T$ where $F_k$ is the discrete-time state transition and $Q = \text{diag}(\sigma_g^2 I_3, \sigma_a^2 I_3)\Delta t$. Rotation covariance grows linearly with $N$; velocity covariance grows as $O(N)$; position covariance grows as $O(N^2)$. This is why long preintegration intervals have high uncertainty.
</details>

---

## Expert Challenges

### Challenge 1: Second-Order Bias Correction
Implement second-order bias correction by including the Hessian term. When does this significantly outperform first-order?

<details><summary>Answer</summary>

Second-order: $\Delta R \approx \Delta R_0 \cdot \text{Exp}(J_1 \delta b + \frac{1}{2} J_2 [\delta b \otimes \delta b])$. Significant improvement when $\|\delta b\| > 0.05$ rad/s or preintegration interval > 1s. Computing $J_2$ requires accumulating second-order Jacobians during preintegration, roughly tripling storage.
</details>

### Challenge 2: Gravity Direction Estimation
Treat gravity direction as an optimization variable $g = g_0 \cdot R_g \hat{z}$ where $R_g \in \text{SO}(3)$ with 2 DOF (gravity magnitude is known). How does this change the preintegration factor?

<details><summary>Answer</summary>

Parameterize gravity on $S^2$ (the 2-sphere) with 2D tangent-space perturbation. The preintegration residual gains additional Jacobians w.r.t. $\delta g$. The rotation component $r_R$ is unchanged; velocity and position residuals gain terms proportional to $\Delta t$ and $\Delta t^2$ times the gravity perturbation. This is used in VIO initialization when gravity direction is unknown.
</details>

### Challenge 3: Discrete-Time Integration Schemes
Compare Euler, midpoint, and RK4 integration for the preintegration step. How does integration order affect drift over 1 second at 200 Hz?

<details><summary>Answer</summary>

At 200 Hz ($\Delta t = 5$ms) with moderate dynamics: Euler error $\sim O(\Delta t) \sim 0.5\%$, midpoint $\sim O(\Delta t^2) \sim 0.0025\%$, RK4 $\sim O(\Delta t^4) \sim 10^{-8}\%$. For typical robotics, Euler is sufficient at $\ge 200$ Hz. Midpoint is standard in GTSAM. RK4 is overkill except for very high dynamics (drone acrobatics).
</details>

---

## Connections
- **Prerequisites**: Day 80 (Exp/Log on SO(3)), Day 85 (manifold optimization), Day 88 (factor graphs)
- **Forward**: Day 91 capstone combines IMU preintegration with visual factors in a VIO system
- **OKS**: IMU preintegration enables efficient wheel-IMU fusion in the navigation estimator

## Self-Check
- [ ] I can explain why preintegration avoids recomputation during optimization
- [ ] I can implement the preintegration equations for $\Delta R$, $\Delta v$, $\Delta p$
- [ ] I understand first-order bias correction and when it breaks down
- [ ] I can compute the 9D preintegration residual connecting two keyframe states
