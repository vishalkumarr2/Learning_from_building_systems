# Day 91: Week 13 Capstone — Mini Visual-Inertial Odometry

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Assemble everything into a working VIO: SE(3) poses, IMU preintegration, landmarks, and sliding-window optimization.*

## Navigation
- **Previous:** [Day 90: Calibration + Trajectory Interpolation](day-90-calibration-interpolation.md)
- **Next:** [Week 14 Overview](../README.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Visual-Inertial Odometry (VIO) is the fallback localization method on the OKS robot when primary sensors (sensorbar, wheel odometry) degrade — e.g., during floor transitions, sensorbar stiction events, or on slippery surfaces. This capstone builds a complete mini-VIO pipeline demonstrating exactly how the robot fuses camera and IMU data to maintain pose estimates under adverse conditions.

---

## Theory (45 min)

### 91.1 VIO Pipeline Overview

A VIO system estimates the robot's trajectory by fusing:
1. **IMU data** (200+ Hz): provides rotation and acceleration between keyframes
2. **Visual data** (10–30 Hz): provides bearing measurements to 3D landmarks

The state at keyframe $k$:

$$\mathcal{X}_k = (T_k \in \text{SE}(3), \; v_k \in \mathbb{R}^3, \; b_k \in \mathbb{R}^6)$$

where $T_k$ is the camera pose, $v_k$ is velocity, and $b_k = (b_a, b_g)$ are IMU biases.

### 91.2 Factor Types

**IMU Preintegration Factor** (between consecutive keyframes):

$$r_{\text{imu}}^{ij} = \begin{pmatrix} \text{Log}(\Delta\tilde{R}_{ij}^T R_i^T R_j) \\ R_i^T(v_j - v_i - g\Delta t) - \Delta\tilde{v}_{ij} \\ R_i^T(p_j - p_i - v_i\Delta t - \frac{1}{2}g\Delta t^2) - \Delta\tilde{p}_{ij} \end{pmatrix} \in \mathbb{R}^9$$

**Visual Landmark Factor** (bearing measurement from keyframe $k$ to landmark $\ell$):

$$r_{\text{vis}}^{k\ell} = \pi(T_k^{-1} \cdot p_\ell) - z_{k\ell} \in \mathbb{R}^2$$

where $\pi$ is the camera projection and $z_{k\ell}$ is the observed pixel coordinate.

**Bias Random Walk Factor** (between consecutive keyframes):

$$r_{\text{bias}}^{ij} = b_j - b_i \in \mathbb{R}^6$$

### 91.3 Sliding-Window Optimization

Instead of optimizing all keyframes (growing cost), keep a **sliding window** of the last $W$ keyframes:

$$\min_{\mathcal{X}_{k-W+1:k}, \, p_{\ell \in \mathcal{L}}} \sum_{\text{factors in window}} \|r\|_{\Omega}^2$$

When a keyframe exits the window, it is **marginalized** (see §91.5), producing a prior on the remaining variables.

### 91.4 Linearization and Structure

The Hessian has block structure:

$$H = \begin{pmatrix} H_{pp} & H_{pl} \\ H_{lp} & H_{ll} \end{pmatrix}$$

where $p$ indexes pose/velocity/bias states and $l$ indexes landmarks. Since $H_{ll}$ is block-diagonal ($3\times3$ blocks, one per landmark), we use the **Schur complement** to solve a reduced system in poses only:

$$\left(H_{pp} - H_{pl} H_{ll}^{-1} H_{lp}\right) \delta_p = b_p - H_{pl} H_{ll}^{-1} b_l$$

### 91.5 Marginalization

When removing old keyframe $x_0$ from the window, we convert its information into a **prior factor** on the remaining variables via the Schur complement:

$$H_{\text{prior}} = H_{rr} - H_{r0} H_{00}^{-1} H_{0r}$$

$$b_{\text{prior}} = b_r - H_{r0} H_{00}^{-1} b_0$$

This prior factor is dense over all variables that were connected to $x_0$, causing **fill-in** that limits the window size in practice.

---

## Implementation (60 min)

```python
"""Mini VIO: SE(3) + IMU preintegration + point landmarks + manifold GN."""
import numpy as np

# === Lie group helpers ===
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def so3_exp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-10: return np.eye(3) + skew(phi)
    K = skew(phi/a)
    return np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K@K)

def so3_log(R):
    ca = np.clip((np.trace(R)-1)/2, -1, 1); a = np.arccos(ca)
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

# === Camera projection ===
def project(T_wc, p_w, fx=200, fy=200, cx=160, cy=120):
    """Project world point p_w into camera at pose T_wc. Returns pixel coords."""
    p_c = se3_inv(T_wc)[:3,:3] @ (p_w - T_wc[:3,3])
    if p_c[2] < 0.1:
        return None
    u = fx * p_c[0] / p_c[2] + cx
    v = fy * p_c[1] / p_c[2] + cy
    return np.array([u, v])

# === Synthetic data generation ===
def generate_vio_data(n_frames=5, n_landmarks=15):
    np.random.seed(42)
    gravity = np.array([0, 0, -9.81])
    dt_frame = 0.1  # 10 Hz keyframes

    # True trajectory: forward motion with gentle turn
    true_poses, true_vels = [np.eye(4)], [np.array([1.0, 0, 0])]
    for i in range(n_frames - 1):
        omega = np.array([0, 0, 0.15])  # gentle yaw
        dR = so3_exp(omega * dt_frame)
        T_prev = true_poses[-1]
        T_new = np.eye(4)
        T_new[:3,:3] = T_prev[:3,:3] @ dR
        T_new[:3,3] = T_prev[:3,3] + true_vels[-1] * dt_frame
        true_poses.append(T_new)
        true_vels.append(T_new[:3,:3] @ np.array([1.0, 0, 0]))

    # Landmarks: scattered in front of trajectory
    landmarks = np.random.randn(n_landmarks, 3) * 2
    landmarks[:, 0] += 3  # in front
    landmarks[:, 2] = np.abs(landmarks[:, 2]) + 0.5

    # IMU preintegrated measurements (simplified)
    imu_factors = []
    for i in range(n_frames - 1):
        dR = true_poses[i][:3,:3].T @ true_poses[i+1][:3,:3]
        dv = true_poses[i][:3,:3].T @ (true_vels[i+1] - true_vels[i] - gravity * dt_frame)
        dp = true_poses[i][:3,:3].T @ (true_poses[i+1][:3,3] - true_poses[i][:3,3]
              - true_vels[i] * dt_frame - 0.5 * gravity * dt_frame**2)
        # Add noise
        dR_noisy = dR @ so3_exp(np.random.randn(3) * 0.005)
        dv_noisy = dv + np.random.randn(3) * 0.01
        dp_noisy = dp + np.random.randn(3) * 0.005
        imu_factors.append((dR_noisy, dv_noisy, dp_noisy, dt_frame))

    # Visual observations
    vis_obs = []
    for k in range(n_frames):
        for l_idx in range(n_landmarks):
            z = project(true_poses[k], landmarks[l_idx])
            if z is not None:
                z_noisy = z + np.random.randn(2) * 1.0  # 1 pixel noise
                vis_obs.append((k, l_idx, z_noisy))

    return true_poses, true_vels, landmarks, imu_factors, vis_obs, gravity

# === Mini VIO optimizer ===
def mini_vio(n_frames, n_landmarks, imu_factors, vis_obs, gravity,
             init_poses, init_vels, init_landmarks, max_iter=15):
    """Manifold GN for VIO: optimize poses (SE3), velocities (R3), landmarks (R3)."""
    # State layout: [pose_0(6), vel_0(3), ..., pose_{n-1}(6), vel_{n-1}(3), lm_0(3), ..., lm_{m-1}(3)]
    pose_dim, vel_dim, lm_dim = 6, 3, 3
    state_dim_per_frame = pose_dim + vel_dim
    n_pose_vars = n_frames * state_dim_per_frame
    n_lm_vars = n_landmarks * lm_dim
    N = n_pose_vars + n_lm_vars

    poses = [T.copy() for T in init_poses]
    vels = [v.copy() for v in init_vels]
    lms = [p.copy() for p in init_landmarks]

    omega_imu_R = np.eye(3) * 1e4
    omega_imu_v = np.eye(3) * 1e4
    omega_imu_p = np.eye(3) * 1e4
    omega_vis = np.eye(2) * (1.0 / 1.0**2)  # 1 pixel sigma

    costs = []
    for it in range(max_iter):
        H = np.zeros((N, N))
        b = np.zeros(N)
        cost = 0.0

        # --- IMU factors ---
        for i in range(n_frames - 1):
            dR_m, dv_m, dp_m, dt = imu_factors[i]
            Ri, Rj = poses[i][:3,:3], poses[i+1][:3,:3]
            ti, tj = poses[i][:3,3], poses[i+1][:3,3]
            vi, vj = vels[i], vels[i+1]

            e_R = so3_log(dR_m.T @ Ri.T @ Rj)
            e_v = Ri.T @ (vj - vi - gravity * dt) - dv_m
            e_p = Ri.T @ (tj - ti - vi * dt - 0.5 * gravity * dt**2) - dp_m

            cost += e_R @ omega_imu_R @ e_R + e_v @ omega_imu_v @ e_v + e_p @ omega_imu_p @ e_p

            # Indices into state vector
            idx_i = i * state_dim_per_frame
            idx_j = (i + 1) * state_dim_per_frame

            # Simplified Jacobians (approximate, small-error regime)
            # d(e_R)/d(pose_i) ~ -I3, d(e_R)/d(pose_j) ~ I3 (rotation part)
            for e_block, omega_block, r_start, r_end in [
                (e_R, omega_imu_R, 0, 3),
                (e_v, omega_imu_v, 0, 3),
                (e_p, omega_imu_p, 0, 3),
            ]:
                A, B = -np.eye(3), np.eye(3)
                # Map to correct sub-indices
                if e_block is e_R:
                    ci, cj = idx_i + 3, idx_j + 3  # rotation part of pose
                elif e_block is e_v:
                    ci, cj = idx_i + 6, idx_j + 6  # velocity
                else:
                    ci, cj = idx_i, idx_j  # translation part of pose

                H[ci:ci+3, ci:ci+3] += A.T @ omega_block @ A
                H[ci:ci+3, cj:cj+3] += A.T @ omega_block @ B
                H[cj:cj+3, ci:ci+3] += B.T @ omega_block @ A
                H[cj:cj+3, cj:cj+3] += B.T @ omega_block @ B
                b[ci:ci+3] += A.T @ omega_block @ e_block
                b[cj:cj+3] += B.T @ omega_block @ e_block

        # --- Visual factors ---
        fx, fy, cx, cy = 200, 200, 160, 120
        for (k, l_idx, z_obs) in vis_obs:
            T_wc = poses[k]
            p_w = lms[l_idx]
            p_c = se3_inv(T_wc)[:3,:3] @ (p_w - T_wc[:3,3])
            if p_c[2] < 0.1:
                continue
            z_pred = np.array([fx*p_c[0]/p_c[2]+cx, fy*p_c[1]/p_c[2]+cy])
            e_vis = z_pred - z_obs
            cost += e_vis @ omega_vis @ e_vis

            # Jacobian w.r.t. landmark (in world frame, projected to pixel)
            dz_dpc = np.array([[fx/p_c[2], 0, -fx*p_c[0]/p_c[2]**2],
                               [0, fy/p_c[2], -fy*p_c[1]/p_c[2]**2]])
            R_cw = se3_inv(T_wc)[:3,:3]
            J_lm = dz_dpc @ R_cw  # 2x3

            lm_idx = n_pose_vars + l_idx * lm_dim
            H[lm_idx:lm_idx+3, lm_idx:lm_idx+3] += J_lm.T @ omega_vis @ J_lm
            b[lm_idx:lm_idx+3] += J_lm.T @ omega_vis @ e_vis

        costs.append(cost)

        # Anchor first pose and velocity
        for d in range(state_dim_per_frame):
            H[d, :] = 0; H[:, d] = 0; H[d, d] = 1e8
            b[d] = 0

        # Damping
        H += 1e-3 * np.eye(N)
        delta = np.linalg.solve(H, -b)

        if np.linalg.norm(delta) < 1e-8:
            break

        # Apply updates
        for k in range(n_frames):
            idx = k * state_dim_per_frame
            poses[k] = poses[k] @ se3_exp(delta[idx:idx+6])
            vels[k] = vels[k] + delta[idx+6:idx+9]
        for l_idx in range(n_landmarks):
            idx = n_pose_vars + l_idx * lm_dim
            lms[l_idx] = lms[l_idx] + delta[idx:idx+3]

    return poses, vels, lms, costs

# === Run ===
n_f, n_l = 5, 15
true_poses, true_vels, true_lms, imu_facs, vis_obs, grav = generate_vio_data(n_f, n_l)

# Noisy initialization
init_poses = [T @ se3_exp(np.random.randn(6)*0.05) for T in true_poses]
init_poses[0] = true_poses[0].copy()  # anchor
init_vels = [v + np.random.randn(3)*0.1 for v in true_vels]
init_vels[0] = true_vels[0].copy()
init_lms = [p + np.random.randn(3)*0.3 for p in true_lms]

opt_poses, opt_vels, opt_lms, costs = mini_vio(
    n_f, n_l, imu_facs, vis_obs, grav, init_poses, init_vels, init_lms)

print(f"VIO optimization: {len(costs)} iterations")
print(f"Cost: {costs[0]:.2f} -> {costs[-1]:.4f}")
print("\nPose errors (6D norm):")
for k in range(n_f):
    err = np.linalg.norm(se3_log(se3_inv(true_poses[k]) @ opt_poses[k]))
    print(f"  Frame {k}: {err:.5f}")
print(f"\nLandmark RMSE: {np.sqrt(np.mean([(np.linalg.norm(opt_lms[i]-true_lms[i]))**2 for i in range(n_l)])):.4f} m")
```

---

## Practice Problems (45 min)

### Problem 1: Factor Count Analysis
For a 5-frame VIO window with 15 landmarks each seen in 3 frames, and 4 IMU factors, how many residual components are there? What is the Hessian dimension?

<details><summary>Answer</summary>

IMU: 4 factors × 9 components = 36. Visual: 15 × 3 × 2 = 90. Bias: 4 × 6 = 24 (not in this simplified version). Total residuals: 126+. State: 5 frames × 9 (6 pose + 3 vel) + 15 landmarks × 3 = 45 + 45 = 90. Hessian: 90 × 90.
</details>

### Problem 2: Schur Complement Application
Apply the Schur complement to the VIO Hessian to eliminate landmarks. What is the reduced system size? How much faster is solving it?

<details><summary>Answer</summary>

Pose/velocity states: 5 × 9 = 45. Landmark states: 15 × 3 = 45. Reduced system: 45 × 45 (from 90 × 90). Since $H_{ll}$ is block-diagonal (fifteen 3×3 blocks), its inverse costs $O(15 \times 27) = O(405)$. Reduced system solve: $O(45^3) \approx 91K$ vs full $O(90^3) \approx 729K$ — about 8× speedup.
</details>

### Problem 3: Sensitivity Analysis
Remove all visual factors and run with IMU only. Then remove IMU factors and run with vision only. Compare drift in each case. What does each sensor contribute?

<details><summary>Answer</summary>

IMU only: drift grows quadratically in position (integration of integration of noise), good short-term relative accuracy, poor absolute. Vision only: scale-ambiguous without IMU, good relative pose from feature matches but scale drift over time. Together: IMU provides scale and high-rate dynamics; vision provides absolute corrections and loop consistency.
</details>

---

## Expert Challenges

### Challenge 1: Marginalization (Schur on Manifold)
Implement marginalization of the oldest keyframe. When frame 0 is removed, create a prior factor on frames 1–4 using the Schur complement. Run the windowed optimizer and verify the prior keeps the solution consistent.

<details><summary>Answer</summary>

Extract the block of $H$ and $b$ connecting frame 0 to the rest. Compute $H_{\text{prior}} = H_{rr} - H_{r0} H_{00}^{-1} H_{0r}$ and $b_{\text{prior}} = b_r - H_{r0} H_{00}^{-1} b_0$. Add this as a constant quadratic factor in subsequent windows. Key subtlety: the linearization point of the prior is fixed — FEJ (First Estimates Jacobians) must be used for the prior Jacobians to maintain consistency.
</details>

### Challenge 2: Fixed-Lag vs Sliding-Window
Compare two marginalization strategies: (a) fixed-lag smoother removing the oldest frame each step, (b) sliding window that keeps every 3rd frame for long-range constraints. Measure trajectory drift over 20 frames.

<details><summary>Answer</summary>

Fixed-lag: constant-size window, bounded compute, but loses long-range information. Sliding with skip: keeps a sparse set of older frames, providing longer baseline constraints at the cost of a larger (but still bounded) system. Skip-frame window typically reduces drift by 30–50% over 20 frames because it maintains observability of scale and orientation over longer intervals.
</details>

### Challenge 3: Observability Analysis
Show that a VIO system with constant velocity has unobservable yaw. How many unobservable directions does a stationary VIO system have?

<details><summary>Answer</summary>

For constant velocity (no excitation): global yaw is unobservable — the system cannot distinguish between a yaw rotation and a corresponding change in velocity direction. The observability matrix has a 4D null space: 3 global translations + 1 global yaw. A stationary system loses additional directions: velocity becomes unobservable (3 more), giving 7 unobservable directions total. Sufficient excitation (acceleration changes) is needed to make the system fully observable.
</details>

---

## Connections
- **Prerequisites**: Day 85 (manifold GN), Day 86–87 (pose graphs), Day 88 (factor graphs), Day 89 (IMU preintegration), Day 90 (calibration)
- **Forward**: this capstone synthesizes all Week 12–13 material into a complete system
- **OKS**: VIO is the backup localization system when sensorbar data is unreliable (stiction, floor transitions, slippery surfaces)

## Self-Check
- [ ] I can describe the VIO pipeline: states, factors, and optimization
- [ ] I can implement a joint optimizer for SE(3) poses + R³ landmarks + IMU factors
- [ ] I understand why the Schur complement is essential for VIO efficiency
- [ ] I can explain marginalization and why it creates a prior for the sliding window
- [ ] I can discuss observability limitations of VIO (unobservable yaw, scale)
