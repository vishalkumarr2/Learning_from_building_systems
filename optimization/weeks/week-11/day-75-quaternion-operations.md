# Day 75: Quaternion Interpolation and Averaging

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *Straight lines on the sphere — how to move smoothly between orientations.*

## Navigation
- **Previous:** [Day 74: Quaternions for Rotation](day-74-quaternions.md)
- **Next:** [Day 76: Rotation Representations Compared](day-76-rotation-representations.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Smooth trajectory generation for the OKS AMR requires interpolating between waypoint orientations. Naive linear interpolation of Euler angles or quaternion components produces jerky, non-uniform motion. SLERP gives constant angular velocity — essential for smooth `cmd_vel` profiles. Quaternion averaging appears in multi-sensor fusion: combining IMU and visual odometry orientation estimates.

---

## Theory (45 min)

### 75.1 Why Linear Interpolation Fails

Given quaternions $q_0$ and $q_1$, the **linear interpolation** (LERP):

$$q(t) = (1-t)\,q_0 + t\,q_1$$

has two problems:
1. $\|q(t)\| \neq 1$ in general — the result is not a unit quaternion
2. The angular velocity is non-uniform (speeds up near endpoints)

Normalizing (NLERP: $q(t) / \|q(t)\|$) fixes problem 1 but not problem 2.

### 75.2 SLERP — Spherical Linear Interpolation

**SLERP** traces a great-circle arc on $S^3$:

$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,q_0 + \frac{\sin(t\Omega)}{\sin\Omega}\,q_1$$

where $\cos\Omega = q_0 \cdot q_1$ (the 4D dot product).

**Properties:**
- Unit quaternion at every $t$ — stays on $S^3$
- Constant angular velocity $\dot{\theta}(t) = \Omega$
- Shortest path when $q_0 \cdot q_1 \geq 0$ (negate one if $< 0$)
- Reduces to LERP when $\Omega \approx 0$

### 75.3 SLERP as Exponential Map

SLERP has an equivalent group-theoretic form:

$$\text{slerp}(q_0, q_1, t) = q_0 \cdot (q_0^{-1} q_1)^t$$

where $(q)^t = \exp(t \cdot \log(q))$. This generalizes to any Lie group: geodesic interpolation is always $g_0 \cdot \exp(t \cdot \log(g_0^{-1} g_1))$.

### 75.4 Quaternion Averaging

The **Fréchet mean** on $S^3$ minimizes total geodesic distance:

$$\bar{q} = \arg\min_{q \in S^3} \sum_{i=1}^N d(q, q_i)^2$$

where $d(q_1, q_2) = 2\arccos|q_1 \cdot q_2|$.

**Iterative algorithm** (Karcher mean):
1. Initialize $\bar{q} = q_1$
2. Compute tangent vectors: $\delta_i = \log(\bar{q}^{-1} q_i)$ for all $i$
3. Update: $\bar{q} \leftarrow \bar{q} \cdot \exp(\frac{1}{N}\sum_i \delta_i)$
4. Repeat until $\|\sum \delta_i\| < \epsilon$

**Why arithmetic mean fails:** $(q_1 + q_2)/2$ is not on $S^3$, and even after normalization it does not minimize geodesic distance.

### 75.5 Eigenvalue Method for Averaging

For small dispersions, a faster method uses the $4 \times 4$ matrix:

$$M = \frac{1}{N}\sum_{i=1}^N q_i q_i^T$$

The mean $\bar{q}$ is the **eigenvector of $M$ corresponding to the largest eigenvalue**.

---

## Implementation (60 min)

```python
import numpy as np
from code.lie_groups.lie_groups import (
    angle_axis_to_quaternion, quaternion_to_rotation, quaternion_slerp
)

def quaternion_dot(q1, q2):
    """4D dot product."""
    return np.dot(q1, q2)

def slerp(q0, q1, t):
    """SLERP with hemisphere check."""
    dot = quaternion_dot(q0, q1)
    if dot < 0:  # ensure shortest path
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel — use LERP
        result = (1 - t) * q0 + t * q1
        return result / np.linalg.norm(result)
    omega = np.arccos(np.clip(dot, -1, 1))
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) / so) * q0 + (np.sin(t * omega) / so) * q1

def nlerp(q0, q1, t):
    """Normalized linear interpolation."""
    dot = quaternion_dot(q0, q1)
    if dot < 0:
        q1 = -q1
    result = (1 - t) * q0 + t * q1
    return result / np.linalg.norm(result)

def quaternion_mean_eigenvalue(quaternions):
    """Quaternion average via eigenvalue method."""
    Q = np.array(quaternions)  # (N, 4)
    M = Q.T @ Q / len(Q)
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs[:, -1]  # largest eigenvalue

# --- SLERP demonstration ---
q0 = angle_axis_to_quaternion([0, 0, 0])           # identity
q1 = angle_axis_to_quaternion([0, 0, np.pi / 2])   # 90° around z

print("=== SLERP between 0° and 90° ===")
for t in np.linspace(0, 1, 5):
    q_t = slerp(q0, q1, t)
    R_t = quaternion_to_rotation(q_t)
    angle = 2 * np.arccos(np.clip(abs(q_t[0]), 0, 1))
    print(f"  t={t:.2f}: angle={np.degrees(angle):6.1f}°, |q|={np.linalg.norm(q_t):.6f}")

# --- Compare SLERP vs NLERP ---
q_far = angle_axis_to_quaternion([0, 0, np.pi * 0.9])  # 162°
print(f"\n=== SLERP vs NLERP at t=0.5 (large angle) ===")
q_slerp = slerp(q0, q_far, 0.5)
q_nlerp = nlerp(q0, q_far, 0.5)
angle_s = np.degrees(2 * np.arccos(np.clip(abs(q_slerp[0]), 0, 1)))
angle_n = np.degrees(2 * np.arccos(np.clip(abs(q_nlerp[0]), 0, 1)))
print(f"  SLERP midpoint: {angle_s:.1f}° (should be 81.0°)")
print(f"  NLERP midpoint: {angle_n:.1f}° (biased)")

# --- Quaternion averaging ---
rng = np.random.default_rng(42)
true_axis_angle = np.array([0.3, -0.2, 0.5])
q_true = angle_axis_to_quaternion(true_axis_angle)

# Generate noisy samples
quats = []
for _ in range(20):
    noise = rng.normal(0, 0.05, size=3)
    q_noisy = angle_axis_to_quaternion(true_axis_angle + noise)
    if quaternion_dot(q_noisy, q_true) < 0:
        q_noisy = -q_noisy
    quats.append(q_noisy)

q_avg = quaternion_mean_eigenvalue(quats)
if quaternion_dot(q_avg, q_true) < 0:
    q_avg = -q_avg

err = np.degrees(2 * np.arccos(np.clip(abs(quaternion_dot(q_avg, q_true)), 0, 1)))
print(f"\n=== Quaternion Averaging (20 samples, σ=0.05) ===")
print(f"  True angle: {np.degrees(np.linalg.norm(true_axis_angle)):.1f}°")
print(f"  Error:      {err:.2f}°")

# --- Arithmetic mean failure ---
q_arith = np.mean(quats, axis=0)
q_arith = q_arith / np.linalg.norm(q_arith)
err_arith = np.degrees(2 * np.arccos(np.clip(abs(quaternion_dot(q_arith, q_true)), 0, 1)))
print(f"  Eigenvalue method error: {err:.2f}°")
print(f"  Arithmetic mean error:   {err_arith:.2f}°")
```

---

## Practice Problems (45 min)

**Problem 1:** Compute $\text{slerp}(q_0, q_1, 0.5)$ by hand where $q_0 = (1,0,0,0)$ (identity) and $q_1 = (0,0,0,1)$ (180° around z).

<details><summary>Answer</summary>

$\cos\Omega = q_0 \cdot q_1 = 0$, so $\Omega = \pi/2$. At $t=0.5$:

$$q(0.5) = \frac{\sin(\pi/4)}{\sin(\pi/2)}(1,0,0,0) + \frac{\sin(\pi/4)}{\sin(\pi/2)}(0,0,0,1) = \frac{\sqrt2/2}{1}(1,0,0,1) = \left(\frac{\sqrt2}{2}, 0, 0, \frac{\sqrt2}{2}\right)$$

This is a 90° rotation around z — exactly the midpoint.
</details>

**Problem 2:** You have 3 IMU readings as quaternions: $q_1$ (from sensor A), $q_2$ (from sensor B), $q_3$ (from visual odometry). How would you compute a weighted average with weights $w = (0.5, 0.3, 0.2)$?

<details><summary>Answer</summary>

Use the weighted eigenvalue method: $M = \sum_i w_i q_i q_i^T$, then take the eigenvector with the largest eigenvalue. Alternatively, use the iterative Karcher mean with weighted tangent vectors: $\bar{q} \leftarrow \bar{q} \cdot \exp(\sum_i w_i \cdot \log(\bar{q}^{-1} q_i))$.
</details>

**Problem 3:** Show that SLERP reduces to LERP (after normalization) when $q_0 \cdot q_1 \approx 1$.

<details><summary>Answer</summary>

When $\Omega \to 0$: $\sin((1-t)\Omega)/\sin\Omega \to (1-t)$ and $\sin(t\Omega)/\sin\Omega \to t$ by L'Hôpital's rule. So $\text{slerp} \to (1-t)q_0 + tq_1 = \text{LERP}(q_0, q_1, t)$, which is already approximately unit-length when $q_0 \approx q_1$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that SLERP gives constant angular velocity, i.e., the angle $\alpha(t)$ between $q_0$ and $q(t)$ is linear in $t$.

<details><summary>Answer</summary>

$\cos\alpha(t) = q_0 \cdot q(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} + 0 = \frac{\sin((1-t)\Omega)}{\sin\Omega}$. Wait — more carefully:

$q_0 \cdot q(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\|q_0\|^2 + \frac{\sin(t\Omega)}{\sin\Omega}(q_0 \cdot q_1) = \frac{\sin((1-t)\Omega)}{\sin\Omega} + \frac{\sin(t\Omega)\cos\Omega}{\sin\Omega}$

Using the identity $\sin(A) + \cos(\Omega)\sin(B) = \sin(A+B)$ when $A = (1-t)\Omega$, $B = t\Omega$: this equals $\cos(t\Omega)$, so $\alpha(t) = t\Omega$ — linear. ✓
</details>

**Challenge 2:** Implement cubic quaternion interpolation (SQUAD) for 4 control quaternions, analogous to cubic Bézier for positions.

<details><summary>Answer</summary>

```python
def squad(q0, q1, q2, q3, t):
    """Spherical and Quadrangle interpolation."""
    s1 = q1 * np.exp(-0.25 * (np.log(q1.conj() * q2) + np.log(q1.conj() * q0)))
    s2 = q2 * np.exp(-0.25 * (np.log(q2.conj() * q3) + np.log(q2.conj() * q1)))
    return slerp(slerp(q1, q2, t), slerp(s1, s2, t), 2*t*(1-t))
```
SQUAD provides C¹ continuity at knot points, suitable for smooth multi-waypoint trajectories.
</details>

**Challenge 3:** The Karcher mean iteration can diverge if initial quaternions span more than a hemisphere. Explain why and propose a fix.

<details><summary>Answer</summary>

The logarithmic map is only well-defined for rotations within $\pi$ of the current estimate. If samples span $> \pi$, some $\log(\bar{q}^{-1}q_i)$ wrap to the wrong direction. Fix: ensure all quaternions are in the same hemisphere by flipping $q_i \to -q_i$ when $q_i \cdot \bar{q} < 0$ before each iteration.
</details>

---

## Connections
- **Prerequisites:** [Day 74](day-74-quaternions.md) — quaternion algebra and rotation correspondence
- **Forward:** [Day 76](day-76-rotation-representations.md) — comparison table including interpolation properties
- **OKS:** SLERP is used in trajectory planners for smooth orientation profiles; quaternion averaging appears in EKF orientation fusion

## Self-Check
- [ ] I can derive the SLERP formula and explain its constant-velocity property
- [ ] I understand why NLERP is faster but non-uniform compared to SLERP
- [ ] I can compute a quaternion average using the eigenvalue method
- [ ] I know when to flip quaternion signs to ensure shortest-path interpolation
