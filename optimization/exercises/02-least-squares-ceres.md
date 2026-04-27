# Exercises: Least Squares and Ceres Solver

### Chapter 03: Nonlinear Least Squares

**Self-assessment guide:** Work each problem before expanding the answer. Focus on formulating the cost function — the solver does the hard work once you set up the problem correctly.

---

## Section A — Conceptual Questions

**A1.** Explain the difference between Gauss-Newton and Levenberg-Marquardt. When does GN fail and LM succeed? When might you prefer GN?

<details><summary>Answer</summary>

**Gauss-Newton:** Approximates the Hessian as $J^T J$ (ignoring second-order terms). Solves $J^T J \Delta x = -J^T r$.

**Levenberg-Marquardt:** Adds damping: $(J^T J + \lambda I) \Delta x = -J^T r$.

**When GN fails:**
- Far from the minimum, $J^T J$ may not be positive definite → step direction is wrong
- When $J$ is rank-deficient → $J^T J$ is singular → can't solve the system
- Large residuals (bad model fit) → second-order terms matter

**When LM succeeds:** The damping $\lambda I$ ensures the system is always solvable and the step is bounded. Large $\lambda$ → gradient descent (safe, slow). Small $\lambda$ → Gauss-Newton (fast near minimum).

**When to prefer GN:** When you're already close to the minimum and the model fits well (small residuals). GN is slightly cheaper per iteration (no $\lambda$ adaptation logic) and converges quadratically near the minimum, same as LM.

**Ceres uses LM by default** — it's the safest choice for general problems.

</details>

**A2.** Why do robust loss functions (Huber, Cauchy) help with outliers? Draw the loss function $\rho(r)$ for Huber and compare it to the standard squared loss $r^2$.

<details><summary>Answer</summary>

Standard squared loss: $\rho(r) = r^2$. The influence function (derivative) is $\rho'(r) = 2r$ — outliers with large $r$ have proportionally large influence on the solution.

Huber loss:
$$\rho(r) = \begin{cases} r^2 & |r| \leq k \\ 2k|r| - k^2 & |r| > k \end{cases}$$

For $|r| > k$, the influence grows **linearly** instead of quadratically. An outlier 10× larger contributes 10× more influence (Huber) vs. 100× more (squared).

Cauchy loss: $\rho(r) = \log(1 + r^2/c^2)$. Even more aggressive — the influence **decreases** for very large residuals, effectively ignoring outliers completely.

**Visual comparison (sketch):**
```
ρ(r)
 │   Squared: r²       /
 │                    /
 │        Huber     /'
 │               /·'
 │            /·'  Cauchy: log(1+r²)
 │         ·'
 │       ·'
 │     ·'
 │   ·'
 │ ·'
 └──────────────────── r
```

</details>

**A3.** Ceres uses `AutoDiffCostFunction` by default. Explain what happens internally when Ceres evaluates the Jacobian. What is a Jet?

<details><summary>Answer</summary>

When you template your cost function:

```cpp
template <typename T>
bool operator()(const T* x, T* residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
}
```

Ceres instantiates this with `T = Jet<double, N>` where `N` is the number of parameters.

A `Jet<double, N>` is a dual number: it stores a value and N partial derivatives simultaneously:
```
jet.a = function value
jet.v[i] = ∂f/∂x_i
```

Arithmetic operations propagate derivatives automatically:
- `jet_a + jet_b` → adds values, adds derivatives
- `jet_a * jet_b` → product rule: $(ab, a'b + ab')$
- `cos(jet_a)` → $(\cos(a), -\sin(a) \cdot a')$

So evaluating the cost function once with Jets computes **both** the residual values **and** the full Jacobian — without finite differences and without hand-coded derivatives.

</details>

---

## Section B — Computation Problems

**B1.** Implement a Ceres-style least-squares solver in Python (from scratch) to fit a circle to noisy 2D points.

Model: Circle with center $(cx, cy)$ and radius $r$.
Residual per point: $r_i = \sqrt{(x_i - cx)^2 + (y_i - cy)^2} - r$

```python
import numpy as np

def generate_circle_data(cx, cy, r, n_points=50, noise_std=0.1, n_outliers=3):
    """Generate noisy points on a circle, plus outliers."""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = cx + r * np.cos(theta) + np.random.randn(n_points) * noise_std
    y = cy + r * np.sin(theta) + np.random.randn(n_points) * noise_std
    # Add outliers
    for _ in range(n_outliers):
        x = np.append(x, cx + np.random.randn() * r * 3)
        y = np.append(y, cy + np.random.randn() * r * 3)
    return x, y

def circle_residuals(params, x_data, y_data):
    """Compute residuals for circle fit."""
    cx, cy, r = params
    # TODO: return array of residuals
    pass

def circle_jacobian(params, x_data, y_data):
    """Compute Jacobian [n_points × 3]."""
    cx, cy, r = params
    # TODO: compute ∂r_i/∂cx, ∂r_i/∂cy, ∂r_i/∂r for each point
    pass

def levenberg_marquardt(residual_fn, jacobian_fn, x0, args, max_iter=50, tol=1e-8):
    """
    Implement LM optimizer.
    Track: cost, lambda, step_type (GN or GD) per iteration.
    """
    # TODO: implement
    pass

# Test: fit a circle to noisy data
x_data, y_data = generate_circle_data(3, -2, 5, n_points=50, noise_std=0.2)
params0 = np.array([0.0, 0.0, 1.0])  # bad initial guess
result = levenberg_marquardt(circle_residuals, circle_jacobian, params0, (x_data, y_data))
# Expected: converges to approximately (3, -2, 5)
```

<details><summary>Solution</summary>

```python
import numpy as np

def circle_residuals(params, x_data, y_data):
    cx, cy, r = params
    d = np.sqrt((x_data - cx)**2 + (y_data - cy)**2)
    return d - r

def circle_jacobian(params, x_data, y_data):
    cx, cy, r = params
    d = np.sqrt((x_data - cx)**2 + (y_data - cy)**2)
    d_safe = np.where(d > 1e-10, d, 1e-10)  # avoid division by zero
    n = len(x_data)
    J = np.zeros((n, 3))
    J[:, 0] = -(x_data - cx) / d_safe  # ∂r_i/∂cx
    J[:, 1] = -(y_data - cy) / d_safe  # ∂r_i/∂cy
    J[:, 2] = -np.ones(n)              # ∂r_i/∂r = -1
    return J

def levenberg_marquardt(residual_fn, jacobian_fn, x0, args, max_iter=50, tol=1e-8):
    x = x0.copy().astype(float)
    lam = 1e-3  # initial damping
    nu = 2.0    # damping update factor
    
    r = residual_fn(x, *args)
    cost = 0.5 * np.dot(r, r)
    
    for i in range(max_iter):
        J = jacobian_fn(x, *args)
        JtJ = J.T @ J
        Jtr = J.T @ r
        
        # Solve (JtJ + λI)Δx = -Jtr
        H_damped = JtJ + lam * np.diag(np.diag(JtJ) + 1e-10)
        dx = np.linalg.solve(H_damped, -Jtr)
        
        # Evaluate candidate
        x_new = x + dx
        r_new = residual_fn(x_new, *args)
        cost_new = 0.5 * np.dot(r_new, r_new)
        
        # Gain ratio: actual improvement / predicted improvement
        predicted = -dx.dot(Jtr) - 0.5 * dx.dot(JtJ @ dx)
        gain_ratio = (cost - cost_new) / (predicted + 1e-20)
        
        if gain_ratio > 0:  # improvement
            x = x_new
            r = r_new
            cost = cost_new
            lam = lam * max(1/3, 1 - (2*gain_ratio - 1)**3)
            nu = 2.0
        else:  # no improvement
            lam *= nu
            nu *= 2
        
        grad_norm = np.linalg.norm(Jtr)
        if grad_norm < tol:
            break
    
    return x, cost, i+1

# Test
np.random.seed(42)
x_data, y_data = generate_circle_data(3, -2, 5, n_points=50, noise_std=0.2)
params0 = np.array([0.0, 0.0, 1.0])
result, cost, n_iter = levenberg_marquardt(circle_residuals, circle_jacobian, params0, (x_data, y_data))
print(f"Result: cx={result[0]:.2f}, cy={result[1]:.2f}, r={result[2]:.2f}")
print(f"Converged in {n_iter} iterations, cost={cost:.4f}")
```

</details>

**B2.** Extend B1: add Huber loss and compare the circle fit with and without outliers.

Hint: With Huber loss, the effective residual becomes $\sqrt{2\rho(r_i)}$ and the Jacobian needs the loss weight $w_i = \rho'(r_i) / r_i$.

<details><summary>Hint for implementation</summary>

Iteratively Reweighted Least Squares (IRLS):
1. Compute residuals $r_i$
2. Compute Huber weights: $w_i = 1$ if $|r_i| \leq k$, else $w_i = k / |r_i|$
3. Solve weighted least squares: $(J^T W J) \Delta x = -J^T W r$ where $W = \text{diag}(w_i)$
4. Repeat

This is what Ceres does internally when you set a loss function.

</details>

---

## Section C — Ceres C++ Problems (Conceptual)

**C1.** Write the Ceres cost functor for fitting a 2D line $y = mx + b$ to data points. Use `AutoDiffCostFunction`.

<details><summary>Answer</summary>

```cpp
struct LineFitCost {
    LineFitCost(double x, double y) : x_(x), y_(y) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = m (slope), params[1] = b (intercept)
        residual[0] = T(y_) - (params[0] * T(x_) + params[1]);
        return true;
    }
    
    double x_, y_;
};

// Usage:
double params[2] = {0.0, 0.0};  // initial guess
ceres::Problem problem;
for (int i = 0; i < n; ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<LineFitCost, 1, 2>(
            new LineFitCost(x_data[i], y_data[i])),
        nullptr,  // no loss function
        params);
}
```

Note: For a line fit, this is overkill — you'd use linear least squares. But the pattern generalizes to any nonlinear model.

</details>

**C2.** The OKS sensorbar measurement model computes a distance from a reflector detection angle. Write a Ceres-style cost functor for: given a robot pose $(x, y, \theta)$ and a known reflector position $(rx, ry)$, the expected bearing is $\alpha = \text{atan2}(ry - y, rx - x) - \theta$, and the measured bearing is $\alpha_{\text{meas}}$.

<details><summary>Answer</summary>

```cpp
struct SensorbarBearingCost {
    SensorbarBearingCost(double rx, double ry, double alpha_measured)
        : rx_(rx), ry_(ry), alpha_measured_(alpha_measured) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residual) const {
        // pose[0] = x, pose[1] = y, pose[2] = theta
        T dx = T(rx_) - pose[0];
        T dy = T(ry_) - pose[1];
        T alpha_expected = ceres::atan2(dy, dx) - pose[2];
        
        // Normalize angle difference to [-pi, pi]
        T diff = alpha_expected - T(alpha_measured_);
        // Use NormalizeAngle helper (wraps to [-pi, pi])
        residual[0] = NormalizeAngle(diff);
        return true;
    }
    
    double rx_, ry_, alpha_measured_;
};

// Key insight: the atan2 handles the quadrant correctly,
// and angle normalization prevents 2π discontinuities.
// Ceres auto-diff handles the atan2 Jacobian automatically.
```

This is conceptually what the OKS estimator does in its sensorbar measurement update, but formulated as an optimization cost instead of an EKF innovation.

</details>

---

## Section D — Debugging Exercises

**D1.** You run a Ceres optimization and get this output:

```
Solver Summary
  Termination:    NO_CONVERGENCE
  Final cost:     1.234567e+03
  Iterations:     50 (max)
  Time:           2.3 sec
  
  Gradient norm:  4.5e-01
  Step norm:      2.3e-04
  Trust region:   1.0e-12
```

What's happening? How would you fix it?

<details><summary>Answer</summary>

**Diagnosis:** The trust region has shrunk to near-zero ($10^{-12}$) while the gradient is still significant ($0.45$). This means:
- The solver keeps proposing steps that **don't reduce the cost** (rejected steps)
- It shrinks the trust region each time
- Eventually it can't take any useful step

**Likely causes:**
1. **Bad Jacobian:** AutoDiff is usually correct, but if using numeric diff, the step size may be wrong
2. **Numerical issues in residual:** NaN, Inf, or discontinuities in the cost function
3. **Very bad initial guess:** The linearization is so poor that no step helps

**Fixes:**
1. Check for NaN in residuals: enable `options.check_gradients = true`
2. Try a better initial guess
3. Increase `max_num_iterations`
4. Switch to `DENSE_QR` (more numerically stable for small problems)
5. Add a robust loss function if outliers are causing issues
6. Check `options.minimizer_progress_to_stdout = true` to see the per-iteration behavior

</details>
