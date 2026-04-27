# Exercises: Numerical Methods and Practical Debugging

### Chapter 07: Numerical Methods and Practical Concerns

**Self-assessment guide:** These exercises test your ability to diagnose and fix numerical issues — the skill that separates textbook knowledge from production debugging.

---

## Section A — Conceptual Questions

**A1.** The normal equations $J^T J \Delta x = -J^T r$ square the condition number: $\kappa(J^T J) = \kappa(J)^2$. For a problem where $\kappa(J) = 10^6$, how many digits of accuracy can you expect in the solution when using:
1. Normal equations (Cholesky on $J^T J$)
2. QR factorization of $J$
3. SVD of $J$

<details><summary>Answer</summary>

With double precision (~16 digits):

1. **Normal equations:** $\kappa(J^T J) = 10^{12}$. Digits lost: $\log_{10}(10^{12}) = 12$. Only **4 digits** of accuracy. Potentially useless.

2. **QR factorization:** Works with $\kappa(J) = 10^6$ directly. Digits lost: 6. About **10 digits** of accuracy. Usually sufficient.

3. **SVD:** Same as QR for well-posed problems (~10 digits). SVD additionally reveals the rank of $J$ and allows truncation of near-zero singular values for rank-deficient problems.

**Lesson:** For $\kappa(J) > 10^8$, avoid normal equations. Ceres' `DENSE_QR` and `SPARSE_SCHUR` avoid this issue.

</details>

**A2.** You're using finite differences to compute a Jacobian. The step size $h$ affects accuracy:
- Too large: truncation error dominates
- Too small: roundoff error dominates

For forward differences, the optimal $h$ is about $\sqrt{\epsilon_{\text{mach}}} \approx 1.5 \times 10^{-8}$. For central differences, it's $\epsilon_{\text{mach}}^{1/3} \approx 6 \times 10^{-6}$. Explain why central differences allow a larger (better) step size.

<details><summary>Answer</summary>

**Forward differences:** $\frac{f(x+h) - f(x)}{h}$
- Truncation error: $O(h)$ (from Taylor expansion, the leading error term is $\frac{h}{2}f''$)
- Roundoff error: $O(\epsilon/h)$ (subtracting two nearby values, then dividing by small $h$)
- Total error: $\frac{h}{2}|f''| + \frac{2\epsilon|f|}{h}$
- Minimize: $h_{\text{opt}} \sim \sqrt{\epsilon} \approx 10^{-8}$

**Central differences:** $\frac{f(x+h) - f(x-h)}{2h}$
- Truncation error: $O(h^2)$ (the first-order error term cancels by symmetry!)
- Roundoff error: $O(\epsilon/h)$ (same)
- Total error: $\frac{h^2}{6}|f'''| + \frac{\epsilon|f|}{h}$
- Minimize: $h_{\text{opt}} \sim \epsilon^{1/3} \approx 10^{-5.3}$

The larger optimal $h$ for central differences means we're further from the roundoff-dominated regime, giving **more accurate results**. Central differences are $O(h^2)$ accurate vs $O(h)$ for forward.

**Trade-off:** Central differences require **2 function evaluations** per parameter (vs 1 for forward). For expensive functions, this doubles the Jacobian computation cost.

</details>

**A3.** What is "fill-in" in sparse matrix factorization, and why does variable ordering matter?

<details><summary>Answer</summary>

**Fill-in:** New non-zero entries that appear in the Cholesky factor $L$ that were zero in the original matrix $H$. Each fill-in entry requires computation and storage.

**Example:** An arrowhead matrix (star graph):
```
H = [X X X X X]    L = [X . . . .]    (good ordering)
    [X X . . .]        [X X . . .]
    [X . X . .]        [X . X . .]
    [X . . X .]        [X . . X .]
    [X . . . X]        [X X X X X]    ← fill-in in last row

vs.

H = [X . . . X]    L = [X . . . .]    (bad ordering — hub first)
    [. X . . X]        [. X . . .]
    [. . X . X]        [. . X . .]
    [. . . X X]        [. . . X .]
    [X X X X X]        [X X X X X]    ← same, but rows above have fill too
```

Actually, putting the "hub" node last produces minimal fill-in (the dense row only fills the last row of $L$). Putting it first causes fill-in to cascade through all subsequent rows.

**AMD (Approximate Minimum Degree):** At each step, eliminate the variable with the fewest connections (smallest degree in the elimination graph). This heuristic minimizes fill-in for most practical problems.

**For SLAM:** Poses at the end of long corridors (few connections) should be eliminated first. Poses at intersections (many connections) should be eliminated last.

</details>

---

## Section B — Computation Problems

**B1.** Write a function that checks the gradient of a cost function using finite differences, similar to Ceres' `CheckGradients` functionality.

```python
import numpy as np

def check_gradient(f, grad_f, x, h=1e-7, tol=1e-5):
    """
    Compare analytic gradient with central-difference approximation.
    Returns: max relative error, per-component errors.
    """
    # TODO: implement
    # For each component, compute central difference
    # Compare with analytic gradient
    # Report max relative error
    pass

# Test with a known function
def f(x):
    return x[0]**2 * np.sin(x[1]) + np.exp(x[0] * x[1])

def grad_f(x):
    return np.array([
        2 * x[0] * np.sin(x[1]) + x[1] * np.exp(x[0] * x[1]),
        x[0]**2 * np.cos(x[1]) + x[0] * np.exp(x[0] * x[1])
    ])

x_test = np.array([1.5, 0.7])
max_err = check_gradient(f, grad_f, x_test)
print(f"Max relative error: {max_err:.2e}")  # Should be < 1e-8
```

<details><summary>Solution</summary>

```python
import numpy as np

def check_gradient(f, grad_f, x, h=1e-7, tol=1e-5):
    n = len(x)
    analytic = grad_f(x)
    numeric = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        numeric[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    errors = np.abs(analytic - numeric)
    # Relative error: use max(|analytic|, |numeric|, 1) to avoid division by zero
    scale = np.maximum(np.abs(analytic), np.maximum(np.abs(numeric), 1.0))
    rel_errors = errors / scale
    
    max_rel_err = np.max(rel_errors)
    
    for i in range(n):
        status = "OK" if rel_errors[i] < tol else "FAIL"
        print(f"  x[{i}]: analytic={analytic[i]:.8f}, "
              f"numeric={numeric[i]:.8f}, "
              f"rel_err={rel_errors[i]:.2e} [{status}]")
    
    return max_rel_err

# Test
def f(x):
    return x[0]**2 * np.sin(x[1]) + np.exp(x[0] * x[1])

def grad_f(x):
    return np.array([
        2 * x[0] * np.sin(x[1]) + x[1] * np.exp(x[0] * x[1]),
        x[0]**2 * np.cos(x[1]) + x[0] * np.exp(x[0] * x[1])
    ])

x_test = np.array([1.5, 0.7])
max_err = check_gradient(f, grad_f, x_test)
print(f"\nMax relative error: {max_err:.2e}")

# Now introduce a deliberate bug:
def grad_f_buggy(x):
    return np.array([
        2 * x[0] * np.sin(x[1]) + x[1] * np.exp(x[0] * x[1]),
        x[0]**2 * np.sin(x[1]) + x[0] * np.exp(x[0] * x[1])  # BUG: sin not cos
    ])

print("\nWith buggy gradient:")
max_err = check_gradient(f, grad_f_buggy, x_test)
print(f"Max relative error: {max_err:.2e}")  # Will show FAIL for x[1]
```

</details>

**B2.** Demonstrate the effect of condition number on solver accuracy.

```python
import numpy as np

def solve_and_check(kappa_target):
    """
    Create a system Ax = b with known condition number,
    solve it, and measure the actual error.
    """
    n = 10
    # Create matrix with specific condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    # Singular values linearly spaced from 1 to kappa
    S = np.linspace(1, kappa_target, n)
    A = U @ np.diag(S) @ U.T
    
    # Known solution
    x_true = np.ones(n)
    b = A @ x_true
    
    # Solve
    x_computed = np.linalg.solve(A, b)
    
    # Measure error
    rel_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    actual_kappa = np.linalg.cond(A)
    
    return rel_error, actual_kappa

# TODO: Run for kappa = 1, 1e3, 1e6, 1e9, 1e12, 1e15
# Plot relative error vs condition number
# At what kappa does the solution become unreliable?
```

<details><summary>Solution</summary>

```python
import numpy as np

def solve_and_check(kappa_target, n=10):
    np.random.seed(42)
    U, _ = np.linalg.qr(np.random.randn(n, n))
    S = np.logspace(0, np.log10(kappa_target), n)
    A = U @ np.diag(S) @ U.T
    
    x_true = np.ones(n)
    b = A @ x_true
    x_computed = np.linalg.solve(A, b)
    
    rel_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    actual_kappa = np.linalg.cond(A)
    
    return rel_error, actual_kappa

print(f"{'κ target':>12s} {'κ actual':>12s} {'Relative Error':>15s} {'Digits Lost':>12s}")
print("-" * 55)
for k in [1, 1e3, 1e6, 1e9, 1e12, 1e15]:
    err, kappa = solve_and_check(k)
    digits_lost = max(0, np.log10(kappa))
    print(f"{k:12.0e} {kappa:12.2e} {err:15.2e} {digits_lost:12.1f}")

# Expected pattern:
# κ=1:    error ~1e-16 (machine precision)
# κ=1e3:  error ~1e-13 (lost 3 digits)
# κ=1e6:  error ~1e-10 (lost 6 digits)
# κ=1e12: error ~1e-4  (lost 12 digits — barely usable)
# κ=1e15: error ~1e-1  (lost 15 digits — garbage)
```

Rule of thumb: you lose $\log_{10}(\kappa)$ digits of accuracy. With 16 digits (double precision), $\kappa > 10^{12}$ means fewer than 4 reliable digits.

</details>

---

## Section C — OKS Debugging Scenarios

**C1.** You're debugging the OKS navigation estimator and find that the covariance matrix $P$ has grown to have a condition number of $10^{10}$. This causes the Kalman gain computation $K = PH^T(HPH^T + R)^{-1}$ to produce inaccurate results.

1. What causes the covariance to become ill-conditioned?
2. How can you detect this in production?
3. What are the remedies?

<details><summary>Answer</summary>

1. **Causes of ill-conditioned covariance:**
   - Unobservable or weakly observable states accumulate infinite/large covariance
   - Long periods without sensor corrections (sensorbar out of range)
   - Model mismatch: process noise $Q$ too small relative to actual dynamics
   - Numerical drift: small asymmetry errors compound over many EKF iterations
   - One state has very different units/scale than others

2. **Detection in production:**
   ```python
   # Check at each EKF update:
   kappa = np.linalg.cond(P)
   if kappa > 1e8:
       log_warning(f"Covariance condition number: {kappa:.2e}")
   
   # Also check symmetry:
   asymmetry = np.max(np.abs(P - P.T)) / np.max(np.abs(P))
   if asymmetry > 1e-10:
       log_warning(f"P asymmetry: {asymmetry:.2e}")
   
   # Check positive definiteness:
   eigenvalues = np.linalg.eigvalsh(P)
   if np.any(eigenvalues < 0):
       log_error("P has negative eigenvalue — EKF diverging!")
   ```

3. **Remedies:**
   - **Joseph form update:** $P_{k|k} = (I - KH)P_{k|k-1}(I - KH)^T + KRK^T$ (numerically stable)
   - **Square root filter:** Track $\sqrt{P}$ instead of $P$ — halves the effective condition number
   - **Symmetry enforcement:** $P = \frac{1}{2}(P + P^T)$ after each update
   - **Covariance reset:** If $\kappa > \text{threshold}$, add small diagonal: $P \leftarrow P + \epsilon I$
   - **Variable scaling:** Ensure all states are O(1) so $P$ entries are comparable

</details>

**C2.** A Ceres optimization for calibrating a sensor model converges to a solution, but the residuals have a clear pattern (not random). Your colleague says "the solver found the minimum, so we're done." What's wrong?

<details><summary>Answer</summary>

**Patterned residuals indicate a model deficiency**, not a solver failure. The solver correctly found the minimum of the **given** cost function, but:

1. **The model doesn't capture the physics.** A pattern in residuals means there's systematic error that the model parameters can't absorb. Examples:
   - Fitting a line when the data is quadratic → residuals show a parabolic pattern
   - Calibrating a sensor without modeling temperature dependence → residuals correlate with temperature

2. **What to do:**
   - Plot residuals vs. each independent variable (time, position, temperature, etc.)
   - The pattern reveals the missing model term
   - Add the missing term (bias, nonlinearity, environmental factor)
   - Re-optimize — residuals should look random (white noise)

3. **In OKS context:** If sensorbar calibration residuals show a pattern vs. angle, it suggests the sensorbar measurement model is missing a nonlinearity (e.g., parallax, mounting offset). The correct fix is to improve the model, not to blame the solver.

4. **Validation:** A proper calibration should pass a **chi-squared test** — the sum of squared normalized residuals should be approximately equal to the degrees of freedom ($m - n$).

</details>
