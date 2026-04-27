"""
Day 17: Newton's Method — Companion Code
Phase II, Week 3

Run: python newtons_method.py
"""

import numpy as np


# =============================================================================
# Test Functions
# =============================================================================

def rosenbrock_fgh(x):
    """Rosenbrock: f, gradient, Hessian."""
    f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    g = np.array([
        -2*(1 - x[0]) + 100 * 2*(x[1] - x[0]**2) * (-2*x[0]),
        100 * 2*(x[1] - x[0]**2)
    ])

    H = np.array([
        [2 + 100*(12*x[0]**2 - 4*x[1]), -400*x[0]],
        [-400*x[0], 200]
    ])

    return f, g, H


def quadratic_fgh(A, b=None):
    """Quadratic f = 0.5 x^T A x - b^T x: returns (f_func, g_func, H_func)."""
    n = A.shape[0]
    if b is None:
        b = np.zeros(n)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def g(x): return A @ x - b
    def H(x): return A

    x_star = np.linalg.solve(A, b)
    return f, g, H, x_star


# =============================================================================
# Exercise 1: Pure Newton's Method
# =============================================================================

def newtons_method(fgh, x0, max_iter=100, tol=1e-12):
    """Pure Newton's method.

    fgh: callable returning (f, grad, hessian).
    """
    x = x0.copy().astype(float)
    history = {"f": [], "grad_norm": [], "x": [], "error": []}

    for k in range(max_iter):
        f, g, H = fgh(x)
        gnorm = np.linalg.norm(g)

        history["f"].append(f)
        history["grad_norm"].append(gnorm)
        history["x"].append(x.copy())

        if gnorm < tol:
            break

        # Newton step: solve H δ = -g
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            print(f"  Singular Hessian at iteration {k}")
            break

        x = x + delta

    return x, history


# =============================================================================
# Exercise 2: Newton on Rosenbrock
# =============================================================================

def test_rosenbrock():
    """Compare Newton vs GD on Rosenbrock."""
    print("=== Exercise 2: Newton vs GD on Rosenbrock ===")
    x0 = np.array([-1.0, 1.0])
    x_star = np.array([1.0, 1.0])

    # Newton
    x_newton, hist_newton = newtons_method(rosenbrock_fgh, x0, max_iter=100)
    iters_n = len(hist_newton["f"])
    err_n = np.linalg.norm(x_newton - x_star)

    # GD with backtracking for comparison
    x_gd, hist_gd = gd_backtracking(x0, max_iter=50000)
    iters_gd = len(hist_gd["f"])
    err_gd = np.linalg.norm(x_gd - x_star)

    print(f"  Newton:  {iters_n:>6} iters, error = {err_n:.2e}")
    print(f"  GD:      {iters_gd:>6} iters, error = {err_gd:.2e}")
    print(f"  Speedup: {iters_gd / max(iters_n, 1):.0f}×")


def gd_backtracking(x0, max_iter=50000, tol=1e-8):
    """Simple GD for comparison."""
    x = x0.copy()
    history = {"f": [], "x": []}

    for k in range(max_iter):
        f, g, _ = rosenbrock_fgh(x)
        history["f"].append(f)
        history["x"].append(x.copy())

        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            break

        # Backtracking
        alpha = 1.0
        d = -g
        for _ in range(50):
            f_new, _, _ = rosenbrock_fgh(x + alpha * d)
            if f_new <= f + 1e-4 * alpha * (g @ d):
                break
            alpha *= 0.5

        x = x + alpha * d

    return x, history


# =============================================================================
# Exercise 3: Quadratic Convergence Verification
# =============================================================================

def verify_quadratic_convergence():
    """Show error_k+1 ≈ C * error_k^2."""
    print("\n=== Exercise 3: Quadratic Convergence ===")
    x0 = np.array([2.0, -3.0])
    x_star = np.array([1.0, 1.0])

    _, hist = newtons_method(rosenbrock_fgh, x0, max_iter=30)

    errors = [np.linalg.norm(np.array(xi) - x_star)
              for xi in hist["x"]]

    print(f"  {'k':>3}  {'||e_k||':>12}  {'||e_{k+1}||/||e_k||^2':>22}")
    for k in range(len(errors) - 1):
        if errors[k] > 1e-15 and errors[k+1] > 0:
            ratio = errors[k+1] / errors[k]**2 if errors[k] > 1e-12 else float('inf')
            print(f"  {k:>3}  {errors[k]:>12.4e}  {ratio:>22.4f}")


# =============================================================================
# Exercise 4: Failure Case — Indefinite Hessian
# =============================================================================

def test_failure_case():
    """Newton fails when started far from minimum (indefinite H)."""
    print("\n=== Exercise 4: Failure Case ===")

    # f(x) = x^4/4 - x^2/2 (non-convex, minima at ±1, saddle at 0)
    def fgh_quartic(x):
        x1 = x[0]
        f = x1**4/4 - x1**2/2
        g = np.array([x1**3 - x1])
        H = np.array([[3*x1**2 - 1]])
        return f, g, H

    x0 = np.array([0.4])  # Hessian = 3*0.16 - 1 = -0.52 < 0 (indefinite!)
    _, g0, H0 = fgh_quartic(x0)
    print(f"  At x0 = {x0[0]}: H = {H0[0,0]:.2f} ({'PD' if H0[0,0] > 0 else 'INDEFINITE'})")

    x, hist = newtons_method(fgh_quartic, x0, max_iter=50)
    print(f"  Pure Newton converges to: x = {x[0]:.6f} (saddle at 0!)")

    # Damped Newton
    x_d, hist_d = damped_newton(fgh_quartic, x0, max_iter=50)
    print(f"  Damped Newton converges to: x = {x_d[0]:.6f} (minimum at ±1)")


# =============================================================================
# Exercise 5: Modified Newton (H + μI)
# =============================================================================

def modified_newton(fgh, x0, max_iter=100, tol=1e-12, mu_init=1e-3):
    """Newton with H + μI regularization for indefinite Hessians."""
    x = x0.copy().astype(float)
    history = {"f": [], "grad_norm": [], "x": []}

    for k in range(max_iter):
        f, g, H = fgh(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(f)
        history["grad_norm"].append(gnorm)
        history["x"].append(x.copy())

        if gnorm < tol:
            break

        # Ensure H + μI is PD
        eigvals = np.linalg.eigvalsh(H)
        min_eig = np.min(eigvals)
        mu = max(0, -min_eig + mu_init)

        H_reg = H + mu * np.eye(len(x))
        delta = np.linalg.solve(H_reg, -g)
        x = x + delta

    return x, history


def damped_newton(fgh, x0, max_iter=100, tol=1e-12):
    """Newton with line search for global convergence."""
    x = x0.copy().astype(float)
    history = {"f": [], "grad_norm": [], "x": []}

    for k in range(max_iter):
        f, g, H = fgh(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(f)
        history["grad_norm"].append(gnorm)
        history["x"].append(x.copy())

        if gnorm < tol:
            break

        # Modified Newton direction
        eigvals = np.linalg.eigvalsh(H)
        min_eig = np.min(eigvals)
        mu = max(0, -min_eig + 1e-4)
        H_reg = H + mu * np.eye(len(x))
        delta = np.linalg.solve(H_reg, -g)

        # Backtracking line search
        alpha = 1.0
        for _ in range(50):
            f_new, _, _ = fgh(x + alpha * delta)
            if f_new <= f + 1e-4 * alpha * (g @ delta):
                break
            alpha *= 0.5

        x = x + alpha * delta

    return x, history


def test_modified_newton():
    """Modified Newton on Rosenbrock from a bad start."""
    print("\n=== Exercise 5: Modified Newton ===")
    x0 = np.array([-5.0, 5.0])
    x_star = np.array([1.0, 1.0])

    # Pure Newton may diverge from far away
    x_pure, hist_pure = newtons_method(rosenbrock_fgh, x0, max_iter=100)
    err_pure = np.linalg.norm(x_pure - x_star)

    # Modified Newton
    x_mod, hist_mod = modified_newton(rosenbrock_fgh, x0, max_iter=100)
    err_mod = np.linalg.norm(x_mod - x_star)

    # Damped Newton (with line search)
    x_damp, hist_damp = damped_newton(rosenbrock_fgh, x0, max_iter=100)
    err_damp = np.linalg.norm(x_damp - x_star)

    print(f"  From x0 = {x0}:")
    print(f"  Pure Newton:     {len(hist_pure['f']):>4} iters, err = {err_pure:.2e}")
    print(f"  Modified (H+μI): {len(hist_mod['f']):>4} iters, err = {err_mod:.2e}")
    print(f"  Damped (LS):     {len(hist_damp['f']):>4} iters, err = {err_damp:.2e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_rosenbrock()
    verify_quadratic_convergence()
    test_failure_case()
    test_modified_newton()
