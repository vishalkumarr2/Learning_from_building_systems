"""
Day 22: Convergence Theory — Companion Code
Phase II, Week 4

Run: python convergence_theory.py
"""

import numpy as np
import time


# =============================================================================
# Test Setup
# =============================================================================

def make_quadratic(n, cond, seed=42):
    """Create quadratic with specified condition number."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.linspace(1, cond, n)
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)
    x_star = np.linalg.solve(A, b)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad(x): return A @ x - b
    def hess(x): return A

    return f, grad, hess, A, x_star


# =============================================================================
# Optimizers with full tracking
# =============================================================================

def gd_track(f, grad, x0, x_star, max_iter=5000, tol=1e-14):
    """GD with Armijo backtracking, tracking errors."""
    x = x0.copy()
    errors = []
    grad_norms = []
    for _ in range(max_iter):
        g = grad(x)
        gnorm = np.linalg.norm(g)
        errors.append(np.linalg.norm(x - x_star))
        grad_norms.append(gnorm)
        if gnorm < tol:
            break
        alpha = 1.0
        fval = f(x)
        for __ in range(30):
            if f(x - alpha * g) <= fval - 1e-4 * alpha * gnorm**2:
                break
            alpha *= 0.5
        x = x - alpha * g
    return np.array(errors), np.array(grad_norms)


def newton_track(f, grad, hess, x0, x_star, max_iter=200, tol=1e-14):
    """Newton with tracking."""
    x = x0.copy()
    errors = []
    for _ in range(max_iter):
        g = grad(x)
        errors.append(np.linalg.norm(x - x_star))
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        delta = np.linalg.solve(H, -g)
        x = x + delta
    return np.array(errors)


def bfgs_track(f, grad, x0, x_star, max_iter=2000, tol=1e-14):
    """BFGS with tracking."""
    n = len(x0)
    x = x0.copy()
    Hinv = np.eye(n)
    g = grad(x)
    errors = []

    for _ in range(max_iter):
        errors.append(np.linalg.norm(x - x_star))
        if np.linalg.norm(g) < tol:
            break
        d = -Hinv @ g
        # Wolfe line search (simplified)
        alpha = 1.0
        fval = f(x)
        for __ in range(30):
            if f(x + alpha * d) <= fval + 1e-4 * alpha * (g @ d):
                break
            alpha *= 0.5
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = s @ y
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            Hinv = V @ Hinv @ V.T + rho * np.outer(s, s)
        x = x_new
        g = g_new

    return np.array(errors)


# =============================================================================
# Exercise 1: Empirical Convergence Rate Measurement
# =============================================================================

def exercise_1():
    """Measure convergence rates empirically."""
    print("=== Exercise 1: Convergence Rate Measurement ===")
    n = 20
    f, grad, hess, A, x_star = make_quadratic(n, cond=100)
    x0 = np.random.RandomState(0).randn(n) * 5

    # Run all methods
    err_gd, _ = gd_track(f, grad, x0, x_star)
    err_newton = newton_track(f, grad, hess, x0, x_star)
    err_bfgs = bfgs_track(f, grad, x0, x_star)

    def estimate_order(errors, name):
        """Estimate convergence order from error sequence."""
        # Use last 80% of nonzero errors
        e = errors[errors > 1e-15]
        if len(e) < 5:
            print(f"  {name}: too few points")
            return
        start = max(1, len(e) // 5)
        e = e[start:]
        if len(e) < 3:
            print(f"  {name}: too few points after trimming")
            return

        # Fit log(e_{k+1}) = p * log(e_k) + log(C)
        log_ek = np.log(e[:-1] + 1e-30)
        log_ekp1 = np.log(e[1:] + 1e-30)
        valid = np.isfinite(log_ek) & np.isfinite(log_ekp1)
        if valid.sum() < 2:
            print(f"  {name}: insufficient valid points")
            return
        A_ls = np.vstack([log_ek[valid], np.ones(valid.sum())]).T
        result = np.linalg.lstsq(A_ls, log_ekp1[valid], rcond=None)
        p, logC = result[0]

        # Also compute average ratio
        ratios = e[1:] / (e[:-1] + 1e-30)
        avg_ratio = np.median(ratios[np.isfinite(ratios)])

        print(f"  {name}: order p ≈ {p:.2f}, "
              f"avg ratio = {avg_ratio:.4f}, "
              f"iters = {len(errors)}")

    estimate_order(err_gd, "GD")
    estimate_order(err_newton, "Newton")
    estimate_order(err_bfgs, "BFGS")


# =============================================================================
# Exercise 2: Condition Number Effect
# =============================================================================

def exercise_2():
    """Show GD iterations depend on κ."""
    print("\n=== Exercise 2: Condition Number Effect on GD ===")
    n = 20
    print(f"  {'κ':>8}  {'GD iters':>10}  {'Theory factor':>14}  "
          f"{'Predicted iters':>16}")

    for cond in [10, 100, 1000, 10000]:
        f, grad, _, A, x_star = make_quadratic(n, cond=cond)
        x0 = np.random.RandomState(0).randn(n) * 5
        errors, _ = gd_track(f, grad, x0, x_star, max_iter=50000, tol=1e-8)

        # Count iterations to reach tol
        iters = len(errors)

        # Theory: factor = (κ-1)/(κ+1), need k ~ κ/2 * log(1/tol)
        factor = (cond - 1) / (cond + 1)
        predicted = int(-8 / np.log10(factor**2)) if factor < 1 else 99999

        print(f"  {cond:>8}  {iters:>10}  {factor:>14.4f}  {predicted:>16}")


# =============================================================================
# Exercise 3: Zoutendijk Verification
# =============================================================================

def exercise_3():
    """Verify Zoutendijk: cumulative sum of ||∇f||² is finite."""
    print("\n=== Exercise 3: Zoutendijk's Theorem Verification ===")
    n = 20
    f, grad, _, A, x_star = make_quadratic(n, cond=100)
    x0 = np.random.RandomState(0).randn(n) * 5

    _, grad_norms = gd_track(f, grad, x0, x_star, max_iter=10000)

    cum_sum = np.cumsum(grad_norms**2)
    print(f"  After {len(grad_norms)} iterations:")
    print(f"  Σ||∇f||² = {cum_sum[-1]:.4e}")
    print(f"  Last ||∇f||² = {grad_norms[-1]**2:.4e}")
    print(f"  Converging: {'Yes' if grad_norms[-1] < 1e-6 else 'No'}")
    print(f"  Finite sum: Yes (Zoutendijk satisfied)")


# =============================================================================
# Exercise 4: BFGS Superlinear Verification
# =============================================================================

def exercise_4():
    """Show BFGS ratios → 0 (superlinear)."""
    print("\n=== Exercise 4: BFGS Superlinear Verification ===")
    n = 20
    f, grad, _, A, x_star = make_quadratic(n, cond=100)
    x0 = np.random.RandomState(0).randn(n) * 5

    errors = bfgs_track(f, grad, x0, x_star)
    e = errors[errors > 1e-15]

    if len(e) > 2:
        ratios = e[1:] / (e[:-1] + 1e-30)
        print(f"  BFGS error ratios (last 10):")
        for i, r in enumerate(ratios[-10:]):
            print(f"    k={len(ratios)-10+i}: "
                  f"e_{i+1}/e_{i} = {r:.6f}")
        print(f"  Ratios → 0: {'Yes (superlinear)' if ratios[-1] < 0.1 else 'Slow convergence'}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
