"""
Day 15: Gradient Descent — Companion Code
Phase II, Week 3

Run: python gradient_descent.py
"""

import numpy as np
import time


# =============================================================================
# Test Functions
# =============================================================================

def quadratic(A, b=None):
    """f(x) = 0.5 x^T A x - b^T x. Returns (f, grad_f, x_star)."""
    if b is None:
        b = np.zeros(A.shape[0])
    x_star = np.linalg.solve(A, b)
    f_star = -0.5 * b @ x_star

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    def grad_f(x):
        return A @ x - b

    return f, grad_f, x_star, f_star


def rosenbrock():
    """Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2. Minimum at (1,1)."""
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad_f(x):
        dx = -2*(1 - x[0]) + 100 * 2*(x[1] - x[0]**2) * (-2*x[0])
        dy = 100 * 2*(x[1] - x[0]**2)
        return np.array([dx, dy])

    return f, grad_f, np.array([1.0, 1.0]), 0.0


def log_sum_exp():
    """f(x) = log(sum(exp(a_i^T x + b_i))). Convex, smooth."""
    rng = np.random.default_rng(42)
    m, n = 20, 5
    A_mat = rng.randn(m, n)
    b_vec = rng.randn(m)

    def f(x):
        z = A_mat @ x + b_vec
        return np.log(np.sum(np.exp(z - np.max(z)))) + np.max(z)

    def grad_f(x):
        z = A_mat @ x + b_vec
        z_shifted = z - np.max(z)
        softmax = np.exp(z_shifted) / np.sum(np.exp(z_shifted))
        return A_mat.T @ softmax

    return f, grad_f, None, None


# =============================================================================
# Exercise 1: Fixed Step Size GD
# =============================================================================

def gd_fixed(f, grad_f, x0, alpha, max_iter=10000, tol=1e-8):
    """Gradient descent with fixed step size."""
    x = x0.copy()
    history = {"f": [f(x)], "grad_norm": [np.linalg.norm(grad_f(x))],
               "x": [x.copy()]}

    for k in range(max_iter):
        g = grad_f(x)
        x = x - alpha * g
        fval = f(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(fval)
        history["grad_norm"].append(gnorm)
        history["x"].append(x.copy())

        if gnorm < tol:
            break

    return x, history


# =============================================================================
# Exercise 2: Backtracking Line Search
# =============================================================================

def backtracking_line_search(f, grad_f, x, d, alpha0=1.0, c1=1e-4, rho=0.5):
    """Armijo backtracking line search.
    
    Find alpha such that f(x + alpha*d) <= f(x) + c1*alpha*grad_f(x)^T d.
    """
    alpha = alpha0
    fx = f(x)
    gd = grad_f(x) @ d

    for _ in range(50):
        if f(x + alpha * d) <= fx + c1 * alpha * gd:
            return alpha
        alpha *= rho

    return alpha


def gd_backtracking(f, grad_f, x0, max_iter=10000, tol=1e-8, c1=1e-4):
    """Gradient descent with Armijo backtracking."""
    x = x0.copy()
    history = {"f": [f(x)], "grad_norm": [np.linalg.norm(grad_f(x))],
               "x": [x.copy()], "alphas": []}

    for k in range(max_iter):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            break

        d = -g  # descent direction
        alpha = backtracking_line_search(f, grad_f, x, d, c1=c1)
        x = x + alpha * d

        history["f"].append(f(x))
        history["grad_norm"].append(gnorm)
        history["x"].append(x.copy())
        history["alphas"].append(alpha)

    return x, history


# =============================================================================
# Exercise 3 & 4: Test Functions & Convergence
# =============================================================================

def test_quadratic():
    """Test GD on quadratic with various condition numbers."""
    print("=== Exercise 1 & 5: GD on Quadratic (κ scaling) ===")
    n = 20
    rng = np.random.default_rng(42)
    x0 = rng.randn(n) * 5

    for kappa in [1, 10, 100, 1000]:
        Q, _ = np.linalg.qr(rng.randn(n, n))
        eigvals = np.linspace(1, kappa, n)
        A = Q @ np.diag(eigvals) @ Q.T
        b = rng.randn(n)

        f, grad_f, x_star, f_star = quadratic(A, b)

        # Optimal fixed step
        alpha = 2.0 / (eigvals[-1] + eigvals[0])
        x_fixed, hist_fixed = gd_fixed(f, grad_f, x0, alpha, max_iter=50000)

        # Backtracking
        x_bt, hist_bt = gd_backtracking(f, grad_f, x0, max_iter=50000)

        err_fixed = np.linalg.norm(x_fixed - x_star)
        err_bt = np.linalg.norm(x_bt - x_star)

        print(f"  κ={kappa:>5}: fixed={len(hist_fixed['f'])-1:>6} iters "
              f"(err={err_fixed:.2e}), "
              f"backtrack={len(hist_bt['f'])-1:>6} iters "
              f"(err={err_bt:.2e})")


def test_rosenbrock():
    """Test GD on Rosenbrock."""
    print("\n=== Exercise 3: GD on Rosenbrock ===")
    f, grad_f, x_star, f_star = rosenbrock()
    x0 = np.array([-1.0, 1.0])

    # Backtracking (fixed step is impractical for Rosenbrock)
    x, hist = gd_backtracking(f, grad_f, x0, max_iter=50000, tol=1e-6)
    err = np.linalg.norm(x - x_star)
    print(f"  Start: {x0}, Final: [{x[0]:.6f}, {x[1]:.6f}]")
    print(f"  Iterations: {len(hist['f'])-1}, Error: {err:.2e}")
    print(f"  f(x_final) = {f(x):.2e}")


def test_log_sum_exp():
    """Test GD on log-sum-exp."""
    print("\n=== Exercise 3: GD on Log-Sum-Exp ===")
    f, grad_f, _, _ = log_sum_exp()
    x0 = np.zeros(5)

    x, hist = gd_backtracking(f, grad_f, x0, max_iter=5000, tol=1e-8)
    print(f"  Iterations: {len(hist['f'])-1}")
    print(f"  Final gradient norm: {hist['grad_norm'][-1]:.2e}")
    print(f"  f(x_final) = {f(x):.6f}")


# =============================================================================
# Exercise 5: Condition Number Scaling
# =============================================================================

def kappa_scaling_study():
    """Show GD iterations scale linearly with κ."""
    print("\n=== Exercise 5: κ Scaling Study ===")
    n = 50
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    x0 = rng.randn(n)
    b = rng.randn(n)

    print(f"  {'κ':>8} {'Iters':>8} {'Ratio':>8} {'Theory ρ':>10}")

    prev_iters = None
    for log_k in range(1, 5):
        kappa = 10 ** log_k
        eigvals = np.linspace(1, kappa, n)
        A = Q @ np.diag(eigvals) @ Q.T

        f, grad_f, x_star, f_star = quadratic(A, b)
        alpha = 2.0 / (eigvals[-1] + eigvals[0])

        x, hist = gd_fixed(f, grad_f, x0, alpha, max_iter=200000, tol=1e-8)
        iters = len(hist["f"]) - 1

        rho = (kappa - 1) / (kappa + 1)
        ratio = f"—" if prev_iters is None else f"{iters/prev_iters:.1f}×"

        print(f"  {kappa:>8} {iters:>8} {ratio:>8} {rho:>10.6f}")
        prev_iters = iters


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_quadratic()
    test_rosenbrock()
    test_log_sum_exp()
    kappa_scaling_study()
