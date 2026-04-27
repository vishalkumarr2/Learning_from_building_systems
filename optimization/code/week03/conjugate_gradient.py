"""
Day 20: Conjugate Gradient for Optimization — Companion Code
Phase II, Week 3

Run: python conjugate_gradient.py
"""

import numpy as np
import time


# =============================================================================
# Test Functions
# =============================================================================

def rosenbrock(x):
    """Extended Rosenbrock."""
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
               for i in range(len(x)-1))


def rosenbrock_grad(x):
    """Extended Rosenbrock gradient."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n-1):
        g[i] += -2*(1-x[i]) + 200*(x[i+1]-x[i]**2)*(-2*x[i])
        g[i+1] += 200*(x[i+1]-x[i]**2)
    return g


def make_quadratic(n, cond=100, seed=42):
    """Create a quadratic f = 0.5 x'Ax - b'x with given condition number."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.linspace(1, cond, n)
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)
    x_star = np.linalg.solve(A, b)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad(x): return A @ x - b

    return f, grad, A, x_star


# =============================================================================
# Wolfe Line Search
# =============================================================================

def wolfe_line_search(f, grad, x, d, c1=1e-4, c2=0.4, max_iter=25):
    """Strong Wolfe conditions with bracket-zoom."""
    alpha = 1.0
    alpha_lo, alpha_hi = 0.0, float('inf')
    f0 = f(x)
    g0 = grad(x)
    dg0 = g0 @ d

    if dg0 >= 0:
        return 1e-6

    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = f(x_new)

        if f_new > f0 + c1 * alpha * dg0:
            alpha_hi = alpha
            alpha = 0.5 * (alpha_lo + alpha_hi)
            continue

        g_new = grad(x_new)
        dg_new = g_new @ d

        if abs(dg_new) <= c2 * abs(dg0):
            return alpha

        if dg_new > 0:
            alpha_hi = alpha
        else:
            alpha_lo = alpha

        if alpha_hi < float('inf'):
            alpha = 0.5 * (alpha_lo + alpha_hi)
        else:
            alpha *= 2.0

    return alpha


# =============================================================================
# Exercise 1: Nonlinear CG
# =============================================================================

def nonlinear_cg(f, grad, x0, beta_type="PR+", max_iter=5000, tol=1e-8,
                 restart_every=None):
    """Nonlinear Conjugate Gradient.

    beta_type: "FR" (Fletcher-Reeves), "PR" (Polak-Ribière), "PR+" (recommended)
    """
    x = x0.copy().astype(float)
    g = grad(x)
    d = -g.copy()
    n = len(x0)

    if restart_every is None:
        restart_every = n  # restart every n steps

    history = {"f": [], "grad_norm": []}

    for k in range(max_iter):
        fval = f(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(fval)
        history["grad_norm"].append(gnorm)

        if gnorm < tol:
            break

        # Line search
        alpha = wolfe_line_search(f, grad, x, d)

        # Update position
        x_new = x + alpha * d
        g_new = grad(x_new)

        # Compute beta
        if beta_type == "FR":
            beta = (g_new @ g_new) / (g @ g + 1e-30)
        elif beta_type == "PR":
            beta = (g_new @ (g_new - g)) / (g @ g + 1e-30)
        elif beta_type == "PR+":
            beta = max(0.0, (g_new @ (g_new - g)) / (g @ g + 1e-30))
        else:
            raise ValueError(f"Unknown beta_type: {beta_type}")

        # Periodic restart
        if (k + 1) % restart_every == 0:
            beta = 0.0

        # Powell restart: if directions lose orthogonality
        if abs(g_new @ g) > 0.2 * (g_new @ g_new):
            beta = 0.0

        # New direction
        d = -g_new + beta * d

        x = x_new
        g = g_new

    return x, history


# =============================================================================
# Exercise 2: n-step Convergence on Quadratic
# =============================================================================

def test_quadratic_convergence():
    """CG should converge in exactly n steps on a quadratic."""
    print("=== Exercise 2: n-step Convergence on Quadratic ===")

    for n in [10, 20, 50]:
        f, grad, A, x_star = make_quadratic(n, cond=100)
        x0 = np.zeros(n)

        x, hist = nonlinear_cg(f, grad, x0, beta_type="PR+",
                                max_iter=2*n, tol=1e-14)
        err = np.linalg.norm(x - x_star)
        iters = len(hist["f"])

        print(f"  n = {n:>3}: converged in {iters:>4} iters "
              f"(theory: ≤{n}), err = {err:.2e}")


# =============================================================================
# Exercise 3: CG vs L-BFGS
# =============================================================================

def lbfgs_simple(f, grad, x0, m=10, max_iter=2000, tol=1e-8):
    """Simplified L-BFGS for comparison."""
    from collections import deque

    x = x0.copy().astype(float)
    g = grad(x)
    s_hist = deque(maxlen=m)
    y_hist = deque(maxlen=m)
    rho_hist = deque(maxlen=m)
    history = {"f": [], "grad_norm": []}

    for k in range(max_iter):
        fval = f(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(fval)
        history["grad_norm"].append(gnorm)

        if gnorm < tol:
            break

        # Two-loop recursion
        q = g.copy()
        alphas = []
        for i in range(len(s_hist)-1, -1, -1):
            a = rho_hist[i] * (s_hist[i] @ q)
            alphas.insert(0, a)
            q -= a * y_hist[i]

        if len(s_hist) > 0:
            gamma = (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
        else:
            gamma = 1.0
        r = gamma * q

        for i in range(len(s_hist)):
            b = rho_hist[i] * (y_hist[i] @ r)
            r += s_hist[i] * (alphas[i] - b)

        d = -r
        alpha = wolfe_line_search(f, grad, x, d)
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = s @ y
        if sy > 1e-10:
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0/sy)

        x = x_new
        g = g_new

    return x, history


def compare_cg_lbfgs():
    """Compare CG and L-BFGS on larger Rosenbrock."""
    print("\n=== Exercise 3: CG vs L-BFGS ===")
    n = 100
    x0 = -np.ones(n)
    x_star = np.ones(n)

    methods = {
        "CG (PR+)": lambda: nonlinear_cg(rosenbrock, rosenbrock_grad, x0,
                                           beta_type="PR+", max_iter=5000),
        "L-BFGS (m=10)": lambda: lbfgs_simple(rosenbrock, rosenbrock_grad,
                                                x0, m=10, max_iter=5000),
    }

    print(f"  n = {n} extended Rosenbrock")
    print(f"  {'Method':<18}  {'Iters':>6}  {'Error':>10}  {'Time (ms)':>10}")
    for name, method in methods.items():
        t0 = time.perf_counter()
        x, hist = method()
        elapsed = (time.perf_counter() - t0) * 1000
        err = np.linalg.norm(x - x_star)
        iters = len(hist["f"])
        print(f"  {name:<18}  {iters:>6}  {err:>10.2e}  {elapsed:>10.1f}")


# =============================================================================
# Exercise 4: Restart Strategy Comparison
# =============================================================================

def compare_restart_strategies():
    """Compare FR, PR, PR+ and periodic restart."""
    print("\n=== Exercise 4: Restart Strategy Comparison ===")
    n = 20
    x0 = -2 * np.ones(n)

    strategies = [
        ("FR", "FR", None),
        ("PR", "PR", None),
        ("PR+", "PR+", None),
        ("FR + restart/n", "FR", n),
    ]

    print(f"  n = {n} extended Rosenbrock")
    print(f"  {'Strategy':<18}  {'Iters':>6}  {'Final ||g||':>12}")
    for name, beta, restart in strategies:
        x, hist = nonlinear_cg(rosenbrock, rosenbrock_grad, x0,
                                beta_type=beta, max_iter=10000, tol=1e-8,
                                restart_every=restart)
        iters = len(hist["f"])
        gnorm = hist["grad_norm"][-1] if hist["grad_norm"] else float('inf')
        print(f"  {name:<18}  {iters:>6}  {gnorm:>12.2e}")


# =============================================================================
# Eigenvalue Clustering Effect
# =============================================================================

def test_eigenvalue_clustering():
    """Show CG converges faster with clustered eigenvalues."""
    print("\n=== Eigenvalue Clustering Effect ===")
    n = 50

    # Well-conditioned (clustered eigenvalues)
    f1, g1, _, xs1 = make_quadratic(n, cond=10)
    x, h1 = nonlinear_cg(f1, g1, np.zeros(n), beta_type="PR+",
                           max_iter=200, tol=1e-14)

    # Ill-conditioned (spread eigenvalues)
    f2, g2, _, xs2 = make_quadratic(n, cond=10000)
    x, h2 = nonlinear_cg(f2, g2, np.zeros(n), beta_type="PR+",
                           max_iter=200, tol=1e-14)

    print(f"  κ = 10:    {len(h1['f']):>4} iters")
    print(f"  κ = 10000: {len(h2['f']):>4} iters")
    print(f"  Theory: O(√κ) → {int(np.sqrt(10))} vs {int(np.sqrt(10000))} steps")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_quadratic_convergence()
    compare_cg_lbfgs()
    compare_restart_strategies()
    test_eigenvalue_clustering()
