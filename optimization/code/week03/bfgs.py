"""
Day 19: Quasi-Newton — BFGS and L-BFGS — Companion Code
Phase II, Week 3

Run: python bfgs.py
"""

import numpy as np
from collections import deque
import time


# =============================================================================
# Test Functions
# =============================================================================

def rosenbrock(x):
    """Rosenbrock function value."""
    return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
               for i in range(len(x)-1))


def rosenbrock_grad(x):
    """Rosenbrock gradient."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n-1):
        g[i] += -2*(1-x[i]) + 200*(x[i+1]-x[i]**2)*(-2*x[i])
        g[i+1] += 200*(x[i+1]-x[i]**2)
    return g


# =============================================================================
# Wolfe Line Search
# =============================================================================

def wolfe_line_search(f, grad, x, d, c1=1e-4, c2=0.9, max_iter=25):
    """Strong Wolfe line search.

    Returns alpha satisfying:
      f(x + alpha*d) <= f(x) + c1*alpha*g'd   (Armijo)
      |g(x+alpha*d)'d| <= c2*|g(x)'d|          (curvature)
    """
    alpha = 1.0
    alpha_lo, alpha_hi = 0.0, float('inf')
    f0 = f(x)
    g0 = grad(x)
    dg0 = g0 @ d

    if dg0 >= 0:
        return 1e-4  # Not a descent direction, use tiny step

    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = f(x_new)

        # Armijo check
        if f_new > f0 + c1 * alpha * dg0:
            alpha_hi = alpha
            alpha = 0.5 * (alpha_lo + alpha_hi)
            continue

        g_new = grad(x_new)
        dg_new = g_new @ d

        # Strong Wolfe curvature check
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
# Exercise 1: Full BFGS
# =============================================================================

def bfgs(f, grad, x0, max_iter=1000, tol=1e-8):
    """Full BFGS with inverse Hessian update.

    Stores the full n×n inverse Hessian approximation.
    """
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)  # Initial inverse Hessian approximation
    g = grad(x)

    history = {"f": [], "grad_norm": [], "n_evals": 0}

    for k in range(max_iter):
        fval = f(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(fval)
        history["grad_norm"].append(gnorm)

        if gnorm < tol:
            break

        # Search direction
        d = -H @ g

        # Wolfe line search
        alpha = wolfe_line_search(f, grad, x, d)
        history["n_evals"] += 1

        # Step
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g

        # BFGS inverse Hessian update
        sy = s @ y
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x = x_new
        g = g_new

    return x, history


# =============================================================================
# Exercise 2: L-BFGS Two-Loop Recursion
# =============================================================================

def lbfgs(f, grad, x0, m=10, max_iter=1000, tol=1e-8):
    """L-BFGS with two-loop recursion.

    Stores only the last m (s, y) pairs — O(mn) memory.
    """
    x = x0.copy().astype(float)
    g = grad(x)

    s_history = deque(maxlen=m)
    y_history = deque(maxlen=m)
    rho_history = deque(maxlen=m)

    history = {"f": [], "grad_norm": [], "n_evals": 0}

    for k in range(max_iter):
        fval = f(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(fval)
        history["grad_norm"].append(gnorm)

        if gnorm < tol:
            break

        # Two-loop recursion to compute H_k @ g
        q = g.copy()
        alphas = []

        # First loop (backward)
        for i in range(len(s_history) - 1, -1, -1):
            alpha_i = rho_history[i] * (s_history[i] @ q)
            alphas.insert(0, alpha_i)
            q = q - alpha_i * y_history[i]

        # Initial scaling
        if len(s_history) > 0:
            s_last = s_history[-1]
            y_last = y_history[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last)
        else:
            gamma = 1.0
        r = gamma * q

        # Second loop (forward)
        for i in range(len(s_history)):
            beta = rho_history[i] * (y_history[i] @ r)
            r = r + s_history[i] * (alphas[i] - beta)

        d = -r  # Search direction

        # Wolfe line search
        alpha = wolfe_line_search(f, grad, x, d)
        history["n_evals"] += 1

        # Update
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g

        sy = s @ y
        if sy > 1e-10:
            s_history.append(s.copy())
            y_history.append(y.copy())
            rho_history.append(1.0 / sy)

        x = x_new
        g = g_new

    return x, history


# =============================================================================
# Exercise 3: Comparison Shootout
# =============================================================================

def gradient_descent(f, grad, x0, max_iter=50000, tol=1e-8):
    """Simple GD with backtracking."""
    x = x0.copy()
    history = {"f": [], "grad_norm": []}

    for k in range(max_iter):
        fval = f(x)
        g = grad(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(fval)
        history["grad_norm"].append(gnorm)

        if gnorm < tol:
            break

        # Backtracking
        alpha = 1.0
        d = -g
        for _ in range(30):
            if f(x + alpha * d) <= fval + 1e-4 * alpha * (g @ d):
                break
            alpha *= 0.5

        x = x + alpha * d

    return x, history


def shootout():
    """Compare GD, BFGS, L-BFGS on 2D Rosenbrock."""
    print("=== Exercise 3: Method Shootout (2D Rosenbrock) ===")
    x0 = np.array([-1.0, 1.0])
    x_star = np.array([1.0, 1.0])

    methods = {
        "GD": lambda: gradient_descent(rosenbrock, rosenbrock_grad, x0),
        "BFGS": lambda: bfgs(rosenbrock, rosenbrock_grad, x0),
        "L-BFGS (m=10)": lambda: lbfgs(rosenbrock, rosenbrock_grad, x0, m=10),
        "L-BFGS (m=3)": lambda: lbfgs(rosenbrock, rosenbrock_grad, x0, m=3),
    }

    print(f"  {'Method':<15}  {'Iters':>6}  {'Error':>10}  {'Time (ms)':>10}")
    for name, method in methods.items():
        t0 = time.perf_counter()
        x, hist = method()
        elapsed = (time.perf_counter() - t0) * 1000
        err = np.linalg.norm(x - x_star)
        iters = len(hist["f"])
        print(f"  {name:<15}  {iters:>6}  {err:>10.2e}  {elapsed:>10.2f}")


# =============================================================================
# Exercise 4: High-Dimensional L-BFGS
# =============================================================================

def high_dimensional_test():
    """Test L-BFGS on n=1000 extended Rosenbrock."""
    print("\n=== Exercise 4: High-Dimensional (n=1000) ===")
    n = 1000
    x0 = -np.ones(n)
    x_star = np.ones(n)

    # L-BFGS should handle this easily
    t0 = time.perf_counter()
    x_lbfgs, hist_lbfgs = lbfgs(rosenbrock, rosenbrock_grad, x0, m=10,
                                  max_iter=2000, tol=1e-6)
    t_lbfgs = (time.perf_counter() - t0) * 1000
    err_lbfgs = np.linalg.norm(x_lbfgs - x_star)

    print(f"  L-BFGS (m=10): {len(hist_lbfgs['f']):>5} iters, "
          f"err = {err_lbfgs:.2e}, time = {t_lbfgs:.0f} ms")
    print(f"  Memory: L-BFGS = {2*10*n*8 / 1024:.0f} KB "
          f"vs BFGS = {n*n*8 / (1024*1024):.0f} MB")


# =============================================================================
# Memory comparison
# =============================================================================

def memory_comparison():
    """Show the memory advantage of L-BFGS."""
    print("\n=== Memory Comparison ===")
    print(f"  {'n':>8}  {'BFGS (MB)':>12}  {'L-BFGS m=10 (KB)':>18}")
    for n in [100, 1000, 10000, 100000]:
        bfgs_mb = n * n * 8 / (1024 * 1024)
        lbfgs_kb = 2 * 10 * n * 8 / 1024
        print(f"  {n:>8}  {bfgs_mb:>12.1f}  {lbfgs_kb:>18.0f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    shootout()
    high_dimensional_test()
    memory_comparison()
