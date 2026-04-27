"""
Day 21: Optimizer Shootout — Benchmarking Framework
Phase II, Week 3

Run: python optimizer_shootout.py
"""

import numpy as np
import time
from collections import deque


# =============================================================================
# Test Functions: (f, grad, hessian_or_None, x_star, name, dim)
# =============================================================================

def rosenbrock_2d():
    """Rosenbrock 2D — narrow curved valley."""
    def f(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def grad(x):
        return np.array([
            -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
            200*(x[1]-x[0]**2)])
    def hess(x):
        return np.array([
            [2+1200*x[0]**2-400*x[1], -400*x[0]],
            [-400*x[0], 200]])
    return f, grad, hess, np.array([1.0, 1.0]), "Rosenbrock 2D", 2


def beale():
    """Beale function — flat regions."""
    def f(x):
        t1 = (1.5 - x[0] + x[0]*x[1])**2
        t2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        t3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return t1 + t2 + t3
    def grad(x):
        g = np.zeros(2)
        r1 = 1.5 - x[0] + x[0]*x[1]
        r2 = 2.25 - x[0] + x[0]*x[1]**2
        r3 = 2.625 - x[0] + x[0]*x[1]**3
        g[0] = 2*r1*(-1+x[1]) + 2*r2*(-1+x[1]**2) + 2*r3*(-1+x[1]**3)
        g[1] = 2*r1*x[0] + 2*r2*2*x[0]*x[1] + 2*r3*3*x[0]*x[1]**2
        return g
    return f, grad, None, np.array([3.0, 0.5]), "Beale 2D", 2


def rosenbrock_nd(n=20):
    """Extended Rosenbrock nD."""
    def f(x):
        return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
                   for i in range(n-1))
    def grad(x):
        g = np.zeros(n)
        for i in range(n-1):
            g[i] += -2*(1-x[i]) + 200*(x[i+1]-x[i]**2)*(-2*x[i])
            g[i+1] += 200*(x[i+1]-x[i]**2)
        return g
    return f, grad, None, np.ones(n), f"Rosenbrock {n}D", n


def quadratic_50d():
    """Quadratic 50D with κ = 1000."""
    rng = np.random.RandomState(42)
    n = 50
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.linspace(1, 1000, n)
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)
    x_star = np.linalg.solve(A, b)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad(x): return A @ x - b
    def hess(x): return A

    return f, grad, hess, x_star, "Quadratic 50D (κ=1000)", n


def styblinski_tang_5d():
    """Styblinski-Tang 5D — many local minima."""
    n = 5
    def f(x):
        return 0.5 * sum(x[i]**4 - 16*x[i]**2 + 5*x[i] for i in range(n))
    def grad(x):
        return np.array([2*x[i]**3 - 16*x[i] + 2.5 for i in range(n)])
    x_star = np.full(n, -2.903534)
    return f, grad, None, x_star, "Styblinski-Tang 5D", n


# =============================================================================
# Optimizers
# =============================================================================

def wolfe_ls(f, grad, x, d, c1=1e-4, c2=0.9, max_iter=20):
    """Wolfe line search."""
    alpha = 1.0
    f0 = f(x)
    g0 = grad(x)
    dg0 = g0 @ d
    if dg0 >= 0:
        return 1e-4
    n_eval = 0
    for _ in range(max_iter):
        n_eval += 1
        fn = f(x + alpha * d)
        if fn > f0 + c1 * alpha * dg0:
            alpha *= 0.5
            continue
        gn = grad(x + alpha * d)
        if abs(gn @ d) <= c2 * abs(dg0):
            return alpha
        if gn @ d > 0:
            alpha *= 0.5
        else:
            alpha *= 2.0
    return alpha


def opt_gd(f, grad, hess, x0, max_iter=10000, tol=1e-8):
    """Gradient descent with backtracking."""
    x = x0.copy()
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return x, k+1
        alpha = 1.0
        fval = f(x)
        for _ in range(30):
            if f(x - alpha*g) <= fval - 1e-4*alpha*(g@g):
                break
            alpha *= 0.5
        x = x - alpha * g
    return x, max_iter


def opt_newton_ls(f, grad, hess, x0, max_iter=500, tol=1e-8):
    """Newton with line search (requires Hessian)."""
    if hess is None:
        return x0, -1  # Can't run
    x = x0.copy()
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return x, k+1
        H = hess(x)
        eigvals = np.linalg.eigvalsh(H)
        mu = max(0, -np.min(eigvals) + 1e-6)
        delta = np.linalg.solve(H + mu*np.eye(len(x)), -g)
        alpha = wolfe_ls(f, grad, x, delta)
        x = x + alpha * delta
    return x, max_iter


def opt_trust_region(f, grad, hess, x0, max_iter=500, tol=1e-8):
    """Trust region with dogleg (requires Hessian)."""
    if hess is None:
        return x0, -1
    x = x0.copy()
    delta_r = 1.0
    for k in range(max_iter):
        g = grad(x)
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            return x, k+1
        H = hess(x)
        eigvals = np.linalg.eigvalsh(H)
        if np.min(eigvals) > 1e-10:
            pn = np.linalg.solve(H, -g)
        else:
            mu = -np.min(eigvals) + 1e-4
            pn = np.linalg.solve(H + mu*np.eye(len(x)), -g)

        if np.linalg.norm(pn) <= delta_r:
            p = pn
        else:
            p = -delta_r / gnorm * g

        fval = f(x)
        fn = f(x + p)
        pred = -(g@p + 0.5*p@H@p)
        if pred > 1e-16:
            rho = (fval - fn) / pred
        else:
            rho = 0.0
        if rho > 0.01:
            x = x + p
        if rho < 0.25:
            delta_r *= 0.25
        elif rho > 0.75:
            delta_r = min(2*delta_r, 100)
    return x, max_iter


def opt_bfgs(f, grad, hess, x0, max_iter=2000, tol=1e-8):
    """BFGS with Wolfe line search."""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)
    g = grad(x)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return x, k+1
        d = -H @ g
        alpha = wolfe_ls(f, grad, x, d)
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = s @ y
        if sy > 1e-10:
            rho = 1.0/sy
            I = np.eye(n)
            V = I - rho*np.outer(s, y)
            H = V @ H @ V.T + rho*np.outer(s, s)
        x = x_new
        g = g_new
    return x, max_iter


def opt_lbfgs(f, grad, hess, x0, m=10, max_iter=2000, tol=1e-8):
    """L-BFGS with two-loop recursion."""
    x = x0.copy()
    g = grad(x)
    s_h = deque(maxlen=m)
    y_h = deque(maxlen=m)
    r_h = deque(maxlen=m)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return x, k+1
        q = g.copy()
        als = []
        for i in range(len(s_h)-1, -1, -1):
            a = r_h[i] * (s_h[i] @ q)
            als.insert(0, a)
            q -= a * y_h[i]
        gam = (s_h[-1]@y_h[-1])/(y_h[-1]@y_h[-1]) if len(s_h) > 0 else 1.0
        r = gam * q
        for i in range(len(s_h)):
            b = r_h[i] * (y_h[i] @ r)
            r += s_h[i] * (als[i] - b)
        d = -r
        alpha = wolfe_ls(f, grad, x, d)
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = s @ y
        if sy > 1e-10:
            s_h.append(s)
            y_h.append(y)
            r_h.append(1.0/sy)
        x = x_new
        g = g_new
    return x, max_iter


def opt_cg(f, grad, hess, x0, max_iter=5000, tol=1e-8):
    """Nonlinear CG with PR+."""
    x = x0.copy()
    g = grad(x)
    d = -g.copy()
    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            return x, k+1
        alpha = wolfe_ls(f, grad, x, d, c2=0.4)
        x_new = x + alpha * d
        g_new = grad(x_new)
        gg = g @ g
        beta = max(0.0, g_new@(g_new - g) / (gg + 1e-30))
        if abs(g_new @ g) > 0.2 * (g_new @ g_new):
            beta = 0.0
        d = -g_new + beta * d
        x = x_new
        g = g_new
    return x, max_iter


# =============================================================================
# Benchmarking Framework
# =============================================================================

def run_shootout():
    """Run all optimizers on all test functions."""
    print("=" * 80)
    print("OPTIMIZER SHOOTOUT — Week 3 Benchmark")
    print("=" * 80)

    problems = [
        rosenbrock_2d(),
        beale(),
        rosenbrock_nd(20),
        quadratic_50d(),
        styblinski_tang_5d(),
    ]

    methods = [
        ("GD", opt_gd),
        ("Newton+LS", opt_newton_ls),
        ("Trust Reg", opt_trust_region),
        ("BFGS", opt_bfgs),
        ("L-BFGS", opt_lbfgs),
        ("CG (PR+)", opt_cg),
    ]

    for f_func, grad_func, hess_func, x_star, name, ndim in problems:
        print(f"\n{'─' * 70}")
        print(f"Problem: {name}")
        print(f"{'─' * 70}")
        print(f"  {'Method':<12}  {'Iters':>7}  {'Error':>10}  "
              f"{'||g||':>10}  {'Time(ms)':>9}")

        # Starting point
        rng = np.random.RandomState(123)
        x0 = rng.uniform(-2, 2, ndim)

        for mname, mfunc in methods:
            t0 = time.perf_counter()
            try:
                x_final, iters = mfunc(f_func, grad_func, hess_func,
                                        x0.copy())
                elapsed = (time.perf_counter() - t0) * 1000

                if iters < 0:
                    print(f"  {mname:<12}  {'N/A (no H)':>7}")
                    continue

                err = np.linalg.norm(x_final - x_star)
                gnorm = np.linalg.norm(grad_func(x_final))

                print(f"  {mname:<12}  {iters:>7}  {err:>10.2e}  "
                      f"{gnorm:>10.2e}  {elapsed:>9.1f}")
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"  {mname:<12}  {'FAIL':>7}  {str(e)[:30]}")


def print_summary():
    """Print decision guide."""
    print(f"\n{'=' * 70}")
    print("DECISION GUIDE")
    print(f"{'=' * 70}")
    print("""
  Small n + Hessian available  → Trust Region (Dogleg) or Newton+LS
  Medium n + no Hessian        → BFGS
  Large n + gradients only     → L-BFGS (m=10)
  Very large n + minimal mem   → CG (PR+)
  Non-smooth or noisy          → GD with momentum (Week 4)

  In Ceres:
    Default               → TRUST_REGION + LEVENBERG_MARQUARDT
    Large sparse           → TRUST_REGION + ITERATIVE_SCHUR
    Smooth trajectory opt  → LINE_SEARCH + LBFGS
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_shootout()
    print_summary()
