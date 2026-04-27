"""
Day 23: Momentum Methods — Companion Code
Phase II, Week 4

Run: python momentum_methods.py
"""

import numpy as np


# =============================================================================
# Test Functions
# =============================================================================

def make_quadratic(n, cond, seed=42):
    """Create quadratic f(x) = 0.5 x^T A x - b^T x with condition number cond."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.linspace(1, cond, n)
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)
    x_star = np.linalg.solve(A, b)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad(x): return A @ x - b

    return f, grad, A, x_star, eigvals


def rosenbrock(x):
    """Rosenbrock function."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))

def rosenbrock_grad(x):
    """Rosenbrock gradient."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i+1] - x[i]**2) + 2 * (x[i] - 1)
        g[i+1] += 200 * (x[i+1] - x[i]**2)
    return g


# =============================================================================
# Exercise 1: Heavy Ball Method
# =============================================================================

def heavy_ball(f, grad, x0, alpha, beta, max_iter=5000, tol=1e-10):
    """
    Heavy ball method: x_{k+1} = x_k - α∇f(x_k) + β(x_k - x_{k-1}).
    
    Parameters:
        alpha: step size
        beta: momentum parameter
    Returns:
        x_history: list of iterates
        f_history: list of function values
    """
    x = x0.copy()
    x_prev = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x)]

    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x_new = x - alpha * g + beta * (x - x_prev)
        x_prev = x.copy()
        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x_history, f_history


def optimal_heavy_ball_params(lam_min, lam_max):
    """Compute optimal α, β for heavy ball on quadratic."""
    alpha = 4.0 / (np.sqrt(lam_max) + np.sqrt(lam_min))**2
    beta = ((np.sqrt(lam_max / lam_min) - 1) /
            (np.sqrt(lam_max / lam_min) + 1))**2
    return alpha, beta


# =============================================================================
# Exercise 2: Nesterov Accelerated Gradient
# =============================================================================

def nesterov_ag(f, grad, x0, alpha, max_iter=5000, tol=1e-10):
    """
    Nesterov accelerated gradient.
    y_k = x_k + (k-1)/(k+2) * (x_k - x_{k-1})
    x_{k+1} = y_k - α ∇f(y_k)
    
    Parameters:
        alpha: step size (typically 1/L)
    Returns:
        x_history, f_history
    """
    x = x0.copy()
    x_prev = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x)]

    for k in range(1, max_iter + 1):
        momentum = (k - 1) / (k + 2)
        y = x + momentum * (x - x_prev)
        g = grad(y)
        if np.linalg.norm(g) < tol:
            break
        x_new = y - alpha * g
        x_prev = x.copy()
        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x_history, f_history


def nesterov_strongly_convex(f, grad, x0, mu, L, max_iter=5000, tol=1e-10):
    """Nesterov for strongly convex functions with known μ and L."""
    kappa = L / mu
    momentum = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    alpha = 1.0 / L
    x = x0.copy()
    x_prev = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x)]

    for _ in range(max_iter):
        y = x + momentum * (x - x_prev)
        g = grad(y)
        if np.linalg.norm(g) < tol:
            break
        x_new = y - alpha * g
        x_prev = x.copy()
        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x_history, f_history


# =============================================================================
# Exercise 3: GD vs Heavy Ball vs Nesterov comparison
# =============================================================================

def gradient_descent(f, grad, x0, alpha=None, max_iter=5000, tol=1e-10):
    """Simple GD with optional backtracking."""
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x)]

    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        if alpha is None:
            a = 1.0
            fval = f(x)
            for __ in range(30):
                if f(x - a * g) <= fval - 1e-4 * a * np.linalg.norm(g)**2:
                    break
                a *= 0.5
        else:
            a = alpha
        x = x - a * g
        x_history.append(x.copy())
        f_history.append(f(x))

    return x_history, f_history


def exercise_3():
    """Compare GD, heavy ball, Nesterov on ill-conditioned quadratic."""
    print("=== Exercise 3: Method Comparison (κ=1000) ===")
    n = 20
    f, grad, A, x_star, eigvals = make_quadratic(n, cond=1000)
    x0 = np.random.RandomState(0).randn(n) * 5
    lam_min, lam_max = eigvals[0], eigvals[-1]
    L = lam_max

    # GD with step 1/L
    _, f_gd = gradient_descent(f, grad, x0, alpha=1.0 / L, max_iter=5000)

    # Heavy ball with optimal params
    alpha_hb, beta_hb = optimal_heavy_ball_params(lam_min, lam_max)
    _, f_hb = heavy_ball(f, grad, x0, alpha_hb, beta_hb, max_iter=5000)

    # Nesterov
    _, f_nag = nesterov_ag(f, grad, x0, alpha=1.0 / L, max_iter=5000)

    f_star = f(x_star)
    tol_target = 1e-6

    def iters_to_tol(fhist, target):
        for i, fv in enumerate(fhist):
            if fv - f_star < target:
                return i
        return len(fhist)

    print(f"  GD iterations to {tol_target} optimality gap: "
          f"{iters_to_tol(f_gd, tol_target)}")
    print(f"  Heavy ball iterations: "
          f"{iters_to_tol(f_hb, tol_target)}")
    print(f"  Nesterov iterations: "
          f"{iters_to_tol(f_nag, tol_target)}")
    print(f"  Theory: GD ~ O(κ)={1000}, "
          f"momentum ~ O(√κ)={int(np.sqrt(1000))}")


# =============================================================================
# Exercise 4: Oscillation with too much momentum
# =============================================================================

def exercise_4():
    """Show heavy ball oscillation/divergence with excessive β."""
    print("\n=== Exercise 4: Oscillation with Excessive Momentum ===")
    n = 10
    f, grad, A, x_star, eigvals = make_quadratic(n, cond=100)
    x0 = np.random.RandomState(0).randn(n)
    lam_min, lam_max = eigvals[0], eigvals[-1]

    alpha_opt, beta_opt = optimal_heavy_ball_params(lam_min, lam_max)

    beta_values = [0.0, beta_opt, 0.95, 0.99, 1.01]
    for beta in beta_values:
        _, fhist = heavy_ball(f, grad, x0, alpha_opt, beta, max_iter=500)
        f_star = f(x_star)
        final_gap = fhist[-1] - f_star
        diverged = final_gap > fhist[0] - f_star or not np.isfinite(final_gap)
        status = "DIVERGED" if diverged else f"gap={final_gap:.2e}"
        print(f"  β={beta:.3f}: iters={len(fhist)}, {status}")


# =============================================================================
# Exercise 5: Nesterov on Rosenbrock
# =============================================================================

def exercise_5():
    """Nesterov on nonconvex Rosenbrock."""
    print("\n=== Exercise 5: Nesterov on Rosenbrock ===")
    n = 4
    x0 = np.array([-1.0, 1.0, -1.0, 1.0])
    alpha = 1e-4  # Small step for Rosenbrock

    _, f_gd = gradient_descent(rosenbrock, rosenbrock_grad, x0,
                               max_iter=20000)
    _, f_nag = nesterov_ag(rosenbrock, rosenbrock_grad, x0,
                           alpha=alpha, max_iter=20000)

    print(f"  GD: {len(f_gd)} iters, final f = {f_gd[-1]:.6f}")
    print(f"  Nesterov: {len(f_nag)} iters, final f = {f_nag[-1]:.6f}")
    print(f"  Optimal: f* = 0.0")

    # Nesterov with restart
    x = x0.copy()
    x_prev = x0.copy()
    f_restart = [rosenbrock(x)]
    for k in range(1, 20001):
        momentum = (k - 1) / (k + 2)
        y = x + momentum * (x - x_prev)
        g = rosenbrock_grad(y)
        x_new = y - alpha * g
        # Restart if function increases
        if rosenbrock(x_new) > rosenbrock(x):
            x_prev = x.copy()
            x_new = x - alpha * rosenbrock_grad(x)
        else:
            x_prev = x.copy()
        x = x_new
        f_restart.append(rosenbrock(x))

    print(f"  Nesterov+restart: {len(f_restart)} iters, "
          f"final f = {f_restart[-1]:.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    exercise_3()
    exercise_4()
    exercise_5()
