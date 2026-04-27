"""
Day 16: Line Search Methods — Companion Code
Phase II, Week 3

Run: python line_search.py
"""

import numpy as np


# =============================================================================
# Test Functions
# =============================================================================

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) + 100 * 2*(x[1] - x[0]**2) * (-2*x[0])
    dy = 100 * 2*(x[1] - x[0]**2)
    return np.array([dx, dy])


def make_quadratic(kappa=100, n=10, seed=42):
    """Create quadratic with specified condition number."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.linspace(1, kappa, n)
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)

    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad_f(x): return A @ x - b

    x_star = np.linalg.solve(A, b)
    return f, grad_f, A, x_star


# =============================================================================
# Exercise 1: Backtracking Line Search (Armijo)
# =============================================================================

def backtracking_armijo(f, grad_f, x, d, alpha0=1.0, c1=1e-4, rho=0.5,
                        max_bt=50):
    """Armijo backtracking line search.

    Returns (alpha, num_evals).
    """
    alpha = alpha0
    fx = f(x)
    gd = grad_f(x) @ d
    evals = 0

    for _ in range(max_bt):
        evals += 1
        if f(x + alpha * d) <= fx + c1 * alpha * gd:
            return alpha, evals
        alpha *= rho

    return alpha, evals


# =============================================================================
# Exercise 2: Wolfe Line Search (Zoom Algorithm)
# =============================================================================

def wolfe_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9, alpha_max=10.0,
                      max_iter=25):
    """Strong Wolfe line search via zoom algorithm (Nocedal & Wright Alg 3.5/3.6).

    Returns (alpha, num_fevals, num_gevals).
    """
    phi0 = f(x)
    dphi0 = grad_f(x) @ d
    fevals, gevals = 1, 1

    def phi(a):
        return f(x + a * d)

    def dphi(a):
        return grad_f(x + a * d) @ d

    def zoom(alo, ahi, phi_lo, phi_hi, dphi_lo):
        """Zoom phase: bisect [alo, ahi] until strong Wolfe holds."""
        nonlocal fevals, gevals
        for _ in range(20):
            # Bisection (could use cubic interpolation for speed)
            aj = 0.5 * (alo + ahi)
            phi_j = phi(aj)
            fevals += 1

            if phi_j > phi0 + c1 * aj * dphi0 or phi_j >= phi_lo:
                ahi = aj
                phi_hi = phi_j
            else:
                dphi_j = dphi(aj)
                gevals += 1
                if abs(dphi_j) <= -c2 * dphi0:
                    return aj  # Strong Wolfe satisfied
                if dphi_j * (ahi - alo) >= 0:
                    ahi = alo
                    phi_hi = phi_lo
                alo = aj
                phi_lo = phi_j
                dphi_lo = dphi_j

        return 0.5 * (alo + ahi)

    # Phase 1: bracket
    alpha_prev = 0.0
    alpha_curr = 1.0
    phi_prev = phi0
    dphi_prev = dphi0
    first = True

    for i in range(max_iter):
        phi_curr = phi(alpha_curr)
        fevals += 1

        if phi_curr > phi0 + c1 * alpha_curr * dphi0 or (phi_curr >= phi_prev and not first):
            return zoom(alpha_prev, alpha_curr, phi_prev, phi_curr, dphi_prev), fevals, gevals

        dphi_curr = dphi(alpha_curr)
        gevals += 1

        if abs(dphi_curr) <= -c2 * dphi0:
            return alpha_curr, fevals, gevals

        if dphi_curr >= 0:
            return zoom(alpha_curr, alpha_prev, phi_curr, phi_prev, dphi_curr), fevals, gevals

        alpha_prev = alpha_curr
        phi_prev = phi_curr
        dphi_prev = dphi_curr
        alpha_curr = min(2 * alpha_curr, alpha_max)
        first = False

    return alpha_curr, fevals, gevals


# =============================================================================
# GD with Various Line Searches
# =============================================================================

def gd_with_linesearch(f, grad_f, x0, ls_method="armijo", max_iter=10000,
                       tol=1e-8, **ls_kwargs):
    """GD using chosen line search method."""
    x = x0.copy()
    history = {"f": [f(x)], "grad_norm": [], "total_fevals": 0}

    for k in range(max_iter):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        history["grad_norm"].append(gnorm)
        if gnorm < tol:
            break

        d = -g

        if ls_method == "armijo":
            alpha, evals = backtracking_armijo(f, grad_f, x, d, **ls_kwargs)
            history["total_fevals"] += evals
        elif ls_method == "wolfe":
            alpha, fe, ge = wolfe_line_search(f, grad_f, x, d, **ls_kwargs)
            history["total_fevals"] += fe
        elif ls_method == "fixed":
            alpha = ls_kwargs.get("alpha", 0.001)
            history["total_fevals"] += 1
        else:
            raise ValueError(f"Unknown: {ls_method}")

        x = x + alpha * d
        history["f"].append(f(x))

    return x, history


# =============================================================================
# Exercise 3: Comparison on Rosenbrock
# =============================================================================

def compare_line_searches():
    """Compare fixed, Armijo, Wolfe on Rosenbrock."""
    print("=== Exercise 3: Line Search Comparison on Rosenbrock ===")
    x0 = np.array([-1.0, 1.0])
    x_star = np.array([1.0, 1.0])
    max_iter = 30000

    methods = [
        ("Fixed (α=0.001)", "fixed", {"alpha": 0.001}),
        ("Armijo (ρ=0.5)", "armijo", {"rho": 0.5}),
        ("Armijo (ρ=0.8)", "armijo", {"rho": 0.8}),
        ("Wolfe", "wolfe", {}),
    ]

    for name, method, kwargs in methods:
        x, hist = gd_with_linesearch(
            rosenbrock, rosenbrock_grad, x0, method, max_iter=max_iter,
            tol=1e-6, **kwargs
        )
        iters = len(hist["f"]) - 1
        err = np.linalg.norm(x - x_star)
        fval = rosenbrock(x)
        print(f"  {name:25s}: {iters:>6} iters, f={fval:.2e}, "
              f"err={err:.2e}, fevals={hist['total_fevals']}")


# =============================================================================
# Exercise 4: Pathological Case
# =============================================================================

def pathological_case():
    """Show where Armijo gives overly short steps."""
    print("\n=== Exercise 4: Pathological Case ===")
    # Long narrow valley: f = 0.01*x^2 + 100*y^2
    def f(x): return 0.01 * x[0]**2 + 100 * x[1]**2
    def grad_f(x): return np.array([0.02 * x[0], 200 * x[1]])

    x0 = np.array([100.0, 1.0])
    x_star = np.array([0.0, 0.0])

    x_arm, h_arm = gd_with_linesearch(
        f, grad_f, x0, "armijo", max_iter=50000, tol=1e-8
    )
    x_wolfe, h_wolfe = gd_with_linesearch(
        f, grad_f, x0, "wolfe", max_iter=50000, tol=1e-8
    )

    print(f"  Armijo:  {len(h_arm['f'])-1:>6} iters, f={f(x_arm):.2e}")
    print(f"  Wolfe:   {len(h_wolfe['f'])-1:>6} iters, f={f(x_wolfe):.2e}")
    print(f"  (Wolfe enforces curvature → longer steps along narrow direction)")


# =============================================================================
# Exercise 5: 1D Slice Visualization (text-based)
# =============================================================================

def visualize_1d_slice():
    """Show Armijo and Wolfe acceptance regions along search direction."""
    print("\n=== Exercise 5: 1D Acceptance Regions ===")
    f, grad_f, A, x_star = make_quadratic(kappa=10, n=2, seed=42)
    x0 = np.array([5.0, -3.0])
    d = -grad_f(x0)  # search direction

    # 1D slice
    fx0 = f(x0)
    dphi0 = grad_f(x0) @ d
    c1 = 1e-4
    c2 = 0.9

    print(f"  x0 = {x0}, d = [{d[0]:.3f}, {d[1]:.3f}]")
    print(f"  φ(0) = {fx0:.4f}, φ'(0) = {dphi0:.4f}")

    alphas = np.linspace(0, 0.5, 21)
    print(f"\n  {'α':>6}  {'φ(α)':>10}  {'Armijo':>8}  {'Curv':>8}  {'Wolfe':>8}")
    print(f"  {'-'*50}")

    for a in alphas:
        xa = x0 + a * d
        phi_a = f(xa)
        dphi_a = grad_f(xa) @ d

        armijo = phi_a <= fx0 + c1 * a * dphi0
        curvature = dphi_a >= c2 * dphi0
        wolfe = armijo and curvature

        print(f"  {a:6.3f}  {phi_a:10.4f}  {'✓' if armijo else '✗':>8}  "
              f"{'✓' if curvature else '✗':>8}  {'✓' if wolfe else '✗':>8}")


# =============================================================================
# Parameter Sensitivity
# =============================================================================

def parameter_sensitivity():
    """Effect of c1 and rho on backtracking."""
    print("\n=== Parameter Sensitivity ===")
    f, grad_f, _, x_star = make_quadratic(kappa=100, n=20, seed=42)
    x0 = np.zeros(20) + 5.0

    print(f"  {'c1':>8}  {'rho':>6}  {'Iters':>8}  {'Error':>10}")
    for c1 in [1e-1, 1e-2, 1e-4, 1e-8]:
        for rho_val in [0.3, 0.5, 0.8]:
            x, hist = gd_with_linesearch(
                f, grad_f, x0, "armijo", max_iter=50000, tol=1e-8,
                c1=c1, rho=rho_val
            )
            err = np.linalg.norm(x - x_star)
            iters = len(hist["f"]) - 1
            print(f"  {c1:>8.0e}  {rho_val:>6.1f}  {iters:>8}  {err:>10.2e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    compare_line_searches()
    pathological_case()
    visualize_1d_slice()
    parameter_sensitivity()
