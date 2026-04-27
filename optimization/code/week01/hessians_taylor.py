"""
Day 4: Hessians and Taylor Expansion — Companion Code
Phase I, Week 1

Run: python hessians_taylor.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Exercise 1: Hessian of Rosenbrock (analytic)
# =============================================================================

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock."""
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])


def rosenbrock_hessian(x: np.ndarray) -> np.ndarray:
    """Analytic Hessian of Rosenbrock."""
    d2fdx2 = 2 - 400 * (x[1] - 3 * x[0]**2)
    d2fdxdy = -400 * x[0]
    d2fdy2 = 200.0
    return np.array([[d2fdx2, d2fdxdy],
                     [d2fdxdy, d2fdy2]])


# =============================================================================
# Exercise 2: Numerical Hessian via finite differences of gradient
# =============================================================================

def numerical_hessian(grad_f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Compute Hessian via central differences of the gradient."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        g_plus = grad_f(x + h * e_i)
        g_minus = grad_f(x - h * e_i)
        H[:, i] = (g_plus - g_minus) / (2 * h)
    # Symmetrize
    return 0.5 * (H + H.T)


# =============================================================================
# Exercise 3: Classify critical points
# =============================================================================

def find_and_classify_critical_points():
    """f(x,y) = x⁴ + y⁴ - 2x² - 2y²"""
    # Critical points: ∇f = 0
    # ∂f/∂x = 4x³ - 4x = 4x(x²-1) = 0 → x = 0, ±1
    # ∂f/∂y = 4y³ - 4y = 4y(y²-1) = 0 → y = 0, ±1
    # 9 critical points

    def f(x):
        return x[0]**4 + x[1]**4 - 2*x[0]**2 - 2*x[1]**2

    def grad_f(x):
        return np.array([4*x[0]**3 - 4*x[0], 4*x[1]**3 - 4*x[1]])

    def hessian_f(x):
        return np.array([[12*x[0]**2 - 4, 0],
                         [0, 12*x[1]**2 - 4]])

    critical_points = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]

    print("=== Critical Point Classification: f(x,y) = x⁴ + y⁴ - 2x² - 2y² ===")
    print(f"{'Point':<12} {'f(x)':<10} {'Eigenvalues':<20} {'Type'}")
    print("-" * 60)

    for (cx, cy) in critical_points:
        pt = np.array([cx, cy], dtype=float)
        fval = f(pt)
        H = hessian_f(pt)
        eigvals = np.linalg.eigvalsh(H)

        if np.all(eigvals > 1e-10):
            ptype = "Local minimum"
        elif np.all(eigvals < -1e-10):
            ptype = "Local maximum"
        elif np.all(eigvals >= -1e-10):
            ptype = "Degenerate (PSD)"
        else:
            ptype = "Saddle point"

        print(f"({cx:+d}, {cy:+d})   {fval:>8.1f}   [{eigvals[0]:>6.1f}, {eigvals[1]:>6.1f}]     {ptype}")


# =============================================================================
# Exercise 4: Taylor Approximation Quality
# =============================================================================

def taylor_approximation_demo():
    """Compare actual Rosenbrock vs quadratic Taylor model."""
    x0 = np.array([0.0, 0.0])
    f0 = rosenbrock(x0)
    g0 = rosenbrock_grad(x0)
    H0 = rosenbrock_hessian(x0)

    # Quadratic model: q(δ) = f(x0) + g^T δ + 0.5 δ^T H δ
    def quadratic_model(delta):
        return f0 + g0 @ delta + 0.5 * delta @ H0 @ delta

    # Evaluate along a line direction
    direction = np.array([1.0, 1.0]) / np.sqrt(2)
    t_vals = np.linspace(-2, 2, 200)

    f_actual = [rosenbrock(x0 + t * direction) for t in t_vals]
    f_quad = [quadratic_model(t * direction) for t in t_vals]

    print("\n=== Taylor Approximation Quality ===")
    print(f"Expansion point: x0 = {x0}")
    print(f"f(x0) = {f0}, ∇f(x0) = {g0}, H(x0) =\n{H0}")
    print("\nDistance from x0 | Actual f | Quadratic model | Error")
    print("-" * 60)
    for t in [0.1, 0.5, 1.0, 2.0]:
        delta = t * direction
        fa = rosenbrock(x0 + delta)
        fq = quadratic_model(delta)
        print(f"  t={t:<4.1f}          | {fa:>10.4f} | {fq:>15.4f} | {abs(fa-fq):>8.4f}")


# =============================================================================
# Exercise 5: Newton Step Visualization
# =============================================================================

def newton_step_demo():
    """Show Newton step on Rosenbrock from a nearby point."""
    x0 = np.array([0.5, 0.5])
    g = rosenbrock_grad(x0)
    H = rosenbrock_hessian(x0)

    # Newton step: δ = -H⁻¹ ∇f
    delta = -np.linalg.solve(H, g)
    x1 = x0 + delta

    print("\n=== Newton Step Demo ===")
    print(f"x0 = {x0}, f(x0) = {rosenbrock(x0):.4f}")
    print(f"∇f(x0) = {g}")
    print(f"H(x0) eigenvalues: {np.linalg.eigvalsh(H)}")
    print(f"Newton step δ = {delta}")
    print(f"x1 = x0 + δ = {x1}, f(x1) = {rosenbrock(x1):.4f}")
    print(f"Minimum at (1,1), f = 0")


if __name__ == "__main__":
    # Verify Hessian
    print("=== Hessian Verification at (1,1) ===")
    x_min = np.array([1.0, 1.0])
    H_analytic = rosenbrock_hessian(x_min)
    H_numerical = numerical_hessian(rosenbrock_grad, x_min)
    print(f"Analytic H:\n{H_analytic}")
    print(f"Numerical H:\n{np.round(H_numerical, 2)}")
    print(f"Error: {np.linalg.norm(H_analytic - H_numerical):.2e}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(H_analytic)} → PD = minimum ✓")

    find_and_classify_critical_points()
    taylor_approximation_demo()
    newton_step_demo()
