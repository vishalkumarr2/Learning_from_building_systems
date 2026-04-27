"""
Day 3: Gradients and Jacobians — Companion Code
Phase I, Week 1

Run: python gradients_jacobians.py
"""

import numpy as np
import sympy as sp


# =============================================================================
# Exercise 1: Symbolic Gradient Verification with SymPy
# =============================================================================

def symbolic_gradient_demo():
    """Compute gradients symbolically and verify."""
    x, y = sp.symbols("x y")

    # f(x,y) = x²y + sin(xy)
    f = x**2 * y + sp.sin(x * y)
    grad_f = [sp.diff(f, var) for var in [x, y]]

    print("=== Symbolic Gradients ===")
    print(f"f(x,y) = {f}")
    print(f"∂f/∂x = {grad_f[0]}")
    print(f"∂f/∂y = {grad_f[1]}")

    # Evaluate at a point
    point = {x: 1.0, y: 2.0}
    grad_at_point = [g.evalf(subs=point) for g in grad_f]
    print(f"\n∇f(1,2) = {[float(g) for g in grad_at_point]}")


# =============================================================================
# Exercise 2: Jacobian of Vector Function
# =============================================================================

def symbolic_jacobian_demo():
    """Compute Jacobian of f: R³ → R³."""
    x, y, z = sp.symbols("x y z")

    f = sp.Matrix([x**2 + y, y * z, x * z**2])
    vars = sp.Matrix([x, y, z])
    J = f.jacobian(vars)

    print("\n=== Symbolic Jacobian ===")
    print(f"f(x,y,z) = {f.T}")
    print(f"J =\n{J}")

    # Evaluate at point
    point = {x: 1.0, y: 2.0, z: 3.0}
    J_at_point = J.evalf(subs=point)
    print(f"\nJ(1,2,3) =\n{np.array(J_at_point, dtype=float)}")


# =============================================================================
# Exercise 3: Numerical Gradient via Finite Differences
# =============================================================================

def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Central difference gradient: ∂f/∂xᵢ ≈ [f(x+h·eᵢ) - f(x-h·eᵢ)] / 2h
    """
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        grad[i] = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)
    return grad


def numerical_jacobian(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central difference Jacobian for f: Rⁿ → Rᵐ."""
    n = len(x)
    f0 = f(x)
    m = len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        J[:, i] = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)
    return J


# =============================================================================
# Exercise 4: Error Analysis — numerical vs analytic
# =============================================================================

def gradient_error_analysis():
    """Plot numerical gradient error vs step size h."""
    # Test function: Rosenbrock
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad_f_analytic(x):
        dfdx = -2 * (1 - x[0]) + 100 * 2 * (x[1] - x[0]**2) * (-2 * x[0])
        dfdy = 100 * 2 * (x[1] - x[0]**2)
        return np.array([dfdx, dfdy])

    x0 = np.array([0.5, 0.5])
    g_analytic = grad_f_analytic(x0)

    h_values = np.logspace(-15, 0, 100)
    errors_forward = []
    errors_central = []

    for h in h_values:
        # Forward difference
        g_fwd = np.zeros(2)
        for i in range(2):
            e_i = np.zeros(2)
            e_i[i] = 1.0
            g_fwd[i] = (f(x0 + h * e_i) - f(x0)) / h

        # Central difference
        g_central = numerical_gradient(f, x0, h)

        errors_forward.append(np.linalg.norm(g_fwd - g_analytic))
        errors_central.append(np.linalg.norm(g_central - g_analytic))

    print("\n=== Gradient Error Analysis ===")
    print("Step size h | Forward Error | Central Error")
    print("-" * 50)
    for h, ef, ec in zip(
        [1e-1, 1e-3, 1e-5, 1e-8, 1e-10, 1e-13],
        [errors_forward[i] for i in [90, 70, 50, 30, 20, 5]],
        [errors_central[i] for i in [90, 70, 50, 30, 20, 5]]
    ):
        print(f"  h={h:.0e}   | {ef:.2e}       | {ec:.2e}")

    # Optimal h analysis
    best_central = min(errors_central)
    best_h_idx = errors_central.index(best_central)
    print(f"\nBest central difference error: {best_central:.2e} at h={h_values[best_h_idx]:.2e}")
    print("Theory: optimal h for central diff ≈ ε^(1/3) ≈ 6e-6 for float64")


# =============================================================================
# Gradient Checker Utility
# =============================================================================

def gradient_check(f, grad_f, x: np.ndarray, h: float = 1e-5, tol: float = 1e-5
                   ) -> bool:
    """
    Check analytic gradient against numerical gradient.
    Returns True if they agree within tolerance.
    """
    g_analytic = grad_f(x)
    g_numerical = numerical_gradient(f, x, h)

    # Relative error with safety denominator
    eps = 1e-7
    rel_errors = np.abs(g_analytic - g_numerical) / (np.abs(g_analytic) + np.abs(g_numerical) + eps)
    max_error = np.max(rel_errors)

    if max_error > tol:
        print(f"  ❌ Gradient check FAILED: max relative error = {max_error:.2e}")
        print(f"     Analytic:  {g_analytic}")
        print(f"     Numerical: {g_numerical}")
        return False
    else:
        print(f"  ✓ Gradient check passed: max relative error = {max_error:.2e}")
        return True


if __name__ == "__main__":
    symbolic_gradient_demo()
    symbolic_jacobian_demo()
    gradient_error_analysis()

    print("\n=== Gradient Checker Demo ===")
    # Rosenbrock
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def rosenbrock_grad(x):
        dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dfdy = 200 * (x[1] - x[0]**2)
        return np.array([dfdx, dfdy])

    for pt in [np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([-1.2, 1.0])]:
        print(f"  At x={pt}:")
        gradient_check(rosenbrock, rosenbrock_grad, pt)
