"""
Day 10: Numerical Differentiation — Companion Code
Phase I, Week 2

Run: python numerical_differentiation.py
"""

import numpy as np
from typing import Callable


# =============================================================================
# Exercise 1: Error vs Step Size (V-shaped curve)
# =============================================================================

def error_vs_h():
    """Plot error vs h for forward, central, complex-step on sin(x)."""
    print("=== Exercise 1: Error vs Step Size ===")
    x0 = 1.0
    f_exact = np.cos(x0)  # d/dx sin(x) at x=1

    hs = np.logspace(-1, -16, 50)

    results = {"forward": [], "central": [], "complex": []}

    for h in hs:
        # Forward difference
        df_fwd = (np.sin(x0 + h) - np.sin(x0)) / h
        results["forward"].append(abs(df_fwd - f_exact))

        # Central difference
        df_cen = (np.sin(x0 + h) - np.sin(x0 - h)) / (2 * h)
        results["central"].append(abs(df_cen - f_exact))

        # Complex-step
        df_cpx = np.sin(complex(x0, h)).imag / h
        results["complex"].append(abs(df_cpx - f_exact))

    # Report optimal h for each method
    for method, errors in results.items():
        best_idx = np.argmin(errors)
        print(f"  {method:>8}: best h = {hs[best_idx]:.2e}, "
              f"best error = {errors[best_idx]:.2e}")


# =============================================================================
# Exercise 2: Optimal h finder
# =============================================================================

def find_optimal_h(f: Callable, x0: float, f_prime_exact: float,
                   method: str = "central") -> tuple:
    """Find optimal h by sweeping."""
    best_h = 1.0
    best_err = float('inf')

    for log_h in np.linspace(-1, -16, 200):
        h = 10**log_h
        if method == "forward":
            df = (f(x0 + h) - f(x0)) / h
        elif method == "central":
            df = (f(x0 + h) - f(x0 - h)) / (2 * h)
        elif method == "complex":
            df = f(complex(x0, h)).imag / h
        else:
            raise ValueError(f"Unknown method: {method}")

        err = abs(df - f_prime_exact)
        if err < best_err:
            best_err = err
            best_h = h

    return best_h, best_err


def demo_optimal_h():
    """Demonstrate optimal h for different methods."""
    print("\n=== Exercise 2: Optimal h ===")
    # f(x) = exp(x), f'(1) = e
    f = lambda x: np.exp(x) if isinstance(x, float) else np.exp(x)
    x0 = 1.0
    exact = np.exp(1.0)

    for method in ["forward", "central", "complex"]:
        h_opt, err_opt = find_optimal_h(f, x0, exact, method)
        print(f"  {method:>8}: optimal h = {h_opt:.2e}, error = {err_opt:.2e}")


# =============================================================================
# Exercise 3: Complex-Step Jacobian
# =============================================================================

def complex_step_jacobian(f: Callable, x: np.ndarray,
                          h: float = 1e-20) -> np.ndarray:
    """Compute Jacobian of f: R^n -> R^m using complex-step method.
    
    Args:
        f: function taking real array, returning real array
        x: point at which to evaluate Jacobian
        h: step size (can be very small, e.g. 1e-20)
    
    Returns:
        J: m×n Jacobian matrix
    """
    n = len(x)
    f0 = f(x)
    m = len(f0)
    J = np.zeros((m, n))

    for j in range(n):
        x_complex = x.astype(complex)
        x_complex[j] += complex(0, h)
        J[:, j] = f(x_complex).imag / h

    return J


def central_diff_jacobian(f: Callable, x: np.ndarray,
                          h: float = 1e-6) -> np.ndarray:
    """Compute Jacobian using central differences."""
    n = len(x)
    f0 = f(x)
    m = len(f0)
    J = np.zeros((m, n))

    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        J[:, j] = (f(x + h * e) - f(x - h * e)) / (2 * h)

    return J


def test_jacobian():
    """Test Jacobian implementations."""
    print("\n=== Exercise 3: Complex-Step Jacobian ===")

    def f(x):
        """f: R^3 -> R^2."""
        return np.array([
            np.sin(x[0]) * x[1] + x[2]**2,
            np.exp(x[0] + x[1]) - x[2]
        ])

    def J_analytic(x):
        return np.array([
            [np.cos(x[0]) * x[1], np.sin(x[0]), 2 * x[2]],
            [np.exp(x[0] + x[1]), np.exp(x[0] + x[1]), -1.0]
        ])

    x0 = np.array([0.5, 1.0, -0.3])

    J_exact = J_analytic(x0)
    J_complex = complex_step_jacobian(f, x0)
    J_central = central_diff_jacobian(f, x0)

    err_complex = np.max(np.abs(J_complex - J_exact))
    err_central = np.max(np.abs(J_central - J_exact))

    print(f"  Complex-step max error: {err_complex:.2e}")
    print(f"  Central diff max error: {err_central:.2e}")
    print(f"  Complex-step is {err_central/max(err_complex, 1e-30):.0f}× more accurate")


# =============================================================================
# Exercise 4: Rounding-Dominated Regime
# =============================================================================

def rounding_regime():
    """Show when numerical differentiation struggles."""
    print("\n=== Exercise 4: Rounding-Dominated Regime ===")

    # f(x) = x + 1e-12 near x = 1 → f'(x) = 1
    offset = 1e-12

    def f(x):
        return x + offset

    x0 = 1.0
    exact = 1.0

    print("  f(x) = x + 1e-12, f'(1) = 1")
    for h_exp in [-4, -8, -12, -14, -16]:
        h = 10.0 ** h_exp
        fwd = (f(x0 + h) - f(x0)) / h
        cen = (f(x0 + h) - f(x0 - h)) / (2 * h)
        cpx = (x0 + complex(0, h) + offset).imag / h  # complex-step

        print(f"  h=1e{h_exp:+d}: fwd={fwd:.10f}, cen={cen:.10f}, cpx={cpx:.10f}")


# =============================================================================
# Gradient Checker Utility
# =============================================================================

def gradient_checker(f: Callable, grad_f: Callable, x: np.ndarray,
                     h: float = 1e-6) -> dict:
    """Compare analytic gradient to numerical. Returns max relative error."""
    g_analytic = grad_f(x)
    n = len(x)
    g_numerical = np.zeros(n)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        g_numerical[i] = (f(x + h * e) - f(x - h * e)) / (2 * h)

    denom = np.maximum(np.abs(g_analytic), np.abs(g_numerical))
    denom = np.maximum(denom, 1.0)
    rel_errors = np.abs(g_analytic - g_numerical) / denom

    return {
        "max_rel_error": np.max(rel_errors),
        "mean_rel_error": np.mean(rel_errors),
        "worst_component": np.argmax(rel_errors),
        "pass": np.max(rel_errors) < 1e-4,
    }


def demo_gradient_checker():
    """Demo the gradient checker."""
    print("\n=== Gradient Checker Demo ===")

    # Correct gradient
    def f(x):
        return np.sum(x**3) + np.sin(x[0] * x[1])

    def grad_correct(x):
        return np.array([3*x[0]**2 + x[1]*np.cos(x[0]*x[1]),
                         3*x[1]**2 + x[0]*np.cos(x[0]*x[1])])

    def grad_wrong(x):
        return np.array([3*x[0]**2 + x[1]*np.sin(x[0]*x[1]),  # bug: sin vs cos
                         3*x[1]**2 + x[0]*np.cos(x[0]*x[1])])

    x0 = np.array([1.5, -0.7])

    result_ok = gradient_checker(f, grad_correct, x0)
    result_bad = gradient_checker(f, grad_wrong, x0)

    print(f"  Correct gradient: max_rel_err={result_ok['max_rel_error']:.2e} "
          f"{'PASS ✓' if result_ok['pass'] else 'FAIL ✗'}")
    print(f"  Wrong gradient:   max_rel_err={result_bad['max_rel_error']:.2e} "
          f"{'PASS ✓' if result_bad['pass'] else 'FAIL ✗'}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    error_vs_h()
    demo_optimal_h()
    test_jacobian()
    rounding_regime()
    demo_gradient_checker()
