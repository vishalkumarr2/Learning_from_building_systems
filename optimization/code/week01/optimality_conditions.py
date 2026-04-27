"""
Day 6: Optimality Conditions — Companion Code
Phase I, Week 1

Run: python optimality_conditions.py
"""

import numpy as np
from typing import Callable


# =============================================================================
# Exercise 1: Find and Classify Stationary Points
# =============================================================================

def find_stationary_2d_grid(grad_f, x_range=(-3, 3), y_range=(-3, 3),
                            n_grid=50, tol=1e-6):
    """Find approximate stationary points via grid search + Newton refinement."""
    candidates = []
    for x in np.linspace(x_range[0], x_range[1], n_grid):
        for y in np.linspace(y_range[0], y_range[1], n_grid):
            pt = np.array([x, y])
            g = grad_f(pt)
            if np.linalg.norm(g) < 0.5:  # coarse filter
                candidates.append(pt)
    return candidates


def classify_stationary_point(hessian_at_point: np.ndarray) -> str:
    """Classify based on Hessian eigenvalues."""
    eigvals = np.linalg.eigvalsh(hessian_at_point)
    if np.all(eigvals > 1e-10):
        return "Strict local minimum (PD)"
    elif np.all(eigvals < -1e-10):
        return "Strict local maximum (ND)"
    elif np.all(eigvals >= -1e-10):
        return "Degenerate — PSD, inconclusive"
    elif np.all(eigvals <= 1e-10):
        return "Degenerate — NSD, inconclusive"
    else:
        return "Saddle point (indefinite)"


def example_classify_all():
    """f(x,y) = x³ + y³ - 3xy — a classic example with 3 stationary points."""
    # ∂f/∂x = 3x² - 3y = 0 → y = x²
    # ∂f/∂y = 3y² - 3x = 0 → x = y²
    # Substituting: x = (x²)² = x⁴ → x(x³-1) = 0 → x=0, x=1
    # Stationary points: (0,0) and (1,1)

    def f(x):
        return x[0]**3 + x[1]**3 - 3 * x[0] * x[1]

    def grad_f(x):
        return np.array([3*x[0]**2 - 3*x[1], 3*x[1]**2 - 3*x[0]])

    def hessian_f(x):
        return np.array([[6*x[0], -3], [-3, 6*x[1]]])

    print("=== f(x,y) = x³ + y³ - 3xy ===")
    for pt_name, pt in [("(0,0)", np.array([0.0, 0.0])),
                        ("(1,1)", np.array([1.0, 1.0]))]:
        g = grad_f(pt)
        H = hessian_f(pt)
        eigvals = np.linalg.eigvalsh(H)
        classification = classify_stationary_point(H)
        print(f"  {pt_name}: f={f(pt):.2f}, ||∇f||={np.linalg.norm(g):.2e}, "
              f"H eigvals={np.round(eigvals, 2)}, → {classification}")


# =============================================================================
# Exercise 2: Saddle Point Detection and Escape Direction
# =============================================================================

def saddle_point_analysis():
    """Analyze saddle points and find descent directions."""
    # f(x,y) = x² - y² (classic saddle)
    def f(x):
        return x[0]**2 - x[1]**2

    def hessian_f(x):
        return np.array([[2.0, 0.0], [0.0, -2.0]])

    print("\n=== Saddle Point Analysis: f(x,y) = x² - y² ===")
    pt = np.array([0.0, 0.0])
    H = hessian_f(pt)
    eigvals, eigvecs = np.linalg.eigh(H)

    print(f"  At origin: H eigenvalues = {eigvals}")
    for i, (val, vec) in enumerate(zip(eigvals, eigvecs.T)):
        direction = "ascent" if val > 0 else "descent"
        print(f"  Eigenvalue {val:+.1f}: eigenvector {np.round(vec, 2)} → {direction} direction")

    # Verify: moving along negative curvature direction decreases f
    neg_dir = eigvecs[:, eigvals < 0].flatten()
    eps = 0.1
    print(f"\n  f(0,0) = {f(pt):.4f}")
    print(f"  f(0 + {eps}*v_neg) = {f(pt + eps * neg_dir):.4f} (decreased ✓)")
    print(f"  f(0 + {eps}*v_pos) = {f(pt + eps * eigvecs[:, 0]):.4f} (increased)")


# =============================================================================
# Exercise 3: Optimality Diagnostics
# =============================================================================

def optimality_diagnostics(f: Callable, grad_f: Callable,
                           hessian_f: Callable, x: np.ndarray,
                           name: str = ""):
    """Full optimality diagnostic at a point."""
    fval = f(x)
    g = grad_f(x)
    H = hessian_f(x)
    eigvals = np.linalg.eigvalsh(H)
    grad_norm = np.linalg.norm(g)
    cond = eigvals[-1] / max(eigvals[0], 1e-16)

    print(f"\n=== Optimality Diagnostics{' — ' + name if name else ''} ===")
    print(f"  Point:           x = {np.round(x, 6)}")
    print(f"  Function value:  f(x) = {fval:.6e}")
    print(f"  Gradient norm:   ||∇f|| = {grad_norm:.6e}")
    print(f"  Hessian eigvals: {np.round(eigvals, 4)}")
    print(f"  Condition number: κ = {abs(cond):.1f}")

    # FONC check
    fonc = grad_norm < 1e-6
    print(f"  FONC (∇f ≈ 0):  {'PASS ✓' if fonc else 'FAIL ✗'}")

    # SONC check
    sonc = fonc and np.all(eigvals > -1e-8)
    print(f"  SONC (H ≥ 0):   {'PASS ✓' if sonc else 'FAIL ✗'}")

    # SOSC check
    sosc = fonc and np.all(eigvals > 1e-8)
    print(f"  SOSC (H > 0):   {'PASS ✓' if sosc else 'N/A or FAIL'}")

    # Classification
    if not fonc:
        print(f"  Verdict: NOT a stationary point")
    elif sosc:
        print(f"  Verdict: STRICT LOCAL MINIMUM (guaranteed)")
    elif sonc:
        print(f"  Verdict: Inconclusive — PSD Hessian, needs higher-order analysis")
    else:
        print(f"  Verdict: SADDLE POINT or MAXIMUM")


# =============================================================================
# Exercise 4: Escaping Saddle Points with Perturbation
# =============================================================================

def escape_saddle_demo():
    """Show that perturbed GD escapes saddle points."""
    # f(x,y) = 0.5*x² - 0.5*y² → saddle at origin
    def f(x):
        return 0.5 * x[0]**2 - 0.5 * x[1]**2

    def grad_f(x):
        return np.array([x[0], -x[1]])

    print("\n=== Escaping Saddle Points ===")

    # Standard GD starts at saddle → stuck
    x = np.array([0.0, 0.0])
    for i in range(10):
        x = x - 0.1 * grad_f(x)
    print(f"  Standard GD from (0,0) after 10 steps: x = {np.round(x, 6)}, f = {f(x):.6f}")
    print(f"  → Stuck at saddle!")

    # Perturbed GD escapes
    rng = np.random.default_rng(42)
    x = np.array([0.0, 0.0])
    x += rng.normal(0, 0.01, size=2)  # small perturbation
    for i in range(50):
        x = x - 0.1 * grad_f(x)
    print(f"  Perturbed GD after 50 steps:           x = {np.round(x, 4)}, f = {f(x):.4f}")
    print(f"  → Escaped along negative curvature direction!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    example_classify_all()
    saddle_point_analysis()

    # Rosenbrock diagnostics at minimum
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    def rosenbrock_grad(x):
        return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                         200*(x[1]-x[0]**2)])
    def rosenbrock_hessian(x):
        return np.array([[2 - 400*(x[1]-3*x[0]**2), -400*x[0]],
                         [-400*x[0], 200.0]])

    optimality_diagnostics(rosenbrock, rosenbrock_grad, rosenbrock_hessian,
                           np.array([1.0, 1.0]), "Rosenbrock at (1,1)")
    optimality_diagnostics(rosenbrock, rosenbrock_grad, rosenbrock_hessian,
                           np.array([0.0, 0.0]), "Rosenbrock at (0,0)")

    # Degenerate case: f(x,y) = x⁴ + y⁴
    def f_degen(x):
        return x[0]**4 + x[1]**4
    def grad_degen(x):
        return np.array([4*x[0]**3, 4*x[1]**3])
    def hess_degen(x):
        return np.array([[12*x[0]**2, 0], [0, 12*x[1]**2]])

    optimality_diagnostics(f_degen, grad_degen, hess_degen,
                           np.array([0.0, 0.0]), "x⁴+y⁴ at origin (degenerate)")

    escape_saddle_demo()
