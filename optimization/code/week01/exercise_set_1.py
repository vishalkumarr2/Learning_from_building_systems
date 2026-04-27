"""
Day 7: Week 1 Comprehensive Exercise Set — Companion Code
Phase I, Week 1

10 problems covering all Week 1 topics.
Run: python exercise_set_1.py
"""

import numpy as np
from typing import Callable


# =============================================================================
# Problem 1: SVD and Rank
# =============================================================================

def problem_1_svd_rank():
    """SVD analysis of a rank-deficient matrix."""
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [1, 1, 1]], dtype=float)

    U, s, Vt = np.linalg.svd(A)
    rank = np.sum(s > 1e-10)

    print("=== Problem 1: SVD and Rank ===")
    print(f"A =\n{A}")
    print(f"Singular values: {np.round(s, 4)}")
    print(f"Rank: {rank}")

    # Null space: columns of V corresponding to zero singular values
    null_space = Vt[rank:].T
    print(f"Null space basis:\n{np.round(null_space, 4)}")

    # Best rank-1 approximation
    A1 = s[0] * np.outer(U[:, 0], Vt[0, :])
    print(f"Best rank-1 approximation:\n{np.round(A1, 4)}")
    print(f"||A - A₁||_F = {np.linalg.norm(A - A1):.4f} (= σ₂ = {s[1]:.4f})")


# =============================================================================
# Problem 2: A^T A is PSD
# =============================================================================

def problem_2_ata_psd():
    """Demonstrate that A^T A is always PSD."""
    print("\n=== Problem 2: A^T A is PSD ===")
    print("Proof: v^T (A^T A) v = (Av)^T (Av) = ||Av||² ≥ 0 for all v")
    print("PD ⟺ Av ≠ 0 for v ≠ 0 ⟺ A has full column rank")

    # Verify numerically
    rng = np.random.default_rng(42)
    for m, n, desc in [(10, 3, "full rank"), (3, 5, "rank deficient")]:
        A = rng.randn(m, n)
        eigvals = np.linalg.eigvalsh(A.T @ A)
        pd = "PD" if np.all(eigvals > 1e-10) else "PSD (not PD)"
        print(f"  A is {m}×{n}: eigvals(A^TA) min={eigvals[0]:.4f} → {pd}")


# =============================================================================
# Problem 3: Gradient of ||Ax - b||²
# =============================================================================

def problem_3_ls_gradient():
    """Derive and verify gradient of least squares."""
    print("\n=== Problem 3: ∇||Ax-b||² ===")
    print("f(x) = (Ax-b)^T(Ax-b) = x^T A^T A x - 2b^T Ax + b^Tb")
    print("∇f = 2A^T Ax - 2A^T b = 2A^T(Ax - b)")

    # Numerical verification
    rng = np.random.default_rng(42)
    A = rng.randn(5, 3)
    b = rng.randn(5)
    x = rng.randn(3)

    # Analytic gradient
    grad_analytic = 2 * A.T @ (A @ x - b)

    # Numerical gradient
    h = 1e-7
    grad_numerical = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        f_plus = np.sum((A @ (x + h*e) - b)**2)
        f_minus = np.sum((A @ (x - h*e) - b)**2)
        grad_numerical[i] = (f_plus - f_minus) / (2*h)

    error = np.linalg.norm(grad_analytic - grad_numerical)
    print(f"  Analytic:  {np.round(grad_analytic, 6)}")
    print(f"  Numerical: {np.round(grad_numerical, 6)}")
    print(f"  Error: {error:.2e} ✓")


# =============================================================================
# Problem 4: Hessian Classification Pipeline
# =============================================================================

def problem_4_classify():
    """f(x,y) = x³ - 3xy² — monkey saddle."""
    print("\n=== Problem 4: f(x,y) = x³ - 3xy² ===")
    # ∇f = (3x² - 3y², -6xy) = 0
    # From -6xy = 0: x=0 or y=0
    # If x=0: 3(0) - 3y² = 0 → y=0
    # If y=0: 3x² = 0 → x=0
    # Only stationary point: (0,0)

    def hessian(x):
        return np.array([[6*x[0], -6*x[1]], [-6*x[1], -6*x[0]]])

    pt = np.array([0.0, 0.0])
    H = hessian(pt)
    eigvals = np.linalg.eigvalsh(H)

    print(f"  Stationary point: (0,0)")
    print(f"  H(0,0) = {H}")
    print(f"  Eigenvalues: {eigvals}")
    print(f"  Classification: Degenerate (H = 0 matrix)")
    print(f"  Actually a 'monkey saddle' — f increases in 3 directions,")
    print(f"  decreases in 3 others (not a standard min/max/saddle)")


# =============================================================================
# Problem 5: Taylor Approximation
# =============================================================================

def problem_5_taylor():
    """Second-order Taylor of e^(x+y) around (0,0)."""
    print("\n=== Problem 5: Taylor of e^(x+y) ===")
    # f = e^(x+y), ∇f = (e^(x+y), e^(x+y)), H = e^(x+y) * ones(2,2)
    # At (0,0): f=1, ∇f=(1,1), H=ones(2,2)
    # T₂(δ) = 1 + δ₁ + δ₂ + 0.5(δ₁² + 2δ₁δ₂ + δ₂²)
    #        = 1 + (δ₁+δ₂) + 0.5(δ₁+δ₂)²

    def f(x, y):
        return np.exp(x + y)

    def taylor2(dx, dy):
        return 1 + dx + dy + 0.5 * (dx**2 + 2*dx*dy + dy**2)

    for dx, dy in [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0)]:
        actual = f(dx, dy)
        approx = taylor2(dx, dy)
        error = abs(actual - approx) / actual * 100
        print(f"  ({dx},{dy}): actual={actual:.6f}, Taylor₂={approx:.6f}, error={error:.2f}%")


# =============================================================================
# Problem 6: Convexity of Max Function
# =============================================================================

def problem_6_convexity_proof():
    """Show that f(x) = max(x_1, ..., x_n) is convex using the definition."""
    print("\n=== Problem 6: Convexity of f(x) = max(x_i) ===")

    # Proof by definition: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
    # For any i: (θx + (1-θ)y)_i = θx_i + (1-θ)y_i ≤ θ max(x) + (1-θ) max(y)
    # Taking max over i on LHS: max(θx + (1-θ)y) ≤ θ max(x) + (1-θ) max(y)  ✓
    print("  Proof: f(x) = max(x_1, ..., x_n) is convex.")
    print("  For any θ ∈ [0,1], any x, y ∈ ℝⁿ:")
    print("    Component i of θx + (1-θ)y = θxᵢ + (1-θ)yᵢ")
    print("    ≤ θ·max(x) + (1-θ)·max(y)  [since xᵢ ≤ max(x), yᵢ ≤ max(y)]")
    print("    Taking max over i on LHS gives the convexity inequality. □")

    # Numerical verification
    rng = np.random.default_rng(42)
    violations = 0
    n_trials = 10000
    for _ in range(n_trials):
        n = rng.integers(2, 10)
        x = rng.randn(n)
        y = rng.randn(n)
        theta = rng.uniform(0, 1)
        z = theta * x + (1 - theta) * y
        lhs = np.max(z)
        rhs = theta * np.max(x) + (1 - theta) * np.max(y)
        if lhs > rhs + 1e-12:
            violations += 1
    print(f"\n  Numerical verification: {n_trials} random trials, {violations} violations")
    print(f"  Convexity {'CONFIRMED' if violations == 0 else 'VIOLATED'}.")


# =============================================================================
# Problem 7: Newton Step on Quadratic
# =============================================================================

def problem_7_newton_quadratic():
    """Newton on f(x,y) = x² + 10y²."""
    print("\n=== Problem 7: Newton vs GD on x² + 10y² ===")

    H = np.array([[2.0, 0.0], [0.0, 20.0]])
    L = 20.0  # largest eigenvalue of Hessian
    alpha_gd = 1.0 / (2 * L)

    x_newton = np.array([5.0, 5.0])
    x_gd = np.array([5.0, 5.0])

    # Newton: one step for a quadratic!
    grad = np.array([2*x_newton[0], 20*x_newton[1]])
    x_newton = x_newton - np.linalg.solve(H, grad)
    print(f"  Newton from (5,5): one step → {x_newton} (exact!)")

    # GD: many steps needed
    print(f"  GD (α=1/{2*L:.0f}={alpha_gd:.4f}):")
    for k in [1, 10, 50, 100, 500]:
        x_gd_k = np.array([5.0, 5.0])
        for _ in range(k):
            grad = np.array([2*x_gd_k[0], 20*x_gd_k[1]])
            x_gd_k = x_gd_k - alpha_gd * grad
        dist = np.linalg.norm(x_gd_k)
        print(f"    k={k:>3}: x={np.round(x_gd_k, 4)}, ||x||={dist:.6f}")


# =============================================================================
# Problem 8: Condition Number Impact
# =============================================================================

def problem_8_condition_number():
    """Sensitivity of Ax=b to perturbations."""
    print("\n=== Problem 8: Condition Number Impact ===")
    rng = np.random.default_rng(42)
    n = 100

    print(f"  {'κ':>10} | {'||δx||/||x||':>15} | {'κ·||δb||/||b||':>15} | Ratio")
    print(f"  {'-'*65}")

    for kappa_target in [10, 100, 1000, 10000]:
        # Create matrix with target condition number
        U, _ = np.linalg.qr(rng.randn(n, n))
        V, _ = np.linalg.qr(rng.randn(n, n))
        s = np.linspace(1, kappa_target, n)
        A = U @ np.diag(s) @ V.T

        b = rng.randn(n)
        x = np.linalg.solve(A, b)

        # Perturb b
        db = rng.randn(n) * 1e-8
        x_pert = np.linalg.solve(A, b + db)
        dx = x_pert - x

        rel_dx = np.linalg.norm(dx) / np.linalg.norm(x)
        rel_db = np.linalg.norm(db) / np.linalg.norm(b)
        bound = kappa_target * rel_db

        print(f"  {kappa_target:>10} | {rel_dx:>15.2e} | {bound:>15.2e} | {rel_dx/rel_db:.1f}")


# =============================================================================
# Problem 9: Point Classifier
# =============================================================================

def classify_point(f: Callable, grad_f: Callable, hessian_f: Callable,
                   x: np.ndarray, grad_tol: float = 1e-6) -> str:
    """Classify a point: not stationary / strict min / strict max / saddle / degenerate."""
    g = grad_f(x)
    if np.linalg.norm(g) > grad_tol:
        return "Not stationary"

    H = hessian_f(x)
    eigvals = np.linalg.eigvalsh(H)

    if np.all(eigvals > 1e-10):
        return "Strict local minimum"
    elif np.all(eigvals < -1e-10):
        return "Strict local maximum"
    elif np.min(eigvals) < -1e-10 and np.max(eigvals) > 1e-10:
        return "Saddle point"
    else:
        return "Degenerate (inconclusive)"


# =============================================================================
# Problem 10: End-to-End Newton on Rosenbrock
# =============================================================================

def problem_10_newton_rosenbrock():
    """Full Newton's method on Rosenbrock."""
    print("\n=== Problem 10: Newton on Rosenbrock from (-1.2, 1.0) ===")

    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad_f(x):
        return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                         200*(x[1]-x[0]**2)])

    def hessian_f(x):
        return np.array([[2 - 400*(x[1]-3*x[0]**2), -400*x[0]],
                         [-400*x[0], 200.0]])

    x = np.array([-1.2, 1.0])
    print(f"  {'Iter':>4} | {'f(x)':>12} | {'||∇f||':>12} | {'x':>24}")
    print(f"  {'-'*60}")
    print(f"  {0:>4} | {f(x):>12.6f} | {np.linalg.norm(grad_f(x)):>12.6f} | {np.round(x, 6)}")

    for k in range(1, 30):
        g = grad_f(x)
        H = hessian_f(x)

        # Check if H is PD, if not use regularization
        eigvals = np.linalg.eigvalsh(H)
        if np.min(eigvals) < 1e-8:
            H += (abs(np.min(eigvals)) + 1e-4) * np.eye(len(x))

        delta = np.linalg.solve(H, -g)

        # Backtracking line search
        alpha = 1.0
        while f(x + alpha * delta) > f(x) + 1e-4 * alpha * g @ delta:
            alpha *= 0.5
            if alpha < 1e-16:
                break

        x = x + alpha * delta
        gnorm = np.linalg.norm(grad_f(x))

        if k <= 10 or k % 5 == 0 or gnorm < 1e-10:
            print(f"  {k:>4} | {f(x):>12.6e} | {gnorm:>12.6e} | {np.round(x, 6)}")

        if gnorm < 1e-10:
            print(f"\n  Converged at iteration {k}!")
            break

    # SOSC verification
    H_final = hessian_f(x)
    eigvals = np.linalg.eigvalsh(H_final)
    print(f"\n  SOSC check at solution:")
    print(f"    ||∇f|| = {np.linalg.norm(grad_f(x)):.2e}")
    print(f"    H eigenvalues: {np.round(eigvals, 2)}")
    print(f"    PD: {'Yes ✓ — confirmed strict local minimum' if np.all(eigvals > 0) else 'No'}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    problem_1_svd_rank()
    problem_2_ata_psd()
    problem_3_ls_gradient()
    problem_4_classify()
    problem_5_taylor()
    problem_6_convexity_proof()
    problem_7_newton_quadratic()
    problem_8_condition_number()

    print("\n=== Problem 9: Point Classifier ===")
    # Test on x² - y² at (0,0) → saddle
    result = classify_point(
        f=lambda x: x[0]**2 - x[1]**2,
        grad_f=lambda x: np.array([2*x[0], -2*x[1]]),
        hessian_f=lambda x: np.array([[2.0, 0.0], [0.0, -2.0]]),
        x=np.array([0.0, 0.0])
    )
    print(f"  x²-y² at (0,0): {result}")

    problem_10_newton_rosenbrock()
