"""
Day 12: Function Landscapes & Conditioning — Companion Code
Phase I, Week 2

Run: python conditioning.py
"""

import numpy as np
import time


# =============================================================================
# Exercise 1: Condition Number Visualization
# =============================================================================

def gradient_descent_quadratic(H: np.ndarray, x0: np.ndarray,
                                lr: float = None,
                                max_iter: int = 500,
                                tol: float = 1e-8) -> dict:
    """Gradient descent on f(x) = 0.5 * x^T H x.
    
    If lr is None, uses optimal lr = 2/(λ_max + λ_min).
    Returns trajectory and iteration count.
    """
    eigvals = np.linalg.eigvalsh(H)
    if lr is None:
        lr = 2.0 / (eigvals[-1] + eigvals[0])

    x = x0.copy()
    trajectory = [x.copy()]

    for i in range(max_iter):
        g = H @ x
        x = x - lr * g
        trajectory.append(x.copy())
        if np.linalg.norm(g) < tol:
            break

    return {
        "trajectory": np.array(trajectory),
        "iterations": len(trajectory) - 1,
        "final_x": x,
        "kappa": eigvals[-1] / eigvals[0],
    }


def condition_number_demo():
    """Show effect of condition number on gradient descent."""
    print("=== Exercise 1: Condition Number & GD ===")
    x0 = np.array([5.0, 5.0])

    for kappa in [1, 10, 100, 1000]:
        # Create 2x2 PD matrix with desired condition number
        # Eigenvalues: 1 and kappa
        theta = np.pi / 6  # rotate to make it non-axis-aligned
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        D = np.diag([1.0, float(kappa)])
        H = R @ D @ R.T

        result = gradient_descent_quadratic(H, x0)
        print(f"  κ = {kappa:>5}: {result['iterations']:>4} iterations, "
              f"final ||x|| = {np.linalg.norm(result['final_x']):.2e}")


# =============================================================================
# Exercise 2: Digits Lost
# =============================================================================

def digits_lost_demo():
    """Show precision loss as condition number increases."""
    print("\n=== Exercise 2: Digits Lost ===")

    for log_kappa in [0, 3, 6, 9, 12, 15]:
        n = 20
        kappa = 10.0 ** log_kappa

        # Create PD matrix with specified condition number
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.randn(n, n))
        eigvals = np.logspace(0, log_kappa, n)
        H = Q @ np.diag(eigvals) @ Q.T

        # True solution
        x_true = rng.randn(n)
        b = H @ x_true

        # Solve
        x_computed = np.linalg.solve(H, b)

        rel_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
        digits_correct = max(0, -np.log10(rel_error + 1e-20))

        print(f"  κ = 10^{log_kappa:>2}: rel_error = {rel_error:.2e}, "
              f"~{digits_correct:.1f} correct digits")


# =============================================================================
# Exercise 3: Preconditioning
# =============================================================================

def conjugate_gradient(H, b, x0=None, M_inv=None, max_iter=None, tol=1e-10):
    """Preconditioned conjugate gradient.
    
    M_inv: function that applies M^{-1} to a vector (preconditioner).
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = 2 * n
    if M_inv is None:
        M_inv = lambda v: v  # no preconditioning

    x = x0.copy()
    r = b - H @ x
    z = M_inv(r)
    p = z.copy()
    rz = r @ z

    iterations = 0
    for i in range(max_iter):
        Hp = H @ p
        alpha = rz / (p @ Hp)
        x = x + alpha * p
        r = r - alpha * Hp

        if np.linalg.norm(r) < tol:
            iterations = i + 1
            break

        z = M_inv(r)
        rz_new = r @ z
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
        iterations = i + 1

    return x, iterations


def preconditioning_demo():
    """Compare CG with and without preconditioning."""
    print("\n=== Exercise 3: Preconditioning ===")

    n = 200
    rng = np.random.default_rng(42)

    # Create ill-conditioned PD matrix
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.logspace(0, 4, n)  # κ = 10^4
    H = Q @ np.diag(eigvals) @ Q.T

    b = rng.randn(n)
    x_true = np.linalg.solve(H, b)

    # No preconditioning
    x_cg, iters_cg = conjugate_gradient(H, b)
    err_cg = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)

    # Jacobi preconditioning: M = diag(H)
    diag_H = np.diag(H)
    M_inv_jacobi = lambda v: v / diag_H

    x_pcg, iters_pcg = conjugate_gradient(H, b, M_inv=M_inv_jacobi)
    err_pcg = np.linalg.norm(x_pcg - x_true) / np.linalg.norm(x_true)

    kappa_orig = eigvals[-1] / eigvals[0]
    # Effective kappa after Jacobi
    H_precond = np.diag(1.0/np.sqrt(diag_H)) @ H @ np.diag(1.0/np.sqrt(diag_H))
    eigvals_precond = np.linalg.eigvalsh(H_precond)
    kappa_precond = eigvals_precond[-1] / eigvals_precond[0]

    print(f"  n = {n}, κ(H) = {kappa_orig:.0f}")
    print(f"  CG without precond: {iters_cg} iters, rel_err = {err_cg:.2e}")
    print(f"  CG with Jacobi:     {iters_pcg} iters, rel_err = {err_pcg:.2e}")
    print(f"  κ after Jacobi: {kappa_precond:.0f}")
    print(f"  Iteration reduction: {iters_cg - iters_pcg} fewer iterations")


# =============================================================================
# Exercise 4: Variable Scaling
# =============================================================================

def variable_scaling_demo():
    """SLAM-like problem with mixed scales."""
    print("\n=== Exercise 4: Variable Scaling ===")

    # Simulate: 2 translation params (meters, ~100) + 1 rotation (radians, ~0.01)
    # Information matrix reflects measurement precision
    rng = np.random.default_rng(42)

    # Unscaled: translation info = 1/σ_t² = 10000, rotation info = 1/σ_θ² = 1
    H_unscaled = np.diag([10000.0, 10000.0, 1.0])
    # Add some coupling
    H_unscaled[0, 2] = 50.0
    H_unscaled[2, 0] = 50.0
    H_unscaled[1, 2] = 30.0
    H_unscaled[2, 1] = 30.0

    b = np.array([100.0, 80.0, 0.5])

    kappa_unscaled = np.linalg.cond(H_unscaled)

    # Scale: s = [0.01, 0.01, 1] (divide translation by 100)
    s = np.array([0.01, 0.01, 1.0])
    S = np.diag(s)
    H_scaled = S @ H_unscaled @ S
    b_scaled = S @ b

    kappa_scaled = np.linalg.cond(H_scaled)

    # Solve both
    x_unscaled = np.linalg.solve(H_unscaled, b)
    x_tilde = np.linalg.solve(H_scaled, b_scaled)
    x_from_scaled = x_tilde / s  # unscale

    print(f"  Unscaled κ = {kappa_unscaled:.0f}")
    print(f"  Scaled κ   = {kappa_scaled:.0f}")
    print(f"  Improvement: {kappa_unscaled/kappa_scaled:.1f}×")
    print(f"  Solutions match: {np.allclose(x_unscaled, x_from_scaled)}")

    # GD comparison
    x0 = np.array([1.0, 1.0, 1.0])
    r_unscaled = gradient_descent_quadratic(H_unscaled, x0)
    r_scaled = gradient_descent_quadratic(H_scaled, x0 * s)

    print(f"  GD iterations unscaled: {r_unscaled['iterations']}")
    print(f"  GD iterations scaled:   {r_scaled['iterations']}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    condition_number_demo()
    digits_lost_demo()
    preconditioning_demo()
    variable_scaling_demo()
