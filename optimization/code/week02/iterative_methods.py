"""
Day 13: Iterative Methods — Conjugate Gradient — Companion Code
Phase I, Week 2

Run: python iterative_methods.py
"""

import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


# =============================================================================
# Exercise 1: CG from Scratch
# =============================================================================

def conjugate_gradient(A, b, x0=None, max_iter=None, tol=1e-10,
                       callback=None):
    """Conjugate Gradient for Ax = b where A is SPD.
    
    Args:
        A: n×n SPD matrix (dense or sparse, or callable for matvec)
        b: n-vector RHS
        x0: initial guess (default: zeros)
        max_iter: maximum iterations (default: n)
        tol: residual tolerance
        callback: function(k, x, r_norm) called each iteration
    
    Returns:
        x: solution
        info: dict with iteration count, residual history
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = n

    # Support matrix or callable
    if callable(A):
        matvec = A
    else:
        matvec = lambda v: A @ v

    x = x0.copy()
    r = b - matvec(x)
    p = r.copy()
    rr = r @ r
    residuals = [np.sqrt(rr)]

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = p @ Ap

        if pAp <= 0:
            # Negative curvature detected
            break

        alpha = rr / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rr_new = r @ r
        residuals.append(np.sqrt(rr_new))

        if callback:
            callback(k + 1, x, np.sqrt(rr_new))

        if np.sqrt(rr_new) < tol:
            break

        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return x, {"iterations": len(residuals) - 1, "residuals": residuals}


def test_cg_exact():
    """Verify CG converges in exactly n steps."""
    print("=== Exercise 1: CG from Scratch ===")

    for n in [5, 10, 20]:
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.randn(n, n))
        D = np.diag(rng.uniform(1, 10, n))
        A = Q @ D @ Q.T  # SPD with moderate κ

        b = rng.randn(n)
        x_true = np.linalg.solve(A, b)

        x_cg, info = conjugate_gradient(A, b)
        error = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)
        print(f"  n={n:>2}: converged in {info['iterations']:>2} iters, "
              f"rel_error = {error:.2e}")


# =============================================================================
# Exercise 2: Convergence Monitoring
# =============================================================================

def convergence_vs_kappa():
    """Show CG convergence depends on condition number."""
    print("\n=== Exercise 2: Convergence vs κ ===")
    n = 100
    rng = np.random.default_rng(42)

    for log_kappa in [1, 2, 3, 4]:
        kappa = 10 ** log_kappa
        Q, _ = np.linalg.qr(rng.randn(n, n))
        eigvals = np.linspace(1, kappa, n)
        A = Q @ np.diag(eigvals) @ Q.T

        b = rng.randn(n)
        _, info = conjugate_gradient(A, b, tol=1e-10)

        # Find iteration where residual drops below threshold
        residuals = info["residuals"]
        r0 = residuals[0]
        iters_6 = next((i for i, r in enumerate(residuals)
                        if r / r0 < 1e-6), len(residuals) - 1)

        theory_bound = int(3 * np.sqrt(kappa))  # rough estimate

        print(f"  κ = 10^{log_kappa}: {info['iterations']} total iters, "
              f"{iters_6} to 1e-6 reduction, theory ≈ {theory_bound}")


# =============================================================================
# Exercise 3: Preconditioned CG
# =============================================================================

def preconditioned_cg(A, b, M_inv, x0=None, max_iter=None, tol=1e-10):
    """Preconditioned CG: M^{-1}A x = M^{-1}b.
    
    M_inv: callable that applies M^{-1} to a vector.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = n

    if callable(A):
        matvec = A
    else:
        matvec = lambda v: A @ v

    x = x0.copy()
    r = b - matvec(x)
    z = M_inv(r)
    p = z.copy()
    rz = r @ z
    residuals = [np.linalg.norm(r)]

    for k in range(max_iter):
        Ap = matvec(p)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        residuals.append(np.linalg.norm(r))
        if np.linalg.norm(r) < tol:
            break

        z = M_inv(r)
        rz_new = r @ z
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, {"iterations": len(residuals) - 1, "residuals": residuals}


def test_preconditioning():
    """Compare CG vs PCG."""
    print("\n=== Exercise 3: Preconditioned CG ===")
    n = 200
    rng = np.random.default_rng(42)

    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.logspace(0, 4, n)  # κ = 10^4
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)
    x_true = np.linalg.solve(A, b)

    # Plain CG
    x_cg, info_cg = conjugate_gradient(A, b)
    err_cg = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)

    # Jacobi PCG
    diag_A = np.diag(A).copy()
    M_inv_jacobi = lambda v: v / diag_A
    x_pcg, info_pcg = preconditioned_cg(A, b, M_inv_jacobi)
    err_pcg = np.linalg.norm(x_pcg - x_true) / np.linalg.norm(x_true)

    print(f"  n = {n}, κ(A) = {eigvals[-1]/eigvals[0]:.0f}")
    print(f"  CG:  {info_cg['iterations']:>4} iters, rel_err = {err_cg:.2e}")
    print(f"  PCG: {info_pcg['iterations']:>4} iters, rel_err = {err_pcg:.2e}")
    print(f"  Speedup: {info_cg['iterations']/max(info_pcg['iterations'],1):.1f}×")


# =============================================================================
# Exercise 4: Sparse PD Benchmark
# =============================================================================

def sparse_benchmark():
    """Compare solvers on a large sparse PD system."""
    print("\n=== Exercise 4: Sparse PD Benchmark ===")
    n = 10000

    # 1D Laplacian (tridiagonal, PD): -1, 2, -1
    A_sparse = sparse.diags([-1, 2.001, -1], [-1, 0, 1],
                            shape=(n, n), format='csc')
    rng = np.random.default_rng(42)
    b = rng.randn(n)

    # Sparse direct solve
    t0 = time.perf_counter()
    x_direct = sp_linalg.spsolve(A_sparse, b)
    t_direct = time.perf_counter() - t0

    # SciPy CG
    t0 = time.perf_counter()
    x_scipy_cg, scipy_info = sp_linalg.cg(A_sparse, b, tol=1e-10, maxiter=500)
    t_scipy = time.perf_counter() - t0
    err_scipy = np.linalg.norm(x_scipy_cg - x_direct) / np.linalg.norm(x_direct)

    # Our CG (with sparse matvec)
    t0 = time.perf_counter()
    x_our_cg, our_info = conjugate_gradient(
        lambda v: A_sparse @ v, b, max_iter=500, tol=1e-10)
    t_our = time.perf_counter() - t0
    err_our = np.linalg.norm(x_our_cg - x_direct) / np.linalg.norm(x_direct)

    # PCG with Jacobi
    diag_vals = A_sparse.diagonal()
    M_inv = lambda v: v / diag_vals

    t0 = time.perf_counter()
    x_pcg, pcg_info = preconditioned_cg(
        lambda v: A_sparse @ v, b, M_inv, max_iter=500, tol=1e-10)
    t_pcg = time.perf_counter() - t0
    err_pcg = np.linalg.norm(x_pcg - x_direct) / np.linalg.norm(x_direct)

    print(f"  n = {n}")
    print(f"  Sparse direct:  {t_direct*1000:>8.1f} ms")
    print(f"  SciPy CG:       {t_scipy*1000:>8.1f} ms, "
          f"info={scipy_info}, err={err_scipy:.2e}")
    print(f"  Our CG:         {t_our*1000:>8.1f} ms, "
          f"{our_info['iterations']} iters, err={err_our:.2e}")
    print(f"  Our PCG:        {t_pcg*1000:>8.1f} ms, "
          f"{pcg_info['iterations']} iters, err={err_pcg:.2e}")


# =============================================================================
# A-conjugacy verification
# =============================================================================

def verify_a_conjugacy():
    """Verify that CG search directions are A-conjugate."""
    print("\n=== Bonus: A-Conjugacy Verification ===")
    n = 8
    rng = np.random.default_rng(42)
    raw = rng.randn(n, n)
    A = raw @ raw.T + np.eye(n)
    b = rng.randn(n)

    # Collect all search directions
    directions = []

    def record_matvec(v):
        return A @ v

    # Manual CG to extract directions
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rr = r @ r

    for k in range(n):
        directions.append(p.copy())
        Ap = A @ p
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        if np.sqrt(rr_new) < 1e-12:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    # Check A-conjugacy: p_i^T A p_j should be ~0 for i ≠ j
    P = np.array(directions).T  # n × k
    gram = P.T @ A @ P
    np.fill_diagonal(gram, 0)  # zero out diagonal
    max_off = np.max(np.abs(gram))
    print(f"  {len(directions)} directions collected")
    print(f"  Max off-diagonal of P^T A P: {max_off:.2e}")
    print(f"  A-conjugacy verified: {max_off < 1e-8}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_cg_exact()
    convergence_vs_kappa()
    test_preconditioning()
    sparse_benchmark()
    verify_a_conjugacy()
