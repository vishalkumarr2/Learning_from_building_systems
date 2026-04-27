"""
Day 8: Matrix Decompositions for Optimization — Companion Code
Phase I, Week 2

Run: python matrix_decompositions.py
"""

import numpy as np
import time


# =============================================================================
# Exercise 1: Cholesky from Scratch
# =============================================================================

def cholesky_manual(A: np.ndarray) -> np.ndarray:
    """Cholesky decomposition: A = L L^T for PD matrix A.
    
    Column-by-column algorithm.
    Raises ValueError if A is not positive definite.
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    for j in range(n):
        # Diagonal entry
        val = A[j, j] - np.sum(L[j, :j]**2)
        if val <= 0:
            raise ValueError(f"Matrix is not PD: pivot {j} = {val:.2e}")
        L[j, j] = np.sqrt(val)

        # Below-diagonal entries in column j
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L


def test_cholesky():
    """Verify against numpy."""
    print("=== Exercise 1: Cholesky from Scratch ===")
    rng = np.random.default_rng(42)
    A_raw = rng.randn(5, 5)
    A = A_raw @ A_raw.T + 0.1 * np.eye(5)  # ensure PD

    L_manual = cholesky_manual(A)
    L_numpy = np.linalg.cholesky(A)

    error = np.linalg.norm(L_manual - L_numpy)
    recon = np.linalg.norm(L_manual @ L_manual.T - A)
    print(f"  ||L_manual - L_numpy|| = {error:.2e}")
    print(f"  ||L L^T - A||         = {recon:.2e}")

    # Test non-PD detection
    A_bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
    try:
        cholesky_manual(A_bad)
        print("  Non-PD test: FAILED (should have raised)")
    except ValueError as e:
        print(f"  Non-PD detection: PASS ✓ ({e})")


# =============================================================================
# Exercise 2: Solve via Cholesky (Forward + Back Substitution)
# =============================================================================

def forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ly = b where L is lower triangular."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_sub(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve Ux = y where U is upper triangular."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b via Cholesky: L L^T x = b."""
    L = cholesky_manual(A)
    y = forward_sub(L, b)
    x = back_sub(L.T, y)
    return x


def test_cholesky_solve():
    """Compare accuracy with np.linalg.solve."""
    print("\n=== Exercise 2: Solve via Cholesky ===")
    rng = np.random.default_rng(42)
    n = 50
    A_raw = rng.randn(n, n)
    A = A_raw @ A_raw.T + 0.1 * np.eye(n)
    b = rng.randn(n)

    x_chol = cholesky_solve(A, b)
    x_numpy = np.linalg.solve(A, b)

    error = np.linalg.norm(x_chol - x_numpy)
    residual = np.linalg.norm(A @ x_chol - b)
    print(f"  ||x_chol - x_numpy|| = {error:.2e}")
    print(f"  ||Ax - b||           = {residual:.2e}")


# =============================================================================
# Exercise 3: Benchmark Decompositions
# =============================================================================

def benchmark_solvers():
    """Time comparison: Cholesky vs solve vs inv."""
    print("\n=== Exercise 3: Benchmark ===")
    rng = np.random.default_rng(42)

    print(f"  {'n':>6} | {'Cholesky':>12} | {'np.solve':>12} | {'np.inv@b':>12}")
    print(f"  {'-'*52}")

    for n in [100, 500, 1000]:
        A_raw = rng.randn(n, n)
        A = A_raw @ A_raw.T + 0.1 * np.eye(n)
        b = rng.randn(n)

        # Cholesky via scipy
        from scipy.linalg import cho_factor, cho_solve
        t0 = time.perf_counter()
        c, low = cho_factor(A)
        x1 = cho_solve((c, low), b)
        t_chol = time.perf_counter() - t0

        # numpy solve
        t0 = time.perf_counter()
        x2 = np.linalg.solve(A, b)
        t_solve = time.perf_counter() - t0

        # inv (BAD practice)
        t0 = time.perf_counter()
        x3 = np.linalg.inv(A) @ b
        t_inv = time.perf_counter() - t0

        print(f"  {n:>6} | {t_chol:>10.4f}ms | {t_solve:>10.4f}ms | {t_inv:>10.4f}ms")


# =============================================================================
# Exercise 5: QR vs Normal Equations
# =============================================================================

def qr_vs_normal_equations():
    """Show QR is more accurate for ill-conditioned LS problems."""
    print("\n=== Exercise 5: QR vs Normal Equations ===")
    rng = np.random.default_rng(42)

    for kappa_target in [10, 1e4, 1e8, 1e12]:
        m, n = 200, 50
        U, _ = np.linalg.qr(rng.randn(m, m))
        V, _ = np.linalg.qr(rng.randn(n, n))
        s = np.logspace(0, np.log10(kappa_target), n)
        A = U[:, :n] @ np.diag(s) @ V.T

        x_true = rng.randn(n)
        b = A @ x_true + rng.randn(m) * 1e-10

        # Normal equations: (A^T A) x = A^T b
        try:
            x_ne = np.linalg.solve(A.T @ A, A.T @ b)
            err_ne = np.linalg.norm(x_ne - x_true) / np.linalg.norm(x_true)
        except np.linalg.LinAlgError:
            err_ne = float('inf')

        # QR
        x_qr, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        err_qr = np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true)

        print(f"  κ={kappa_target:.0e}: NormalEq err={err_ne:.2e}, QR err={err_qr:.2e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_cholesky()
    test_cholesky_solve()
    benchmark_solvers()
    qr_vs_normal_equations()
