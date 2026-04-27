"""
Day 2: Eigenvalues, Positive Definiteness, SVD — Companion Code
Phase I, Week 1

Run: python eigenvalues_svd.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Exercise 1: Power Iteration for dominant eigenvalue
# =============================================================================

def power_iteration(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-10
                    ) -> tuple[float, np.ndarray]:
    """
    Power iteration to find the dominant eigenvalue and eigenvector.
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    eigenvalue = 0.0
    for i in range(max_iter):
        Av = A @ v
        eigenvalue_new = v @ Av  # Rayleigh quotient
        v = Av / np.linalg.norm(Av)
        if abs(eigenvalue_new - eigenvalue) < tol:
            print(f"  Power iteration converged in {i+1} iterations")
            return eigenvalue_new, v
        eigenvalue = eigenvalue_new

    return eigenvalue, v


# =============================================================================
# Exercise 2: Positive Definiteness Classification
# =============================================================================

def classify_matrix(A: np.ndarray) -> str:
    """Classify a symmetric matrix as PD, PSD, ND, NSD, or Indefinite."""
    eigenvalues = np.linalg.eigvalsh(A)

    if np.all(eigenvalues > 1e-12):
        return "Positive Definite"
    elif np.all(eigenvalues >= -1e-12):
        return "Positive Semidefinite"
    elif np.all(eigenvalues < -1e-12):
        return "Negative Definite"
    elif np.all(eigenvalues <= 1e-12):
        return "Negative Semidefinite"
    else:
        return "Indefinite"


def check_pd_three_ways(A: np.ndarray) -> dict:
    """Check PD via eigenvalues, Cholesky, and quadratic form."""
    results = {}

    # Method 1: Eigenvalues
    eigvals = np.linalg.eigvalsh(A)
    results["eigenvalues"] = eigvals
    results["pd_by_eigenvalues"] = bool(np.all(eigvals > 1e-12))

    # Method 2: Cholesky attempt
    try:
        L = np.linalg.cholesky(A)
        results["pd_by_cholesky"] = True
    except np.linalg.LinAlgError:
        results["pd_by_cholesky"] = False

    # Method 3: Quadratic form with random vectors
    n_tests = 1000
    all_positive = True
    for _ in range(n_tests):
        x = np.random.randn(A.shape[0])
        if x @ A @ x <= 0:
            all_positive = False
            break
    results["pd_by_quadratic_form"] = all_positive

    return results


# =============================================================================
# Exercise 3: SVD and Low-Rank Approximation
# =============================================================================

def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """Compute rank-k approximation of A using SVD."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Keep only the top k components
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# =============================================================================
# Exercise 4: Condition Number and Error Amplification
# =============================================================================

def condition_number_demo(kappa_target: float = 100.0, n: int = 10):
    """Show how condition number amplifies perturbation errors."""
    # Create a PD matrix with specified condition number
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    lambdas = np.linspace(1.0, kappa_target, n)
    A = Q @ np.diag(lambdas) @ Q.T

    actual_kappa = np.linalg.cond(A)

    # Solve Ax = b
    x_true = np.random.randn(n)
    b = A @ x_true

    # Perturb b
    delta_b = np.random.randn(n) * 1e-8
    b_perturbed = b + delta_b

    x_perturbed = np.linalg.solve(A, b_perturbed)
    delta_x = x_perturbed - x_true

    # Error amplification
    relative_b_error = np.linalg.norm(delta_b) / np.linalg.norm(b)
    relative_x_error = np.linalg.norm(delta_x) / np.linalg.norm(x_true)
    amplification = relative_x_error / relative_b_error

    return {
        "kappa": actual_kappa,
        "relative_b_error": relative_b_error,
        "relative_x_error": relative_x_error,
        "amplification": amplification,
    }


# =============================================================================
# Main demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Power Iteration")
    print("=" * 60)
    A = np.array([[4, 1], [1, 3]], dtype=float)
    eigval, eigvec = power_iteration(A)
    print(f"  Dominant eigenvalue: {eigval:.6f}")
    print(f"  Eigenvector: {eigvec}")
    print(f"  NumPy eigenvalues: {np.linalg.eigvalsh(A)}")

    print("\n" + "=" * 60)
    print("Exercise 2: PD Classification")
    print("=" * 60)
    matrices = {
        "[[2,1],[1,2]]": np.array([[2, 1], [1, 2]], dtype=float),
        "[[1,2],[2,1]]": np.array([[1, 2], [2, 1]], dtype=float),
        "[[4,2],[2,1]]": np.array([[4, 2], [2, 1]], dtype=float),
        "diag(1,0,1)": np.diag([1.0, 0.0, 1.0]),
        "I_3": np.eye(3),
    }
    for name, M in matrices.items():
        classification = classify_matrix(M)
        results = check_pd_three_ways(M)
        print(f"\n  {name}: {classification}")
        print(f"    Eigenvalues: {np.round(results['eigenvalues'], 6)}")
        print(f"    Cholesky: {'✓' if results['pd_by_cholesky'] else '✗'}")

    print("\n" + "=" * 60)
    print("Exercise 3: Low-Rank Approximation")
    print("=" * 60)
    np.random.seed(42)
    A = np.random.randn(10, 8)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  Singular values: {np.round(S, 3)}")
    for k in [1, 2, 4, 8]:
        A_k = low_rank_approximation(A, k)
        error = np.linalg.norm(A - A_k, "fro")
        print(f"  Rank-{k} approx error: {error:.4f} (theory: {np.sqrt(sum(S[k:]**2)):.4f})")

    print("\n" + "=" * 60)
    print("Exercise 4: Condition Number → Error Amplification")
    print("=" * 60)
    for kappa in [2, 10, 100, 1000, 10000]:
        result = condition_number_demo(kappa)
        print(f"  κ={result['kappa']:.0f}: "
              f"δb/b={result['relative_b_error']:.2e}, "
              f"δx/x={result['relative_x_error']:.2e}, "
              f"amplification={result['amplification']:.1f}×")
