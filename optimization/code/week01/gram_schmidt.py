"""
Day 1: Vectors, Matrices, Transformations — Companion Code
Phase I, Week 1

Run: python gram_schmidt.py
"""

import numpy as np


# =============================================================================
# Exercise 1: Gram-Schmidt Orthogonalization (from scratch)
# =============================================================================

def gram_schmidt(V: np.ndarray) -> np.ndarray:
    """
    Classical Gram-Schmidt orthogonalization.

    Args:
        V: Matrix whose columns are vectors to orthogonalize (n × k)

    Returns:
        Q: Matrix with orthonormal columns (n × k)
    """
    n, k = V.shape
    Q = np.zeros((n, k))

    for j in range(k):
        v = V[:, j].copy()
        # Subtract projections onto previous orthonormal vectors
        for i in range(j):
            v -= np.dot(Q[:, i], V[:, j]) * Q[:, i]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError(f"Column {j} is linearly dependent on previous columns")
        Q[:, j] = v / norm

    return Q


# =============================================================================
# Exercise 2: QR Decomposition via Gram-Schmidt
# =============================================================================

def qr_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition using Gram-Schmidt.

    A = QR where Q is orthogonal and R is upper triangular.
    """
    n, k = A.shape
    Q = gram_schmidt(A)

    # R = Q^T A
    R = Q.T @ A

    # Clean up: R should be upper triangular
    R = np.triu(R)

    return Q, R


# =============================================================================
# Exercise 3: Projection onto subspace
# =============================================================================

def project_onto_subspace(b: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Project vector b onto the column space of A.

    proj = A (A^T A)^{-1} A^T b
    """
    ATA = A.T @ A
    ATb = A.T @ b
    # Solve the normal equation A^T A x = A^T b
    x = np.linalg.solve(ATA, ATb)
    return A @ x


# =============================================================================
# Exercise 4: Verify against NumPy
# =============================================================================

def verify_gram_schmidt():
    """Compare our Gram-Schmidt with numpy.linalg.qr."""
    np.random.seed(42)
    A = np.random.randn(5, 3)

    # Our implementation
    Q_ours, R_ours = qr_gram_schmidt(A)

    # NumPy's QR (thin QR)
    Q_np, R_np = np.linalg.qr(A)

    # Q columns should span the same space (may differ by sign)
    print("=== QR Decomposition Verification ===")
    print(f"A shape: {A.shape}")
    print(f"Q shape: {Q_ours.shape}, R shape: {R_ours.shape}")

    # Check orthonormality: Q^T Q should be I
    QTQ = Q_ours.T @ Q_ours
    print(f"\nQ^T Q (should be I):\n{np.round(QTQ, 10)}")

    # Check reconstruction: QR should equal A
    reconstruction_error = np.linalg.norm(Q_ours @ R_ours - A)
    print(f"\n||QR - A|| = {reconstruction_error:.2e} (should be ~0)")

    # Check R is upper triangular
    print(f"\nR (should be upper triangular):\n{np.round(R_ours, 6)}")


def verify_projection():
    """Verify projection against numpy.linalg.lstsq."""
    np.random.seed(42)

    # Subspace defined by two columns
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0]])
    b = np.array([1.0, 2.0, 3.0])

    proj = project_onto_subspace(b, A)
    print("\n=== Projection Verification ===")
    print(f"b = {b}")
    print(f"Projection onto C(A) = {proj}")
    print(f"Expected: [1, 2, 0] (component in xy-plane)")

    # Verify: residual should be orthogonal to column space
    residual = b - proj
    print(f"Residual = {residual}")
    print(f"A^T @ residual = {A.T @ residual} (should be ~0)")


def demo_four_subspaces():
    """Demonstrate the four fundamental subspaces."""
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [1, 3, 4]])

    print("\n=== Four Fundamental Subspaces ===")
    print(f"A =\n{A}")
    print(f"Rank = {np.linalg.matrix_rank(A)}")

    # Null space via SVD
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)
    null_space = Vt[rank:].T

    print(f"\nSingular values: {np.round(S, 6)}")
    print(f"Null space basis (columns):\n{np.round(null_space, 6)}")

    # Verify: A @ null_vector should be 0
    if null_space.shape[1] > 0:
        test = A @ null_space[:, 0]
        print(f"A @ null_vector = {np.round(test, 10)} (should be ~0)")


if __name__ == "__main__":
    verify_gram_schmidt()
    verify_projection()
    demo_four_subspaces()
