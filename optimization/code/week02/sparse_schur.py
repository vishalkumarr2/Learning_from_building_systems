"""
Day 9: Sparse Matrices & Schur Complement — Companion Code
Phase I, Week 2

Run: python sparse_schur.py
"""

import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


# =============================================================================
# Exercise 1: Tridiagonal System — Dense vs Sparse
# =============================================================================

def tridiagonal_benchmark():
    """Create and solve a tridiagonal PD system, dense vs sparse."""
    print("=== Exercise 1: Tridiagonal System ===")
    n = 3000

    # Build tridiagonal: 2 on diagonal, -1 on off-diagonals (PD)
    diag_main = 4.0 * np.ones(n)
    diag_off = -1.0 * np.ones(n - 1)

    # Dense version
    A_dense = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
    b = np.random.default_rng(42).randn(n)

    t0 = time.perf_counter()
    x_dense = np.linalg.solve(A_dense, b)
    t_dense = time.perf_counter() - t0

    # Sparse version
    A_sparse = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1],
                            shape=(n, n), format='csc')

    t0 = time.perf_counter()
    x_sparse = sp_linalg.spsolve(A_sparse, b)
    t_sparse = time.perf_counter() - t0

    error = np.linalg.norm(x_dense - x_sparse)
    print(f"  n = {n}")
    print(f"  Dense solve:  {t_dense*1000:.1f} ms, memory ~{A_dense.nbytes/1e6:.1f} MB")
    print(f"  Sparse solve: {t_sparse*1000:.1f} ms, memory ~{A_sparse.data.nbytes/1e3:.1f} KB")
    print(f"  Speedup: {t_dense/t_sparse:.1f}×")
    print(f"  Solution diff: {error:.2e}")


# =============================================================================
# Exercise 2: Sparsity Visualization
# =============================================================================

def sparsity_patterns():
    """Create and display various sparsity patterns."""
    print("\n=== Exercise 2: Sparsity Patterns ===")
    n = 20

    # Tridiagonal
    A_tri = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    print(f"  Tridiagonal {n}×{n}: nnz={A_tri.nnz}, density={A_tri.nnz/n**2:.3f}")

    # Block-diagonal (5 blocks of 4×4)
    blocks = [np.random.randn(4, 4) for _ in range(5)]
    A_blk = sparse.block_diag(blocks)
    print(f"  Block-diagonal {A_blk.shape}: nnz={A_blk.nnz}, density={A_blk.nnz/A_blk.shape[0]**2:.3f}")

    # Pose-graph: chain + loop closure
    rows, cols, data = [], [], []
    for i in range(n - 1):  # chain
        rows.extend([i, i+1, i, i+1])
        cols.extend([i, i+1, i+1, i])
        data.extend([2, 2, -1, -1])
    # Loop closure: connect first and last
    rows.extend([0, n-1, 0, n-1])
    cols.extend([0, n-1, n-1, 0])
    data.extend([1, 1, -1, -1])
    A_pg = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    print(f"  Pose-graph {n}×{n}: nnz={A_pg.nnz}, density={A_pg.nnz/n**2:.3f}")

    # Arrow-shaped (dense first row/col + diagonal)
    A_arrow = sparse.eye(n, format='lil') * 2.0
    for i in range(1, n):
        A_arrow[0, i] = 0.5
        A_arrow[i, 0] = 0.5
    A_arrow = A_arrow.tocsr()
    print(f"  Arrow {n}×{n}: nnz={A_arrow.nnz}, density={A_arrow.nnz/n**2:.3f}")


# =============================================================================
# Exercise 3: Schur Complement
# =============================================================================

def schur_complement_demo():
    """Schur complement elimination on a block system."""
    print("\n=== Exercise 3: Schur Complement ===")
    rng = np.random.default_rng(42)

    na, nb = 5, 10  # cameras (small), landmarks (large)
    n = na + nb

    # Build PD block system
    raw = rng.randn(n, n)
    H = raw @ raw.T + 5 * np.eye(n)
    g = rng.randn(n)

    # Extract blocks
    Haa = H[:na, :na]
    Hab = H[:na, na:]
    Hba = H[na:, :na]
    Hbb = H[na:, na:]
    ga = g[:na]
    gb = g[na:]

    # Full solve (reference)
    x_full = np.linalg.solve(H, g)
    xa_full = x_full[:na]
    xb_full = x_full[na:]

    # Schur complement: S = Haa - Hab @ Hbb^{-1} @ Hba
    Hbb_inv = np.linalg.inv(Hbb)
    S = Haa - Hab @ Hbb_inv @ Hba
    gs = ga - Hab @ Hbb_inv @ gb

    # Solve reduced system
    xa_schur = np.linalg.solve(S, gs)

    # Back-substitute
    xb_schur = Hbb_inv @ (gb - Hba @ xa_schur)

    err_a = np.linalg.norm(xa_schur - xa_full)
    err_b = np.linalg.norm(xb_schur - xb_full)
    print(f"  System: {n}×{n}, partitioned {na}+{nb}")
    print(f"  Schur complement S: {na}×{na}")
    print(f"  ||xa_schur - xa_full|| = {err_a:.2e}")
    print(f"  ||xb_schur - xb_full|| = {err_b:.2e}")

    # Block-diagonal Hbb demo (BA-style)
    print("\n  BA-style: Hbb block-diagonal (each landmark independent)")
    n_landmarks = 100
    landmark_dim = 3
    nb2 = n_landmarks * landmark_dim
    na2 = 12  # 2 cameras × 6 params

    Hbb_blocks = [rng.randn(3, 3) @ rng.randn(3, 3).T + np.eye(3)
                  for _ in range(n_landmarks)]
    Hbb_bd = sparse.block_diag(Hbb_blocks).toarray()

    t0 = time.perf_counter()
    # Full inversion of Hbb
    Hbb_inv_full = np.linalg.inv(Hbb_bd)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Block inversion (exploit structure)
    Hbb_inv_blocks = [np.linalg.inv(B) for B in Hbb_blocks]
    t_block = time.perf_counter() - t0

    print(f"  Full Hbb^{{-1}} ({nb2}×{nb2}): {t_full*1000:.2f} ms")
    print(f"  Block Hbb^{{-1}} ({n_landmarks}×3×3): {t_block*1000:.2f} ms")
    print(f"  Speedup: {t_full/max(t_block, 1e-9):.1f}×")


# =============================================================================
# Exercise 4: Fill-in with Ordering
# =============================================================================

def fill_in_demo():
    """Show fill-in depends on variable ordering."""
    print("\n=== Exercise 4: Fill-in with Ordering ===")

    # Arrow matrix: eliminating first variable creates massive fill-in
    n = 50
    A = sparse.eye(n, format='lil') * 3.0
    for i in range(1, n):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    A = A.tocsc()

    print(f"  Arrow matrix {n}×{n}: nnz(A) = {A.nnz}")

    # Natural ordering (bad: first variable connects to everything)
    from scipy.sparse.linalg import splu
    LU_natural = splu(A.tocsc())
    nnz_natural = LU_natural.L.nnz + LU_natural.U.nnz
    print(f"  Natural ordering: nnz(L+U) = {nnz_natural}")

    # Reverse ordering (good: eliminate leaves first)
    perm = np.arange(n-1, -1, -1)
    A_perm = A[perm][:, perm]
    LU_reverse = splu(A_perm.tocsc())
    nnz_reverse = LU_reverse.L.nnz + LU_reverse.U.nnz
    print(f"  Reverse ordering: nnz(L+U) = {nnz_reverse}")
    print(f"  Fill-in reduction: {nnz_natural - nnz_reverse} fewer nonzeros")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    tridiagonal_benchmark()
    sparsity_patterns()
    schur_complement_demo()
    fill_in_demo()
