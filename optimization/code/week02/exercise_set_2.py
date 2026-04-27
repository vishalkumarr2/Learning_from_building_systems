"""
Day 14: Week 2 Comprehensive Exercise Set — Companion Code
Phase I, Week 2

Covers: gradient checking, sparse tridiagonal, AD Jacobians,
condition number study, preconditioned CG, Schur complement.

Run: python exercise_set_2.py
"""

import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


# =============================================================================
# Problem 1: Gradient Checker
# =============================================================================

def check_gradient(f, grad_f, x, h=1e-7, tol=1e-5):
    """Check analytic gradient against central differences.
    
    Returns:
        passed: bool
        errors: array of relative errors per component
    """
    n = len(x)
    g_analytic = grad_f(x)
    g_numerical = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        g_numerical[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    # Relative error with denominator safety
    denom = np.maximum(np.abs(g_analytic) + np.abs(g_numerical), 1e-15)
    errors = np.abs(g_analytic - g_numerical) / denom
    passed = np.all(errors < tol)

    return passed, errors


def problem_1():
    """Gradient checker demo."""
    print("=== Problem 1: Gradient Checker ===")

    rng = np.random.default_rng(42)
    A = rng.randn(5, 3)
    b = rng.randn(5)

    f = lambda x: np.linalg.norm(A @ x - b) ** 2
    grad_f_correct = lambda x: 2 * A.T @ (A @ x - b)
    grad_f_buggy = lambda x: -2 * A.T @ (A @ x - b)  # sign error

    x = rng.randn(3)

    passed, errors = check_gradient(f, grad_f_correct, x)
    print(f"  Correct gradient: PASS={passed}, max_err={np.max(errors):.2e}")

    passed, errors = check_gradient(f, grad_f_buggy, x)
    print(f"  Buggy gradient:   PASS={passed}, max_err={np.max(errors):.2e}")


# =============================================================================
# Problem 2: Sparse Tridiagonal System
# =============================================================================

def problem_2():
    """Sparse tridiagonal solve comparison."""
    print("\n=== Problem 2: Sparse Tridiagonal ===")
    n = 50000

    # Create tridiagonal: 4 on diag, -1 on off-diags
    A = sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format='csc')
    rng = np.random.default_rng(42)
    b = rng.randn(n)

    # (a) Sparse direct
    t0 = time.perf_counter()
    x_direct = sp_linalg.spsolve(A, b)
    t_direct = time.perf_counter() - t0

    # (b) SciPy CG
    t0 = time.perf_counter()
    x_cg, info_cg = sp_linalg.cg(A, b, tol=1e-10, maxiter=1000)
    t_cg = time.perf_counter() - t0
    err_cg = np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct)

    # (c) PCG with Jacobi
    diag_vals = A.diagonal()
    M_inv = sp_linalg.LinearOperator((n, n), matvec=lambda v: v / diag_vals)
    t0 = time.perf_counter()
    x_pcg, info_pcg = sp_linalg.cg(A, b, M=M_inv, tol=1e-10, maxiter=1000)
    t_pcg = time.perf_counter() - t0
    err_pcg = np.linalg.norm(x_pcg - x_direct) / np.linalg.norm(x_direct)

    print(f"  n = {n}")
    print(f"  Direct:  {t_direct*1000:>8.1f} ms")
    print(f"  CG:      {t_cg*1000:>8.1f} ms, info={info_cg}, err={err_cg:.2e}")
    print(f"  PCG:     {t_pcg*1000:>8.1f} ms, info={info_pcg}, err={err_pcg:.2e}")


# =============================================================================
# Problem 3: Forward-Mode AD Jacobian
# =============================================================================

class Dual:
    """Dual number for forward-mode AD."""
    def __init__(self, val, deriv=0.0):
        self.val = val
        self.deriv = deriv

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val + other.val, self.deriv + other.deriv)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val * other.val,
                    self.val * other.deriv + self.deriv * other.val)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val / other.val,
                    (self.deriv * other.val - self.val * other.deriv) / other.val**2)

    def __pow__(self, n):
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.deriv)

    def __neg__(self):
        return Dual(-self.val, -self.deriv)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other


def dual_sin(d):
    return Dual(np.sin(d.val), np.cos(d.val) * d.deriv)

def dual_cos(d):
    return Dual(np.cos(d.val), -np.sin(d.val) * d.deriv)

def dual_exp(d):
    e = np.exp(d.val)
    return Dual(e, e * d.deriv)


def f_vec(x1, x2, x3):
    """Vector function for Jacobian test."""
    f1 = x1**2 + dual_sin(x2 * x3)
    f2 = dual_exp(x1) * dual_cos(x2)
    f3 = x3 / (1 + x1**2)
    return [f1, f2, f3]


def problem_3():
    """Forward-mode AD Jacobian."""
    print("\n=== Problem 3: AD Jacobian ===")
    x = [1.0, 2.0, 3.0]

    # Forward-mode AD: seed each input direction
    J_ad = np.zeros((3, 3))
    for j in range(3):
        duals = [Dual(x[i], 1.0 if i == j else 0.0) for i in range(3)]
        result = f_vec(duals[0], duals[1], duals[2])
        for i in range(3):
            J_ad[i, j] = result[i].deriv

    # Numerical Jacobian (central differences)
    def f_numpy(xv):
        return np.array([
            xv[0]**2 + np.sin(xv[1] * xv[2]),
            np.exp(xv[0]) * np.cos(xv[1]),
            xv[2] / (1 + xv[0]**2),
        ])

    J_num = np.zeros((3, 3))
    h = 1e-7
    xv = np.array(x)
    for j in range(3):
        ep = xv.copy(); em = xv.copy()
        ep[j] += h; em[j] -= h
        J_num[:, j] = (f_numpy(ep) - f_numpy(em)) / (2 * h)

    print("  AD Jacobian:")
    for row in J_ad:
        print(f"    [{', '.join(f'{v:>9.5f}' for v in row)}]")

    print("  Numerical Jacobian:")
    for row in J_num:
        print(f"    [{', '.join(f'{v:>9.5f}' for v in row)}]")

    print(f"  Max difference: {np.max(np.abs(J_ad - J_num)):.2e}")


# =============================================================================
# Problem 4: Condition Number Study
# =============================================================================

def simple_gd(H, b, max_iter=100000, tol=1e-6):
    """Gradient descent on 0.5 x^T H x - b^T x."""
    eigvals = np.linalg.eigvalsh(H)
    lr = 2.0 / (eigvals[-1] + eigvals[0])
    x_true = np.linalg.solve(H, b)

    x = np.zeros(len(b))
    for k in range(max_iter):
        g = H @ x - b
        if np.linalg.norm(x - x_true) / np.linalg.norm(x_true) < tol:
            return k
        x = x - lr * g
    return max_iter


def simple_cg(H, b, tol=1e-6):
    """CG iteration count to tolerance."""
    n = len(b)
    x_true = np.linalg.solve(H, b)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    rr = r @ r

    for k in range(2 * n):
        Hp = H @ p
        alpha = rr / (p @ Hp)
        x += alpha * p
        r -= alpha * Hp
        rr_new = r @ r
        if np.linalg.norm(x - x_true) / np.linalg.norm(x_true) < tol:
            return k + 1
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return 2 * n


def problem_4():
    """Condition number vs convergence study."""
    print("\n=== Problem 4: Condition Number Study ===")
    n = 100
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    b = rng.randn(n)

    print(f"  {'κ':>10} {'Digits':>8} {'GD iters':>10} {'CG iters':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for log_k in [1, 2, 3, 4, 5, 6]:
        kappa = 10 ** log_k
        eigvals = np.linspace(1, kappa, n)
        H = Q @ np.diag(eigvals) @ Q.T

        # Precision
        x_true = np.linalg.solve(H, b)
        x_check = np.linalg.solve(H, H @ x_true)
        rel_err = np.linalg.norm(x_check - x_true) / np.linalg.norm(x_true)
        digits = max(0, -np.log10(rel_err + 1e-20))

        # GD iterations (cap at 50000)
        gd_iters = simple_gd(H, b, max_iter=50000)

        # CG iterations
        cg_iters = simple_cg(H, b)

        print(f"  10^{log_k:<5} {digits:>8.1f} {gd_iters:>10} {cg_iters:>10}")


# =============================================================================
# Problem 5: Preconditioning Deep Dive
# =============================================================================

def problem_5():
    """PCG with various preconditioners."""
    print("\n=== Problem 5: Preconditioning Deep Dive ===")
    n = 100
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    eigvals = np.logspace(0, 4, n)  # κ = 10^4
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.randn(n)
    x_true = np.linalg.solve(A, b)

    # No preconditioning
    x1, info1 = sp_linalg.cg(A, b, tol=1e-10, maxiter=n)
    err1 = np.linalg.norm(x1 - x_true) / np.linalg.norm(x_true)

    # Jacobi preconditioning
    diag_A = np.diag(A)
    M_jacobi = sp_linalg.LinearOperator(
        (n, n), matvec=lambda v: v / diag_A)
    x2, info2 = sp_linalg.cg(A, b, M=M_jacobi, tol=1e-10, maxiter=n)
    err2 = np.linalg.norm(x2 - x_true) / np.linalg.norm(x_true)

    # "Perfect" preconditioning: M = A
    A_inv = np.linalg.inv(A)
    M_perfect = sp_linalg.LinearOperator(
        (n, n), matvec=lambda v: A_inv @ v)
    x3, info3 = sp_linalg.cg(A, b, M=M_perfect, tol=1e-10, maxiter=n)
    err3 = np.linalg.norm(x3 - x_true) / np.linalg.norm(x_true)

    # Effective condition numbers
    kappa_none = eigvals[-1] / eigvals[0]
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_A))
    A_jacobi = D_inv_sqrt @ A @ D_inv_sqrt
    ev_jacobi = np.linalg.eigvalsh(A_jacobi)
    kappa_jacobi = ev_jacobi[-1] / ev_jacobi[0]
    kappa_perfect = 1.0  # M^{-1}A = I

    print(f"  κ(A)       = {kappa_none:.0f}")
    print(f"  No precond:  info={info1}, err={err1:.2e}")
    print(f"  Jacobi:      info={info2}, err={err2:.2e}, κ_eff={kappa_jacobi:.0f}")
    print(f"  Perfect:     info={info3}, err={err3:.2e}, κ_eff={kappa_perfect:.0f}")


# =============================================================================
# Problem 6: Schur Complement Solver
# =============================================================================

def problem_6():
    """Schur complement solve for BA-like structure."""
    print("\n=== Problem 6: Schur Complement ===")
    rng = np.random.default_rng(42)

    nc, nl = 10, 90  # camera params, landmark params
    n = nc + nl

    # B: camera-camera block (dense SPD)
    B_raw = rng.randn(nc, nc)
    B = B_raw @ B_raw.T + 5 * np.eye(nc)

    # C: block-diagonal landmark block (each 2x2 SPD)
    C = np.zeros((nl, nl))
    for i in range(0, nl, 2):
        c_raw = rng.randn(2, 2)
        C[i:i+2, i:i+2] = c_raw @ c_raw.T + 3 * np.eye(2)

    # E: camera-landmark coupling
    E = rng.randn(nc, nl) * 0.5

    # Full system
    H = np.block([[B, E], [E.T, C]])
    g = rng.randn(n)

    # (a) Full dense solve
    x_full = np.linalg.solve(H, g)

    # (b) Schur complement: eliminate landmarks
    # S = B - E C^{-1} E^T
    C_inv = np.linalg.inv(C)
    S = B - E @ C_inv @ E.T

    # Reduced RHS: g_c - E C^{-1} g_l
    g_c = g[:nc]
    g_l = g[nc:]
    g_reduced = g_c - E @ C_inv @ g_l

    # Solve for cameras
    x_c = np.linalg.solve(S, g_reduced)

    # Back-substitute for landmarks
    x_l = C_inv @ (g_l - E.T @ x_c)

    x_schur = np.concatenate([x_c, x_l])

    err = np.linalg.norm(x_schur - x_full) / np.linalg.norm(x_full)
    print(f"  Full system: {n}×{n}")
    print(f"  Schur complement: {nc}×{nc}")
    print(f"  Solutions match: {err < 1e-10} (err = {err:.2e})")


# =============================================================================
# Problem 8: Mini-SLAM Integration
# =============================================================================

def problem_8():
    """Mini-SLAM: assemble info matrix, fix gauge, solve."""
    print("\n=== Problem 8: Mini-SLAM Integration ===")
    rng = np.random.default_rng(42)

    n_poses = 5
    n_landmarks = 20
    pose_dim = 3  # [x, y, theta]
    lm_dim = 2    # [lx, ly]
    n_total = n_poses * pose_dim + n_landmarks * lm_dim

    # Generate random poses and landmarks
    poses = np.cumsum(rng.randn(n_poses, pose_dim) * [1.0, 1.0, 0.1], axis=0)
    landmarks = rng.randn(n_landmarks, 2) * 5

    # Build information matrix from observations
    H = np.zeros((n_total, n_total))
    g = np.zeros(n_total)

    # Odometry factors (between consecutive poses)
    sigma_odom = np.diag([0.1, 0.1, 0.01])
    info_odom = np.linalg.inv(sigma_odom @ sigma_odom)

    for i in range(n_poses - 1):
        idx = slice(i * pose_dim, (i + 2) * pose_dim)
        # Simplified: info matrix connects pose i and i+1
        J_local = np.zeros((pose_dim, 2 * pose_dim))
        J_local[:, :pose_dim] = -np.eye(pose_dim)
        J_local[:, pose_dim:] = np.eye(pose_dim)
        H[idx, idx] += J_local.T @ info_odom @ J_local

    # Observation factors (each pose observes some landmarks)
    sigma_obs = np.diag([0.05, 0.05])
    info_obs = np.linalg.inv(sigma_obs @ sigma_obs)

    for i in range(n_poses):
        pose_idx = i * pose_dim
        for j in range(n_landmarks):
            if rng.random() < 0.3:  # 30% visibility
                lm_idx = n_poses * pose_dim + j * lm_dim
                # Simplified bearing-range Jacobian
                J = np.zeros((lm_dim, pose_dim + lm_dim))
                J[:, :2] = -np.eye(2)  # d/d(pose x,y)
                J[:, pose_dim:pose_dim + lm_dim] = np.eye(2)  # d/d(landmark)

                # Add to H
                rows = list(range(pose_idx, pose_idx + 2)) + \
                       list(range(lm_idx, lm_idx + lm_dim))
                for ri, r in enumerate(rows):
                    for ci, c in enumerate(rows):
                        block = J.T @ info_obs @ J
                        if ri < block.shape[0] and ci < block.shape[1]:
                            H[r, c] += block[ri, ci]

    # Check conditioning before gauge fix
    eigvals_pre = np.linalg.eigvalsh(H)
    n_zero = np.sum(np.abs(eigvals_pre) < 1e-10)

    # Fix gauge: anchor pose 0
    for k in range(pose_dim):
        H[k, k] += 100.0  # strong prior on pose 0

    eigvals_post = np.linalg.eigvalsh(H)
    kappa_post = eigvals_post[-1] / max(eigvals_post[0], 1e-15)

    # Solve with CG
    g = rng.randn(n_total) * 0.1  # small perturbation as RHS
    x_direct = np.linalg.solve(H, g)
    x_cg, info_cg = sp_linalg.cg(H, g, tol=1e-10, maxiter=500)
    err = np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct)

    print(f"  System size: {n_total}×{n_total}")
    print(f"  Before gauge fix: {n_zero} near-zero eigenvalues")
    print(f"  After gauge fix: κ = {kappa_post:.0f}")
    print(f"  CG info={info_cg}, rel_error={err:.2e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
    problem_4()
    problem_5()
    problem_6()
    problem_8()
