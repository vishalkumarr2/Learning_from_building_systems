"""
Matrix Mastery — Python Implementations
==========================================
Companion code for Chapter 08: Matrix Essentials for Optimization.

Covers:
  - Matrix classification and property detection
  - Positive definiteness tests (6 methods)
  - Nearest PD matrix (Higham 1988)
  - SVD analysis and four fundamental subspaces
  - Condition number diagnostics
  - Tikhonov and truncated SVD regularization
  - L-curve method for choosing λ
  - Levenberg-Marquardt with conditioning diagnostics
  - Matrix doctor (comprehensive diagnosis)

Usage:
    python matrix_essentials.py          # Run all demos
    python matrix_essentials.py --test   # Run self-tests only
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import sys


# ============================================================
# 1. Matrix Classification
# ============================================================

def classify_matrix(A, tol=1e-10):
    """Classify a matrix's structural and spectral properties.
    
    Returns dict with keys: symmetric, orthogonal, pd, psd, indefinite,
    singular, condition_number, rank.
    """
    m, n = A.shape
    result = {}
    is_square = m == n

    result["symmetric"] = is_square and np.allclose(A, A.T, atol=tol)
    result["skew_symmetric"] = is_square and np.allclose(A, -A.T, atol=tol)

    if is_square:
        result["orthogonal"] = np.allclose(A.T @ A, np.eye(n), atol=tol)
    else:
        result["orthogonal"] = False

    U, s, Vt = np.linalg.svd(A)
    result["rank"] = int(np.sum(s > tol * s[0] * max(m, n)))
    result["singular"] = result["rank"] < min(m, n)
    result["condition_number"] = s[0] / s[-1] if s[-1] > tol else np.inf

    if result["symmetric"]:
        eigvals = np.linalg.eigvalsh(A)
        if np.all(eigvals > tol):
            result["pd"] = True
            result["psd"] = True
            result["indefinite"] = False
        elif np.all(eigvals > -tol):
            result["pd"] = False
            result["psd"] = True
            result["indefinite"] = False
        else:
            result["pd"] = False
            result["psd"] = False
            result["indefinite"] = True
    else:
        result["pd"] = None
        result["psd"] = None
        result["indefinite"] = None

    return result


# ============================================================
# 2. Six Tests for Positive Definiteness
# ============================================================

def test_pd_eigenvalues(A, tol=1e-10):
    """Test 1: All eigenvalues positive."""
    eigvals = np.linalg.eigvalsh(A)
    return bool(np.all(eigvals > tol)), eigvals


def test_pd_cholesky(A):
    """Test 2: Cholesky factorization succeeds."""
    try:
        L = np.linalg.cholesky(A)
        return True, L
    except np.linalg.LinAlgError:
        return False, None


def test_pd_sylvester(A, tol=1e-10):
    """Test 3: All leading principal minors positive."""
    n = A.shape[0]
    minors = []
    for k in range(1, n + 1):
        minor = np.linalg.det(A[:k, :k])
        minors.append(minor)
        if minor <= tol:
            return False, minors
    return True, minors


def test_pd_pivots(A, tol=1e-10):
    """Test 5: All pivots in Gaussian elimination are positive."""
    n = A.shape[0]
    U = A.astype(float).copy()
    pivots = []
    for k in range(n):
        if abs(U[k, k]) <= tol:
            return False, pivots
        pivots.append(U[k, k])
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k:] -= factor * U[k, k:]
    return all(p > tol for p in pivots), pivots


def test_pd_diagonal_dominance(A):
    """Test 6: Strictly diagonally dominant (sufficient, not necessary)."""
    n = A.shape[0]
    for i in range(n):
        if A[i, i] <= np.sum(np.abs(A[i, :])) - np.abs(A[i, i]):
            return False
    return True


def run_all_pd_tests(A):
    """Run all six PD tests and print results."""
    print("Positive Definiteness Tests")
    print("-" * 40)

    is_pd, eigvals = test_pd_eigenvalues(A)
    print(f"1. Eigenvalue test:    {'PASS' if is_pd else 'FAIL'}  λ = {np.round(eigvals, 4)}")

    is_pd, L = test_pd_cholesky(A)
    print(f"2. Cholesky test:      {'PASS' if is_pd else 'FAIL'}")

    is_pd, minors = test_pd_sylvester(A)
    print(f"3. Sylvester criterion: {'PASS' if is_pd else 'FAIL'}  minors = {[round(m, 4) for m in minors]}")

    is_pd, pivots = test_pd_pivots(A)
    print(f"5. Pivot test:         {'PASS' if is_pd else 'FAIL'}  pivots = {[round(p, 4) for p in pivots]}")

    is_dd = test_pd_diagonal_dominance(A)
    print(f"6. Diag. dominance:    {'PASS' if is_dd else 'FAIL (not necessary)'}")


# ============================================================
# 3. Nearest PD Matrix (Higham 1988)
# ============================================================

def nearest_pd(A, epsilon=1e-6):
    """Nearest positive definite matrix via eigenvalue clipping."""
    A_sym = (A + A.T) / 2
    eigvals, Q = np.linalg.eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, epsilon)
    A_pd = Q @ np.diag(eigvals_clipped) @ Q.T
    return (A_pd + A_pd.T) / 2


# ============================================================
# 4. SVD Analysis
# ============================================================

def svd_analysis(A, tol=None, plot=True, save_path=None):
    """Comprehensive SVD analysis: rank, condition, subspaces, energy."""
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    if tol is None:
        tol = max(m, n) * s[0] * np.finfo(float).eps

    r = int(np.sum(s > tol))
    total_energy = np.sum(s ** 2)
    energy = np.cumsum(s ** 2) / total_energy if total_energy > 0 else np.zeros_like(s)

    print("=" * 60)
    print(f"SVD Analysis: {m}×{n} matrix")
    print("=" * 60)
    print(f"Rank: {r} / {min(m, n)}")
    kappa = s[0] / s[min(m, n) - 1] if s[min(m, n) - 1] > 0 else np.inf
    print(f"Condition number: {kappa:.4e}")

    print(f"\nSingular values (top {min(10, len(s))}):")
    for i in range(min(10, len(s))):
        bar = "█" * max(1, int(40 * s[i] / s[0]))
        pct = energy[i] * 100 if i < len(energy) else 100.0
        print(f"  σ_{i + 1:2d} = {s[i]:12.6e}  {bar}  ({pct:.1f}%)")

    print(f"\nFour Fundamental Subspaces:")
    print(f"  C(A):  dim {r}     | N(A):  dim {n - r}")
    print(f"  C(Aᵀ): dim {r}     | N(Aᵀ): dim {m - r}")

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.bar(range(1, len(s) + 1), s, color="steelblue")
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Singular Value")
        ax1.set_title("Singular Value Spectrum")
        ax1.set_yscale("log")

        ax2.plot(range(1, len(s) + 1), energy * 100, "ro-")
        ax2.axhline(y=99, color="gray", linestyle="--", alpha=0.5, label="99%")
        ax2.set_xlabel("Components")
        ax2.set_ylabel("Cumulative Energy (%)")
        ax2.set_title("Energy Distribution")
        ax2.legend()

        plt.tight_layout()
        path = save_path or "svd_analysis.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\nPlot saved to {path}")

    return {"U": U, "s": s, "Vt": Vt, "rank": r, "condition": kappa, "energy": energy}


# ============================================================
# 5. Regularization
# ============================================================

def tikhonov_solve(A, b, lam):
    """Tikhonov-regularized least-squares: min ‖Ax-b‖² + λ‖x‖²."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


def truncated_svd_solve(A, b, k):
    """Truncated SVD solution keeping top k singular values."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    x = np.zeros(A.shape[1])
    for i in range(min(k, len(s))):
        if s[i] > 1e-15:
            x += (U[:, i] @ b / s[i]) * Vt[i, :]
    return x


def l_curve(A, b, lambdas=None, save_path=None):
    """Plot L-curve and return optimal λ at maximum curvature."""
    if lambdas is None:
        lambdas = np.logspace(-16, 2, 300)

    residuals = []
    sol_norms = []
    for lam in lambdas:
        x_lam = tikhonov_solve(A, b, lam)
        residuals.append(np.linalg.norm(A @ x_lam - b))
        sol_norms.append(np.linalg.norm(x_lam))

    residuals = np.array(residuals)
    sol_norms = np.array(sol_norms)

    # Curvature in log-log space
    lr = np.log(residuals + 1e-30)
    ln = np.log(sol_norms + 1e-30)
    dr = np.gradient(lr)
    dn = np.gradient(ln)
    ddr = np.gradient(dr)
    ddn = np.gradient(dn)
    curvature = np.abs(dr * ddn - dn * ddr) / (dr ** 2 + dn ** 2 + 1e-30) ** 1.5
    margin = max(10, len(lambdas) // 20)
    corner_idx = np.argmax(curvature[margin:-margin]) + margin

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(residuals, sol_norms, "b-", linewidth=1.5)
    ax.plot(residuals[corner_idx], sol_norms[corner_idx], "ro", markersize=10)
    ax.annotate(
        f"λ = {lambdas[corner_idx]:.2e}",
        xy=(residuals[corner_idx], sol_norms[corner_idx]),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=10,
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xlabel(r"$\|Ax_\lambda - b\|_2$")
    ax.set_ylabel(r"$\|x_\lambda\|_2$")
    ax.set_title("L-Curve for Tikhonov Regularization")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_path or "l_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"L-curve saved to {path}. Optimal λ ≈ {lambdas[corner_idx]:.2e}")

    return lambdas[corner_idx]


# ============================================================
# 6. Levenberg-Marquardt with Diagnostics
# ============================================================

def levenberg_marquardt(residual_fn, jacobian_fn, x0, max_iter=100,
                        lam0=1e-3, tol=1e-10):
    """Simplified LM solver with condition-number tracking."""
    x = x0.astype(float).copy()
    lam = lam0
    n = len(x)
    history = {"cost": [], "lambda": [], "kappa_H": [], "kappa_damped": []}

    for it in range(max_iter):
        r = residual_fn(x)
        J = jacobian_fn(x)
        cost = 0.5 * r @ r
        H = J.T @ J
        g = J.T @ r

        kappa_H = np.linalg.cond(H)
        kappa_d = np.linalg.cond(H + lam * np.eye(n))
        history["cost"].append(cost)
        history["lambda"].append(lam)
        history["kappa_H"].append(kappa_H)
        history["kappa_damped"].append(kappa_d)

        dx = np.linalg.solve(H + lam * np.eye(n), -g)
        x_new = x + dx
        cost_new = 0.5 * residual_fn(x_new) @ residual_fn(x_new)
        gain = 0.5 * dx @ (lam * dx - g)
        rho = (cost - cost_new) / (gain + 1e-30)

        if rho > 0:
            x = x_new
            lam = max(lam / 3, 1e-15)
            if np.linalg.norm(dx) < tol:
                break
        else:
            lam = min(lam * 2, 1e10)

    return x, history


def demo_lm_curve_fit(save_path=None):
    """Demo: fit y = a·exp(-b·x) + c with LM."""
    np.random.seed(42)
    x_data = np.linspace(0, 5, 50)
    a_true, b_true, c_true = 3.0, 1.5, 0.5
    y_data = a_true * np.exp(-b_true * x_data) + c_true + 0.1 * np.random.randn(50)

    def residual(p):
        return p[0] * np.exp(-p[1] * x_data) + p[2] - y_data

    def jacobian(p):
        J = np.zeros((len(x_data), 3))
        e = np.exp(-p[1] * x_data)
        J[:, 0] = e
        J[:, 1] = -p[0] * x_data * e
        J[:, 2] = 1.0
        return J

    p0 = np.array([1.0, 0.5, 0.0])
    p_opt, hist = levenberg_marquardt(residual, jacobian, p0)
    print(f"LM result: a={p_opt[0]:.4f}, b={p_opt[1]:.4f}, c={p_opt[2]:.4f}")
    print(f"Truth:     a={a_true}, b={b_true}, c={c_true}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].semilogy(hist["cost"])
    axes[0, 0].set_title("Cost")
    axes[0, 1].semilogy(hist["lambda"])
    axes[0, 1].set_title("λ (damping)")
    axes[1, 0].semilogy(hist["kappa_H"], label="κ(H)")
    axes[1, 0].semilogy(hist["kappa_damped"], label="κ(H+λI)")
    axes[1, 0].legend()
    axes[1, 0].set_title("Condition Number")
    axes[1, 1].scatter(x_data, y_data, s=10, alpha=0.5, label="Data")
    axes[1, 1].plot(
        x_data,
        p_opt[0] * np.exp(-p_opt[1] * x_data) + p_opt[2],
        "r-",
        lw=2,
        label="Fit",
    )
    axes[1, 1].legend()
    axes[1, 1].set_title("Curve Fit")
    for ax in axes.flat:
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    path = save_path or "lm_diagnostics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"LM diagnostics saved to {path}")


# ============================================================
# 7. Matrix Doctor
# ============================================================

def matrix_doctor(A, verbose=True):
    """Comprehensive matrix diagnosis with solver recommendation."""
    m, n = A.shape
    is_square = m == n
    report = {"shape": (m, n), "square": is_square}

    report["symmetric"] = is_square and np.allclose(A, A.T, atol=1e-12)
    report["orthogonal"] = is_square and np.allclose(A.T @ A, np.eye(n), atol=1e-10) if is_square else False
    report["diagonal"] = is_square and np.allclose(A, np.diag(np.diag(A)), atol=1e-12)

    nnz = np.count_nonzero(np.abs(A) > 1e-15)
    report["sparsity"] = 1 - nnz / (m * n)
    report["sparse"] = report["sparsity"] > 0.5

    _, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(m, n) * s[0] * np.finfo(float).eps
    r = int(np.sum(s > tol))
    report["rank"] = r
    report["full_rank"] = r == min(m, n)
    report["condition_number"] = s[0] / s[r - 1] if r > 0 else np.inf

    if report["symmetric"]:
        eigvals = np.linalg.eigvalsh(A)
        if np.all(eigvals > tol):
            report["definiteness"] = "positive definite"
        elif np.all(eigvals > -tol):
            report["definiteness"] = "positive semidefinite"
        elif np.all(eigvals < -tol):
            report["definiteness"] = "negative definite"
        else:
            report["definiteness"] = "indefinite"

    warnings = []
    kappa = report["condition_number"]
    if kappa > 1e12:
        warnings.append(f"SEVERELY ILL-CONDITIONED (κ={kappa:.1e})")
    elif kappa > 1e6:
        warnings.append(f"ILL-CONDITIONED (κ={kappa:.1e})")
    if not report["full_rank"]:
        warnings.append(f"RANK DEFICIENT: {r}/{min(m, n)}")
    if report.get("definiteness") == "indefinite":
        warnings.append("INDEFINITE — not a valid minimum Hessian")
    report["warnings"] = warnings

    # Solver recommendation
    if not is_square:
        solver = "QR (overdetermined)" if m > n else "SVD (underdetermined)"
        if not report["full_rank"]:
            solver = "SVD / pseudoinverse (rank-deficient)"
    elif report.get("definiteness") == "positive definite":
        solver = "Sparse Cholesky" if report["sparse"] and n > 100 else "Dense Cholesky"
    elif report.get("definiteness") == "positive semidefinite":
        solver = "LDLᵀ or regularised Cholesky"
    elif report.get("definiteness") == "indefinite":
        solver = "Modified Cholesky or LDLᵀ with pivoting"
    else:
        solver = "LU with partial pivoting"
    report["solver"] = solver

    if verbose:
        print("=" * 60)
        print("MATRIX DOCTOR REPORT")
        print("=" * 60)
        tags = [k for k in ("symmetric", "orthogonal", "diagonal", "sparse") if report.get(k)]
        print(f"Shape: {m}×{n} | Properties: {', '.join(tags) or 'general'}")
        if "definiteness" in report:
            print(f"Definiteness: {report['definiteness']}")
        print(f"Rank: {r}/{min(m, n)} | κ = {kappa:.2e}")
        for w in warnings:
            print(f"  ⚠ {w}")
        print(f"→ Recommended solver: {solver}")
        print("=" * 60)

    return report


# ============================================================
# Self-tests
# ============================================================

def run_tests():
    """Verify implementations against known results."""
    passed = 0
    failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")

    print("Running self-tests...\n")

    # classify_matrix
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
    c = classify_matrix(A)
    check("classify: symmetric PD", c["symmetric"] and c["pd"])

    Q = np.array([[1, 2], [2, 3]])
    c = classify_matrix(Q)
    check("classify: indefinite", c["indefinite"])

    R = np.array([[0, -1], [1, 0]], dtype=float)
    c = classify_matrix(R)
    check("classify: orthogonal", c["orthogonal"])

    # PD tests
    check("pd_eigenvalues", test_pd_eigenvalues(A)[0])
    check("pd_cholesky", test_pd_cholesky(A)[0])
    check("pd_sylvester", test_pd_sylvester(A)[0])
    check("pd_pivots", test_pd_pivots(A)[0])

    check("not_pd_eigenvalues", not test_pd_eigenvalues(Q)[0])
    check("not_pd_cholesky", not test_pd_cholesky(Q)[0])
    check("not_pd_sylvester", not test_pd_sylvester(Q)[0])

    # nearest_pd
    bad = np.array([[4.0, 1.0, 3.1], [1.0, 2.0, -0.5], [3.1, -0.5, 1.0]])
    fixed = nearest_pd(bad)
    check("nearest_pd: result is PD", test_pd_cholesky(fixed)[0])

    # tikhonov
    A2 = np.diag([1.0, 1e-8])
    b2 = A2 @ np.array([1.0, 1.0])
    x_tikh = tikhonov_solve(A2, b2, lam=0)
    check("tikhonov(λ=0) recovers exact", np.allclose(x_tikh, [1, 1], atol=1e-6))

    # truncated_svd
    A3 = np.array([[1, 1], [1, 1], [0, 0]], dtype=float)
    b3 = np.array([3.0, 3.0, 0.0])
    x_tsvd = truncated_svd_solve(A3, b3, k=1)
    check("tsvd solution", np.allclose(A3 @ x_tsvd, [3, 3, 0], atol=1e-10))

    # matrix_doctor
    report = matrix_doctor(A, verbose=False)
    check("doctor: PD detected", report.get("definiteness") == "positive definite")
    check("doctor: cholesky recommended", "Cholesky" in report["solver"])

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)

    print("=" * 60)
    print("Matrix Essentials — Demo Suite")
    print("=" * 60)

    # Demo 1: PD tests
    print("\n--- Demo 1: Positive Definiteness Tests ---")
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
    run_all_pd_tests(A)

    # Demo 2: Nearest PD
    print("\n--- Demo 2: Nearest PD Matrix ---")
    bad = np.array([[4.0, 1.0, 3.1], [1.0, 2.0, -0.5], [3.1, -0.5, 1.0]])
    print(f"Original eigenvalues: {np.round(np.linalg.eigvalsh(bad), 4)}")
    fixed = nearest_pd(bad)
    print(f"Fixed eigenvalues:    {np.round(np.linalg.eigvalsh(fixed), 4)}")

    # Demo 3: SVD analysis
    print("\n--- Demo 3: SVD Analysis ---")
    np.random.seed(42)
    J = np.random.randn(8, 5) @ np.random.randn(5, 6)
    svd_analysis(J, save_path="demo_svd.png")

    # Demo 4: Regularization comparison
    print("\n--- Demo 4: Regularization ---")
    np.random.seed(42)
    n = 15
    s_vals = np.logspace(0, -8, n)
    U0, _ = np.linalg.qr(np.random.randn(n, n))
    V0, _ = np.linalg.qr(np.random.randn(n, n))
    A_ill = U0 @ np.diag(s_vals) @ V0.T
    x_true = np.ones(n)
    b_ill = A_ill @ x_true + 1e-6 * np.random.randn(n)
    print(f"κ(A) = {np.linalg.cond(A_ill):.2e}")

    x_direct = np.linalg.solve(A_ill, b_ill)
    x_tikh = tikhonov_solve(A_ill, b_ill, lam=1e-6)
    x_tsvd = truncated_svd_solve(A_ill, b_ill, k=8)
    print(f"Direct error:  {np.linalg.norm(x_direct - x_true):.4e}")
    print(f"Tikhonov:      {np.linalg.norm(x_tikh - x_true):.4e}")
    print(f"Truncated SVD: {np.linalg.norm(x_tsvd - x_true):.4e}")

    lam_opt = l_curve(A_ill, b_ill, save_path="demo_l_curve.png")

    # Demo 5: LM
    print("\n--- Demo 5: Levenberg-Marquardt ---")
    demo_lm_curve_fit(save_path="demo_lm.png")

    # Demo 6: Matrix doctor
    print("\n--- Demo 6: Matrix Doctor ---")
    matrix_doctor(A_ill)

    print("\nAll demos complete. Check *.png for plots.")
