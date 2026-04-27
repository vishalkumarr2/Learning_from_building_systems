"""
Day 25: Multi-Objective and Regularization — Companion Code
Phase II, Week 4

Run: python regularization.py
"""

import numpy as np


# =============================================================================
# Exercise 1: Ridge Regression
# =============================================================================

def ridge_closed_form(A, b, lam):
    """Solve ridge regression: min ||Ax - b||^2 + λ||x||^2."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


def ridge_iterative(A, b, lam, max_iter=2000, tol=1e-10):
    """Solve ridge regression via gradient descent."""
    n = A.shape[1]
    x = np.zeros(n)
    # f(x) = ||Ax - b||^2 + λ||x||^2
    # ∇f = 2(A^T A x - A^T b + λ x) = 2(A^T A + λI)x - 2A^T b
    H = A.T @ A + lam * np.eye(n)
    L = np.linalg.norm(H, 2)  # Lipschitz constant
    alpha = 1.0 / L

    for _ in range(max_iter):
        grad = 2 * (H @ x - A.T @ b)
        if np.linalg.norm(grad) < tol:
            break
        x = x - alpha * grad

    return x


def exercise_1():
    """Ridge regression: closed-form vs iterative."""
    print("=== Exercise 1: Ridge Regression ===")
    rng = np.random.RandomState(42)
    m, n = 100, 20
    A = rng.randn(m, n)
    x_true = rng.randn(n)
    b = A @ x_true + 0.5 * rng.randn(m)

    for lam in [0.01, 0.1, 1.0, 10.0]:
        x_cf = ridge_closed_form(A, b, lam)
        x_it = ridge_iterative(A, b, lam)
        print(f"  λ={lam:>5.2f}  ||x_cf||={np.linalg.norm(x_cf):>7.4f}  "
              f"||x_it||={np.linalg.norm(x_it):>7.4f}  "
              f"diff={np.linalg.norm(x_cf - x_it):>10.2e}  "
              f"||x-x*||={np.linalg.norm(x_cf - x_true):>7.4f}")


# =============================================================================
# Exercise 2: ISTA for LASSO
# =============================================================================

def soft_threshold(v, lam):
    """Proximal operator for L1 norm: soft-thresholding."""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)


def ista(A, b, lam, max_iter=5000, tol=1e-8):
    """
    ISTA for LASSO: min (1/2)||Ax - b||^2 + λ||x||_1.
    Proximal gradient: x_{k+1} = prox_{αλ||·||_1}(x_k - α A^T(Ax_k - b))
    """
    n = A.shape[1]
    x = np.zeros(n)
    L = np.linalg.norm(A.T @ A, 2)  # Lipschitz constant of smooth part
    alpha = 1.0 / L
    losses = []

    for _ in range(max_iter):
        residual = A @ x - b
        loss = 0.5 * np.linalg.norm(residual)**2 + lam * np.linalg.norm(x, 1)
        losses.append(loss)
        grad = A.T @ residual
        x = soft_threshold(x - alpha * grad, alpha * lam)

        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < tol:
            break

    return x, losses


def fista(A, b, lam, max_iter=5000, tol=1e-8):
    """FISTA: accelerated proximal gradient for LASSO."""
    n = A.shape[1]
    x = np.zeros(n)
    x_prev = x.copy()
    t = 1.0
    L = np.linalg.norm(A.T @ A, 2)
    alpha = 1.0 / L
    losses = []

    for _ in range(max_iter):
        residual = A @ x - b
        loss = 0.5 * np.linalg.norm(residual)**2 + lam * np.linalg.norm(x, 1)
        losses.append(loss)

        # Nesterov momentum
        y = x + ((t - 1) / (t + 2)) * (x - x_prev)
        grad = A.T @ (A @ y - b)
        x_new = soft_threshold(y - alpha * grad, alpha * lam)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2

        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < tol:
            break

        x_prev = x.copy()
        x = x_new
        t = t_new

    return x, losses


def exercise_2():
    """ISTA for LASSO — show sparsity."""
    print("\n=== Exercise 2: ISTA for LASSO ===")
    rng = np.random.RandomState(42)
    m, n = 100, 50
    A = rng.randn(m, n)
    # Sparse true solution
    x_true = np.zeros(n)
    x_true[:5] = rng.randn(5) * 3
    b = A @ x_true + 0.3 * rng.randn(m)

    for lam in [0.01, 0.1, 0.5, 1.0]:
        x_ista, losses = ista(A, b, lam)
        nnz = np.sum(np.abs(x_ista) > 1e-6)
        print(f"  λ={lam:>4.2f}  nnz={nnz:>3}  "
              f"loss={losses[-1]:>10.4f}  "
              f"||x-x*||={np.linalg.norm(x_ista - x_true):>7.4f}  "
              f"iters={len(losses)}")

    print(f"  True solution: nnz={np.sum(np.abs(x_true) > 1e-6)}")


# =============================================================================
# Exercise 3: Regularization Path
# =============================================================================

def exercise_3():
    """Plot-ready regularization path data."""
    print("\n=== Exercise 3: Regularization Path ===")
    rng = np.random.RandomState(42)
    m, n = 50, 10
    A = rng.randn(m, n)
    x_true = np.zeros(n)
    x_true[:3] = [2.0, -1.5, 1.0]
    b = A @ x_true + 0.3 * rng.randn(m)

    lambdas = np.logspace(-3, 2, 30)

    print("  L2 path (Ridge):")
    print(f"  {'λ':>10} | {'||x||':>8} | {'nnz(>0.1)':>9}")
    for lam in lambdas[::5]:
        x = ridge_closed_form(A, b, lam)
        nnz = np.sum(np.abs(x) > 0.1)
        print(f"  {lam:>10.4f} | {np.linalg.norm(x):>8.4f} | {nnz:>9}")

    print("\n  L1 path (LASSO):")
    print(f"  {'λ':>10} | {'||x||_1':>8} | {'nnz(>1e-4)':>9}")
    for lam in lambdas[::5]:
        x, _ = ista(A, b, lam, max_iter=2000)
        nnz = np.sum(np.abs(x) > 1e-4)
        print(f"  {lam:>10.4f} | {np.linalg.norm(x, 1):>8.4f} | {nnz:>9}")


# =============================================================================
# Exercise 4: Cross-Validation
# =============================================================================

def k_fold_cv(A, b, lam, k=5, method='ridge'):
    """K-fold cross-validation for regularization parameter selection."""
    n = A.shape[0]
    indices = np.arange(n)
    fold_size = n // k
    errors = []

    for fold in range(k):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size],
                                    indices[(fold + 1) * fold_size:]])
        A_train, b_train = A[train_idx], b[train_idx]
        A_val, b_val = A[val_idx], b[val_idx]

        if method == 'ridge':
            x = ridge_closed_form(A_train, b_train, lam)
        else:
            x, _ = ista(A_train, b_train, lam, max_iter=1000)

        errors.append(np.mean((A_val @ x - b_val)**2))

    return np.mean(errors)


def exercise_4():
    """Cross-validation for λ selection."""
    print("\n=== Exercise 4: Cross-Validation ===")
    rng = np.random.RandomState(42)
    m, n = 200, 20
    A = rng.randn(m, n)
    x_true = rng.randn(n)
    b = A @ x_true + 0.5 * rng.randn(m)

    lambdas = np.logspace(-4, 2, 20)
    print("  Ridge CV:")
    best_lam, best_err = None, np.inf
    for lam in lambdas:
        cv_err = k_fold_cv(A, b, lam, k=5, method='ridge')
        if cv_err < best_err:
            best_err = cv_err
            best_lam = lam

    print(f"  Best λ = {best_lam:.4f}, CV error = {best_err:.6f}")
    x_best = ridge_closed_form(A, b, best_lam)
    print(f"  ||x_best - x_true|| = {np.linalg.norm(x_best - x_true):.6f}")
    x_unreg = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"  ||x_unreg - x_true|| = {np.linalg.norm(x_unreg - x_true):.6f}")


# =============================================================================
# Exercise 5: Regularized Sensor Calibration
# =============================================================================

def exercise_5():
    """Regularized sensor calibration — prevent overfitting to noise."""
    print("\n=== Exercise 5: Regularized Sensor Calibration ===")
    rng = np.random.RandomState(42)

    # True calibration parameters (6 params: 3 scale + 3 bias)
    params_true = np.array([1.02, 0.98, 1.01, 0.05, -0.03, 0.02])

    # Generate calibration data
    n_samples = 30  # Few samples — overfitting risk
    n_params = 6
    A = np.zeros((n_samples, n_params))
    for i in range(n_samples):
        raw = rng.randn(3) * 5
        A[i, :3] = raw
        A[i, 3:] = 1.0
    b = A @ params_true + 0.3 * rng.randn(n_samples)

    # Test data (larger)
    n_test = 200
    A_test = np.zeros((n_test, n_params))
    for i in range(n_test):
        raw = rng.randn(3) * 5
        A_test[i, :3] = raw
        A_test[i, 3:] = 1.0
    b_test = A_test @ params_true + 0.3 * rng.randn(n_test)

    # Compare unregularized vs regularized
    x_unreg = np.linalg.lstsq(A, b, rcond=None)[0]
    train_err_unreg = np.mean((A @ x_unreg - b)**2)
    test_err_unreg = np.mean((A_test @ x_unreg - b_test)**2)

    print(f"  {'Method':>20}  {'Train MSE':>10}  {'Test MSE':>10}  "
          f"{'||x-x*||':>10}")
    print(f"  {'Unregularized':>20}  {train_err_unreg:>10.6f}  "
          f"{test_err_unreg:>10.6f}  "
          f"{np.linalg.norm(x_unreg - params_true):>10.6f}")

    for lam in [0.001, 0.01, 0.1, 1.0]:
        x_reg = ridge_closed_form(A, b, lam)
        train_err = np.mean((A @ x_reg - b)**2)
        test_err = np.mean((A_test @ x_reg - b_test)**2)
        print(f"  {'Ridge λ=' + str(lam):>20}  {train_err:>10.6f}  "
              f"{test_err:>10.6f}  "
              f"{np.linalg.norm(x_reg - params_true):>10.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
