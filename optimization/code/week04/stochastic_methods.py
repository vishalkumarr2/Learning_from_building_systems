"""
Day 24: Stochastic and Mini-Batch Methods — Companion Code
Phase II, Week 4 (ENRICHMENT)

Run: python stochastic_methods.py
"""

import numpy as np
import time


# =============================================================================
# Data Generation
# =============================================================================

def generate_linear_data(n_samples=10000, n_features=10, noise_std=0.5, seed=42):
    """Generate linear regression data: y = X @ w_true + noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    w_true = rng.randn(n_features)
    y = X @ w_true + noise_std * rng.randn(n_samples)
    return X, y, w_true


# =============================================================================
# Exercise 1: SGD for sum-of-functions
# =============================================================================

def sgd(X, y, x0, alpha=0.01, max_epochs=50, seed=0):
    """
    Stochastic Gradient Descent for linear regression.
    f(w) = (1/N) Σ (y_i - x_i^T w)^2 / 2
    ∇f_i(w) = -x_i (y_i - x_i^T w)
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = x0.copy()
    loss_history = []

    for epoch in range(max_epochs):
        # Compute full loss for logging
        residuals = y - X @ w
        loss = 0.5 * np.mean(residuals**2)
        loss_history.append(loss)

        # Shuffle indices
        indices = rng.permutation(n)
        for i in indices:
            xi = X[i]
            yi = y[i]
            grad_i = -xi * (yi - xi @ w)
            w = w - alpha * grad_i

    return w, loss_history


# =============================================================================
# Exercise 2: Mini-Batch GD
# =============================================================================

def mini_batch_gd(X, y, x0, batch_size=32, alpha=0.01,
                  max_epochs=50, seed=0):
    """
    Mini-batch gradient descent.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = x0.copy()
    loss_history = []

    for epoch in range(max_epochs):
        residuals = y - X @ w
        loss = 0.5 * np.mean(residuals**2)
        loss_history.append(loss)

        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = indices[start:start + batch_size]
            X_b = X[batch]
            y_b = y[batch]
            grad_b = -X_b.T @ (y_b - X_b @ w) / len(batch)
            w = w - alpha * grad_b

    return w, loss_history


# =============================================================================
# Exercise 3: Full GD for comparison
# =============================================================================

def full_gd(X, y, x0, alpha=0.01, max_epochs=50):
    """Full-batch gradient descent."""
    n, d = X.shape
    w = x0.copy()
    loss_history = []

    for epoch in range(max_epochs):
        residuals = y - X @ w
        loss = 0.5 * np.mean(residuals**2)
        loss_history.append(loss)
        grad = -X.T @ residuals / n
        w = w - alpha * grad

    return w, loss_history


def exercise_3():
    """Compare GD, SGD, mini-batch on linear regression."""
    print("=== Exercise 3: GD vs SGD vs Mini-Batch ===")
    X, y, w_true = generate_linear_data(n_samples=10000, n_features=10)
    x0 = np.zeros(10)
    epochs = 20
    alpha = 0.001

    # Full GD
    t0 = time.time()
    w_gd, loss_gd = full_gd(X, y, x0, alpha=alpha, max_epochs=epochs)
    t_gd = time.time() - t0

    # SGD
    t0 = time.time()
    w_sgd, loss_sgd = sgd(X, y, x0, alpha=alpha * 0.1,
                          max_epochs=epochs)
    t_sgd = time.time() - t0

    # Mini-batch variants
    results = {}
    for bs in [32, 128, 512]:
        t0 = time.time()
        w_mb, loss_mb = mini_batch_gd(X, y, x0, batch_size=bs,
                                      alpha=alpha, max_epochs=epochs)
        t_mb = time.time() - t0
        results[bs] = (w_mb, loss_mb, t_mb)

    print(f"  {'Method':>15}  {'Final Loss':>12}  {'Time (s)':>10}  "
          f"{'||w - w*||':>12}")
    print(f"  {'Full GD':>15}  {loss_gd[-1]:>12.6f}  {t_gd:>10.3f}  "
          f"{np.linalg.norm(w_gd - w_true):>12.6f}")
    print(f"  {'SGD':>15}  {loss_sgd[-1]:>12.6f}  {t_sgd:>10.3f}  "
          f"{np.linalg.norm(w_sgd - w_true):>12.6f}")
    for bs, (w_mb, loss_mb, t_mb) in results.items():
        print(f"  {'MB-' + str(bs):>15}  {loss_mb[-1]:>12.6f}  "
              f"{t_mb:>10.3f}  {np.linalg.norm(w_mb - w_true):>12.6f}")


# =============================================================================
# Exercise 4: Learning Rate Schedules
# =============================================================================

def sgd_with_schedule(X, y, x0, schedule_fn, max_epochs=50, seed=0):
    """SGD with custom learning rate schedule."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = x0.copy()
    loss_history = []
    step = 0

    for epoch in range(max_epochs):
        residuals = y - X @ w
        loss = 0.5 * np.mean(residuals**2)
        loss_history.append(loss)

        indices = rng.permutation(n)
        for start in range(0, n, 128):
            batch = indices[start:start + 128]
            X_b = X[batch]
            y_b = y[batch]
            grad_b = -X_b.T @ (y_b - X_b @ w) / len(batch)
            alpha = schedule_fn(step)
            w = w - alpha * grad_b
            step += 1

    return w, loss_history


def exercise_4():
    """Compare learning rate schedules."""
    print("\n=== Exercise 4: Learning Rate Schedules ===")
    X, y, w_true = generate_linear_data()
    x0 = np.zeros(10)
    epochs = 30
    total_steps = epochs * (len(y) // 128)

    schedules = {
        "constant": lambda k: 0.001,
        "1/k": lambda k: 0.01 / (1 + 0.001 * k),
        "cosine": lambda k: 0.001 * 0.5 * (1 + np.cos(np.pi * k / total_steps)),
    }

    print(f"  {'Schedule':>12}  {'Final Loss':>12}  {'||w - w*||':>12}")
    for name, sched in schedules.items():
        w, losses = sgd_with_schedule(X, y, x0, sched, max_epochs=epochs)
        print(f"  {name:>12}  {losses[-1]:>12.6f}  "
              f"{np.linalg.norm(w - w_true):>12.6f}")


# =============================================================================
# Exercise 5: Adam from Scratch
# =============================================================================

def adam(X, y, x0, alpha=0.001, beta1=0.9, beta2=0.999,
        eps=1e-8, batch_size=128, max_epochs=50, seed=0):
    """
    Adam optimizer from scratch.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = x0.copy()
    m = np.zeros(d)  # First moment
    v = np.zeros(d)  # Second moment
    loss_history = []
    t = 0

    for epoch in range(max_epochs):
        residuals = y - X @ w
        loss = 0.5 * np.mean(residuals**2)
        loss_history.append(loss)

        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            t += 1
            batch = indices[start:start + batch_size]
            X_b = X[batch]
            y_b = y[batch]
            grad_b = -X_b.T @ (y_b - X_b @ w) / len(batch)

            m = beta1 * m + (1 - beta1) * grad_b
            v = beta2 * v + (1 - beta2) * grad_b**2

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)

    return w, loss_history


def exercise_5():
    """Compare Adam to SGD."""
    print("\n=== Exercise 5: Adam vs SGD ===")
    X, y, w_true = generate_linear_data()
    x0 = np.zeros(10)
    epochs = 20

    w_adam, loss_adam = adam(X, y, x0, alpha=0.001, max_epochs=epochs)
    w_sgd_mb, loss_sgd = mini_batch_gd(X, y, x0, batch_size=128,
                                       alpha=0.001, max_epochs=epochs)

    print(f"  Adam:    loss={loss_adam[-1]:.6f}, "
          f"||w-w*||={np.linalg.norm(w_adam - w_true):.6f}")
    print(f"  SGD-MB:  loss={loss_sgd[-1]:.6f}, "
          f"||w-w*||={np.linalg.norm(w_sgd_mb - w_true):.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    exercise_3()
    exercise_4()
    exercise_5()
