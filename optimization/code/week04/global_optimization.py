"""
Day 26: Global Optimization Methods — Companion Code
Phase II, Week 4

Run: python global_optimization.py
"""

import numpy as np
import time

try:
    from scipy.optimize import minimize, differential_evolution, basinhopping
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Test Functions
# =============================================================================

def rastrigin(x):
    """Rastrigin function — many local minima. Global min = 0 at origin."""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    """Gradient of Rastrigin."""
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def ackley(x):
    """Ackley function. Global min = 0 at origin."""
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e

def styblinski_tang(x):
    """Styblinski-Tang function. Global min ≈ -39.166*n at x ≈ -2.903."""
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


# =============================================================================
# Exercise 1: Random Restart + L-BFGS
# =============================================================================

def random_restart(f, bounds, n_restarts=50, seed=42):
    """
    Random restart with L-BFGS-B.
    
    Parameters:
        f: objective function
        bounds: list of (low, high) per dimension
        n_restarts: number of random starting points
    Returns:
        best_x, best_f, all_results
    """
    rng = np.random.RandomState(seed)
    n = len(bounds)
    best_x = None
    best_f = np.inf
    all_results = []

    for i in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        if HAS_SCIPY:
            res = minimize(f, x0, method='L-BFGS-B', bounds=bounds)
            x_opt, f_opt = res.x, res.fun
        else:
            # Simple GD fallback
            x = x0.copy()
            for _ in range(500):
                g = _numerical_grad(f, x)
                x = x - 0.01 * g
                x = np.clip(x, [lo for lo, hi in bounds],
                           [hi for lo, hi in bounds])
            x_opt, f_opt = x, f(x)

        all_results.append((x_opt, f_opt))
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()

    return best_x, best_f, all_results


def _numerical_grad(f, x, h=1e-7):
    """Central difference gradient."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        g[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return g


def exercise_1():
    """Random restart on Rastrigin."""
    print("=== Exercise 1: Random Restart + L-BFGS ===")
    n_dim = 5
    bounds = [(-5.12, 5.12)] * n_dim

    t0 = time.time()
    best_x, best_f, results = random_restart(
        rastrigin, bounds, n_restarts=100)
    elapsed = time.time() - t0

    all_f = [r[1] for r in results]
    print(f"  Dim={n_dim}, Restarts=100")
    print(f"  Best f = {best_f:.6f} (global min = 0)")
    print(f"  Best x ≈ {np.round(best_x, 3)}")
    print(f"  Mean f across restarts = {np.mean(all_f):.4f}")
    print(f"  Found global (f < 1e-4): "
          f"{sum(1 for fv in all_f if fv < 1e-4)}/100")
    print(f"  Time: {elapsed:.2f}s")


# =============================================================================
# Exercise 2: Simulated Annealing
# =============================================================================

def simulated_annealing(f, bounds, T_init=10.0, T_min=1e-6,
                         cooling_rate=0.995, max_iter=50000,
                         perturbation=0.5, seed=42):
    """
    Simulated annealing.
    
    Parameters:
        T_init: initial temperature
        T_min: stop when T < T_min
        cooling_rate: T *= cooling_rate each step
        perturbation: std of Gaussian perturbation
    """
    rng = np.random.RandomState(seed)
    n = len(bounds)
    x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
    f_x = f(x)
    best_x, best_f = x.copy(), f_x
    T = T_init
    accepted = 0

    for i in range(max_iter):
        if T < T_min:
            break

        # Propose
        x_new = x + perturbation * rng.randn(n)
        # Clip to bounds
        x_new = np.clip(x_new,
                       [lo for lo, hi in bounds],
                       [hi for lo, hi in bounds])
        f_new = f(x_new)
        delta = f_new - f_x

        # Accept/reject
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            x = x_new
            f_x = f_new
            accepted += 1
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

        T *= cooling_rate

    return best_x, best_f, accepted


def exercise_2():
    """Simulated annealing on Rastrigin."""
    print("\n=== Exercise 2: Simulated Annealing ===")
    n_dim = 5
    bounds = [(-5.12, 5.12)] * n_dim

    t0 = time.time()
    best_x, best_f, accepted = simulated_annealing(
        rastrigin, bounds, T_init=50, cooling_rate=0.9995,
        max_iter=100000, perturbation=0.5)
    elapsed = time.time() - t0

    print(f"  Dim={n_dim}")
    print(f"  Best f = {best_f:.6f} (global min = 0)")
    print(f"  Best x ≈ {np.round(best_x, 3)}")
    print(f"  Accepted: {accepted}/100000")
    print(f"  Time: {elapsed:.2f}s")


# =============================================================================
# Exercise 3: Rastrigin — local vs global
# =============================================================================

def exercise_3():
    """Compare local vs global on Rastrigin."""
    print("\n=== Exercise 3: Local vs Global on Rastrigin ===")
    n_dim = 5
    bounds = [(-5.12, 5.12)] * n_dim

    # Single local start (likely finds local minimum)
    rng = np.random.RandomState(0)
    x0 = rng.uniform(-5.12, 5.12, n_dim)

    if HAS_SCIPY:
        res_local = minimize(rastrigin, x0, method='L-BFGS-B',
                            jac=rastrigin_grad, bounds=bounds)
        print(f"  Local L-BFGS-B from random x0:")
        print(f"    f = {res_local.fun:.6f}, "
              f"x ≈ {np.round(res_local.x, 3)}")
    else:
        print("  (scipy not available, skipping local comparison)")

    # Random restart
    best_x, best_f, _ = random_restart(rastrigin, bounds, n_restarts=50)
    print(f"  Random restart (50 trials):")
    print(f"    f = {best_f:.6f}, x ≈ {np.round(best_x, 3)}")

    # Simulated annealing
    best_x, best_f, _ = simulated_annealing(
        rastrigin, bounds, T_init=50, max_iter=50000)
    print(f"  Simulated annealing:")
    print(f"    f = {best_f:.6f}, x ≈ {np.round(best_x, 3)}")


# =============================================================================
# Exercise 4: scipy comparison
# =============================================================================

def exercise_4():
    """Compare scipy global optimizers."""
    if not HAS_SCIPY:
        print("\n=== Exercise 4: Skipped (scipy not available) ===")
        return

    print("\n=== Exercise 4: scipy Global Optimizers ===")
    n_dim = 5
    bounds = [(-5.12, 5.12)] * n_dim

    # differential_evolution
    t0 = time.time()
    res_de = differential_evolution(rastrigin, bounds, seed=42,
                                   maxiter=500, tol=1e-10)
    t_de = time.time() - t0

    # basinhopping
    x0 = np.random.RandomState(42).uniform(-5.12, 5.12, n_dim)
    t0 = time.time()
    res_bh = basinhopping(rastrigin, x0, niter=100, seed=42,
                          minimizer_kwargs={'method': 'L-BFGS-B',
                                           'bounds': bounds})
    t_bh = time.time() - t0

    # local minimize
    t0 = time.time()
    res_lm = minimize(rastrigin, x0, method='L-BFGS-B', bounds=bounds)
    t_lm = time.time() - t0

    print(f"  {'Method':>25}  {'f(x*)':>10}  {'Time (s)':>10}")
    print(f"  {'Local (L-BFGS-B)':>25}  {res_lm.fun:>10.6f}  {t_lm:>10.3f}")
    print(f"  {'differential_evolution':>25}  {res_de.fun:>10.6f}  "
          f"{t_de:>10.3f}")
    print(f"  {'basinhopping':>25}  {res_bh.fun:>10.6f}  {t_bh:>10.3f}")


# =============================================================================
# Exercise 5: Scan Matching Initialization
# =============================================================================

def exercise_5():
    """Scan matching: find rotation/translation for point cloud alignment."""
    print("\n=== Exercise 5: Scan Matching Initialization ===")
    rng = np.random.RandomState(42)

    # Generate source point cloud (2D)
    n_points = 50
    source = rng.randn(n_points, 2) * 3

    # True transformation
    theta_true = 1.2  # radians (~69 degrees)
    tx_true, ty_true = 2.0, -1.5
    R_true = np.array([[np.cos(theta_true), -np.sin(theta_true)],
                       [np.sin(theta_true),  np.cos(theta_true)]])
    target = (R_true @ source.T).T + np.array([tx_true, ty_true])
    target += 0.1 * rng.randn(n_points, 2)  # noise

    def alignment_error(params):
        """Sum of squared distances after applying transform."""
        theta, tx, ty = params
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        transformed = (R @ source.T).T + np.array([tx, ty])
        return np.sum((transformed - target)**2)

    # Single local start (likely fails with bad initial rotation)
    x0_bad = np.array([0.0, 0.0, 0.0])
    if HAS_SCIPY:
        res_local = minimize(alignment_error, x0_bad, method='L-BFGS-B')
        print(f"  Single start (θ=0): f={res_local.fun:.4f}, "
              f"θ={res_local.x[0]:.3f} "
              f"(true={theta_true:.3f})")

    # Random restart over rotations
    best_f = np.inf
    best_params = None
    for i in range(36):  # 10-degree increments
        theta0 = i * np.pi / 18
        x0 = np.array([theta0, 0.0, 0.0])
        if HAS_SCIPY:
            res = minimize(alignment_error, x0, method='L-BFGS-B')
            if res.fun < best_f:
                best_f = res.fun
                best_params = res.x

    if best_params is not None:
        print(f"  Random restart (36 rotations): f={best_f:.4f}, "
              f"θ={best_params[0]:.3f}, "
              f"t=({best_params[1]:.3f}, {best_params[2]:.3f})")
        print(f"  True: θ={theta_true:.3f}, "
              f"t=({tx_true:.3f}, {ty_true:.3f})")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
