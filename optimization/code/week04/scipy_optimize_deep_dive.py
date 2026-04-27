"""
Day 27: scipy.optimize Deep Dive — Companion Code
Phase II, Week 4

Run: python scipy_optimize_deep_dive.py
"""

import numpy as np
import time

try:
    from scipy.optimize import minimize, least_squares, rosen, rosen_der
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Test Functions
# =============================================================================

def rosenbrock(x):
    """Rosenbrock function. Global min = 0 at (1, 1, ..., 1)."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))

def rosenbrock_grad(x):
    """Analytic gradient of Rosenbrock."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        g[i+1] += 200 * (x[i+1] - x[i]**2)
    return g

def rosenbrock_hess(x):
    """Analytic Hessian of Rosenbrock."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += -400 * (x[i+1] - 3 * x[i]**2) + 2 + 800 * x[i]**2
        H[i, i+1] += -400 * x[i]
        H[i+1, i] += -400 * x[i]
        H[i+1, i+1] += 200
    # Fix: first element
    H[0, 0] = -400 * (x[1] - 3 * x[0]**2) + 2 + 800 * x[0]**2
    return H

def ill_conditioned_quad(x, kappa=1e4):
    """Quadratic with condition number kappa."""
    n = len(x)
    diag = np.logspace(0, np.log10(kappa), n)
    return 0.5 * np.sum(diag * x**2)

def ill_conditioned_quad_grad(x, kappa=1e4):
    n = len(x)
    diag = np.logspace(0, np.log10(kappa), n)
    return diag * x


# =============================================================================
# Exercise 1: Every Method Comparison
# =============================================================================

def exercise_1():
    """Compare all scipy methods on Rosenbrock."""
    if not HAS_SCIPY:
        print("=== Exercise 1: Skipped (scipy not available) ===")
        return

    print("=== Exercise 1: Method Comparison on Rosenbrock 10D ===")
    n = 10
    x0 = np.zeros(n)

    methods_no_hess = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B']
    methods_hess = ['Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact',
                    'trust-krylov']

    print(f"  {'Method':>15}  {'f(x*)':>12}  {'nit':>6}  "
          f"{'nfev':>6}  {'time(s)':>8}  {'ok?':>4}")
    print("  " + "-" * 65)

    for method in methods_no_hess:
        try:
            t0 = time.time()
            if method in ['CG', 'BFGS', 'L-BFGS-B']:
                res = minimize(rosenbrock, x0, method=method,
                              jac=rosenbrock_grad,
                              options={'maxiter': 10000})
            else:
                res = minimize(rosenbrock, x0, method=method,
                              options={'maxiter': 10000})
            elapsed = time.time() - t0
            print(f"  {method:>15}  {res.fun:>12.6e}  {res.nit:>6}  "
                  f"{res.nfev:>6}  {elapsed:>8.3f}  "
                  f"{'✓' if res.success else '✗':>4}")
        except Exception as e:
            print(f"  {method:>15}  ERROR: {e}")

    for method in methods_hess:
        try:
            t0 = time.time()
            res = minimize(rosenbrock, x0, method=method,
                          jac=rosenbrock_grad, hess=rosenbrock_hess,
                          options={'maxiter': 10000})
            elapsed = time.time() - t0
            print(f"  {method:>15}  {res.fun:>12.6e}  {res.nit:>6}  "
                  f"{res.nfev:>6}  {elapsed:>8.3f}  "
                  f"{'✓' if res.success else '✗':>4}")
        except Exception as e:
            print(f"  {method:>15}  ERROR: {str(e)[:40]}")


# =============================================================================
# Exercise 2: Callback Logging
# =============================================================================

def exercise_2():
    """Callback-based convergence logging."""
    if not HAS_SCIPY:
        print("\n=== Exercise 2: Skipped (scipy not available) ===")
        return

    print("\n=== Exercise 2: Callback Convergence Logging ===")
    n = 5
    x0 = np.zeros(n)
    x_star = np.ones(n)

    for method in ['CG', 'BFGS', 'Newton-CG']:
        history = {'f': [], 'dist': [], 'grad_norm': []}

        def callback(xk):
            f_val = rosenbrock(xk)
            dist = np.linalg.norm(xk - x_star)
            grad = rosenbrock_grad(xk)
            history['f'].append(f_val)
            history['dist'].append(dist)
            history['grad_norm'].append(np.linalg.norm(grad))

        kwargs = {'jac': rosenbrock_grad, 'callback': callback,
                  'options': {'maxiter': 5000}}
        if method == 'Newton-CG':
            kwargs['hess'] = rosenbrock_hess

        res = minimize(rosenbrock, x0, method=method, **kwargs)
        print(f"  {method:>12}: iters={len(history['f']):>5}  "
              f"f_final={history['f'][-1] if history['f'] else 'N/A':>10.2e}  "
              f"dist_final={history['dist'][-1] if history['dist'] else 'N/A':>10.2e}")


# =============================================================================
# Exercise 3: Bounds and Scaling
# =============================================================================

def exercise_3():
    """Variable scaling for BFGS."""
    if not HAS_SCIPY:
        print("\n=== Exercise 3: Skipped (scipy not available) ===")
        return

    print("\n=== Exercise 3: Bounds and Variable Scaling ===")

    # Badly scaled: x1 ∈ [0, 1], x2 ∈ [0, 10000]
    # f(x) = (x1 - 0.5)^2 + (x2 - 5000)^2
    def f_unscaled(x):
        return (x[0] - 0.5)**2 + (x[1] - 5000)**2

    def f_unscaled_grad(x):
        return np.array([2*(x[0] - 0.5), 2*(x[1] - 5000)])

    x0 = np.array([0.0, 0.0])
    bounds = [(0, 1), (0, 10000)]

    # Unscaled
    res_unscaled = minimize(f_unscaled, x0, method='L-BFGS-B',
                           jac=f_unscaled_grad, bounds=bounds)

    # Scaled: normalize to [0, 1] each
    def f_scaled(z):
        x1 = z[0]  # already [0, 1]
        x2 = z[1] * 10000  # [0, 1] → [0, 10000]
        return (x1 - 0.5)**2 + (x2 - 5000)**2

    def f_scaled_grad(z):
        x1 = z[0]
        x2 = z[1] * 10000
        return np.array([2*(x1 - 0.5), 2*(x2 - 5000) * 10000])

    z0 = np.array([0.0, 0.0])
    bounds_scaled = [(0, 1), (0, 1)]
    res_scaled = minimize(f_scaled, z0, method='L-BFGS-B',
                         jac=f_scaled_grad, bounds=bounds_scaled)

    print(f"  Unscaled: nit={res_unscaled.nit}, nfev={res_unscaled.nfev}, "
          f"x={res_unscaled.x}")
    x_from_scaled = np.array([res_scaled.x[0], res_scaled.x[1] * 10000])
    print(f"  Scaled:   nit={res_scaled.nit}, nfev={res_scaled.nfev}, "
          f"x={x_from_scaled}")


# =============================================================================
# Exercise 4: Analytic Derivatives Speedup
# =============================================================================

def exercise_4():
    """Compare finite-difference vs analytic gradient."""
    if not HAS_SCIPY:
        print("\n=== Exercise 4: Skipped (scipy not available) ===")
        return

    print("\n=== Exercise 4: Derivative Supply Comparison (Rosenbrock 20D) ===")
    n = 20
    x0 = np.zeros(n)

    configs = [
        ("No jac (2-point FD)", {'method': 'BFGS'}),
        ("jac='3-point'", {'method': 'BFGS', 'jac': '3-point'}),
        ("Analytic jac", {'method': 'BFGS', 'jac': rosenbrock_grad}),
        ("Analytic jac+hess", {'method': 'Newton-CG',
                               'jac': rosenbrock_grad,
                               'hess': rosenbrock_hess}),
    ]

    print(f"  {'Config':>25}  {'f(x*)':>10}  {'nfev':>6}  "
          f"{'njev':>6}  {'time(s)':>8}")
    for name, kwargs in configs:
        t0 = time.time()
        res = minimize(rosenbrock, x0, options={'maxiter': 10000}, **kwargs)
        elapsed = time.time() - t0
        njev = getattr(res, 'njev', 'N/A')
        print(f"  {name:>25}  {res.fun:>10.2e}  {res.nfev:>6}  "
              f"{str(njev):>6}  {elapsed:>8.3f}")


# =============================================================================
# Exercise 5: Custom Optimizer Wrapping
# =============================================================================

def bfgs_custom(f, x0, grad_f, max_iter=1000, tol=1e-8):
    """Our BFGS from Week 3, simplified."""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)

    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        d = -H @ g

        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4
        fx = f(x)
        while f(x + alpha * d) > fx + c1 * alpha * g.dot(d):
            alpha *= 0.5
            if alpha < 1e-16:
                break

        s = alpha * d
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g

        sy = s.dot(y)
        if sy > 1e-10:
            rho = 1.0 / sy
            V = np.eye(n) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x = x_new

    return x, f(x), k


def exercise_5():
    """Compare custom BFGS with scipy BFGS."""
    print("\n=== Exercise 5: Custom BFGS vs scipy BFGS ===")
    n = 10
    x0 = np.zeros(n)

    # Custom
    t0 = time.time()
    x_cust, f_cust, iters_cust = bfgs_custom(
        rosenbrock, x0, rosenbrock_grad)
    t_cust = time.time() - t0

    print(f"  Custom BFGS: f={f_cust:.6e}, iters={iters_cust}, "
          f"time={t_cust:.3f}s")

    # scipy
    if HAS_SCIPY:
        t0 = time.time()
        res = minimize(rosenbrock, x0, method='BFGS', jac=rosenbrock_grad)
        t_scipy = time.time() - t0
        print(f"  scipy BFGS:  f={res.fun:.6e}, iters={res.nit}, "
              f"time={t_scipy:.3f}s")


# =============================================================================
# Exercise 6: Wheel Radius Calibration
# =============================================================================

def exercise_6():
    """Wheel radius calibration with least_squares."""
    print("\n=== Exercise 6: Wheel Radius Calibration ===")
    rng = np.random.RandomState(42)

    # True params: [wheel_radius, wheelbase]
    params_true = np.array([0.075, 0.42])  # 7.5cm radius, 42cm base

    # Generate data: encoder ticks → measured displacement
    n_data = 50
    ticks_left = rng.randint(100, 1000, n_data).astype(float)
    ticks_right = rng.randint(100, 1000, n_data).astype(float)
    ticks_per_rev = 360.0

    # Forward model: d = r * ticks * 2π / ticks_per_rev
    def forward_model(params, ticks_l, ticks_r):
        r, b = params
        dl = r * ticks_l * 2 * np.pi / ticks_per_rev
        dr = r * ticks_r * 2 * np.pi / ticks_per_rev
        linear = (dl + dr) / 2
        angular = (dr - dl) / b
        return np.column_stack([linear, angular])

    # "True" measurements with noise
    true_disp = forward_model(params_true, ticks_left, ticks_right)
    noise = 0.002 * rng.randn(*true_disp.shape)
    measured_disp = true_disp + noise

    # Residual function for least_squares
    def residuals(params):
        pred = forward_model(params, ticks_left, ticks_right)
        return (pred - measured_disp).ravel()

    # Initial guess (deliberately off)
    x0 = np.array([0.08, 0.45])

    if HAS_SCIPY:
        # least_squares
        t0 = time.time()
        res_ls = least_squares(residuals, x0, method='trf',
                              bounds=([0.05, 0.3], [0.15, 0.6]))
        t_ls = time.time() - t0

        # minimize (for comparison)
        def objective(params):
            r = residuals(params)
            return 0.5 * np.sum(r**2)

        t0 = time.time()
        res_min = minimize(objective, x0, method='L-BFGS-B',
                          bounds=[(0.05, 0.15), (0.3, 0.6)])
        t_min = time.time() - t0

        print(f"  True params:     r={params_true[0]:.4f}, "
              f"b={params_true[1]:.4f}")
        print(f"  least_squares:   r={res_ls.x[0]:.4f}, "
              f"b={res_ls.x[1]:.4f}  "
              f"(nfev={res_ls.nfev}, time={t_ls:.4f}s)")
        print(f"  minimize(L-BFGS): r={res_min.x[0]:.4f}, "
              f"b={res_min.x[1]:.4f}  "
              f"(nfev={res_min.nfev}, time={t_min:.4f}s)")
    else:
        print("  (scipy not available)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()
