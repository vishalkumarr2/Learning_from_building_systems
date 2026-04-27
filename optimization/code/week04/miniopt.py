"""
Day 28: miniopt — A Minimal Optimization Library
Phase II Capstone, Week 4

Usage:
    from miniopt import minimize, OptimizeResult
    result = minimize(rosenbrock, x0, method='bfgs', jac=rosenbrock_grad)

Run standalone: python miniopt.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple


# =============================================================================
# Result container
# =============================================================================

@dataclass
class OptimizeResult:
    """Result of an optimization run (mirrors scipy.optimize.OptimizeResult)."""
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    ngrad: int
    success: bool
    message: str
    history: List[Tuple[float, np.ndarray]] = field(default_factory=list)


# =============================================================================
# Standard test functions
# =============================================================================

def rosenbrock(x):
    """Rosenbrock: min=0 at (1,...,1)."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))

def rosenbrock_grad(x):
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        g[i+1] += 200 * (x[i+1] - x[i]**2)
    return g

def sphere(x):
    """Sphere: min=0 at origin."""
    return np.sum(x**2)

def sphere_grad(x):
    return 2 * x

def beale(x):
    """Beale (2D only): min=0 at (3, 0.5)."""
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def booth(x):
    """Booth (2D only): min=0 at (1, 3)."""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def styblinski_tang(x):
    """Styblinski-Tang: min ≈ -39.166*n at x ≈ -2.903."""
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)


# =============================================================================
# Utilities
# =============================================================================

def _numerical_gradient(f, x, h=1e-7):
    """Central difference gradient."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def _backtracking_line_search(f, x, d, g, alpha_init=1.0,
                               c1=1e-4, rho=0.5, max_ls=40):
    """Backtracking line search satisfying Armijo condition."""
    alpha = alpha_init
    fx = f(x)
    slope = g.dot(d)
    for _ in range(max_ls):
        if f(x + alpha * d) <= fx + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


# =============================================================================
# Optimization methods
# =============================================================================

def _gradient_descent(f, x0, jac, max_iter, tol, lr):
    """Gradient descent with backtracking line search."""
    x = x0.copy()
    nfev, ngrad = 0, 0
    history = []

    for k in range(max_iter):
        fx = f(x); nfev += 1
        g = jac(x); ngrad += 1
        history.append((fx, x.copy()))

        if np.linalg.norm(g) < tol:
            return OptimizeResult(x=x, fun=fx, nit=k, nfev=nfev,
                                ngrad=ngrad, success=True,
                                message="Converged", history=history)

        alpha = _backtracking_line_search(f, x, -g, g)
        nfev += 20  # approximate line search evals
        x = x - alpha * g

    fx = f(x); nfev += 1
    return OptimizeResult(x=x, fun=fx, nit=max_iter, nfev=nfev,
                         ngrad=ngrad, success=False,
                         message="Max iterations reached", history=history)


def _momentum(f, x0, jac, max_iter, tol, lr, beta=0.9):
    """Heavy ball momentum method."""
    x = x0.copy()
    v = np.zeros_like(x)
    nfev, ngrad = 0, 0
    history = []

    for k in range(max_iter):
        fx = f(x); nfev += 1
        g = jac(x); ngrad += 1
        history.append((fx, x.copy()))

        if np.linalg.norm(g) < tol:
            return OptimizeResult(x=x, fun=fx, nit=k, nfev=nfev,
                                ngrad=ngrad, success=True,
                                message="Converged", history=history)

        v = beta * v - lr * g
        x = x + v

    fx = f(x); nfev += 1
    return OptimizeResult(x=x, fun=fx, nit=max_iter, nfev=nfev,
                         ngrad=ngrad, success=False,
                         message="Max iterations reached", history=history)


def _nesterov(f, x0, jac, max_iter, tol, lr, beta=0.9):
    """Nesterov accelerated gradient."""
    x = x0.copy()
    v = np.zeros_like(x)
    nfev, ngrad = 0, 0
    history = []

    for k in range(max_iter):
        fx = f(x); nfev += 1
        # Look-ahead gradient
        g = jac(x + beta * v); ngrad += 1
        nfev += len(x) * 2  # numerical grad cost if applicable
        history.append((fx, x.copy()))

        if np.linalg.norm(g) < tol:
            return OptimizeResult(x=x, fun=fx, nit=k, nfev=nfev,
                                ngrad=ngrad, success=True,
                                message="Converged", history=history)

        v = beta * v - lr * g
        x = x + v

    fx = f(x); nfev += 1
    return OptimizeResult(x=x, fun=fx, nit=max_iter, nfev=nfev,
                         ngrad=ngrad, success=False,
                         message="Max iterations reached", history=history)


def _bfgs(f, x0, jac, max_iter, tol, **kwargs):
    """BFGS quasi-Newton method."""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)
    nfev, ngrad = 0, 0
    history = []

    g = jac(x); ngrad += 1

    for k in range(max_iter):
        fx = f(x); nfev += 1
        history.append((fx, x.copy()))

        if np.linalg.norm(g) < tol:
            return OptimizeResult(x=x, fun=fx, nit=k, nfev=nfev,
                                ngrad=ngrad, success=True,
                                message="Converged", history=history)

        d = -H @ g
        alpha = _backtracking_line_search(f, x, d, g)
        nfev += 20

        s = alpha * d
        x_new = x + s
        g_new = jac(x_new); ngrad += 1
        y = g_new - g

        sy = s.dot(y)
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x = x_new
        g = g_new

    fx = f(x); nfev += 1
    return OptimizeResult(x=x, fun=fx, nit=max_iter, nfev=nfev,
                         ngrad=ngrad, success=False,
                         message="Max iterations reached", history=history)


def _lbfgs(f, x0, jac, max_iter, tol, m=10, **kwargs):
    """L-BFGS with two-loop recursion."""
    n = len(x0)
    x = x0.copy()
    nfev, ngrad = 0, 0
    history = []
    s_list, y_list, rho_list = [], [], []

    g = jac(x); ngrad += 1

    for k in range(max_iter):
        fx = f(x); nfev += 1
        history.append((fx, x.copy()))

        if np.linalg.norm(g) < tol:
            return OptimizeResult(x=x, fun=fx, nit=k, nfev=nfev,
                                ngrad=ngrad, success=True,
                                message="Converged", history=history)

        # Two-loop recursion
        q = g.copy()
        alphas = []
        for i in range(len(s_list) - 1, -1, -1):
            a = rho_list[i] * s_list[i].dot(q)
            alphas.append(a)
            q = q - a * y_list[i]
        alphas.reverse()

        # Initial Hessian approximation
        if len(s_list) > 0:
            gamma = s_list[-1].dot(y_list[-1]) / y_list[-1].dot(y_list[-1])
            r = gamma * q
        else:
            r = q.copy()

        for i in range(len(s_list)):
            beta = rho_list[i] * y_list[i].dot(r)
            r = r + (alphas[i] - beta) * s_list[i]

        d = -r
        alpha = _backtracking_line_search(f, x, d, g)
        nfev += 20

        s = alpha * d
        x_new = x + s
        g_new = jac(x_new); ngrad += 1
        y = g_new - g

        sy = s.dot(y)
        if sy > 1e-10:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        x = x_new
        g = g_new

    fx = f(x); nfev += 1
    return OptimizeResult(x=x, fun=fx, nit=max_iter, nfev=nfev,
                         ngrad=ngrad, success=False,
                         message="Max iterations reached", history=history)


def _adam(f, x0, jac, max_iter, tol, lr=0.001,
          beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
    """Adam optimizer."""
    x = x0.copy()
    m_vec = np.zeros_like(x)
    v_vec = np.zeros_like(x)
    nfev, ngrad = 0, 0
    history = []

    for k in range(1, max_iter + 1):
        fx = f(x); nfev += 1
        g = jac(x); ngrad += 1
        history.append((fx, x.copy()))

        if np.linalg.norm(g) < tol:
            return OptimizeResult(x=x, fun=fx, nit=k, nfev=nfev,
                                ngrad=ngrad, success=True,
                                message="Converged", history=history)

        m_vec = beta1 * m_vec + (1 - beta1) * g
        v_vec = beta2 * v_vec + (1 - beta2) * g**2

        m_hat = m_vec / (1 - beta1**k)
        v_hat = v_vec / (1 - beta2**k)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

    fx = f(x); nfev += 1
    return OptimizeResult(x=x, fun=fx, nit=max_iter, nfev=nfev,
                         ngrad=ngrad, success=False,
                         message="Max iterations reached", history=history)


# =============================================================================
# Main entry point
# =============================================================================

METHODS = {
    'gd': _gradient_descent,
    'momentum': _momentum,
    'nesterov': _nesterov,
    'bfgs': _bfgs,
    'lbfgs': _lbfgs,
    'adam': _adam,
}


def minimize(fun: Callable, x0: np.ndarray, method: str = 'bfgs',
             jac: Optional[Callable] = None,
             max_iter: int = 5000, tol: float = 1e-8,
             lr: float = 0.01, **kwargs) -> OptimizeResult:
    """
    Minimize a scalar function of one or more variables.

    Parameters:
        fun: Objective function f(x) → float
        x0: Initial guess (1D array)
        method: One of 'gd', 'momentum', 'nesterov', 'bfgs', 'lbfgs', 'adam'
        jac: Gradient function. If None, uses numerical differentiation.
        max_iter: Maximum iterations
        tol: Gradient norm tolerance for convergence
        lr: Learning rate (for gd, momentum, nesterov, adam)
        **kwargs: Additional method-specific parameters

    Returns:
        OptimizeResult with x, fun, nit, nfev, ngrad, success, message, history
    """
    x0 = np.asarray(x0, dtype=float).copy()

    method_lower = method.lower()
    if method_lower not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}")

    # Default to numerical gradient if not provided
    if jac is None:
        jac = lambda x: _numerical_gradient(fun, x)

    solver = METHODS[method_lower]
    return solver(fun, x0, jac, max_iter, tol, lr=lr, **kwargs)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("miniopt — Phase II Capstone Optimization Library")
    print("=" * 60)

    # Test on Rosenbrock 5D
    n = 5
    x0 = np.zeros(n)
    x_star = np.ones(n)

    print(f"\nRosenbrock {n}D — all methods:")
    print(f"  {'Method':>10}  {'f(x*)':>12}  {'||x-x*||':>10}  "
          f"{'nit':>6}  {'ok?':>4}")
    print("  " + "-" * 55)

    configs = [
        ('gd', {'lr': 0.001, 'max_iter': 20000}),
        ('momentum', {'lr': 0.0005, 'max_iter': 20000}),
        ('nesterov', {'lr': 0.0005, 'max_iter': 20000}),
        ('bfgs', {}),
        ('lbfgs', {}),
        ('adam', {'lr': 0.005, 'max_iter': 20000}),
    ]

    for method, opts in configs:
        res = minimize(rosenbrock, x0, method=method,
                      jac=rosenbrock_grad, **opts)
        dist = np.linalg.norm(res.x - x_star)
        print(f"  {method:>10}  {res.fun:>12.6e}  {dist:>10.6f}  "
              f"{res.nit:>6}  {'✓' if res.success else '✗':>4}")

    # Quick tests on other functions
    print("\nSphere 10D (BFGS):")
    res = minimize(sphere, np.ones(10) * 5, method='bfgs', jac=sphere_grad)
    print(f"  f={res.fun:.2e}, ||x||={np.linalg.norm(res.x):.2e}, "
          f"nit={res.nit}, success={res.success}")

    print("\nBooth 2D (L-BFGS):")
    res = minimize(booth, np.array([0.0, 0.0]), method='lbfgs')
    print(f"  f={res.fun:.2e}, x={res.x}, nit={res.nit}")

    print("\nAvailable methods:", list(METHODS.keys()))
