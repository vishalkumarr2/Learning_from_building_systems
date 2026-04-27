"""
Day 5: Convexity — Companion Code
Phase I, Week 1

Run: python convexity.py
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Exercise 1: Convexity Classification via Hessian
# =============================================================================

def classify_convexity_1d(f, f_hessian, domain=(-5, 5), n_samples=1000):
    """Classify 1D function convexity by checking Hessian sign."""
    x_vals = np.linspace(domain[0], domain[1], n_samples)
    h_vals = np.array([f_hessian(x) for x in x_vals])

    if np.all(h_vals > -1e-10):
        if np.all(h_vals > 1e-10):
            return "Strictly Convex"
        return "Convex"
    elif np.all(h_vals < 1e-10):
        return "Concave"
    else:
        return "Neither"


def classify_convexity_nd(hessian_func, points: list[np.ndarray]):
    """Classify nD function by checking Hessian eigenvalues at multiple points."""
    all_psd = True
    all_nsd = True

    for x in points:
        H = hessian_func(x)
        eigvals = np.linalg.eigvalsh(H)
        if np.any(eigvals < -1e-10):
            all_psd = False
        if np.any(eigvals > 1e-10):
            all_nsd = False

    if all_psd:
        return "Convex (PSD Hessian everywhere sampled)"
    elif all_nsd:
        return "Concave"
    else:
        return "Neither (or non-convex in sampled region)"


# =============================================================================
# Exercise 2: First-Order Condition Visualization
# =============================================================================

def plot_first_order_condition():
    """Show tangent planes lie below convex function."""
    def f(x):
        return x**2

    def grad_f(x):
        return 2 * x

    x_range = np.linspace(-3, 3, 200)
    f_vals = f(x_range)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(x_range, f_vals, "b-", linewidth=2, label="f(x) = x²")

    # Tangent lines at several points
    for x0 in [-2, -0.5, 1, 2.5]:
        tangent = f(x0) + grad_f(x0) * (x_range - x0)
        ax.plot(x_range, tangent, "--", alpha=0.7, label=f"Tangent at x={x0}")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("First-Order Convexity: Tangent lines lie BELOW the function")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convexity_first_order.png", dpi=100)
    print("  Saved: convexity_first_order.png")
    plt.close()


# =============================================================================
# Exercise 3: GD on Convex vs Non-Convex
# =============================================================================

def gd_comparison():
    """Run GD on convex and non-convex functions."""
    # Convex: quadratic
    def f_convex(x):
        return 0.5 * (x[0]**2 + 10 * x[1]**2)

    def grad_convex(x):
        return np.array([x[0], 10 * x[1]])

    # Non-convex: Rastrigin
    def f_nonconvex(x):
        A = 10
        return A * 2 + (x[0]**2 - A * np.cos(2 * np.pi * x[0])) + \
               (x[1]**2 - A * np.cos(2 * np.pi * x[1]))

    def grad_nonconvex(x):
        A = 10
        return np.array([
            2 * x[0] + A * 2 * np.pi * np.sin(2 * np.pi * x[0]),
            2 * x[1] + A * 2 * np.pi * np.sin(2 * np.pi * x[1]),
        ])

    def gradient_descent(f, grad_f, x0, lr=0.01, n_iter=500):
        x = x0.copy()
        trajectory = [x.copy()]
        for _ in range(n_iter):
            g = grad_f(x)
            x = x - lr * g
            trajectory.append(x.copy())
        return np.array(trajectory), f(x)

    print("\n=== GD on Convex vs Non-Convex ===")

    # Convex: always finds global min regardless of starting point
    for start in [np.array([3.0, 3.0]), np.array([-5.0, 2.0]), np.array([0.1, -4.0])]:
        traj, fval = gradient_descent(f_convex, grad_convex, start, lr=0.05)
        print(f"  Convex | Start: {start} → End: {np.round(traj[-1], 4)}, f = {fval:.6f}")

    # Non-convex: gets stuck in local minima
    print()
    for start in [np.array([3.0, 3.0]), np.array([0.5, 0.5]), np.array([-1.2, 2.3])]:
        traj, fval = gradient_descent(f_nonconvex, grad_nonconvex, start, lr=0.001)
        print(f"  Non-convex | Start: {start} → End: {np.round(traj[-1], 4)}, f = {fval:.4f}")
    print("  (Global minimum is at (0,0) with f=0. Non-convex GD misses it.)")


# =============================================================================
# Exercise 4: Convexity-Preserving Operations
# =============================================================================

def convexity_operations_demo():
    """Demonstrate operations that preserve convexity."""
    print("\n=== Convexity-Preserving Operations ===")

    # Sum of convex functions is convex
    def f1(x): return x**2
    def f2(x): return abs(x)
    def f_sum(x): return f1(x) + f2(x)

    x_vals = np.linspace(-3, 3, 200)

    # Pointwise max of convex functions is convex
    def g1(x): return x + 1
    def g2(x): return -x + 1
    def f_max(x): return np.maximum(g1(x), g2(x))  # = |x| + 1... convex!

    print("  f₁(x) = x²: Convex ✓")
    print("  f₂(x) = |x|: Convex ✓")
    print("  f₁ + f₂ = x² + |x|: Convex ✓ (sum preserves convexity)")
    print("  max(x+1, -x+1): Convex ✓ (pointwise max preserves convexity)")
    print("  f(Ax+b) where f convex: Convex ✓ (composition with affine)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Convexity Classification")
    print("=" * 60)

    functions_1d = {
        "x²": (lambda x: x**2, lambda x: 2.0),
        "log(x)": (lambda x: np.log(x), lambda x: -1/x**2),
        "x³": (lambda x: x**3, lambda x: 6*x),
        "eˣ": (lambda x: np.exp(x), lambda x: np.exp(x)),
        "-log(x)": (lambda x: -np.log(x), lambda x: 1/x**2),
        "softplus": (lambda x: np.log(1 + np.exp(x)),
                     lambda x: np.exp(x) / (1 + np.exp(x))**2),
    }

    for name, (f, h) in functions_1d.items():
        domain = (0.1, 5) if "log" in name else (-5, 5)
        result = classify_convexity_1d(f, h, domain)
        print(f"  {name:<15}: {result}")

    # nD classification
    print("\n  nD functions:")

    # Least squares ||Ax-b||² — always convex
    A = np.random.randn(5, 3)
    b = np.random.randn(5)
    def ls_hessian(x):
        return 2 * A.T @ A
    points = [np.random.randn(3) for _ in range(100)]
    print(f"  ||Ax-b||²     : {classify_convexity_nd(ls_hessian, points)}")

    # Rosenbrock — not globally convex
    def rosenbrock_hessian(x):
        return np.array([
            [2 - 400*(x[1] - 3*x[0]**2), -400*x[0]],
            [-400*x[0], 200.0]
        ])
    points = [np.random.randn(2) * 2 for _ in range(100)]
    print(f"  Rosenbrock    : {classify_convexity_nd(rosenbrock_hessian, points)}")

    plot_first_order_condition()
    gd_comparison()
    convexity_operations_demo()
