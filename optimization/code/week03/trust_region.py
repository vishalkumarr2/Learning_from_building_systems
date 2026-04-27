"""
Day 18: Trust Region Methods — Companion Code
Phase II, Week 3

Run: python trust_region.py
"""

import numpy as np


# =============================================================================
# Test Functions
# =============================================================================

def rosenbrock_fgh(x):
    """Rosenbrock: f, gradient, Hessian."""
    f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    g = np.array([
        -2*(1 - x[0]) + 100 * 2*(x[1] - x[0]**2) * (-2*x[0]),
        100 * 2*(x[1] - x[0]**2)
    ])
    H = np.array([
        [2 + 100*(12*x[0]**2 - 4*x[1]), -400*x[0]],
        [-400*x[0], 200]
    ])
    return f, g, H


def saddle_fgh(x):
    """f = x^2 - y^2 — saddle point at origin."""
    f = x[0]**2 - x[1]**2
    g = np.array([2*x[0], -2*x[1]])
    H = np.array([[2.0, 0.0], [0.0, -2.0]])
    return f, g, H


# =============================================================================
# Exercise 1: Cauchy Point Trust Region
# =============================================================================

def cauchy_point(g, H, delta):
    """Compute the Cauchy point within trust region ||p|| <= delta."""
    gnorm = np.linalg.norm(g)
    if gnorm < 1e-16:
        return np.zeros_like(g)

    gHg = g @ H @ g

    if gHg > 0:
        tau = min(1.0, gnorm**3 / (delta * gHg))
    else:
        tau = 1.0

    p_c = -tau * (delta / gnorm) * g
    return p_c


def trust_region_cauchy(fgh, x0, delta0=1.0, max_iter=500, tol=1e-10):
    """Trust region method using Cauchy point only."""
    x = x0.copy().astype(float)
    delta = delta0
    history = {"f": [], "grad_norm": [], "delta": [], "rho": []}

    for k in range(max_iter):
        f, g, H = fgh(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(f)
        history["grad_norm"].append(gnorm)
        history["delta"].append(delta)

        if gnorm < tol:
            break

        # Cauchy point
        p = cauchy_point(g, H, delta)

        # Evaluate actual vs predicted reduction
        f_new, _, _ = fgh(x + p)
        actual_red = f - f_new
        predicted_red = -(g @ p + 0.5 * p @ H @ p)

        if predicted_red < 1e-16:
            rho = 0.0
        else:
            rho = actual_red / predicted_red

        history["rho"].append(rho)

        # Update x
        if rho > 0.01:
            x = x + p

        # Update radius
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-10:
            delta = min(2 * delta, 100.0)

    return x, history


# =============================================================================
# Exercise 2: Dogleg Trust Region
# =============================================================================

def dogleg_step(g, H, delta):
    """Compute the dogleg step within trust region ||p|| <= delta."""
    gnorm = np.linalg.norm(g)
    if gnorm < 1e-16:
        return np.zeros_like(g)

    # Full Newton step
    try:
        eigvals = np.linalg.eigvalsh(H)
        if np.min(eigvals) > 1e-10:
            p_n = np.linalg.solve(H, -g)
        else:
            # Hessian not PD — use regularized
            mu = max(0, -np.min(eigvals) + 1e-4)
            p_n = np.linalg.solve(H + mu * np.eye(len(g)), -g)
    except np.linalg.LinAlgError:
        p_n = -g  # fallback to gradient

    # If Newton step is inside trust region, use it
    if np.linalg.norm(p_n) <= delta:
        return p_n

    # Cauchy point (steepest descent direction scaled)
    gHg = g @ H @ g
    if gHg > 0:
        alpha_sd = gnorm**2 / gHg
    else:
        alpha_sd = delta / gnorm
    p_c = -alpha_sd * g

    # If Cauchy point is outside trust region, scale gradient to boundary
    if np.linalg.norm(p_c) >= delta:
        return -delta / gnorm * g

    # Dogleg: find tau such that ||p_c + tau*(p_n - p_c)|| = delta
    diff = p_n - p_c
    a = diff @ diff
    b = 2 * p_c @ diff
    c = p_c @ p_c - delta**2
    discriminant = b**2 - 4*a*c

    if discriminant < 0 or a < 1e-16:
        return -delta / gnorm * g

    tau = (-b + np.sqrt(discriminant)) / (2 * a)
    tau = np.clip(tau, 0.0, 1.0)

    return p_c + tau * diff


def trust_region_dogleg(fgh, x0, delta0=1.0, max_iter=500, tol=1e-10):
    """Trust region method with dogleg step."""
    x = x0.copy().astype(float)
    delta = delta0
    history = {"f": [], "grad_norm": [], "delta": [], "rho": []}

    for k in range(max_iter):
        f, g, H = fgh(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(f)
        history["grad_norm"].append(gnorm)
        history["delta"].append(delta)

        if gnorm < tol:
            break

        # Dogleg step
        p = dogleg_step(g, H, delta)

        # Evaluate ratio
        f_new, _, _ = fgh(x + p)
        actual_red = f - f_new
        predicted_red = -(g @ p + 0.5 * p @ H @ p)

        if predicted_red < 1e-16:
            rho = 0.0
        else:
            rho = actual_red / predicted_red

        history["rho"].append(rho)

        # Accept/reject
        if rho > 0.01:
            x = x + p

        # Radius update
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-8 * delta:
            delta = min(2 * delta, 100.0)

    return x, history


# =============================================================================
# Exercise 3: Comparison with Line Search Newton
# =============================================================================

def newton_line_search(fgh, x0, max_iter=200, tol=1e-10):
    """Newton with backtracking for comparison."""
    x = x0.copy().astype(float)
    history = {"f": [], "grad_norm": []}

    for k in range(max_iter):
        f, g, H = fgh(x)
        gnorm = np.linalg.norm(g)
        history["f"].append(f)
        history["grad_norm"].append(gnorm)

        if gnorm < tol:
            break

        # Modified Newton direction
        eigvals = np.linalg.eigvalsh(H)
        mu = max(0, -np.min(eigvals) + 1e-4)
        delta_n = np.linalg.solve(H + mu * np.eye(len(g)), -g)

        # Backtracking
        alpha = 1.0
        for _ in range(50):
            f_new, _, _ = fgh(x + alpha * delta_n)
            if f_new <= f + 1e-4 * alpha * (g @ delta_n):
                break
            alpha *= 0.5

        x = x + alpha * delta_n

    return x, history


def compare_methods():
    """Compare Cauchy, Dogleg, and Line Search Newton."""
    print("=== Exercise 3: Method Comparison on Rosenbrock ===")
    x_star = np.array([1.0, 1.0])

    starts = [
        np.array([-1.0, 1.0]),
        np.array([5.0, -5.0]),
        np.array([-3.0, 3.0]),
    ]

    for x0 in starts:
        x_c, h_c = trust_region_cauchy(rosenbrock_fgh, x0, delta0=1.0)
        x_d, h_d = trust_region_dogleg(rosenbrock_fgh, x0, delta0=1.0)
        x_n, h_n = newton_line_search(rosenbrock_fgh, x0)

        print(f"\n  x0 = [{x0[0]:>5.1f}, {x0[1]:>5.1f}]:")
        print(f"    Cauchy TR:  {len(h_c['f']):>5} iters, "
              f"err = {np.linalg.norm(x_c - x_star):.2e}")
        print(f"    Dogleg TR:  {len(h_d['f']):>5} iters, "
              f"err = {np.linalg.norm(x_d - x_star):.2e}")
        print(f"    Newton LS:  {len(h_n['f']):>5} iters, "
              f"err = {np.linalg.norm(x_n - x_star):.2e}")


# =============================================================================
# Exercise 4: Saddle Point Handling
# =============================================================================

def test_saddle_point():
    """Trust region navigates away from saddle point."""
    print("\n=== Exercise 4: Saddle Point f = x² - y² ===")
    x0 = np.array([0.1, 0.1])

    x_tr, h_tr = trust_region_dogleg(saddle_fgh, x0, delta0=0.5, max_iter=100)
    print(f"  Dogleg TR from {x0}: converged to [{x_tr[0]:.4f}, {x_tr[1]:.4f}], "
          f"f = {saddle_fgh(x_tr)[0]:.4f}")
    print(f"  (Should move toward y → ±∞ since f = x² - y² is unbounded below)")


# =============================================================================
# Exercise 5: Radius Adaptation Visualization (text)
# =============================================================================

def visualize_radius_adaptation():
    """Show how trust region radius evolves."""
    print("\n=== Exercise 5: Radius Adaptation ===")
    x0 = np.array([-2.0, 2.0])

    _, hist = trust_region_dogleg(rosenbrock_fgh, x0, delta0=0.5, max_iter=100)

    print(f"  {'Iter':>4}  {'f':>12}  {'||g||':>10}  {'Δ':>10}  {'ρ':>8}")
    n_show = min(20, len(hist["f"]))
    for i in range(n_show):
        rho = hist["rho"][i] if i < len(hist["rho"]) else float('nan')
        print(f"  {i:>4}  {hist['f'][i]:>12.4f}  "
              f"{hist['grad_norm'][i]:>10.2e}  "
              f"{hist['delta'][i]:>10.4f}  {rho:>8.3f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    compare_methods()
    test_saddle_point()
    visualize_radius_adaptation()
