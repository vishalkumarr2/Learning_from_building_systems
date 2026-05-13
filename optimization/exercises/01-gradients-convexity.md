# Exercises: Gradients, Hessians, and Convexity

### Chapters 01 & 02: Mathematical Foundations & Unconstrained Optimization

**Self-assessment guide:** Work each problem on paper before expanding the answer. If you can solve 80% without peeking, move on to `02-least-squares-ceres.md`.

---

## Section A — Conceptual Questions

**A1.** Given $f(x, y) = x^2 y + \sin(xy)$, compute the gradient $\nabla f$ and the Hessian $H_f$. At what point(s) is the gradient zero?

<details><summary>Answer</summary>

**Gradient:**
$$\nabla f = \begin{bmatrix} 2xy + y\cos(xy) \\ x^2 + x\cos(xy) \end{bmatrix}$$

**Hessian:**
$$H_f = \begin{bmatrix} 2y - y^2\sin(xy) & 2x + \cos(xy) - xy\sin(xy) \\ 2x + \cos(xy) - xy\sin(xy) & -x^2\sin(xy) \end{bmatrix}$$

At the origin $(0, 0)$:
- $\nabla f = (0, 0)$ — it's a stationary point
- $H_f = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ with eigenvalues $\pm 1$ — indefinite → saddle point

</details>

**A2.** Prove that $f(x) = \|Ax - b\|^2$ is convex for any matrix $A$ and vector $b$. Under what condition is it strictly convex?

<details><summary>Answer</summary>

Expand: $f(x) = x^T A^T A x - 2b^T A x + b^T b$

The Hessian is $H = 2A^T A$.

A function is convex iff its Hessian is positive semidefinite everywhere.

$A^T A$ is always PSD: for any $v$, $v^T A^T A v = \|Av\|^2 \geq 0$.

Therefore $f$ is convex.

**Strictly convex** iff $A^T A \succ 0$ (positive definite), which requires $A$ to have full column rank (i.e., $\text{rank}(A) = n$ where $A \in \mathbb{R}^{m \times n}$). This ensures $\|Av\|^2 > 0$ for all $v \neq 0$.

If $A$ doesn't have full column rank, there exists $v \neq 0$ with $Av = 0$, so the function is flat in that direction — convex but not strictly convex, and the minimum is not unique.

</details>

**A3.** A colleague says: "Gradient descent always converges to the global minimum if you set the learning rate small enough." Is this correct? Provide a counterexample.

<details><summary>Answer</summary>

**Incorrect.** Gradient descent converges to a **local** minimum (or saddle point), not necessarily the global minimum.

**Counterexample:** $f(x) = x^4 - 2x^2 + 0.5$

This has two local minima at $x = \pm 1$ and a local maximum at $x = 0$. Starting from $x_0 = -0.5$, gradient descent converges to $x = -1$ regardless of learning rate, missing the global minimum at $x = 1$ (both have the same value here, but in general the left/right minima could differ).

For non-convex functions, GD is only guaranteed to find a local minimum (and with small enough step size, it converges to a stationary point, which could even be a saddle point if the direction of negative curvature isn't explored).

The statement is true **only for convex functions**, where every local minimum is global.

</details>

**A4.** Why does Newton's method converge in exactly 1 step for a quadratic function? What does this tell you about its behavior near any minimum?

<details><summary>Answer</summary>

For a quadratic $f(x) = \frac{1}{2}x^T H x + g^T x + c$:
- Gradient: $\nabla f(x) = Hx + g$
- Hessian: $\nabla^2 f(x) = H$ (constant)

Newton step: $x_{k+1} = x_k - H^{-1}(Hx_k + g) = -H^{-1}g = x^*$

The Newton step takes you **directly to the minimum** because the second-order Taylor approximation is **exact** for a quadratic.

**Near any minimum:** Any smooth function is approximately quadratic near its minimum (Taylor expansion). So Newton's method converges **quadratically** near the minimum — each step roughly doubles the number of correct digits. This is why it converges so fast in the final iterations.

</details>

---

## Section B — Computation Problems

**B1.** Implement gradient descent and Newton's method for the Rosenbrock function:

$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$

Compare:
- Number of iterations to converge ($\|\nabla f\| < 10^{-6}$)
- Path taken (plot the trajectory)
- Effect of step size on GD convergence

```python
import numpy as np

def rosenbrock(x):
    """f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Gradient of Rosenbrock function"""
    # TODO: compute and return [df/dx, df/dy]
    pass

def rosenbrock_hess(x):
    """Hessian of Rosenbrock function"""
    # TODO: compute and return 2x2 Hessian matrix
    pass

def gradient_descent(f, grad, x0, lr=0.001, tol=1e-6, max_iter=100000):
    """
    Returns: (x_final, trajectory, num_iterations)
    """
    # TODO: implement
    pass

def newton_method(f, grad, hess, x0, tol=1e-6, max_iter=100):
    """
    Returns: (x_final, trajectory, num_iterations)
    """
    # TODO: implement with line search for safety
    pass

# Start from (-1, 1)
x0 = np.array([-1.0, 1.0])
# Compare both methods. Expected: GD takes ~10,000+ iterations, Newton ~20.
```

<details><summary>Solution</summary>

```python
import numpy as np

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) + 200*(x[1] - x[0]**2)*(-2*x[0])
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

def rosenbrock_hess(x):
    dxx = 2 - 400*x[1] + 1200*x[0]**2
    dxy = -400*x[0]
    dyy = 200.0
    return np.array([[dxx, dxy], [dxy, dyy]])

def gradient_descent(f, grad, x0, lr=0.001, tol=1e-6, max_iter=100000):
    x = x0.copy()
    trajectory = [x.copy()]
    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return x, trajectory, i
        x = x - lr * g
        trajectory.append(x.copy())
    return x, trajectory, max_iter

def newton_method(f, grad, hess, x0, tol=1e-6, max_iter=100):
    x = x0.copy()
    trajectory = [x.copy()]
    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return x, trajectory, i
        H = hess(x)
        # Add damping if H is not positive definite
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dx = -g  # fallback to gradient step
        
        # Backtracking line search
        alpha = 1.0
        while f(x + alpha * dx) > f(x) + 1e-4 * alpha * g.dot(dx):
            alpha *= 0.5
            if alpha < 1e-16:
                break
        
        x = x + alpha * dx
        trajectory.append(x.copy())
    return x, trajectory, max_iter

x0 = np.array([-1.0, 1.0])

x_gd, traj_gd, n_gd = gradient_descent(rosenbrock, rosenbrock_grad, x0, lr=0.001)
print(f"GD: {n_gd} iterations, x = {x_gd}")

x_nt, traj_nt, n_nt = newton_method(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0)
print(f"Newton: {n_nt} iterations, x = {x_nt}")
```

Typical results: GD ~10,000+ iterations (with lr=0.001), Newton ~15–25 iterations. The GD trajectory zig-zags along the narrow valley; Newton cuts straight through.

</details>

**B2.** For the function $f(x) = x_1^2 + 10 x_2^2$ (an elongated bowl):

1. What is the condition number of the Hessian?
2. How many iterations does gradient descent take vs. Newton?
3. What happens to GD if you change it to $f(x) = x_1^2 + 1000 x_2^2$?

<details><summary>Answer</summary>

1. $H = \begin{bmatrix} 2 & 0 \\ 0 & 20 \end{bmatrix}$. $\kappa(H) = 20/2 = 10$.

2. GD convergence rate: $((\kappa - 1)/(\kappa + 1))^2 = (9/11)^2 \approx 0.67$ per iteration. Need about $\frac{\log(1/\epsilon)}{-\log(0.67)} \approx 35$ iterations for $\epsilon = 10^{-6}$. Newton: 1 iteration (quadratic function).

3. For $f(x) = x_1^2 + 1000 x_2^2$: $\kappa = 1000$. Rate $= (999/1001)^2 \approx 0.996$. Need $\approx 3,450$ iterations. The worse the condition number, the more GD zig-zags.

**Lesson:** Condition number directly controls GD convergence speed. Newton is invariant to conditioning.

</details>

---

## Section C — OKS Connection

**C1.** The OKS navigation estimator's EKF state includes $(x, y, \theta, v, \omega)$. The position variables are in metres ($\sim 0$ to $100$), while the heading $\theta$ is in radians ($\sim 0$ to $6.28$).

1. Why might this cause numerical issues if you were solving an optimization problem with these variables?
2. How would you rescale? What scaling factors would you choose?
3. Why does the EKF avoid this issue (mostly)?

<details><summary>Answer</summary>

1. The Hessian will have entries of very different magnitudes — position-related entries could be $10^4 \times$ larger than heading-related entries. This leads to a poorly conditioned Hessian, causing slow convergence or inaccurate steps.

2. Scale all variables to $O(1)$:
   - Position: divide by typical range ($\sim 50$ m) → scaled position $\in [-1, 2]$
   - Heading: already $O(1)$, no scaling needed
   - Velocity: divide by max velocity ($\sim 1$ m/s) → scaled $\in [0, 1]$
   - Angular velocity: divide by max ($\sim 1$ rad/s) → already $O(1)$

3. The EKF processes one measurement at a time, updating the state incrementally. It never solves a large system simultaneously where different-scale variables interact in the same Hessian. The covariance matrix $P$ naturally tracks the different scales. The issue mainly arises in batch optimization where all variables are solved at once.

</details>
