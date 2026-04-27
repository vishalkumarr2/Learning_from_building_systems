"""
Day 11: Automatic Differentiation — Companion Code
Phase I, Week 2

Run: python automatic_differentiation.py
"""

import numpy as np
from typing import Union


# =============================================================================
# Exercise 1: Dual Number Class
# =============================================================================

class DualNumber:
    """Dual number a + b*ε where ε² = 0."""

    def __init__(self, real: float, dual: float = 0.0):
        self.real = float(real)
        self.dual = float(dual)

    def __repr__(self):
        return f"Dual({self.real:.6f} + {self.dual:.6f}ε)"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.real + other, self.dual)
        return DualNumber(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.real - other, self.dual)
        return DualNumber(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(other - self.real, -self.dual)
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.real * other, self.dual * other)
        return DualNumber(self.real * other.real,
                         self.real * other.dual + self.dual * other.real)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.real / other, self.dual / other)
        denom = other.real ** 2
        real = self.real / other.real
        dual = (self.dual * other.real - self.real * other.dual) / denom
        return DualNumber(real, dual)

    def __pow__(self, n):
        if isinstance(n, (int, float)):
            return DualNumber(self.real ** n,
                            n * self.real ** (n - 1) * self.dual)
        raise NotImplementedError("Dual^Dual not implemented")

    def __neg__(self):
        return DualNumber(-self.real, -self.dual)


def dual_sin(x: DualNumber) -> DualNumber:
    return DualNumber(np.sin(x.real), np.cos(x.real) * x.dual)


def dual_cos(x: DualNumber) -> DualNumber:
    return DualNumber(np.cos(x.real), -np.sin(x.real) * x.dual)


def dual_exp(x: DualNumber) -> DualNumber:
    ex = np.exp(x.real)
    return DualNumber(ex, ex * x.dual)


def dual_log(x: DualNumber) -> DualNumber:
    return DualNumber(np.log(x.real), x.dual / x.real)


def dual_sqrt(x: DualNumber) -> DualNumber:
    s = np.sqrt(x.real)
    return DualNumber(s, x.dual / (2 * s))


def test_dual_numbers():
    """Test dual number AD."""
    print("=== Exercise 1: Dual Number Class ===")

    # f(x) = sin(x), f'(x) = cos(x), test at x=1
    x = DualNumber(1.0, 1.0)  # x + ε
    result = dual_sin(x)
    print(f"  sin(1 + ε) = {result}")
    print(f"  f(1)  = {result.real:.10f}, exact = {np.sin(1.0):.10f}")
    print(f"  f'(1) = {result.dual:.10f}, exact = {np.cos(1.0):.10f}")

    # f(x) = x³ + 2x at x=3
    x = DualNumber(3.0, 1.0)
    result = x ** 3 + 2 * x
    print(f"\n  x³+2x at x=3: {result}")
    print(f"  f(3)  = {result.real:.1f}, exact = {33.0:.1f}")
    print(f"  f'(3) = {result.dual:.1f}, exact = {29.0:.1f}")

    # f(x) = exp(sin(x)) at x=0.5
    x = DualNumber(0.5, 1.0)
    result = dual_exp(dual_sin(x))
    exact_val = np.exp(np.sin(0.5))
    exact_der = np.cos(0.5) * np.exp(np.sin(0.5))
    print(f"\n  exp(sin(0.5)): {result}")
    print(f"  f(0.5)  = {result.real:.10f}, exact = {exact_val:.10f}")
    print(f"  f'(0.5) = {result.dual:.10f}, exact = {exact_der:.10f}")


# =============================================================================
# Exercise 2: Forward-Mode AD for Multi-Variable Functions
# =============================================================================

def forward_mode_gradient(f, x: np.ndarray) -> tuple:
    """Compute gradient of scalar f: R^n -> R using forward-mode AD."""
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        # Create dual number for each variable
        x_dual = [DualNumber(x[j], 1.0 if j == i else 0.0)
                  for j in range(n)]
        result = f(x_dual)
        grad[i] = result.dual

    return f([DualNumber(xi, 0.0) for xi in x]).real, grad


def test_forward_mode():
    """Test forward-mode gradient computation."""
    print("\n=== Exercise 2: Forward-Mode AD ===")

    # f(x,y) = x*y + sin(x)
    def f(x):
        return x[0] * x[1] + dual_sin(x[0])

    point = np.array([1.5, -0.7])
    val, grad = forward_mode_gradient(f, point)

    exact_grad = np.array([
        point[1] + np.cos(point[0]),  # df/dx = y + cos(x)
        point[0]                       # df/dy = x
    ])

    print(f"  f(1.5, -0.7) = {val:.6f}")
    print(f"  AD gradient:    [{grad[0]:.6f}, {grad[1]:.6f}]")
    print(f"  Exact gradient: [{exact_grad[0]:.6f}, {exact_grad[1]:.6f}]")
    print(f"  Error: {np.linalg.norm(grad - exact_grad):.2e}")


# =============================================================================
# Exercise 3: Reverse-Mode AD (Simple Computation Graph)
# =============================================================================

class Var:
    """Variable in a computation graph for reverse-mode AD."""
    _tape = []  # global tape

    def __init__(self, value: float, children=(), grad_fns=()):
        self.value = float(value)
        self.grad = 0.0
        self._children = children
        self._grad_fns = grad_fns
        Var._tape.append(self)

    def __repr__(self):
        return f"Var({self.value:.6f})"

    def backward(self):
        """Backpropagate from this node."""
        self.grad = 1.0
        for node in reversed(Var._tape):
            for child, grad_fn in zip(node._children, node._grad_fns):
                child.grad += node.grad * grad_fn()

    @staticmethod
    def clear_tape():
        Var._tape = []

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Var(other)
        return Var(self.value + other.value,
                  (self, other),
                  (lambda: 1.0, lambda: 1.0))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Var(other)
        s, o = self, other
        return Var(self.value * other.value,
                  (self, other),
                  (lambda: o.value, lambda: s.value))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Var(-self.value, (self,), (lambda: -1.0,))

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, n):
        s = self
        return Var(self.value ** n,
                  (self,),
                  (lambda: n * s.value ** (n-1),))


def var_sin(x: Var) -> Var:
    return Var(np.sin(x.value), (x,),
              (lambda: np.cos(x.value),))


def var_exp(x: Var) -> Var:
    ex = np.exp(x.value)
    return Var(ex, (x,), (lambda: ex,))


def test_reverse_mode():
    """Test reverse-mode AD."""
    print("\n=== Exercise 3: Reverse-Mode AD ===")

    # f(x,y) = x*y + sin(x) + y²
    Var.clear_tape()
    x = Var(1.5)
    y = Var(-0.7)
    z = x * y + var_sin(x) + y ** 2

    z.backward()

    exact_dfdx = y.value + np.cos(x.value)  # y + cos(x)
    exact_dfdy = x.value + 2 * y.value       # x + 2y

    print(f"  f(1.5, -0.7) = {z.value:.6f}")
    print(f"  Reverse df/dx = {x.grad:.6f}, exact = {exact_dfdx:.6f}")
    print(f"  Reverse df/dy = {y.grad:.6f}, exact = {exact_dfdy:.6f}")
    print(f"  Note: ONE backward pass gave BOTH partial derivatives!")


# =============================================================================
# Exercise 4: JAX Gradients and Hessians
# =============================================================================

def jax_demo():
    """Demonstrate JAX AD capabilities (if available)."""
    print("\n=== Exercise 4: JAX AD ===")
    try:
        import jax
        import jax.numpy as jnp

        # Rosenbrock function
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        x0 = jnp.array([0.5, 0.5])

        # Gradient (reverse mode)
        grad_f = jax.grad(rosenbrock)
        g = grad_f(x0)
        print(f"  Rosenbrock at [0.5, 0.5]:")
        print(f"  f(x)  = {rosenbrock(x0):.4f}")
        print(f"  grad  = [{g[0]:.4f}, {g[1]:.4f}]")

        # Hessian
        hess_f = jax.hessian(rosenbrock)
        H = hess_f(x0)
        print(f"  Hessian = \n    {H}")

        # Forward vs reverse Jacobian timing
        def f_vec(x):
            return jnp.array([jnp.sin(x[0]) * x[1],
                             jnp.exp(x[0] + x[1]),
                             x[0]**2 - x[1]])

        J_fwd = jax.jacfwd(f_vec)(x0)
        J_rev = jax.jacrev(f_vec)(x0)
        print(f"\n  Forward Jacobian == Reverse Jacobian: "
              f"{jnp.allclose(J_fwd, J_rev)}")

    except ImportError:
        print("  JAX not installed. Install with: pip install jax jaxlib")
        print("  Showing manual forward/reverse comparison instead.")

        # Fallback: compare our implementations
        print("\n  Forward-mode (n passes for n inputs):")
        print("    Best for f: R^n -> R^m where n << m")
        print("  Reverse-mode (1 pass for scalar output):")
        print("    Best for f: R^n -> R where n is large")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_dual_numbers()
    test_forward_mode()
    test_reverse_mode()
    jax_demo()
