# Day 11: Automatic Differentiation

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *AD gives exact derivatives at the cost of one extra function evaluation — no approximation, no truncation error.*

## Navigation
- **Previous:** [Day 10: Numerical Differentiation](day-10-numerical-differentiation.md)
- **Next:** [Day 12: Conditioning & Function Landscapes](day-12-conditioning.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Ceres Solver's `AutoDiffCostFunction` is the default and recommended way to define cost functions. It uses forward-mode AD with dual numbers (Jet types) to compute exact Jacobians automatically. JAX uses reverse-mode AD for efficient gradient computation in Python prototyping. Understanding both modes tells you when each is efficient and how Ceres Jet types work under the hood.

---

## Theory (45 min)

### 11.1 The AD Landscape

|  | Numerical | Symbolic | Automatic |
|--|-----------|----------|-----------|
| Accuracy | $O(h^k)$ | Exact | Exact (machine precision) |
| Expression swell | No | Yes | No |
| Implementation | Perturb inputs | Manipulate expressions | Transform program |
| Cost | $O(n)$ evals for gradient | Explosion for large graphs | 1 forward or 1 reverse pass |

### 11.2 Forward Mode: Dual Numbers

A **dual number** is $a + b\epsilon$ where $\epsilon^2 = 0$ (not zero itself, but its square is).

Arithmetic:
$$
(a + b\epsilon) + (c + d\epsilon) = (a+c) + (b+d)\epsilon
$$
$$
(a + b\epsilon) \cdot (c + d\epsilon) = ac + (ad + bc)\epsilon
$$

Evaluating $f(x + \epsilon)$ gives $f(x) + f'(x)\epsilon$ — the derivative rides along for free!

**For a function $f: \mathbb{R}^n \to \mathbb{R}^m$:**
- Set input $x_i = x_i + \delta_{ij}\epsilon$ to get column $j$ of the Jacobian
- Need $n$ passes for the full Jacobian
- **Efficient when $m \gg n$** (few inputs, many outputs)

### 11.3 Ceres Jet Type

Ceres implements dual numbers as `Jet<T, N>`:
```cpp
// Jet<double, 3> for a function with 3 parameters
// Stores: value (double) + 3 partial derivatives
struct Jet {
    double a;           // f(x)
    double v[N];        // [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
};
```
When you write `AutoDiffCostFunction`, Ceres evaluates your cost function with `Jet` types instead of `double`, automatically computing all derivatives.

### 11.4 Reverse Mode: Backpropagation

Build a **computation graph** (DAG) during the forward pass, then propagate **adjoints** backward.

For $f: \mathbb{R}^n \to \mathbb{R}$ (scalar output):
- Forward: compute $f(x)$ and store intermediate values
- Backward: propagate $\bar{v}_i = \partial f / \partial v_i$ from output to inputs

$$\bar{v}_i = \sum_{j \in \text{children}(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i}$$

**One reverse pass gives the full gradient** $\nabla f \in \mathbb{R}^n$ — cost is $O(1) \times$ the forward pass, regardless of $n$.

### 11.5 When to Use Which

| Scenario | Best Mode | Why |
|----------|-----------|-----|
| $f: \mathbb{R}^n \to \mathbb{R}$ (loss function) | Reverse | 1 pass for full gradient |
| $f: \mathbb{R}^3 \to \mathbb{R}^{1000}$ (cost function, few params) | Forward | $n=3$ passes total |
| Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$ | Forward if $n < m$, Reverse if $m < n$ | Minimize passes |
| Ceres cost function | Forward (Jet) | Built-in, per-residual |
| Training neural nets | Reverse (backprop) | Scalar loss, many params |

### 11.6 JAX: Composable AD

JAX provides both modes:
```python
jax.grad(f)(x)        # Reverse mode gradient
jax.jacfwd(f)(x)      # Forward mode Jacobian
jax.jacrev(f)(x)      # Reverse mode Jacobian
jax.hessian(f)(x)     # Hessian via nested AD
```

JAX transformations compose: `grad(grad(f))` gives $f''$, `vmap(grad(f))` batches gradient computation.

---

## Implementation (60 min)

### Exercise 1: Dual Number Class
```python
# See: code/week02/automatic_differentiation.py
```
Implement a `DualNumber` class supporting `+`, `*`, `sin`, `cos`, `exp`, `pow`. Verify $f'(x) = \cos(x)$ for $f(x) = \sin(x)$.

### Exercise 2: Forward-Mode AD Engine
Build a mini forward-mode AD that computes the gradient of arbitrary compositions.

### Exercise 3: Reverse-Mode Computation Graph
Implement a simple reverse-mode AD with a computation graph and backward pass.

### Exercise 4: JAX Gradients and Hessians
Use JAX's `grad`, `jacfwd`, `jacrev`, `hessian` on optimization-relevant functions (Rosenbrock, quadratic forms). Compare forward vs reverse Jacobian computation time for different input/output dimensions.

---

## Practice (45 min)

1. Evaluate $f(x) = x^3 + 2x$ at $x = 3$ using dual numbers. Show the full calculation.
2. For $f(x,y) = xy + \sin(x)$, how many forward-mode passes to get $\nabla f$? How many reverse-mode passes?
3. Why is Ceres forward-mode AD efficient? (Hint: typical cost function has 6-15 parameters.)
4. Implement $\sqrt{x}$ for dual numbers. What is the derivative at $x=0$?
5. If $f: \mathbb{R}^{1000} \to \mathbb{R}^{1000}$, what's the cost of the full Jacobian via forward vs reverse mode?
6. JAX's `jit(grad(f))` compiles the gradient computation. Why is compilation important for repeated evaluations?
7. Can AD handle `if` statements? What about `while` loops? What limitations exist?
8. Explain: "AD is not symbolic differentiation — it operates on programs, not expressions."

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Higher-order dual numbers</summary>

**Problem:** Extend dual numbers to compute second derivatives: use $a + b\epsilon_1 + c\epsilon_2 + d\epsilon_1\epsilon_2$ where $\epsilon_1^2 = \epsilon_2^2 = 0$ (hyper-dual numbers). Show that $d$ gives the second derivative.

**Solution:** Evaluate $f(x + \epsilon_1 + \epsilon_2)$:
$$f(x) + f'(x)(\epsilon_1 + \epsilon_2) + \frac{1}{2}f''(x) \cdot 2\epsilon_1\epsilon_2$$
The coefficient of $\epsilon_1\epsilon_2$ is $f''(x)$. This avoids the instability of finite-differencing the first derivative. Ceres doesn't use this (it typically only needs first derivatives), but it's useful for Newton methods where you need the Hessian exactly.
</details>

<details>
<summary>🎯 Challenge 2: Reverse-mode for a neural network layer</summary>

**Problem:** For one fully-connected layer $y = \sigma(Wx + b)$ with $W \in \mathbb{R}^{m \times n}$, write the forward and backward pass explicitly. Verify the gradient $\frac{\partial L}{\partial W}$ matches what you'd get from JAX.

**Solution:** Forward: $z = Wx + b$, $y = \sigma(z)$. Given upstream gradient $\bar{y} = \partial L / \partial y$:
- $\bar{z} = \bar{y} \odot \sigma'(z)$ (element-wise)
- $\bar{W} = \bar{z} \cdot x^T$ (outer product, $m \times n$)
- $\bar{b} = \bar{z}$
- $\bar{x} = W^T \bar{z}$ (for propagating to previous layer)

Total cost: one forward pass + one backward pass, each $O(mn)$. The gradient w.r.t. all $mn + m$ parameters is computed in one backward pass — this is why reverse mode is the foundation of deep learning.
</details>

<details>
<summary>🎯 Challenge 3: AD through an iterative solver</summary>

**Problem:** You solve $Ax = b$ inside your cost function (e.g., physics simulation). Can AD differentiate through the solve? What are the options?

**Solution:** Option 1: **Unroll** the iterative solver (e.g., 50 CG iterations) and let AD propagate through each step. Pros: automatic. Cons: memory ($O(\text{iterations})$), numerical instability for many iterations. Option 2: **Implicit differentiation.** At convergence, $Ax^* = b$. Differentiating: $A \frac{dx^*}{d\theta} = \frac{db}{d\theta} - \frac{dA}{d\theta} x^*$. Solve this linear system for $\frac{dx^*}{d\theta}$. Pros: memory-efficient, stable. Cons: requires implementing the adjoint. This is what JAX's `custom_vjp` and Ceres's `LocalParameterization` patterns enable. The implicit approach is almost always preferred for optimization-in-the-loop.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement and use dual numbers for forward-mode AD
- [ ] I understand reverse-mode AD (backpropagation) and when it's efficient
- [ ] I know why Ceres uses forward-mode Jet types
- [ ] I can use JAX's `grad`, `jacfwd`, `jacrev`, and `hessian`

---

## Key Takeaways

1. **Forward mode** uses dual numbers — efficient when few inputs (Ceres cost functions, $n \leq 15$).
2. **Reverse mode** builds and traverses a computation graph — efficient when few outputs (loss functions, neural nets).
3. **Ceres AutoDiff** = forward mode with Jet types = exact derivatives at ~3× the cost of a function evaluation.
4. **JAX** provides composable forward and reverse AD — prototype in Python, then port to Ceres C++ for production.
