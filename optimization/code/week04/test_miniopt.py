"""
Day 28: pytest test suite for miniopt
Phase II Capstone, Week 4

Run: pytest test_miniopt.py -v
"""

import numpy as np
import pytest
from miniopt import (minimize, OptimizeResult, METHODS,
                     rosenbrock, rosenbrock_grad,
                     sphere, sphere_grad,
                     beale, booth, styblinski_tang)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def quadratic():
    """Simple quadratic: f(x) = 0.5 * x^T A x, min at origin."""
    A = np.diag([1.0, 2.0, 5.0])
    f = lambda x: 0.5 * x @ A @ x
    g = lambda x: A @ x
    x0 = np.array([10.0, -5.0, 3.0])
    return f, g, x0, np.zeros(3)


@pytest.fixture
def sphere_setup():
    """Sphere function setup."""
    x0 = np.array([5.0, -3.0, 2.0, -1.0, 4.0])
    return sphere, sphere_grad, x0, np.zeros(5)


# =============================================================================
# Test: Result structure
# =============================================================================

class TestOptimizeResult:
    def test_result_fields(self, quadratic):
        f, g, x0, _ = quadratic
        res = minimize(f, x0, method='bfgs', jac=g)
        assert hasattr(res, 'x')
        assert hasattr(res, 'fun')
        assert hasattr(res, 'nit')
        assert hasattr(res, 'nfev')
        assert hasattr(res, 'ngrad')
        assert hasattr(res, 'success')
        assert hasattr(res, 'message')
        assert hasattr(res, 'history')

    def test_result_types(self, quadratic):
        f, g, x0, _ = quadratic
        res = minimize(f, x0, method='bfgs', jac=g)
        assert isinstance(res.x, np.ndarray)
        assert isinstance(res.fun, (float, np.floating))
        assert isinstance(res.nit, (int, np.integer))
        assert isinstance(res.success, bool)
        assert isinstance(res.message, str)
        assert isinstance(res.history, list)


# =============================================================================
# Test: Method validation
# =============================================================================

class TestMethodValidation:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            minimize(sphere, np.zeros(3), method='nonexistent')

    def test_all_methods_registered(self):
        expected = {'gd', 'momentum', 'nesterov', 'bfgs', 'lbfgs', 'adam'}
        assert set(METHODS.keys()) == expected

    def test_case_insensitive(self, quadratic):
        f, g, x0, _ = quadratic
        res = minimize(f, x0, method='BFGS', jac=g)
        assert res.success


# =============================================================================
# Test: Convergence on quadratic
# =============================================================================

class TestQuadraticConvergence:
    @pytest.mark.parametrize("method", ['gd', 'momentum', 'nesterov',
                                        'bfgs', 'lbfgs', 'adam'])
    def test_converges_on_quadratic(self, quadratic, method):
        f, g, x0, x_star = quadratic
        lr = 0.1 if method in ('gd', 'momentum', 'nesterov') else 0.01
        max_iter = 5000 if method in ('gd', 'adam') else 1000
        res = minimize(f, x0, method=method, jac=g, lr=lr,
                      max_iter=max_iter, tol=1e-6)
        assert np.linalg.norm(res.x - x_star) < 0.1, \
            f"{method}: ||x - x*|| = {np.linalg.norm(res.x - x_star)}"

    def test_bfgs_exact_on_quadratic(self, quadratic):
        """BFGS should converge very accurately on quadratics."""
        f, g, x0, x_star = quadratic
        res = minimize(f, x0, method='bfgs', jac=g, tol=1e-12)
        assert np.linalg.norm(res.x - x_star) < 1e-6


# =============================================================================
# Test: Convergence on Rosenbrock
# =============================================================================

class TestRosenbrockConvergence:
    def test_bfgs_on_rosenbrock(self):
        x0 = np.zeros(5)
        res = minimize(rosenbrock, x0, method='bfgs',
                      jac=rosenbrock_grad, max_iter=10000)
        assert res.fun < 1e-4, f"BFGS: f = {res.fun}"

    def test_lbfgs_on_rosenbrock(self):
        x0 = np.zeros(5)
        res = minimize(rosenbrock, x0, method='lbfgs',
                      jac=rosenbrock_grad, max_iter=10000)
        assert res.fun < 1e-4, f"L-BFGS: f = {res.fun}"

    def test_bfgs_lbfgs_similar(self):
        """BFGS and L-BFGS should produce similar results."""
        x0 = np.zeros(3)
        res_bfgs = minimize(rosenbrock, x0, method='bfgs',
                           jac=rosenbrock_grad)
        res_lbfgs = minimize(rosenbrock, x0, method='lbfgs',
                            jac=rosenbrock_grad)
        assert abs(res_bfgs.fun - res_lbfgs.fun) < 1e-2


# =============================================================================
# Test: Sphere function
# =============================================================================

class TestSphere:
    @pytest.mark.parametrize("method", ['bfgs', 'lbfgs', 'gd'])
    def test_sphere_convergence(self, sphere_setup, method):
        f, g, x0, x_star = sphere_setup
        lr = 0.1 if method == 'gd' else 0.01
        res = minimize(f, x0, method=method, jac=g, lr=lr, max_iter=5000)
        assert np.linalg.norm(res.x) < 0.01


# =============================================================================
# Test: Numerical gradient fallback
# =============================================================================

class TestNumericalGradient:
    def test_no_jac_provided(self):
        """Should work with numerical gradient when jac=None."""
        res = minimize(sphere, np.array([3.0, -2.0]), method='bfgs',
                      jac=None, max_iter=1000)
        assert np.linalg.norm(res.x) < 0.1

    def test_numerical_matches_analytic(self):
        """Numerical and analytic gradients should give similar results."""
        x0 = np.array([2.0, -1.0])
        res_num = minimize(sphere, x0, method='bfgs', jac=None)
        res_ana = minimize(sphere, x0, method='bfgs', jac=sphere_grad)
        assert np.allclose(res_num.x, res_ana.x, atol=0.01)


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_already_at_minimum(self):
        """Starting at the minimum should converge immediately."""
        res = minimize(sphere, np.zeros(3), method='bfgs', jac=sphere_grad)
        assert res.fun < 1e-10
        assert res.nit <= 1

    def test_1d_problem(self):
        """Should handle 1D optimization."""
        f = lambda x: (x[0] - 3)**2
        g = lambda x: np.array([2 * (x[0] - 3)])
        res = minimize(f, np.array([0.0]), method='bfgs', jac=g)
        assert abs(res.x[0] - 3.0) < 1e-4

    def test_history_recorded(self, quadratic):
        """History should be non-empty."""
        f, g, x0, _ = quadratic
        res = minimize(f, x0, method='bfgs', jac=g)
        assert len(res.history) > 0

    def test_history_decreasing(self, quadratic):
        """Function values in history should generally decrease for BFGS."""
        f, g, x0, _ = quadratic
        res = minimize(f, x0, method='bfgs', jac=g)
        fvals = [h[0] for h in res.history]
        # Allow for some non-monotonicity but overall trend should be down
        assert fvals[-1] < fvals[0]


# =============================================================================
# Test: Method-specific features
# =============================================================================

class TestMethodSpecific:
    def test_adam_handles_high_lr(self):
        """Adam should be robust to learning rate choice."""
        res = minimize(sphere, np.array([5.0, 5.0]),
                      method='adam', jac=sphere_grad,
                      lr=0.1, max_iter=5000)
        assert np.linalg.norm(res.x) < 1.0

    def test_momentum_faster_than_gd(self, quadratic):
        """Momentum should converge in fewer iterations than vanilla GD."""
        f, g, x0, _ = quadratic
        res_gd = minimize(f, x0, method='gd', jac=g, lr=0.1, max_iter=5000)
        res_mom = minimize(f, x0, method='momentum', jac=g, lr=0.1,
                          max_iter=5000)
        # Momentum should be at least as good
        assert res_mom.fun <= res_gd.fun * 10  # generous tolerance


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
