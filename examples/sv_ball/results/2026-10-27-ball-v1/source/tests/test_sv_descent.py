"""Legacy boundary solver regression checks; public fits now use the SV ball."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit
from hapc import hapc_core as native
from hapc.core import pcghal
from hapc.single import _calibrate_logistic_intercept, single_pcghal


class TestSVDescent(unittest.TestCase):
    def check_path(self, out, V, y, Z, logistic=False):
        hist = out.alphaiters
        radii = np.abs(hist @ V.T).sum(axis=1)
        assert_allclose(radii, radii[0], rtol=2e-12, atol=2e-14)
        if logistic:
            risks = []
            for a in hist:
                eta = Z @ a
                eta += _calibrate_logistic_intercept((y+1)/2, eta)
                risks.append(np.logaddexp(0, -y*eta).mean())
        else:
            risks = np.mean((y[None, :] - hist @ Z.T)**2, axis=1)
        # NumPy BLAS and Eigen may round large squared residual sums
        # differently. Bound only floating-point summation error, not a
        # statistical optimisation tolerance.
        risk_roundoff = 64*np.finfo(float).eps*max(1., np.max(np.abs(risks)))
        self.assertTrue(np.all(np.diff(risks) <= risk_roundoff))
        assert_allclose(out.alpha, hist[-1])
        assert_allclose(out.risk, risks[-1], rtol=1e-10, atol=1e-12)

    def test_zero_beta_cancellation(self):
        V = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
        Z, y, a = np.eye(2), np.array([10., 0.]), np.ones(2)
        out = native.pcghal_boundary_call(y, Z, V, a, 500, 1e-10, .8, False, "grad")
        self.check_path(out, V, y, Z)
        self.assertLess(out.risk, np.mean((y-Z@a)**2))
        self.assertGreater(out.iter, 0)

    def test_random_gaussian_and_logistic_paths(self):
        for logistic in (False, True):
            for seed in range(20):
                rng = np.random.default_rng(seed)
                V = np.linalg.qr(rng.normal(size=(9, 4)))[0]
                Z = rng.normal(size=(30, 4))*10
                a = rng.normal(size=4)
                # A genuine zero coefficient with orthonormal V, rather than
                # an arbitrary non-orthogonal penalty transform.
                a[-1] = -(V[0, :-1]@a[:-1])/V[0, -1]
                y = rng.choice([-1., 1.], 30) if logistic else rng.normal(size=30)*100
                if logistic:
                    out = native.pcghal_boundary_classi_call(y, Z, V, a, 250, 1e-10, .8, False)
                else:
                    out = native.pcghal_boundary_call(y, Z, V, a, 250, 1e-10, .8, False, "grad")
                self.check_path(out, V, y, Z, logistic)
                self.assertGreater(out.iter, 0)

    def test_smooth_face_known_minimum(self):
        # Positive orthant, C=2: Euclidean projection of y onto sum(alpha)=2.
        Z, V, y = np.eye(2), np.eye(2), np.array([2., 1.])
        out = native.pcghal_boundary_call(y, Z, V, np.ones(2), 3000, 1e-10, .8, False, "grad")
        assert_allclose(out.alpha, [1.5, .5], atol=2e-6)
        self.check_path(out, V, y, Z)

    def test_zero_radius_and_no_steps(self):
        for a in (np.zeros(2), np.ones(2)):
            out = native.pcghal_boundary_call(np.ones(2), np.eye(2), np.eye(2), a, 0, 1e-8, .8, False, "grad")
            assert_allclose(out.alpha, a)
            self.assertEqual(out.iter, 0)

    def test_small_nonzero_radius(self):
        a = np.array([2e-8, 1e-8])
        out = native.pcghal_boundary_call(np.array([10., 0.]), np.eye(2), np.eye(2), a, 100, 1e-20, .8, False, "grad")
        self.check_path(out, np.eye(2), np.array([10., 0.]), np.eye(2))

    def test_stationarity_is_not_global_optimality(self):
        # Explicitly preserve this warning: multiplicative updates cannot
        # activate the second PC. Its tangent score can vanish at a bad point.
        out = native.pcghal_boundary_call(np.array([0., 2.]), np.eye(2), np.eye(2), np.array([1., 0.]), 100, 1e-8, .8, False, "grad")
        self.assertEqual(out.status, "tangent_stationary")
        self.assertGreater(out.risk, .5)  # feasible [0,1] has risk .5

    def test_intercept_with_extreme_offsets_and_one_class(self):
        eta = np.array([-1000., -999., 800., 900.])
        y = np.array([0., 1., 0., 0.])
        b = _calibrate_logistic_intercept(y, eta)
        self.assertLess(abs(np.mean(expit(eta+b))-y.mean()), 1e-11)
        for yy in (np.zeros(4), np.ones(4)):
            out = native.pcghal_boundary_classi_call(2*yy-1, np.c_[eta, np.ones(4)], np.eye(2), np.ones(2), 20, 1e-8, .8, False)
            self.assertTrue(np.isfinite(out.risk))
            self.check_path(out, np.eye(2), 2*yy-1, np.c_[eta, np.ones(4)], True)

    def test_validation_and_public_diagnostics(self):
        args = [np.ones(2), np.eye(2), np.eye(2), np.ones(2)]
        for factor in (0, 1, -1, np.nan):
            with self.assertRaises(RuntimeError):
                native.pcghal_boundary_call(*args, 100, 1e-8, factor, False, "grad")
        with self.assertRaises(RuntimeError):
            native.pcghal_boundary_call(np.ones(3), *args[1:], 100, 1e-8, .8, False, "grad")
        out = pcghal(*args)
        self.assertNotEqual(out.status, "not_reported")

    def test_public_single_uses_actual_history_and_predictions(self):
        rng = np.random.default_rng(8)
        X = rng.uniform(size=(35, 2)); y = np.sin(X[:, 0]*6)+X[:, 1]
        out = single_pcghal(X, y, 1, 10, .01, predict=X, ini="2", tol=1e-8)
        opt = out.optimizer_output
        self.assertEqual(len(opt.alphaiters), opt.iter+1)
        des = native.pchal_des(X, 1, 10, True)
        assert_allclose(opt.beta, des.V@opt.alpha, atol=1e-9)
        assert_allclose(out.predictions, (des.U*des.d)@opt.alpha+y.mean(), atol=1e-9)
        self.assertNotEqual(opt.status, "not_reported")


if __name__ == "__main__":
    unittest.main()
