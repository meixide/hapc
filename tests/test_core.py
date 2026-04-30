"""Smoke tests for the low-level Python HAPC core functions."""

import numpy as np
import pytest

from hapc import (
    cross_kernel_hapc,
    design_hapc,
    fast_pchal,
    kernel_hapc,
    pcghal,
    pcghal_classification,
    pc_hal_classi,
    ridge_regression,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    Y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.1
    return X, Y


@pytest.fixture
def sample_binary_data():
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    Y = np.where(X[:, 0] + X[:, 1] > 0, 1.0, -1.0)
    return X, Y


class TestDesign:
    def test_design_shape(self, sample_data):
        X, _ = sample_data
        des = design_hapc(X, max_degree=2, npcs=5, center=True)
        assert des.U.shape == (100, 5)
        assert des.d.shape == (5,)
        assert des.V.shape[1] == 5

    def test_design_no_center(self, sample_data):
        X, _ = sample_data
        a = design_hapc(X, max_degree=2, npcs=5, center=True)
        b = design_hapc(X, max_degree=2, npcs=5, center=False)
        assert not np.allclose(a.U, b.U)


class TestRidge:
    def test_ridge_regression(self, sample_data):
        X, Y = sample_data
        des = design_hapc(X, max_degree=2, npcs=5)
        U = des.U[:, :3]
        D2 = des.d[:3] ** 2
        beta = ridge_regression(Y, U, D2, lambda_=0.01)
        assert beta.shape == (3,)
        assert np.all(np.isfinite(beta))

    def test_ridge_shrinks_with_lambda(self, sample_data):
        X, Y = sample_data
        des = design_hapc(X, max_degree=2, npcs=5)
        U = des.U[:, :3]
        D2 = des.d[:3] ** 2
        small = ridge_regression(Y, U, D2, lambda_=0.001)
        large = ridge_regression(Y, U, D2, lambda_=1.0)
        assert np.linalg.norm(large) <= np.linalg.norm(small)


class TestKernel:
    def test_kernel_hapc_symmetric(self, sample_data):
        X, _ = sample_data
        K = kernel_hapc(X, max_degree=2)
        assert K.shape == (100, 100)
        assert np.allclose(K, K.T)

    def test_cross_kernel_shape(self, sample_data):
        X, _ = sample_data
        Xte = np.random.default_rng(0).standard_normal((20, 5))
        Kx = cross_kernel_hapc(X, Xte, max_degree=2)
        assert Kx.shape == (20, 100)


class TestOptimizer:
    def test_pcghal(self, sample_data):
        X, Y = sample_data
        des = design_hapc(X, max_degree=2, npcs=5)
        k = des.d.shape[0]
        Xtilde = des.U[:, :k] * des.d[:k]
        ENn = des.V[:, :k]
        Yc = Y - Y.mean()
        alpha0 = ridge_regression(Yc, des.U[:, :k], des.d[:k] ** 2, 0.01)
        out = pcghal(Yc, Xtilde, ENn, alpha0, max_iter=10, tol=1e-6,
                     crit="grad")
        assert out.alpha.shape == (k,)
        assert np.isfinite(out.risk)

    def test_pcghal_classification(self, sample_binary_data):
        X, Y = sample_binary_data
        des = design_hapc(X, max_degree=2, npcs=5)
        k = des.d.shape[0]
        Xtilde = des.U[:, :k] * des.d[:k]
        ENn = des.V[:, :k]
        alpha0 = np.ones(k)
        out = pcghal_classification(Y, Xtilde, ENn, alpha0, max_iter=10)
        assert out.alpha.shape == (k,)
        assert np.isfinite(out.risk)

    def test_pc_hal_classi_matches_pcghal_classification(self, sample_binary_data):
        X, Y = sample_binary_data
        des = design_hapc(X, max_degree=2, npcs=5)
        k = des.d.shape[0]
        Xtilde = des.U[:, :k] * des.d[:k]
        ENn = des.V[:, :k]
        alpha0 = np.ones(k)
        a = pcghal_classification(Y, Xtilde, ENn, alpha0, max_iter=10)
        b = pc_hal_classi(Y, Xtilde, ENn, alpha0, max_iter=10)
        np.testing.assert_allclose(a.alpha, b.alpha)
        np.testing.assert_allclose(a.risk, b.risk)
    def test_fast_pchal(self, sample_data):
        X, Y = sample_data
        des = design_hapc(X, max_degree=2, npcs=5)
        beta = fast_pchal(des.U[:, :3], des.d[:3] ** 2, Y, lambda_=0.01)
        assert beta.shape == (3,)
        assert np.all(np.isfinite(beta))
