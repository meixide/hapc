"""Tests for HAPC core functions."""

import numpy as np
import pytest
from hapc.core import (
    pchal_design, ridge_regression, mkernel, kernel_cross, 
    pcghal, pcghal_classification, fast_pchal
)

@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    return X, Y

@pytest.fixture
def sample_binary_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    Y = np.where(X[:, 0] + X[:, 1] > 0, 1.0, -1.0)
    return X, Y

class TestDesign:
    def test_pchal_design_shape(self, sample_data):
        X, _ = sample_data
        des = pchal_design(X, maxdeg=2, npc=5, center=True)
        
        assert des.U.shape == (100, 5)
        assert des.d.shape == (5,)
        assert des.V.shape[1] == 5
        assert des.H.shape[0] == 100

    def test_pchal_design_no_center(self, sample_data):
        X, _ = sample_data
        des_center = pchal_design(X, maxdeg=2, npc=5, center=True)
        des_no_center = pchal_design(X, maxdeg=2, npc=5, center=False)
        
        # Should be different
        assert not np.allclose(des_center.U, des_no_center.U)

class TestRidge:
    def test_ridge_regression(self, sample_data):
        X, Y = sample_data
        des = pchal_design(X, maxdeg=2, npc=5, center=True)
        
        U = des.U[:, :3]
        D2 = des.d[:3] ** 2
        beta = ridge_regression(Y, U, D2, lambda_=0.01)
        
        assert beta.shape == (3,)
        assert np.all(np.isfinite(beta))

    def test_ridge_lambda_effect(self, sample_data):
        X, Y = sample_data
        des = pchal_design(X, maxdeg=2, npc=5, center=True)
        
        U = des.U[:, :3]
        D2 = des.d[:3] ** 2
        
        beta_small = ridge_regression(Y, U, D2, lambda_=0.001)
        beta_large = ridge_regression(Y, U, D2, lambda_=1.0)
        
        # Larger lambda should shrink coefficients more
        assert np.linalg.norm(beta_large) <= np.linalg.norm(beta_small)

class TestKernel:
    def test_mkernel_shape(self, sample_data):
        X, _ = sample_data
        K = mkernel(X, m=2, center=True)
        
        assert K.shape == (100, 100)
        assert np.allclose(K, K.T)  # Should be symmetric

    def test_mkernel_no_center(self, sample_data):
        X, _ = sample_data
        K_center = mkernel(X, m=2, center=True)
        K_no_center = mkernel(X, m=2, center=False)
        
        assert not np.allclose(K_center, K_no_center)

    def test_kernel_cross_shape(self, sample_data):
        X, _ = sample_data
        X_test = np.random.randn(20, 5)
        K_cross = kernel_cross(X, X_test, m=2, center=True)
        
        assert K_cross.shape == (20, 100)

class TestOptimizer:
    def test_pcghal_regression(self, sample_data):
        X, Y = sample_data
        des = pchal_design(X, maxdeg=2, npc=5, center=True)
        
        final_npc = des.d.shape[0]
        Xtilde = des.U[:, :final_npc] @ np.diag(des.d[:final_npc])
        ENn = des.V[:, :final_npc]
        
        Y_centered = Y - Y.mean()
        alpha0 = ridge_regression(Y_centered, des.U[:, :final_npc], 
                                 des.d[:final_npc]**2, 0.01)
        
        result = pcghal(Y_centered, Xtilde, ENn, alpha0, 
                       max_iter=10, tol=1e-6, verbose=False, crit="grad")
        
        assert result.alpha.shape == (final_npc,)
        assert np.isfinite(result.risk)
        assert result.iter > 0

    def test_pcghal_classification(self, sample_binary_data):
        X, Y = sample_binary_data
        des = pchal_design(X, maxdeg=2, npc=5, center=True)
        
        final_npc = des.d.shape[0]
        Xtilde = des.U[:, :final_npc] @ np.diag(des.d[:final_npc])
        ENn = des.V[:, :final_npc]
        
        alpha0 = np.ones(final_npc)
        
        result = pcghal_classification(Y, Xtilde, ENn, alpha0, 
                                       max_iter=10, verbose=False)
        
        assert result.alpha.shape == (final_npc,)
        assert np.isfinite(result.risk)

class TestFastPchal:
    def test_fast_pchal(self, sample_data):
        X, Y = sample_data
        des = pchal_design(X, maxdeg=2, npc=5, center=True)
        
        U = des.U[:, :3]
        D2 = des.d[:3] ** 2
        
        beta = fast_pchal(U, D2, Y, lambda_=0.01)
        
        assert beta.shape == (3,)
        assert np.all(np.isfinite(beta))
