"""Tests for high-level HAPC API."""

import numpy as np
import pytest
from hapc.single import single_pcghal
from hapc.cv import pcghal_cv

@pytest.fixture
def regression_data():
    """Generate regression test data."""
    np.random.seed(42)
    n, p = 150, 5
    X = np.random.randn(n, p)
    Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    return X, Y

class TestSinglePcghal:
    def test_single_fit(self, regression_data):
        X, Y = regression_data
        result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01,
                              max_iter=50, verbose=False)
        
        assert result.risk > 0
        assert result.iter > 0
        assert result.lambda_ == 0.01

    def test_single_fit_with_predictions(self, regression_data):
        X, Y = regression_data
        X_test = np.random.randn(20, 5)
        
        result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01,
                              predict=X_test)
        
        assert result.predictions is not None
        assert result.predictions.shape == (20,)

    def test_single_fit_no_center(self, regression_data):
        X, Y = regression_data
        result_center = single_pcghal(X, Y, maxdeg=2, npc=5, 
                                     single_lambda=0.01, center=True)
        result_no_center = single_pcghal(X, Y, maxdeg=2, npc=5, 
                                        single_lambda=0.01, center=False)
        
        # Should be different
        assert result_center.risk != result_no_center.risk

class TestCv:
    def test_cv_fit(self, regression_data):
        X, Y = regression_data
        lambdas = np.logspace(-4, 0, 5)
        
        cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, 
                             nfolds=3, verbose=False)
        
        assert cv_result.best_lambda in lambdas
        assert cv_result.mses.shape == (5,)
        assert np.all(np.isfinite(cv_result.mses))

    def test_cv_with_predictions(self, regression_data):
        X, Y = regression_data
        X_test = np.random.randn(20, 5)
        lambdas = np.logspace(-4, 0, 3)
        
        cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, 
                             nfolds=3, predict=X_test)
        
        assert cv_result.predictions is not None
        assert cv_result.predictions.shape == (20,)
