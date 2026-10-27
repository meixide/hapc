"""Smoke tests for the high-level Python HAPC API."""

import numpy as np
import pytest

from hapc import cv_hapc, hapc, pcghal_cv, single_pcghal


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n, p = 150, 5
    X = rng.standard_normal((n, p))
    Y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.1
    return X, Y


class TestSinglePcghal:
    def test_basic_fit(self, regression_data):
        X, Y = regression_data
        res = single_pcghal(X, Y, max_degree=2, npcs=5, lambda_=0.01,
                            max_iter=50, verbose=False)
        assert res.risk > 0
        assert res.iter >= 0
        assert res.lambda_ == 0.01

    def test_fit_with_predictions(self, regression_data):
        X, Y = regression_data
        Xte = np.random.default_rng(0).standard_normal((20, 5))
        res = single_pcghal(X, Y, max_degree=2, npcs=5, lambda_=0.01,
                            predict=Xte)
        assert res.predictions is not None
        assert res.predictions.shape == (20,)

    def test_no_center_changes_risk(self, regression_data):
        X, Y = regression_data
        a = single_pcghal(X, Y, max_degree=2, npcs=5, lambda_=0.01, center=True)
        b = single_pcghal(X, Y, max_degree=2, npcs=5, lambda_=0.01, center=False)
        assert a.risk != b.risk


class TestHapcDispatcher:
    @pytest.mark.parametrize("norm", ["sv", "1", "2"])
    def test_gaussian_norms(self, regression_data, norm):
        X, Y = regression_data
        res = hapc(X, Y, family="gaussian", max_degree=2, npcs=5,
                   lambda_=0.01, norm=norm)
        assert res.alpha is not None


class TestCv:
    def test_pcghal_cv(self, regression_data):
        X, Y = regression_data
        lambdas = np.logspace(-4, 0, 5)
        res = pcghal_cv(X, Y, max_degree=2, npcs=5, lambdas=lambdas,
                        nfolds=3, verbose=False)
        assert res.best_lambda in lambdas
        assert res.mses.shape == (5,)
        assert np.all(np.isfinite(res.mses))

    def test_pcghal_cv_with_predictions(self, regression_data):
        X, Y = regression_data
        Xte = np.random.default_rng(0).standard_normal((20, 5))
        lambdas = np.logspace(-4, 0, 3)
        res = pcghal_cv(X, Y, max_degree=2, npcs=5, lambdas=lambdas,
                        nfolds=3, predict=Xte)
        assert res.predictions is not None
        assert res.predictions.shape == (20,)

    @pytest.mark.parametrize("norm", ["sv", "1", "2"])
    def test_cv_hapc_all_norms(self, regression_data, norm):
        X, Y = regression_data
        res = cv_hapc(X, Y, family="gaussian", max_degree=1, npcs=5,
                      grid_length=4, nfolds=3, norm=norm)
        assert res.lambdas.shape == (4,)
        assert res.best_lambda > 0
