"""Tests for binomial / logistic-regression HAPC paths in Python."""

import numpy as np
import pytest

from hapc import (
    SinglePcghalClassificationResult,
    cv_hapc,
    hapc,
    single_pcghal_classification,
    single_pcghal_classification_ridge_only,
)


@pytest.fixture
def binary_01():
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    Y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, 1, 0)
    return X, Y


@pytest.fixture
def binary_pm1(binary_01):
    X, Y = binary_01
    return X, np.where(Y == 0, -1, 1)


class TestSinglePcghalClassification:
    def test_basic_fit_01(self, binary_01):
        X, Y = binary_01
        res = single_pcghal_classification(
            X, Y, max_degree=2, npcs=10, lambda_=0.01,
        )
        assert isinstance(res, SinglePcghalClassificationResult)
        assert res.alpha.shape[0] > 0
        assert np.isfinite(res.risk)
        assert res.iter >= 0
        assert res.lambda_ == 0.01

    def test_encodings_match(self, binary_01, binary_pm1):
        _, Y_01 = binary_01
        X, Y_pm1 = binary_pm1
        a = single_pcghal_classification(X, Y_01, max_degree=2, npcs=10,
                                         lambda_=0.01)
        b = single_pcghal_classification(X, Y_pm1, max_degree=2, npcs=10,
                                         lambda_=0.01)
        assert np.isclose(a.risk, b.risk, rtol=1e-5)
        assert np.allclose(a.alpha, b.alpha, rtol=1e-5)

    def test_predictions(self, binary_01):
        X, Y = binary_01
        Xte = np.random.default_rng(0).standard_normal((20, 5))
        res = single_pcghal_classification(
            X, Y, max_degree=2, npcs=10, lambda_=0.01, predict=Xte,
        )
        assert res.predictions.shape == (20,)
        assert res.probabilities.shape == (20,)
        assert res.predicted_classes.shape == (20,)
        assert np.all((res.probabilities >= 0) & (res.probabilities <= 1))
        assert np.all(np.isin(res.predicted_classes, [-1, 1]))

    def test_invalid_y(self, binary_01):
        X, Y = binary_01
        bad = Y.copy().astype(float); bad[0] = 2.0
        with pytest.raises(ValueError):
            single_pcghal_classification(X, bad, max_degree=2, npcs=10,
                                         lambda_=0.01)


class TestRidgeOnly:
    def test_runs(self, binary_01):
        X, Y = binary_01
        res = single_pcghal_classification_ridge_only(
            X, Y, max_degree=2, npcs=10, lambda_=0.01,
        )
        assert isinstance(res, SinglePcghalClassificationResult)
        assert np.isfinite(res.risk)
        assert res.iter == 0


class TestHapcBinomial:
    def test_dispatch(self, binary_01):
        X, Y = binary_01
        a = hapc(X, Y, family="binomial", max_degree=2, npcs=10,
                 lambda_=0.01, norm="sv")
        b = single_pcghal_classification(X, Y, max_degree=2, npcs=10,
                                         lambda_=0.01)
        assert np.isclose(a.risk, b.risk, rtol=1e-5)
        assert np.allclose(a.alpha, b.alpha, rtol=1e-5)

    def test_norm_2_uses_ridge_only(self, binary_01):
        X, Y = binary_01
        res = hapc(X, Y, family="binomial", max_degree=2, npcs=10,
                   lambda_=0.01, norm="2")
        assert res.iter == 0  # ridge-only path


class TestCvHapcBinomial:
    """Binomial CV must always use logistic loss (deviance), never MSE."""

    @pytest.mark.parametrize("norm", ["sv", "1", "2"])
    def test_logistic_cv_runs(self, binary_01, norm):
        X, Y = binary_01
        res = cv_hapc(
            X, Y, family="binomial", max_degree=2, npcs=10,
            grid_length=4, nfolds=3, norm=norm,
        )
        assert res.lambdas.shape == (4,)
        assert res.best_lambda > 0
        assert np.all(np.isfinite(res.mses))
        assert np.all(res.mses >= 0)


class TestSinglePcghalLasso:
    def test_basic_fit(self, binary_01):
        from hapc import single_pcghal_classification_lasso
        X, Y = binary_01
        res = single_pcghal_classification_lasso(
            X, Y, max_degree=2, npcs=10, lambda_=0.05,
        )
        assert res.alpha.shape[0] > 0
        assert np.isfinite(res.risk)
        assert res.lambda_ == 0.05

    def test_increasing_lambda_increases_sparsity(self, binary_01):
        from hapc import single_pcghal_classification_lasso
        X, Y = binary_01
        small = single_pcghal_classification_lasso(
            X, Y, max_degree=2, npcs=10, lambda_=1e-3,
        )
        big = single_pcghal_classification_lasso(
            X, Y, max_degree=2, npcs=10, lambda_=1.0,
        )
        # LASSO with very large λ should not have *more* nonzero coefficients
        assert int((big.alpha != 0).sum()) <= int((small.alpha != 0).sum())
