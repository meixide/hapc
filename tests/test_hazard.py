"""Tests for the discrete-time logistic hazard wrapper (family="logit-hazard")."""

import numpy as np
import pytest

from hapc import HazardResult, cv_hapc, hapc, hazard_hapc
from hapc.hazard import _infer_time_grid, _person_period


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture
def survival_data():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([rng.uniform(size=n), rng.integers(0, 2, n)]).astype(float)
    grid = np.arange(1, 7, dtype=float)

    def haz(t, x):
        return _sigmoid(-2.6 + 0.30 * t + 1.3 * x[0] - 0.9 * x[1])

    Tev = np.full(n, grid.max(), dtype=float)
    for i in range(n):
        for t in grid:
            if rng.random() < haz(t, X[i]):
                Tev[i] = t
                break
    C = rng.choice(grid, n)
    Tobs = np.minimum(Tev, C)
    Delta = (Tev <= C).astype(float)
    return X, Tobs, Delta, grid, haz


class TestExpansion:
    def test_infer_integer_grid_fills_gaps(self):
        # observed times skip 4, but the integer grid must still include it
        T = np.array([1.0, 2.0, 3.0, 5.0])
        grid = _infer_time_grid(T)
        assert np.array_equal(grid, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_person_period_labels(self):
        X = np.array([[0.5], [0.2]])
        T = np.array([3.0, 2.0])
        Delta = np.array([1.0, 0.0])  # subject 0 event at 3, subject 1 censored at 2
        grid = np.array([1.0, 2.0, 3.0])
        ids, times, Xpp, Y = _person_period(X, T, Delta, grid)
        # subject 0: rows at 1,2,3 (Y=0,0,1); subject 1: rows at 1,2 (Y=0,0)
        assert ids.tolist() == [0, 0, 0, 1, 1]
        assert times.tolist() == [1, 2, 3, 1, 2]
        assert Y.tolist() == [0, 0, 1, 0, 0]
        # design = [time, X]
        assert Xpp.shape == (5, 2)
        assert np.allclose(Xpp[:, 0], times)


class TestHazardHapc:
    @pytest.mark.parametrize("norm", ["1", "2"])
    def test_fit_runs_and_recovers_hazard(self, survival_data, norm):
        X, T, Delta, grid, haz = survival_data
        fit = hazard_hapc(X, T, Delta, norm=norm, max_degree=2, time_grid=grid,
                          log_lambda_min=-6, log_lambda_max=0, grid_length=10)
        assert isinstance(fit, HazardResult)
        assert fit.hazard.shape == fit.times.shape
        assert np.all(np.isfinite(fit.hazard))
        assert np.all((fit.hazard >= 0) & (fit.hazard <= 1))
        truth = np.array([haz(t, X[i]) for t, i in zip(fit.times, fit.ids)])
        # the fit should be positively correlated with the truth
        assert np.corrcoef(truth, fit.hazard)[0, 1] > 0.3

    def test_dispatch_via_cv_hapc(self, survival_data):
        X, T, Delta, grid, _ = survival_data
        fit = cv_hapc(X, T, family="logit-hazard", Delta=Delta, norm="1",
                      max_degree=1, time_grid=grid)
        assert isinstance(fit, HazardResult)
        assert fit.best_lambda > 0

    def test_sv_is_flagged(self, survival_data):
        X, T, Delta, grid, _ = survival_data
        with pytest.raises(NotImplementedError, match="norm='sv'"):
            hazard_hapc(X, T, Delta, norm="sv")

    def test_missing_delta_flagged(self, survival_data):
        X, T, _, grid, _ = survival_data
        with pytest.raises(ValueError, match="Delta"):
            cv_hapc(X, T, family="logit-hazard", norm="1")

    def test_single_hapc_redirects(self, survival_data):
        X, T, _, _, _ = survival_data
        with pytest.raises(ValueError, match="hazard_hapc"):
            hapc(X, T, family="logit-hazard")

    def test_predict_surface_and_survival(self, survival_data):
        X, T, Delta, grid, _ = survival_data
        X_new = np.array([[0.2, 0.0], [0.9, 1.0]])
        fit = hazard_hapc(X, T, Delta, norm="1", max_degree=2, time_grid=grid,
                          predict=X_new)
        K = grid.size
        assert fit.predict_hazard.shape == (2, K)
        assert fit.predict_survival.shape == (2, K)
        # survival is non-increasing along the time grid and in [0, 1]
        assert np.all(np.diff(fit.predict_survival, axis=1) <= 1e-9)
        assert np.all((fit.predict_survival >= 0) & (fit.predict_survival <= 1))

    def test_interior_flag_detects_boundary(self, survival_data):
        X, T, Delta, grid, _ = survival_data
        # an absurdly high lambda window forces the optimum to the boundary
        fit = hazard_hapc(X, T, Delta, norm="2", max_degree=1, time_grid=grid,
                          log_lambda_min=3, log_lambda_max=6, grid_length=6)
        assert fit.interior is False
