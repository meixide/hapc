"""Smoke tests for ate_hapc (HAPC + outcome undersmoothing)."""

from __future__ import annotations

import numpy as np
import pytest

from hapc import ATEResult, ate_hapc

# Shared small grids for tests (prop and outcome grids match unless we split).
_GRID_AB = dict(
    log_lambda_prop_min=-4,
    log_lambda_prop_max=-2,
    grid_length_prop=4,
    log_lambda_out_min=-4,
    log_lambda_out_max=-2,
    grid_length_out=4,
)


def _simulate_simple(n: int = 150, seed: int = 0):
    """Simple DGP with continuous outcome that does not depend on A → true ATE = 0."""
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-2.0, 2.0, n)
    W2 = rng.normal(0.0, 0.5, n)
    W = np.column_stack([W1, W2])
    eta = W1 + 0.5 * W2
    p = 1.0 / (1.0 + np.exp(-eta))
    A = rng.binomial(1, p, n).astype(np.float64)
    eps = rng.normal(0.0, 0.5, n)
    # Y depends only on W, not on A → true ATE = 0
    Y = 2.0 * W1 - 1.0 * (W2**2) + 0.4 * W2 + 0.5 + eps
    return W, Y, A


def _simulate_with_effect(n: int = 200, seed: int = 1, effect: float = 1.0):
    """Same as above but with a known additive treatment effect."""
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-2.0, 2.0, n)
    W2 = rng.normal(0.0, 0.5, n)
    W = np.column_stack([W1, W2])
    eta = W1 + 0.5 * W2
    p = 1.0 / (1.0 + np.exp(-eta))
    A = rng.binomial(1, p, n).astype(np.float64)
    eps = rng.normal(0.0, 0.5, n)
    Y = effect * A + 2.0 * W1 + 0.5 + eps
    return W, Y, A


class TestAteHapc:
    def test_returns_three_finite_numbers(self):
        W, Y, A = _simulate_simple(n=120, seed=0)
        res = ate_hapc(
            W, Y, A, alpha=0.05,
            max_degree=2, npcs=20,
            nfolds=3, norm="2",
            **_GRID_AB,
        )
        assert isinstance(res, ATEResult)
        assert np.isfinite(res.estimate)
        assert np.isfinite(res.lower)
        assert np.isfinite(res.upper)
        assert res.lower <= res.estimate <= res.upper

    def test_ci_widens_with_smaller_alpha(self):
        W, Y, A = _simulate_simple(n=150, seed=2)
        kw = dict(max_degree=2, npcs=20, nfolds=3, norm="2", **_GRID_AB)
        wide = ate_hapc(W, Y, A, alpha=0.01, **kw)
        narrow = ate_hapc(W, Y, A, alpha=0.10, **kw)
        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)

    def test_alpha_validation(self):
        W, Y, A = _simulate_simple(n=80, seed=3)
        g = dict(
            log_lambda_prop_min=-4, log_lambda_prop_max=-2, grid_length_prop=3,
            log_lambda_out_min=-4, log_lambda_out_max=-2, grid_length_out=3,
        )
        with pytest.raises(ValueError):
            ate_hapc(W, Y, A, alpha=0.0, max_degree=1, npcs=10,
                     nfolds=2, norm="2", **g)
        with pytest.raises(ValueError):
            ate_hapc(W, Y, A, alpha=1.0, max_degree=1, npcs=10,
                     nfolds=2, norm="2", **g)

    def test_accepts_pm1_treatment(self):
        W, Y, A = _simulate_simple(n=100, seed=4)
        Apm1 = (2.0 * A - 1.0)
        base = dict(max_degree=2, npcs=15, nfolds=2, norm="2", **_GRID_AB)
        res01 = ate_hapc(W, Y, A, alpha=0.05, **base)
        rpm1 = ate_hapc(W, Y, Apm1, alpha=0.05, **base)
        assert np.isclose(res01.estimate, rpm1.estimate, atol=1e-10)
        assert np.isclose(res01.lower, rpm1.lower, atol=1e-10)
        assert np.isclose(res01.upper, rpm1.upper, atol=1e-10)

    def test_signal_recovery_loose(self):
        """ATE should be in the right ballpark when there is a true effect."""
        W, Y, A = _simulate_with_effect(n=300, seed=5, effect=1.0)
        res = ate_hapc(
            W, Y, A, alpha=0.05,
            max_degree=2, npcs=40,
            log_lambda_prop_min=-5,
            log_lambda_prop_max=-2,
            grid_length_prop=6,
            log_lambda_out_min=-5,
            log_lambda_out_max=-2,
            grid_length_out=6,
            nfolds=3, norm="2",
        )
        # Loose tolerance: only a sanity check, not a coverage guarantee.
        assert -0.5 < res.estimate < 2.5

    def test_unpacking_three_numbers(self):
        W, Y, A = _simulate_simple(n=100, seed=6)
        g = dict(
            log_lambda_prop_min=-4, log_lambda_prop_max=-2, grid_length_prop=3,
            log_lambda_out_min=-4, log_lambda_out_max=-2, grid_length_out=3,
        )
        est, lo, hi = ate_hapc(W, Y, A, alpha=0.05, max_degree=1, npcs=10,
                               nfolds=2, norm="2", **g)
        assert lo <= est <= hi

    def test_plot_diagnostics_no_display(self, monkeypatch):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        W, Y, A = _simulate_simple(n=90, seed=8)
        res = ate_hapc(
            W, Y, A, alpha=0.05,
            max_degree=2, npcs=15,
            nfolds=3, norm="2",
            plot_diagnostics=True,
            **_GRID_AB,
        )
        assert isinstance(res, ATEResult)
        plt.close("all")

    def test_split_grids_prop_vs_out(self):
        """Outcome grid can differ from propensity grid (undersmoothing on outcome only)."""
        rng = np.random.default_rng(0)
        n = 100
        W = rng.standard_normal((n, 3))
        A = (W[:, 0] + rng.standard_normal(n) > 0).astype(np.float64)
        Y = W[:, 1] + 0.3 * A + rng.standard_normal(n) * 0.2
        res = ate_hapc(
            W, Y, A,
            log_lambda_prop_min=-5,
            log_lambda_prop_max=-3,
            grid_length_prop=5,
            log_lambda_out_min=-2,
            log_lambda_out_max=0,
            grid_length_out=6,
            npcs=15,
            max_degree=1,
            nfolds=3,
            norm="2",
        )
        assert isinstance(res, ATEResult)
        assert np.isfinite(res.estimate)
