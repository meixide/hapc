"""Reproducible ``ate_hapc`` diagnostic demo (built-in DGP, L1 norm, split λ grids).

The exploratory figure ``ate_hapc_diagnostics_demo.png`` in the repository root
can be regenerated from the package root::

    python tests/test_ate_hapc_diagnostics_example.py

This uses ``alpha=0.05`` with the **moderate** DGP from the original
``ate/simulate_data.py`` script (vendored below — exact same draws thanks to
``np.random.seed`` + the same ``np.random.uniform`` / ``normal`` /
``binomial`` call order).  ``ate_hapc`` is run with ``npcs = n - 1``.

* ``W1 ~ Uniform(-2, 2)``
* ``W2 ~ Normal(0, 0.5)``
* ``A | W ~ Bernoulli(expit(W1 + 0.5*W2 + W1*W2 + 0.3*W2**2))``
* ``Y | A,W = 2*W1 - 2*W2**2 + W2 + W1*W2 + 0.5 + ε``,
  ``ε ~ Normal(0, sigma_eps=0.5)`` — true ATE = 0 (no A in mean).

A lightweight pytest checks that the point estimate and 95% CI match the
pinned values (guards against silent numerical drift).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hapc import ATEResult

_REPO = Path(__file__).resolve().parent.parent

# --- Pinned exploratory settings (see docstring) --------------------------------
DEMO_SEED = 456
DEMO_N = 300
DEMO_ALPHA = 0.05
DEMO_MAX_DEGREE = 2
DEMO_NFOLDS = 4
DEMO_NORM = "1"

LOG_LAMBDA_PROP_MIN = -5.0
LOG_LAMBDA_PROP_MAX = -2.0
GRID_LENGTH_PROP = 8

LOG_LAMBDA_OUT_MIN = -4.0
LOG_LAMBDA_OUT_MAX = -1.0
GRID_LENGTH_OUT = 8

FIGURE_NAME = "ate_hapc_diagnostics_demo.png"

# Pinned outputs (``alpha=0.05``, ``npcs = n - 1``, current C++/Python stack)
_EXPECTED_ESTIMATE = 0.07963839541495547
_EXPECTED_LOWER = -0.048798044399148574
_EXPECTED_UPPER = 0.2080748352290595


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _true_log_odds_A(W1: np.ndarray, W2: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return W1 + 0.5 * W2
    if mode == "moderate":
        return W1 + 0.5 * W2 + W1 * W2 + 0.3 * W2**2
    if mode == "complex":
        return 0.5 * W2**2 - 0.5 * np.exp(W1 / 2.0)
    raise ValueError(f"Unknown mode: {mode!r}")


def load_demo_data(
    n: int = None,  # type: ignore[assignment]
    seed: int = None,  # type: ignore[assignment]
    *,
    mode: str = "moderate",
    sigma_eps: float = 0.5,
):
    """Vendored DGP from ``ate/simulate_data.simulate_data`` (mode-aware).

    Numerically identical to the original function (same legacy ``np.random``
    state and call order: ``uniform``, ``normal``, ``binomial``, ``normal``).
    """
    if n is None:
        n = DEMO_N
    if seed is None:
        seed = DEMO_SEED

    np.random.seed(seed)
    W1 = np.random.uniform(-2.0, 2.0, size=n)
    W2 = np.random.normal(loc=0.0, scale=0.5, size=n)
    propensity = _expit(_true_log_odds_A(W1, W2, mode=mode))
    A = np.random.binomial(n=1, p=propensity, size=n).astype(np.float64)
    eps = np.random.normal(loc=0.0, scale=sigma_eps, size=n)
    Y = 2.0 * W1 - 2.0 * W2**2 + W2 + W1 * W2 + 0.5 + eps

    W = np.column_stack([W1, W2]).astype(np.float64)
    return W, A, Y.astype(np.float64)


def run_ate_hapc_demo(
    *,
    plot_diagnostics: bool = False,
) -> "ATEResult":
    """Run ``ate_hapc`` with the pinned demo hyperparameters.

    Uses ``npcs = n - 1`` (sample size from ``load_demo_data``) for both
    propensity and outcome stages, matching the usual HAL rank cap.
    """
    from hapc import ate_hapc

    W, A, Y = load_demo_data()
    npcs = int(W.shape[0]) - 1
    return ate_hapc(
        W,
        Y,
        A,
        alpha=DEMO_ALPHA,
        max_degree=DEMO_MAX_DEGREE,
        npcs=npcs,
        log_lambda_prop_min=LOG_LAMBDA_PROP_MIN,
        log_lambda_prop_max=LOG_LAMBDA_PROP_MAX,
        grid_length_prop=GRID_LENGTH_PROP,
        log_lambda_out_min=LOG_LAMBDA_OUT_MIN,
        log_lambda_out_max=LOG_LAMBDA_OUT_MAX,
        grid_length_out=GRID_LENGTH_OUT,
        nfolds=DEMO_NFOLDS,
        norm=DEMO_NORM,
        plot_diagnostics=plot_diagnostics,
    )


def save_diagnostics_figure(
    dest: Path | None = None,
    *,
    dpi: int = 150,
):
    """Regenerate the three-panel diagnostic PNG (requires matplotlib).

    Returns
    -------
    path : pathlib.Path
        Where the figure was written.
    result : ATEResult
        ``ate_hapc`` outcome for the same run (``alpha=DEMO_ALPHA``).
    """
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required to save the figure; pip install matplotlib"
        ) from e

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = dest or (_REPO / FIGURE_NAME)
    _orig_show = plt.show

    def _save_instead(*_a, **_k):
        plt.gcf().savefig(out, dpi=dpi, bbox_inches="tight")

    plt.show = _save_instead
    try:
        res = run_ate_hapc_demo(plot_diagnostics=True)
    finally:
        plt.show = _orig_show
        plt.close("all")
    return out, res


def test_ate_hapc_demo_dgp_matches_pinned_ci():
    """Regression: demo DGP + grids + L1 norm → fixed ATE (stack drift guard)."""
    res = run_ate_hapc_demo(plot_diagnostics=False)
    assert np.isclose(res.estimate, _EXPECTED_ESTIMATE, rtol=0, atol=1e-9)
    assert np.isclose(float(res.lower), _EXPECTED_LOWER, rtol=0, atol=1e-9)
    assert np.isclose(float(res.upper), _EXPECTED_UPPER, rtol=0, atol=1e-9)


if __name__ == "__main__":
    path, r = save_diagnostics_figure()
    print(f"Saved: {path}")
    print(f"ATE (alpha={DEMO_ALPHA}): estimate={r.estimate}, CI=({r.lower}, {r.upper})")
