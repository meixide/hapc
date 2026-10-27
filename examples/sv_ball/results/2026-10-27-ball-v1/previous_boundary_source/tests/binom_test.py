"""
Ad-hoc script (not collected by pytest: name is not test_*.py).

Run from repo root:
  python tests/binom_test.py

Requires matplotlib: pip install matplotlib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hapc import cv_hapc

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for this script. Install with: pip install matplotlib"
    ) from e


def main() -> None:
    rng = np.random.default_rng(0)
    n, p = 500, 2
    X = rng.standard_normal((n, p))
    logit = 0.4 * X[:, 0] - 0.2 * X[:, 1]
    prob = 1.0 / (1.0 + np.exp(-logit))
    Y = (rng.uniform(size=n) < prob).astype(float)  # {0,1}

    X_test = rng.standard_normal((30, p))
    # Same DGP as training for interpretable "true" probabilities on test rows
    logit_test = 0.4 * X_test[:, 0] - 0.2 * X_test[:, 1]
    prob_true_test = 1.0 / (1.0 + np.exp(-logit_test))

    fit = cv_hapc(
        X,
        Y,
        family="binomial",
        norm="1",
        max_degree=1,
        npcs=150,
        log_lambda_min=-4.0,
        log_lambda_max=1.0,
        grid_length=20,
        nfolds=5,
        predict=X_test,
    )

    import hapc as hapc_pkg

    print("hapc version:", hapc_pkg.__version__)
    print("best_lambda:", fit.best_lambda)
    print("best_model_alpha shape:", fit.best_model_alpha.shape)
    if fit.predictions is not None:
        print("test predictions (probabilities) shape:", fit.predictions.shape)

    # CVResult.mses is named for API parity; for binomial it is mean CV
    # logistic deviance per fold (not squared-error MSE).
    fig, (ax_cv, ax_cal) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[1, 1])

    ax_cv.semilogx(fit.lambdas, fit.mses, marker="o", ms=3, lw=1)
    ax_cv.axvline(
        fit.best_lambda,
        color="C1",
        ls="--",
        lw=1,
        label=f"best λ = {fit.best_lambda:.4g}",
    )
    ax_cv.set_xlabel(r"$\lambda$")
    ax_cv.set_ylabel("mean CV deviance\n(binomial: logistic loss on held-out folds)")
    ax_cv.set_title("Binomial HAPC (norm=1): cross-validated risk vs λ")
    ax_cv.grid(True, which="both", alpha=0.3)
    ax_cv.legend(loc="best")

    pred = fit.predictions
    if pred is None:
        ax_cal.text(0.5, 0.5, "no test predictions", ha="center", va="center", transform=ax_cal.transAxes)
    else:
        lo, hi = 0.0, 1.0
        ax_cal.scatter(prob_true_test, pred, alpha=0.75, edgecolors="k", lw=0.3, s=40)
        ax_cal.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax_cal.set_aspect("equal", adjustable="box")
        ax_cal.set_xlim(lo, hi)
        ax_cal.set_ylim(lo, hi)
        ax_cal.set_xlabel("true P(Y=1 | X) (simulation DGP)")
        ax_cal.set_ylabel("predicted probability (CV winner on train)")
        ax_cal.set_title("Test calibration: predicted vs true probability")
        ax_cal.grid(True, alpha=0.3)
        ax_cal.legend(loc="lower right")

    fig.tight_layout()

    out = Path(__file__).resolve().parent / "binom_test_cv_curve.png"
    fig.savefig(out, dpi=150)
    print("saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
