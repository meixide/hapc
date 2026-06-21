"""family = "logit-hazard": discrete-time logistic hazard via HAPC (Python).

Five worked examples mirroring ``examples/hazard_logit_hazard_examples.R``. The
unified pipeline in ``hazard.R`` shipped two setups; these are three more, plus
two that echo the original flavours. Each example:

  1. simulates baseline covariates X, a latent event time T_event from a known
     discrete hazard, and an independent censoring time C;
  2. forms the *observed* data the user actually has:
        T     = min(T_event, C)        (observed time)
        Delta = 1(T_event <= C)        (event indicator)
  3. fits ``hazard_hapc(X, T, Delta, ...)`` -- the family = "logit-hazard" wrapper;
  4. draws TWO diagnostics:
        (left)  CV risk (logistic deviance) vs lambda, with the selected lambda
                marked -- we check it is an *interior* grid point;
        (right) true hazard vs estimated hazard on the person-period rows.

Run:  python examples/hazard_logit_hazard_examples.py
Output: examples/hazard_logit_hazard_examples_py.png  + a console summary table.
"""

import os
import sys

import numpy as np

# Allow running from a source checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hapc import hazard_hapc


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate_survival(rng, n, grid, true_haz, gen_X, cens_grid=None):
    """Simulate one right-censored discrete-survival data set."""
    if cens_grid is None:
        cens_grid = grid
    X = gen_X(rng, n)
    Tev = np.full(n, grid.max(), dtype=float)
    for i in range(n):
        for t in grid:
            if rng.random() < true_haz(t, X[i]):
                Tev[i] = t
                break
    C = rng.choice(cens_grid, n)
    Tobs = np.minimum(Tev, C)
    Delta = (Tev <= C).astype(float)
    return dict(X=X, T=Tobs, Delta=Delta, grid=grid, true_haz=true_haz)


def run_example(name, sim, max_degree, norm="1",
                log_lambda_min=-7, log_lambda_max=3, grid_length=18,
                max_expand=4):
    """Fit and *adaptively widen* the log-lambda grid until the CV optimum is
    bracketed (interior). This makes the interior-optimum check hold robustly
    regardless of the random data / language RNG, instead of relying on
    hand-tuned ranges."""
    lo, hi = log_lambda_min, log_lambda_max
    for _ in range(max_expand + 1):
        fit = hazard_hapc(
            sim["X"], sim["T"], sim["Delta"],
            norm=norm, max_degree=max_degree, time_grid=sim["grid"],
            log_lambda_min=lo, log_lambda_max=hi,
            grid_length=grid_length, nfolds=5,
        )
        if fit.interior:
            break
        if int(np.argmin(fit.risk)) == 0:
            lo -= 2
        else:
            hi += 2
    truth = np.array([sim["true_haz"](t, sim["X"][i])
                      for t, i in zip(fit.times, fit.ids)])
    slope = np.polyfit(truth, fit.hazard, 1)[0]
    rcorr = float(np.corrcoef(truth, fit.hazard)[0, 1])
    return dict(name=name, fit=fit, truth=truth, rcorr=rcorr, slope=slope)


def plot_example(ax_left, ax_right, res):
    fit, name = res["fit"], res["name"]
    # (left) CV risk vs lambda
    ax_left.semilogx(fit.lambdas, fit.risk, "o", color="darkgreen")
    ax_left.axvline(fit.best_lambda, color="red", ls="--")
    ax_left.set_xlabel(r"$\lambda$")
    ax_left.set_ylabel("CV logistic deviance")
    ax_left.set_title(f"{name}\nCV risk vs lambda  (interior: {fit.interior})")
    ax_left.text(0.05, 0.92, f"best lambda = {fit.best_lambda:.3g}",
                 transform=ax_left.transAxes, fontsize=8)
    # (right) true vs estimated hazard
    ax_right.scatter(res["truth"], fit.hazard, s=6, alpha=0.35, color="blue")
    lo = min(res["truth"].min(), fit.hazard.min())
    hi = max(res["truth"].max(), fit.hazard.max())
    ax_right.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax_right.set_xlabel("True hazard")
    ax_right.set_ylabel("Estimated hazard")
    ax_right.set_title(f"{name}\nr = {res['rcorr']:.3f}, slope = {res['slope']:.3f}")


def main():
    rng = np.random.default_rng(2024)

    examples = []

    # (1) Linear-in-time, additive covariates (echoes hazard.R setup_1)
    examples.append(dict(
        name="linear_additive", max_degree=1, norm="2",
        sim=simulate_survival(
            rng, n=300, grid=np.arange(1, 7),
            true_haz=lambda t, x: sigmoid(-2.6 + 0.28 * t + 1.2 * x[0] - 0.9 * x[1]),
            gen_X=lambda rng, n: np.column_stack([rng.uniform(size=n),
                                                  rng.uniform(size=n)]))))

    # (2) Non-linear time effect with a bump at t = 3 (echoes setup_2)
    def bump_haz(t, x):
        te = 1.5 if t == 3 else 0.2 * t
        return sigmoid(-2.5 + te + 2.0 * x[0])
    examples.append(dict(
        name="bump_time", max_degree=2, norm="1",
        sim=simulate_survival(
            rng, n=300, grid=np.arange(1, 6),
            true_haz=bump_haz,
            gen_X=lambda rng, n: rng.uniform(0.1, 0.9, size=(n, 1)))))

    # (3) NEW: time x covariate interaction
    examples.append(dict(
        name="time_interaction", max_degree=2, norm="2",
        sim=simulate_survival(
            rng, n=300, grid=np.arange(1, 8),
            true_haz=lambda t, x: sigmoid(-3.0 + 0.15 * t + 1.4 * x[0]
                                          + 0.45 * t * x[0] - 0.6 * x[1]),
            gen_X=lambda rng, n: np.column_stack([rng.uniform(size=n),
                                                  rng.uniform(size=n)]))))

    # (4) NEW: U-shaped (non-monotone) hazard in time
    examples.append(dict(
        name="ushaped_time", max_degree=2, norm="1",
        sim=simulate_survival(
            rng, n=300, grid=np.arange(1, 7),
            true_haz=lambda t, x: sigmoid(-2.9 + 0.55 * abs(t - 3.5) + 1.1 * x[0]),
            gen_X=lambda rng, n: rng.uniform(size=(n, 1)))))

    # (5) NEW: three covariates with a nonlinear (threshold) effect
    examples.append(dict(
        name="three_cov_nonlinear", max_degree=2, norm="2",
        sim=simulate_survival(
            rng, n=300, grid=np.arange(1, 9),
            true_haz=lambda t, x: sigmoid(-2.7 + 0.18 * t + 1.3 * x[0] - 1.0 * x[1]
                                          + 0.9 * float(x[2] > 0.5)),
            gen_X=lambda rng, n: np.column_stack([rng.uniform(size=n),
                                                  rng.uniform(size=n),
                                                  rng.uniform(size=n)]))))

    results = [run_example(e["name"], e["sim"], e["max_degree"], e["norm"])
               for e in examples]

    fig, axes = plt.subplots(5, 2, figsize=(11, 20))
    for row, res in zip(axes, results):
        plot_example(row[0], row[1], res)
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__),
                           "hazard_logit_hazard_examples_py.png")
    fig.savefig(out_png, dpi=130)
    print(f"Saved figure to {out_png}\n")

    # Summary table
    print("Summary (interior = best lambda strictly inside the CV grid):")
    header = f"{'example':22s} {'pp_rows':>7s} {'events':>6s} " \
             f"{'best_lambda':>11s} {'interior':>8s} {'r':>6s} {'slope':>6s}"
    print(header)
    all_interior = True
    for res in results:
        fit = res["fit"]
        all_interior &= fit.interior
        print(f"{res['name']:22s} {fit.times.size:7d} {int(fit.Y.sum()):6d} "
              f"{fit.best_lambda:11.4g} {str(fit.interior):>8s} "
              f"{res['rcorr']:6.3f} {res['slope']:6.3f}")
    if all_interior:
        print("\nAll five CV optima are interior grid points.")
    else:
        print("\nNOTE: widen log_lambda_min/max for examples with interior = False.")


if __name__ == "__main__":
    main()
