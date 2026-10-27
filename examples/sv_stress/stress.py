"""Deterministic geometry stress: each norm gets radius C=1 in its own space.
Not a comparison at a common lambda, common SV, CV, or prediction efficiency.
Run: python examples/sv_stress/stress.py --output DIRECTORY
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from hapc import hapc_core as core


def calibrate(fit, measure, C=1.):
    lo, hi = 0., 1.
    while measure(fit(hi)) > C:
        hi *= 2
    for _ in range(90):
        mid = (lo+hi)/2
        if measure(fit(mid)) > C:
            lo = mid
        else:
            hi = mid
    return hi, fit(hi)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (32, 64, 128, 256):
        rng = np.random.default_rng(20261027+n)
        x = (np.arange(n)+.5)/n
        knots = (x[:-1]+x[1:])/2
        H = (x[:, None] >= knots).astype(float)  # distinct univariate HAL atoms
        means = H.mean(axis=0)
        U, D, Vt = np.linalg.svd(H-means, full_matrices=False)
        k = min(12, n-1)
        U, D, V = U[:, :k], D[:k], Vt[:k].T
        Z = U*D
        target = np.r_[2., np.repeat(.02, k-1)]
        y = Z@target + rng.normal(0, .03, n)
        y -= y.mean()
        lasso = lambda lam: core.fast_pchal_call(U, D**2, y, lam)
        ridge = lambda lam: core.ridge_call(y, U, D**2, lam)
        lam1, a1 = calibrate(lasso, lambda a: np.linalg.norm(a, 1))
        lam2, a2 = calibrate(ridge, np.linalg.norm)
        lamsv, initial = calibrate(ridge, lambda a: np.linalg.norm(V@a, 1))
        sv = core.pcghal_boundary_call(y, Z, V, initial, 3000, 1e-10, .8, False, "grad")
        path_sv = np.abs(sv.alphaiters@V.T).sum(axis=1)
        assert np.max(np.abs(path_sv-1)) < 2e-12
        for norm, a, lam in (("1", a1, lam1), ("2", a2, lam2), ("sv", sv.alpha, lamsv)):
            beta = V@a
            coeff_sv = np.linalg.norm(beta, 1)
            # Fitted function is ymean + sum beta_j(H_j - means_j), ymean=0.
            anchored_intercept = -means@beta
            rows.append(dict(n=n, q=n-1, k=k, norm=norm, C=1., lambda_=lam,
                alpha_l1=np.linalg.norm(a,1), alpha_l2=np.linalg.norm(a),
                coefficient_sv=coeff_sv, full_sv=abs(anchored_intercept)+coeff_sv,
                training_mse=np.mean((y-Z@a)**2),
                status=sv.status if norm=="sv" else "closed_form",
                iterations=sv.iter if norm=="sv" else 0,
                max_path_sv_error=np.max(abs(path_sv-1)) if norm=="sv" else ""))
        # Geometric separation is the property under test, not a risk ranking.
        assert np.linalg.norm(V@a1,1)>1.5 and np.linalg.norm(V@a2,1)>1.5
    with (output/"norm_comparison.csv").open("w") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (output/"design.json").write_text(json.dumps(dict(seed="20261027+n", n=[32,64,128,256],
        C=1, knots="adjacent grid midpoints, distinct univariate indicator atoms", k=12,
        tuning="bisection over native penalty paths to own-norm radius C; SV radius from ridge",
        sv_constraint="equality on coefficient variation; intercept excluded",
        caveats=["synthetic geometry demonstration, not CV", "equal numeric radii are different classes",
                 "multiplicative solver can stall or lock zero PC coordinates",
                 "full anchored variation is reported separately"]), indent=2)+"\n")
    for row in rows:
        print(f"n={row['n']:3d} norm={row['norm']:2s} coefficient_SV={row['coefficient_sv']:.8f} status={row['status']}")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
